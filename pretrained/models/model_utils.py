"""
model_utils.py
--------
Self-contained utilities for FTW embedding extraction with Clay, TerraFM, and DINOv3.
"""

import os
import json
import torch
import rasterio
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
from box import Box
from .clay.finetune.segment.factory import SegmentEncoder as ClayEncoder
from .TerraFM.terrafm_segment import TerraFMEncoderWrapper as TerraFMEncoder
from .dinov3.dinov3_segmentor import SegmentEncoder as DinoV3Encoder
from .terramind.terramind import SegmentEncoder as TeraMindEncoder
from datetime import datetime, timedelta
import math

# ============================================================
# 1️⃣ IMAGE I/O
# ============================================================
def load_image(path: str, select_rgb: bool = False):
    """Load a single Sentinel-2 image and return image tensor and center lat/lon.

    Args:
        path: Path to Sentinel-2 .tif file (e.g., window_a or window_b)
        select_rgb: If True, keeps only the first 3 channels (RGB)

    Returns:
        (torch.Tensor[C,H,W], lat, lon)
    """
    with rasterio.open(path) as f:
        img = f.read().astype(np.float32)
        r, c = f.height // 2, f.width // 2
        lon, lat = rasterio.transform.xy(f.transform, r, c)

    if select_rgb:
        img = img[:3]

    return torch.from_numpy(img), lat, lon


def preprocess_general(sample: dict, norm_const=3000.0) -> dict:
    """Normalize Sentinel-2 reflectance for TerraFM (L2A scaled by 10 000)."""
    sample["image"] = sample["image"] / norm_const
    return sample

class preprocess_dinov3:
    def __init__(self, mean=None, std=None, norm_constant=3000.0):
        self.mean = torch.as_tensor(mean if mean is not None else [0.430, 0.411, 0.296], dtype=torch.float32)
        self.std  = torch.as_tensor(std  if std  is not None else [0.213, 0.156, 0.143], dtype=torch.float32)
        self.norm_constant = norm_constant

    def __call__(self, sample: dict) -> dict:
        image = sample["image"][:3,:,:] / self.norm_constant        # [C,H,W]
        mean = self.mean.to(image.device).view(-1,1,1)  # [C,1,1]
        std  = self.std.to(image.device).view(-1,1,1)
        sample["image"] = (image - mean) / std
        return sample


class preprocess_clay:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]  # shape: [C, H, W]
        mean = self.mean.to(image.device).view(-1, 1, 1)  # reshape to [C, 1, 1]
        std = self.std.to(image.device).view(-1, 1, 1)

        sample["image"] = (image - mean) / std
        return sample


# ============================================================
# 3️⃣ METADATA HELPERS
# ============================================================
def normalize_timestamp(date,hour=False):
    # import code;code.interact(local=dict(globals(), **locals()));
    week = date.isocalendar().week * 2 * np.pi / 52
    if hour:
        hour = date.hour * 2 * np.pi / 24
    else:
        hour = 12 #approximate 12pm to be default time if time is not given

    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))



def get_median_date(start: str, end: str) -> datetime:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    delta_days = (end_dt - start_dt).days
    median_date = start_dt + timedelta(days=delta_days // 2)

    return median_date


def extract_season_dates(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    seasons = data["seasons"]

    # Handle both dict and list of dicts
    if isinstance(seasons, list):
        seasons = seasons[0]  # just take the first entry

    window_a_start = seasons["window_a"]["start"]
    window_a_end = seasons["window_a"]["end"]
    window_b_start = seasons["window_b"]["start"]
    window_b_end = seasons["window_b"]["end"]

    return {
        "window_a": get_median_date(window_a_start, window_a_end),
        "window_b": get_median_date(window_b_start, window_b_end),
    }

def prepare_general_sample(
    image_path: str,
    preprocess: callable,
):
    image, lat, lon = load_image(image_path)
    image = preprocess({"image":image})
    return image

def prepare_clay_sample(
    image_path: str,
    preprocess: callable,
    gsd: torch.Tensor,
    waves: torch.Tensor,
    data_root: str = "/u/subashk/storage/ftw-prue/data/ftw",
):
    """
    Prepare a Sentinel-2 image sample for CLAY encoder inference.
    Automatically infers window type (A/B) and extracts temporal + spatial encodings.

    Args:
        image_path: Path to a Sentinel-2 image (.tif)
        device: torch.device
        preprocess: normalization transform (preprocess_clay instance)
        gsd: Ground Sampling Distance tensor
        waves: Band wavelength tensor

    Returns:
        dict ready for ClayEncoder.forward() with batched tensors.
        {
            'platform': 'sentinel-2-l2a',
            'image':  [C,H,W],
            'time':   [T],
            'latlon': [4],
            'gsd':    tensor,
            'waves':  tensor
        }
    """

    image_path_lower = str(image_path).lower()
    if "window_a" in image_path_lower:
        window_type = "a"
    elif "window_b" in image_path_lower:
        window_type = "b"
    else:
        raise ValueError(f"Cannot infer window type from path: {image_path}")

    # determine country and get timestamps
    country = Path(image_path).parts[-4]
    json_fn = f"{data_root}/{country}/data_config_{country}.json"
    times_json = extract_season_dates(json_fn)
    timestamp = times_json[f"window_{window_type}"]

    # load + preprocess image
    image, lat, lon = load_image(image_path)
    image = preprocess({"image": image})["image"]
    
    # temporal encoding
    week_norm, hour_norm = normalize_timestamp(timestamp)
    time_vec = torch.tensor(list(week_norm) + list(hour_norm), dtype=torch.float32)

    # spatial encoding
    latlon_encoded = normalize_latlon(lat, lon)
    lat_vec = torch.tensor(latlon_encoded[0], dtype=torch.float32)
    lon_vec = torch.tensor(latlon_encoded[1], dtype=torch.float32)
    latlon_vec = torch.cat([lat_vec, lon_vec], dim=-1) 

    return {
        "platform": "sentinel-2-l2a",
        "image": image,
        "time": time_vec,
        "latlon": latlon_vec,
        "gsd": gsd,
        "waves": waves,
    }


def prepare_clay_batch(
    image_paths: list[str],
    device: torch.device,
    preprocess: callable,
    gsd: torch.Tensor,
    waves: torch.Tensor,
    data_root: str = "/u/subashk/storage/ftw-prue/data/ftw",
):
    """
    Prepare a batch of Sentinel-2 image samples for CLAY encoder inference.
    Automatically infers window type (A/B) and extracts temporal + spatial encodings.

    Args:
        image_paths: List of Sentinel-2 image paths (.tif)
        device: torch.device
        preprocess: normalization transform (preprocess_clay instance)
        gsd: Ground Sampling Distance tensor
        waves: Band wavelength tensor

    Returns:
        dict ready for ClayEncoder.forward() with batched tensors.
        {
            'platform': 'sentinel-2-l2a',
            'image':  [B,C,H,W],
            'time':   [B,T],
            'latlon': [B,4],
            'gsd':    tensor,
            'waves':  tensor
        }
    """
    images, times, latlons = [], [], []

    for image_path in image_paths:
        image_path_lower = str(image_path).lower()
        if "window_a" in image_path_lower:
            window_type = "a"
        elif "window_b" in image_path_lower:
            window_type = "b"
        else:
            raise ValueError(f"Cannot infer window type from path: {image_path}")

        # determine country and get timestamps
        country = Path(image_path).parts[-4]
        json_fn = f"{data_root}/{country}/data_config_{country}.json"
        times_json = extract_season_dates(json_fn)
        timestamp = times_json[f"window_{window_type}"]

        # load + preprocess image
        image, lat, lon = load_image(image_path)
        sample = {"image": image}
        sample = preprocess(sample)
        images.append(sample["image"])

        # temporal encoding
        week_norm, hour_norm = normalize_timestamp(timestamp)
        time_vec = torch.tensor(list(week_norm) + list(hour_norm), dtype=torch.float32)
        times.append(time_vec)

        # spatial encoding
        latlon_encoded = normalize_latlon(lat, lon)
        lat_vec = torch.tensor(latlon_encoded[0], dtype=torch.float32)
        lon_vec = torch.tensor(latlon_encoded[1], dtype=torch.float32)
        latlon_vec = torch.cat([lat_vec, lon_vec], dim=-1)
        latlons.append(latlon_vec)

    # stack across batch dimension
    images = torch.stack(images).to(device)        # [B,C,H,W]
    times = torch.stack(times).to(device)          # [B,T]
    latlons = torch.stack(latlons).to(device)      # [B,4]

    return {
        "platform": "sentinel-2-l2a",
        "image": images,
        "time": times,
        "latlon": latlons,
        "gsd": gsd.to(device),
        "waves": waves.to(device),
    }



# ============================================================
# 5️⃣ MODEL + PREPROCESS WRAPPER
# ============================================================
def get_model_and_preprocess(model_name: str, device: torch.device, metadata_path: str):
    """Return encoder, preprocessing function, and metadata tensors."""
    model_name = model_name.lower()

    # -------------------- CLAY --------------------
    if model_name == "clay":
        weights = "/projects/bdbk/subashk/ckpts/CLAY/clay-v1.5.ckpt"
        metadata = Box(yaml.safe_load(open(metadata_path, "r")))
        platform = "sentinel-2-l2a"
        bands = ["red", "green", "blue", "nir"]

        mean = torch.tensor([metadata[platform].bands.mean[str(b)] for b in bands])
        std = torch.tensor([metadata[platform].bands.std[str(b)] for b in bands])
        preprocess_fn = preprocess_clay(mean, std)

        wavelength = [metadata[platform].bands.wavelength[str(b)] for b in bands]
        gsd = torch.tensor(metadata[platform].gsd, dtype=torch.float32)
        waves = torch.tensor(wavelength, dtype=torch.float32)

        encoder = ClayEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
            ckpt_path=weights,
            freeze_encoder="all",
        ).to(device)
        encoder.eval()
        return encoder, preprocess_fn, gsd, waves

    # -------------------- TERRAFM --------------------
    elif model_name == "terrafm":
        weights = "/projects/bdbk/subashk/ckpts/TERRAFM/TerraFM-B.pth"
        preprocess_fn = preprocess_general
        encoder = TerraFMEncoder(
            ckpt_path=weights, in_chans=4,
            device=device, freeze_encoder="all"
        ).to(device)
        encoder.eval()
        return encoder, preprocess_fn, None, None

    elif model_name == "terramind":
        encoder = TeraMindEncoder().to(device)
        encoder.eval()
        preprocess_fn = preprocess_general
        return encoder, preprocess_fn, None, None

    # -------------------- DINOV3 --------------------
    elif model_name == "dinov3":
        weights = "/projects/bdbk/subashk/ckpts/DINOV3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        preprocess_fn = preprocess_dinov3()
        encoder = DinoV3Encoder(ckpt_path=weights).to(device)
        encoder.eval()
        return encoder, preprocess_fn, None, None

    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

def get_preprocessor(model_name: str, metadata_path: str):
    """Return only the preprocessing function and metadata tensors (no model)."""
    model_name = model_name.lower()

    if model_name == "unet":
        preprocess_fn = preprocess_general
        return preprocess_fn, None, None

    # -------------------- CLAY --------------------
    elif model_name == "clay":
        metadata = Box(yaml.safe_load(open(metadata_path, "r")))
        platform = "sentinel-2-l2a"
        bands = ["red", "green", "blue", "nir"]

        mean = torch.tensor([metadata[platform].bands.mean[str(b)] for b in bands])
        std = torch.tensor([metadata[platform].bands.std[str(b)] for b in bands])
        preprocess_fn = preprocess_clay(mean, std)

        wavelength = [metadata[platform].bands.wavelength[str(b)] for b in bands]
        gsd = torch.tensor(metadata[platform].gsd, dtype=torch.float32)
        waves = torch.tensor(wavelength, dtype=torch.float32)

        return preprocess_fn, gsd, waves

    # -------------------- TERRAFM --------------------
    elif model_name == "terrafm":
        preprocess_fn = preprocess_general
        return preprocess_fn, None, None
    
    # -------------------- TERRAFM --------------------
    elif model_name == "terramind":
        preprocess_fn = preprocess_general
        return preprocess_fn, None, None

    # -------------------- DINOV3 --------------------
    elif model_name == "dinov3":
        preprocess_fn = preprocess_dinov3()
        return preprocess_fn, None, None

    else:
        raise ValueError(f"Unsupported model: {model_name}")