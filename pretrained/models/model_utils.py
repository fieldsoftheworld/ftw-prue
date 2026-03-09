"""
Self-contained utilities for FTW embedding extraction with pretrained encoders.
"""

import os
import json
import torch
import rasterio
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from box import Box
import math

from .clay.finetune.segment.factory import SegmentEncoder as ClayEncoder
from .TerraFM.terrafm_segment import TerraFMEncoderWrapper as TerraFMEncoder
from .dinov3.dinov3_segmentor import SegmentEncoder as DinoV3Encoder
from .terramind.terramind import SegmentEncoder as TeraMindEncoder
from ..path_config import get_model_path, get_data_root, get_metadata_path


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
        self.std = torch.as_tensor(std if std is not None else [0.213, 0.156, 0.143], dtype=torch.float32)
        self.norm_constant = norm_constant

    def __call__(self, sample: dict) -> dict:
        image = sample["image"][:3, :, :] / self.norm_constant
        mean = self.mean.to(image.device).view(-1, 1, 1)
        std = self.std.to(image.device).view(-1, 1, 1)
        sample["image"] = (image - mean) / std
        return sample


class preprocess_clay:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        mean = self.mean.to(image.device).view(-1, 1, 1)
        std = self.std.to(image.device).view(-1, 1, 1)
        sample["image"] = (image - mean) / std
        return sample


class preprocess_galileo:
    """Preprocessing for Galileo benchmark models (identity - handled in wrapper)."""
    def __call__(self, sample: dict) -> dict:
        return sample


def normalize_timestamp(date, hour=False):
    week = date.isocalendar().week * 2 * np.pi / 52
    if hour:
        hour = date.hour * 2 * np.pi / 24
    else:
        hour = 12

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
    if isinstance(seasons, list):
        seasons = seasons[0]

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
    data_root: str = None,
):
    """
    Prepare a Sentinel-2 image sample for CLAY encoder inference.
    Automatically infers window type (A/B) and extracts temporal + spatial encodings.

    Args:
        image_path: Path to a Sentinel-2 image (.tif)
        preprocess: normalization transform (preprocess_clay instance)
        gsd: Ground Sampling Distance tensor
        waves: Band wavelength tensor
        data_root: Root directory for FTW data (defaults to path_config.get_data_root())

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
    if data_root is None:
        data_root = str(get_data_root())

    image_path_lower = str(image_path).lower()
    if "window_a" in image_path_lower:
        window_type = "a"
    elif "window_b" in image_path_lower:
        window_type = "b"
    else:
        raise ValueError(f"Cannot infer window type from path: {image_path}")

    country = Path(image_path).parts[-4]
    json_fn = f"{data_root}/{country}/data_config_{country}.json"
    times_json = extract_season_dates(json_fn)
    timestamp = times_json[f"window_{window_type}"]

    image, lat, lon = load_image(image_path)
    image = preprocess({"image": image})["image"]

    week_norm, hour_norm = normalize_timestamp(timestamp)
    time_vec = torch.tensor(list(week_norm) + list(hour_norm), dtype=torch.float32)

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
    data_root: str = None,
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
    if data_root is None:
        data_root = str(get_data_root())
    
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

        image, lat, lon = load_image(image_path)
        sample = {"image": image}
        sample = preprocess(sample)
        images.append(sample["image"])

        week_norm, hour_norm = normalize_timestamp(timestamp)
        time_vec = torch.tensor(list(week_norm) + list(hour_norm), dtype=torch.float32)
        times.append(time_vec)

        latlon_encoded = normalize_latlon(lat, lon)
        lat_vec = torch.tensor(latlon_encoded[0], dtype=torch.float32)
        lon_vec = torch.tensor(latlon_encoded[1], dtype=torch.float32)
        latlon_vec = torch.cat([lat_vec, lon_vec], dim=-1)
        latlons.append(latlon_vec)

    images = torch.stack(images).to(device)
    times = torch.stack(times).to(device)
    latlons = torch.stack(latlons).to(device)

    return {
        "platform": "sentinel-2-l2a",
        "image": images,
        "time": times,
        "latlon": latlons,
        "gsd": gsd.to(device),
        "waves": waves.to(device),
    }


def get_model_and_preprocess(model_name: str, device: torch.device, metadata_path: str = None, weights_path: str = None):
    """
    Return encoder, preprocessing function, and metadata tensors.
    
    Args:
        model_name: Name of the model
        device: torch.device
        metadata_path: Path to metadata YAML (defaults to path_config.get_metadata_path())
        weights_path: Path to model weights (defaults to path_config.get_model_path())
    
    Returns:
        Tuple of (encoder, preprocess_fn, gsd, waves)
    """
    model_name = model_name.lower()
    
    if metadata_path is None:
        metadata_path = str(get_metadata_path())
    
    if weights_path is None:
        weights_path = str(get_model_path(model_name))

    if model_name == "clay":
        weights = weights_path
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

    elif model_name == "terrafm":
        weights = weights_path
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

    elif model_name == "dinov3":
        weights = weights_path
        preprocess_fn = preprocess_dinov3()
        encoder = DinoV3Encoder(ckpt_path=weights).to(device)
        encoder.eval()
        return encoder, preprocess_fn, None, None

    elif model_name == "croma":
        from .galileo_benchmark.galileo_wrappers import CROMAEncoder
        weights = weights_path
        preprocess_fn = preprocess_galileo()
        encoder = CROMAEncoder(
            ckpt_path=weights,
            size="base",
            freeze_encoder="all",
            device=device
        )
        encoder.eval()
        return encoder, preprocess_fn, None, None

    elif model_name == "decur":
        from .galileo_benchmark.galileo_wrappers import DeCurEncoder
        weights = weights_path
        preprocess_fn = preprocess_galileo()
        encoder = DeCurEncoder(
            ckpt_path=weights,
            freeze_encoder="all",
            device=device
        )
        encoder.eval()
        return encoder, preprocess_fn, None, None

    elif model_name == "dofa":
        from .galileo_benchmark.galileo_wrappers import DOFAEncoder
        weights = weights_path
        preprocess_fn = preprocess_galileo()
        encoder = DOFAEncoder(
            ckpt_path=weights,
            size="base",
            freeze_encoder="all",
            device=device
        )
        encoder.eval()
        return encoder, preprocess_fn, None, None

    elif model_name == "prithvi":
        from .galileo_benchmark.galileo_wrappers import PrithviEncoder
        weights = weights_path
        preprocess_fn = preprocess_galileo()
        encoder = PrithviEncoder(
            ckpt_path=weights,
            freeze_encoder="all",
            device=device
        )
        encoder.eval()
        return encoder, preprocess_fn, None, None

    elif model_name == "satlas":
        from .galileo_benchmark.galileo_wrappers import SatlasEncoder
        weights = weights_path
        preprocess_fn = preprocess_galileo()
        encoder = SatlasEncoder(
            ckpt_path=weights,
            size="base",
            freeze_encoder="all",
            device=device
        )
        encoder.eval()
        return encoder, preprocess_fn, None, None

    elif model_name == "softcon":
        from .galileo_benchmark.galileo_wrappers import SoftConEncoder
        weights = weights_path
        preprocess_fn = preprocess_galileo()
        encoder = SoftConEncoder(
            ckpt_path=weights,
            size="base",
            freeze_encoder="all",
            device=device
        )
        encoder.eval()
        return encoder, preprocess_fn, None, None

    elif model_name == "galileo":
        from .galileo_benchmark.galileo_wrappers import GalileoEncoder
        weights = weights_path
        preprocess_fn = preprocess_galileo()
        encoder = GalileoEncoder(
            ckpt_path=weights,
            freeze_encoder="all",
            device=device
        )
        encoder.eval()
        return encoder, preprocess_fn, None, None

    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

def get_preprocessor(preprocessing: str, metadata_path: str = None):
    """
    Return only the preprocessing function and metadata tensors (no model).
    
    Args:
        preprocessing: Name of preprocessing method
        metadata_path: Path to metadata YAML (defaults to path_config.get_metadata_path())
    
    Returns:
        Tuple of (preprocess_fn, gsd, waves)
    """
    if metadata_path is None:
        metadata_path = str(get_metadata_path())
    
    # Handle "none" string (in addition to None)
    if preprocessing == "none" or preprocessing is None:
        preprocess_fn = None
        return preprocess_fn, None, None
    
    if preprocessing == "unet":
        preprocess_fn = preprocess_general
        return preprocess_fn, None, None

    elif preprocessing == "clay":
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

    elif preprocessing == "terrafm":
        preprocess_fn = preprocess_general
        return preprocess_fn, None, None

    elif preprocessing == "terramind":
        preprocess_fn = preprocess_general
        return preprocess_fn, None, None

    elif preprocessing == "dinov3":
        preprocess_fn = preprocess_dinov3()
        return preprocess_fn, None, None

    elif preprocessing in ["croma", "decur", "dofa", "prithvi", "satlas", "softcon", "galileo"]:
        preprocess_fn = preprocess_galileo()
        return preprocess_fn, None, None
    else:
        raise ValueError(f"Unsupported preprocessing: {preprocessing}")