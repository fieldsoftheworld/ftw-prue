import hashlib
import logging
import os
from typing import Union
import click
import pandas as pd
import scipy
import scipy.stats
import xarray as xr
import rasterio
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime, timedelta
from functools import lru_cache
import math
from box import Box
import yaml
# Harvest day raster paths from https://github.com/ucg-uv/research_products/tree/main
SUMMER_START_RASTER_PATH = "assets/global_crop_calendar/sc_sos_3x3_v2_cog.tiff"
SUMMER_END_RASTER_PATH = "assets/global_crop_calendar/sc_eos_3x3_v2_cog.tiff"

logger = logging.getLogger()


def normalize_timestamp(date: datetime, hour: bool = False):
    """Encodes a date/time into sine and cosine components for continuity."""
    week = date.isocalendar().week * 2 * np.pi / 52
    
    if hour:
        hour_val = date.hour * 2 * np.pi / 24
    else:
        # Approximate 12pm (noon) to be default time if time is not given
        hour_val = 12 * 2 * np.pi / 24 

    return (math.sin(week), math.cos(week)), (math.sin(hour_val), math.cos(hour_val))

def normalize_latlon(lat: float, lon: float):
    """Encodes latitude and longitude into sine and cosine components."""
    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180

    return (math.sin(lat_rad), math.cos(lat_rad)), (math.sin(lon_rad), math.cos(lon_rad))

def get_median_date(start: str, end: str) -> datetime:
    """Calculates the date at the midpoint of a start/end string date range."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    delta_days = (end_dt - start_dt).days
    median_date = start_dt + timedelta(days=delta_days // 2)

    return median_date

@lru_cache(maxsize=None)
def extract_season_dates(json_path):
    """
    Loads season dates from a JSON file. Cached by lru_cache to prevent
    re-reading the same file repeatedly.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    seasons = data["seasons"]

    # Handle both dict and list of dicts
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

def load_image(path: str, select_rgb: bool = False):
    """Load a single Sentinel-2 image and return image tensor and center lat/lon."""
    with rasterio.open(path) as f:
        img = f.read().astype(np.float32)
        r, c = f.height // 2, f.width // 2
        lon, lat = rasterio.transform.xy(f.transform, r, c)

    if select_rgb:
        img = img[:3]

    return torch.from_numpy(img), lat, lon


def load_with_metadata(
    image_path: str,
    root_data_path: str,
    gsd: torch.Tensor,
    waves: torch.Tensor,
    metadata_on: bool = True,
):
    """
    Loads a single Sentinel-2 image sample, preprocesses it, and optionally
    extracts image-level metadata (temporal and spatial encodings).
    """
    image_path_lower = str(image_path).lower()

    # --- 1. Load and Preprocess Image ---
    # NOTE: load_image is updated to return a torch.Tensor
    image_tensor, lat, lon = load_image(image_path)
    sample = {"image": image_tensor}
    
    image_tensor = sample["image"] 

    # --- 2. Handle Metadata Extraction (Conditional) ---
    if not metadata_on:
        return {"image": image_tensor}

    # Infer Window Type
    if "window_a" in image_path_lower:
        window_type = "a"
    elif "window_b" in image_path_lower:
        window_type = "b"
    else:
        raise ValueError(f"Cannot infer window type from path: {image_path}")

    # Determine Country and Get Timestamps (Uses CACHED function)
    country = Path(image_path).parts[-4]
    json_fn = Path(root_data_path) / country / f"data_config_{country}.json"
        
    # extract_season_dates is now cached!
    times_json = extract_season_dates(str(json_fn))
    timestamp = times_json[f"window_{window_type}"]

    # Temporal Encoding
    week_norm, hour_norm = normalize_timestamp(timestamp)
    time_vec = torch.tensor(list(week_norm) + list(hour_norm), dtype=torch.float32)

    # Spatial Encoding
    latlon_encoded = normalize_latlon(lat, lon)
    lat_vec = torch.tensor(latlon_encoded[0], dtype=torch.float32)
    lon_vec = torch.tensor(latlon_encoded[1], dtype=torch.float32)
    latlon_vec = torch.cat([lat_vec, lon_vec], dim=-1)

    # --- 3. Return Full Metadata Dictionary ---
    return {
        "platform": "sentinel-2-l2a",
        "image": image_tensor,
        "time": time_vec,
        "latlon": latlon_vec,
        "gsd": gsd,
        "waves": waves,
    }


# ============================================================
# Preprocessors for different GFMs
# ============================================================

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
        image = sample["image"][:3, :, :] / self.norm_constant        # [C,H,W]
        mean = self.mean.to(image.device).view(-1,1,1)  # [C,1,1]
        std  = self.std.to(image.device).view(-1,1,1)
        sample["image"] = (image - mean) / std
        return sample


class preprocess_clay:
    def __init__(self, metadata_path):
        metadata = Box(yaml.safe_load(open(metadata_path, "r")))
        platform = "sentinel-2-l2a"
        bands = ["red", "green", "blue", "nir"] 

        self.mean = torch.tensor([metadata[platform].bands.mean[str(b)] for b in bands])
        self.std = torch.tensor([metadata[platform].bands.std[str(b)] for b in bands])

        wavelength = [metadata[platform].bands.wavelength[str(b)] for b in bands]
        self.gsd = torch.tensor(metadata[platform].gsd, dtype=torch.float32)
        self.waves = torch.tensor(wavelength, dtype=torch.float32)

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        mean = self.mean.to(image.device).view(-1, 1, 1)
        std = self.std.to(image.device).view(-1, 1, 1)
        sample["image"] = (image - mean) / std
        return sample


def preprocess_none(sample: dict, **kwargs) -> dict:
    """No preprocessing; return sample as-is."""
    return sample

class PreProcessorWrapper:
    def __init__(self, preprocess_type: str, **kwargs):
        preprocessor_map = {
            "none": preprocess_none,
            "terrafm": preprocess_general,
            "terramind": preprocess_general,
            "dinov3": preprocess_dinov3,
            "clay": preprocess_clay,
        }

        Preprocessor = preprocessor_map[preprocess_type]

        self.kwargs = kwargs

        if isinstance(Preprocessor, type):
            self.preprocessor = Preprocessor(**kwargs)
            self.is_function = False
        else:
            self.preprocessor = Preprocessor
            self.is_function = True

    def __call__(self, sample: dict) -> dict:
        if self.is_function:
            return self.preprocessor(sample, **self.kwargs)
        else:
            return self.preprocessor(sample)


    @property
    def gsd(self) -> Union[torch.Tensor, None]:
        """Returns the GSD tensor if available on the underlying preprocessor."""
        return getattr(self.preprocessor, 'gsd', None)

    @property
    def waves(self) -> Union[torch.Tensor, None]:
        """Returns the WAVES tensor if available on the underlying preprocessor."""
        return getattr(self.preprocessor, 'waves', None)

# ============================================================
# Some utilities for dataset/datamodule
# ============================================================

def compute_md5(file_path):
    """Compute the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except FileNotFoundError:
        return None
    return hash_md5.hexdigest()


def validate_checksums(checksum_file, root_directory):
    """Validate checksums stored in a checksum file."""
    if not os.path.isfile(checksum_file):
        print(f"Checksum file not found: {checksum_file}")
        return

    with open(checksum_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        stored_checksum, file_path = parts
        file_path = os.path.join(root_directory, file_path)
        current_checksum = compute_md5(file_path)

        if current_checksum != stored_checksum:
            print("Checksum mismatch: {file_path}")
            return False
    return True


def harvest_to_datetime(harvest_day: int, year: int) -> pd.Timestamp:
    """
    Convert a harvest integer (day of the year) to a datetime object.

    Args:
        harvest_day (int): Day of the year (1-365).
        year (int): The year for which the date is to be calculated.

    Returns:
        pd.Timestamp: Corresponding datetime object.
    """
    return pd.to_datetime(f"{year}-{harvest_day}", format="%Y-%j")


# to-do func to get harvest integer from user provided bbox
def get_harvest_integer_from_bbox(
    bbox: list[float],
    start_year_raster_path: str = SUMMER_START_RASTER_PATH,
    end_year_raster_path: str = SUMMER_END_RASTER_PATH,
) -> list[int]:
    """
    Gets harvest integer from a user-provided bounding box. Note currently just uses summer crops.

    Args:
        bbox (str): Bounding box in the format 'minx,miny,maxx,maxy'.

    Returns:
        list: start and end harvest integer (day of the year).
    """

    start_harvest_dset = xr.open_dataset(start_year_raster_path, engine="rasterio")
    end_harvest_dset = xr.open_dataset(end_year_raster_path, engine="rasterio")

    # Clip the datasets to the bounding box
    start_value = start_harvest_dset.rio.clip_box(
        bbox[0], bbox[1], bbox[2], bbox[3], allow_one_dimensional_raster=True
    )

    if len(start_value["band_data"][0][0]) > 1:
        start_days = start_value["band_data"].values[0][0]
        logger.info(
            f"Multiple dates found in area of interest {start_days}. Using circular mean to determine harvest day."
        )
        start_value = int(round(scipy.stats.circmean(start_days, high=365, low=1)))
    else:
        start_value = int(start_value["band_data"].values[0][0][0])

    end_value = end_harvest_dset.rio.clip_box(
        bbox[0], bbox[1], bbox[2], bbox[3], allow_one_dimensional_raster=True
    )
    if len(end_value["band_data"][0][0]) > 1:
        end_days = end_value["band_data"].values[0][0]
        logger.info(
            f"Multiple dates found in area of interest {end_days}. Using circular mean to determine harvest day."
        )
        end_value = int(round(scipy.stats.circmean(end_days, high=365, low=1)))

    else:
        end_value = int(end_value["band_data"].values[0][0][0])

    return [start_value, end_value]


def parse_bbox(ctx, param, value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise click.BadParameter("Bounding box must be a string")
    values = value.split(",")
    if len(values) != 4:
        raise click.BadParameter("Bounding box must contain exactly 4 values")
    for i, v in enumerate(values):
        try:
            values[i] = float(v)
        except ValueError:
            raise click.BadParameter(
                f"Invalid value for element {i} in bounding box: {v}"
            )
    return values
