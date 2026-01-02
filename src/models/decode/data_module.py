import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
from scipy.ndimage import maximum_filter, minimum_filter

# Utility functions that handles presence case
## With a simple 3x3 neighborhood difference
def get_boundary(mask):
    m = mask.copy()
    m[m == 3] = 0
    field_mask = (m > 0).astype(np.uint8)

    local_max = maximum_filter(m, size=3)
    local_min = minimum_filter(m, size=3)
    boundary = ((local_max != local_min) & (field_mask > 0)).astype(np.float32)

    # valid = (m != 3)
    # boundary = ((local_max != local_min) & valid & field_mask).astype(np.float32)
    return boundary
    


## with dilation
# def get_boundary(mask, dilation_iter=1):
#     m = mask.astype(np.int32)
#     if (m == 3).any():
#         m = m.copy()
#         m[m == 3] = 0
#     binmask = (m > 0).astype(np.uint8)
#     dilated = cv2.dilate(binmask, kernel=np.ones((3, 3), np.uint8), iterations=dilation_iter)
#     boundary = dilated - binmask
#     return (boundary > 0).astype(np.float32)

# ## with distance transform for edges
# def get_boundary(mask):
#     m = mask.copy()
#     m[m == 3] = 0
#     binmask = (m > 0).astype(np.uint8)
#     distance = ndimage.distance_transform_edt(binmask)
#     boundary = ((distance > 0) & (distance <= 1)).astype(np.float32)
#     return boundary

## with morphological gradient
# def get_boundary(mask):
#     m = mask.copy()
#     m[m == 3] = 0
#     binmask = (m > 0).astype(np.uint8)
#     boundary = cv2.morphologyEx(binmask, cv2.MORPH_GRADIENT, kernel=np.ones((3, 3), np.uint8))
#     return boundary.astype(np.float32)

    
def get_distance(mask):
    m = mask.astype(np.int32)
    if (m == 3).any():
        m = m.copy()
        m[m == 3] = 0
    binmask = (m > 0).astype(np.uint8)
    distance_map = ndimage.distance_transform_edt(binmask)
    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()
    return distance_map.astype(np.float32)


class FTWMultiCountryDataset(Dataset):
    valid_countries = [
        "austria", "belgium", "brazil", "cambodia", "corsica", "croatia", "denmark",
        "estonia", "finland", "france", "germany", "india", "kenya", "latvia",
        "lithuania", "luxembourg", "netherlands", "portugal", "rwanda", "slovakia",
        "slovenia", "south_africa", "spain", "sweden", "vietnam"
    ]
    valid_splits = ["train", "val", "test"]
    valid_temporal_options = ["stacked", "windowA", "windowB"]

    def __init__(self, root_dir, countries, split="train", load_boundaries=False,
                 temporal_option="stacked", crop_size=(256, 256), num_samples=-1):

        # Normalize and validate countries
        if isinstance(countries, str):
            countries = [countries]
        for country in countries:
            assert country in self.valid_countries, f"Invalid country: {country}"
        self.countries = countries

        # Validate split and temporal option
        assert split in self.valid_splits, f"Invalid split: {split}, must be one of {self.valid_splits}"
        assert temporal_option in self.valid_temporal_options, (
            f"Invalid temporal_option: {temporal_option}, must be one of {self.valid_temporal_options}"
        )

        self.root_dir = root_dir
        self.split = split
        self.load_boundaries = load_boundaries
        self.temporal_option = temporal_option
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.file_list = []

        self._build_file_list()

        print(f"Running {self.split} for {self.countries}")

    def _build_file_list(self):
        for country in self.countries:
            country_dir = os.path.join(self.root_dir, country)
            chips_path = os.path.join(country_dir, f"chips_{country}.parquet")
            if not os.path.exists(chips_path):
                continue

            df = pd.read_parquet(chips_path)
            df = df[df["split"] == self.split]
            aoi_ids = df["aoi_id"].values

            for aoi in aoi_ids:
                harvest_path = os.path.join(country_dir, "s2_images/window_b", f"{aoi}.tif")
                planting_path = os.path.join(country_dir, "s2_images/window_a", f"{aoi}.tif")
                mask_dir = "label_masks/semantic_3class" if self.load_boundaries else "label_masks/semantic_2class"
                mask_path = os.path.join(country_dir, mask_dir, f"{aoi}.tif")

                if os.path.exists(harvest_path) and os.path.exists(planting_path) and os.path.exists(mask_path):
                    self.file_list.append({
                        "image_a": planting_path,
                        "image_b": harvest_path,
                        "mask": mask_path
                    })

        if self.num_samples != -1:
            self.file_list = random.sample(self.file_list, min(self.num_samples, len(self.file_list)))

    def __getitem__(self, idx):
        files = self.file_list[idx]

        image_a = self._read_image(files["image_a"])
        image_b = self._read_image(files["image_b"])
        mask = self._read_mask(files["mask"])

        # mask[mask == 3] = 0 # needs to be commented if I want to ignore presence only in the loss computation

        if self.temporal_option == "stacked":
            image = np.concatenate([image_a, image_b], axis=0)
            # image = np.concatenate([image_b, image_a], axis=0)
        elif self.temporal_option == "windowA":
            image = image_a
        elif self.temporal_option == "windowB":
            image = image_b

        boundary = get_boundary(mask[0])
        distance = get_distance(mask[0])

        # Convert to torch.Tensor
        return (
            torch.from_numpy(image_a),
            torch.from_numpy(image_b),
            torch.from_numpy(image),
            torch.from_numpy(mask),
            torch.from_numpy(np.expand_dims(boundary, 0)),
            torch.from_numpy(np.expand_dims(distance, 0))
        )

    def __len__(self):
        return len(self.file_list)

    def _read_image(self, path):
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
            img = img.astype(np.float32) / 3000.0
        return img

    def _read_mask(self, path):
        with rasterio.open(path) as src:
            mask = src.read(1)
            mask = mask.astype(np.float32)
        return np.expand_dims(mask, axis=0)
