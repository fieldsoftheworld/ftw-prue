"""FTW dataset."""

import os
import random
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import rasterio
import torch
from scipy.ndimage import maximum_filter, minimum_filter, distance_transform_edt
from torch import Tensor
from torchgeo.datasets import NonGeoDataset, RasterDataset

from ftw_tools.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS
from ftw_tools.utils import validate_checksums
from pretrained.models.model_utils import get_preprocessor, prepare_clay_sample, prepare_general_sample


def get_boundary(mask):
    """Compute boundary from mask."""
    m = mask.copy()
    m[m == 3] = 0
    field_mask = (m > 0).astype(np.uint8)
    local_max = maximum_filter(m, size=3)
    local_min = minimum_filter(m, size=3)
    boundary = ((local_max != local_min) & (field_mask > 0)).astype(np.float32)
    return boundary


def get_distance(mask):
    """Compute distance map from mask."""
    m = mask.astype(np.int32)
    if (m == 3).any():
        m = m.copy()
        m[m == 3] = 0
    binmask = (m > 0).astype(np.uint8)
    distance_map = distance_transform_edt(binmask)
    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()
    return distance_map.astype(np.float32)


def sample_points_from_mask(mask, n=1, is_training=True):
    """Sample points from binary field mask for SAM-2 prompts.
    
    Args:
        mask: Binary field mask (numpy array, float32, 0 or 1)
        n: Number of points to sample (default: 1 for training, 3 for testing)
        is_training: If True, sample single point; if False, sample n points
    
    Returns:
        points: Array of shape [n, 2] with (x, y) coordinates, or None if no points
        labels: Array of shape [n] with all 1s (positive points), or None
    """
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None, None
    
    if is_training:
        # Training: sample single point
        idx = np.random.randint(len(xs))
        points = np.array([[xs[idx], ys[idx]]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
    else:
        # Testing: sample n points
        n_samples = min(n, len(xs))
        idx = np.random.choice(len(xs), size=n_samples, replace=False)
        points = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
        labels = np.ones(len(points), dtype=np.int32)
    
    return points, labels


def prepare_binary_field_mask(mask):
    """Convert 3-class mask to binary field mask (classes 1 and 2 -> 1, else 0).
    
    Args:
        mask: 3-class mask (numpy array)
    
    Returns:
        Binary field mask (float32, 0 or 1)
    """
    return ((mask == 1) | (mask == 2)).astype(np.float32)

class SingleRasterDataset(RasterDataset):
    """A torchgeo dataset that loads a single raster file."""

    def __init__(self, fn: str, transforms: Optional[Callable] = None):
        """Initialize the SingleRasterDataset class.

        Args:
            fn (str): The path to the raster file.
            transforms (Optional[Callable], optional): The transforms to apply to the
                raster file. Defaults to None.
        """
        path = os.path.abspath(fn)
        self.filename_regex = os.path.basename(path)
        super().__init__(paths=os.path.dirname(path), transforms=transforms)


def reshape_feat(feat):
    if feat.dim() == 3:
        d, h, w = feat.shape
        feat = feat.permute(1, 2, 0).reshape(h * w, d)
    return feat


class FTW(NonGeoDataset):
    valid_splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = "data/ftw",
        countries: Union[Sequence[str], str] = None,
        split: str = "train",
        preprocessing: str = "none",
        checksum: bool = False,
        load_boundaries: bool = False,
        temporal_options: str = "stacked",
        swap_order: bool = False,
        num_samples: int = -1,
        input_type: str = "images",
        feat_root: Optional[str] = None,
        metadata_path: str = None,
        sam2_max_image_size: int = 1024,
        sam2_num_points: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize a new FTW dataset instance.

        Args:
            root: root directory where dataset can be found, this should contain the
                country folder
            countries: the countries to load the dataset from, e.g. "france"
            split: string specifying what split to load (e.g. "train", "val", "test")
            preprocessing: a string specifying the preprocessing to apply to each
                entry and returns a transformed version , options are "none", "ftw", or gfm model name like | clay | croma | decur | dofa | dinov3 | galileo | prithvi | satlas | softcon | terrafm | terramind
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            load_boundaries: if True, load the 3 class masks with boundaries
            temporal_options : for abalation study, valid option are (stacked, windowA, windowB, median, rgb, random_window)
            swap_order: if True, swap the order of temporal data (i.e. use window A first)
            input_type: if "images" we are using raw images, if "features" we are using precomputed features , "images_noaug" for GFM experiments
            feat_root: root directory where precomputed features are stored
            metadata_path: path to metadata file
        Raises:
            AssertionError: if ``countries`` argument is invalid
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if data is not found, or checksums don't match
        """
        self.root = root
        self.input_type = input_type
        self.feat_root = feat_root
        if countries is None:
            raise ValueError("Please specify the countries to load the dataset from")
        if temporal_options not in TEMPORAL_OPTIONS:
            raise ValueError(f"Invalid temporal option {temporal_options}")

        if isinstance(countries, str):
            countries = [countries]
        countries = [country.lower() for country in countries]
        for country in countries:
            assert country in ALL_COUNTRIES, f"Invalid country {country}"

        self.countries = countries
        assert split in self.valid_splits
        self.preprocessing = preprocessing
        self.preprocessor, self.gsd, self.waves = get_preprocessor(preprocessing=preprocessing, metadata_path=metadata_path)
        self.checksum = checksum
        self.load_boundaries = load_boundaries
        self.temporal_options = temporal_options
        self.num_samples = num_samples
        self.swap_order = swap_order
        self.split = split
        self.compute_boundary_distance = False
        self.sam2_mode = (temporal_options == "sam2")
        # sam2_max_image_size is kept for backward compatibility but not used (FTW images are 256x256)
        self.sam2_max_image_size = sam2_max_image_size
        self.sam2_num_points = sam2_num_points
        if metadata_path != None and self.preprocessing == "clay":
            self.with_metadata = True
        else:
            self.with_metadata = False
            self.metadata_path = metadata_path
        print("METADATA used: ",  self.with_metadata)
        print("Preprocessing method: ", self.preprocessing)

        if self.load_boundaries:
            print("Loading 3 Class Masks, with Boundaries")
        else:
            print("Loading 2 Class Masks, without Boundaries")

        print("Temporal option: ", temporal_options)
        if swap_order:
            if temporal_options not in ("stacked", "rgb"):
                raise ValueError(
                    "Can only use swap_order with temporal_options stacked or rgb"
                )
            print("Using window A first, then window B")
        else:
            print("Using window B first, then window A")

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found at root directory or corrupted.  Download dataset with `ftw data download`"
            )

        if checksum:
            assert self._checksum(), "Checksum of dataset does not match"

        self.img_filenames = []
        all_img_filenames = []
        self.feat_filenames = []
        all_feat_filenames = []
        ignore_list = ["g14-2_00059_7","g50_00037_0","g2_00036_9","g25_00014_11"] # filter out some bad samples for which I had corrupted features extracted/saved??
        
        for country in self.countries:
            country_root = os.path.join(self.root, country)
            chips_fn = os.path.join(country_root, f"chips_{country}.parquet")
            chips_df = gpd.read_parquet(str(chips_fn))
            chips_df = chips_df[chips_df["split"] == split]
            aoi_ids = chips_df["aoi_id"].values
            aoi_ids = [id for id in aoi_ids if id not in ignore_list]  
            for idx in aoi_ids:
                window_b_fn = Path(
                    os.path.join(country_root, "s2_images/window_b", f"{idx}.tif")
                )
                window_a_fn = Path(
                    os.path.join(country_root, "s2_images/window_a", f"{idx}.tif")
                )
                masks_2c_fn = Path(
                    os.path.join(
                        country_root, "label_masks/semantic_2class", f"{idx}.tif"
                    )
                )
                masks_3c_fn = Path(
                    os.path.join(
                        country_root, "label_masks/semantic_3class", f"{idx}.tif"
                    )
                )

                # Skip the image AOI's which does not have all four corresponding files
                if not (
                    window_b_fn.exists()
                    and window_a_fn.exists()
                    and masks_2c_fn.exists()
                    and masks_3c_fn.exists()
                ):
                    continue

                if self.load_boundaries:
                    mask_fn = os.path.join(
                        country_root, "label_masks/semantic_3class", f"{idx}.tif"
                    )
                else:
                    mask_fn = os.path.join(
                        country_root, "label_masks/semantic_2class", f"{idx}.tif"
                    )

                if os.path.exists(mask_fn):
                    if "images" in self.input_type:
                        all_img_filenames.append(
                            {
                                "window_b": os.path.join(
                                    country_root, "s2_images/window_b", f"{idx}.tif"
                                ),
                                "window_a": os.path.join(
                                    country_root, "s2_images/window_a", f"{idx}.tif"
                                ),
                                "mask": mask_fn,
                            }
                        )
                    if "features" in self.input_type:
                        country = Path(country_root).name
                        model = Path(self.feat_root).name
                        country_feat_root = os.path.join(self.feat_root, country)
                        all_feat_filenames.append(
                            {
                                "window_b_feats": os.path.join(
                                    country_feat_root, "window_b", f"{model}_{idx}.npz"
                                ),
                                "window_a_feats": os.path.join(
                                    country_feat_root, "window_a", f"{model}_{idx}.npz"
                                ),
                                "mask": mask_fn,
                            }
                        )


        if self.num_samples == -1:  # select all samples
            self.img_filenames = all_img_filenames
            self.feat_filenames = all_feat_filenames
        else:
            raise ValueError("Currently only -1 (all samples) is supported for num_samples")
           
        print("Selecting : ", len(self.feat_filenames), " feat samples")
        print("Selecting : ", len(self.img_filenames), "  image samples")

    def _checksum(self) -> bool:
        """Check the checksum of the dataset.

        Returns:
            True if the checksum matches, else False
        """
        for country in ALL_COUNTRIES:
            print(f"Validating checksums for {country}")
            for checksum_file in [
                "distances_checksums.md5",
                "masks_checksums.md5",
                "window_b_checksums.md5",
                "window_a_checksums.md5",
            ]:
                checksum_file = os.path.join(self.root, country, checksum_file)
                if not os.path.exists(checksum_file):
                    print(f"Checksum file {checksum_file} not found")
                    return False
                if not validate_checksums(checksum_file, self.root):
                    return False
        return True

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """

        for country in self.countries:
            if country not in ALL_COUNTRIES:
                print(f"Invalid country {country}")
                return False

            country_dir: str = os.path.join(self.root, country)
            if not os.path.exists(country_dir):
                print(f"Country directory {country_dir} not found")
                return False

            chips_fns = list(Path(country_dir).glob(f"chips_*.parquet"))
            if len(chips_fns) != 1:
                print(f"Country {country} does not have chips file")
                return False

            if self.load_boundaries:
                if not all(
                    [
                        os.path.exists(os.path.join(country_dir, "s2_images/window_b")),
                        os.path.exists(os.path.join(country_dir, "s2_images/window_a")),
                        os.path.exists(
                            os.path.join(country_dir, "label_masks/semantic_3class")
                        ),
                    ]
                ):
                    print(f"Country {country} does not have all required directories")
                    return False
            else:
                if not all(
                    [
                        os.path.exists(os.path.join(country_dir, "s2_images/window_b")),
                        os.path.exists(os.path.join(country_dir, "s2_images/window_a")),
                        os.path.exists(
                            os.path.join(country_dir, "label_masks/semantic_2class")
                        ),
                    ]
                ):
                    print(f"Country {country} does not have all required directories")
                    return False
        return True

    def __len__(self) -> int:
        if "features" in self.input_type:
            return len(self.feat_filenames)
        else:
            return len(self.img_filenames)
    

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            dictionary containing "image" and "mask" PyTorch tensors
        """
        
        sample = {}
        if "features" in self.input_type:
            feat_filenames = self.feat_filenames[index]

            window_b_feat = np.load(feat_filenames["window_b_feats"])["embedding"]  
            window_a_feat = np.load(feat_filenames["window_a_feats"])["embedding"]


            window_b_feat = reshape_feat(torch.from_numpy(window_b_feat).float())
            window_a_feat = reshape_feat(torch.from_numpy(window_a_feat).float())

            # Handle temporal options just like image mode
            if self.temporal_options == "stacked":
                feat = torch.stack([window_b_feat, window_a_feat], dim=0)
            elif self.temporal_options == "windowA":
                feat = window_a_feat
            elif self.temporal_options == "windowB":
                feat = window_b_feat
            elif self.temporal_options == "random_window":
                feat = window_a_feat if random.random() < 0.5 else window_b_feat
            elif self.temporal_options == "median":
                feat = (window_a_feat + window_b_feat) / 2
            else:
                raise ValueError(f"Unsupported temporal option for features: {self.temporal_options}")

            with rasterio.open(feat_filenames["mask"]) as f:
                mask = torch.from_numpy(f.read(1)).long()

            sample["feat"] = feat
            sample["mask"] = mask

        if "images" in self.input_type:
            img_filenames = self.img_filenames[index]
            
            # --- SAM-2 Special Handling (check early to bypass normal loading) ---
            if self.sam2_mode:
                # SAM-2 needs raw int16 data, normalize by 3000
                # FTW images are 256x256, so no resizing needed
                # Read raw data directly from files (bypass preprocessing)
                with rasterio.open(img_filenames["window_a"]) as f:
                    img_a_raw = f.read()[:3].astype(np.float32)  # [C, H, W], int16 -> float32
                with rasterio.open(img_filenames["window_b"]) as f:
                    img_b_raw = f.read()[:3].astype(np.float32)  # [C, H, W], int16 -> float32
                
                # Load mask
                with rasterio.open(img_filenames["mask"]) as f:
                    mask = f.read(1)
                
                # Normalize by 3000 (FTW standard normalization for int16 data)
                FTW_NORM_CONST = 3000.0
                img_a_np = (img_a_raw / FTW_NORM_CONST).transpose(1, 2, 0)  # [H, W, C], now [0, 1]
                img_b_np = (img_b_raw / FTW_NORM_CONST).transpose(1, 2, 0)  # [H, W, C], now [0, 1]
                
                # No resizing needed - FTW images are already 256x256
                mask_resized = mask
                
                # Clip to [0, 1] range (in case of any floating point issues)
                img_a_np = np.clip(img_a_np, 0.0, 1.0)
                img_b_np = np.clip(img_b_np, 0.0, 1.0)
                
                # Convert to uint8 [0, 255] for storage (matching original SAM-2 scripts)
                # This allows us to match the original implementation while preserving normalization
                img_a_np = (img_a_np * 255.0).astype(np.uint8)
                img_b_np = (img_b_np * 255.0).astype(np.uint8)
                
                # Convert to tensors [C, H, W] as float (will be divided by 255 in trainer)
                img_a = torch.from_numpy(img_a_np).permute(2, 0, 1).float()
                img_b = torch.from_numpy(img_b_np).permute(2, 0, 1).float()
                
                # Prepare binary field mask
                field_mask = prepare_binary_field_mask(mask_resized)
                
                # Sample points (training: 1 point, validation/test: sam2_num_points)
                is_training = (self.split == "train")
                points, labels = sample_points_from_mask(
                    field_mask, 
                    n=self.sam2_num_points, 
                    is_training=is_training
                )
                
                # Store SAM-2 specific data
                sample["window_a"] = img_a
                sample["window_b"] = img_b
                sample["field_mask"] = torch.from_numpy(field_mask).float()
                sample["mask_3class"] = torch.from_numpy(mask_resized).long()
                if points is not None:
                    sample["points"] = torch.from_numpy(points).float()
                    sample["point_labels"] = torch.from_numpy(labels).long()
                else:
                    # No points available (empty mask)
                    sample["points"] = None
                    sample["point_labels"] = None
                
                # Also store regular mask for compatibility
                sample["mask"] = sample["field_mask"].long()
                
                return sample
            
            # --- Normal image loading path (for non-SAM-2 modes) ---
            # Lists to collect the components
            images = []
            time_vectors = [] 
            metadata_dict = None

            current_gsd = self.gsd
            current_waves = self.waves
            
            # --- 1. Determine and Load Windows ---
            windows_to_load = []
            if self.temporal_options in ("stacked", "median", "windowB", "rgb"):
                windows_to_load.append("window_b")
            if self.temporal_options in ("stacked", "median", "windowA", "rgb"):
                windows_to_load.append("window_a")
            if self.temporal_options == "random_window":
                windows_to_load.append("window_a" if random.random() < 0.5 else "window_b")
            
            # Process the determined windows
            for window_key in windows_to_load:
                # Load the data dictionary
                if self.preprocessing == "clay":
                    data_dict = prepare_clay_sample(
                        image_path=img_filenames[window_key], 
                        preprocess=self.preprocessor,
                        gsd=current_gsd,                   
                        waves=current_waves,
                        data_root=self.root,               
                    )
                else:
                    data_dict = prepare_general_sample(
                        image_path=img_filenames[window_key], 
                        preprocess=self.preprocessor,
                    )
        
                # Always collect the image tensor
                images.append(data_dict['image'])
                
                # Conditionally collect time vector and static metadata
                if self.with_metadata:
                    time_vectors.append(data_dict["time"])
                    if metadata_dict is None:
                        # Store the first dictionary for static metadata
                        metadata_dict = data_dict 

            # Handle swapping order
            if self.swap_order and len(images) == 2:
                images = [images[1], images[0]]
                if self.with_metadata:
                    time_vectors = [time_vectors[1], time_vectors[0]]

            # --- 2. Load Mask for regular processing ---
            with rasterio.open(img_filenames["mask"]) as f:
                mask = f.read(1)

            # --- 3. Image Combination Logic (for non-SAM-2 modes) ---
            if self.temporal_options == "median":
                images_np = np.stack([img.numpy() for img in images], axis=0).astype(np.float32)
                image = torch.from_numpy(np.median(images_np, axis=0)).float()
            else:
                image = torch.cat(images, dim=0).float()
                    
            # --- 4. Finalize Sample Dictionary ---
            sample["image"] = image
            
            if self.with_metadata:
                # Combine time vectors and add all static metadata
                sample["time"] = torch.cat(time_vectors, dim=0) if len(time_vectors) > 1 else time_vectors[0]
                sample["latlon"] = metadata_dict["latlon"]
                sample["gsd"] = metadata_dict["gsd"]
                sample["waves"] = metadata_dict["waves"]
                sample["platform"] = metadata_dict["platform"]

            # --- 5. Load Mask and Apply Transforms (for non-SAM-2 modes) ---
            sample["mask"] = torch.from_numpy(mask).long()
            
            if self.compute_boundary_distance:
                boundary = get_boundary(mask)
                distance = get_distance(mask)
                sample["boundary"] = torch.from_numpy(np.expand_dims(boundary, 0)).float()
                sample["distance"] = torch.from_numpy(np.expand_dims(distance, 0)).float()

        return sample