"""FTW dataset."""

import os
import random
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset, RasterDataset

from ftw_tools.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS
from ftw_tools.utils import validate_checksums
from pretrained.models.model_utils import get_preprocessor, prepare_clay_sample, prepare_general_sample

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
        metadata_path: str = None
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
        self.preprocessor, self.gsd, self.waves = get_preprocessor(preprocessing, metadata_path=metadata_path)
        self.checksum = checksum
        self.load_boundaries = load_boundaries
        self.temporal_options = temporal_options
        self.num_samples = num_samples
        self.swap_order = swap_order
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

            # --- 2. Image Combination Logic ---
            if self.temporal_options == "median":
                images_np = np.stack([img.numpy() for img in images], axis=0).astype(np.float32)
                image = torch.from_numpy(np.median(images_np, axis=0)).float()
            else:
                image = torch.cat(images, dim=0).float()
                    
            # --- 3. Finalize Sample Dictionary ---
            sample["image"] = image
            
            if self.with_metadata:
                # Combine time vectors and add all static metadata
                sample["time"] = torch.cat(time_vectors, dim=0) if len(time_vectors) > 1 else time_vectors[0]
                sample["latlon"] = metadata_dict["latlon"]
                sample["gsd"] = metadata_dict["gsd"]
                sample["waves"] = metadata_dict["waves"]
                sample["platform"] = metadata_dict["platform"]

            # --- 4. Load Mask and Apply Transforms ---
            with rasterio.open(img_filenames["mask"]) as f:
                mask = f.read(1)
            sample["mask"] = torch.from_numpy(mask).long()

        return sample