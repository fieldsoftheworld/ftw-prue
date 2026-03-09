#!/usr/bin/env python3
"""
FTW to COCO Dataset Converter

Converts Fields of the World dataset into COCO detection, panoptic, and semantic segmentation formats.
This script combines multiple country folders into a single COCO dataset, maintaining the original splits.

Example usage:
python ftw_to_coco.py \
    --data_root /path/to/ftw/data \
    --output_dir /path/to/output/coco \
    --countries austria,france,germany \
    --splits train,val,test \
    --window a \
    --num_workers 8
"""

import os
import json
import time
import math
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import multiprocessing as mp

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import box, shape
import PIL.Image as Image
from tqdm import tqdm
import fsspec
import shutil
import pycocotools.mask as mask_util

class InstanceInfo(NamedTuple):
    """Store information about a single instance (field)"""
    id: int
    mask: np.ndarray
    bbox: List[float]
    area: float
    color: Optional[np.ndarray] = None

@dataclass
class CategoryInfo:
    """Store category information for COCO format"""
    id: int
    name: str
    supercategory: str
    isthing: bool = True
    color: Optional[List[int]] = None

def rgb2id(color: np.ndarray) -> int:
    """Convert RGB color to unique ID using the COCO panopticapi formula"""
    if len(color) == 4:  # RGBA
        return int(color[0]) + 256 * int(color[1]) + 256 * 256 * int(color[2])
    return int(color[0]) + 256 * int(color[1]) + 256 * 256 * int(color[2])

class FTWToCOCOConverter:
    """Convert FTW dataset to COCO format (instance and panoptic segmentation)"""
    
    def __init__(
        self,
        data_root: str,
        output_dir: str,
        countries: List[str],
        splits: List[str] = None,
        window: str = "a",
        min_area: int = 0,
        num_workers: int = None,
        background_id: int = 4988569,
        field_category_id: int = 1,
        background_category_id: int = 2,
        field_color: List[int] = [100, 204, 25], # Green
        background_color: List[int] = [153, 30, 76], # Dark red
        sample_fraction: float = 1.0,
        random_seed: int = 42,
        verbose: bool = True,
        parallel_processing: bool = True
    ):
        """Initialize the converter.
        
        Args:
            data_root: Root directory of FTW dataset
            output_dir: Output directory for COCO dataset
            countries: List of countries to include
            splits: List of splits to include (train, val, test)
            window: Which temporal window to use (a or b)
            min_area: Minimum area of instances to include (in pixels)
            num_workers: Number of worker processes
            background_id: ID for background in panoptic segmentation
            field_category_id: Category ID for agricultural fields
            background_category_id: Category ID for background
            sample_fraction: Fraction of dataset to sample (for debugging)
            random_seed: Random seed for reproducibility
            verbose: Whether to print verbose output
            parallel_processing: Whether to use parallel processing
        """

        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.countries = countries
        self.splits = splits or ["train", "val", "test"]
        self.window = window.lower()
        self.min_area = min_area
        self.num_workers = num_workers or mp.cpu_count()
        self.background_id = background_id
        self.field_category_id = field_category_id
        self.background_category_id = background_category_id
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed
        self.verbose = verbose
        self.parallel_processing = parallel_processing

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Define field base color and background color
        self.field_color = field_color 
        self.background_color = background_color  

        # Define categories
        self.categories = [
            CategoryInfo(
                id=field_category_id,
                name="ag_field",
                supercategory="landcover",
                isthing=True,
                color=self.field_color
            ),
            CategoryInfo(
                id=background_category_id,
                name="background",
                supercategory="background",
                isthing=False,
                color=self.background_color
            )
        ]

        # Setup directories
        self._setup_directories()

        # Initialize counters for unique IDs
        self.image_id_counter = 0
        self.annotation_id_counter = 0
        
        if self.verbose:
            print(f"Initialized FTW to COCO converter with:")
            print(f"  Data root: {data_root}")
            print(f"  Output directory: {output_dir}")
            print(f"  Countries: {', '.join(countries)}")
            print(f"  Splits: {', '.join(splits)}")
            print(f"  Window: {window}")
            print(f"  Workers: {self.num_workers}")

    def _setup_directories(self):
        """Create necessary output directories"""
        # Main directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        
        # Create directories for each split
        for split in self.splits:
            # Images directory
            (self.output_dir / split).mkdir(exist_ok=True)
            
            # Panoptic directories
            (self.output_dir / f"panoptic_{split}").mkdir(exist_ok=True)
            (self.output_dir / f"panoptic_semseg_{split}").mkdir(exist_ok=True)
            
    def _get_next_image_id(self) -> int:
        """Get next unique image ID and increment counter"""
        image_id = self.image_id_counter
        self.image_id_counter += 1
        return image_id
        
    def _get_next_annotation_id(self) -> int:
        """Get next unique annotation ID and increment counter"""
        annotation_id = self.annotation_id_counter
        self.annotation_id_counter += 1
        return annotation_id

    def _load_chips_for_country(self, country: str) -> gpd.GeoDataFrame:
        """Load chips file for a country
        
        Args:
            country: Country name
            
        Returns:
            GeoDataFrame with chips information
        """
        chips_file = self.data_root / country / f"chips_{country}.parquet"
        if not chips_file.exists():
            warnings.warn(f"Chips file not found for {country}")
            return gpd.GeoDataFrame()
            
        try:
            # Read chips file
            chips_gdf = gpd.read_parquet(chips_file)
            
            # Add country column if not present
            if "country" not in chips_gdf.columns:
                chips_gdf["country"] = country
                
            return chips_gdf
            
        except Exception as e:
            warnings.warn(f"Error loading chips for {country}: {str(e)}")
            return gpd.GeoDataFrame()

    def _load_split_aois(self) -> Dict[str, Dict[str, List[str]]]:
        """Load AOI IDs for each country and split
        
        Returns:
            Dictionary mapping split -> country -> list of AOI IDs
        """
        result = {split: {} for split in self.splits}
        
        for country in self.countries:
            # Load chips
            chips_gdf = self._load_chips_for_country(country)
            
            if chips_gdf.empty:
                continue
                
            # Group by split
            for split in self.splits:
                # Filter by split
                split_chips = chips_gdf[chips_gdf["split"] == split]
                
                # Sample if needed
                if self.sample_fraction < 1.0:
                    split_chips = split_chips.sample(
                        frac=self.sample_fraction, 
                        random_state=self.random_seed
                    )
                
                # Get AOI IDs
                aoi_ids = split_chips["aoi_id"].tolist()
                
                # Store in result
                if aoi_ids:
                    result[split][country] = aoi_ids
                    
        return result

    def _get_field_geometries(self, country: str, aoi_id: str) -> gpd.GeoDataFrame:
        """Get field geometries for a specific AOI"""
        try:
            # Check both window_a and window_b for the image file
            window_path_a = self.data_root / country / "s2_images" / "window_a" / f"{aoi_id}.tif"
            window_path_b = self.data_root / country / "s2_images" / "window_b" / f"{aoi_id}.tif"
            
            if window_path_a.exists():
                window_path = window_path_a
            elif window_path_b.exists():
                window_path = window_path_b
            else:
                raise FileNotFoundError(f"Image file not found in either window for {country}/{aoi_id}")
                
            try:
                with rio.open(window_path) as src:
                    bounds = src.bounds
                    transform = src.transform
                    crs = src.crs
            except Exception as e:
                raise RuntimeError(f"Failed to open image file: {str(e)}")
                    
            # Create bbox geometry
            bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Determine which local geoparquet file to use based on country
            if country in ['austria','belgium','cambodia','corsica','denmark','estonia','finland','latvia','lithuania','portugal','rwanda','slovakia','slovenia','sweden','vietnam']:
                geoparquet_filename = f"boundaries_{country}_2021.parquet"
            elif country in ['brazil','france','spain']:
                geoparquet_filename = f"boundaries_{country}_2020.parquet"
            elif country in ['croatia']:
                geoparquet_filename = f"boundaries_{country}_2023.parquet"
            elif country in ['germany']:
                geoparquet_filename = f"boundaries_{country}_2018_2019.parquet"
            elif country in ['india']:
                geoparquet_filename = f"boundaries_{country}_2016.parquet"
            elif country in ['kenya', 'luxembourg','netherlands']:
                geoparquet_filename = f"boundaries_{country}_2022.parquet"
            elif country in ['south_africa']:
                geoparquet_filename = f"boundaries_south_africa_2018.parquet"
            else:
                return gpd.GeoDataFrame()
            
            # Construct path to local file (allow overriding via environment variable)
            geoparquet_root = Path(os.environ.get("FTW_GEOPARQUET_ROOT", "/path/to/ftw/geoparquets"))
            geoparquet_path = geoparquet_root / geoparquet_filename
            
            if not geoparquet_path.exists():
                raise FileNotFoundError(f"Local GeoParquet file not found: {geoparquet_path}")
            
            # Read the local geoparquet file with better error handling
            try:
                # First try to filter by aoi_id if supported
                try:
                    gdf = gpd.read_parquet(geoparquet_path, filters=[('aoi_id', '=', aoi_id)])
                except Exception as e1:
                    # If filtering by aoi_id fails, try spatial filter
                    try:
                        gdf = gpd.read_parquet(geoparquet_path, bbox=bbox_geom)
                    except Exception as e2:
                        # If spatial filter fails, read everything and filter manually
                        try:
                            gdf = gpd.read_parquet(geoparquet_path)
                        except Exception as e3:
                            raise RuntimeError(f"All attempts to read GeoParquet failed: {str(e3)}")
            except Exception as e:
                raise RuntimeError(f"Failed to read GeoParquet file: {str(e)}")
                
            # If no CRS in GeoParquet, use the image CRS
            if gdf.crs is None:
                gdf.set_crs(crs, inplace=True)
                
            # If CRS doesn't match, reproject
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)
                
            # If no aoi_id column in GeoParquet, use spatial filtering
            if 'aoi_id' not in gdf.columns:
                # Filter by intersection with bbox
                gdf = gdf[gdf.intersects(bbox_geom)]
            else:
                # Filter by aoi_id
                gdf = gdf[gdf['aoi_id'] == aoi_id]
                
            return gdf
                
        except Exception as e:
            # Add more context to the error message
            error_msg = f"Error getting field geometries for {country}/{aoi_id}: {str(e)}"
            # Re-raise with more context
            raise type(e)(error_msg) from e

    def _generate_instance_color(self) -> np.ndarray:
        """Generate a unique color for instance segmentation
        
        Returns:
            RGB color as numpy array
        """
        while True:
            # Start with field base color
            base_color = np.array(self.field_color)
            
            # Add random jitter while keeping in green hue range
            jitter = np.random.randint(-30, 31, size=3)
            jitter[0] -= 20  # Reduce red
            jitter[2] -= 20  # Reduce blue
            
            # Apply jitter and clip to valid range
            color = np.clip(base_color + jitter, 0, 255)
            
            # Ensure ID doesn't match background ID
            if rgb2id(color) != self.background_id:
                return color

    def _create_rle_from_binary_mask(self, binary_mask: np.ndarray) -> Dict:
        """Create RLE encoded mask from binary mask
        
        Args:
            binary_mask: Binary mask as numpy array
            
        Returns:
            RLE encoded mask in COCO format
        """
        # Ensure mask is binary and correct data type
        binary_mask = binary_mask.astype(np.uint8)
        
        # Encode mask
        rle = mask_util.encode(np.asfortranarray(binary_mask))
        
        # Convert to COCO format (bytes to string)
        rle['counts'] = rle['counts'].decode('utf-8')
        
        return rle

    def _create_instance(
        self,
        geom,
        transform: rio.transform.Affine,
        height: int,
        width: int,
        instance_id: int,
        generate_color: bool = True
    ) -> Optional[InstanceInfo]:
        """Create instance from geometry
        
        Args:
            geom: Shapely geometry
            transform: Affine transform
            height: Image height
            width: Image width
            instance_id: Instance ID
            generate_color: Whether to generate color for panoptic segmentation
            
        Returns:
            InstanceInfo object or None if invalid
        """
        try:
            # Rasterize geometry
            mask = rasterize(
                [(geom, 1)],
                out_shape=(height, width),
                transform=transform,
                dtype=np.uint8,
                all_touched=True
            )
            
            # Validate mask
            if mask is None or not mask.any():
                return None
                
            # Check minimum area
            area = float(mask.sum())
            if area < self.min_area:
                return None
                
            # Compute bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return None
                
            # Get bbox coordinates (x, y, width, height)
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                return None
                
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            bbox = [
                float(x_min),
                float(y_min),
                float(x_max - x_min + 1),
                float(y_max - y_min + 1)
            ]
            
            # Generate color for panoptic segmentation if needed
            color = self._generate_instance_color() if generate_color else None
            
            return InstanceInfo(
                id=instance_id,
                mask=mask,
                bbox=bbox,
                area=area,
                color=color
            )
            
        except Exception as e:
            warnings.warn(f"Error creating instance: {str(e)}")
            return None

    def _process_single_image(
        self,
        country: str,
        aoi_id: str,
        split: str,
        image_id: int,
        generate_panoptic: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Process a single image with improved handling of instance masks"""
        try:
            # Construct paths for both window_a and window_b
            image_path_a = self.data_root / country / "s2_images" / "window_a" / f"{aoi_id}.tif"
            image_path_b = self.data_root / country / "s2_images" / "window_b" / f"{aoi_id}.tif"
            
            # Check for semantic and instance masks
            mask_path_2class = self.data_root / country / "label_masks" / "semantic_2class" / f"{aoi_id}.tif"
            mask_path_3class = self.data_root / country / "label_masks" / "semantic_3class" / f"{aoi_id}.tif"
            mask_path_instance = self.data_root / country / "label_masks" / "instance" / f"{aoi_id}.tif"
            
            
            if self.window == "ab":
                try:
                    with rio.open(image_path_a) as src_a:
                        img_a = src_a.read()
                        profile = src_a.profile.copy()
                        height, width = src_a.height, src_a.width
                        transform = src_a.transform
                        crs = src_a.crs
                
                    with rio.open(image_path_b) as src_b:
                        img_b = src_b.read()
                        
                    # Verify both have the same shape
                    if img_a.shape != img_b.shape:
                        if self.verbose:
                            print(f"Skipping {country}/{aoi_id} - window_a and window_b have different shapes")
                        return None

                    # Create the stacked image (RGBNRGBN)
                    stacked_img = np.vstack([img_b, img_a]) # switched to ba
                        
                    # Create NPZ output directory
                    npz_output_dir = self.output_dir / split / "npz"
                    npz_output_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Save stacked image as NPZ
                    npz_file_path = npz_output_dir / f"{country}_{aoi_id}.npz"
                    
                    # Save stacked image with appropriate metadata
                    np.savez_compressed(
                        npz_file_path, 
                        image=stacked_img,
                        window_a=img_a,
                        window_b=img_b,
                        transform=transform,
                        crs=str(crs),
                        height=height,
                        width=width
                    )

                    # Use the npz file path for further processing
                    image_path = npz_file_path
                    window_used = "ab"
                except Exception as e:
                    if self.verbose:
                        print(f"Error stacking images for {country}/{aoi_id}: {str(e)}")
                    return None

            # Otherwise, check if image file exists in either window
            elif self.window == "a":
                if image_path_a.exists():
                    image_path = image_path_a
                    window_used = "a"
                elif image_path_b.exists():
                    image_path = image_path_b
                    window_used = "b"
                with rio.open(image_path) as src:
                    height, width = src.height, src.width
                    transform = src.transform
                    profile = src.profile
                    crs = src.crs
            elif self.window == "b": 
                if image_path_b.exists():
                    image_path = image_path_b
                    window_used = "b"
                elif image_path_a.exists():
                    image_path = image_path_a
                    window_used = "a"
                with rio.open(image_path) as src:
                    height, width = src.height, src.width
                    transform = src.transform
                    profile = src.profile
                    crs = src.crs
            else:
                raise FileNotFoundError(f"Image file not found in either window for {country}/{aoi_id}")
            
            # Flags to track data sources
            using_mask = False
            using_instance_mask = False
            mask_path_used = None
            is_negative_example = False
            
            # Try to get field geometries from GeoParquet
            try:
                fields_gdf = self._get_field_geometries(country, aoi_id)
            except Exception as e:
                # If GeoParquet loading fails, create empty GeoDataFrame
                fields_gdf = gpd.GeoDataFrame()
                    
            # If no geometries were found in GeoParquet, try to use instance mask as primary fallback
            if fields_gdf.empty:
                if mask_path_instance.exists():
                    try:
                        # Read the instance mask
                        with rio.open(mask_path_instance) as src:
                            instance_mask = src.read()[0]
                            
                        # Check if there are any non-zero values (instance IDs)
                        if np.any(instance_mask > 0):
                            using_instance_mask = True
                            mask_path_used = "instance"
                        else:
                            # No instances found, mark as negative example
                            is_negative_example = True
                    except Exception as e:
                        # If instance mask reading fails, log and continue to next fallback
                        warnings.warn(f"Failed to read instance mask file {mask_path_instance}: {str(e)}")
                
                # If instance mask isn't available or readable, try semantic masks
                if not using_instance_mask:
                    # Check if either semantic mask file exists
                    if mask_path_2class.exists():
                        mask_path = mask_path_3class
                        mask_path_used = "3class"
                    elif mask_path_3class.exists():
                        mask_path = mask_path_2class
                        mask_path_used = "2class"
                    else:
                        # Neither mask file exists, so we'll treat this as a negative example
                        is_negative_example = True
                        
                    # If a semantic mask file was found, use it to generate field geometries
                    if mask_path_used in ["2class", "3class"]:
                        try:
                            # Read the mask
                            with rio.open(mask_path) as src:
                                mask_array = src.read(1)
                                
                            # Check mask values to determine class interpretation
                            unique_values = np.unique(mask_array)
                            
                            # Determine which values represent fields based on mask type
                            if mask_path_used == "2class":
                                # In 2-class mask, field class is usually 1
                                # but check for presence of 1 in the data
                                # in FTW: 2=boundary, 0=background, 1=crop
                                if 1 in unique_values:
                                    field_mask = mask_array == 1
                                else:
                                    field_mask = mask_array == 0
                            else:  # 3-class
                                # In 3-class mask, field class is usually 1
                                field_mask = mask_array == 1
                                
                            # If there are no field pixels, mark as negative example
                            if not np.any(field_mask):
                                is_negative_example = True
                            else:
                                # We're using the mask
                                using_mask = True
                        except Exception as e:
                            # If mask reading fails, treat as negative example
                            is_negative_example = True
                            warnings.warn(f"Failed to read mask file {mask_path}: {str(e)}")
            
            # Process each field geometry if any exist
            instances = []
            instance_anns = []
            panoptic_segments = []
            
            # Skip instance generation for negative examples
            if not is_negative_example:
                # Process vector geometries from GeoParquet
                if not fields_gdf.empty:
                    for idx, row in fields_gdf.iterrows():
                        # Get annotation ID for this instance
                        annotation_id = self._get_next_annotation_id()
                        
                        # Create instance
                        try:
                            instance = self._create_instance(
                                row.geometry, 
                                transform, 
                                height, 
                                width, 
                                annotation_id,
                                generate_color=generate_panoptic
                            )
                        except Exception as e:
                            # Skip this instance but continue with others
                            continue
                            
                        if instance is None:
                            continue
                            
                        # Add to instances list
                        instances.append(instance)
                        
                        rle_mask = self._create_rle_from_binary_mask(instance.mask)

                        # Create instance annotation
                        instance_ann = {
                            # COCO requirement: annotation ids must be globally unique within the file.
                            # Use the monotonically increasing annotation_id instead of color-based IDs.
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": self.field_category_id,
                            "area": float(instance.area),
                            "bbox": [float(x) for x in instance.bbox],
                            "segmentation": rle_mask,
                            "iscrowd": 0
                        }
                        
                        # Add to instance annotations list
                        instance_anns.append(instance_ann)
                        
                        # Add to panoptic segments if needed
                        if generate_panoptic and not is_negative_example:
                            panoptic_segments.append({
                                # For panoptic annotations, the 'id' field is the segment id within this image.
                                # It can be derived from the color and does not need to be globally unique.
                                "id": rgb2id(instance.color),
                                "category_id": self.field_category_id,
                                "area": float(instance.area),
                                "bbox": [float(x) for x in instance.bbox],
                                "iscrowd": 0
                            })
                
                # Process instance mask if using it
                elif using_instance_mask:
                    # Get unique instance IDs (excluding 0 which is background)
                    instance_ids = np.unique(instance_mask)
                    instance_ids = instance_ids[instance_ids > 0]
                    
                    # Process each instance ID as a separate field
                    for instance_id in instance_ids:
                        # Get annotation ID for this instance
                        annotation_id = self._get_next_annotation_id()
                        
                        # Create binary mask for this instance
                        instance_binary = (instance_mask == instance_id).astype(np.uint8)
                        
                        # Check minimum area
                        area = float(instance_binary.sum())
                        if area < self.min_area:
                            continue
                        
                        # Compute bounding box
                        rows = np.any(instance_binary, axis=1)
                        cols = np.any(instance_binary, axis=0)
                        if not rows.any() or not cols.any():
                            continue
                        
                        # Get bbox coordinates (x, y, width, height)
                        y_indices = np.where(rows)[0]
                        x_indices = np.where(cols)[0]
                        
                        if len(y_indices) == 0 or len(x_indices) == 0:
                            continue
                        
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        
                        bbox = [
                            float(x_min),
                            float(y_min),
                            float(x_max - x_min + 1),
                            float(y_max - y_min + 1)
                        ]
                        
                        # Generate color for panoptic segmentation
                        color = self._generate_instance_color() if generate_panoptic else None
                        
                        # Create instance
                        instance = InstanceInfo(
                            id=annotation_id,
                            mask=instance_binary,
                            bbox=bbox,
                            area=area,
                            color=color
                        )
                        
                        # Add to instances list
                        instances.append(instance)
                        
                        rle_mask = self._create_rle_from_binary_mask(instance.mask)
                        
                        # Create instance annotation
                        instance_ann = {
                            # Use globally unique annotation_id for instance annotations
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": self.field_category_id,
                            "area": float(instance.area),
                            "bbox": [float(x) for x in instance.bbox],
                            "segmentation": rle_mask,
                            "iscrowd": 0
                        }
                        
                        # Add to instance annotations list
                        instance_anns.append(instance_ann)
                        
                        # Add to panoptic segments if needed
                        if generate_panoptic and not is_negative_example:
                            panoptic_segments.append({
                                "id": rgb2id(instance.color),
                                "category_id": self.field_category_id,
                                "area": float(instance.area),
                                "bbox": [float(x) for x in instance.bbox],
                                "iscrowd": 0
                            })
                            
                # Process semantic mask (labeled connected components) if using it
                elif using_mask:
                    # Process field mask to create instances
                    from skimage import measure
                    
                    # Find connected components in the field mask
                    labeled_mask, num_labels = measure.label(field_mask, return_num=True, connectivity=2)
                    
                    # Process each connected component as an instance
                    for label in range(1, num_labels + 1):
                        # Get annotation ID for this instance
                        annotation_id = self._get_next_annotation_id()
                        
                        # Create binary mask for this component
                        component_mask = (labeled_mask == label).astype(np.uint8)
                        
                        # Check minimum area
                        area = float(component_mask.sum())
                        if area < self.min_area:
                            continue
                        
                        # Compute bounding box
                        rows = np.any(component_mask, axis=1)
                        cols = np.any(component_mask, axis=0)
                        if not rows.any() or not cols.any():
                            continue
                        
                        # Get bbox coordinates (x, y, width, height)
                        y_indices = np.where(rows)[0]
                        x_indices = np.where(cols)[0]
                        
                        if len(y_indices) == 0 or len(x_indices) == 0:
                            continue
                        
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        
                        bbox = [
                            float(x_min),
                            float(y_min),
                            float(x_max - x_min + 1),
                            float(y_max - y_min + 1)
                        ]
                        
                        # Generate color for panoptic segmentation
                        color = self._generate_instance_color() if generate_panoptic else None
                        
                        # Create instance
                        instance = InstanceInfo(
                            id=annotation_id,
                            mask=component_mask,
                            bbox=bbox,
                            area=area,
                            color=color
                        )
                        
                        # Add to instances list
                        instances.append(instance)
                        
                        rle_mask = self._create_rle_from_binary_mask(instance.mask)
                        
                        # Create instance annotation
                        instance_ann = {
                            # Use globally unique annotation_id for instance annotations
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": self.field_category_id,
                            "area": float(instance.area),
                            "bbox": [float(x) for x in instance.bbox],
                            "segmentation": rle_mask,
                            "iscrowd": 0
                        }
                        
                        # Add to instance annotations list
                        instance_anns.append(instance_ann)
                        
                        # Add to panoptic segments if needed
                        if generate_panoptic and not is_negative_example:
                            panoptic_segments.append({
                                "id": rgb2id(instance.color),
                                "category_id": self.field_category_id,
                                "area": float(instance.area),
                                "bbox": [float(x) for x in instance.bbox],
                                "iscrowd": 0
                            })
            
            # Create image info (regardless of whether instances were found)
            if self.window == "ab":
                image_filename = f"{country}_{aoi_id}.npz"
            else:
                image_filename = f"{country}_{aoi_id}.tif"
            sem_seg_filename = f"{country}_{aoi_id}.png"  # Semantic segmentation filename
            panoptic_filename = f"{country}_{aoi_id}.png"  # Panoptic segmentation filename

            image_info = {
                "id": image_id,
                "file_name": image_filename,
                "height": height,
                "width": width,
                "country": country,
                "aoi_id": aoi_id,
                "window": window_used,
                "data_source": ("geoparquet" if not fields_gdf.empty else 
                            "instance_mask" if using_instance_mask else 
                            f"mask_{mask_path_used}" if using_mask else 
                            "negative_example"),
                "is_negative_example": is_negative_example or not instances,
                "sem_seg_file_name": f"panoptic_semseg_{split}/{sem_seg_filename}",
                "pan_seg_file_name": f"panoptic_{split}/{panoptic_filename}"
            }
            
            # Create result dictionary
            result = {
                "image_info": image_info,
                "image_path": str(image_path),
                "instances": instances,
                "instance_anns": instance_anns,
                "has_instances": len(instances) > 0,
                "is_negative_example": is_negative_example or not instances
            }
            
            # Add panoptic annotation if needed
            if generate_panoptic:
                # Create background mask 
                # For negative examples, this is the entire image
                # For positive examples, this is all area not covered by instances
                background_mask = np.ones((height, width), dtype=np.uint8)
                
                # Update background mask for positive examples
                if instances:
                    for instance in instances:
                        # background_mask &= (instance.mask == 0)
                        background_mask[instance.mask == 1] = 0
                        
                # Add background segment
                panoptic_segments.append({
                    "id": self.background_id,
                    "category_id": self.background_category_id,
                    "area": float(background_mask.sum()),
                    "bbox": [0, 0, width, height],
                    "iscrowd": 0
                })
                
                # Create panoptic annotation
                panoptic_ann = {
                    "image_id": image_id,
                    "file_name": image_filename.replace(".tif", ".png"),
                    "segments_info": panoptic_segments
                }
                
                result["panoptic_ann"] = panoptic_ann
                result["background_mask"] = background_mask
                
            return result
            
        except Exception as e:
            # Add more context to the error message
            error_msg = f"Error processing {country}/{aoi_id}: {str(e)}"
            # Re-raise with more context
            raise type(e)(error_msg) from e

    def _generate_and_save_images(
        self,
        result: Dict[str, Any],
        split: str
    ):
        """Generate and save images for visualization and panoptic segmentation"""
        try:
            # Extract data
            image_info = result["image_info"]
            image_path = result["image_path"]
            image_filename = image_info["file_name"]
            base_filename = os.path.splitext(image_filename)[0]
            
            # Paths for output images
            img_output_path = self.output_dir / split / image_filename
            panoptic_output_path = self.output_dir / f"panoptic_{split}" / f"{base_filename}.png"
            semseg_output_path = self.output_dir / f"panoptic_semseg_{split}" / f"{base_filename}.png"
            
            if image_info.get('window', 'ab'):
                pass
            else:
                # Copy the original image file. Use shutil.copy2 to preserve metadata
                shutil.copy2(image_path, img_output_path)

            # Generate and save panoptic segmentation
            if "panoptic_ann" in result:
                instances = result["instances"]
                background_mask = result["background_mask"]
                height, width = image_info["height"], image_info["width"]
                
                # Initialize arrays
                panoptic_seg = np.zeros((height, width, 3), dtype=np.uint8)
                # semantic_seg = np.full((height, width), 255, dtype=np.uint8)
                semantic_seg = np.zeros((height,width), dtype=np.uint8)

                # Fill with background first (contiguous ID 1)
                semantic_seg.fill(1) # background class has contiguous ID 1 # CHECK
                
                # Set background color for the entire image
                for i in range(3):
                    panoptic_seg[:, :, i] = self.background_color[i]
                
                # Add instances if they exist
                if instances:
                    for instance in instances:
                        # Update semantic segmentation with field class (contiguous ID 0)
                        semantic_seg[instance.mask == 1] = 0 # Field class has contiguous ID 0
                        
                        # Update panoptic segmentation
                        for i in range(3):
                            panoptic_seg[:, :, i][instance.mask == 1] = instance.color[i]
                
                # Save images
                Image.fromarray(panoptic_seg).save(panoptic_output_path)
                Image.fromarray(semantic_seg).save(semseg_output_path)
                
        except Exception as e:
            warnings.warn(f"Error generating images for {image_filename}: {str(e)}")
            
    def _process_country_split(
        self,
        country: str,
        split: str,
        aoi_ids: List[str],
        generate_panoptic: bool = True
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Process all images for a country and split
        
        Args:
            country: Country name
            split: Dataset split
            aoi_ids: List of AOI IDs
            generate_panoptic: Whether to generate panoptic segmentation
            
        Returns:
            Tuple of (image_infos, instance_anns, panoptic_anns)
        """
        if self.verbose:
            print(f"Processing {country} {split} split with {len(aoi_ids)} images")
            
        image_infos = []
        instance_anns = []
        panoptic_anns = []
        
        # Define chunk size
        chunk_size = max(1, min(100, len(aoi_ids) // (self.num_workers * 2)))
        
        # Split into chunks
        chunks = [aoi_ids[i:i + chunk_size] for i in range(0, len(aoi_ids), chunk_size)]
        
        if self.verbose:
            print(f"Split into {len(chunks)} chunks of ~{chunk_size} images each")
            
        # Track progress
        processed_count = 0
        failed_count = 0
        negative_example_count = 0
        positive_example_count = 0
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(tqdm(chunks, desc=f"{country} {split}", disable=not self.verbose)):
            # We'll use a list to collect results from parallel workers
            chunk_results = []
            chunk_image_ids = {}  # Map from aoi_id to image_id
            error_reasons = {}  # Track specific error reasons by aoi_id
            
            # Assign image IDs before parallelization
            for aoi_id in chunk:
                chunk_image_ids[aoi_id] = self._get_next_image_id()
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit tasks
                futures = []
                for aoi_id in chunk:
                    image_id = chunk_image_ids[aoi_id]
                    future = executor.submit(
                        self._process_single_image,
                        country,
                        aoi_id,
                        split,
                        image_id,
                        generate_panoptic
                    )
                    futures.append((aoi_id, future))
                
                # Collect results
                for aoi_id, future in futures:
                    try:
                        result = future.result()
                        if result is not None:
                            chunk_results.append(result)
                        else:
                            failed_count += 1
                            error_reasons[aoi_id] = "Result was None; missing window_a and/or window_b"
                    except Exception as e:
                        failed_count += 1
                        error_reasons[aoi_id] = str(e)
                        warnings.warn(f"Error processing {country}/{aoi_id}: {str(e)}")
            
            # Process results for this chunk
            for result in chunk_results:
                # Add the image info to our collection
                image_infos.append(result["image_info"])
                
                # Add instance annotations (could be empty for negative examples)
                if "instance_anns" in result and result["instance_anns"]:
                    instance_anns.extend(result["instance_anns"])
                
                # Add panoptic annotation (should always exist, with at least background)
                if generate_panoptic and "panoptic_ann" in result:
                    panoptic_anns.append(result["panoptic_ann"])
                    
                # Generate and save images
                self._generate_and_save_images(result, split)
                
                processed_count += 1

                if result.get("is_negative_example", False): # CHECK: shouldn't this be flipped?
                    negative_example_count += 1
                else:
                    positive_example_count += 1
            
            # Print error statistics after each chunk
            if error_reasons and self.verbose:
                print(f"\nChunk {chunk_idx+1}/{len(chunks)} error summary:")
                error_counts = {}
                for reason in error_reasons.values():
                    error_counts[reason] = error_counts.get(reason, 0) + 1
                
                # Print the top 5 most common error types
                for reason, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {count} failures due to: {reason}")
                
                # Log detailed errors to a file for later analysis
                error_log_path = self.output_dir / f"error_log_{country}_{split}.txt"
                with open(error_log_path, "a") as f:
                    f.write(f"\n--- Chunk {chunk_idx+1}/{len(chunks)} errors ---\n")
                    for aoi_id, reason in error_reasons.items():
                        f.write(f"{aoi_id}: {reason}\n")
        
        if self.verbose:
            print(f"Processed {processed_count} images, failed {failed_count} images")
            print(f"Positive examples: {positive_example_count}, Negative examples: {negative_example_count}")
            
        return image_infos, instance_anns, panoptic_anns

    def convert_dataset(self, generate_panoptic: bool = True):
        """Convert the entire dataset
        
        Args:
            generate_panoptic: Whether to generate panoptic segmentation
        """
        # Record start time
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting conversion at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        # Load split AOIs
        split_aois = self._load_split_aois()
        
        # Process each split
        for split in self.splits:
            if self.verbose:
                print(f"\nProcessing {split} split")
                
            # Skip if no images in split
            if not split_aois[split]:
                if self.verbose:
                    print(f"No images found for {split} split")
                continue
                
            # Initialize result containers
            all_image_infos = []
            all_instance_anns = []
            all_panoptic_anns = []
            
            # Process each country
            for country, aoi_ids in split_aois[split].items():
                # Process country
                image_infos, instance_anns, panoptic_anns = self._process_country_split(
                    country, split, aoi_ids, generate_panoptic
                )
                
                # Add to results
                all_image_infos.extend(image_infos)
                all_instance_anns.extend(instance_anns)
                
                if generate_panoptic:
                    all_panoptic_anns.extend(panoptic_anns)
                    
            # Skip if no images processed
            if not all_image_infos:
                if self.verbose:
                    print(f"No images processed for {split} split")
                continue
                
            # Save instance annotations
            instances_dict = {
                "images": all_image_infos,
                "annotations": all_instance_anns,
                "categories": [
                    {
                        "id": cat.id,
                        "name": cat.name,
                        "supercategory": cat.supercategory,
                        "isthing": 1 if cat.isthing else 0,
                        "color": cat.color
                    }
                    for cat in self.categories # if cat.isthing # CHECK
                ]
            }
            
            instance_path = self.output_dir / "annotations" / f"instances_{split}.json"
            
            with open(instance_path, "w") as f:
                json.dump(instances_dict, f)
                
            if self.verbose:
                print(f"Saved {len(all_image_infos)} images and {len(all_instance_anns)} instances to {instance_path}")
                
            # Save panoptic annotations
            if generate_panoptic and all_panoptic_anns:
                panoptic_dict = {
                    "images": all_image_infos,
                    "annotations": all_panoptic_anns,
                    "categories": [
                        {
                            "id": cat.id,
                            "name": cat.name,
                            "supercategory": cat.supercategory,
                            "isthing": 1 if cat.isthing else 0, # CHECK
                            "color": cat.color
                        }
                        for cat in self.categories
                    ]
                }
                
                panoptic_path = self.output_dir / "annotations" / f"panoptic_{split}.json"
                
                with open(panoptic_path, "w") as f:
                    json.dump(panoptic_dict, f)
                    
                if self.verbose:
                    print(f"Saved {len(all_panoptic_anns)} panoptic annotations to {panoptic_path}")
                    
        # Report elapsed time
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"\nConversion completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Fields of the World dataset to COCO format"
    )
    parser.add_argument("--data_root", required=True, help="Root directory of FTW dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory for COCO dataset")
    parser.add_argument("--countries", required=True, help="Comma-separated list of countries to include")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated list of splits to include")
    parser.add_argument("--window", default="a", choices=["a", "b", "ab"], help="Which temporal window to use (a or b or ab)")
    parser.add_argument("--min_area", type=int, default=0, help="Minimum area of instances to include (in pixels)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--no_panoptic", action="store_true", help="Skip panoptic segmentation generation")
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of dataset to sample (for debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--sequential", action="store_true", help="Use sequential processing instead of parallel")
    
    args = parser.parse_args()
    
    # Parse comma-separated lists
    countries = args.countries.split(",")
    splits = args.splits.split(",")
    
    # Create converter
    converter = FTWToCOCOConverter(
        data_root=args.data_root,
        output_dir=args.output_dir,
        countries=countries,
        splits=splits,
        window=args.window,
        min_area=args.min_area,
        num_workers=args.num_workers,
        sample_fraction=args.sample,
        random_seed=args.seed,
        verbose=not args.quiet,
        parallel_processing=not args.sequential
    )
    
    # Convert dataset
    converter.convert_dataset(generate_panoptic=not args.no_panoptic)


