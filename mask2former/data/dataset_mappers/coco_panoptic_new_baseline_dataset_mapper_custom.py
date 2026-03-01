# Modified from Bowen Cheng's Mask2Former COCO dataset mappers.
# This dataset mapper is specifically designed for handling multi-spectral satellite imagery
# with panoptic segmentation annotations in COCO format.
# See also for inspiration: https://github.com/PatBall1/detectree2/blob/master/detectree2/models/train.py#L48
import os
import logging

import numpy as np
import torch
# from cv2 import imread, cvtColor, COLOR_BGR2RGB

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances, BoxMode

from mask2former.data.transforms.augmentations import (
    SafeTransformMixin, 
    build_transform_gen,
    build_transform_gen_separated
)

class COCOPanopticNewBaselineDatasetMapperCustom(SafeTransformMixin):
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        geometric_tfm_gens=None,
        radiometric_tfm_gens=None,
        tfm_gens=None,
        image_format,
        debug_aug=False
    ):
        # Use separated transforms if provided, otherwise fall back to combined
        if geometric_tfm_gens is not None and radiometric_tfm_gens is not None:
            self.geometric_tfm_gens = geometric_tfm_gens
            self.radiometric_tfm_gens = radiometric_tfm_gens
            self.tfm_gens = geometric_tfm_gens + radiometric_tfm_gens
        elif tfm_gens is not None:
            # Backward compatibility: separate automatically if not pre-separated
            from mask2former.data.transforms.augmentations import is_geometric_transform
            self.geometric_tfm_gens = [t for t in tfm_gens if is_geometric_transform(t)]
            self.radiometric_tfm_gens = [t for t in tfm_gens if not is_geometric_transform(t)]
            self.tfm_gens = tfm_gens
        else:
            raise ValueError("Must provide either (geometric_tfm_gens, radiometric_tfm_gens) or tfm_gens")
        
        self.logger = logging.getLogger(__name__)

        self.img_format = image_format
        self.is_train = is_train

        self.debug_aug = debug_aug
        if self.debug_aug:
            self.logger.info(
                "[COCOPanopticNewBaselineDatasetMapperCustom] Geometric transforms: %s",
                str(self.geometric_tfm_gens)
            )
            self.logger.info(
                "[COCOPanopticNewBaselineDatasetMapperCustom] Radiometric transforms: %s",
                str(self.radiometric_tfm_gens)
            )
        

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build separated transforms for proper geometric/radiometric handling
        geometric_tfm_gens, radiometric_tfm_gens = build_transform_gen_separated(cfg, is_train)
        
        return {
            "is_train": is_train,
            "geometric_tfm_gens": geometric_tfm_gens,
            "radiometric_tfm_gens": radiometric_tfm_gens,
            # Keep backward compatibility: combine for legacy code
            "tfm_gens": geometric_tfm_gens + radiometric_tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "debug_aug": (cfg.DEBUG.AUGMENTATION if hasattr(cfg, 'DEBUG') and hasattr(cfg.DEBUG, 'AUGMENTATION') else False)
        }

    def read_image(self, dataset_dict):
        """Read image, handling both regular images and GeoTIFFs/NPZs."""
        file_name = dataset_dict.get("file_name", "")
        
        if not file_name:
            raise ValueError("No file_name found in dataset_dict")
        
        try:
            # Use custom GeoTIFF/NPZ reader for satellite imagery
            if file_name.endswith((".tif", ".tiff", ".npz")):
                image = utils.read_geotiff(file_name, format=self.img_format)
            else:
                # Use standard image reader for regular images
                image = utils.read_image(file_name, format=self.img_format)

            # Ensure the image array is writable to avoid PyTorch warnings
            # Only copy if necessary (most arrays from image readers are already writable)
            if not image.flags.writeable:
                image = np.array(image, copy=True)
            
            utils.check_image_size(dataset_dict, image)
            
            return image
            
        except (LookupError, RuntimeError) as e:
            # Handle Python 3.11 encoding issues with NPZ files (cp437 encoding error)
            if ("encoding" in str(e).lower() or "cp437" in str(e).lower()) and file_name.endswith(".npz"):
                # Log the error clearly - this file should be skipped
                self.logger.error(
                    f"Skipping NPZ file {file_name} due to encoding error (Python 3.11 compatibility issue): {e}. "
                    f"This file should be filtered out at dataset registration time or recreated."
                )
                # Re-raise to let the error propagate - the file will be skipped by the dataloader error handling
                # or can be filtered at dataset registration time
                raise
        except Exception as e:
            self.logger.error(f"Failed to read image {file_name}: {e}")
            raise

    def _detect_dataset_type(self, file_name, dataset_dict):
        """
        Detect dataset type and extract metadata from file path and name.
        Returns a dictionary with dataset_type, country, and other metadata.
        
        Performance: Fast string operations, negligible overhead (<0.1ms per image).
        """
        base_name = os.path.basename(file_name)
        
        # Initialize result
        result = {
            "dataset_type": "unknown",
            "country": None,
            "file_format": "unknown"
        }
        
        # Detect file format (fast string checks)
        if file_name.endswith(".npz"):
            result["file_format"] = "npz"
        elif file_name.endswith((".tif", ".tiff")):
            result["file_format"] = "geotiff"
        elif file_name.endswith((".jpg", ".jpeg", ".png")):
            result["file_format"] = "image"
        
        # Detect dataset type and country from path and filename
        path_parts = file_name.split(os.sep)
        
        # Check for FTW dataset indicators
        if "npz" in path_parts or file_name.endswith(".npz"):
            result["dataset_type"] = "ftw"
            # Extract country from filename (e.g., "austria_12345.npz")
            if "_" in base_name:
                result["country"] = base_name.split("_")[0].lower()
        
        # Fallback: try to extract country from filename if it follows the pattern
        elif "_" in base_name:
            potential_country = base_name.split("_")[0].lower()
            # Common FTW countries (using set for O(1) lookup)
            ftw_countries = {"austria", "belgium", "brazil", "cambodia", "corsica", "croatia", 
                           "denmark", "estonia", "finland", "france", "germany", "india", 
                           "kenya", "latvia", "lithuania", "luxembourg", "netherlands", 
                           "portugal", "rwanda", "slovakia", "slovenia", "south_africa", 
                           "spain", "sweden", "vietnam"}
            if potential_country in ftw_countries:
                result["dataset_type"] = "ftw"
                result["country"] = potential_country
        
        return result

    def __call__(self, dataset_dict):
        """
        Apply transformations to the image and annotations.
        
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
    
        # Extract dataset information from filename and path
        file_name = dataset_dict["file_name"]
        base_name = os.path.basename(file_name)
        
        # Detect dataset type and country (fast operation, <0.1ms overhead)
        dataset_info = self._detect_dataset_type(file_name, dataset_dict)
        country = dataset_info.get("country")
        dataset_type = dataset_info.get("dataset_type")
        
        if self.debug_aug:
            self.logger.info(f"Processing {dataset_type} dataset, country: {country}, file: {base_name}")

        # Read image
        image = self.read_image(dataset_dict)
        original_image_shape = image.shape[:2]  # h, w
        
        # STEP 1: Apply geometric transforms to image (affect spatial coordinates)
        # These transforms will be applied to both image and segmentation labels
        image, geometric_transforms = T.apply_transform_gens(self.geometric_tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # STEP 2: Process panoptic segmentation if available
        if "pan_seg_file_name" in dataset_dict:
            # Read and process panoptic segmentation
            pan_seg_gt = utils.read_image(dataset_dict["pan_seg_file_name"], "RGB")
            # Ensure the array is writable to avoid PyTorch warnings (only copy if necessary)
            if not pan_seg_gt.flags.writeable:
                pan_seg_gt = np.array(pan_seg_gt, copy=True)
            segments_info = dataset_dict["segments_info"]

            # Apply the SAME geometric transforms to segmentation
            # (radiometric transforms will NOT be applied to segmentation)
            pan_seg_gt = geometric_transforms.apply_segmentation(pan_seg_gt)
            
            # Convert panoptic segmentation
            from panopticapi.utils import rgb2id
            pan_seg_gt = rgb2id(pan_seg_gt)
            
            # Create instances
            instances = Instances(image_shape)
            classes = []
            masks = []
            
            # Process each segment (segment_info is a dict, each dict is a segment).
            # segments_info currently not getting transformed, just used to access the segment id and category_id)
            for segment_info in segments_info:
                if not segment_info["iscrowd"]:
                    class_id = segment_info["category_id"]
                    mask = pan_seg_gt == segment_info["id"]
                    
                    if mask.sum() > 0:  # Only add if mask is not empty
                        classes.append(class_id)
                        masks.append(mask)
            
            # Create instance annotations
            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) > 0:
                # Ensure mask arrays are writable before converting to tensors (only copy if necessary)
                writable_masks = []
                for mask in masks:
                    if mask.flags.writeable and mask.flags.c_contiguous:
                        writable_masks.append(mask)
                    else:
                        writable_masks.append(np.ascontiguousarray(mask))
                masks = BitMasks(torch.stack([torch.from_numpy(mask) for mask in writable_masks]))
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()
                
                if self.debug_aug:
                    print(f"  Created instances with {len(classes)} masks")
                    print(f"  Classes: {instances.gt_classes}")
                    print(f"  Mask shapes: {instances.gt_masks.shape}")
            else: # Some image does not have annotation (all ignored)
                if self.debug_aug:
                    print("  No valid masks found, creating empty instances")
                instances.gt_classes = torch.zeros(0, dtype=torch.int64)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            
            dataset_dict["instances"] = instances

        # STEP 3: Apply radiometric transforms ONLY to the image (NOT to labels)
        # These transforms only affect pixel values and should not change segmentation masks
        if self.radiometric_tfm_gens:
            image, _ = T.apply_transform_gens(self.radiometric_tfm_gens, image)

        # Convert to tensor format (C, H, W)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32)

        return dataset_dict

class COCOPanopticCachingDatasetMapper:
    """
    Dataset mapper that caches processed samples to disk for faster loading.
    
    This mapper processes samples once and caches them to disk for repeated use.
    This significantly reduces data loading time for repeated training runs.

    NB: I'm not convinced this actually speeds up data reading and am not currently using it
    """
    def __init__(self, cfg, is_train=True, cache_dir=None):
        """
        Args:
            cfg: Detectron2 config
            is_train: Whether this is for training
            cache_dir: Directory to store cached samples
        """
        self.cfg = cfg
        self.is_train = is_train
        self.logger = logging.getLogger(__name__)
        
        # Create the original mapper
        from mask2former.data.dataset_mappers import COCOPanopticNewBaselineDatasetMapperCustom
        self.original_mapper = COCOPanopticNewBaselineDatasetMapperCustom(cfg, is_train)
        
        # Setup cache directory
        self.cache_dir = cache_dir or os.path.join(cfg.OUTPUT_DIR, "dataset_cache", "train" if is_train else "val")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Initialized caching dataset mapper with cache at {self.cache_dir}")
        
        # Keep a memory cache for quick access
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _get_cache_path(self, dataset_dict):
        """Get the cache file path for a dataset dict"""
        # Use image_id as unique identifier if available
        if 'image_id' in dataset_dict and dataset_dict['image_id'] is not None:
            identifier = str(dataset_dict['image_id'])
        elif 'file_name' in dataset_dict:
            # Otherwise use filename hash
            identifier = os.path.basename(dataset_dict['file_name'])
        else:
            # Last resort: hash the dataset dict itself
            import hashlib
            identifier = hashlib.md5(str(dataset_dict).encode()).hexdigest()
            
        return os.path.join(self.cache_dir, f"{identifier}.pt")
        
    def __call__(self, dataset_dict):
        # More efficient memory cache check
        dataset_id = dataset_dict.get('image_id', '') or dataset_dict.get('file_name', '')
        
        # Use a lightweight in-memory reference cache
        if dataset_id in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[dataset_id]
        
        # Check disk cache more efficiently
        cache_path = self._get_cache_path(dataset_dict)
        if os.path.exists(cache_path):
            try:
                # Use memory mapping for faster loading from disk
                result = torch.load(cache_path, map_location='cpu')
                # Don't store the full result in memory, just a reference
                self.memory_cache[dataset_id] = result
                self.cache_hits += 1
                return result
            except Exception as e:
                # Log less frequently to reduce overhead
                if self.cache_misses % 100 == 0:
                    self.logger.warning(f"Failed to load cache: {e}")
        
        # Process and cache with reduced logging
        self.cache_misses += 1
        result = self.original_mapper(dataset_dict)
        
        try:
            torch.save(result, cache_path)
            # Store reference not copy
            self.memory_cache[dataset_id] = result
        except Exception as e:
            if self.cache_misses % 100 == 0:
                self.logger.warning(f"Failed to save to cache: {e}")
                
        return result
