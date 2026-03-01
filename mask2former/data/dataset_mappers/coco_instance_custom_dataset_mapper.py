# Modified by Zhanpei Fang from COCOPanopticNewBaselineDatasetMapperCustom.
# This dataset mapper is specifically designed for handling multi-spectral satellite imagery
# with instance segmentation annotations in COCO format.
import copy
import os
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances

from mask2former.data.transforms.augmentations import (
    SafeTransformMixin, 
    build_transform_gen_separated
)


class COCOInstanceCustomDatasetMapper(SafeTransformMixin):
    """
    Custom dataset mapper for instance segmentation with multi-spectral satellite imagery.
    
    Handles:
    - 8-channel RGBNRGBN input format
    - NPZ file loading (FTW stacked format)
    - GeoTIFF reading
    - Custom augmentations (FTW, Prue, SatTrivial, etc.)
    - Instance segmentation format (bboxes, masks, classes)
    """
    
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
                "[COCOInstanceCustomDatasetMapper] Geometric transforms: %s",
                str(self.geometric_tfm_gens)
            )
            self.logger.info(
                "[COCOInstanceCustomDatasetMapper] Radiometric transforms: %s",
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

    def __call__(self, dataset_dict):
        """
        Map a dataset dict to the format expected by Mask-RCNN.
        
        Args:
            dataset_dict: Metadata of one image, in Detectron2 Dataset format.
                Should contain:
                - "file_name": path to image
                - "height", "width": image dimensions
                - "annotations": list of annotation dicts with:
                  - "bbox": [x, y, w, h] in XYWH_ABS format
                  - "segmentation": polygon or RLE mask
                  - "category_id": class ID
                  - "iscrowd": 0 or 1
        
        Returns:
            dict: Format expected by Mask-RCNN:
                - "image": tensor of shape (C, H, W)
                - "instances": Instances object with:
                  - gt_boxes: Boxes tensor
                  - gt_classes: tensor of class IDs
                  - gt_masks: BitMasks tensor
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        # Read image
        image = self.read_image(dataset_dict)
        original_image_shape = image.shape[:2]  # h, w
        
        # STEP 1: Apply geometric transforms to image (affect spatial coordinates)
        # These transforms will be applied to both image and instance annotations
        image, geometric_transforms = T.apply_transform_gens(self.geometric_tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # STEP 2: Process instance annotations if available
        if "annotations" in dataset_dict:
            # Transform instance annotations (bboxes, masks, etc.)
            annos = [
                utils.transform_instance_annotations(obj, geometric_transforms, image_shape)
                for obj in dataset_dict["annotations"]
                if obj.get("iscrowd", 0) == 0
            ]
            
            # Convert annotations to Instances object
            instances = utils.annotations_to_instances(annos, image_shape)
            
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. Recompute bounding boxes from masks.
            if hasattr(instances, 'gt_masks') and len(instances.gt_masks) > 0:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            
            # Filter empty instances (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            
            dataset_dict["instances"] = instances

        # STEP 3: Apply radiometric transforms ONLY to the image (NOT to labels)
        # These transforms only affect pixel values and should not change instance masks
        if self.radiometric_tfm_gens:
            image, _ = T.apply_transform_gens(self.radiometric_tfm_gens, image)

        # Convert to tensor format (C, H, W)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32)

        # Remove annotations key (already converted to instances)
        dataset_dict.pop("annotations", None)
        
        return dataset_dict

