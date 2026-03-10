# Custom instance dataset mapper for multispectral satellite imagery.
# Extends the upstream COCOInstanceNewBaselineDatasetMapper to read GeoTIFFs
# with N channels (4, 6, 8) instead of 3-channel RGB images.
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances

from pycocotools import mask as coco_mask

from .coco_instance_new_baseline_dataset_mapper import build_transform_gen, convert_coco_poly_to_mask
from .coco_panoptic_new_baseline_dataset_mapper_custom import read_image_multichannel

__all__ = ["COCOInstanceCustomDatasetMapper"]

logger = logging.getLogger(__name__)


class COCOInstanceCustomDatasetMapper:
    """Dataset mapper for instance segmentation with multispectral satellite imagery.

    Reads GeoTIFF files with N channels and applies standard augmentations.
    """

    @configurable
    def __init__(self, is_train=True, *, tfm_gens, image_format):
        self.tfm_gens = tfm_gens
        self.img_format = image_format
        self.is_train = is_train
        logger.info(f"[COCOInstanceCustomDatasetMapper] TransformGens: {self.tfm_gens}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        tfm_gens = build_transform_gen(cfg, is_train)
        return {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = read_image_multichannel(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)
            h, w = instances.image_size
            if hasattr(instances, "gt_masks"):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            dataset_dict["instances"] = instances

        return dataset_dict
