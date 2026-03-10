# Custom panoptic dataset mapper for multispectral satellite imagery.
# Extends the upstream COCOPanopticNewBaselineDatasetMapper to read GeoTIFFs
# with N channels (4, 6, 8) instead of 3-channel RGB images.
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances

from .coco_panoptic_new_baseline_dataset_mapper import build_transform_gen

__all__ = ["COCOPanopticNewBaselineDatasetMapperCustom"]

logger = logging.getLogger(__name__)


def read_image_multichannel(file_name):
    """Read a GeoTIFF or regular image, returning HWC numpy array."""
    if file_name.endswith(".tif") or file_name.endswith(".tiff"):
        import rasterio

        with rasterio.open(file_name) as src:
            image = src.read()  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        return image
    else:
        return utils.read_image(file_name, format="RGB")


class COCOPanopticNewBaselineDatasetMapperCustom:
    """Dataset mapper for panoptic segmentation with multispectral satellite imagery.

    Reads GeoTIFF files with N channels and applies standard augmentations.
    """

    @configurable
    def __init__(self, is_train=True, *, tfm_gens, image_format):
        self.tfm_gens = tfm_gens
        self.img_format = image_format
        self.is_train = is_train
        logger.info(f"[COCOPanopticNewBaselineDatasetMapperCustom] TransformGens: {self.tfm_gens}")

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

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])

            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]))
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            dataset_dict["instances"] = instances

        return dataset_dict
