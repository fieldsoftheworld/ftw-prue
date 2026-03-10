"""Convert model-specific outputs to intermediate formats (SemanticOutput/InstanceOutput/PanopticOutput)."""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Union, Optional
from .intermediate_formats import SemanticOutput, InstanceOutput, PanopticOutput


def convert_baseline_output(model_output: Union[np.ndarray, torch.Tensor], image_id: Optional[int] = None) -> SemanticOutput:
    """Baseline model (UNet, DeepLabV3+, etc.) logits → SemanticOutput."""
    if isinstance(model_output, torch.Tensor):
        model_output = model_output.detach().cpu().numpy()
    if model_output.ndim == 4:
        if model_output.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {model_output.shape[0]}")
        model_output = model_output[0]

    if not np.allclose(model_output.sum(axis=0), 1.0, atol=0.1):
        exp = np.exp(model_output - np.max(model_output, axis=0, keepdims=True))
        model_output = exp / exp.sum(axis=0, keepdims=True)

    return SemanticOutput(logits=model_output, image_id=image_id, metadata={"model_type": "baseline"})


def convert_decode_output(model_output: Union[Tuple, List], image_id: Optional[int] = None) -> SemanticOutput:
    """DECODE 3-tuple (seg_logits, boundary_logits, distance) → SemanticOutput with auxiliary."""
    seg_logits, boundary_logits, distance = model_output

    if isinstance(seg_logits, torch.Tensor):
        seg_logits = seg_logits.detach().cpu().numpy()
        boundary_logits = boundary_logits.detach().cpu().numpy()
        distance = distance.detach().cpu().numpy()

    if seg_logits.ndim == 4:
        if seg_logits.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {seg_logits.shape[0]}")
        seg_logits = seg_logits[0]
        boundary_logits = boundary_logits[0]
        distance = distance[0]

    if not np.allclose(seg_logits.sum(axis=0), 1.0, atol=0.1):
        exp = np.exp(seg_logits - np.max(seg_logits, axis=0, keepdims=True))
        seg_logits = exp / exp.sum(axis=0, keepdims=True)

    if not np.allclose(boundary_logits.sum(axis=0), 1.0, atol=0.1):
        exp = np.exp(boundary_logits - np.max(boundary_logits, axis=0, keepdims=True))
        boundary_logits = exp / exp.sum(axis=0, keepdims=True)

    return SemanticOutput(
        logits=seg_logits,
        auxiliary={"boundary_logits": boundary_logits, "distance": distance},
        image_id=image_id,
        metadata={"model_type": "decode"},
    )


def convert_sam_output(model_output: List[Dict[str, Any]], image_id: Optional[int] = None) -> InstanceOutput:
    """SAM output (list of dicts with 'segmentation', 'predicted_iou') → InstanceOutput."""
    if not model_output:
        return InstanceOutput(
            masks=np.zeros((0, 256, 256), dtype=np.uint8),
            scores=np.array([]),
            image_id=image_id,
            metadata={"model_type": "sam"},
        )

    masks, scores = [], []
    for pred in model_output:
        seg = pred["segmentation"]
        if isinstance(seg, dict):
            import pycocotools.mask as mask_util
            seg = mask_util.decode(seg)
        masks.append(seg.astype(np.uint8))

        iou = pred.get("predicted_iou")
        if iou is not None:
            scores.append(float(np.clip(iou, 0.0, 1.0)))
        else:
            scores.append(float(np.clip(pred.get("stability_score", 1.0), 0.0, 1.0)))

    return InstanceOutput(masks=np.stack(masks), scores=np.array(scores), image_id=image_id, metadata={"model_type": "sam"})


def convert_delineate_anything_output(model_output, image_id: Optional[int] = None) -> InstanceOutput:
    """Ultralytics YOLO Results → InstanceOutput."""
    if isinstance(model_output, list):
        if len(model_output) != 1:
            raise ValueError(f"Expected single result, got {len(model_output)}")
        model_output = model_output[0]

    if model_output.masks is None or len(model_output.masks) == 0:
        return InstanceOutput(
            masks=np.zeros((0, 256, 256), dtype=np.uint8),
            scores=np.array([]),
            image_id=image_id,
            metadata={"model_type": "delineate_anything"},
        )

    masks = model_output.masks.data.cpu().numpy()
    target_shape = model_output.masks.orig_shape

    if masks.shape[1:] != target_shape:
        import cv2
        masks = np.stack([
            cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            for m in masks
        ])

    masks = (masks > 0.5).astype(np.uint8)
    scores = model_output.boxes.conf.cpu().numpy()

    return InstanceOutput(masks=masks, scores=scores, image_id=image_id, metadata={"model_type": "delineate_anything"})


def convert_d2_panoptic_output(model_output: Dict[str, Any], image_id: Optional[int] = None) -> PanopticOutput:
    """Detectron2 panoptic output → PanopticOutput. Works with Mask2Former, MaskDINO, OneFormer."""
    if isinstance(model_output, tuple) and len(model_output) == 2:
        seg_map, segments_info = model_output
    else:
        if "panoptic_seg" not in model_output:
            raise ValueError("Model output must contain 'panoptic_seg' key")
        seg_map, segments_info = model_output["panoptic_seg"]

    if isinstance(seg_map, torch.Tensor):
        seg_map = seg_map.detach().cpu().numpy()

    return PanopticOutput(seg_map=seg_map.astype(np.int32), segments_info=segments_info, image_id=image_id, metadata={"model_type": "d2_panoptic"})
