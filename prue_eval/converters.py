"""
Converter functions to transform model-specific outputs to intermediate formats.
The module contains all model-specific conversion logic, keeping the intermediate
formats and Detections class model-agnostic.

Usage:
    model_output = model.predict(image)
    intermediate = convert_baseline_output(model_output)
    detections = intermediate.to_detections()
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Union, Optional
from .intermediate_formats import SemanticOutput, InstanceOutput, PanopticOutput


# ============================================================================
# Semantic Segmentation Models
# ============================================================================


def convert_baseline_output(
    model_output: Union[np.ndarray, torch.Tensor], image_id: Optional[int] = None
) -> SemanticOutput:
    """
    Convert baseline model (UNet, DeepLabV3+, etc.) output to SemanticLogits.

    Args:
        model_output: Model logits of shape (num_classes, H, W) or (B, num_classes, H, W)
        image_id: Optional image identifier

    Returns:
        SemanticLogits object
    """
    # Convert torch to numpy if needed
    if isinstance(model_output, torch.Tensor):
        model_output = model_output.detach().cpu().numpy()

    # Handle batch dimension
    if model_output.ndim == 4:
        if model_output.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {model_output.shape[0]}")
        model_output = model_output[0]  # Remove batch dimension

    # Apply softmax if needed (assumes logits are raw)
    # Check if already probabilities (sum to ~1)
    if not np.allclose(model_output.sum(axis=0), 1.0, atol=0.1):
        # Apply softmax
        exp_logits = np.exp(model_output - np.max(model_output, axis=0, keepdims=True))
        model_output = exp_logits / exp_logits.sum(axis=0, keepdims=True)

    return SemanticOutput(logits=model_output, image_id=image_id, metadata={"model_type": "baseline"})


def convert_decode_output(model_output: Union[Tuple, List], image_id: Optional[int] = None) -> SemanticOutput:
    """
    Convert DECODE model output to SemanticLogits.

    DECODE produces a 3-tuple:
    - preds[0]: segmentation logits (B, num_classes, H, W)
    - preds[1]: boundary logits (B, 2, H, W)
    - preds[2]: distance to boundary (B, 1, H, W)

    Args:
        model_output: Tuple of (seg_logits, boundary_logits, distance)
        image_id: Optional image identifier

    Returns:
        SemanticLogits object with auxiliary outputs
    """
    seg_logits, boundary_logits, distance = model_output

    # Convert torch to numpy if needed
    if isinstance(seg_logits, torch.Tensor):
        seg_logits = seg_logits.detach().cpu().numpy()
        boundary_logits = boundary_logits.detach().cpu().numpy()
        distance = distance.detach().cpu().numpy()

    # Handle batch dimension
    if seg_logits.ndim == 4:
        if seg_logits.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {seg_logits.shape[0]}")
        seg_logits = seg_logits[0]
        boundary_logits = boundary_logits[0]
        distance = distance[0]

    # Apply softmax to segmentation logits
    if not np.allclose(seg_logits.sum(axis=0), 1.0, atol=0.1):
        exp_logits = np.exp(seg_logits - np.max(seg_logits, axis=0, keepdims=True))
        seg_logits = exp_logits / exp_logits.sum(axis=0, keepdims=True)

    # Apply softmax to boundary logits
    if not np.allclose(boundary_logits.sum(axis=0), 1.0, atol=0.1):
        exp_boundary = np.exp(boundary_logits - np.max(boundary_logits, axis=0, keepdims=True))
        boundary_logits = exp_boundary / exp_boundary.sum(axis=0, keepdims=True)

    return SemanticOutput(
        logits=seg_logits,
        auxiliary={"boundary_logits": boundary_logits, "distance": distance},
        image_id=image_id,
        metadata={"model_type": "decode"},
    )


# ============================================================================
# Instance Segmentation Models
# ============================================================================


def convert_sam_output(model_output: List[Dict[str, Any]], image_id: Optional[int] = None) -> InstanceOutput:
    """
    Convert SAM (Segment Anything) output to InstanceMasks.

    SAM produces a list of dicts, each with:
    - 'segmentation': binary mask (H, W) or RLE
    - 'predicted_iou': IoU score
    - 'stability_score': stability score

    Args:
        model_output: List of prediction dicts from SAM
        image_id: Optional image identifier

    Returns:
        InstanceMasks object
    """
    if len(model_output) == 0:
        # Return empty masks
        return InstanceOutput(
            masks=np.zeros((0, 256, 256), dtype=np.uint8),
            scores=np.array([]),
            image_id=image_id,
            metadata={"model_type": "sam"},
        )

    masks = []
    scores = []

    for pred in model_output:
        # Handle RLE format if needed
        segmentation = pred["segmentation"]
        if isinstance(segmentation, dict):  # RLE format
            import pycocotools.mask as mask_util

            segmentation = mask_util.decode(segmentation)

        masks.append(segmentation.astype(np.uint8))
        # Use predicted_iou as confidence score (clamp to [0, 1] for safety)
        pred_iou = pred.get("predicted_iou")
        if pred_iou is not None:
            # Clamp to [0, 1] in case of numerical precision issues
            score = float(np.clip(pred_iou, 0.0, 1.0))
        else:
            # Fallback to stability_score if predicted_iou not available
            stability = pred.get("stability_score", 1.0)
            score = float(np.clip(stability, 0.0, 1.0))
        scores.append(score)

    return InstanceOutput(
        masks=np.stack(masks, axis=0), scores=np.array(scores), image_id=image_id, metadata={"model_type": "sam"}
    )


def convert_delineate_anything_output(
    model_output,  # ultralytics.engine.results.Results
    image_id: Optional[int] = None,
) -> InstanceOutput:
    """
    Convert Delineate Anything (YOLO) output to InstanceMasks.

    Args:
        model_output: Results object from Ultralytics YOLO
        image_id: Optional image identifier

    Returns:
        InstanceMasks object
    """
    # Handle list of Results (batch)
    if isinstance(model_output, list):
        if len(model_output) != 1:
            raise ValueError(f"Expected single result, got {len(model_output)}")
        model_output = model_output[0]

    # Extract masks and scores
    if model_output.masks is None or len(model_output.masks) == 0:
        # No detections
        return InstanceOutput(
            masks=np.zeros((0, 256, 256), dtype=np.uint8),
            scores=np.array([]),
            image_id=image_id,
            metadata={"model_type": "delineate_anything"},
        )

    # Get masks as numpy arrays at the original (resized) resolution
    masks = model_output.masks.data.cpu().numpy()  # (N, H_resized, W_resized)

    # Get the target shape from orig_shape (this is the patch_size set in DelineateAnything)
    target_shape = model_output.masks.orig_shape  # (H_target, W_target)

    # Rescale masks to target shape if they don't match
    if masks.shape[1:] != target_shape:
        import cv2

        rescaled_masks = []
        for mask in masks:
            # Resize mask to target shape
            # Use INTER_NEAREST for segmentation masks to avoid subpixel shifts
            # INTER_LINEAR can introduce spatial offsets
            rescaled_mask = cv2.resize(
                mask,
                (target_shape[1], target_shape[0]),  # cv2 uses (width, height)
                interpolation=cv2.INTER_NEAREST,
            )
            rescaled_masks.append(rescaled_mask)
        masks = np.stack(rescaled_masks, axis=0)

    # Binarize masks (may already be binary with INTER_NEAREST)
    masks = (masks > 0.5).astype(np.uint8)

    # Get confidence scores
    scores = model_output.boxes.conf.cpu().numpy()  # (N,)

    return InstanceOutput(masks=masks, scores=scores, image_id=image_id, metadata={"model_type": "delineate_anything"})


# ============================================================================
# Panoptic Segmentation Models (Mask2Former, etc.)
# ============================================================================


def convert_d2_panoptic_output(model_output: Dict[str, Any], image_id: Optional[int] = None) -> PanopticOutput:
    """
    Convert Mask2Former/etc panoptic output to PanopticOutput.
    Hypothetically should support other panoptic-segmentation models like MaskDINO and OneFormer which are also based on detectron2.

    Model outputs a dict with:
    - 'panoptic_seg': Tuple of (seg_map, segments_info)
        - seg_map: Tensor of shape (H, W) with segment IDs
        - segments_info: List of dicts with 'id', 'category_id', 'isthing', 'score', etc.

    Args:
        model_output: Model output dict containing 'panoptic_seg'
        image_id: Optional image identifier

    Returns:
        PanopticOutput object
    """
    # Accept either a full output dict with key 'panoptic_seg' or a direct tuple (seg_map, segments_info)
    if isinstance(model_output, tuple) and len(model_output) == 2:
        seg_map, segments_info = model_output
    else:
        if "panoptic_seg" not in model_output:
            raise ValueError("Model output must contain 'panoptic_seg' key")
        seg_map, segments_info = model_output["panoptic_seg"]

    # Convert torch to numpy if needed
    if isinstance(seg_map, torch.Tensor):
        seg_map = seg_map.detach().cpu().numpy()

    # Ensure int type
    seg_map = seg_map.astype(np.int32)

    return PanopticOutput(
        seg_map=seg_map, segments_info=segments_info, image_id=image_id, metadata={"model_type": "d2_panoptic"}
    )
