"""
Pixel-space postprocessing utilities for agricultural field boundary delineation.

This module provides functions to filter and process masks/segments before vectorization.
Pixel-space operations are more efficient than geometric operations on polygons.

These functions operate on numpy arrays and segments_info lists, before conversion
to geographic coordinates. See trainer/postprocessing.py for conversion functions.

All operations work in pixel space only - no geographic transformations or CRS handling.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from skimage import morphology
except ImportError:
    morphology = None
    logger.warning("skimage not available, small object removal will be disabled")


def filter_segments_by_confidence(
    segments_info: List[Dict],
    confidence_threshold: float = 0.8,  # Default matches MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
) -> List[Dict]:
    """
    Filter segments_info by confidence threshold.

    Args:
        segments_info: List of segment information dictionaries
        confidence_threshold: Minimum confidence score

    Returns:
        Filtered list of segments_info
    """
    filtered = []
    for segment in segments_info:
        # Try to get confidence from multiple possible keys
        confidence = segment.get("confidence", None)
        if confidence is None:
            # Try "score" key (common in Mask2Former output)
            confidence = segment.get("score", 0.0)
        if confidence is None or confidence == 0.0:
            # Try computing from score and mask_score
            score_val = segment.get("score", 0.0)
            mask_score_val = segment.get("mask_score", 1.0)
            if score_val > 0:
                confidence = score_val * mask_score_val
            else:
                confidence = 0.0

        if confidence >= confidence_threshold:
            filtered.append(segment)
    return filtered


def filter_segments_by_category(
    segments_info: List[Dict], category_id: int = 0, exclude_background: bool = True
) -> List[Dict]:
    """
    Filter segments_info by category ID.

    Args:
        segments_info: List of segment information dictionaries
        category_id: Contiguous category ID to keep (0 = ag_field, 1 = background)
        exclude_background: Whether to exclude background category (category_id=1)

    Returns:
        Filtered list of segments_info
    """
    filtered = []
    for segment in segments_info:
        seg_category = segment.get("category_id", 0)

        # Skip background if requested
        if exclude_background and seg_category == 1:
            continue

        # Keep only specified category
        if seg_category == category_id:
            filtered.append(segment)

    return filtered


def filter_segments_by_isthing(segments_info: List[Dict], keep_things: bool = True) -> List[Dict]:
    """
    Filter segments_info by isthing flag.

    Args:
        segments_info: List of segment information dictionaries
        keep_things: If True, keep only thing instances; if False, keep only stuff

    Returns:
        Filtered list of segments_info
    """
    filtered = []
    for segment in segments_info:
        isthing = segment.get("isthing", True)
        if (keep_things and isthing) or (not keep_things and not isthing):
            filtered.append(segment)
    return filtered


def filter_mask_by_area_pixels(
    mask: np.ndarray, min_area_pixels: int = 100, max_area_pixels: Optional[int] = None
) -> Tuple[np.ndarray, bool]:
    """
    Filter mask by area threshold (in pixels).

    Args:
        mask: Binary mask array (H, W)
        min_area_pixels: Minimum area in pixels
        max_area_pixels: Maximum area in pixels (None = no max)

    Returns:
        Tuple of (filtered_mask, is_valid)
        - filtered_mask: Original mask if valid, zeros if invalid
        - is_valid: Boolean indicating if mask passes area filter
    """
    area_pixels = np.sum(mask > 0)

    if area_pixels < min_area_pixels:
        return np.zeros_like(mask), False

    if max_area_pixels is not None and area_pixels > max_area_pixels:
        return np.zeros_like(mask), False

    return mask, True


def detect_edge_polygons(mask: np.ndarray, chip_size: Tuple[int, int], edge_threshold: int = 5) -> bool:
    """
    Detect if mask touches chip edges.

    Args:
        mask: Binary mask array (H, W)
        chip_size: Tuple of (height, width) of the chip
        edge_threshold: Number of pixels from edge to consider as "edge"

    Returns:
        True if mask touches edges, False otherwise
    """
    h, w = mask.shape
    chip_h, chip_w = chip_size

    # Check if mask extends to edges
    # Top edge
    if np.any(mask[:edge_threshold, :] > 0):
        return True

    # Bottom edge
    if np.any(mask[h - edge_threshold :, :] > 0):
        return True

    # Left edge
    if np.any(mask[:, :edge_threshold] > 0):
        return True

    # Right edge
    if np.any(mask[:, w - edge_threshold :] > 0):
        return True

    return False


def remove_small_objects(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Remove small objects from mask using morphological operations.

    Args:
        mask: Binary mask array (H, W)
        min_size: Minimum size in pixels for objects to keep

    Returns:
        Filtered mask with small objects removed
    """
    if morphology is None:
        logger.warning("skimage not available, returning original mask")
        return mask

    # Use skimage morphology to remove small objects
    cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)

    return cleaned.astype(mask.dtype)


def filter_panoptic_segments_pixel_space(
    panoptic_seg: np.ndarray,
    segments_info: List[Dict],
    confidence_threshold: float = 0.8,  # Default matches MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
    min_area_pixels: int = 10,  # Small default - just removes tiny fragments; bulk filtering at geographic level
    max_area_pixels: Optional[int] = None,
    chip_size: Optional[Tuple[int, int]] = None,
    edge_threshold: int = 5,
    remove_small_objects_size: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Apply all pixel-space filtering to panoptic segmentation.

    This is a convenience function that applies multiple filters in sequence.
    All operations work in pixel space only - no geographic transformations.

    Args:
        panoptic_seg: Panoptic segmentation array (H, W)
        segments_info: List of segment information dictionaries
        confidence_threshold: Minimum confidence score
        min_area_pixels: Minimum area in pixels
        max_area_pixels: Maximum area in pixels (None = no max)
        chip_size: Tuple of (height, width) for edge detection (None = skip)
        edge_threshold: Number of pixels from edge for edge detection
        remove_small_objects_size: Minimum size in pixels for small object removal (None = skip)

    Returns:
        Tuple of (filtered_panoptic_seg, filtered_segments_info)
    """
    # Start with copy of panoptic segmentation
    filtered_panoptic_seg = panoptic_seg.copy()

    # Step 1: Filter by confidence
    filtered_segments_info = filter_segments_by_confidence(segments_info, confidence_threshold)

    # Step 2: Filter by category (keep only ag fields, exclude background)
    # category_id=0 is ag_field (contiguous ID), category_id=1 is background
    filtered_segments_info = filter_segments_by_category(filtered_segments_info, category_id=0, exclude_background=True)

    # Step 3: Filter by isthing (keep only instances)
    filtered_segments_info = filter_segments_by_isthing(filtered_segments_info, keep_things=True)

    # Step 4: Apply pixel-space filters to each segment
    valid_segments = []
    valid_segment_ids = set()

    for segment in filtered_segments_info:
        segment_id = segment["id"]
        mask = (panoptic_seg == segment_id).astype(np.uint8)

        # Skip if mask is empty
        if np.sum(mask) == 0:
            continue

        # Apply area filter (pixel-based)
        filtered_mask, is_valid = filter_mask_by_area_pixels(mask, min_area_pixels, max_area_pixels)
        if not is_valid:
            continue

        # Check edge detection if chip_size provided
        if chip_size is not None:
            touches_edge = detect_edge_polygons(mask, chip_size, edge_threshold)
            if touches_edge:
                # Mark as edge polygon (we'll filter later in geographic space)
                segment["touches_edge"] = True
            else:
                segment["touches_edge"] = False

        # Apply small object removal if requested
        if remove_small_objects_size is not None:
            filtered_mask = remove_small_objects(filtered_mask, remove_small_objects_size)
            # Re-check area after small object removal
            filtered_mask, is_valid = filter_mask_by_area_pixels(filtered_mask, min_area_pixels, max_area_pixels)
            if not is_valid:
                continue

        # Update panoptic segmentation with filtered mask
        # Remove old segment ID
        filtered_panoptic_seg[filtered_panoptic_seg == segment_id] = 0
        # Add back filtered mask
        filtered_panoptic_seg[filtered_mask > 0] = segment_id

        valid_segments.append(segment)
        valid_segment_ids.add(segment_id)

    # Remove segment IDs that are no longer valid
    filtered_panoptic_seg[~np.isin(filtered_panoptic_seg, list(valid_segment_ids))] = 0

    return filtered_panoptic_seg, valid_segments
