"""
Raw mask evaluation to match ftw_tools baseline methodology exactly.

This module provides evaluation functions that work directly with raw model outputs
(binary masks) rather than reconstructed masks from detections, ensuring exact
alignment with the paper baseline methodology.
"""

import numpy as np
from typing import List, Dict, Any
from evaluator import get_object_level_metrics


def evaluate_raw_masks(
    gt_masks: List[np.ndarray], 
    pred_masks: List[np.ndarray], 
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate raw binary masks using the exact same methodology as ftw_tools baseline.
    
    This matches the evaluation in ftw_tools.models.baseline_eval.py exactly:
    - Works directly with raw model outputs (no polygonization/reconstruction)
    - Uses ftw_tools.postprocess.metrics.get_object_level_metrics
    - No area filtering or postprocessing
    
    Args:
        gt_masks: List of ground truth binary masks (0=background, 1=field)
        pred_masks: List of predicted binary masks (0=background, 1=field)
        iou_threshold: IoU threshold for object matching
        
    Returns:
        Dictionary with object-level metrics matching paper baseline
    """
    if len(gt_masks) != len(pred_masks):
        raise ValueError(f"GT masks ({len(gt_masks)}) and pred masks ({len(pred_masks)}) must have same length")
    
    all_tps = 0
    all_fps = 0
    all_fns = 0
    
    # Import ftw_tools metrics to ensure exact same methodology
    try:
        import sys
        from pathlib import Path
        ftw_baselines_path = Path(__file__).parent.parent.parent / "ftw-baselines"
        if str(ftw_baselines_path) not in sys.path:
            sys.path.insert(0, str(ftw_baselines_path))
        
        from ftw_tools.postprocess.metrics import get_object_level_metrics as ftw_get_object_level_metrics
        
        # Use ftw_tools methodology exactly
        for i, (gt_mask, pred_mask) in enumerate(zip(gt_masks, pred_masks)):
            tps, fps, fns = ftw_get_object_level_metrics(
                gt_mask, pred_mask, iou_threshold=iou_threshold
            )
            all_tps += tps
            all_fps += fps
            all_fns += fns
            
    except ImportError:
        print("Warning: Could not import ftw_tools.metrics. Using fallback evaluation.")
        # Fallback to your implementation (should be identical now)
        for i, (gt_mask, pred_mask) in enumerate(zip(gt_masks, pred_masks)):
            # Convert to Detections format for evaluation
            from detections import Detections
            
            # Create temporary detections from masks
            gt_dets = Detections.from_semantic_logits(
                type('SemanticOutput', (), {
                    'logits': np.zeros((2, *gt_mask.shape)),
                    'get_field_mask': lambda: gt_mask
                })(),
                min_area=0
            )
            
            pred_dets = Detections.from_semantic_logits(
                type('SemanticOutput', (), {
                    'logits': np.zeros((2, *pred_mask.shape)),
                    'get_field_mask': lambda: pred_mask
                })(),
                min_area=0
            )
            
            tps, fps, fns = get_object_level_metrics(gt_dets, pred_dets, iou_threshold)
            all_tps += tps
            all_fps += fps
            all_fns += fns
    
    # Calculate metrics exactly like ftw_tools baseline_eval.py
    if all_tps + all_fps > 0:
        object_precision = all_tps / (all_tps + all_fps)
    else:
        object_precision = float("nan")

    if all_tps + all_fns > 0:
        object_recall = all_tps / (all_tps + all_fns)
    else:
        object_recall = float("nan")
    
    object_f1 = 2 * object_precision * object_recall / (object_precision + object_recall) if (object_precision + object_recall) > 0 else 0
    
    return {
        "object_precision": object_precision * 100,
        "object_recall": object_recall * 100,
        "object_f1": object_f1 * 100,
        "object_tps": all_tps,
        "object_fps": all_fps,
        "object_fns": all_fns,
    }


def extract_raw_masks_from_detections(detections_list: List, target_shape: tuple) -> List[np.ndarray]:
    """
    Extract raw binary masks from detections for fair comparison.
    
    This tries to reconstruct the original model output masks before
    any polygonization or postprocessing was applied.
    
    Args:
        detections_list: List of Detections objects
        target_shape: Target mask shape (height, width)
        
    Returns:
        List of binary masks (0=background, 1=field)
    """
    masks = []
    
    for detections in detections_list:
        # Create binary mask from detections
        binary_mask = np.zeros(target_shape, dtype=np.uint8)
        
        if detections.mask is not None:
            for mask in detections.mask:
                # Ensure mask is same shape as target
                if mask.shape != target_shape:
                    from skimage.transform import resize
                    mask = resize(mask, target_shape, preserve_range=True, anti_aliasing=True)
                    mask = (mask > 0.5).astype(np.uint8)
                
                # Combine masks using logical OR (same as to_binary_mask)
                binary_mask = np.logical_or(binary_mask, mask > 0)
        
        masks.append(binary_mask.astype(np.uint8))
    
    return masks


def compare_evaluation_methods(
    gt_detections_list: List, 
    pred_detections_list: List,
    gt_masks: List[np.ndarray],
    target_shape: tuple
) -> Dict[str, Dict[str, float]]:
    """
    Compare different evaluation methodologies to identify differences.
    
    Args:
        gt_detections_list: Ground truth detections
        pred_detections_list: Predicted detections  
        gt_masks: Ground truth binary masks
        target_shape: Target mask shape
        
    Returns:
        Dictionary comparing different evaluation methods
    """
    results = {}
    
    # Method 1: Your current evaluation (detections-based)
    from evaluator import get_object_level_metrics
    
    all_tps_1, all_fps_1, all_fns_1 = 0, 0, 0
    for gt_dets, pred_dets in zip(gt_detections_list, pred_detections_list):
        tps, fps, fns = get_object_level_metrics(gt_dets, pred_dets)
        all_tps_1 += tps
        all_fps_1 += fps
        all_fns_1 += fns
    
    results["detections_based"] = {
        "precision": (all_tps_1 / (all_tps_1 + all_fps_1)) * 100 if (all_tps_1 + all_fps_1) > 0 else 0,
        "recall": (all_tps_1 / (all_tps_1 + all_fns_1)) * 100 if (all_tps_1 + all_fns_1) > 0 else 0,
        "tps": all_tps_1,
        "fps": all_fps_1,
        "fns": all_fns_1,
    }
    
    # Method 2: Raw mask evaluation (ftw_tools style)
    pred_masks = extract_raw_masks_from_detections(pred_detections_list, target_shape)
    results["raw_mask_based"] = evaluate_raw_masks(gt_masks, pred_masks)
    
    return results
