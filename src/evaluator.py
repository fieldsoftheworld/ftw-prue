import numpy as np
import rasterio.features
import shapely.geometry
import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detections import Detections
import rasterio.features
import shapely.geometry


def get_object_level_metrics_from_semantic_masks(
    gt_mask: np.ndarray, 
    pred_mask: np.ndarray, 
    iou_threshold=0.5
):
    """
    Get object level metrics from semantic masks (matches ftw_tools.training.metrics.get_object_level_metrics).
    
    This function extracts connected components from semantic masks where val == 1,
    which is the correct approach for evaluating semantic segmentation models.
    
    Args:
        gt_mask: Ground truth semantic mask (binary: 0=background, 1=field)
        pred_mask: Predicted semantic mask (binary: 0=background, 1=field)
        iou_threshold: IoU threshold for matching predictions to ground truths
        
    Returns:
        tuple (int, int, int): Number of true positives, false positives, and false negatives
    """
    if iou_threshold < 0.5:
        raise ValueError("iou_threshold must be greater than 0.5")
    
    # Extract shapes from semantic masks (connected components where val == 1)
    # This matches the old evaluation behavior
    gt_shapes = []
    for geom, val in rasterio.features.shapes(gt_mask.astype(np.uint8)):
        if val == 1:
            gt_shapes.append(shapely.geometry.shape(geom))
    
    pred_shapes = []
    for geom, val in rasterio.features.shapes(pred_mask.astype(np.uint8)):
        if val == 1:
            pred_shapes.append(shapely.geometry.shape(geom))
    
    # Compute matching using IoU (greedy algorithm, matches ftw_tools)
    tps = 0
    fns = 0
    matched_pred_indices = set()
    
    for i, gt_shape in enumerate(gt_shapes):
        matching_j = None
        for j, pred_shape in enumerate(pred_shapes):
            if gt_shape.intersects(pred_shape):
                intersection = gt_shape.intersection(pred_shape)
                union = gt_shape.union(pred_shape)
                iou = intersection.area / union.area if union.area > 0 else 0
                if iou > iou_threshold:
                    matching_j = j
                    matched_pred_indices.add(j)
                    break  # Stop at first match above threshold (greedy matching)
        
        if matching_j is not None:
            tps += 1
        else:
            fns += 1
    
    fps = len(pred_shapes) - len(matched_pred_indices)
    
    return (tps, fps, fns)


def get_object_level_metrics(gt_detections: Detections, pred_detections: Detections, iou_threshold=0.5):
    """
    Unified object level metrics function that works with Detections objects.
    
    This function extracts polygons from both ground truth and predicted detections
    and computes object-level metrics using IoU-based matching.
    
    Use this for instance segmentation models where both GT and predictions are instances.
    For semantic segmentation models, use get_object_level_metrics_from_semantic_masks instead.
    
    Args:
        gt_detections: Ground truth Detections object
        pred_detections: Predicted Detections object  
        iou_threshold: IoU threshold for matching predictions to ground truths
        
    Returns:
        tuple (int, int, int): Number of true positives, false positives, and false negatives
    """
    if iou_threshold < 0.5:
        raise ValueError("iou_threshold must be greater than 0.5")
    
    # Extract polygons from detections using the unified method (no min_area filtering to match ftw_tools)
    gt_polygons = gt_detections.compute_polygons(min_area=0)
    pred_polygons = pred_detections.compute_polygons(min_area=0)
    
    # Compute matching using IoU (match ftw_tools greedy algorithm)
    tps = 0
    fns = 0
    matched_pred_indices = set()
    
    for i, gt_polygon in enumerate(gt_polygons):
        matching_j = None
        
        for j, pred_polygon in enumerate(pred_polygons):
            if gt_polygon.intersects(pred_polygon):
                intersection = gt_polygon.intersection(pred_polygon)
                union = gt_polygon.union(pred_polygon)
                iou = intersection.area / union.area if union.area > 0 else 0
                
                if iou > iou_threshold:
                    matching_j = j
                    matched_pred_indices.add(j)
                    break  # Stop at first match above threshold (greedy matching)
        
        if matching_j is not None:
            tps += 1
        else:
            fns += 1
    
    fps = len(pred_polygons) - len(matched_pred_indices)
    
    return (tps, fps, fns)


def get_pixel_level_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute pixel-level metrics from binary masks.
    
    Args:
        gt_mask: Ground truth mask (0=background, 1=field, 2=boundary for 3-class)
        pred_mask: Predicted mask (0=background, 1=field, 2=boundary for 3-class)
        
    Returns:
        Dictionary with pixel-level metrics
    """
    # DEBUG: Print mask statistics (only for first few calls)
    # import os
    # debug_count = int(os.environ.get('DEBUG_COUNT', '0'))
    # if debug_count < 3:
    #     print(f"DEBUG: GT mask shape: {gt_mask.shape}, dtype: {gt_mask.dtype}")
    #     print(f"DEBUG: GT mask unique values: {np.unique(gt_mask)}")
    #     print(f"DEBUG: GT mask value counts: {np.bincount(gt_mask.flatten())}")
    #     print(f"DEBUG: Pred mask shape: {pred_mask.shape}, dtype: {pred_mask.dtype}")
    #     print(f"DEBUG: Pred mask unique values: {np.unique(pred_mask)}")
    #     print(f"DEBUG: Pred mask value counts: {np.bincount(pred_mask.flatten())}")
    #     os.environ['DEBUG_COUNT'] = str(debug_count + 1)
    
    # Handle both 2-class and 3-class masks
    # For 3-class: 0=background, 1=field, 2=boundary
    # For 2-class: 0=background, 1=field
    if gt_mask.max() > 1 or pred_mask.max() > 1:
        # 3-class system: map 1=field, 0&2=background (following baseline_eval.py)
        # if debug_count < 3:
        #     print("DEBUG: Using 3-class system (1=field, 0&2=background)")
        gt_binary = (gt_mask == 1).astype(np.int64)
        pred_binary = (pred_mask == 1).astype(np.int64)
    else:
        # 2-class system: 0=background, 1=field
        # if debug_count < 3:
        #     print("DEBUG: Using 2-class system (0=background, 1=field)")
        gt_binary = (gt_mask == 1).astype(np.int64)
        pred_binary = (pred_mask == 1).astype(np.int64)
    
    # DEBUG: Print binary mask statistics
    # if debug_count < 3:
    #     print(f"DEBUG: GT binary unique values: {np.unique(gt_binary)}")
    #     print(f"DEBUG: GT binary value counts: {np.bincount(gt_binary.flatten())}")
    #     print(f"DEBUG: Pred binary unique values: {np.unique(pred_binary)}")
    #     print(f"DEBUG: Pred binary value counts: {np.bincount(pred_binary.flatten())}")
    
    # Confusion matrix components
    tp = np.sum((gt_binary == 1) & (pred_binary == 1))  # True positives
    fp = np.sum((gt_binary == 0) & (pred_binary == 1))  # False positives
    fn = np.sum((gt_binary == 1) & (pred_binary == 0))  # False negatives
    tn = np.sum((gt_binary == 0) & (pred_binary == 0))  # True negatives
    
    # DEBUG: Print confusion matrix
    # if debug_count < 3:
    #     print(f"DEBUG: Confusion matrix - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    #     print(f"DEBUG: Total pixels: {tp + fp + fn + tn}")
    #     print(f"DEBUG: GT field pixels: {np.sum(gt_binary == 1)}")
    #     print(f"DEBUG: Pred field pixels: {np.sum(pred_binary == 1)}")
    #     print("DEBUG: " + "="*50)
    
    # Pixel-level metrics
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Field class metrics (IoU, precision, recall, F1)
    field_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    field_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    field_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    field_f1 = 2 * field_precision * field_recall / (field_precision + field_recall) if (field_precision + field_recall) > 0 else 0
    
    # Background class metrics
    background_iou = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
    background_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    background_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Mean IoU
    mean_iou = (field_iou + background_iou) / 2
    
    # Multi-class averaged metrics
    pixel_precision_multiclass = (field_precision + background_precision) / 2
    pixel_recall_multiclass = (field_recall + background_recall) / 2
    pixel_f1_multiclass = 2 * pixel_precision_multiclass * pixel_recall_multiclass / (pixel_precision_multiclass + pixel_recall_multiclass) if (pixel_precision_multiclass + pixel_recall_multiclass) > 0 else 0
    
    return {
        "pixel_accuracy": pixel_accuracy * 100,
        "mean_iou": mean_iou * 100,
        "pixel_precision": pixel_precision_multiclass * 100,
        "pixel_recall": pixel_recall_multiclass * 100,
        "pixel_f1": pixel_f1_multiclass * 100,
        "pixel_iou_field": field_iou * 100,
        "pixel_precision_field": field_precision * 100,
        "pixel_recall_field": field_recall * 100,
        "pixel_f1_field": field_f1 * 100,
        "pixel_tp": int(tp),
        "pixel_fp": int(fp),
        "pixel_fn": int(fn),
        "pixel_tn": int(tn),
    }



class Evaluator:
    """
    Comprehensive evaluator for object detection and segmentation models.
    Supports pixel-level, object-level, and COCO metrics.
    """
    def __init__(
        self, 
        iou_threshold=0.5, 
        metrics: List[str] = ["pixel", "object", "coco"],
        output_dir: Optional[str] = None,
        gt_masks: Optional[List[np.ndarray]] = None,
        image_ids: Optional[List[int]] = None,
        use_semantic_masks_for_object_metrics: bool = False
    ):
        """
        Args:
            iou_threshold: IoU threshold for object-level metrics
            metrics: List of metrics to compute ("pixel", "object", "coco")
            output_dir: Directory to save results
            gt_masks: List of ground truth binary masks for pixel-level metrics
            image_ids: List of image IDs for COCO format
            use_semantic_masks_for_object_metrics: If True, use semantic masks (connected components) 
                for object-level metrics instead of instance masks. This is the correct approach 
                for evaluating semantic segmentation models, matching ftw_tools.training.metrics behavior.
        """
        self.iou_threshold = iou_threshold
        self.metrics = metrics
        self.output_dir = output_dir
        self.gt_masks = gt_masks
        self.image_ids = image_ids
        self.use_semantic_masks_for_object_metrics = use_semantic_masks_for_object_metrics
        self.results = {}
        
        # Initialize accumulators
        self._reset_accumulators()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
            self.logger.addHandler(logging.StreamHandler())
        
        # Validate semantic mask mode
        if use_semantic_masks_for_object_metrics and gt_masks is None:
            raise ValueError("gt_masks must be provided when use_semantic_masks_for_object_metrics=True")

    def _reset_accumulators(self):
        """Reset all metric accumulators."""
        # Pixel-level metrics
        if "pixel" in self.metrics:
            self._confusion_matrix = np.zeros((2, 2), dtype=np.int64)
        
        # Object-level metrics  
        if "object" in self.metrics:
            self._total_tps = 0
            self._total_fps = 0
            self._total_fns = 0
            self._total_gt_instances = 0
            self._total_pred_instances = 0
            self._total_confidence_sum = 0.0
            self._total_confidence_count = 0
        
        # COCO metrics
        if "coco" in self.metrics:
            self._coco_predictions = []
            self._coco_gt_annotations = []
            self._next_annotation_id = 1  # Track next available annotation ID
            self._all_image_ids = set()   # Track all evaluated image IDs (even with zero GT)

    def evaluate(self, y_true: List[Detections], y_pred: List[Detections]):
        """
        Evaluate a list of predictions against a list of ground truths.
        
        Args:
            y_true: List of ground truth Detections objects
            y_pred: List of predicted Detections objects
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Ground truth and predictions must have same length")
        
        self._reset_accumulators()
        
        # Store y_true and y_pred for statistics computation
        self._y_true = y_true
        self._y_pred = y_pred
        
        for i, (gt_dets, pred_dets) in enumerate(zip(y_true, y_pred)):
            # Note: Instance counting is now done in the object metrics section
            # to handle both semantic mask and instance-based modes correctly
            
            # Get image ID for COCO format
            image_id = self.image_ids[i] if self.image_ids else i
            # Track all image IDs encountered
            if "coco" in self.metrics:
                self._all_image_ids.add(image_id)
            
            # Pixel-level metrics
            if "pixel" in self.metrics and self.gt_masks:
                gt_mask = self.gt_masks[i]
                
                # Select field_class_id based on prediction class ids: prefer 1, else 0, else include all
                field_class_id = None
                try:
                    pred_class_ids = getattr(pred_dets, "class_id", None)
                    if pred_class_ids is not None and len(pred_class_ids) > 0:
                        unique_ids = set(int(x) for x in np.unique(pred_class_ids))
                        if 1 in unique_ids:
                            field_class_id = 1
                        elif 0 in unique_ids:
                            field_class_id = 0
                        else:
                            field_class_id = None
                except Exception:
                    field_class_id = None
                
                pred_mask = pred_dets.to_binary_mask(gt_mask.shape, field_class_id=field_class_id)
                
                # DEBUG: Print info about this image (only first 3 images)
                # if i < 3:
                #     print(f"DEBUG: Processing image {i} (image_id: {image_id})")
                #     print(f"DEBUG: GT detections: {len(gt_dets)} instances")
                #     print(f"DEBUG: Pred detections: {len(pred_dets)} instances")
                #     print(f"DEBUG: Using field_class_id={field_class_id} for pixel metrics")
                
                pixel_metrics = get_pixel_level_metrics(gt_mask, pred_mask)
                self._update_confusion_matrix(gt_mask, pred_mask)
            
            # Object-level metrics
            if "object" in self.metrics:
                if self.use_semantic_masks_for_object_metrics:
                    # Use semantic masks for object metrics (matches ftw_tools.training.metrics)
                    # This is correct for semantic segmentation models
                    gt_mask = self.gt_masks[i]
                    
                    # Convert prediction Detections to binary mask using the same method as pixel metrics
                    # Determine field_class_id (prefer 1, else 0)
                    field_class_id = None
                    try:
                        pred_class_ids = getattr(pred_dets, "class_id", None)
                        if pred_class_ids is not None and len(pred_class_ids) > 0:
                            unique_ids = set(int(x) for x in np.unique(pred_class_ids))
                            if 1 in unique_ids:
                                field_class_id = 1
                            elif 0 in unique_ids:
                                field_class_id = 0
                    except Exception:
                        field_class_id = None
                    
                    # Convert Detections to binary mask (combines all instance masks)
                    pred_mask = pred_dets.to_binary_mask(gt_mask.shape, field_class_id=field_class_id)
                    
                    # Use semantic mask-based metrics (extracts connected components)
                    tps, fps, fns = get_object_level_metrics_from_semantic_masks(
                        gt_mask, pred_mask, self.iou_threshold
                    )
                    
                    # Count instances from semantic masks (connected components)
                    # This matches the old evaluation behavior
                    gt_shapes = [geom for geom, val in rasterio.features.shapes(gt_mask.astype(np.uint8)) if val == 1]
                    pred_shapes = [geom for geom, val in rasterio.features.shapes(pred_mask.astype(np.uint8)) if val == 1]
                    self._total_gt_instances += len(gt_shapes)
                    self._total_pred_instances += len(pred_shapes)
                else:
                    # Use instance-based metrics (for instance segmentation models)
                    tps, fps, fns = get_object_level_metrics(gt_dets, pred_dets, self.iou_threshold)
                    self._total_gt_instances += len(gt_dets)
                    self._total_pred_instances += len(pred_dets)
                    
                    # Accumulate confidence scores (only for instance-based mode)
                    if pred_dets.confidence is not None and len(pred_dets.confidence) > 0:
                        self._total_confidence_sum += np.sum(pred_dets.confidence)
                        self._total_confidence_count += len(pred_dets.confidence)
                
                self._total_tps += tps
                self._total_fps += fps
                self._total_fns += fns
            
            # COCO format predictions
            if "coco" in self.metrics:
                # DEBUG: Check confidence scores (only first few images)
                if i < 3:
                    # self.logger.info(f"DEBUG: Image {i} (image_id={image_id}):")
                    # self.logger.info(f"  - Pred detections: {len(pred_dets)} instances")
                    
                    # Check if confidence is available and meaningful
                    if pred_dets.confidence is not None and len(pred_dets.confidence) > 0:
                        conf_array = np.array(pred_dets.confidence)
                        unique_vals = len(np.unique(conf_array))
                        conf_min = float(np.min(conf_array))
                        conf_max = float(np.max(conf_array))
                        conf_mean = float(np.mean(conf_array))
                        conf_std = float(np.std(conf_array))
                        
                        # self.logger.info(f"  - Confidence scores: min={conf_min:.4f}, max={conf_max:.4f}, "
                        #                f"mean={conf_mean:.4f}, std={conf_std:.4f}")
                        # self.logger.info(f"  - Unique confidence values: {unique_vals}")
                        
                        # Check if all scores are the same (not meaningful)
                        if unique_vals == 1:
                            self.logger.warning(f"  - WARNING: All predictions have same confidence ({conf_min:.4f})!")
                            self.logger.warning(f"    COCO mAP will compute but precision-recall curves won't reflect confidence ordering.")
                        elif unique_vals < len(conf_array) * 0.1:  # Less than 10% unique values
                            self.logger.warning(f"  - WARNING: Only {unique_vals}/{len(conf_array)} unique confidence values.")
                            self.logger.warning(f"    Confidence scores may not be meaningful for thresholding.")
                        
                        # Check if scores are in reasonable range [0, 1]
                        if conf_min < 0 or conf_max > 1:
                            self.logger.warning(f"  - WARNING: Confidence scores outside [0, 1] range!")
                    else:
                        self.logger.warning(f"  - WARNING: No confidence scores available!")
                        self.logger.warning(f"    All predictions will get score=1.0 in COCO format.")
                        self.logger.warning(f"    COCO mAP will compute but precision-recall curves won't reflect confidence ordering.")
                
                coco_preds = pred_dets.to_coco_format(image_id, self._next_annotation_id)
                # Normalize prediction category_id to 0 (ag_field) to match GT category id
                for p in coco_preds:
                    p["category_id"] = 0
                
                self._coco_predictions.extend(coco_preds)
                self._next_annotation_id += len(coco_preds)
                
                # Convert GT to COCO format as well
                coco_gt = gt_dets.to_coco_format(image_id, self._next_annotation_id)
                # Normalize GT category_id to 0 (ag_field)
                for g in coco_gt:
                    g["category_id"] = 0
                # if i < 3:
                #     print(f"DEBUG: Generated {len(coco_gt)} COCO ground truth annotations")
                #     if len(coco_gt) > 0:
                #         print(f"DEBUG: First GT annotation keys: {list(coco_gt[0].keys())}")
                #         print(f"DEBUG: First GT annotation: {coco_gt[0]}")
                
                self._coco_gt_annotations.extend(coco_gt)
                self._next_annotation_id += len(coco_gt)
        
        # Compute final results
        self.results = self._compute_final_results()
        
        # Save results if output directory specified
        if self.output_dir:
            self._save_results()
        
        return self.results
    

    def _update_confusion_matrix(self, gt_mask: np.ndarray, pred_mask: np.ndarray):
        """Update confusion matrix for pixel-level metrics."""
        # Handle both 2-class and 3-class masks
        if gt_mask.max() > 1 or pred_mask.max() > 1:
            # 3-class system: map 1=field, 0&2=background (following baseline_eval.py)
            gt_binary = (gt_mask == 1).astype(np.int64)
            pred_binary = (pred_mask == 1).astype(np.int64)
        else:
            # 2-class system: 0=background, 1=field
            gt_binary = (gt_mask == 1).astype(np.int64)
            pred_binary = (pred_mask == 1).astype(np.int64)
        
        for gt_val in [0, 1]:
            for pred_val in [0, 1]:
                self._confusion_matrix[gt_val, pred_val] += np.sum(
                    (gt_binary == gt_val) & (pred_binary == pred_val)
                )

    def _compute_final_results(self) -> Dict[str, Any]:
        """Compute final results from accumulated metrics."""
        results = {}
        
        # Pixel-level metrics
        if "pixel" in self.metrics and hasattr(self, '_confusion_matrix'):
            confusion_matrix = self._confusion_matrix
            tp = confusion_matrix[1, 1]
            fp = confusion_matrix[0, 1]
            fn = confusion_matrix[1, 0]
            tn = confusion_matrix[0, 0]
            
            # Compute pixel-level metrics
            pixel_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            field_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            field_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            field_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            field_f1 = 2 * field_precision * field_recall / (field_precision + field_recall) if (field_precision + field_recall) > 0 else 0
            
            results.update({
                "pixel_accuracy": pixel_accuracy * 100,
                "pixel_iou_field": field_iou * 100,
                "pixel_precision_field": field_precision * 100,
                "pixel_recall_field": field_recall * 100,
                "pixel_f1_field": field_f1 * 100,
                "pixel_tp": int(tp),
                "pixel_fp": int(fp),
                "pixel_fn": int(fn),
                "pixel_tn": int(tn),
            })
        
        # Object-level metrics
        if "object" in self.metrics:
            object_precision = self._total_tps / (self._total_tps + self._total_fps) if (self._total_tps + self._total_fps) > 0 else 0
            object_recall = self._total_tps / (self._total_tps + self._total_fns) if (self._total_tps + self._total_fns) > 0 else 0
            object_f1 = 2 * object_precision * object_recall / (object_precision + object_recall) if (object_precision + object_recall) > 0 else 0
            
            # Compute average instances per image
            num_images = len(self._y_true) if hasattr(self, '_y_true') else 0
            avg_gt_instances = self._total_gt_instances / num_images if num_images > 0 else 0
            avg_pred_instances = self._total_pred_instances / num_images if num_images > 0 else 0
            
            # Compute average confidence
            avg_confidence = self._total_confidence_sum / self._total_confidence_count if self._total_confidence_count > 0 else 0.0
            
            results.update({
                "object_precision": object_precision * 100,
                "object_recall": object_recall * 100,
                "object_f1": object_f1 * 100,
                "object_tps": self._total_tps,
                "object_fps": self._total_fps,
                "object_fns": self._total_fns,
                "total_gt_instances": self._total_gt_instances,
                "total_pred_instances": self._total_pred_instances,
                "avg_gt_instances_per_image": avg_gt_instances,
                "avg_pred_instances_per_image": avg_pred_instances,
                "avg_confidence": avg_confidence,
            })
        
        # COCO metrics
        if "coco" in self.metrics and len(self._coco_predictions) > 0:
            coco_results = self._compute_coco_metrics()
            results.update(coco_results)
        
        return results

    def _compute_coco_metrics(self) -> Dict[str, float]:
        """Compute COCO metrics using pycocotools."""
        try:
            self.logger.info(f"Computing COCO metrics: {len(self._coco_predictions)} predictions, "
                           f"{len(self._coco_gt_annotations)} GT annotations")
            
            # Check confidence scores across all predictions
            if len(self._coco_predictions) > 0:
                scores = [p.get("score", 1.0) for p in self._coco_predictions]
                scores_array = np.array(scores)
                unique_scores = len(np.unique(scores_array))
                score_min = float(np.min(scores_array))
                score_max = float(np.max(scores_array))
                score_mean = float(np.mean(scores_array))
                score_std = float(np.std(scores_array))
                
                self.logger.info(f"COCO prediction scores: min={score_min:.4f}, max={score_max:.4f}, "
                               f"mean={score_mean:.4f}, std={score_std:.4f}, unique={unique_scores}")
                
                if unique_scores == 1:
                    self.logger.warning(f"All {len(scores)} predictions have same score ({score_min:.4f})!")
                    self.logger.warning("COCO mAP will compute correctly, but precision-recall curves won't reflect confidence ordering.")
                elif unique_scores < len(scores) * 0.1:
                    self.logger.warning(f"Only {unique_scores}/{len(scores)} unique scores - confidence may not be meaningful for thresholding.")
                
                # Check score range
                if score_min < 0 or score_max > 1:
                    self.logger.warning(f"Scores outside [0, 1] range: [{score_min:.4f}, {score_max:.4f}]")
            
            # Create COCO ground truth dataset
            coco_gt = self._create_coco_gt()
            
            # Filter predictions to only those image_ids present in COCO GT
            valid_image_ids = set(coco_gt.imgs.keys())
            filtered_preds = [p for p in self._coco_predictions if p.get("image_id") in valid_image_ids]
            if len(filtered_preds) != len(self._coco_predictions):
                self.logger.warning(f"Filtered {len(self._coco_predictions) - len(filtered_preds)} predictions with unknown image_ids")
            
            # Load predictions
            coco_dt = coco_gt.loadRes(filtered_preds)
            
            # Run evaluation
            coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
            
            # Log COCOeval parameters
            self.logger.info(f"COCOeval parameters: {len(coco_eval.params.iouThrs)} IoU thresholds "
                           f"({coco_eval.params.iouThrs[0]:.2f} to {coco_eval.params.iouThrs[-1]:.2f}), "
                           f"{len(coco_eval.params.recThrs)} recall thresholds, "
                           f"maxDets={coco_eval.params.maxDets}")
            
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            coco_results = {}
            if coco_eval.eval:
                stats = coco_eval.stats
                # print(f"DEBUG: COCO stats: {stats}")
                coco_results = {
                    "coco_AP": float(stats[0]) * 100,      # AP @ IoU=0.50:0.95
                    "coco_AP50": float(stats[1]) * 100,    # AP @ IoU=0.50
                    "coco_AP75": float(stats[2]) * 100,    # AP @ IoU=0.75
                    "coco_APs": float(stats[3]) * 100,     # AP for small objects
                    "coco_APm": float(stats[4]) * 100,     # AP for medium objects
                    "coco_APl": float(stats[5]) * 100,     # AP for large objects
                    # AR metrics (default IoU=0.50:0.95)
                    "coco_AR1": float(stats[6]) * 100,     # AR @ maxDets=1
                    "coco_AR10": float(stats[7]) * 100,    # AR @ maxDets=10
                    "coco_AR100": float(stats[8]) * 100,   # AR @ maxDets=100
                    "coco_ARs": float(stats[9]) * 100,     # AR for small objects
                    "coco_ARm": float(stats[10]) * 100,    # AR for medium objects
                    "coco_ARl": float(stats[11]) * 100,    # AR for large objects
                }
                # print(f"DEBUG: COCO results: {coco_results}")
            # else:
            #     print(f"DEBUG: COCO evaluation failed - no eval results")
            
            return coco_results
            
        except Exception as e:
            self.logger.warning(f"COCO evaluation failed: {str(e)}")
            return {
                "coco_AP": float('nan'),
                "coco_AP50": float('nan'),
                "coco_AP75": float('nan'),
                "coco_APs": float('nan'),
                "coco_APm": float('nan'),
                "coco_APl": float('nan'),
            }

    def _create_coco_gt(self) -> COCO:
        """Create COCO ground truth dataset from annotations."""
        # Create categories
        categories = [
            {"id": 0, "name": "ag_field", "supercategory": "landcover"},
        ]
        
        # Create images (assuming 256x256 for now)
        images = []
        unique_image_ids = set()
        
        # Include all evaluated image IDs, even if no GT annotations
        if hasattr(self, "_all_image_ids") and len(self._all_image_ids) > 0:
            unique_image_ids.update(self._all_image_ids)
        
        # Also include any image_ids present in GT annotations (for safety)
        for ann in self._coco_gt_annotations:
            unique_image_ids.add(ann["image_id"])
        
        # print(f"DEBUG: Found {len(unique_image_ids)} unique image IDs in GT annotations/evaluated set")
        # print(f"DEBUG: Image IDs: {sorted(list(unique_image_ids))[:10]}{'...' if len(unique_image_ids) > 10 else ''}")
        
        for image_id in sorted(list(unique_image_ids)):
            images.append({
                "id": image_id,
                "width": 256,
                "height": 256,
                "file_name": f"image_{image_id}.png"
            })
        
        # print(f"DEBUG: Created {len(images)} image entries")
        # print(f"DEBUG: GT annotations sample: {self._coco_gt_annotations[:2] if self._coco_gt_annotations else 'None'}")
        
        # Create dataset
        coco_dataset = {
            "info": {
                "description": "Fields of the World",
                "version": "1.0",
                "year": 2024,
                "contributor": "FTW Bakeoff",
                "date_created": "2025-11-18"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": "https://fieldsofthe.world"
                }
            ],
            "images": images,
            "annotations": self._coco_gt_annotations,
            "categories": categories
        }
        
        # print(f"DEBUG: COCO dataset structure:")
        # print(f"DEBUG: - Images: {len(coco_dataset['images'])}")
        # print(f"DEBUG: - Annotations: {len(coco_dataset['annotations'])}")
        # print(f"DEBUG: - Categories: {len(coco_dataset['categories'])}")
        
        # Create COCO object
        coco_gt = COCO()
        coco_gt.dataset = coco_dataset
        coco_gt.createIndex()
        
        # print(f"DEBUG: COCO GT index created successfully")
        
        return coco_gt

    def _save_results(self):
        """Save evaluation results to output directory."""
        if not self.output_dir:
            return
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save main results
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save COCO predictions if available
        if "coco" in self.metrics and len(self._coco_predictions) > 0:
            coco_path = os.path.join(self.output_dir, "coco_predictions.json")
            with open(coco_path, 'w') as f:
                json.dump(self._coco_predictions, f, indent=2)

    def print_results(self):
        """Print evaluation results in a formatted table."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Pixel-level results
        if any(k.startswith('pixel_') for k in self.results):
            print("\nPixel-level Metrics:")
            print("-" * 30)
            pixel_metrics = {k: v for k, v in self.results.items() if k.startswith('pixel_')}
            for metric, value in pixel_metrics.items():
                if isinstance(value, float):
                    print(f"{metric:25}: {value:8.2f}")
                else:
                    print(f"{metric:25}: {value:8}")
        
        # Object-level results
        if any(k.startswith('object_') for k in self.results):
            print("\nObject-level Metrics:")
            print("-" * 30)
            object_metrics = {k: v for k, v in self.results.items() if k.startswith('object_')}
            for metric, value in object_metrics.items():
                if isinstance(value, float):
                    # Add % symbol for percentage metrics (those multiplied by 100)
                    if 'precision' in metric or 'recall' in metric or 'f1' in metric:
                        print(f"{metric:25}: {value:8.2f}%")
                    else:
                        print(f"{metric:25}: {value:8.2f}")
                else:
                    print(f"{metric:25}: {value:8}")
        
        # COCO results
        if any(k.startswith('coco_') for k in self.results):
            print("\nCOCO Metrics:")
            print("-" * 30)
            coco_metrics = {k: v for k, v in self.results.items() if k.startswith('coco_')}
            for metric, value in coco_metrics.items():
                if isinstance(value, float) and not np.isnan(value):
                    print(f"{metric:25}: {value:8.2f}")
                else:
                    print(f"{metric:25}: {'N/A':8}")
        
        print("="*60)
