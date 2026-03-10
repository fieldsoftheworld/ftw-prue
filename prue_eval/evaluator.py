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

from .detections import Detections


def get_object_level_metrics_from_semantic_masks(gt_mask: np.ndarray, pred_mask: np.ndarray, iou_threshold=0.5):
    """Object-level TP/FP/FN from semantic masks via connected components (matches ftw_tools)."""
    if iou_threshold < 0.5:
        raise ValueError("iou_threshold must be >= 0.5")

    gt_shapes = [shapely.geometry.shape(geom) for geom, val in rasterio.features.shapes(gt_mask.astype(np.uint8)) if val == 1]
    pred_shapes = [shapely.geometry.shape(geom) for geom, val in rasterio.features.shapes(pred_mask.astype(np.uint8)) if val == 1]

    tps = 0
    fns = 0
    matched_pred_indices = set()

    for gt_shape in gt_shapes:
        matched = False
        for j, pred_shape in enumerate(pred_shapes):
            if gt_shape.intersects(pred_shape):
                intersection = gt_shape.intersection(pred_shape)
                union = gt_shape.union(pred_shape)
                iou = intersection.area / union.area if union.area > 0 else 0
                if iou > iou_threshold:
                    matched = True
                    matched_pred_indices.add(j)
                    break
        if matched:
            tps += 1
        else:
            fns += 1

    fps = len(pred_shapes) - len(matched_pred_indices)
    return (tps, fps, fns)


def get_object_level_metrics(gt_detections: Detections, pred_detections: Detections, iou_threshold=0.5):
    """Object-level TP/FP/FN from Detections via polygon IoU matching."""
    if iou_threshold < 0.5:
        raise ValueError("iou_threshold must be >= 0.5")

    gt_polygons = gt_detections.compute_polygons(min_area=0)
    pred_polygons = pred_detections.compute_polygons(min_area=0)

    tps = 0
    fns = 0
    matched_pred_indices = set()

    for gt_polygon in gt_polygons:
        matched = False
        for j, pred_polygon in enumerate(pred_polygons):
            if gt_polygon.intersects(pred_polygon):
                intersection = gt_polygon.intersection(pred_polygon)
                union = gt_polygon.union(pred_polygon)
                iou = intersection.area / union.area if union.area > 0 else 0
                if iou > iou_threshold:
                    matched = True
                    matched_pred_indices.add(j)
                    break
        if matched:
            tps += 1
        else:
            fns += 1

    fps = len(pred_polygons) - len(matched_pred_indices)
    return (tps, fps, fns)


def get_pixel_level_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """Pixel-level metrics (IoU, precision, recall, F1) from binary masks."""
    # Map to binary: 1=field, 0=everything else
    gt_binary = (gt_mask == 1).astype(np.int64)
    pred_binary = (pred_mask == 1).astype(np.int64)

    tp = np.sum((gt_binary == 1) & (pred_binary == 1))
    fp = np.sum((gt_binary == 0) & (pred_binary == 1))
    fn = np.sum((gt_binary == 1) & (pred_binary == 0))
    tn = np.sum((gt_binary == 0) & (pred_binary == 0))

    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    field_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    field_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    field_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    field_f1 = (
        2 * field_precision * field_recall / (field_precision + field_recall)
        if (field_precision + field_recall) > 0
        else 0
    )

    background_iou = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
    background_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    background_recall = tn / (tn + fp) if (tn + fp) > 0 else 0

    mean_iou = (field_iou + background_iou) / 2
    pixel_precision_mc = (field_precision + background_precision) / 2
    pixel_recall_mc = (field_recall + background_recall) / 2
    pixel_f1_mc = (
        2 * pixel_precision_mc * pixel_recall_mc / (pixel_precision_mc + pixel_recall_mc)
        if (pixel_precision_mc + pixel_recall_mc) > 0
        else 0
    )

    return {
        "pixel_accuracy": pixel_accuracy * 100,
        "mean_iou": mean_iou * 100,
        "pixel_precision": pixel_precision_mc * 100,
        "pixel_recall": pixel_recall_mc * 100,
        "pixel_f1": pixel_f1_mc * 100,
        "pixel_iou_field": field_iou * 100,
        "pixel_precision_field": field_precision * 100,
        "pixel_recall_field": field_recall * 100,
        "pixel_f1_field": field_f1 * 100,
        "pixel_tp": int(tp),
        "pixel_fp": int(fp),
        "pixel_fn": int(fn),
        "pixel_tn": int(tn),
    }


def _resolve_field_class_id(pred_dets: "Detections") -> Optional[int]:
    """Pick field class id from prediction class_ids: prefer 1, else 0, else None."""
    try:
        pred_class_ids = getattr(pred_dets, "class_id", None)
        if pred_class_ids is not None and len(pred_class_ids) > 0:
            unique_ids = set(int(x) for x in np.unique(pred_class_ids))
            if 1 in unique_ids:
                return 1
            if 0 in unique_ids:
                return 0
    except Exception:
        pass
    return None


class Evaluator:
    """Evaluator for segmentation models. Supports pixel, object, and COCO metrics."""

    def __init__(
        self,
        iou_threshold=0.5,
        metrics: List[str] = ["pixel", "object", "coco"],
        output_dir: Optional[str] = None,
        gt_masks: Optional[List[np.ndarray]] = None,
        image_ids: Optional[List[int]] = None,
        use_semantic_masks_for_object_metrics: bool = False,
    ):
        self.iou_threshold = iou_threshold
        self.metrics = metrics
        self.output_dir = output_dir
        self.gt_masks = gt_masks
        self.image_ids = image_ids
        self.use_semantic_masks_for_object_metrics = use_semantic_masks_for_object_metrics
        self.results = {}

        self._reset_accumulators()

        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

        if use_semantic_masks_for_object_metrics and gt_masks is None:
            raise ValueError("gt_masks required when use_semantic_masks_for_object_metrics=True")

    def _reset_accumulators(self):
        if "pixel" in self.metrics:
            self._confusion_matrix = np.zeros((2, 2), dtype=np.int64)

        if "object" in self.metrics:
            self._total_tps = 0
            self._total_fps = 0
            self._total_fns = 0
            self._total_gt_instances = 0
            self._total_pred_instances = 0
            self._total_confidence_sum = 0.0
            self._total_confidence_count = 0

        if "coco" in self.metrics:
            self._coco_predictions = []
            self._coco_gt_annotations = []
            self._next_annotation_id = 1
            self._all_image_ids = set()

    def evaluate(self, y_true: List[Detections], y_pred: List[Detections]):
        if len(y_true) != len(y_pred):
            raise ValueError("Ground truth and predictions must have same length")

        self._reset_accumulators()
        self._y_true = y_true
        self._y_pred = y_pred

        for i, (gt_dets, pred_dets) in enumerate(zip(y_true, y_pred)):
            image_id = self.image_ids[i] if self.image_ids else i
            if "coco" in self.metrics:
                self._all_image_ids.add(image_id)

            field_class_id = _resolve_field_class_id(pred_dets)

            if "pixel" in self.metrics and self.gt_masks:
                gt_mask = self.gt_masks[i]
                pred_mask = pred_dets.to_binary_mask(gt_mask.shape, field_class_id=field_class_id)
                self._update_confusion_matrix(gt_mask, pred_mask)

            if "object" in self.metrics:
                if self.use_semantic_masks_for_object_metrics:
                    gt_mask = self.gt_masks[i]
                    pred_mask = pred_dets.to_binary_mask(gt_mask.shape, field_class_id=field_class_id)
                    tps, fps, fns = get_object_level_metrics_from_semantic_masks(gt_mask, pred_mask, self.iou_threshold)
                    gt_shapes = [g for g, v in rasterio.features.shapes(gt_mask.astype(np.uint8)) if v == 1]
                    pred_shapes = [g for g, v in rasterio.features.shapes(pred_mask.astype(np.uint8)) if v == 1]
                    self._total_gt_instances += len(gt_shapes)
                    self._total_pred_instances += len(pred_shapes)
                else:
                    tps, fps, fns = get_object_level_metrics(gt_dets, pred_dets, self.iou_threshold)
                    self._total_gt_instances += len(gt_dets)
                    self._total_pred_instances += len(pred_dets)
                    if pred_dets.confidence is not None and len(pred_dets.confidence) > 0:
                        self._total_confidence_sum += np.sum(pred_dets.confidence)
                        self._total_confidence_count += len(pred_dets.confidence)

                self._total_tps += tps
                self._total_fps += fps
                self._total_fns += fns

            if "coco" in self.metrics:
                coco_preds = pred_dets.to_coco_format(image_id, self._next_annotation_id)
                for p in coco_preds:
                    p["category_id"] = 0
                self._coco_predictions.extend(coco_preds)
                self._next_annotation_id += len(coco_preds)

                coco_gt = gt_dets.to_coco_format(image_id, self._next_annotation_id)
                for g in coco_gt:
                    g["category_id"] = 0
                self._coco_gt_annotations.extend(coco_gt)
                self._next_annotation_id += len(coco_gt)

        self.results = self._compute_final_results()

        if self.output_dir:
            self._save_results()

        return self.results

    def _update_confusion_matrix(self, gt_mask: np.ndarray, pred_mask: np.ndarray):
        gt_binary = (gt_mask == 1).astype(np.int64)
        pred_binary = (pred_mask == 1).astype(np.int64)
        for gt_val in [0, 1]:
            for pred_val in [0, 1]:
                self._confusion_matrix[gt_val, pred_val] += np.sum((gt_binary == gt_val) & (pred_binary == pred_val))

    def _compute_final_results(self) -> Dict[str, Any]:
        results = {}

        if "pixel" in self.metrics and hasattr(self, "_confusion_matrix"):
            cm = self._confusion_matrix
            tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]

            pixel_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            field_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            field_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            field_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            field_f1 = (
                2 * field_precision * field_recall / (field_precision + field_recall)
                if (field_precision + field_recall) > 0
                else 0
            )

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

        if "object" in self.metrics:
            obj_prec = self._total_tps / (self._total_tps + self._total_fps) if (self._total_tps + self._total_fps) > 0 else 0
            obj_rec = self._total_tps / (self._total_tps + self._total_fns) if (self._total_tps + self._total_fns) > 0 else 0
            obj_f1 = 2 * obj_prec * obj_rec / (obj_prec + obj_rec) if (obj_prec + obj_rec) > 0 else 0

            num_images = len(self._y_true) if hasattr(self, "_y_true") else 0
            avg_gt = self._total_gt_instances / num_images if num_images > 0 else 0
            avg_pred = self._total_pred_instances / num_images if num_images > 0 else 0
            avg_conf = self._total_confidence_sum / self._total_confidence_count if self._total_confidence_count > 0 else 0.0

            results.update({
                "object_precision": obj_prec * 100,
                "object_recall": obj_rec * 100,
                "object_f1": obj_f1 * 100,
                "object_tps": self._total_tps,
                "object_fps": self._total_fps,
                "object_fns": self._total_fns,
                "total_gt_instances": self._total_gt_instances,
                "total_pred_instances": self._total_pred_instances,
                "avg_gt_instances_per_image": avg_gt,
                "avg_pred_instances_per_image": avg_pred,
                "avg_confidence": avg_conf,
            })

        if "coco" in self.metrics and len(self._coco_predictions) > 0:
            coco_results = self._compute_coco_metrics()
            results.update(coco_results)

        return results

    def _compute_coco_metrics(self) -> Dict[str, float]:
        try:
            self.logger.info(
                f"COCO metrics: {len(self._coco_predictions)} predictions, "
                f"{len(self._coco_gt_annotations)} GT annotations"
            )

            if self._coco_predictions:
                scores = np.array([p.get("score", 1.0) for p in self._coco_predictions])
                if len(np.unique(scores)) == 1:
                    self.logger.warning(f"All {len(scores)} predictions have same score ({scores[0]:.4f})")

            coco_gt = self._create_coco_gt()

            valid_ids = set(coco_gt.imgs.keys())
            filtered = [p for p in self._coco_predictions if p.get("image_id") in valid_ids]
            if len(filtered) != len(self._coco_predictions):
                self.logger.warning(f"Filtered {len(self._coco_predictions) - len(filtered)} predictions with unknown image_ids")

            coco_dt = coco_gt.loadRes(filtered)
            coco_eval = COCOeval(coco_gt, coco_dt, "segm")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            if coco_eval.eval:
                s = coco_eval.stats
                return {
                    "coco_AP": float(s[0]) * 100,
                    "coco_AP50": float(s[1]) * 100,
                    "coco_AP75": float(s[2]) * 100,
                    "coco_APs": float(s[3]) * 100,
                    "coco_APm": float(s[4]) * 100,
                    "coco_APl": float(s[5]) * 100,
                    "coco_AR1": float(s[6]) * 100,
                    "coco_AR10": float(s[7]) * 100,
                    "coco_AR100": float(s[8]) * 100,
                    "coco_ARs": float(s[9]) * 100,
                    "coco_ARm": float(s[10]) * 100,
                    "coco_ARl": float(s[11]) * 100,
                }
            return {}

        except Exception as e:
            self.logger.warning(f"COCO evaluation failed: {e}")
            return {k: float("nan") for k in ["coco_AP", "coco_AP50", "coco_AP75", "coco_APs", "coco_APm", "coco_APl"]}

    def _create_coco_gt(self) -> COCO:
        categories = [{"id": 0, "name": "ag_field", "supercategory": "landcover"}]

        unique_ids = set()
        if hasattr(self, "_all_image_ids"):
            unique_ids.update(self._all_image_ids)
        for ann in self._coco_gt_annotations:
            unique_ids.add(ann["image_id"])

        images = [{"id": iid, "width": 256, "height": 256, "file_name": f"image_{iid}.png"} for iid in sorted(unique_ids)]

        coco_gt = COCO()
        coco_gt.dataset = {
            "info": {"description": "FTW COCO GT", "version": "1.0", "year": 2025},
            "licenses": [],
            "images": images,
            "annotations": self._coco_gt_annotations,
            "categories": categories,
        }
        coco_gt.createIndex()
        return coco_gt

    def _save_results(self):
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, "evaluation_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

        if "coco" in self.metrics and self._coco_predictions:
            with open(os.path.join(self.output_dir, "coco_predictions.json"), "w") as f:
                json.dump(self._coco_predictions, f, indent=2)

    def print_results(self):
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        for prefix, label in [("pixel_", "Pixel-level"), ("object_", "Object-level"), ("coco_", "COCO")]:
            section = {k: v for k, v in self.results.items() if k.startswith(prefix)}
            if not section:
                continue
            print(f"\n{label} Metrics:")
            print("-" * 30)
            for metric, value in section.items():
                if isinstance(value, float):
                    if not np.isnan(value):
                        print(f"{metric:25}: {value:8.2f}")
                    else:
                        print(f"{metric:25}: {'N/A':8}")
                else:
                    print(f"{metric:25}: {value:8}")

        print("=" * 60)
