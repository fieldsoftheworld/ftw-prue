import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import defaultdict, OrderedDict
from typing import Optional, List, Dict, Any, Union
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.structures import BoxMode
from detectron2.utils.logger import create_small_table
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.events import get_event_storage

import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import rasterio.features
import shapely.geometry
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches




def get_object_level_metrics(y_true, y_pred, iou_threshold=0.5):
    """Get object level metrics for a single mask / prediction pair.

    Args:
        y_true (np.ndarray): Ground truth mask.
        y_pred (np.ndarray): Predicted mask.
        iou_threshold (float, optional): IoU threshold for matching predictions to ground truths. Defaults to 0.5.

    Returns
        tuple (int, int, int): Number of true positives, false positives, and false negatives.
    """
    if iou_threshold < 0.5:
        raise ValueError("iou_threshold must be greater than 0.5")  # If we go lower than 0.5 then it is possible for a single prediction to match with multiple ground truths and we have to do de-duplication
    y_true_shapes = []
    for geom, val in rasterio.features.shapes(y_true):
        if val == 1:
            y_true_shapes.append(shapely.geometry.shape(geom))

    y_pred_shapes = []
    for geom, val in rasterio.features.shapes(y_pred):
        if val == 1:
            y_pred_shapes.append(shapely.geometry.shape(geom))

    tps = 0
    fns = 0
    tp_is = set()  # keep track of which of the true shapes are true positives
    tp_js = set()  # keep track of which of the predicted shapes are true positives
    fn_is = set()  # keep track of which of the true shapes are false negatives
    matched_js = set()
    for i, y_true_shape in enumerate(y_true_shapes):
        matching_j = None
        for j, y_pred_shape in enumerate(y_pred_shapes):
            if y_true_shape.intersects(y_pred_shape):
                intersection = y_true_shape.intersection(y_pred_shape)
                union = y_true_shape.union(y_pred_shape)
                iou = intersection.area / union.area
                if iou > iou_threshold:
                    matching_j = j
                    matched_js.add(j)
                    tp_js.add(j)
                    break
        if matching_j is not None:
            tp_is.add(i)
            tps += 1
        else:
            fn_is.add(i)
            fns += 1
    fps = len(y_pred_shapes) - len(matched_js)
    fp_js = set(range(len(y_pred_shapes))) - matched_js  # compute which of the predicted shapes are false positives

    return (tps, fps, fns)

def get_object_level_metrics_panoptic(gt_panoptic, gt_segments_info, pred_panoptic, pred_segments_info, iou_threshold=0.5):
    """Get object level metrics using panoptic segmentation format
    
    Args:
        gt_panoptic: Ground truth panoptic mask (numpy array)
        gt_segments_info: Ground truth segments info list
        pred_panoptic: Predicted panoptic mask (numpy array)
        pred_segments_info: Predicted segments info list
        iou_threshold: IoU threshold for matching
        
    Returns:
        tuple (int, int, int): Number of true positives, false positives, and false negatives
    """
    # Extract ground truth shapes for ag_field instances only
    y_true_shapes = []
    for segment in gt_segments_info:
        if segment.get("isthing", True): #and segment.get("category_id") == 0
            mask = gt_panoptic == segment["id"]
            if mask.any():
                # Find contours using rasterio. Want list of shapely.geometry.Polygon objects
                import rasterio.features
                geoms = [shapely.geometry.shape(geom) for geom, _ in rasterio.features.shapes(mask.astype(np.uint8), mask=(mask))]
                for geom in geoms:
                    y_true_shapes.append(geom)
    
    # Extract predicted shapes for ag_field instances only
    y_pred_shapes = []
    for segment in pred_segments_info:
        if segment.get("isthing", True): #and segment.get("category_id") == 0
            mask = pred_panoptic == segment["id"]
            if mask.any():
                import rasterio.features
                geoms = [shapely.geometry.shape(geom) for geom, _ in rasterio.features.shapes(mask.astype(np.uint8), mask=(mask))]
                # do filtering to remove small shapes (<50 pixels)
                for geom in geoms:
                    if geom.area > 50:
                        y_pred_shapes.append(geom)
    
    # Now use the same logic as the original get_object_level_metrics
    tps = 0
    fns = 0
    matched_js = set()
    
    # both are lists of dicts with keys 'type', 'coordinates'
    for i, y_true_shape in enumerate(y_true_shapes):
        matching_j = None
        for j, y_pred_shape in enumerate(y_pred_shapes):
            # convert y_true_shape and y_pred_shape to shapely.geometry.Polygon objects
            if y_true_shape.intersects(y_pred_shape):
                intersection = y_true_shape.intersection(y_pred_shape)
                union = y_true_shape.union(y_pred_shape)
                iou = intersection.area / union.area
                if iou > iou_threshold:
                    matching_j = j
                    matched_js.add(j)
                    break
        
        if matching_j is not None:
            tps += 1
        else:
            fns += 1
    
    fps = len(y_pred_shapes) - len(matched_js)
    
    return (tps, fps, fns)


##### FTW evaluator class definition #####
class FTWEvaluator(DatasetEvaluator):
    """
    Evaluator for Fields of the World dataset.
    Computes pixel-level and object-level metrics as in the FTW original dataset paper, 
    as well as COCO metrics.
    """
    
    def __init__(
        self,
        dataset_name: str,
        distributed: bool = True,
        output_dir: Optional[str] = None,
        country_names: List[str] = None,
        iou_threshold: float = 0.5,
        metrics: List[str] = ["pixel", "object", "coco"],
    ):
        """
        Args:
            dataset_name: Name of the dataset
            distributed: If True, will collect results from all ranks for evaluation
            output_dir: Directory to save evaluation results
            country_names: List of country names for per-country evaluation
            iou_threshold: IoU threshold for object-level metrics
            metrics: List of metrics to compute. Can include "pixel", "object", "coco"
        """
        import logging
        import sys
        
        # Configure root logger if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                ]
            )
        
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        
        # Force logger to use stdout
        if not self._logger.handlers:
            self._logger.addHandler(logging.StreamHandler(sys.stdout))
        
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._country_names = country_names
        self._iou_threshold = iou_threshold
        self._metrics = metrics
        self._prediction_type_counts = {"panoptic": 0, "semantic": 0, "instance": 0, "unknown": 0}
        # Disable expensive COCO debug visualizations by default to avoid long runtimes
        self._enable_coco_debug = False
        
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        
        # Get ground truth annotations
        self._dataset_dicts = DatasetCatalog.get(dataset_name) # category ids 0 and 1 are ag_field and background
        
        # Map file_name to country for per-country evaluation
        if self._country_names:
            self._file_to_country = {}
            for record in self._dataset_dicts:
                for country in self._country_names:
                    if country.lower() in record["file_name"].lower():
                        self._file_to_country[record["file_name"]] = country
                        break
        
        # Initialize COCO API if needed
        if "coco" in self._metrics:
            # COCO ground truth will be created manually from panoptic files in evaluate() method
            # No need to initialize here since the JSON file doesn't contain RLE masks
            pass
    
    def reset(self):
        """Reset evaluator"""
        # Pixel-level metrics accumulators
        if "pixel" in self._metrics:
            self._confusion_matrix = np.zeros((2, 2), dtype=np.int64)
            if self._country_names:
                self._country_confusion_matrices = {
                    country: np.zeros((2, 2), dtype=np.int64) for country in self._country_names
                }
        
        # Object-level metrics accumulators
        if "object" in self._metrics:
            self._total_tps = 0
            self._total_fps = 0
            self._total_fns = 0
            if self._country_names:
                self._country_tps = {country: 0 for country in self._country_names}
                self._country_fps = {country: 0 for country in self._country_names}
                self._country_fns = {country: 0 for country in self._country_names}
        
        # COCO metrics predictions
        if "coco" in self._metrics:
            self._coco_predictions = []
        
        # Store predictions for visualization or further analysis
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Process one batch of inputs and outputs.
        """
        # print("Starting process")

        for input_record, output in zip(inputs, outputs):
            file_name = input_record["file_name"]
            
            # Get country if doing per-country evaluation
            country = None
            if self._country_names:
                for c in self._country_names:
                    if c.lower() in file_name.lower():
                        country = c
                        break
            
            # Get ground truth masks separately for pixel and object metrics
            gt_binary = self._get_gt_mask(input_record)  # Binary mask for pixel metrics
            gt_panoptic, gt_segments_info = self._get_gt_panoptic(input_record)  # Panoptic for object metrics
            
            if gt_binary is None:
                continue
            
            # Get prediction mask for pixel-level metrics
            pred_binary = None
            panoptic_for_object = None
            
            if "panoptic_seg" in output:
                panoptic_seg, segments_info = output["panoptic_seg"]

                # Store original panoptic for object metrics
                panoptic_for_object = (panoptic_seg.to(self._cpu_device).numpy(), segments_info)
                
                # Convert to binary for pixel metrics
                pred_binary = self._panoptic_to_binary(
                    panoptic_seg.to(self._cpu_device).numpy(), 
                    segments_info
                    # is_prediction=True
                )

            elif "sem_seg" in output:
                pred_sem = output["sem_seg"].argmax(dim=0).to(self._cpu_device).numpy()
                pred_binary = (pred_sem == 0).astype(np.uint8)
            else:
                self._logger.warning(f"No supported prediction format found for {file_name}")
                continue

            # Process predictions for pixel-level metrics using binary masks
            self._update_confusion_matrix(gt_binary, pred_binary, country)
            
            # Process predictions for object-level metrics
            try:
                if panoptic_for_object is not None and gt_panoptic is not None:
                    # Use panoptic format for object metrics
                    pred_panoptic, pred_segments_info = panoptic_for_object
                    tps, fps, fns = get_object_level_metrics_panoptic(
                        gt_panoptic, gt_segments_info,
                        pred_panoptic, pred_segments_info,
                        iou_threshold=self._iou_threshold
                    )
                else:
                    # Fallback to rasterio-based method
                    tps, fps, fns = get_object_level_metrics(
                        gt_binary, pred_binary, iou_threshold=self._iou_threshold
                    )
                
                self._total_tps += tps
                self._total_fps += fps
                self._total_fns += fns
                
                if country and country in self._country_tps:
                    self._country_tps[country] += tps
                    self._country_fps[country] += fps
                    self._country_fns[country] += fns
            except Exception as e:
                self._logger.warning(f"Error computing object-level metrics for {file_name}: {str(e)}")
            
            # Update COCO format predictions
            if "coco" in self._metrics and panoptic_for_object is not None:
                pred_panoptic, pred_segments_info = panoptic_for_object
                coco_instances = []
                
                for segment in pred_segments_info:
                    if segment.get("isthing", True):  # check if it's a thing class
                        mask = (pred_panoptic == segment["id"]).astype(np.uint8)
                        if mask.sum() > 10:
                            # Convert mask to RLE
                            rle = mask_util.encode(np.asfortranarray(mask))
                            rle["counts"] = rle["counts"].decode("utf-8")
                            
                            # Get bbox from mask
                            bbox = mask_util.toBbox(rle).tolist()
                            
                            # Use the actual category_id from the prediction - no mapping needed
                            # Ground truth is remapped to match predictions (0 for ag_field)
                            pred_category_id = segment.get("category_id", 0)
                            # No mapping needed - use the original category_id
                            coco_category_id = pred_category_id

                            coco_instances.append({
                                "image_id": input_record["image_id"], # file_to_id.get(input_record["file_name"], input_record["image_id"]),  # Use the image_id from input record to match ground truth
                                "file_name": input_record["file_name"],  # Add file_name for mapping
                                "category_id": coco_category_id,  # Use original category_id (no mapping needed)
                                "segmentation": rle,
                                "bbox": bbox,
                                "score": segment.get("score", 1.0)  # Use score if available
                            })
                
                if coco_instances:
                    self._coco_predictions.extend(coco_instances)

    def _get_gt_mask(self, input_record):
        """
        Get ground truth mask from input record - prioritize panoptic for better comparison
        """
        # First try to get panoptic segmentation mask for consistency with predicted panoptic output
        if "pan_seg_file_name" in input_record:
            # Get the root directory
            dataset_dir = os.path.dirname(self._output_dir) if self._output_dir else ""
            
            # Try direct path first
            pan_seg_file = input_record["pan_seg_file_name"]
            
            # If not absolute, try relative to dataset dir
            if not os.path.isabs(pan_seg_file) and dataset_dir:
                pan_seg_file = os.path.join(dataset_dir, pan_seg_file)
            
            # Try to open the file
            try:
                from PIL import Image
                with PathManager.open(pan_seg_file, "rb") as f:
                    pan_seg = np.array(Image.open(f))

                
                # Extract segments info if available
                segments_info = input_record.get("segments_info", [])
                if segments_info:
                    from panopticapi.utils import rgb2id
                    return self._panoptic_to_binary(rgb2id(pan_seg), segments_info)#, is_prediction=False)
                else:
                    # If no segments_info, assume 0 = ag_field, other values = background
                    return (pan_seg == 0).astype(np.uint8)
            except Exception as e:
                self._logger.warning(f"Error reading panoptic mask {pan_seg_file}: {str(e)}")
        
        # Fallback to semantic segmentation
        elif "sem_seg_file_name" in input_record:
            # Get the root directory (parent of output_dir)
            dataset_dir = os.path.dirname(self._output_dir) if self._output_dir else ""
            
            # Try direct path first
            sem_seg_file = input_record["sem_seg_file_name"]
            
            # If not absolute, try relative to dataset dir
            if not os.path.isabs(sem_seg_file) and dataset_dir:
                sem_seg_file = os.path.join(dataset_dir, sem_seg_file)
            
            # Try to open the file
            try:
                from PIL import Image
                with PathManager.open(sem_seg_file, "rb") as f:
                    gt_mask = np.array(Image.open(f))
                    
                # Convert to binary mask where 1 = field (for metrics calculation)
                # In the semantic segmentation PNG, 0 = ag_field, 1 = background
                gt_binary = (gt_mask == 0).astype(np.uint8)
                return gt_binary
            except Exception as e:
                self._logger.warning(f"Error reading semantic mask {sem_seg_file}: {str(e)}")
        
        # If all else fails, try to infer the path
        try:
            # Get the dataset directory and split
            split = self._dataset_name.split('_')[-1]  # e.g., 'test' from 'ftw_test'
            image_path = input_record["file_name"]
            
            # If image_path is relative, make it absolute
            if not os.path.isabs(image_path) and self._output_dir:
                dataset_dir = os.path.dirname(self._output_dir)
                image_path = os.path.join(dataset_dir, image_path)
            
            # Extract base name
            base_name = os.path.basename(image_path).replace(".npz", ".png")
            
            # Try panoptic_semseg path
            sem_seg_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), f"panoptic_{split}")
            sem_seg_path = os.path.join(sem_seg_dir, base_name)
            
            if os.path.exists(sem_seg_path):
                from PIL import Image
                with open(sem_seg_path, "rb") as f:
                    gt_mask = np.array(Image.open(f))
                # Convert to binary where 1 = field
                return (gt_mask == 0).astype(np.uint8)
        except Exception as e:
            self._logger.warning(f"Error inferring mask path: {str(e)}")
        
        self._logger.warning(f"No ground truth mask found for input: {input_record['file_name']}")
        return None

    def _get_gt_panoptic(self, input_record):
        """
        Get ground truth panoptic segmentation from input record
        
        Returns:
            Tuple[np.ndarray, List[Dict]]: (panoptic_seg, segments_info)
        """
        if "pan_seg_file_name" in input_record:
            # Get the root directory
            dataset_dir = os.path.dirname(self._output_dir) if self._output_dir else ""
            
            # Try direct path first
            pan_seg_file = input_record["pan_seg_file_name"]
            
            # If not absolute, try relative to dataset dir
            if not os.path.isabs(pan_seg_file) and dataset_dir:
                pan_seg_file = os.path.join(dataset_dir, pan_seg_file)
            
            # Try to open the file
            try:
                from PIL import Image
                with PathManager.open(pan_seg_file, "rb") as f:
                    pan_seg = np.array(Image.open(f))
                    # if pan_seg is 3 channel, convert to 1 channel with rgb2id
                    if pan_seg.shape[2] == 3:
                        from panopticapi.utils import rgb2id
                        pan_seg = rgb2id(pan_seg)
                
                # Extract segments info if available
                segments_info = input_record.get("segments_info", [])
                
                return pan_seg, segments_info
            except Exception as e:
                self._logger.warning(f"Error reading ground truth panoptic mask {pan_seg_file}: {str(e)}")
        
        return None, None

    def _panoptic_to_binary(self, panoptic_seg, segments_info):#    , is_prediction=True):
        """
        Convert panoptic segmentation to binary mask (1 for agricultural field, 0 for background)
        """
        binary_mask = np.zeros_like(panoptic_seg, dtype=np.uint8)
        
        # For both predictions and ground truth, only include thing classes (ag_field instances)
        # Background (stuff class) should not be included in binary mask
        for segment in segments_info:
            # Only include agricultural field instances (isthing=True and category_id=0 for ag_field)
            if segment.get("isthing", True) and segment.get("category_id", 0) == 0:
                binary_mask[panoptic_seg == segment["id"]] = 1
        
        return binary_mask
    
    def _update_confusion_matrix(self, gt_mask, pred_mask, country=None):
        """
        Update confusion matrix for pixel-level metrics
        
        Args:
            gt_mask: Ground truth mask
            pred_mask: Prediction mask
            country: Country name for per-country evaluation
        """
        # Ensure masks are binary (0 for background, 1 for field)
        gt_binary = (gt_mask == 1).astype(np.int64)
        pred_binary = (pred_mask == 1).astype(np.int64)
        
        # Update overall confusion matrix
        for gt_val in [0, 1]:
            for pred_val in [0, 1]:
                self._confusion_matrix[gt_val, pred_val] += np.sum((gt_binary == gt_val) & (pred_binary == pred_val))
        
        # Update per-country confusion matrix
        if country and country in self._country_confusion_matrices:
            for gt_val in [0, 1]:
                for pred_val in [0, 1]:
                    self._country_confusion_matrices[country][gt_val, pred_val] += np.sum(
                        (gt_binary == gt_val) & (pred_binary == pred_val)
                    )

    def evaluate(self):
        """
        Evaluate/summarize the performance
        """
        if self._distributed:
            comm.synchronize()
            
            # Gather metrics from all processes
            print("Gathering metrics from all processes")
            if "pixel" in self._metrics:
                confusion_matrices = comm.gather(self._confusion_matrix, dst=0)
                if self._country_names:
                    country_conf_matrices = comm.gather(self._country_confusion_matrices, dst=0)
            
            if "object" in self._metrics:
                tps = comm.gather(self._total_tps, dst=0)
                fps = comm.gather(self._total_fps, dst=0)
                fns = comm.gather(self._total_fns, dst=0)
                if self._country_names:
                    country_tps = comm.gather(self._country_tps, dst=0)
                    country_fps = comm.gather(self._country_fps, dst=0)
                    country_fns = comm.gather(self._country_fns, dst=0)
            
            if "coco" in self._metrics:
                predictions = comm.gather(self._coco_predictions, dst=0)
                predictions = list(itertools.chain(*predictions))
            
            if not comm.is_main_process():
                return {}
            
            # Combine metrics from all processes
            if "pixel" in self._metrics:
                confusion_matrix = np.zeros_like(self._confusion_matrix)
                for cm in confusion_matrices:
                    confusion_matrix += cm
                
                if self._country_names:
                    country_confusion_matrices = {
                        country: np.zeros_like(self._confusion_matrix) 
                        for country in self._country_names
                    }
                    for cm_dict in country_conf_matrices:
                        for country, cm in cm_dict.items():
                            country_confusion_matrices[country] += cm
            
            if "object" in self._metrics:
                total_tps = sum(tps)
                total_fps = sum(fps)
                total_fns = sum(fns)
                
                if self._country_names:
                    country_total_tps = {country: 0 for country in self._country_names}
                    country_total_fps = {country: 0 for country in self._country_names}
                    country_total_fns = {country: 0 for country in self._country_names}
                    
                    for tps_dict in country_tps:
                        for country, val in tps_dict.items():
                            country_total_tps[country] += val
                    for fps_dict in country_fps:
                        for country, val in fps_dict.items():
                            country_total_fps[country] += val
                    for fns_dict in country_fns:
                        for country, val in fns_dict.items():
                            country_total_fns[country] += val
        else:
            # Single process evaluation
            if "pixel" in self._metrics:
                confusion_matrix = self._confusion_matrix
                if self._country_names:
                    country_confusion_matrices = self._country_confusion_matrices
            
            if "object" in self._metrics:
                total_tps = self._total_tps
                total_fps = self._total_fps
                total_fns = self._total_fns
                if self._country_names:
                    country_total_tps = self._country_tps
                    country_total_fps = self._country_fps
                    country_total_fns = self._country_fns
            
            if "coco" in self._metrics:
                predictions = self._coco_predictions

        # Initialize results dictionary
        results = {}

        # Compute pixel-level metrics
        if "pixel" in self._metrics:
            pixel_results = self._compute_metrics(
                confusion_matrix, total_tps, total_fps, total_fns, prefix=""
            )
            results.update(pixel_results)
            
            # Per-country pixel/object metrics disabled for training-time evaluation (too verbose)
            # Only aggregate metrics are needed for monitoring training progress
            # if self._country_names:
            #     for country in self._country_names:
            #         country_results = self._compute_metrics(
            #             country_confusion_matrices[country],
            #             country_total_tps[country],
            #             country_total_fps[country],
            #             country_total_fns[country],
            #             prefix=f"{country}/"
            #         )
            #         results.update(country_results)

        # COCO metrics
        if "coco" in self._metrics and len(predictions) > 0:
            # predictions are in the format of 
            # {'image_id': 62979, 'category_id': 1, 'segmentation': {'size': [256, 256], 'counts': 'a]`11l75N2M2O1O1O1O2M2O1O4L5K2M2O0O1O2O00O10@TOmIj0e60N2M2O12N4K3N1O1O1O1N2O1O1O1O001O001N101O1N2N2N3MTR4'}, 'bbox': [193.0, 153.0, 46.0, 50.0], 'score': 0.9891765713691711}
            # rather than instances
            coco_results = predictions
            
            # Extract image IDs from predictions to limit ground truth loading
            evaluated_image_ids = set(p.get("image_id") for p in coco_results if p.get("image_id") is not None)
            
            # Also extract file_names from predictions for matching
            evaluated_file_names = set(p.get("file_name") for p in coco_results if p.get("file_name") is not None)
            
            # Debug: Check if we have image_ids and file_names
            if len(evaluated_image_ids) == 0:
                self._logger.warning("No image_ids found in predictions! Using file_names for matching.")
                # Fallback: try to match by file_name
                evaluated_image_ids = None
            else:
                self._logger.debug(f"Evaluating {len(evaluated_image_ids)} unique images by image_id")
            
            # Save predictions if output directory is specified (silently)
            if self._output_dir:
                PathManager.mkdirs(self._output_dir)
                file_path = os.path.join(self._output_dir, "coco_instances_results.json")
                with PathManager.open(file_path, "w") as f:
                    json.dump(coco_results, f)

            # Create custom COCO ground truth with RLE masks, limited to evaluated images only
            # If image_ids don't match, try matching by file_name
            coco_gt = self._create_coco_gt_with_rle(
                selected_image_ids=evaluated_image_ids,
                selected_file_names=evaluated_file_names if evaluated_image_ids is None else None
            )
            
            # Check if we have valid ground truth annotations
            if len(coco_gt.getAnnIds()) == 0:
                self._logger.error("No ground truth annotations found! Cannot compute COCO metrics.")
                self._logger.error(f"Predicted image_ids: {list(evaluated_image_ids)[:10] if evaluated_image_ids else 'None'}")
                self._logger.error(f"Predicted file_names: {list(evaluated_file_names)[:10] if evaluated_file_names else 'None'}")
                # Set all COCO metrics to NaN
                metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
                coco_results = {}
                for metric in metrics:
                    coco_results[f"coco_segm_{metric}"] = float('nan')
                results.update(coco_results)
                return results
            
            # Create a mapping from file_name to image_id for predictions that might have mismatched image_ids
            # This ensures predictions are matched to the correct ground truth images
            file_name_to_gt_image_id = {}
            for img_info in coco_gt.imgs.values():
                file_name_to_gt_image_id[img_info["file_name"]] = img_info["id"]
            
            # Remap prediction image_ids to match ground truth image_ids if needed
            # This handles cases where image_ids don't match but file_names do
            remapped_coco_results = []
            for pred in coco_results:
                pred_file_name = pred.get("file_name")
                pred_image_id = pred.get("image_id")
                
                # If we have a file_name and it matches a GT image, use the GT image_id
                if pred_file_name and pred_file_name in file_name_to_gt_image_id:
                    gt_image_id = file_name_to_gt_image_id[pred_file_name]
                    # Only remap if image_ids don't match
                    if pred_image_id != gt_image_id:
                        self._logger.debug(f"Remapping prediction image_id {pred_image_id} -> {gt_image_id} for file {pred_file_name}")
                        pred = pred.copy()  # Don't modify original
                        pred["image_id"] = gt_image_id
                remapped_coco_results.append(pred)
            
            # Create COCO API for predictions (using remapped image_ids)
            coco_dt = coco_gt.loadRes(remapped_coco_results)

            # Run evaluation for both segmentation and bounding box tasks
            
            # Evaluate segmentation metrics
            segm_eval_results = self._compute_coco_metrics(coco_gt, coco_dt, iou_type="segm")
            
            # Evaluate bounding box metrics
            bbox_eval_results = self._compute_coco_metrics(coco_gt, coco_dt, iou_type="bbox")
            
            # Extract metrics for both tasks
            metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
            coco_results = {}
            
            # Process segmentation results
            if segm_eval_results is not None:
                segm_stats = self._summarize_coco_metrics(segm_eval_results)
                
                # Extract metrics (reduced logging)
                for idx, metric in enumerate(metrics):
                    value = float(segm_stats[idx] * 100 if segm_stats[idx] >= 0 else float('nan'))
                    coco_results[f"coco_segm_{metric}"] = value
                
                # Save detailed segmentation results
                if self._output_dir:
                    segm_detailed_results = self._prepare_detailed_results(segm_eval_results, segm_stats)
                    segm_detailed_path = os.path.join(self._output_dir, "coco_segm_detailed_results.json")
                    with PathManager.open(segm_detailed_path, "w") as f:
                        json.dump(segm_detailed_results, f)
            else:
                self._logger.error("COCO segmentation evaluation failed - no results generated")
                for metric in metrics:
                    coco_results[f"coco_segm_{metric}"] = float('nan')
            
            # Process bounding box results
            if bbox_eval_results is not None:
                bbox_stats = self._summarize_coco_metrics(bbox_eval_results)
                
                # Extract metrics (reduced logging)
                for idx, metric in enumerate(metrics):
                    value = float(bbox_stats[idx] * 100 if bbox_stats[idx] >= 0 else float('nan'))
                    coco_results[f"coco_bbox_{metric}"] = value
                
                # Save detailed bounding box results
                if self._output_dir:
                    bbox_detailed_results = self._prepare_detailed_results(bbox_eval_results, bbox_stats)
                    bbox_detailed_path = os.path.join(self._output_dir, "coco_bbox_detailed_results.json")
                    with PathManager.open(bbox_detailed_path, "w") as f:
                        json.dump(bbox_detailed_results, f)
            else:
                self._logger.error("COCO bounding box evaluation failed - no results generated")
                for metric in metrics:
                    coco_results[f"coco_bbox_{metric}"] = float('nan')
            
            results.update(coco_results)

            # Per-country COCO metrics disabled for training-time evaluation (too verbose)
            # Skip entirely to reduce output - only aggregate metrics are needed for monitoring
            # if self._country_names:
            #     ... (per-country COCO evaluation code) ...

        # Print summary table (suppressed for training-time evaluation to reduce verbosity)
        # Only log aggregate metrics for monitoring training progress
        # self._print_summary_table(results)
        
        # Filter results to only include aggregate metrics for TensorBoard (remove per-country, individual COCO, and counts)
        filtered_results = {}
        for key, value in results.items():
            # Skip per-country metrics (contain '/')
            if '/' in key:
                continue
            # Skip individual COCO metrics (keep only aggregate coco_segm_AP and coco_bbox_AP)
            if key.startswith('coco_'):
                # Only keep main AP metrics, skip AP50, AP75, APs, APm, APl
                if key in ['coco_segm_AP', 'coco_bbox_AP']:
                    filtered_results[key] = value
                continue
            # Skip absolute counts (TP/FP/TN/FN)
            if any(count_key in key for count_key in ['_tp', '_fp', '_tn', '_fn', '_tps', '_fps', '_fns']):
                continue
            # Keep all other metrics (pixel-level, object-level aggregate metrics)
            filtered_results[key] = value
        
        # Save full results (including filtered) for later analysis
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "ftw_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(results, f)  # Save full results

        return filtered_results  # Return filtered results for TensorBoard logging

    def _summarize_coco_metrics(self, eval_results):
        """
        Compute and log summary metrics for COCO evaluation.
        """
        if eval_results is None:
            self._logger.error("No evaluation results to summarize!")
            return np.zeros(12) * np.nan
            
        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = eval_results["params"]
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = eval_results["precision"]
                if s is None:
                    self._logger.error("No precision data in evaluation results!")
                    return float("nan")
                    
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
                
            else:
                # dimension of recall: [TxKxAxM]
                s = eval_results["recall"]
                if s is None:
                    self._logger.error("No recall data in evaluation results!")
                    return float("nan")
                    
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
                
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
                
            # Suppress verbose logging for training-time evaluation
            # self._logger.info(
            #     f" {' ' if ap == 1 else '  '}{'Average Precision' if ap == 1 else 'Average Recall'}"
            #     f"  (AP{'r' if ap == 0 else ''}) "
            #     f"@[ IoU={'0.50:0.95' if iouThr is None else f'{iouThr:0.2f}'} | "
            #     f"area={areaRng:>6s} | "
            #     f"maxDets={maxDets:>3d} ] = "
            #     f"{mean_s:0.3f}"
            # )
            return mean_s
            
        stats = np.zeros((12,))
        stats[0] = _summarize(1)                          # AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        stats[1] = _summarize(1, iouThr=.5)              # AP @[ IoU=0.50      | area=   all | maxDets=100 ]
        stats[2] = _summarize(1, iouThr=.75)             # AP @[ IoU=0.75      | area=   all | maxDets=100 ]
        stats[3] = _summarize(1, areaRng="small")        # AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
        stats[4] = _summarize(1, areaRng="medium")       # AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
        stats[5] = _summarize(1, areaRng="large")        # AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
        stats[6] = _summarize(0, maxDets=1)              # AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
        stats[7] = _summarize(0, maxDets=10)             # AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
        stats[8] = _summarize(0, maxDets=100)            # AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        stats[9] = _summarize(0, areaRng="small")        # AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
        stats[10] = _summarize(0, areaRng="medium")      # AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
        stats[11] = _summarize(0, areaRng="large")       # AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
        
        return stats

    def _visualize_masks(self, image_id, gt_anns, pred_anns, output_dir, img_size=(256, 256)):
        """
        Visualize GT and predicted masks overlaid on RGB input image with transparency.
        GT: semi-transparent green, Pred: semi-transparent red, Overlap: semi-transparent yellow
        Handles 8-channel (stacked) and 4-channel images as in viz_dataset_mapper.py.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to find the corresponding input image
        input_image = None
        for dataset_dict in self._dataset_dicts:
            if dataset_dict.get("image_id") == image_id:
                try:
                    # Read the input image (could be 8, 4, or 3 channels)
                    if dataset_dict.get("file_name", "").endswith(".npz"):
                        data = np.load(dataset_dict["file_name"])
                        image = data['image']
                        if image.shape[0] == 8:
                            rgb = image[:, :, :3]
                        elif image.shape[0] >= 3:
                            rgb = image[:, :, :3]
                        else:
                            rgb = np.tile(image[:, :, 0:1], (1, 1, 3))
                        # normalize to min max
                        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
                        rgb = (rgb * 255).astype(np.uint8)
                        input_image = rgb
                    elif dataset_dict.get("file_name", "").endswith(".tif"):
                        from detectron2.data import detection_utils as utils
                        image = utils.read_geotiff(dataset_dict["file_name"], format=None)
                        # image shape: HWC or CHW? Ensure CHW
                        if image.ndim == 3 and image.shape[0] in [3, 4, 8]:
                            # CHW -> HWC
                            image = np.transpose(image, (1, 2, 0))
                        # Now image is HWC
                        if image.shape[2] == 8:
                            # Stacked: use first 3 channels of window A
                            rgb = image[:, :, :3]
                        elif image.shape[2] >= 3:
                            rgb = image[:, :, :3]
                        else:
                            # Fallback: tile single channel to 3
                            rgb = np.tile(image[:, :, 0:1], (1, 1, 3))
                        # Normalize to 0-255 if needed
                        if rgb.max() <= 1.0:
                            rgb = (rgb * 255).astype(np.uint8)
                        else:
                            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                        input_image = rgb
                    else:
                        from PIL import Image
                        with PathManager.open(dataset_dict["file_name"], "rb") as f:
                            img = np.array(Image.open(f))
                        if img.ndim == 2:
                            input_image = np.stack([img]*3, axis=-1)
                        elif img.shape[2] >= 3:
                            input_image = img[:, :, :3]
                        else:
                            input_image = img
                    break
                except Exception as e:
                    self._logger.warning(f"Could not load input image for {image_id}: {str(e)}")
                    break
        
        # If no input image found, create a black background
        if input_image is None:
            input_image = np.zeros((*img_size, 3), dtype=np.uint8)
        else:
            # Ensure correct size for overlay
            if input_image.shape[:2] != img_size:
                from skimage.transform import resize
                input_image = resize(input_image, img_size, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            if input_image.shape[2] > 3:
                input_image = input_image[:, :, :3]
        
        # Create mask overlays
        gt_mask = np.zeros(img_size, dtype=np.uint8)
        pred_mask = np.zeros(img_size, dtype=np.uint8)
        
        for ann in gt_anns:
            gt_mask |= mask_util.decode(ann['segmentation'])
        for ann in pred_anns:
            pred_mask |= mask_util.decode(ann['segmentation'])
        
        # Create visualization with transparency
        vis = input_image.copy().astype(np.float32)
        
        # Define colors with transparency (alpha = 0.6)
        gt_color = np.array([0, 255, 0])  # Green
        pred_color = np.array([255, 0, 0])  # Red
        overlap_color = np.array([255, 255, 0])  # Yellow
        alpha = 0.6
        
        # Overlay ground truth (green)
        gt_overlay = gt_mask & ~pred_mask  # GT only
        vis[gt_overlay == 1] = vis[gt_overlay == 1] * (1 - alpha) + gt_color * alpha
        
        # Overlay predictions (red)
        pred_overlay = pred_mask & ~gt_mask  # Pred only
        vis[pred_overlay == 1] = vis[pred_overlay == 1] * (1 - alpha) + pred_color * alpha
        
        # Overlay overlap (yellow)
        overlap = gt_mask & pred_mask
        vis[overlap == 1] = vis[overlap == 1] * (1 - alpha) + overlap_color * alpha
        
        # Convert back to uint8
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(vis)
        plt.axis('off')
        plt.title(f"Image ID: {image_id}\nGT: Green, Pred: Red, Overlap: Yellow")
        
        # Create legend
        legend_patches = [
            mpatches.Patch(color='green', alpha=alpha, label='Ground Truth'),
            mpatches.Patch(color='red', alpha=alpha, label='Prediction'),
            mpatches.Patch(color='yellow', alpha=alpha, label='Overlap'),
        ]
        plt.legend(handles=legend_patches, loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{image_id}_vis.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _print_per_image_iou(self, image_id, gt_anns, pred_anns):
        """Print IoU matrix for a single image."""
        if not gt_anns or not pred_anns:
            print(f"Image {image_id}: No GT or predictions.")
            return
        gt_masks = [mask_util.decode(ann['segmentation']) for ann in gt_anns]
        pred_masks = [mask_util.decode(ann['segmentation']) for ann in pred_anns]
        iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
        for i, g in enumerate(gt_masks):
            for j, p in enumerate(pred_masks):
                intersection = np.logical_and(g, p).sum()
                union = np.logical_or(g, p).sum()
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0
        print(f"Image {image_id} IoU matrix:")
        print(iou_matrix)

    def _compute_coco_metrics(self, coco_gt, coco_dt, iou_type):
        """Modified COCO metrics computation with debugging and visualization."""
        logger = logging.getLogger(__name__)
        from pycocotools.coco import COCO
        import copy
        gt_dataset = copy.deepcopy(coco_gt.dataset)
        cat_mapping = {1: 0, 2: 1}
        remapped_cats = []
        for cat in gt_dataset['categories']:
            if cat['id'] in cat_mapping:
                original_id = cat['id']
                cat['id'] = cat_mapping[original_id]
                remapped_cats.append(cat)
        gt_dataset['categories'] = remapped_cats
        for ann in gt_dataset['annotations']:
            if ann['category_id'] in cat_mapping:
                ann['category_id'] = cat_mapping[ann['category_id']]
        coco_gt_remapped = COCO()
        coco_gt_remapped.dataset = gt_dataset
        coco_gt_remapped.createIndex()
        gt_img_ids = coco_gt_remapped.getImgIds()
        dt_img_ids = coco_dt.getImgIds()
        common_img_ids = set(gt_img_ids) & set(dt_img_ids)
        if len(common_img_ids) == 0:
            logger.error("No common image IDs between ground truth and predictions!")
            return None
        # Set custom area bins for small images
        from pycocotools.cocoeval import COCOeval
        coco_eval = COCOeval(coco_gt_remapped, coco_dt, iou_type)
        # coco_eval.params.areaRng = [
        #     [0**2, 256*256],   # all
        #     [0**2, 32**2],     # small
        #     [32**2, 128**2],   # medium
        #     [128**2, 256*256]  # large
        # ]
        # coco_eval.params.areaRngLbl = ["all", "small", "medium", "large"]
        all_common_img_ids = list(common_img_ids)
        coco_eval.params.imgIds = all_common_img_ids
        # Optional visualization (disabled by default; enable by setting self._enable_coco_debug = True)
        if getattr(self, "_enable_coco_debug", False):
            debug_dir = os.path.join(self._output_dir, "debug_visualizations") if self._output_dir else "debug_visualizations"
            import random
            sample_k = min(10, len(all_common_img_ids))
            sample_img_ids = random.sample(all_common_img_ids, sample_k)
            for img_id in sample_img_ids:
                gt_anns = coco_gt_remapped.loadAnns(coco_gt_remapped.getAnnIds(imgIds=[img_id]))
                pred_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[img_id]))
                self._visualize_masks(img_id, gt_anns, pred_anns, debug_dir)
        coco_eval.evaluate()
        coco_eval.accumulate()
        # Suppress verbose output from summarize() for training-time evaluation
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            coco_eval.summarize()
        finally:
            sys.stdout = old_stdout
        return {
            "precision": coco_eval.eval["precision"] if coco_eval.eval else None,
            "recall": coco_eval.eval["recall"] if coco_eval.eval else None,
            "scores": coco_eval.eval["scores"] if coco_eval.eval else None,
            "params": coco_eval.params
        }

    def _compute_metrics(self, confusion_matrix, tps, fps, fns, prefix=""):
        """
        Compute metrics with both per-class and averaged multi-class metrics
        """
        # Extract confusion matrix values
        tn = confusion_matrix[0, 0]  # Background correctly predicted as background
        fp = confusion_matrix[0, 1]  # Background incorrectly predicted as field
        fn = confusion_matrix[1, 0]  # Field incorrectly predicted as background
        tp = confusion_matrix[1, 1]  # Field correctly predicted as field
        
        # Pixel-level metrics
        
        # Overall pixel accuracy
        pixel_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Class IoU metrics
        field_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        background_iou = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
        mean_iou = (field_iou + background_iou) / 2
        
        # Per-class metrics
        field_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        field_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        background_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        background_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Multi-class averaged metrics
        # Treat each class equally in the averaging
        pixel_precision_multiclass = (field_precision + background_precision) / 2
        pixel_recall_multiclass = (field_recall + background_recall) / 2
        
        # Alternative: Weighted by actual class frequency
        total_gt_positives = tp + fn  # Total actual fields
        total_gt_negatives = tn + fp  # Total actual background
        total_pixels = total_gt_positives + total_gt_negatives
        
        if total_pixels > 0:
            field_weight = total_gt_positives / total_pixels
            background_weight = total_gt_negatives / total_pixels
            
            pixel_precision_weighted = (field_precision * field_weight + background_precision * background_weight)
            pixel_recall_weighted = (field_recall * field_weight + background_recall * background_weight)
        else:
            pixel_precision_weighted = 0
            pixel_recall_weighted = 0
        
        # F1 scores
        field_f1 = 2 * field_precision * field_recall / (field_precision + field_recall) if (field_precision + field_recall) > 0 else 0
        multiclass_f1 = 2 * pixel_precision_multiclass * pixel_recall_multiclass / (pixel_precision_multiclass + pixel_recall_multiclass) if (pixel_precision_multiclass + pixel_recall_multiclass) > 0 else 0
        
        # Object-level metrics
        object_precision = tps / (tps + fps) if (tps + fps) > 0 else 0
        object_recall = tps / (tps + fns) if (tps + fns) > 0 else 0
        object_f1 = 2 * object_precision * object_recall / (object_precision + object_recall) if (object_precision + object_recall) > 0 else 0
        
        return {
            # Overall metrics
            f"{prefix}pixel_accuracy": pixel_accuracy * 100,
            f"{prefix}mean_iou": mean_iou * 100,
            
            # Multi-class averaged metrics (what you want)
            f"{prefix}pixel_precision": pixel_precision_multiclass * 100,  # Averaged across both classes
            f"{prefix}pixel_recall": pixel_recall_multiclass * 100,  # Averaged across both classes
            f"{prefix}pixel_f1": multiclass_f1 * 100,
            
            # Weighted multi-class metrics (based on class frequency)
            f"{prefix}pixel_precision_weighted": pixel_precision_weighted * 100,
            f"{prefix}pixel_recall_weighted": pixel_recall_weighted * 100,
            
            # Field class metrics (this is what authors call "ignore presence-only bg")
            f"{prefix}pixel_iou_field": field_iou * 100,
            f"{prefix}pixel_precision_field": field_precision * 100,
            f"{prefix}pixel_recall_field": field_recall * 100,
            f"{prefix}pixel_f1_field": field_f1 * 100,
            
            # Object-level metrics
            f"{prefix}object_precision": object_precision * 100,
            f"{prefix}object_recall": object_recall * 100,
            f"{prefix}object_f1": object_f1 * 100,
            
            # Raw counts for debugging
            f"{prefix}object_tps": tps,
            f"{prefix}object_fps": fps,
            f"{prefix}object_fns": fns,
            f"{prefix}pixel_tp": int(tp),
            f"{prefix}pixel_fp": int(fp),
            f"{prefix}pixel_fn": int(fn),
            f"{prefix}pixel_tn": int(tn),
        }

    def _prepare_detailed_results(self, eval_results, stats):
        """
        Prepare detailed COCO evaluation results for JSON serialization.
        
        Args:
            eval_results: COCO evaluation results
            stats: Summary statistics
            
        Returns:
            Dictionary with detailed results
        """
        # Convert Params object to dictionary for JSON serialization
        params_dict = {
            "iouThrs": eval_results["params"].iouThrs.tolist(),
            "recThrs": eval_results["params"].recThrs.tolist(),
            "maxDets": eval_results["params"].maxDets.tolist() if hasattr(eval_results["params"].maxDets, 'tolist') else list(eval_results["params"].maxDets),
            "areaRng": eval_results["params"].areaRng,
            "areaRngLbl": eval_results["params"].areaRngLbl,
            "useCats": eval_results["params"].useCats,
            "catIds": [int(x) for x in eval_results["params"].catIds] if eval_results["params"].catIds is not None else None,
            "imgIds": [int(x) for x in eval_results["params"].imgIds] if eval_results["params"].imgIds is not None else None,
        }
        
        detailed_results = {
            "stats": stats.tolist(),
            "params": params_dict,
            "precision": eval_results["precision"].tolist() if eval_results["precision"] is not None else None,
            "recall": eval_results["recall"].tolist() if eval_results["recall"] is not None else None,
            "scores": eval_results["scores"].tolist() if eval_results["scores"] is not None else None,
        }
        
        return detailed_results

    def _print_summary_table(self, results):
        """
        Print summary table of evaluation results matching paper format
        """
        # Extract overall metrics
        overall_metrics = {k: v for k, v in results.items() if '/' not in k}
        
        # Create pixel-level metrics table
        pixel_table_data = {
            "Pixel-level IoU (Field)": f"{overall_metrics.get('pixel_iou_field', 0):.2f}",
            "Pixel-level Precision (Field)": f"{overall_metrics.get('pixel_precision_field', 0):.2f}",
            "Pixel-level Recall (Field)": f"{overall_metrics.get('pixel_recall_field', 0):.2f}",
            "Pixel-level F1 (Field)": f"{overall_metrics.get('pixel_f1_field', 0):.2f}",
        }
        
        # Create object-level metrics table
        object_table_data = {
            "Object-level Precision": f"{overall_metrics.get('object_precision', 0):.2f}",
            "Object-level Recall": f"{overall_metrics.get('object_recall', 0):.2f}",
            "Object-level F1": f"{overall_metrics.get('object_f1', 0):.2f}",
            "Object TPs": f"{overall_metrics.get('object_tps', 0)}",
            "Object FPs": f"{overall_metrics.get('object_fps', 0)}",
            "Object FNs": f"{overall_metrics.get('object_fns', 0)}",
        }
        
        # Create COCO metrics table if available
        if any(k.startswith('coco_') for k in overall_metrics):
            # Segmentation metrics
            segm_table_data = {
                "AP": f"{overall_metrics.get('coco_segm_AP', 0):.2f}",
                "AP50": f"{overall_metrics.get('coco_segm_AP50', 0):.2f}",
                "AP75": f"{overall_metrics.get('coco_segm_AP75', 0):.2f}",
                "APs": f"{overall_metrics.get('coco_segm_APs', 0):.2f}",
                "APm": f"{overall_metrics.get('coco_segm_APm', 0):.2f}",
                "APl": f"{overall_metrics.get('coco_segm_APl', 0):.2f}",
            }
            
            # Bounding box metrics
            bbox_table_data = {
                "AP": f"{overall_metrics.get('coco_bbox_AP', 0):.2f}",
                "AP50": f"{overall_metrics.get('coco_bbox_AP50', 0):.2f}",
                "AP75": f"{overall_metrics.get('coco_bbox_AP75', 0):.2f}",
                "APs": f"{overall_metrics.get('coco_bbox_APs', 0):.2f}",
                "APm": f"{overall_metrics.get('coco_bbox_APm', 0):.2f}",
                "APl": f"{overall_metrics.get('coco_bbox_APl', 0):.2f}",
            }
        
        # Log results
        self._logger.info("FTW Evaluation Results (matching paper format):")
        self._logger.info("Pixel-level Metrics:")
        self._logger.info(create_small_table(pixel_table_data))
        self._logger.info("Object-level Metrics:")
        self._logger.info(create_small_table(object_table_data))
        
        if any(k.startswith('coco_') for k in overall_metrics):
            self._logger.info("COCO-style Metrics:")
            self._logger.info("Segmentation Metrics:")
            self._logger.info(create_small_table(segm_table_data))
            self._logger.info("Bounding Box Metrics:")
            self._logger.info(create_small_table(bbox_table_data))
        
        # Per-country results disabled for training-time evaluation (too verbose)
        # Uncomment if needed for detailed analysis
        # if self._country_names:
        #     self._logger.info("\nPer-Country Results:")
        #     country_metrics = defaultdict(dict)
        #     
        #     for key, value in results.items():
        #         if '/' in key:
        #             country, metric = key.split('/', 1)
        #             if metric in ["pixel_iou_field", "pixel_precision_field", "pixel_recall_field", 
        #                         "pixel_f1_field", "object_precision", "object_recall", "object_f1"]:
        #                 country_metrics[country][metric] = f"{value:.2f}"
        #     
        #     for country, metrics in country_metrics.items():
        #         country_table_data = {
        #             "Pixel IoU": metrics.get("pixel_iou_field", "N/A"),
        #             "Pixel Prec": metrics.get("pixel_precision_field", "N/A"),
        #             "Pixel Rec": metrics.get("pixel_recall_field", "N/A"),
        #             "Obj Prec": metrics.get("object_precision", "N/A"),
        #             "Obj Rec": metrics.get("object_recall", "N/A"),
        #             "Obj F1": metrics.get("object_f1", "N/A"),
        #         }
        #         self._logger.info(f"{country}:")
        #         self._logger.info(create_small_table(country_table_data))

    def _get_country_image_ids(self, country_name: str):
        """
        Collect image_ids belonging to a given country based on file_name matching.
        """
        ids = set()
        cname = country_name.lower()
        for dataset_dict in self._dataset_dicts:
            if cname in dataset_dict.get("file_name", "").lower():
                ids.add(dataset_dict["image_id"])
        return ids

    def _load_gt_annotations_with_rle(self, selected_image_ids: set = None, selected_file_names: set = None):
        """
        Load ground truth annotations with RLE masks from panoptic files.
        This is needed because the COCO JSON file doesn't contain RLE masks.
        
        Args:
            selected_image_ids: Optional set of image_ids to filter by
            selected_file_names: Optional set of file_names to filter by (used as fallback if image_ids don't match)
        
        Returns:
            List of ground truth annotations in COCO format with RLE masks
        """
        gt_annotations = []
        processed_images = 0
        skipped_images = 0
        
        for dataset_dict in self._dataset_dicts:
            # Optional filtering by a subset of images
            # Try image_id first, then fallback to file_name
            if selected_image_ids is not None:
                if dataset_dict["image_id"] not in selected_image_ids:
                    # Fallback to file_name matching if image_ids don't match
                    if selected_file_names is None or dataset_dict.get("file_name") not in selected_file_names:
                        continue
            elif selected_file_names is not None:
                if dataset_dict.get("file_name") not in selected_file_names:
                    continue
            image_id = dataset_dict["image_id"]
            file_name = dataset_dict["file_name"]
            
            # Get panoptic segmentation file path
            if "pan_seg_file_name" in dataset_dict:
                pan_seg_file = dataset_dict["pan_seg_file_name"]
                
                # Get the root directory
                dataset_dir = os.path.dirname(self._output_dir) if self._output_dir else ""
                
                # Try direct path first
                if not os.path.isabs(pan_seg_file) and dataset_dir:
                    pan_seg_file = os.path.join(dataset_dir, pan_seg_file)
                
                try:
                    # Read panoptic segmentation
                    from PIL import Image
                    with PathManager.open(pan_seg_file, "rb") as f:
                        pan_seg = np.array(Image.open(f))
                    
                    # Convert RGB to ID if needed
                    if len(pan_seg.shape) == 3:
                        from panopticapi.utils import rgb2id
                        pan_seg = rgb2id(pan_seg)
                    
                    # Get segments info from dataset dict
                    segments_info = dataset_dict.get("segments_info", [])
                    
                    if not segments_info:
                        self._logger.warning(f"No segments_info found for {file_name}")
                        skipped_images += 1
                        continue
                    
                    # Create annotations for each segment
                    image_annotations = 0
                    for segment in segments_info:
                        if segment.get("isthing", True):  # Only process thing classes (ag_field)
                            segment_id = segment["id"]
                            category_id = segment["category_id"]
                            
                            # Create binary mask for this segment
                            mask = (pan_seg == segment_id).astype(np.uint8)
                            
                            if mask.sum() > 0:  # Only include if mask has sufficient area
                                # Convert mask to RLE
                                rle = mask_util.encode(np.asfortranarray(mask))
                                rle["counts"] = rle["counts"].decode("utf-8")
                                
                                # Get bbox from mask
                                bbox = mask_util.toBbox(rle).tolist()
                                
                                # Create annotation
                                annotation = {
                                    "id": len(gt_annotations) + 1,  # Unique annotation ID
                                    "image_id": image_id,
                                    "category_id": category_id,
                                    "segmentation": rle,
                                    "area": float(mask.sum()),
                                    "bbox": bbox,
                                    "iscrowd": 0
                                }
                                
                                gt_annotations.append(annotation)
                                image_annotations += 1
                    
                    if image_annotations > 0:
                        processed_images += 1
                    else:
                        skipped_images += 1
                        self._logger.debug(f"No valid annotations found for {file_name}")
                    
                except Exception as e:
                    self._logger.warning(f"Error loading panoptic file {pan_seg_file}: {str(e)}")
                    skipped_images += 1
                    continue
            else:
                self._logger.warning(f"No pan_seg_file_name found for {file_name}")
                skipped_images += 1
        
        self._logger.info(f"Loaded {len(gt_annotations)} ground truth annotations with RLE masks")
        self._logger.info(f"Processed {processed_images} images, skipped {skipped_images} images")
        return gt_annotations

    def _create_coco_gt_with_rle(self, selected_image_ids: set = None, selected_file_names: set = None):
        """
        Create a COCO ground truth object with RLE masks from panoptic files.
        
        Args:
            selected_image_ids: Optional set of image_ids to filter by
            selected_file_names: Optional set of file_names to filter by (used as fallback if image_ids don't match)
        
        Returns:
            COCO object with ground truth annotations
        """
        # Load ground truth annotations with RLE masks (optionally filtered)
        gt_annotations = self._load_gt_annotations_with_rle(selected_image_ids, selected_file_names)
        
        # Get image information from dataset dicts
        images = []
        for dataset_dict in self._dataset_dicts:
            # Match by image_id first, then fallback to file_name
            if selected_image_ids is not None:
                if dataset_dict["image_id"] not in selected_image_ids:
                    # Fallback to file_name matching if image_ids don't match
                    if selected_file_names is None or dataset_dict.get("file_name") not in selected_file_names:
                        continue
            elif selected_file_names is not None:
                if dataset_dict.get("file_name") not in selected_file_names:
                    continue
            # Try to get actual dimensions from panoptic file
            height, width = 256, 256  # Default values
            if "pan_seg_file_name" in dataset_dict:
                pan_seg_file = dataset_dict["pan_seg_file_name"]
                dataset_dir = os.path.dirname(self._output_dir) if self._output_dir else ""
                
                if not os.path.isabs(pan_seg_file) and dataset_dir:
                    pan_seg_file = os.path.join(dataset_dir, pan_seg_file)
                
                try:
                    from PIL import Image
                    with PathManager.open(pan_seg_file, "rb") as f:
                        pan_seg = np.array(Image.open(f))
                    height, width = pan_seg.shape[:2]  # Get actual dimensions
                except Exception as e:
                    self._logger.warning(f"Could not get dimensions from {pan_seg_file}: {str(e)}")
            
            image_info = {
                "id": dataset_dict["image_id"],
                "file_name": dataset_dict["file_name"],
                "height": height,
                "width": width,
            }
            images.append(image_info)
        
        # Create categories (same as in metadata.py)
        categories = [
            {
                "id": 1,
                "name": "ag_field",
                "supercategory": "landcover",
                "isthing": 1
            },
            {
                "id": 2,
                "name": "background", 
                "supercategory": "background",
                "isthing": 0
            }
        ]
        
        # Create COCO dataset structure
        coco_dataset = {
            "info": {
                "description": "Fields of The World Dataset",
                "version": "1.0",
                "year": 2024
            },
            "licenses": [],
            "images": images,
            "annotations": gt_annotations,
            "categories": categories
        }
        
        # Create COCO object
        coco_gt = COCO()
        coco_gt.dataset = coco_dataset
        coco_gt.createIndex()
        
        self._logger.info(f"Created COCO ground truth with {len(images)} images and {len(gt_annotations)} annotations")
        return coco_gt