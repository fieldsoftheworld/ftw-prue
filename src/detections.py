from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import warnings
from shapely.geometry import Polygon
import shapely

from intermediate_formats import SemanticOutput, InstanceOutput, PanopticOutput

@dataclass
class Detections:
    """
    A dataclass to store and handle detections from various models.
    Supports both mask-based and polygon-based formats for comprehensive evaluation.
    """
    xyxy: np.ndarray
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
    polygons: Optional[List[Polygon]] = None  # For object-level metrics
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], Optional[float], Optional[int], Optional[int], Optional[Polygon]]]:
        """
        Iterates over the Detections object and yields a tuple of
        (xyxy, mask, confidence, class_id, tracker_id, polygon) for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.mask[i] if self.mask is not None else None,
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
                self.polygons[i] if self.polygons is not None else None,
            )
    
    @classmethod
    def from_semantic_logits(cls, semantic_logits: SemanticOutput, field_class_id: int = 1, min_area: int = 0) -> Detections:
        """
        Creates a Detections instance from SemanticOutput.
        
        Uses rasterio.features.shapes() to match the paper's polygonization approach exactly.
        
        Args:
            semantic_logits: SemanticOutput object containing model outputs
            field_class_id: Class ID for field class (default: 1)
            min_area: Minimum area threshold for instances (in pixels)
            
        Returns:
            Detections object with field instances extracted from semantic segmentation
        """
        import rasterio.features
        import shapely.geometry
        
        # Get field mask from semantic logits
        field_mask = semantic_logits.get_field_mask(field_class_id=field_class_id)
        
        # Ensure mask is uint8 for rasterio (required for rasterio.features.shapes)
        field_mask = field_mask.astype(np.uint8)
        
        masks = []
        xyxys = []
        confidences = []
        class_ids = []
        
        # Extract shapes using rasterio (matches paper's approach exactly)
        for geom, val in rasterio.features.shapes(field_mask):
            if val == 1:  # Only process field pixels
                shapely_geom = shapely.geometry.shape(geom)
                
                # Skip small areas
                if shapely_geom.area < min_area:
                    continue
                
                # Create mask for this shape
                mask = rasterio.features.rasterize(
                    [shapely_geom], 
                    out_shape=field_mask.shape,
                    fill=0,
                    default_value=1,
                    dtype=np.uint8
                )
                masks.append(mask)
                
                # Get bounding box from shapely geometry
                bounds = shapely_geom.bounds
                xyxys.append([bounds[0], bounds[1], bounds[2], bounds[3]])
                
                # Get confidence from semantic logits (mean probability of field class in this instance)
                field_probs = semantic_logits.logits[field_class_id]
                mask_confidence = np.mean(field_probs[mask == 1])
                confidences.append(mask_confidence)
                
                # Use provided field class ID
                class_ids.append(field_class_id)
        
        return cls(
            xyxy=np.array(xyxys) if xyxys else np.empty((0, 4)),
            mask=np.array(masks) if masks else None,
            confidence=np.array(confidences) if confidences else None,
            class_id=np.array(class_ids) if class_ids else None
        )

    @classmethod
    def from_instance_masks(cls, instance_masks: InstanceOutput, min_area: int = 0, score_threshold: float = 0.0) -> Detections:
        """
        Creates a Detections instance from InstanceOutput.
        
        Args:
            instance_masks: InstanceOutput object containing model outputs
            min_area: Minimum area threshold for instances
            score_threshold: Score threshold for filtering instances
            
        Returns:
            Detections object with instance detections
        """
        import cv2
        
        # Filter by score if threshold provided
        if score_threshold > 0:
            instance_masks = instance_masks.filter(score_threshold=score_threshold, min_area=min_area)
        
        if instance_masks.num_instances == 0:
            return cls(xyxy=np.empty((0, 4)))
        
        masks = []
        xyxys = []
        confidences = []
        class_ids = []
        
        for i in range(instance_masks.num_instances):
            mask = instance_masks.masks[i]
            
            # Ensure mask is binary
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8)
            
            # Skip if mask is too small
            if np.sum(mask) < min_area:
                continue
            
            masks.append(mask)
            
            # Compute bounding box
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            xyxys.append([x_min, y_min, x_max, y_max])
            
            # Get score and class
            confidences.append(float(instance_masks.scores[i]))
            
            if instance_masks.class_ids is not None:
                class_ids.append(int(instance_masks.class_ids[i]))
            else:
                class_ids.append(0)
        
        return cls(
            xyxy=np.array(xyxys) if xyxys else np.empty((0, 4)),
            mask=np.array(masks) if masks else None,
            confidence=np.array(confidences) if confidences else None,
            class_id=np.array(class_ids) if class_ids else None
        )

    @classmethod
    def from_panoptic_output(cls, panoptic_output: PanopticOutput, min_area: int = 0, include_stuff: bool = False) -> Detections:
        """
        Creates a Detections instance from PanopticOutput.
        
        Args:
            panoptic_output: PanopticOutput object containing model outputs
            min_area: Minimum area threshold for instances
            include_stuff: Whether to include "stuff" classes (default: False, only "things")
            
        Returns:
            Detections object with instance detections
        """
        # Extract thing instances first
        thing_instances = panoptic_output.to_instance_masks()
        
        # Optionally include stuff segments that correspond to the field class (category_id==1)
        if include_stuff:
            masks = []
            scores = []
            class_ids = []
            if thing_instances.num_instances > 0:
                masks.append(thing_instances.masks)
                scores.append(thing_instances.scores)
                class_ids.append(thing_instances.class_ids if thing_instances.class_ids is not None else np.zeros(thing_instances.num_instances, dtype=int))
            # Add stuff segments with category_id==1
            for seg in panoptic_output.segments_info:
                if not seg.get('isthing', False) and int(seg.get('category_id', -1)) == 1:
                    seg_id = int(seg['id'])
                    mask = (panoptic_output.seg_map == seg_id).astype(np.uint8)
                    masks.append(mask[None, ...])
                    # Warn if score is missing and default to 1.0
                    _score = seg.get('score', None)
                    if _score is None:
                        try:
                            import warnings
                            warnings.warn(
                                f"Panoptic segment missing 'score'; defaulting to 1.0 (image_id={panoptic_output.image_id}, seg_id={seg_id})"
                            )
                        except Exception:
                            pass
                        _score = 1.0
                    scores.append(np.array([float(_score)]))
                    class_ids.append(np.array([1]))
            if masks:
                masks = np.concatenate(masks, axis=0)
                scores = np.concatenate(scores, axis=0)
                class_ids = np.concatenate(class_ids, axis=0)
                all_instances = InstanceOutput(masks=masks, scores=scores, class_ids=class_ids)
            else:
                all_instances = thing_instances
            return cls.from_instance_masks(all_instances, min_area=min_area)
        
        # Things only
        return cls.from_instance_masks(thing_instances, min_area=min_area)


    @classmethod
    def from_gt(cls, instance_mask: np.ndarray, min_area: int = 0) -> Detections:
        """
        Creates a Detections instance from ground truth instance masks.
        This preserves individual field instances without fusion.
        
        Args:
            instance_mask: Instance mask where each unique value represents a field instance
            min_area: Minimum area threshold for instances
            
        Returns:
            Detections object with individual field instances
        """
        import cv2
        import rasterio.features
        from shapely.geometry import shape
        
        # Get unique instance IDs (excluding 0 which is background)
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids > 0]
        
        masks = []
        xyxys = []
        confidences = []
        class_ids = []
        polygons = []
        
        for instance_id in instance_ids:
            # Create binary mask for this instance
            instance_binary = (instance_mask == instance_id).astype(np.uint8)
            
            # Check minimum area
            area = np.sum(instance_binary)
            if area < min_area:
                continue
            
            masks.append(instance_binary)
            
            # Compute bounding box
            rows = np.any(instance_binary, axis=1)
            cols = np.any(instance_binary, axis=0)
            if not rows.any() or not cols.any():
                continue
                
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            xyxys.append([x_min, y_min, x_max, y_max])
            
            # Use area as confidence (normalized)
            confidences.append(min(area / 1000.0, 1.0))
            class_ids.append(0)  # 0 for ag_field
            
            # Extract polygon from mask for object-level metrics
            try:
                # Use rasterio.features.shapes to get polygon
                shapes = list(rasterio.features.shapes(instance_binary, mask=instance_binary))
                if shapes:
                    # Get the largest polygon (should be the only one for a single instance)
                    largest_shape = max(shapes, key=lambda x: shape(x[0]).area)
                    polygon = shape(largest_shape[0])
                    polygons.append(polygon)
                else:
                    polygons.append(None)
            except Exception as e:
                print(f"Warning: Could not extract polygon for instance {instance_id}: {str(e)}")
                polygons.append(None)
        
        return cls(
            xyxy=np.array(xyxys) if xyxys else np.empty((0, 4)),
            mask=np.array(masks) if masks else None,
            confidence=np.array(confidences) if confidences else None,
            class_id=np.array(class_ids) if class_ids else None,
            polygons=polygons if polygons else None
        )

    def to_coco_format(self, image_id: int, start_annotation_id: int = 1) -> List[Dict]:
        """
        Convert Detections to COCO format with RLE masks.
        
        Args:
            image_id: Image ID for COCO format
            start_annotation_id: Starting annotation ID (for unique IDs across images)
            
        Returns:
            List of COCO-style annotations with RLE masks
        """
        import pycocotools.mask as mask_util
        
        if self.mask is None or len(self.mask) == 0:
            return []
        
        coco_annotations = []
        
        for i, mask in enumerate(self.mask):
            if np.sum(mask) < 10:  # Skip very small masks
                continue
                
            # Convert mask to RLE
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            
            # Get bounding box from mask
            bbox = mask_util.toBbox(rle).tolist()
            
            # Get confidence and class_id
            confidence = float(self.confidence[i]) if self.confidence is not None else 1.0
            class_id = int(self.class_id[i]) if self.class_id is not None else 0
            
            annotation = {
                "id": start_annotation_id + len(coco_annotations),  # Unique annotation ID
                "image_id": image_id,
                "category_id": class_id,
                "segmentation": rle,
                "bbox": bbox,
                "area": float(np.sum(mask)),
                "score": confidence,
                "iscrowd": 0
            }
            
            coco_annotations.append(annotation)
        
        return coco_annotations
    
    def to_binary_mask(self, shape: Tuple[int, int], field_class_id: Optional[int] = None) -> np.ndarray:
        """
        Convert all masks to a single binary mask.
        
        Args:
            shape: (height, width) of the output mask
            field_class_id: If provided, only include masks whose class_id matches this value
            
        Returns:
            Binary mask where 1 indicates any detection, 0 indicates background
        """
        import os
        debug_count = int(os.environ.get('DEBUG_DETECTIONS_COUNT', '0'))
        
        if self.mask is None or len(self.mask) == 0:
            return np.zeros(shape, dtype=np.uint8)

        binary_mask = np.zeros(shape, dtype=np.uint8)
        
        for i, mask in enumerate(self.mask):
            # If filtering by class id, skip non-matching masks
            if field_class_id is not None and self.class_id is not None and len(self.class_id) > i:
                if self.class_id[i] != field_class_id:
                    continue
            
            # Ensure mask is same shape as target
            if mask.shape != shape:
                # Resize mask to target shape
                from skimage.transform import resize
                mask = resize(mask, shape, preserve_range=True, anti_aliasing=True)
                mask = (mask > 0.5).astype(np.uint8)
            
            binary_mask = np.logical_or(binary_mask, mask > 0)
        
        return binary_mask.astype(np.uint8)
    
    def compute_polygons(self, min_area: int = 0) -> List[Polygon]:
        """
        Compute polygons from masks for object-level metrics.
        
        Args:
            min_area: Minimum area threshold for polygons
            
        Returns:
            List of shapely Polygon objects
        """
        import rasterio.features
        
        if self.polygons is not None:
            # Return pre-computed polygons, filtering by area
            # Suppress warnings when checking area
            result = []
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                for p in self.polygons:
                    if p is not None and not p.is_empty:
                        try:
                            area = p.area
                            if np.isfinite(area) and area >= min_area:
                                result.append(p)
                        except (RuntimeWarning, ValueError, RuntimeError):
                            continue
            return result
        
        polygons = []
        # Suppress warnings during polygon extraction and area calculation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
            if self.mask is not None:
                for mask in self.mask:
                    if mask.sum() > 0:
                        # Extract polygons from mask using rasterio (match ftw_tools behavior)
                        import shapely.geometry
                        for geom, val in rasterio.features.shapes(mask.astype(np.uint8)):
                            if val == 1:  # Only extract shapes with value 1 (match ftw_tools)
                                try:
                                    shapely_geom = shapely.geometry.shape(geom)
                                    # Make valid if needed
                                    if not shapely_geom.is_valid:
                                        shapely_geom = shapely.make_valid(shapely_geom)
                                        # Extract first Polygon if it's a collection
                                        if hasattr(shapely_geom, 'geoms'):
                                            shapely_geom = next((g for g in shapely_geom.geoms if isinstance(g, Polygon)), None)
                                            if shapely_geom is None:
                                                continue
                                    if shapely_geom is not None and not shapely_geom.is_empty and shapely_geom.is_valid:
                                        area = shapely_geom.area
                                        if np.isfinite(area) and area >= min_area:
                                            polygons.append(shapely_geom)
                                except (RuntimeWarning, ValueError, RuntimeError, Exception):
                                    continue
            else:
                # Fall back to bounding boxes as rectangles
                for bbox in self.xyxy:
                    try:
                        x_min, y_min, x_max, y_max = bbox
                        import shapely.geometry
                        polygon = shapely.geometry.box(x_min, y_min, x_max, y_max)
                        if polygon.is_valid and not polygon.is_empty:
                            area = polygon.area
                            if np.isfinite(area) and area >= min_area:
                                polygons.append(polygon)
                    except (RuntimeWarning, ValueError, RuntimeError, Exception):
                        continue
        
        return polygons
