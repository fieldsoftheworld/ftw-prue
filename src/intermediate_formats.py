"""
Intermediate representation formats for model outputs.

This module provides standardized intermediate representations that sit between
raw model outputs and the unified Detections format. This allows model-specific
conversion logic to be separated from evaluation logic.

Workflow:
    Model Output → Intermediate Format → Detections → Evaluator
"""

import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class SemanticOutput:
    """
    Intermediate representation for semantic segmentation model outputs.
    
    Used by models that produce per-pixel class probabilities/logits.
    Models: Baseline (UNet, DeepLabV3+, etc.), Galileo, Clay, SatMAE, Croma, DECODE
    
    Attributes:
        logits: Class probabilities/logits of shape (num_classes, H, W)
        auxiliary: Optional auxiliary outputs (e.g., DECODE's boundary/distance predictions)
        image_id: Optional image identifier for tracking
        metadata: Optional metadata (e.g., original image shape, class names)
    """
    logits: np.ndarray  # Shape: (num_classes, H, W)
    auxiliary: Optional[Dict[str, np.ndarray]] = None
    image_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the logits shape."""
        if self.logits.ndim != 3:
            raise ValueError(f"logits must be 3D (num_classes, H, W), got shape {self.logits.shape}")
        if self.logits.shape[0] < 1:
            raise ValueError(f"logits must have at least 1 class, got {self.logits.shape[0]}")
    
    @property
    def num_classes(self) -> int:
        """Number of classes in the segmentation."""
        return self.logits.shape[0]
    
    @property
    def shape(self) -> tuple:
        """Spatial shape (H, W) of the segmentation."""
        return self.logits.shape[1:]
    
    def get_field_mask(self, field_class_id: int = 1, threshold: float = 0.5) -> np.ndarray:
        """
        Extract binary field mask from logits.
        
        Args:
            field_class_id: Class ID for field class (default: 1)
            threshold: Probability threshold for binarization (default: 0.5)
            
        Returns:
            Binary mask of shape (H, W) where 1=field, 0=background
        """
        if field_class_id >= self.num_classes:
            raise ValueError(f"field_class_id {field_class_id} >= num_classes {self.num_classes}")
        
        # For 2-class: just threshold the field class
        # For 3-class: threshold the field class (class 2 is boundary)
        field_probs = self.logits[field_class_id]
        binary_mask = (field_probs > threshold).astype(np.uint8)
        
        return binary_mask
    
    def get_class_mask(self) -> np.ndarray:
        """
        Get the predicted class for each pixel via argmax.
        
        Returns:
            Class mask of shape (H, W) with values in [0, num_classes-1]
        """
        return np.argmax(self.logits, axis=0).astype(np.uint8)
    
    def to_detections(self, field_class_id: int = 1, min_area: int = 0):
        """
        Convert to Detections format.
        
        Args:
            field_class_id: Class ID for field class (default: 1)
            min_area: Minimum area threshold for filtering detections
            
        Returns:
            Detections object
        """
        from detections import Detections
        return Detections.from_semantic_logits(self, field_class_id=field_class_id, min_area=min_area)


@dataclass
class InstanceOutput:
    """
    Intermediate representation for instance segmentation model outputs.
    
    Used by models that produce per-instance binary masks with scores.
    Models: SAM, Delineate Anything, Mask2Former/OneFormer/MaskDINO (instance mode)
    
    Attributes:
        masks: Binary instance masks of shape (N, H, W) or list of (H, W) arrays
        scores: Confidence scores of shape (N,)
        class_ids: Optional class labels of shape (N,). If None, assumes single class.
        image_id: Optional image identifier for tracking
        metadata: Optional metadata (e.g., original image shape, model info)
    """
    masks: np.ndarray  # Shape: (N, H, W) or List[np.ndarray]
    scores: np.ndarray  # Shape: (N,)
    class_ids: Optional[np.ndarray] = None  # Shape: (N,)
    image_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and normalize the masks."""
        # Convert list to array if needed
        if isinstance(self.masks, list):
            if len(self.masks) == 0:
                self.masks = np.zeros((0, 256, 256), dtype=np.uint8)
            else:
                self.masks = np.stack(self.masks, axis=0)
        
        # Validate shapes
        if self.masks.ndim != 3:
            raise ValueError(f"masks must be 3D (N, H, W), got shape {self.masks.shape}")
        
        num_instances = self.masks.shape[0]
        
        if self.scores.shape != (num_instances,):
            raise ValueError(f"scores shape {self.scores.shape} doesn't match num instances {num_instances}")
        
        if self.class_ids is not None and self.class_ids.shape != (num_instances,):
            raise ValueError(f"class_ids shape {self.class_ids.shape} doesn't match num instances {num_instances}")
        
        # Ensure binary masks
        if self.masks.dtype != np.uint8 or not np.all(np.isin(self.masks, [0, 1])):
            self.masks = (self.masks > 0).astype(np.uint8)
    
    @property
    def num_instances(self) -> int:
        """Number of instance masks."""
        return self.masks.shape[0]
    
    @property
    def shape(self) -> tuple:
        """Spatial shape (H, W) of the masks."""
        return self.masks.shape[1:]
    
    def filter(self, score_threshold: float = 0.0, min_area: int = 0) -> 'InstanceMasks':
        """
        Filter instances by score and area.
        
        Args:
            score_threshold: Minimum confidence score
            min_area: Minimum mask area in pixels
            
        Returns:
            Filtered InstanceMasks object
        """
        if self.num_instances == 0:
            return self
        
        # Score filtering
        keep_score = self.scores >= score_threshold
        
        # Area filtering
        if min_area > 0:
            areas = np.sum(self.masks, axis=(1, 2))
            keep_area = areas >= min_area
            keep = keep_score & keep_area
        else:
            keep = keep_score
        
        if not np.any(keep):
            # Return empty instance
            return InstanceOutput(
                masks=np.zeros((0, *self.shape), dtype=np.uint8),
                scores=np.array([]),
                class_ids=np.array([]) if self.class_ids is not None else None,
                image_id=self.image_id,
                metadata=self.metadata
            )
        
        return InstanceOutput(
            masks=self.masks[keep],
            scores=self.scores[keep],
            class_ids=self.class_ids[keep] if self.class_ids is not None else None,
            image_id=self.image_id,
            metadata=self.metadata
        )
    
    def to_detections(self, min_area: int = 0, score_threshold: float = 0.0):
        """
        Convert to Detections format.
        
        Args:
            min_area: Minimum area threshold for instances (in pixels)
            score_threshold: Score threshold for filtering instances
        
        Returns:
            Detections object
        """
        from detections import Detections
        return Detections.from_instance_masks(self, min_area=min_area, score_threshold=score_threshold)


@dataclass
class PanopticOutput:
    """
    Intermediate representation for panoptic segmentation model outputs.
    
    Used by models that produce panoptic segmentation (stuff + things).
    Models: Mask2Former/OneFormer/MaskDINO (panoptic mode)
    
    Attributes:
        seg_map: Segmentation map of shape (H, W) where each pixel value is a segment ID
        segments_info: List of segment metadata dicts with keys:
            - 'id': segment ID
            - 'category_id': class/category ID
            - 'isthing': whether segment is a thing (instance) or stuff (semantic)
            - 'score': optional confidence score
            - 'area': optional area in pixels
        image_id: Optional image identifier for tracking
        metadata: Optional metadata (e.g., category names, model info)
    """
    seg_map: np.ndarray  # Shape: (H, W), dtype: int
    segments_info: List[Dict[str, Any]]
    image_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the segmentation map and segments info."""
        if self.seg_map.ndim != 2:
            raise ValueError(f"seg_map must be 2D (H, W), got shape {self.seg_map.shape}")
        
        # Verify all segment IDs in segments_info exist in seg_map
        seg_ids_in_info = {seg['id'] for seg in self.segments_info}
        seg_ids_in_map = set(np.unique(self.seg_map))
        seg_ids_in_map.discard(0)  # 0 is typically background/no segment
        
        # It's okay if seg_map has more IDs (e.g., filtered segments), but segments_info should be subset
        if not seg_ids_in_info.issubset(seg_ids_in_map | {0}):
            extra_ids = seg_ids_in_info - seg_ids_in_map - {0}
            if extra_ids:
                print(f"Warning: segments_info contains IDs not in seg_map: {extra_ids}")
    
    @property
    def shape(self) -> tuple:
        """Spatial shape (H, W) of the segmentation."""
        return self.seg_map.shape
    
    @property
    def num_segments(self) -> int:
        """Total number of segments (things + stuff)."""
        return len(self.segments_info)
    
    def get_things(self) -> List[Dict[str, Any]]:
        """Get only the 'thing' (instance) segments."""
        return [seg for seg in self.segments_info if seg.get('isthing', False)]
    
    def get_stuff(self) -> List[Dict[str, Any]]:
        """Get only the 'stuff' (semantic) segments."""
        return [seg for seg in self.segments_info if not seg.get('isthing', False)]
    
    def to_instance_masks(self, field_category_id: Optional[int] = None) -> InstanceOutput:
        """
        Extract instance masks from panoptic output.
        
        Args:
            field_category_id: If provided, only extract instances of this category.
                              If None, extract all 'thing' instances.
        
        Returns:
            InstanceMasks object containing only instance masks
        """
        # Get thing segments
        thing_segments = self.get_things()
        
        # Filter by category if specified
        if field_category_id is not None:
            thing_segments = [s for s in thing_segments if s['category_id'] == field_category_id]
        
        if len(thing_segments) == 0:
            # Return empty instance masks
            return InstanceOutput(
                masks=np.zeros((0, *self.shape), dtype=np.uint8),
                scores=np.array([]),
                class_ids=np.array([]),
                image_id=self.image_id,
                metadata=self.metadata
            )
        
        # Extract masks for each thing segment
        masks = []
        scores = []
        class_ids = []
        
        for seg in thing_segments:
            seg_id = seg['id']
            mask = (self.seg_map == seg_id).astype(np.uint8)
            masks.append(mask)
            scores.append(seg.get('score', 1.0))
            class_ids.append(seg['category_id'])
        
        return InstanceOutput(
            masks=np.stack(masks, axis=0),
            scores=np.array(scores),
            class_ids=np.array(class_ids),
            image_id=self.image_id,
            metadata=self.metadata
        )
    
    def to_binary_mask(self, field_category_id: int = 0) -> np.ndarray:
        """
        Convert panoptic output directly to binary mask (for pixel-level metrics).
        This matches the M2F evaluator behavior.
        
        Args:
            field_category_id: Category ID for field class (default: 0)
        
        Returns:
            Binary mask where 1 = field, 0 = background
        """
        binary_mask = np.zeros_like(self.seg_map, dtype=np.uint8)
        
        # Include all thing instances with matching category_id
        # This matches M2F evaluator's _panoptic_to_binary method
        for segment in self.segments_info:
            if segment.get("isthing", True) and segment.get("category_id", 0) == field_category_id:
                binary_mask[self.seg_map == segment["id"]] = 1
        
        return binary_mask
    
    def to_detections(self, min_area: int = 0, include_stuff: bool = False):
        """
        Convert to Detections format (things only; stuff is ignored for instance eval).
        
        Args:
            min_area: Minimum area threshold for instances
            include_stuff: Ignored; Detections uses thing segments only (matches ag-seg/COCO instance eval).
        
        Returns:
            Detections object
        """
        from detections import Detections
        return Detections.from_panoptic_output(self, min_area=min_area)

