"""Intermediate output formats: Model Output → SemanticOutput/InstanceOutput/PanopticOutput → Detections."""

import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class SemanticOutput:
    """Per-pixel class probabilities from semantic segmentation models (UNet, DeepLabV3+, DECODE, etc.)."""

    logits: np.ndarray  # (num_classes, H, W)
    auxiliary: Optional[Dict[str, np.ndarray]] = None
    image_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.logits.ndim != 3:
            raise ValueError(f"logits must be 3D (num_classes, H, W), got shape {self.logits.shape}")

    @property
    def num_classes(self) -> int:
        return self.logits.shape[0]

    @property
    def shape(self) -> tuple:
        return self.logits.shape[1:]

    def get_field_mask(self, field_class_id: int = 1, threshold: float = 0.5) -> np.ndarray:
        """Binary field mask from logits: 1=field, 0=background."""
        if field_class_id >= self.num_classes:
            raise ValueError(f"field_class_id {field_class_id} >= num_classes {self.num_classes}")
        return (self.logits[field_class_id] > threshold).astype(np.uint8)

    def get_class_mask(self) -> np.ndarray:
        """Argmax class mask of shape (H, W)."""
        return np.argmax(self.logits, axis=0).astype(np.uint8)

    def to_detections(self, field_class_id: int = 1, min_area: int = 0):
        from .detections import Detections
        return Detections.from_semantic_logits(self, field_class_id=field_class_id, min_area=min_area)


@dataclass
class InstanceOutput:
    """Per-instance binary masks with scores from instance segmentation models (SAM, DA, Mask2Former)."""

    masks: np.ndarray  # (N, H, W)
    scores: np.ndarray  # (N,)
    class_ids: Optional[np.ndarray] = None  # (N,)
    image_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if isinstance(self.masks, list):
            self.masks = np.stack(self.masks, axis=0) if self.masks else np.zeros((0, 256, 256), dtype=np.uint8)

        if self.masks.ndim != 3:
            raise ValueError(f"masks must be 3D (N, H, W), got shape {self.masks.shape}")

        n = self.masks.shape[0]
        if self.scores.shape != (n,):
            raise ValueError(f"scores shape {self.scores.shape} != ({n},)")
        if self.class_ids is not None and self.class_ids.shape != (n,):
            raise ValueError(f"class_ids shape {self.class_ids.shape} != ({n},)")

        if self.masks.dtype != np.uint8 or not np.all(np.isin(self.masks, [0, 1])):
            self.masks = (self.masks > 0).astype(np.uint8)

    @property
    def num_instances(self) -> int:
        return self.masks.shape[0]

    @property
    def shape(self) -> tuple:
        return self.masks.shape[1:]

    def filter(self, score_threshold: float = 0.0, min_area: int = 0) -> "InstanceOutput":
        """Filter instances by score and area."""
        if self.num_instances == 0:
            return self

        keep = self.scores >= score_threshold
        if min_area > 0:
            keep &= np.sum(self.masks, axis=(1, 2)) >= min_area

        if not np.any(keep):
            return InstanceOutput(
                masks=np.zeros((0, *self.shape), dtype=np.uint8),
                scores=np.array([]),
                class_ids=np.array([]) if self.class_ids is not None else None,
                image_id=self.image_id,
                metadata=self.metadata,
            )

        return InstanceOutput(
            masks=self.masks[keep],
            scores=self.scores[keep],
            class_ids=self.class_ids[keep] if self.class_ids is not None else None,
            image_id=self.image_id,
            metadata=self.metadata,
        )

    def to_detections(self, min_area: int = 0, score_threshold: float = 0.0):
        from .detections import Detections
        return Detections.from_instance_masks(self, min_area=min_area, score_threshold=score_threshold)


@dataclass
class PanopticOutput:
    """Panoptic segmentation output (stuff + things) from Mask2Former/OneFormer."""

    seg_map: np.ndarray  # (H, W), each pixel = segment ID
    segments_info: List[Dict[str, Any]]
    image_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.seg_map.ndim != 2:
            raise ValueError(f"seg_map must be 2D (H, W), got shape {self.seg_map.shape}")

        seg_ids_in_info = {seg["id"] for seg in self.segments_info}
        seg_ids_in_map = set(np.unique(self.seg_map)) - {0}
        extra = seg_ids_in_info - seg_ids_in_map - {0}
        if extra:
            print(f"Warning: segments_info contains IDs not in seg_map: {extra}")

    @property
    def shape(self) -> tuple:
        return self.seg_map.shape

    @property
    def num_segments(self) -> int:
        return len(self.segments_info)

    def get_things(self) -> List[Dict[str, Any]]:
        return [seg for seg in self.segments_info if seg.get("isthing", False)]

    def get_stuff(self) -> List[Dict[str, Any]]:
        return [seg for seg in self.segments_info if not seg.get("isthing", False)]

    def to_instance_masks(self, field_category_id: Optional[int] = None) -> InstanceOutput:
        """Extract instance masks for thing segments."""
        things = self.get_things()
        if field_category_id is not None:
            things = [s for s in things if s["category_id"] == field_category_id]

        if not things:
            return InstanceOutput(
                masks=np.zeros((0, *self.shape), dtype=np.uint8),
                scores=np.array([]),
                class_ids=np.array([]),
                image_id=self.image_id,
                metadata=self.metadata,
            )

        masks = [(self.seg_map == seg["id"]).astype(np.uint8) for seg in things]
        scores = [seg.get("score", 1.0) for seg in things]
        class_ids = [seg["category_id"] for seg in things]

        return InstanceOutput(
            masks=np.stack(masks),
            scores=np.array(scores),
            class_ids=np.array(class_ids),
            image_id=self.image_id,
            metadata=self.metadata,
        )

    def to_binary_mask(self, field_category_id: int = 0) -> np.ndarray:
        """Binary mask where 1=field (thing segments with matching category), 0=background."""
        binary = np.zeros_like(self.seg_map, dtype=np.uint8)
        for seg in self.segments_info:
            if seg.get("isthing", True) and seg.get("category_id", 0) == field_category_id:
                binary[self.seg_map == seg["id"]] = 1
        return binary

    def to_detections(self, min_area: int = 0, include_stuff: bool = False):
        from .detections import Detections
        return Detections.from_panoptic_output(self, min_area=min_area)
