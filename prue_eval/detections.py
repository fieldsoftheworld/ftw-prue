from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import warnings
from shapely.geometry import Polygon
import shapely

from .intermediate_formats import SemanticOutput, InstanceOutput, PanopticOutput


@dataclass
class Detections:
    """Stores detections (mask-based or polygon-based) from any model backend."""

    xyxy: np.ndarray
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    polygons: Optional[List[Polygon]] = None
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)
    image_filename: Optional[str] = None

    def __len__(self):
        return len(self.xyxy)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], Optional[float], Optional[int], Optional[Polygon]]]:
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.mask[i] if self.mask is not None else None,
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.polygons[i] if self.polygons is not None else None,
            )

    @classmethod
    def from_semantic_logits(cls, semantic_logits: SemanticOutput, field_class_id: int = 1, min_area: int = 0) -> Detections:
        """Extract field instances from semantic segmentation via connected components."""
        import rasterio.features
        import shapely.geometry

        field_mask = semantic_logits.get_field_mask(field_class_id=field_class_id).astype(np.uint8)

        masks, xyxys, confidences, class_ids = [], [], [], []
        for geom, val in rasterio.features.shapes(field_mask):
            if val != 1:
                continue
            shapely_geom = shapely.geometry.shape(geom)
            if shapely_geom.area < min_area:
                continue

            mask = rasterio.features.rasterize([shapely_geom], out_shape=field_mask.shape, fill=0, default_value=1, dtype=np.uint8)
            masks.append(mask)
            bounds = shapely_geom.bounds
            xyxys.append([bounds[0], bounds[1], bounds[2], bounds[3]])
            confidences.append(np.mean(semantic_logits.logits[field_class_id][mask == 1]))
            class_ids.append(field_class_id)

        return cls(
            xyxy=np.array(xyxys) if xyxys else np.empty((0, 4)),
            mask=np.array(masks) if masks else None,
            confidence=np.array(confidences) if confidences else None,
            class_id=np.array(class_ids) if class_ids else None,
        )

    @classmethod
    def from_instance_masks(cls, instance_masks: InstanceOutput, min_area: int = 0, score_threshold: float = 0.0) -> Detections:
        """Create Detections from InstanceOutput."""
        if score_threshold > 0:
            instance_masks = instance_masks.filter(score_threshold=score_threshold, min_area=min_area)
        if instance_masks.num_instances == 0:
            return cls(xyxy=np.empty((0, 4)))

        masks, xyxys, confidences, class_ids = [], [], [], []
        for i in range(instance_masks.num_instances):
            mask = instance_masks.masks[i]
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8)
            if np.sum(mask) < min_area:
                continue

            y_idx, x_idx = np.where(mask > 0)
            if len(y_idx) == 0:
                continue

            masks.append(mask)
            xyxys.append([x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()])
            confidences.append(float(instance_masks.scores[i]))
            class_ids.append(int(instance_masks.class_ids[i]) if instance_masks.class_ids is not None else 0)

        return cls(
            xyxy=np.array(xyxys) if xyxys else np.empty((0, 4)),
            mask=np.array(masks) if masks else None,
            confidence=np.array(confidences) if confidences else None,
            class_id=np.array(class_ids) if class_ids else None,
        )

    @classmethod
    def from_panoptic_output(cls, panoptic_output: PanopticOutput, min_area: int = 0) -> Detections:
        """Create Detections from PanopticOutput (thing instances only)."""
        return cls.from_instance_masks(panoptic_output.to_instance_masks(), min_area=min_area)

    @classmethod
    def from_gt(cls, instance_mask: np.ndarray, min_area: int = 0) -> Detections:
        """Create Detections from ground truth instance mask (each unique value = one field)."""
        import rasterio.features
        from shapely.geometry import shape

        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids > 0]

        masks, xyxys, confidences, class_ids, polygons = [], [], [], [], []
        for iid in instance_ids:
            binary = (instance_mask == iid).astype(np.uint8)
            area = np.sum(binary)
            if area < min_area:
                continue

            rows = np.any(binary, axis=1)
            cols = np.any(binary, axis=0)
            if not rows.any() or not cols.any():
                continue
            y_idx = np.where(rows)[0]
            x_idx = np.where(cols)[0]

            masks.append(binary)
            xyxys.append([x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()])
            confidences.append(min(area / 1000.0, 1.0))
            class_ids.append(0)

            try:
                shapes = list(rasterio.features.shapes(binary, mask=binary))
                if shapes:
                    polygons.append(shape(max(shapes, key=lambda x: shape(x[0]).area)[0]))
                else:
                    polygons.append(None)
            except Exception:
                polygons.append(None)

        return cls(
            xyxy=np.array(xyxys) if xyxys else np.empty((0, 4)),
            mask=np.array(masks) if masks else None,
            confidence=np.array(confidences) if confidences else None,
            class_id=np.array(class_ids) if class_ids else None,
            polygons=polygons or None,
        )

    def to_coco_format(self, image_id: int, start_annotation_id: int = 1) -> List[Dict]:
        """Convert to COCO annotations with RLE masks."""
        import pycocotools.mask as mask_util

        if self.mask is None or len(self.mask) == 0:
            return []

        annotations = []
        for i, mask in enumerate(self.mask):
            if np.sum(mask) < 10:
                continue
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            annotations.append({
                "id": start_annotation_id + len(annotations),
                "image_id": image_id,
                "category_id": int(self.class_id[i]) if self.class_id is not None else 0,
                "segmentation": rle,
                "bbox": mask_util.toBbox(rle).tolist(),
                "area": float(np.sum(mask)),
                "score": float(self.confidence[i]) if self.confidence is not None else 1.0,
                "iscrowd": 0,
            })
        return annotations

    def to_binary_mask(self, shape: Tuple[int, int], field_class_id: Optional[int] = None) -> np.ndarray:
        """Merge all instance masks into a single binary mask."""
        if self.mask is None or len(self.mask) == 0:
            return np.zeros(shape, dtype=np.uint8)

        binary = np.zeros(shape, dtype=np.uint8)
        for i, mask in enumerate(self.mask):
            if field_class_id is not None and self.class_id is not None and len(self.class_id) > i:
                if self.class_id[i] != field_class_id:
                    continue
            if mask.shape != shape:
                from skimage.transform import resize
                mask = (resize(mask, shape, preserve_range=True, anti_aliasing=True) > 0.5).astype(np.uint8)
            binary = np.logical_or(binary, mask > 0)

        return binary.astype(np.uint8)

    def compute_polygons(self, min_area: int = 0) -> List[Polygon]:
        """Extract polygons from masks for object-level metrics."""
        import rasterio.features

        if self.polygons is not None:
            result = []
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                for p in self.polygons:
                    if p is not None and not p.is_empty:
                        try:
                            if np.isfinite(p.area) and p.area >= min_area:
                                result.append(p)
                        except (RuntimeWarning, ValueError, RuntimeError):
                            continue
            return result

        polygons = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
            if self.mask is not None:
                for mask in self.mask:
                    if mask.sum() == 0:
                        continue
                    import shapely.geometry
                    for geom, val in rasterio.features.shapes(mask.astype(np.uint8)):
                        if val != 1:
                            continue
                        try:
                            sg = shapely.geometry.shape(geom)
                            if not sg.is_valid:
                                sg = shapely.make_valid(sg)
                                if hasattr(sg, "geoms"):
                                    sg = next((g for g in sg.geoms if isinstance(g, Polygon)), None)
                                    if sg is None:
                                        continue
                            if sg is not None and not sg.is_empty and sg.is_valid:
                                if np.isfinite(sg.area) and sg.area >= min_area:
                                    polygons.append(sg)
                        except Exception:
                            continue
            else:
                for bbox in self.xyxy:
                    try:
                        import shapely.geometry
                        poly = shapely.geometry.box(*bbox)
                        if poly.is_valid and not poly.is_empty and np.isfinite(poly.area) and poly.area >= min_area:
                            polygons.append(poly)
                    except Exception:
                        continue
        return polygons
