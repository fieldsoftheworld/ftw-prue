# prue_eval

Unified evaluation framework for field boundary segmentation.

- `detections.py` — `Detections` dataclass (mask, polygon, COCO conversions)
- `evaluator.py` — `Evaluator` (pixel/object/COCO metrics)
- `converters.py` — model output → `SemanticOutput`/`InstanceOutput`/`PanopticOutput`
- `intermediate_formats.py` — intermediate output dataclasses
- `models/` — per-backend adapters (SAM, DECODE, DA, Mask2Former) registered via `models/registry.py`
