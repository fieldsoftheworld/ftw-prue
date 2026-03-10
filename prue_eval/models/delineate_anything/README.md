# DelineateAnything Adapter

YOLO-based field delineation from satellite imagery. Registered as `"delineate_anything"` / `"da"`.

## Variants

| Variant | Backbone | Params |
|---------|----------|--------|
| `DelineateAnything` | YOLO11x | ~100M |
| `DelineateAnything-S` | YOLO11n | ~3M |

## Usage

```python
from prue_eval.models.registry import create_segmenter

seg = create_segmenter("da", model_weights="DelineateAnything-S", patch_size=256, device="cuda")
outputs = list(seg.predict(images))  # → InstanceOutput
detections = outputs[0].to_detections(min_area=100)
```

## Input

- Shape: `(B, C, H, W)` where C >= 3 (uses RGB only)
- Values: raw Sentinel-2 DN (0-3000 range); model applies adaptive percentile normalization
- Temporal: use `--temporal_options windowA` (not `stacked`)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_weights` | -- | Path or variant name (`"DelineateAnything"`, `"DelineateAnything-S"`) |
| `patch_size` | 256 | Input patch resolution |
| `resize_factor` | 2 | Upscale factor before inference |
| `max_detections` | 100 | Max instances per image |
| `iou_threshold` | 0.3 | NMS IoU threshold |
| `conf_threshold` | 0.05 | Detection confidence threshold (YOLO default, not 0.5) |
