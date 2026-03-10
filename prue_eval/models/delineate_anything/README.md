# DelineateAnything Segmenter

This module provides a unified segmenter adapter for **DelineateAnything**, a YOLO-based model for delineating agricultural fields from satellite imagery.

## Overview

DelineateAnything is an instance segmentation model that:
- Takes RGB satellite imagery as input (extracts RGB from multi-channel inputs)
- Uses YOLO11 architecture for field detection and delineation
- Produces instance masks with confidence scores
- Supports two model variants: full (YOLO11x) and small (YOLO11n)

## Usage

### Basic Usage

```python
import sys
sys.path.insert(0, 'src')
import models  # Triggers registration
from models.registry import create_segmenter
import torch

# Create segmenter
seg = create_segmenter(
    'delineate_anything',
    model_weights='DelineateAnything-S',  # Or path to local checkpoint
    patch_size=256,
    resize_factor=2,
    conf_threshold=0.05,
    device='cuda'
)

# Run inference on RGBN image (DA extracts RGB automatically)
image = torch.rand((1, 4, 256, 256))  # (B, C, H, W) - windowA or windowB
outputs = list(seg.predict(image))

# Convert to Detections
detections = outputs[0].to_detections(
    min_area=0,
    score_threshold=0.0  # Model already applied conf_threshold
)
```

### Choosing Temporal Window

When using multi-temporal Sentinel-2 data, use `--temporal_options` to select which window:

```bash
# Use Window A (more recent) - DEFAULT for DA
python scripts/run_model_inference.py \
    --model delineate_anything \
    --temporal_options windowA \
    --data_dir /path/to/data \
    --output_dir /path/to/output

# Use Window B (older)
python scripts/run_model_inference.py \
    --model delineate_anything \
    --temporal_options windowB \
    --data_dir /path/to/data \
    --output_dir /path/to/output
```

**Note:** DelineateAnything uses only RGB channels. For Sentinel-2 data:
- `temporal_options=windowA` → loads 4-channel RGBN from Window A, extracts RGB
- `temporal_options=windowB` → loads 4-channel RGBN from Window B, extracts RGB
- Do NOT use `stacked` - DA cannot use 8-channel input

### Using the "da" Alias

```python
# Short alias for convenience
seg = create_segmenter('da', model_weights='DelineateAnything')
```

### Using Local Checkpoint

```python
seg = create_segmenter(
    'delineate_anything',
    model_weights='/path/to/checkpoint.pt',
    model_variant='DelineateAnything',  # Explicit variant
    conf_threshold=0.05,
)
```

## Parameters

- **model_weights**: Path to checkpoint or model variant name
  - `"DelineateAnything"`: Full model (YOLO11x)
  - `"DelineateAnything-S"`: Small model (YOLO11n)
  - Or path to local `.pt` file

- **model_variant**: Explicit variant (if None, inferred from model_weights)
  - `"DelineateAnything"` or `"DelineateAnything-S"`

- **patch_size**: Size of input patches (default: 256)
  - Images are processed at this resolution

- **resize_factor**: Factor to resize patches before inference (default: 2)
  - Actual inference size = patch_size × resize_factor

- **max_detections**: Maximum detections per image (default: 100)

- **iou_threshold**: IoU threshold for NMS (default: 0.3)

- **conf_threshold**: Confidence threshold for detections (default: 0.05)
  - YOLO default is 0.05, not 0.5 like other models

- **device**: Device to run on (`"cuda"` or `"cpu"`)

## Model Variants

### DelineateAnything (Full)
- Based on YOLO11x
- Higher accuracy, slower inference
- ~100M parameters
- Best for production use

### DelineateAnything-S (Small)
- Based on YOLO11n
- Faster inference, good accuracy
- ~3M parameters
- Good for rapid prototyping and testing

## Input Format

DelineateAnything expects:
- **Shape**: `(B, C, H, W)` where C >= 3
- **Channels**: Uses only RGB (first 3 channels)
  - For RGBN input (4 channels), automatically extracts RGB
  - For stacked temporal data (8 channels), extracts RGB from first window
- **Values**: Expects raw Sentinel-2 values (typically 0-3000 range)
  - Model applies **adaptive percentile-based normalization** (1st-99th percentile)
  - Matches original Delineate Anything implementation
  - Better handles varying atmospheric conditions, seasons, and regions
  - Utilizes full dynamic range for improved contrast

## Output Format

Returns `InstanceOutput` objects with:
- **masks**: Binary instance masks `(N, H, W)`
- **scores**: Confidence scores `(N,)` from YOLO detection
- **image_id**: Image index in batch (0, 1, 2, ...)
- **metadata**: Model type and other info

## Integration with Pipeline

The segmenter integrates seamlessly with the rest of the pipeline:

```python
# Full pipeline
seg = create_segmenter('da', model_weights='DelineateAnything-S')
outputs = list(seg.predict(images))

# Convert to Detections
detections = outputs[0].to_detections(min_area=100)

# Evaluate
from evaluator import Evaluator
evaluator = Evaluator()
metrics = evaluator.evaluate(detections, ground_truth)
```

## Normalization

This implementation uses **adaptive percentile-based normalization**, matching the original Delineate Anything codebase:

### How It Works
1. For each input image, computes 1st and 99th percentile per RGB channel
2. Uses only positive (non-zero) values for percentile calculation
3. Normalizes each channel: `(x - p1) / (p99 - p1)` then clips to [0, 1]
4. Averages normalization bounds across batch

### Benefits Over Fixed Normalization
- **Adaptive to conditions**: Automatically adjusts for atmospheric conditions, cloud cover, seasons
- **Better dynamic range**: Uses full [0, 1] range regardless of input distribution
- **Region-agnostic**: Works equally well across different geographic regions
- **Contrast enhancement**: Removes outliers for better feature visibility

### Comparison
- **Fixed (old)**: `x / 3000.0` → assumes all images in [0-3000] range
- **Adaptive (new)**: `(x - p1) / (p99 - p1)` → uses actual data distribution

This matches the original Delineate Anything implementation and should improve performance, especially for images with varying atmospheric conditions or acquisition dates.

## Notes

- **RGB-only**: DA requires RGB input; other channels are ignored
- **Confidence filtering**: YOLO applies `conf_threshold` during inference
  - No need for additional score filtering in `to_detections()`
  - Set `score_threshold=0.0` to avoid double filtering
- **Default threshold**: Use 0.05 (YOLO default), not 0.5
- **Checkpoints**: Models download automatically from HuggingFace if not found locally

## Testing

Run tests with:

```bash
cd /nfs/stak/users/fangzha/projects/ftw-prue
pytest tests/test_delineate_anything_segmenter.py -v
```

## Registry Names

This segmenter is registered under two names:
- `"delineate_anything"`: Full name
- `"da"`: Short alias

Both can be used interchangeably with `create_segmenter()`.
