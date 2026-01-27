# MY SAM-2 QUICK COMMANDS (paths my need to be updated)

### Training
```bash
python -m ftw_tools.cli model fit --config sam2_ftw/config_sam_rebuttal.yaml
```

### Testing
```bash
python -m ftw_tools.cli model test   --model /u/gmuhawenayo/projects/PRUE-CVPR/ftw-prue/logs/sam2-ftw-rebuttal/FTW-project/5294stag/checkpoints/last.ckpt   --countries germany   --test_split test   --input_type images   --temporal_options sam2   --dir /u/gmuhawenayo/datasets/FTW-Dataset/ftw   --gpu 0   --out sam2_ftw/test_results_germany.json
```

To test for all countries check [test_ftw_sam.sh](test_ftw_sam.sh])

### Visualization
```bash
python viz_sam2_ftw.py   --model /u/gmuhawenayo/projects/PRUE-CVPR/ftw-prue/logs/sam2-ftw-rebuttal/FTW-project/5294stag/checkpoints/last.ckpt   --data_root /u/gmuhawenayo/datasets/FTW-Dataset/ftw   --country cambodia   --split test   --num_samples 5   --out_dir sam2_ftw/viz
```

# SAM-2 FTW Integration

This directory contains the integration of SAM-2 (Segment Anything Model 2) into the FTW training pipeline.

## Overview

SAM-2 is fine-tuned for field boundary segmentation on the FTW dataset using:
- **Temporal memory**: Uses `window_a` → `window_b` sequence
- **Multi-modal prompts**: Points + mask prompts
- **Selective fine-tuning**: Only the mask decoder is trained (image encoder and prompt encoder are frozen)

## Integration Details

### 1. Dataset Support (`ftw_tools/torchgeo/datasets.py`)

Added support for SAM-2 format:
- **Temporal option**: `"sam2"` - Returns `window_a` and `window_b` separately
- **Point sampling**: Automatically samples points from field masks
- **Binary masks**: Converts 3-class masks to binary field masks
- **Image preprocessing**: Resizes to `sam2_max_image_size` (default: 1024px), RGB only

**New functions:**
- `sample_points_from_mask()`: Samples points from binary field masks
- `prepare_binary_field_mask()`: Converts 3-class to binary masks

**Dataset returns:**
- `window_a`: [B, 3, H, W] - Earlier temporal window (RGB, float32, 0-255 range)
- `window_b`: [B, 3, H, W] - Later temporal window (RGB, float32, 0-255 range)
- `field_mask`: [B, H, W] - Binary field mask (float32, 0 or 1)
- `points`: [B, N, 2] - Point coordinates (x, y) or None
- `point_labels`: [B, N] - Point labels (all 1s for positive points) or None
- `mask_3class`: [B, H, W] - Original 3-class mask (for reference)

**Normalization flow:**
1. Read raw int16 data [0, 3000] from Sentinel-2 images
2. Normalize by dividing by 3000 → [0, 1] (FTW standard normalization)
3. No resizing needed (FTW images are already 256x256)
4. Scale to uint8 [0, 255] for storage (matching original SAM-2 scripts)
5. In trainer: Divide by 255.0 → [0, 1] for model input

This ensures proper normalization from int16 Sentinel-2 data while maintaining compatibility with SAM-2's expected input format. Since FTW images are 256x256, no resizing is performed.

### 2. Model Integration (`ftw_tools/torchgeo/trainers.py`)

**Model configuration:**
- Model type: `"sam2"`
- Loss: `"sam2"` or `"bce"` (binary cross-entropy)
- Only mask decoder parameters are trainable

**Forward pass:**
- Processes `window_a` first (no gradients) to build temporal memory
- Processes `window_b` with temporal memory
- Uses points + mask prompts
- Returns binary predictions

**Optimizer:**
- Only optimizes `model.sam_mask_decoder.parameters()`

### 3. DataModule (`ftw_tools/torchgeo/datamodules.py`)

**SAM-2 specific settings:**
- No augmentations (images are preprocessed in dataset)
- Passes `sam2_max_image_size` and `sam2_num_points` to dataset
- Batch size should be smaller (SAM-2 is memory intensive)

### 4. Configuration File

Example config: `sam2_ftw/config_example.yaml`

**Key parameters:**
```yaml
model:
  model: "sam2"
  loss: "sam2"
  model_kwargs:
    sam2_repo_path: null  # Path to sam2 repo (or null)
    model_cfg: "sam2_hiera_s.yaml"
    checkpoint_path: "/path/to/sam2.1_hiera_small.pt"

data:
  temporal_options: "sam2"
  sam2_max_image_size: 1024
  sam2_num_points: 1  # Training: 1, Validation/Test: 3
  batch_size: 4  # Smaller batch size
```

## Usage

### Training

```bash
python -m ftw_tools.cli model fit \
  --config sam2_ftw/config_example.yaml \
  --model.model_kwargs.checkpoint_path /path/to/sam2.1_hiera_small.pt
```

### Testing

```bash
python -m ftw_tools.cli model test \
  --model logs/sam2-ftw-3-class/checkpoints/last.ckpt \
  --countries france \
  --test_split test \
  --input_type images \
  --temporal_options sam2 \
  --dir /u/gmuhawenayo/datasets/FTW-Dataset/ftw \
  --gpu -1 \
  --out sam2_ftw/test_results.json
```

Or use the provided test script:

```bash
# Basic usage (uses defaults)
./sam2_ftw/test_model.sh

# Custom checkpoint and output
./sam2_ftw/test_model.sh \
  logs/sam2-ftw-3-class/checkpoints/best.ckpt \
  /u/gmuhawenayo/datasets/FTW-Dataset/ftw \
  sam2_ftw/my_results.json \
  -1  # GPU (-1 for CPU, 0 for GPU)
```

## Differences from Standalone Scripts

The integrated version differs from `sam2_ftw_train.py` and `sam2_ftw_test.py`:

1. **Lightning integration**: Uses PyTorch Lightning for training loop, logging, checkpointing
2. **Config-based**: All settings via YAML config files
3. **Metrics**: Integrated with existing FTW metrics (IoU, precision, recall)
4. **Data loading**: Uses FTW dataset class with SAM-2 mode
5. **Batch processing**: Supports batched training (original scripts use batch_size=1)

## Requirements

- SAM-2 repository must be accessible (either in system path or via `sam2_repo_path`)
- Base SAM-2 checkpoint (e.g., `sam2.1_hiera_small.pt`)
- FTW dataset with `window_a`, `window_b`, and `semantic_3class` masks

## Notes

- **Memory**: SAM-2 is memory intensive. Use smaller batch sizes (2-4)
- **Point sampling**: Training uses 1 point, validation/test can use more (3)
- **Image size**: Images are resized to max 1024px to fit in memory
- **Temporal memory**: `window_a` is processed without gradients to build temporal context

