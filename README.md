# ftw-prue: Field Boundary Segmentation

Official repository for "PRUE: A Practical Recipe for Field Boundary Segmentation at Scale".


## Overview

This codebase provides:
- Support for multiple model architectures (standard segmentation models, pretrained encoders, and custom models)
- Training and evaluation pipelines for FTW based field boundary segmentation
- Feature extraction utilities for precomputing embeddings
- Unified interface for working with different model architectures

## Data Setup

1. Download the FTW (Fields of the World) dataset following instructions in the [ftw-baselines repository](https://github.com/fieldsoftheworld/ftw-baselines)
2. Place the dataset under `./data/ftw`

## Environment Setup

Create and activate the Conda environment:

```bash
conda env create -f env.yaml
conda activate ftw
```

## Available Models

The repository supports multiple model architectures organized into categories:

### Standard Segmentation Models

These models work directly on images and use standard backbones:

- **`unet`**: U-Net architecture with various backbones (e.g., `efficientnet-b3`, `resnet50`)
- **`deeplabv3+`**: DeepLabV3+ with encoder backbones
- **`fcn`**: Fully Convolutional Network
- **`upernet`**: UPerNet architecture
- **`segformer`**: SegFormer transformer-based model
- **`dpt`**: Dense Prediction Transformer

**Usage**: Set `model: "unet"` (or other model name) and `backbone: "efficientnet-b3"` in config.

### Pretrained Foundation Models (GFM)

These models use pretrained encoders with a segmentation decoder:

- **Foundation models**: `clay`, `terrafm`, `dinov3`, `terramind`
- **Galileo benchmark models**: `croma`, `decur`, `dofa`, `prithvi`, `satlas`, `softcon`, `galileo`

**Usage**: Set `model: "gfm"` and `backbone: "clay"` (or other encoder name) in config.

Model checkpoints should be placed in `gfm_ckpts/encoders/` (or set `FTW_CKPT_BASE_DIR` environment variable).

### Feature-Based Models

- **`pretrained`**: Uses precomputed features from any encoder

**Usage**: Set `model: "pretrained"` and provide precomputed features via `feat_root`.

### Custom Models

- **`decode`**: FracTAL ResUNet with multi-task learning (segmentation, boundary, distance)

**Usage**: Set `model: "decode"` and `loss: "decode"` in config. See [decode/README.md](decode/README.md) for details.

For detailed usage and API documentation, see [pretrained/README.md](pretrained/README.md).

## Training

### Using Training Scripts

For GFM models (pretrained encoders), use `train_gfm.sh`:

```bash
./train_gfm.sh <model_name> <input_type> [<feat_root>] [<log_mode>]
```

**Arguments:**
- `model_name`: Model to train (e.g., `clay`, `terrafm`, `croma`, `decode`)
- `input_type`: `images_noaug` (raw images) or `features` (precomputed embeddings)
- `feat_root`: Path to precomputed features (required when `input_type=features`)
- `log_mode`: Logging mode (default: `disabled`)

**Examples:**
```bash
./train_gfm.sh clay images_noaug disabled
./train_gfm.sh terrafm features /path/to/features disabled
./train_gfm.sh decode images_noaug disabled
```

### Using Lightning CLI

For all models, you can use the Lightning CLI with a config file:

```bash
python -m ftw_tools.cli model fit --config <config_file.yaml>
```

Example config files are available in `configs/release/` and `decode/config_example.yaml`.

## Evaluation

### Using Evaluation Scripts

For GFM models, use `eval_gfm.sh`:

```bash
./eval_gfm.sh <model_filter> <experiment> <input_type> [<feat_root_base>]
```

**Arguments:**
- `model_filter`: `all` or specific model name (e.g., `clay`, `decode`)
- `experiment`: `main` or `supp`
- `input_type`: `images_noaug` or `features`
- `feat_root_base`: Directory with precomputed features (required when `input_type=features`)

**Examples:**
```bash
./eval_gfm.sh all main features /path/to/features
./eval_gfm.sh clay main images_noaug
./eval_gfm.sh decode main images_noaug
```

Decoder checkpoints should be placed under:
- `gfm_ckpts/decoders/main/<model_name>/`
- `gfm_ckpts/decoders/supp/<model_name>/`

### Using Lightning CLI

For all models, you can use the Lightning CLI:

```bash
python -m ftw_tools.cli model test \
  --model <checkpoint_path> \
  --countries france \
  --test_split test \
  --input_type images \
  --dir ./data/ftw \
  --gpu 0 \
  --out results.json
```

## Feature Extraction

To precompute embeddings for the entire dataset:

```bash
python -m pretrained.models.compute_feats --model <model_name> --batch_size 32
```

This extracts embeddings for all Sentinel-2 images and saves them as `.npz` files. See `pretrained/README.md` for detailed options.
