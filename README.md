# ftw-prue: Field Boundary Segmentation

Official repository for "PRUE: A Practical Recipe for Field Boundary Segmentation at Scale".


## Overview

This codebase provides:
- Support for multiple pretrained encoders (CLAY, TerraFM, DINOv3, TeraMind, and Galileo benchmark models)
- Training and evaluation pipelines for FTW based field boundary segmentation
- Feature extraction utilities for precomputing embeddings
- Unified interface for working with different encoder architectures

## Data Setup

1. Download the FTW (Fields of the World) dataset following instructions in the [ftw-baselines repository](https://github.com/fieldsoftheworld/ftw-baselines)
2. Place the dataset under `./data/ftw`

## Environment Setup

Create and activate the Conda environment:

```bash
conda env create -f env.yaml
conda activate ftw
```

## Pretrained Models

The repository supports multiple pretrained encoders for feature extraction:

- Foundation models: `clay`, `terrafm`, `dinov3`, `terramind`
- Galileo benchmark models: `croma`, `decur`, `dofa`, `prithvi`, `satlas`, `softcon`, `galileo`

Model checkpoints should be placed in `gfm_ckpts/encoders/` (or set `FTW_CKPT_BASE_DIR` environment variable).

For detailed usage and API documentation, see [pretrained/README.md](pretrained/README.md).

## Training

Training is performed using the `train_gfm.sh` script:

```bash
./train_gfm.sh <model_name> <input_type> [<feat_root>] [<log_mode>]
```

**Arguments:**
- `model_name`: Model to train (e.g., `clay`, `terrafm`, `croma`)
- `input_type`: `images_noaug` (raw images) or `features` (precomputed embeddings)
- `feat_root`: Path to precomputed features (required when `input_type=features`)
- `log_mode`: Logging mode (default: `disabled`)

**Example:**
```bash
./train_gfm.sh clay images_noaug disabled
./train_gfm.sh terrafm features /path/to/features disabled
```

## Evaluation

Evaluation requires both encoder and decoder checkpoints. Decoder checkpoints should be placed under:
- `gfm_ckpts/decoders/main/<model_name>/`
- `gfm_ckpts/decoders/supp/<model_name>/`

Run evaluation:

```bash
./eval_gfm.sh <model_filter> <experiment> <input_type> [<feat_root_base>]
```

**Arguments:**
- `model_filter`: `all` or specific model name (e.g., `clay`)
- `experiment`: `main` or `supp`
- `input_type`: `images_noaug` or `features`
- `feat_root_base`: Directory with precomputed features (required when `input_type=features`)

**Example:**
```bash
./eval_gfm.sh all main features /path/to/features
./eval_gfm.sh clay main images_noaug
```

## Feature Extraction

To precompute embeddings for the entire dataset:

```bash
python -m pretrained.models.compute_feats --model <model_name> --batch_size 32
```

This extracts embeddings for all Sentinel-2 images and saves them as `.npz` files. See `pretrained/README.md` for detailed options.
