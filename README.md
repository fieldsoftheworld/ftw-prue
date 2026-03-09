# PRUE: A Practical Recipe for Field Boundary Segmentation at Scale

Official repository for "PRUE: A Practical Recipe for Field Boundary Segmentation at Scale" (CVPR 2025).

<!--
TODO: Add arxiv link and bibtex when available
-->

## Overview

Benchmark for field boundary segmentation on the [Fields of the World (FTW)](https://github.com/fieldsoftheworld/ftw-baselines) dataset. Supports:

- **Standard segmentation models**: U-Net, DeepLabV3+, FCN, UPerNet, SegFormer, DPT
- **Foundation model encoders**: Clay, TerraFM, DINOv3, TerraMind, CROMA, DeCUR, DOFA, Prithvi, SatLAS, SoftCon, Galileo
- **Custom architectures**: DECODE (FracTAL ResUNet), SAM2
- **Multi-task learning**: segmentation + boundary detection + distance regression
- Feature extraction and precomputed embedding pipelines
- Per-country evaluation across 25 countries

## Installation

```bash
pip install -e .

# With foundation model support
pip install -e ".[gfm]"

# With dev tools (testing, linting)
pip install -e ".[dev]"
```

For SAM2 support, install [SAM2](https://github.com/facebookresearch/sam2) separately.

## Data Setup

1. Download the FTW dataset following [ftw-baselines](https://github.com/fieldsoftheworld/ftw-baselines)
2. Place under `./data/ftw` (or set `FTW_DATA_DIR` / `FTW_DATA_ROOT` env var)

## Repository Structure

```
ftw_tools/          # Core package: training, eval, data, losses, metrics, postprocessing
pretrained/         # Foundation model encoder wrappers + feature extraction
decode/             # FracTAL ResUNet multi-task model
sam2_ftw/           # SAM2 finetuning pipeline
configs/            # Training configs (2-class, 3-class, ViT)
tools/              # Utilities: throughput benchmark, COCO converter, split search
GFMs/               # Scripts for extracting GFM embeddings (CROMA, DeCUR, etc.)
tests/              # Unit tests
```

## Available Models

### Standard Segmentation Models

Direct image-to-segmentation with standard backbones.

| Model | Config key | Example backbone |
|-------|-----------|-----------------|
| U-Net | `unet` | `efficientnet-b3`, `resnet50` |
| DeepLabV3+ | `deeplabv3+` | `resnet50` |
| FCN | `fcn` | `resnet50` |
| UPerNet | `upernet` | `resnet50` |
| SegFormer | `segformer` | `mit_b2` |
| DPT | `dpt` | `vit_base_patch16_384` |

### Foundation Models (GFM)

Pretrained satellite encoders with segmentation decoders. Set `model: "gfm"`.

| Encoder | Key | Source |
|---------|-----|--------|
| Clay v1.5 | `clay` | [Made With Clay](https://github.com/Clay-foundation/model) |
| TerraFM | `terrafm` | IBM |
| DINOv3 | `dinov3` | - |
| TerraMind | `terramind` | DLR |
| CROMA / DeCUR / DOFA / Prithvi / SatLAS / SoftCon / Galileo | respective keys | [Galileo benchmark](https://github.com/nasaharvest/galileo) |

Encoder checkpoints: `gfm_ckpts/encoders/` (or `GFM_CKPT_DIR` env var).

### Custom Models

- **DECODE** (`model: "decode"`): FracTAL ResUNet with multi-task head. See [`decode/README.md`](decode/README.md).
- **SAM2** (`model: "sam2"`): SAM2 finetuned for field segmentation. See [`sam2_ftw/README.md`](sam2_ftw/README.md).

## Training

### With training scripts

```bash
# GFM encoder (images)
./train_gfm.sh clay images_noaug

# GFM encoder (precomputed features)
./train_gfm.sh terrafm features /path/to/features

# DECODE model
./train_gfm.sh decode images_noaug

# Clay finetuning (encoder + decoder)
./train_clay.sh a online
```

### With Lightning CLI

```bash
python -m ftw_tools.cli model fit --config configs/release/3_class/full-ftw.yaml
```

Example configs in [`configs/release/`](configs/release/) and [`decode/config_example.yaml`](decode/config_example.yaml).

## Evaluation

### With eval scripts

```bash
# All GFM models
./eval_gfm.sh all main features /path/to/features

# Single model
./eval_gfm.sh clay main images_noaug

# Clay finetuned
./eval_clay.sh main
```

### With Lightning CLI

```bash
python -m ftw_tools.cli model test \
  --model checkpoint.ckpt \
  --countries france \
  --test_split test \
  --input_type images \
  --dir ./data/ftw \
  --gpu 0 \
  --out results.json
```

Decoder checkpoints: `gfm_ckpts/decoders/{main,supp}/<model_name>/`

## Feature Extraction

Precompute embeddings for the full dataset:

```bash
python -m pretrained.models.compute_feats --model clay --batch_size 32
```

Per-model embedding extraction scripts in [`GFMs/`](GFMs/).

## Tools

| Tool | Description |
|------|-------------|
| `tools/benchmark_throughput.py` | Benchmark model inference speed (km²/s) |
| `tools/ftw_to_coco.py` | Convert FTW dataset to COCO format |
| `tools/search_ftw_image_splits.py` | Look up image filenames by split index |
| `aggregate.py` | Aggregate per-country metrics into overall results |
| `visualize.py` | Visualize model predictions vs ground truth |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FTW_DATA_DIR` / `FTW_DATA_ROOT` | `./data/ftw` | FTW dataset root |
| `GFM_CKPT_DIR` | `./gfm_ckpts/encoders` | GFM encoder checkpoints |
| `CLAY_CKPT_PATH` | - | Clay finetuned checkpoint (for eval_clay.sh) |
| `SAM2_REPO_PATH` | - | Path to cloned SAM2 repo |
| `SAM2_CHECKPOINT_PATH` | - | SAM2 base checkpoint |

## Testing

```bash
pytest tests/ -v
```

## Citation

```bibtex
@inproceedings{prue2025,
  title={PRUE: A Practical Recipe for Field Boundary Segmentation at Scale},
  author={TODO},
  booktitle={CVPR},
  year={2025}
}
```

## License

[MIT](LICENSE)
