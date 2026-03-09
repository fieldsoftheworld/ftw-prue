# Changelog — Codebase Consolidation

Summary of changes made to consolidate the multi-branch PhD student codebase into a clean, usable state for the CVPR 2025 paper release.

## Branches Merged

| Branch | What it adds | Merge type |
|--------|-------------|------------|
| `prue-galileo-decode` | DECODE training/eval scripts, GFM embedding extraction (`GFMs/`) | Full merge |
| `ftw_sam2v2` | SAM2 finetuning pipeline (`sam2_ftw/`) | Full merge |
| `prue-m2f` | Mask2Former panoptic segmentation pipeline (vendored detectron2/mask2former/panopticapi, trainer, scripts, configs) | Full merge |
| `clay-finetune-rebuttal` | Clay finetuning scripts, eval scripts, visualization | Full merge |
| `prue-eval` | `tools/` (throughput benchmark, COCO converter, split search) | Cherry-picked `tools/` only |

### Branches NOT merged (and why)

| Branch | Reason |
|--------|--------|
| `prue-eval` (full) | 1080 files — includes vendored detectron2 + panopticapi (~143k lines). Only `tools/` cherry-picked. |
| `prue-m2f` | ~~Previously skipped~~ — merged with vendored detectron2/mask2former/panopticapi (modified, can't use submodules) |
| `ftw_sam2` | Superseded by `ftw_sam2v2` |
| `prue-sam` | Older SAM approach, superseded by SAM2 pipeline |

## Submodules & Dependencies

| Dependency | Strategy | Reason |
|-----------|----------|--------|
| Galileo (`nasaharvest/galileo`) | Git submodule at `pretrained/models/galileo_benchmark/galileo` | Not on PyPI; `galileo_wrappers.py` imports from `galileo/src/` |
| SAM2 (`facebookresearch/sam2`) | pip dep (`sam2>=1.0`) | On PyPI; removed all `sys.path` hacks and `SAM2_REPO_PATH` |
| Clay (`Clay-foundation/model`) | Vendored (modified) | Custom `SegmentEncoder` inherits from vendored `Encoder` class; converted sys.path hack to proper relative import |
| TerraFM / DINOv3 / TerraMind | pip deps (thin wrappers) | Already just adapter code around installed packages |
| DECODE (FracTAL ResUNet) | In-repo (original) | No external upstream |

### Package Manager

Switched to **uv** as recommended package manager. `pip install -e .` still works.

### Dependency Groups (pyproject.toml)

| Group | Install | What |
|-------|---------|------|
| core | `uv pip install -e .` | Training + eval with standard models |
| `[gfm]` | `uv pip install -e ".[gfm]"` | Foundation model encoders (terratorch, transformers, etc.) |
| `[sam2]` | `uv pip install -e ".[sam2]"` | SAM2 finetuning (sam2, hydra-core) |
| `[dev]` | `uv pip install -e ".[dev]"` | Testing + formatting (pytest, ruff) |
| `[all]` | `uv pip install -e ".[all]"` | Everything |

### Formatting

Ruff configured for **formatting only** (no linting). Excludes:
- `pretrained/models/galileo_benchmark/galileo/` (submodule)
- `pretrained/models/clay/src/` (vendored upstream code)

Applied `ruff format .` to all 1st-party code (81 files).

## Mask2Former Integration (prue-m2f)

Merged the `prue-m2f` branch which adds a Mask2Former panoptic segmentation pipeline.

### Vendored Code (modified — cannot use submodules)

| Directory | Source | Why vendored |
|-----------|--------|-------------|
| `detectron2/` | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) | Modified for multispectral (4/8-channel) satellite input |
| `mask2former/` | [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former) | Plugin with custom dataset mappers |
| `panopticapi/` | [cocodataset/panopticapi](https://github.com/cocodataset/panopticapi) | Evaluation utilities |

### 1st-Party Code Added

| Directory/File | Purpose |
|---------------|---------|
| `trainer/` | Custom detectron2 trainer, evaluation (FTWEvaluator), hooks, postprocessing, visualization |
| `scripts/train_panoptic.py` | Main training entry point with query expansion for transfer learning |
| `scripts/setup.py` | Shared config/dataset registration |
| `scripts/predict_geotiff_to_geojson.py` | GeoTIFF inference → GeoJSON output |
| `scripts/predict_and_visualize.py` | Prediction + visualization |
| `scripts/evaluate.py` | Evaluation script |
| `configs/ftw/panoptic-segmentation/` | Mask2Former configs (R50, Swin-S) |
| `create_env.sh` | Conda environment setup for M2F pipeline |

### Ruff Exclusions

Added `detectron2/`, `mask2former/`, `panopticapi/` to ruff `extend-exclude` — vendored code is not formatted.

### Dependency Group

Added `[m2f]` optional dep group: `pillow`, `pycocotools`, `fvcore`, `iopath`. Detectron2 must be installed separately: `pip install -e detectron2/`.

### Removed

| File | Reason |
|------|--------|
| `requirements.txt` | Redundant with `pyproject.toml` |

## Files Removed

| File/Directory | Reason |
|---------------|--------|
| `pretrained/models/galileo_benchmark/galileo/` | Vendored external repo (nasaharvest/galileo). ~100 files. Should be installed as pip dependency. |
| `clay_finetuned_results.txt` | Stale single-run experiment results |
| `predictions/*.png` | Result artifacts (7 PNGs, ~14MB), not source code |
| `decode/base_config.yaml` | Superseded by `decode/config_example.yaml` |
| `configs/release/run_eval.py` | Deprecated eval script using old CLI interface |

## Hardcoded Paths Fixed

All user-specific absolute paths (`/u/`, `/projects/`, `/nfs/`) replaced with environment variables or sensible defaults.

| File | What changed |
|------|-------------|
| `visualize.py` | Removed hardcoded checkpoint default; made `--ckpt_path` required; fixed global variable bug (`img_tensor_device`) |
| `eval_clay.sh` | `CLAY_CKPT_PATH` env var + `FTW_DATA_DIR` with `./data/ftw` default |
| `train_clay.sh` | `GFM_CKPT_DIR` env var with `./gfm_ckpts/encoders` default |
| `sam2_ftw/build_sam_v2.py` | `SAM2_ROOT` env var with auto-detect from installed package |
| `sam2_ftw/sam2_ftw_test.py` | `SAM2_REPO_PATH`, `FTW_DATA_ROOT`, `SAM2_CHECKPOINT_PATH` env vars |
| `sam2_ftw/sam2_ftw_train.py` | `SAM2_REPO_PATH`, `FTW_DATA_ROOT`, `SAM2_CHECKPOINT_PATH`, `SAM2_MODEL_CFG` env vars |
| `sam2_ftw/sam2_ftw_eval.py` | Same env vars as above |
| `sam2_ftw/test_model.sh` | `FTW_DATA_ROOT` default |
| `sam2_ftw/README.md` | Replaced specific paths with `/path/to/...` placeholders |
| `sam2_ftw/config_sam_rebuttal.yaml` | Nullified user-specific paths with instructional comments |
| `sam2_ftw/config_example_3class.yaml` | Same as above |
| `tools/search_ftw_image_splits.py` | Fixed shebang (`#!/usr/bin/env python3`), default data path |

## New Files Added

| File | Purpose |
|------|---------|
| `pyproject.toml` | Proper Python packaging: dependencies, optional groups (`gfm`, `sam2`, `dev`), entry points, ruff + pytest config |
| `tests/test_losses.py` | Unit tests for PixelWeightedCE, logCoshDice, logCoshDiceCE |
| `tests/test_segmentor.py` | Unit tests for SegmentationHead across all model configs (parametrized) |
| `tests/test_decode.py` | Unit tests for FracTAL_ResUNet_cmtsk forward pass and output validation |
| `tests/test_settings.py` | Validates constants (country list, temporal options) |
| `tests/test_utils.py` | Tests for harvest_to_datetime, parse_bbox, compute_md5 |
| `tests/conftest.py` | Shared fixtures; skips if torch unavailable |
| `tools/benchmark_throughput.py` | Benchmark inference speed in km²/s (from prue-eval) |
| `tools/ftw_to_coco.py` | Convert FTW dataset to COCO format (from prue-eval) |
| `tools/search_ftw_image_splits.py` | Lookup image filenames by split index (from prue-eval) |
| `CHANGELOG.md` | This file |

## Files Updated

| File | What changed |
|------|-------------|
| `README.md` | Complete rewrite: CVPR 2025 header, pip install instructions, repo structure, env var docs, model tables, citation block |
| `.gitignore` | Replaced bloated GitHub template (~210 lines) with focused project-specific patterns (~40 lines) |

## Environment Variables Reference

| Variable | Default | Used by |
|----------|---------|---------|
| `FTW_DATA_DIR` | `./data/ftw` | `eval_clay.sh` |
| `FTW_DATA_ROOT` | `./data/ftw` | SAM2 scripts |
| `GFM_CKPT_DIR` | `./gfm_ckpts/encoders` | `train_clay.sh` |
| `CLAY_CKPT_PATH` | *(required)* | `eval_clay.sh` |
| `SAM2_ROOT` | auto-detect | `build_sam_v2.py` |
| `SAM2_CHECKPOINT_PATH` | *(required)* | SAM2 train/test/eval scripts |
| `SAM2_MODEL_CFG` | `sam2_hiera_s.yaml` | `sam2_ftw_train.py` |
