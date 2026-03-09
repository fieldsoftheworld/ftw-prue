# PRUE

**A Practical Recipe for Field Boundary Segmentation at Scale**

CVPR 2025 &middot; [Paper (coming soon)]() &middot; [Fields of the World](https://github.com/fieldsoftheworld/ftw-baselines)

---

This repo benchmarks field boundary segmentation across 25 countries using standard segmentation models, geospatial foundation model (GFM) encoders, and custom architectures (DECODE, SAM2) on the [Fields of the World (FTW)](https://github.com/fieldsoftheworld/ftw-baselines) dataset.

## Setup

```bash
git clone --recurse-submodules https://github.com/fieldsoftheworld/ftw-prue.git
cd ftw-prue

# using uv (recommended)
uv pip install -e .            # core: training + eval
uv pip install -e ".[gfm]"    # + foundation model encoders
uv pip install -e ".[sam2]"   # + SAM2 finetuning
uv pip install -e ".[m2f]"   # + Mask2Former (also need: pip install -e detectron2/)
uv pip install -e ".[dev]"    # + pytest, ruff
uv pip install -e ".[all]"    # everything
```

`pip install -e .` works too if you don't have [uv](https://docs.astral.sh/uv/).

Download the FTW dataset per the [ftw-baselines instructions](https://github.com/fieldsoftheworld/ftw-baselines) and place it at `./data/ftw` (or set `FTW_DATA_DIR`).

## Repo layout

```
ftw_tools/       Core package — datasets, trainers, losses, metrics, postprocessing
pretrained/      GFM encoder wrappers + feature extraction
decode/          DECODE (FracTAL ResUNet) multi-task model
sam2_ftw/        SAM2 finetuning pipeline
detectron2/      Vendored detectron2 (modified for multispectral input)
mask2former/     Vendored Mask2Former (modified)
panopticapi/     Vendored panoptic evaluation utilities
trainer/         Mask2Former training infrastructure (custom trainer, eval, hooks)
scripts/         Mask2Former training/inference entry points
configs/         Training configs (2-class, 3-class, ViT, Mask2Former panoptic)
GFMs/            Embedding extraction scripts (CROMA, DeCUR, DOFA, …)
tools/           Throughput benchmark, COCO converter, split search
tests/           Unit tests
```

## Models

**Standard decoders** — U-Net, DeepLabV3+, FCN, UPerNet, SegFormer, DPT (via [smp](https://github.com/qubvel-org/segmentation_models.pytorch) / [timm](https://github.com/huggingface/pytorch-image-models))

**GFM encoders** — Clay, TerraFM, DINOv3, TerraMind, CROMA, DeCUR, DOFA, Prithvi, SatLAS, SoftCon, Galileo

**Custom** — DECODE (FracTAL ResUNet multi-task), SAM2 (temporal propagation), Mask2Former (panoptic segmentation)

## Training

```bash
# GFM encoder (from images)
./train_gfm.sh clay images_noaug

# GFM encoder (from precomputed features)
./train_gfm.sh terrafm features /path/to/features

# Clay finetuning (encoder + decoder end-to-end)
./train_clay.sh a online

# DECODE
./train_gfm.sh decode images_noaug

# Mask2Former panoptic segmentation
python scripts/train_panoptic.py \
  --config-file configs/ftw/panoptic-segmentation/maskformer2_R50_ftw_panoptic.yaml \
  --coco-root /path/to/coco/output

# Lightning CLI directly
python -m ftw_tools.cli model fit --config configs/release/3_class/full-ftw.yaml
```

## Evaluation

```bash
# All GFM models
./eval_gfm.sh all main features /path/to/features

# Single model
./eval_gfm.sh clay main images_noaug

# Clay finetuned
./eval_clay.sh main

# Lightning CLI
python -m ftw_tools.cli model test \
  --model checkpoint.ckpt \
  --countries france \
  --test_split test \
  --input_type images \
  --dir ./data/ftw \
  --gpu 0 \
  --out results.json
```

## Feature extraction

Precompute embeddings for the full dataset:

```bash
python -m pretrained.models.compute_feats --model clay --batch_size 32
```

Per-model scripts in [`GFMs/`](GFMs/).

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FTW_DATA_DIR` / `FTW_DATA_ROOT` | `./data/ftw` | Dataset root |
| `GFM_CKPT_DIR` | `./gfm_ckpts/encoders` | Encoder checkpoints |
| `CLAY_CKPT_PATH` | *(required)* | Clay checkpoint for `eval_clay.sh` |
| `SAM2_CHECKPOINT_PATH` | *(required)* | SAM2 base checkpoint |
| `SAM2_MODEL_CFG` | `sam2_hiera_s.yaml` | SAM2 model config |
| `FTW_GEOPARQUET_ROOT` | *(required)* | Geoparquet root for `tools/ftw_to_coco.py` |

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
