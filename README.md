# PRUE

**A Practical Recipe for Field Boundary Segmentation at Scale**

CVPR 2025 &middot; [Paper (coming soon)]() &middot; [Fields of the World](https://github.com/fieldsoftheworld/ftw-baselines)

---

This repo benchmarks field boundary segmentation across 25 countries using standard segmentation models, geospatial foundation model (GFM) encoders, and custom architectures (DECODE, SAM2) on the [Fields of the World (FTW)](https://github.com/fieldsoftheworld/ftw-baselines) dataset.

## Setup

Requires Python >=3.11, <3.14. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone --recurse-submodules https://github.com/fieldsoftheworld/ftw-prue.git
cd ftw-prue

uv pip install -e .               # core: training + eval
uv pip install -e ".[gfm]"       # + foundation model encoders
uv pip install -e ".[sam2]"      # + SAM2 finetuning
uv pip install -e ".[eval]"      # + PRUE evaluation framework
uv pip install -e ".[dev]"       # + pytest, ruff
uv pip install -e ".[all]"       # everything

# Mask2Former (requires vendored detectron2)
uv pip install -e vendor/detectron2/ --no-build-isolation
uv pip install -e ".[m2f]"
```

Download the FTW dataset per the [ftw-baselines instructions](https://github.com/fieldsoftheworld/ftw-baselines) and place it at `./data/ftw` (or set `FTW_DATA_DIR`). 

If you prefer not to compute embeddings locally, you can download [precomputed FTW GFM embeddings](https://source.coop/mvrl/ftw-inference-gfm/precomputed_feats) from Source Cooperative.


## Repo layout

```text
ftw_tools/         Core — datasets, trainers, losses, metrics, postprocessing
pretrained/        GFM encoder wrappers + feature extraction
prue_eval/         Unified evaluation framework (registry, intermediate formats, metrics)
decode/            DECODE (FracTAL ResUNet) multi-task model
trainer/           Mask2Former training infrastructure
sam2_ftw/          SAM2 finetuning pipeline
vendor/            Vendored third-party (detectron2, mask2former, panopticapi)
scripts/           Training, inference, evaluation, embedding extraction
configs/           Model and training configs
tests/             Unit tests
```

## Models

**Standard decoders** — U-Net, DeepLabV3+, FCN, UPerNet, SegFormer, DPT (via [smp](https://github.com/qubvel-org/segmentation_models.pytorch) / [timm](https://github.com/huggingface/pytorch-image-models))

**GFM encoders** — Clay, TerraFM, DINOv3, TerraMind, CROMA, DeCUR, DOFA, Prithvi, SatLAS, SoftCon, Galileo

**Custom** — DECODE (FracTAL ResUNet multi-task), SAM/SAM2 (instance segmentation), Mask2Former (panoptic segmentation), DelineateAnything (YOLO)

## Training

```bash
# GFM encoder (from images)
scripts/train_gfm.sh clay images_noaug

# GFM encoder (from precomputed features)
scripts/train_gfm.sh terrafm features /path/to/features

# Clay finetuning (encoder + decoder end-to-end)
scripts/train_clay.sh a online

# DECODE
scripts/train_gfm.sh decode images_noaug

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
scripts/eval_gfm.sh all main features /path/to/features

# Single model
scripts/eval_gfm.sh clay main images_noaug

# Clay finetuned
scripts/eval_clay.sh main

# Lightning CLI
python -m ftw_tools.cli model test \
  --model checkpoint.ckpt \
  --countries france \
  --test_split test \
  --input_type images \
  --dir ./data/ftw \
  --gpu 0 \
  --out results.json

# PRUE unified evaluation (SAM, DECODE, DelineateAnything, Mask2Former)
python scripts/run_model_inference.py --model decode --model_weights /path/to/ckpt --data_dir ./data/ftw --output_dir ./results
python scripts/evaluate_by_country.py --model_detections '{"decode": "./results/decode_detections_belgium.pkl"}' --data_dir ./data/ftw --output_dir ./results
```

## Feature extraction

Precompute embeddings for the full dataset:

```bash
python -m pretrained.models.compute_feats --model clay --batch_size 32
```

Per-model scripts in [`scripts/embeddings/`](scripts/embeddings/).

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FTW_DATA_DIR` / `FTW_DATA_ROOT` | `./data/ftw` | Dataset root |
| `GFM_CKPT_DIR` | `./gfm_ckpts/encoders` | Encoder checkpoints |
| `CLAY_CKPT_PATH` | *(required)* | Clay checkpoint for `scripts/eval_clay.sh` |
| `SAM2_CHECKPOINT_PATH` | *(required)* | SAM2 base checkpoint |
| `SAM2_MODEL_CFG` | `sam2_hiera_s.yaml` | SAM2 model config |
| `FTW_GEOPARQUET_ROOT` | *(required)* | Geoparquet root for `scripts/tools/ftw_to_coco.py` |

## Testing

```bash
pytest tests/ -v
```

## Citation

```bibtex
@misc{muhawenayo2026pruepracticalrecipefield,
      title={PRUE: A Practical Recipe for Field Boundary Segmentation at Scale}, 
      author={Gedeon Muhawenayo and Caleb Robinson and Subash Khanal and Zhanpei Fang and Isaac Corley and Alexander Wollam and Tianyi Gao and Leonard Strnad and Ryan Avery and Lyndon Estes and Ana M. Tárano and Nathan Jacobs and Hannah Kerner},
      year={2026},
      eprint={2603.27101},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.27101}, 
}
```

## License

[MIT](LICENSE)
