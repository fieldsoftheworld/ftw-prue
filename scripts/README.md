# Scripts for PRUE inference and evaluation

## 1. Prerequisites

- **Environment**: activate the Conda/venv that contains the FTW baselines dependencies (`pip install -r requirements.txt`). PyTorch with CUDA is
  required for most models.
- **Data layout**: point `--data_dir` at the FTW dataset root that contains folders such as `belgium/label_masks/...` and the `chips_{country}.parquet` metadata files.
- **Working directories**:
  - `eval_outputs/model_detections/*.pkl` for model predictions.
  - `eval_outputs/results/` for per-country metrics, JSON summaries, and optional plots.

> Tip: you might consider writing shell scripts which source a `shell/common_setup.sh` to export these paths for your convenience. When running the Python scripts directly you pass the exact paths via CLI flags, so no repository modifications are required.

---

## 2. Workflow

1. **Run model inference**
   ```bash
   python scripts/run_model_inference.py \
     --model ftw \
     --data_dir /path/to/ftw/data \
     --model_weights /path/to/checkpoint.ckpt \
     --output_dir eval_outputs/model_detections \
     --countries france spain germany \
     --split test
   ```
   Loads the requested model (DECODE, SAM variants, etc.), runs
   inference with consistent AOI ordering, converts outputs to the unified
   `Detections` format, and stores them. Use `--countries all` to process all
   countries.

2. **Evaluate by country**
   ```bash
   python scripts/evaluate_by_country.py \
     --model_detections '{"baseline": "eval_outputs/model_detections/ftw_detections_france-spain-germany.pkl"}' \
     --output_dir eval_outputs/results \
     --data_dir /path/to/ftw/data \
     --countries france spain germany \
     --metrics pixel object coco
   ```
   Loads GT semantic masks directly from the dataset, converts model Detections to binary masks for comparison, and computes metrics per
   country and overall. Object metrics use semantic masks with connected components,
   matching the FTW baseline methodology (`ftw_tools.training.metrics.get_object_level_metrics`).
   The `--model_detections` argument can be a JSON string mapping model names to detection
   file paths, or a path to a JSON file. Use `--countries all` to process all countries.

---
## Troubleshooting

- **NaN COCO metrics**: ensure predictions were generated with the same `--countries`
  and `--split` arguments as used during evaluation.
- **Mismatched sample counts**: regenerate predictions with the same `--countries`,
  `--split`, and filtering flags as used during evaluation.
- **Slow evaluation**: pass `--metrics pixel object` to skip COCO, or limit the
  `--countries` list. COCO AP requires encoding masks to RLE and can be memory
  intensive.