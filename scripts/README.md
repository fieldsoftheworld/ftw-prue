# Harmonizing and evaluating model predictions with FTW-Bakeoff


Most collaborators should interact with the project through the Python entry points in `scripts/` and the shared evaluation utilities in `src/`. Start here!

This is the overall workflow for evaluating the models included in the FTW bakeoff (exact commands are below):
1. Run `run_model_inference.py` to generate model Detections (for whichever sets of weights you'd like to test) 
that are saved to a pkl file;
2. Run `evaluate_by_country.py` which loads GT masks directly from the dataset and compares model Detections 
against them to compute all or a subset of metrics (pixel, object, COCO). Object metrics use semantic masks 
with connected components, matching the FTW baseline methodology.

---

## 1. Prerequisites

- **Environment**: activate the Conda/venv that contains the FTW baselines
  dependencies (`pip install -r requirements.txt`). PyTorch with CUDA is
  required for most models.
- **Data layout**: point `--data_dir` at the FTW dataset root that contains
  folders such as `belgium/label_masks/...` and the `chips_{country}.parquet`
  metadata files.
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
   Loads the requested model (Mask2Former, DECODE, SAM variants, etc.), runs
   inference with consistent AOI ordering, converts outputs to the unified
   `Detections` format, and stores them. For Mask2Former, also provide
   `--config_file /path/to/config.yaml`. Use `--countries all` to process all
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
   Loads GT semantic masks directly from the dataset (no GT Detections pickle needed),
   converts model Detections to binary masks for comparison, and computes metrics per
   country and overall. Object metrics use semantic masks with connected components,
   matching the FTW baseline methodology (`ftw_tools.training.metrics.get_object_level_metrics`).
   The `--model_detections` argument can be a JSON string mapping model names to detection
   file paths, or a path to a JSON file. Use `--countries all` to process all countries.

---

## 3. Script Reference

| Script | Purpose | Key arguments | Outputs |
| ------ | ------- | ------------- | ------- |
| `run_model_inference.py` | Standardizes inference across diverse model families. Handles dataset ordering, image preprocessing, and conversion to semantic/instance/panoptic intermediate classes before producing `Detections`. | `--model`, `--data_dir`, `--model_weights`, `--output_dir`, `--countries`, `--split`, `--config_file` (for detectron2 models), `--batch_size`, `--temporal_options`. | `output_dir/{model}_detections_{countries}.pkl` plus optional raw logits. |
| `evaluate_by_country.py` | Loads GT masks directly from dataset, converts model Detections to binary masks, and computes metrics per country. Object metrics use semantic masks with connected components (matching FTW baseline). | `--model_detections` (JSON string or file), `--data_dir` (for GT masks and country/AOI mapping), `--countries`, `--metrics`, `--output_dir`, `--iou_threshold`, `--split`. | `output_dir/country_evaluation_results.json` and optional CSV. |


---

## 4. Relationship to `src/`

- `src/detections.py`: defines the core `Detections` data structure with mask,
  polygon, and COCO conversion helpers. All scripts serialize/deserialize this
  class, so keeping schema compatibility is critical.
- `src/evaluator.py`: houses the `Evaluator` class that computes pixel/object/
  COCO metrics. `evaluate_by_country.py` instantiates it and passes the masks
  or detections; shell scripts ultimately call the same code path.
- `src/converters.py` and `src/intermediate_formats.py`: adapters that turn
  raw model outputs (Mask2Former, SAM, DECODE, etc.) into the unified formats
  used downstream.
- `src/models/`: contains model-specific loading utilities referenced by
  `run_model_inference.py`.

If you need to extend the evaluation logic (e.g., add a new metric), update
`src/` first, then expose the option via the relevant script CLI.

---

## 5. Troubleshooting

- **NaN COCO metrics**: ensure predictions were generated with the same `--countries`
  and `--split` arguments as used during evaluation.
- **Mismatched sample counts**: regenerate predictions with the same `--countries`,
  `--split`, and filtering flags as used during evaluation.
- **Slow evaluation**: pass `--metrics pixel object` to skip COCO, or limit the
  `--countries` list. COCO AP requires encoding masks to RLE and can be memory
  intensive.
