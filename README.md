# ftw-prue branch: Evaluation

Official code for the paper "PRUE: A Practical Recipe for Field Boundary Segmentation at Scale". 

## Harmonizing and evaluating model predictions

This branch is an effort to unify the different models tested for the bake-off into a format (referred to as Detections) which can then be evaluated in a unified manner. 

Current status: First commit, testing now. Mask2Former integration is in progress but is a bit delicate because the codebase is so different from torchgeo and the semantic-output models. 

## Code layout
This branch is laid out as follows:
```
    configs/ 
        model_name/config.yaml #, etc. settings for each model
    scripts/ # scripts to run from the command line for inference and evaluation
    src/ 
    tools/ # contains other tools for model profiling, namely throughput as reported in the paper
```

The overall workflow is: Model Output → Intermediate Format → Detections → Evaluator.

## Get started
Most users should interact with the project through the Python entry points in `scripts/` and the shared evaluation utilities in `src/`.

This is the overall workflow for evaluating the models included in the FTW bakeoff:
1. Run `run_model_inference.py` to generate model Detections (for whichever sets of weights you'd like to test) that are saved to a pkl file;
2. Run `evaluate_by_country.py` which loads GT masks from the dataset and compares model Detections against them to compute all or a subset of metrics (pixel, object, COCO). Object metrics use semantic masks with a simple connected components.

For more details see the readme in the `scripts/` directory. 

| Script | Purpose | Key arguments | Outputs |
| ------ | ------- | ------------- | ------- |
| `run_model_inference.py` | Standardizes inference across diverse model families. Handles dataset ordering, image preprocessing, and conversion to semantic/instance/panoptic intermediate classes before producing `Detections`. | `--model`, `--data_dir`, `--model_weights`, `--output_dir`, `--countries`, `--split`, `--batch_size`, `--temporal_options`. | `output_dir/{model}_detections_{countries}.pkl` plus optional raw logits. |
| `evaluate_by_country.py` | Loads GT masks from dataset, converts model Detections to binary masks, and computes metrics per country. Object metrics use semantic masks with connected components (matching FTW baseline). | `--model_detections` (JSON string or file), `--data_dir` (for GT masks and country/AOI mapping), `--countries`, `--metrics`, `--output_dir`, `--iou_threshold`, `--split`. | `output_dir/country_evaluation_results.json` and optional CSV. |




