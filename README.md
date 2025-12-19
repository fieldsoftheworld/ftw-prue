# 🌾 ftw-prue: Field Boundary Segmentation

Official repository for "PRUE: A Practical Recipe for Field Boundary Segmentation at Scale".

This branch focuses on experiments related to GFM (Geospatial Feature Matching) described in the paper.

---

## 💾 Data Setup

1. Download the FTW (Fields of the World) dataset by following instructions in the [ftw-baselines repository](https://github.com/fieldsoftheworld/ftw-baselines).
2. Place the dataset under the project’s `./data` directory.

---

## 🛠️ Environment Setup

Create and activate the Conda environment using the provided env.yaml:

conda create -f env.yaml  
conda activate ftw

---

## 🚀 Training

Training is performed via the ./train_gfm.sh script.

Command structure:

`./train_gfm.sh <model_filter> <input_type> [<feat_root_base>] [<wandb_mode>]`

Arguments:

- `<input_type>`: Specifies the type of input data.  
    • `"images_noaug"` → use raw images from the FTW dataset  
    • `"features"` → use precomputed embeddings  
- `<feat_root_base>`: Path where precomputed features are stored (required only when using `input_type="features"`).

(💡 For full details including model list, overrides, and logging options, inspect `train_gfm.sh`.)

---

## 📊 Evaluation

To run evaluation you must first download **both encoders and decoders** used in the GFM experiments.

Each model has **its own decoder directory** under:  
- gfm_ckpts/decoders/main/<model_name>/  
- gfm_ckpts/decoders/supp/<model_name>/


---

Evaluation is run using:

`./eval_gfm.sh <model_filter> <experiment> <input_type> [<feat_root_base>]`

Arguments:

- `<model_filter>`: Which models to evaluate.  
    • `"all"` or a specific model like `"clay"`  
- `<experiment>`: Which experiment configuration to load.  
    • `"main"` or `"supp"`  
- `<input_type>`: Input type to evaluate on.  
    • `"images_noaug"` or `"features"`  
- `<feat_root_base>`: Directory containing precomputed features (required only when `input_type="features"`).

(💡 Refer to `eval_gfm.sh` for full configuration details and usage examples.)
