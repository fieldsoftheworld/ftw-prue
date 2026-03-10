# FTW GFM Embedding Extraction

This directory contains scripts to extract embeddings from various foundation models for the FTW dataset.

## Prerequisites

1. Clone the Galileo repository:
   ```bash
   git clone git@github.com:nasaharvest/galileo.git
   ```

2. Install required dependencies (follow the installation instructions in the [ftw-baselines](https://github.com/fieldsoftheworld/ftw-baselines) and [Galileo](https://github.com/nasaharvest/galileo) repositories).

3. Download model checkpoints for each model you want to use. Follow each model's documentation:
   - **CROMA**: Download checkpoint to `ckpt_base_dir/croma`
   - **DeCur**: Download checkpoint to `ckpt_base_dir/decur`
   - **DOFA**: Download checkpoint to `ckpt_base_dir/dofa`
   - **Galileo**: Download checkpoint to `ckpt_base_dir/base`
   - **Prithvi**: Download checkpoint to `ckpt_base_dir/prithvi`
   - **Satlas**: Download checkpoint to `ckpt_base_dir/satlas`
   - **SoftCon**: Download checkpoint to `ckpt_base_dir/softcon`

## Configuration

Before running the scripts, update the paths in the `if __name__ == "__main__"` block of each script:

1. **base_dir**: Path to your FTW-Dataset directory
   ```python
   base_dir = Path("/path/to/FTW-Dataset")
   ```

2. **ckpt_base_dir**: Path to your baseline models directory
   ```python
   ckpt_base_dir = Path("/path/to/baseline_models")
   ```

## Usage

Run any of the extraction scripts:

```bash
python extract_ftw_croma_emb.py
python extract_ftw_decur_emb.py
python extract_ftw_dofa_emb.py
python extract_ftw_galileo_emb.py
python extract_ftw_prithvi_emb.py
python extract_ftw_satlas_emb.py
python extract_ftw_softcon_emb.py
```

Each script will:
- Process all countries from `ftw_tools.settings.ALL_COUNTRIES`
- Extract embeddings from Sentinel-2 images in `window_a` and `window_b` directories
- Save embeddings as `.npy` files in the corresponding output directories

## Output Structure

Embeddings are saved in directories parallel to the input structure:
- Input: `FTW-Dataset/ftw/{country}/s2_images/{window}/`
- Output: `FTW-{MODEL}-Embeddings/ftw/{country}/{model_name}/{window}/`

## Notes

- The scripts automatically skip files that have already been processed
- Failed files will print error messages without stopping the entire process
- Make sure you have sufficient disk space for the embedding files

