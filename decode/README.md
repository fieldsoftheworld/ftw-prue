# PRUE DECODE Experiment

Implementaions for training DECODE (pytorch version) on FTW dataset.

## Files and folders

- `fractal_resunet`: Package containing the FracTAL ResUNet model implementation
- `train_test.py`: Training script that trains the model and runs evaluation
- `eval.py`: Evaluation script for testing trained models
- `data_module.py`: Dataset class for loading multi-country field boundary data
- `visuals_and_inference.ipynb`: Jupyter notebook for visualization and inference on trained models


## Configuration

### Training Configuration

Before running training, create a `base_config.yaml` file in the same directory as `train_test.py`. The config file should contain:

- `experiment_name`: Name for the experiment
- `save_dir`: Directory where results will be saved
- `data`: Configuration for data loading
  - `root_dir`: Root directory containing country data
  - `countries`: List of countries to use
  - `n_classes`: Number of classes
  - `in_channels`: Number of input channels
  - `temporal_option`: Temporal option ("stacked", "windowA", or "windowB")
  - `crop_size`: Crop size as a list
  - `num_samples`: Number of samples (-1 for all)
  - `presence_only`: Boolean flag for presence-only handling
- `model`: Model configuration
  - `nfilters_init`: Initial number of filters
  - `depth`: Model depth
  - `ftdepth`: Feature depth
  - `psp_depth`: PSP depth
  - `norm_type`: Normalization type
  - `norm_groups`: Normalization groups
  - `nheads_start`: Number of attention heads
- `train`: Training configuration
  - `batch_size`: Batch size
  - `num_workers`: Number of data loading workers
  - `lr`: Learning rate
  - `num_epochs`: Number of training epochs
  - `patience`: Early stopping patience
- `loss`: Loss weights
  - `seg_weight`: Segmentation loss weight
  - `bound_weight`: Boundary loss weight
  - `dist_weight`: Distance loss weight

### Evaluation Configuration

For evaluation, the script reads from a config file at `logs-decode/exp0910-2classes-2win/config.yaml`. Update the path in `eval.py` (line 23) to point to your trained model's config file.

## Usage

### Training

To train the model:

```bash
python train_test.py
```

This will:
1. Load configuration from `base_config.yaml`
2. Create the experiment directory and save a copy of the config
3. Train the model for the specified number of epochs
4. Save the best model checkpoint to `{save_dir}/{experiment_name}/best_model.pth`
5. Generate loss plots and learning rate plots
6. Run evaluation on test data and save results to CSV

Outputs:
- `{save_dir}/{experiment_name}/best_model.pth`: Best model checkpoint
- `{save_dir}/{experiment_name}/train_loss.csv`: Training loss per epoch
- `{save_dir}/{experiment_name}/val_loss.csv`: Validation loss per epoch
- `{save_dir}/{experiment_name}/loss_plot.png`: Training/validation loss plot
- `{save_dir}/{experiment_name}/lr_plot.png`: Learning rate schedule plot
- `{save_dir}/{experiment_name}/test_results.csv`: Test evaluation results per country

### Evaluation

To evaluate a trained model:

```bash
python eval.py
```

This will:
1. Load the model configuration from the specified config path
2. Load the trained model checkpoint from `{save_dir}/{experiment_name}/best_model.pth`
3. Evaluate the model on test data for each country
4. Save results to `{save_dir}/{experiment_name}/test_results_final_flipped.csv`

The evaluation computes:
- Pixel-level metrics: IoU, precision, recall
- Object-level metrics: precision, recall

For presence-only countries (brazil, india, kenya, rwanda), only recall metrics are computed.


## Notes

- The model uses multi-task learning with segmentation, boundary, and distance prediction heads
- Presence-only countries are handled differently during evaluation (only recall is computed)
- Early stopping is implemented based on validation loss
- The training uses cosine annealing learning rate scheduling

