# DECODE Model for FTW Field Boundary Segmentation

DECODE (FracTAL ResUNet) implementation integrated with the FTW training pipeline.

## Model Architecture

The model uses FracTAL ResUNet with multi-task learning:
- **Segmentation head**: Predicts field boundaries
- **Boundary head**: Predicts boundary pixels
- **Distance head**: Predicts distance to boundaries

## Integration with FTW Pipeline

DECODE is now fully integrated into the FTW training pipeline and can be used like other models (unet, fcn, etc.).

### Training

Use the FTW CLI with a config file:

```bash
python -m ftw_tools.cli model fit --config decode/config_example.yaml
```

Or use the training script:

```bash
./train_gfm.sh decode images
```

### Evaluation

```bash
python -m ftw_tools.cli model test \
  --model <checkpoint_path> \
  --countries france \
  --test_split test \
  --input_type images \
  --dir ./data/ftw \
  --gpu 0 \
  --out results.json
```

## Configuration

See `config_example.yaml` for a complete example. Key parameters:

### Model Parameters (`model_kwargs`)

- `nfilters_init`: Initial number of filters (default: 32)
- `depth`: Model depth (default: 6)
- `ftdepth`: Feature depth (default: 5)
- `psp_depth`: PSP pooling depth (default: 4)
- `norm_type`: Normalization type ("BatchNorm" or "GroupNorm")
- `norm_groups`: Number of groups for GroupNorm (default: 4)
- `nheads_start`: Initial number of attention heads (default: 4)
- `seg_weight`: Segmentation loss weight (default: 1.0)
- `bound_weight`: Boundary loss weight (default: 1.0)
- `dist_weight`: Distance loss weight (default: 5.0)

### Model Settings

- `model`: Set to `"decode"`
- `loss`: Set to `"decode"` (uses MultiTaskLoss)
- `in_channels`: 4 for single window, 8 for stacked
- `num_classes`: 2 or 3
- `presence_only`: Boolean for presence-only handling

### Data Settings

The dataset automatically computes boundary and distance labels when using decode model.

## Model Outputs

DECODE returns three outputs:
1. **Segmentation logits**: `[B, num_classes, H, W]` - Main segmentation prediction
2. **Boundary logits**: `[B, num_classes, H, W]` - Boundary prediction
3. **Distance map**: `[B, 1, H, W]` - Distance to boundaries

For metrics and evaluation, only the segmentation output is used.

## Files

- `fractal_resunet/`: Model implementation
- `config_example.yaml`: Example configuration file
- `visuals_and_inference.ipynb`: Visualization notebook (optional)

## Notes

- The model uses multi-task learning with segmentation, boundary, and distance prediction
- Boundary and distance labels are automatically computed from masks
- Presence-only countries are handled via the `presence_only` parameter
- Checkpoints are saved in Lightning format and can be loaded via `CustomSemanticSegmentationTask.load_from_checkpoint()`
