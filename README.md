# SAM Running instructions

Below are instructions on running the training and eval scripts.

# Training Script Usage

The `train_sam.sh` script runs SAM training using a YAML config file.

## Basic Usage

```bash
./train_sam.sh
```

This uses the default config:

```
configs/sam/e2e_config.yaml
```

## Override the Config File

```bash
./train_sam.sh --config path/to/your_config.yaml
```

## Output

Training output, checkpoints, and logs depend on the settings inside the YAML
configuration and the SAMTrainer implementation.


# Evaluation Script Usage

The `run_full_eval.sh` script runs the full evaluation pipeline, including:
1. Model inference (producing model detections)
2. Evaluation using those detections

## Basic Usage

```bash
./run_full_eval.sh
```

## Arguments

Override any parameters:

```bash
./run_full_eval.sh \
  --data_dir /path/to/data \
  --countries "us uk fr" \
  --eval_base_dir my_eval \
  --model_weights /path/to/checkpoint.ckpt \
  --model sam \
  --split test
```

### Supported Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--data_dir PATH` | Path to dataset | `/data` |
| `--countries LIST` | Space-separated list of countries used in both steps | `all` |
| `--eval_base_dir PATH` | Base directory for outputs | `eval_outputs` |
| `--model_weights PATH` | Model checkpoint path | `/checkpoint/checkpoint.ckpt` |
| `--model NAME` | Model name | `sam` |
| `--split NAME` | Dataset split | `test` |

## Output Locations

- Model detections:  
  `<eval_base_dir>/model_detections/`
- Evaluation results:  
  `<eval_base_dir>/results/`

Both directories are created automatically.

