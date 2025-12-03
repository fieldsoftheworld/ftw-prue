#!/bin/bash
#SBATCH --account=bdbk-tgirails
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=ftw-train  # overridden by sbatch --job-name

# ==============================
# Setup
# ==============================
set -e
export EXP_NAME=${SLURM_JOB_NAME:-ftw-train}
export WORK_DIR="/u/subashk/storage/ftw-ablation/FTW-Bakeoff/ftw-baselines-2"
export LOG_DIR="$WORK_DIR/logs"
export CONFIG_PATH="$WORK_DIR/configs/release/3_class/vit.yaml"
PROJECT="FTW-pretrained-ablation-final"

cd "$WORK_DIR"
source /u/subashk/miniconda3/bin/activate ftw

echo "🚀 Starting GPU training for: $EXP_NAME"
echo "Working directory: $WORK_DIR"
echo "Logs → $LOG_DIR/${EXP_NAME}.log"
echo "=============================="

# ==============================
# Per-model configuration
# ==============================
case $EXP_NAME in
  clay)
    hidden_dim=1024; patch=8;  input_size=256 ;;
  croma)
    hidden_dim=768;  patch=8;  input_size=120 ;;
  decur)
    hidden_dim=384;  patch=16; input_size=224 ;;
  dofa)
    hidden_dim=1024; patch=16; input_size=224 ;;
  dinov3)
    hidden_dim=1024; patch=16; input_size=224 ;;
  galileo)
    hidden_dim=768;  patch=4;  input_size=256 ;;
  prithvi)
    hidden_dim=1024; patch=16; input_size=224 ;;
  satlas)
    hidden_dim=768;  patch=16; input_size=256 ;;
  softcon)
    hidden_dim=384;  patch=14; input_size=224 ;;
  terrafm)
    hidden_dim=768;  patch=16; input_size=224 ;;
  terramind)
    hidden_dim=768;  patch=16; input_size=224 ;;
  *)
    echo "❌ Unknown experiment: $EXP_NAME"
    echo "Valid options: clay | croma | decur | dofa | dinov3 | galileo | prithvi | satlas | softcon | terrafm | terramind"
    exit 1 ;;
esac

# ==============================
# Launch training
# ==============================
python -m ftw_tools.cli model fit \
  --config "$CONFIG_PATH" -- \
  --model.model_kwargs.hidden_dim "$hidden_dim" \
  --model.model_kwargs.patch_size "$patch" \
  --model.model_kwargs.original_input_size "$input_size" \
  --data.dict_kwargs.feat_root "/projects/benq/ftw-data/precomputed_feats/${EXP_NAME}" \
  --log_mode online \
  --project "$PROJECT" \
  --run_name "$EXP_NAME" \
  > "$LOG_DIR/${EXP_NAME}.log" 2>&1

echo "✅ Training completed for: $EXP_NAME"
