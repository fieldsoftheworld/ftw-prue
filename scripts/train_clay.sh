#!/bin/bash
set -e

# ==========================================================
# USAGE
# ==========================================================
#   ./train_clay.sh <key> [log_mode]
#
# KEYS → LR MAPPING:
#   a : 1e-5
#   b : 3e-5
#   c : 1e-4
#   d : 3e-4
#   e : 3e-3
#
# EXAMPLES:
#   ./train_clay.sh a online
#   FTW_DATA_DIR=/path/to/ftw ./train_clay.sh a online
# ==========================================================

# Run from repo root so config/data paths and PYTHONPATH are correct
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/pretrained/models/clay:${PYTHONPATH:-}"

# ------------------------------
# REQUIRED ARG
# ------------------------------
KEY="$1"
LOG_MODE="${2:-disabled}"

if [ -z "$KEY" ]; then
  echo "❌ ERROR: Provide sweep key: a b c d e"
  exit 1
fi

# ------------------------------
# LR MAP: {1,3} × {-5,-4,-3}
# ------------------------------
declare -A LR_MAP=(
  [a]=1e-5
  [b]=3e-5
  [c]=1e-4
  [d]=3e-4
  [e]=3e-3
)

LR="${LR_MAP[$KEY]}"

if [ -z "$LR" ]; then
  echo "❌ Invalid key: $KEY"
  echo "Valid keys: a b c d e"
  exit 1
fi

# ------------------------------
# FIXED SETTINGS (Clay only)
# ------------------------------
EXP_NAME="clay"
INPUT_TYPE="images_noaug"

WORK_DIR="$(pwd)"
DATA_ROOT="${FTW_DATA_DIR:-$WORK_DIR/data/ftw}"
LOG_DIR="$WORK_DIR/logs"
CONFIG_PATH="$WORK_DIR/configs/release/3_class/vit.yaml"
PROJECT="FTW-gfm"

mkdir -p "$LOG_DIR"

if [ ! -d "$DATA_ROOT" ]; then
  echo "❌ ERROR: Data dir not found: $DATA_ROOT (set FTW_DATA_DIR if needed)"
  exit 1
fi

echo "🚀 Clay finetuning (images_noaug)"
echo "🔑 Sweep key : $KEY"
echo "🔁 model.lr : $LR"
echo "📊 log_mode : $LOG_MODE"
echo "=============================="

# ------------------------------
# Encoder weights
# ------------------------------
ENCODER_DIR="${GFM_CKPT_DIR:-./gfm_ckpts/encoders}"
WEIGHTS_PATH="$ENCODER_DIR/clay/clay-v1.5.ckpt"

# ------------------------------
# Clay model params
# ------------------------------
hidden_dim=1024
patch=8
input_size=256

# ------------------------------
# Data pipeline (Clay, full finetuning: encoder + decoder, images_noaug)
# ------------------------------
DATA_ARGS="
  --data.dict_kwargs.root $DATA_ROOT
  --data.input_type images_noaug
  --data.preprocessing clay
  --data.dict_kwargs.metadata_path $WORK_DIR/configs/metadata.yaml
"

# ------------------------------
# Launch training (full finetuning: encoder + decoder, encoder NOT frozen)
# ------------------------------
python -m ftw_tools.cli model fit \
  --config "$CONFIG_PATH" -- \
  --model.model gfm \
  --model.backbone clay \
  --model.weights "$WEIGHTS_PATH" \
  --model.freeze_backbone false \
  --model.lr "$LR" \
  --model.model_kwargs.hidden_dim "$hidden_dim" \
  --model.model_kwargs.patch_size "$patch" \
  --model.model_kwargs.original_input_size "$input_size" \
  $DATA_ARGS \
  --log_mode "$LOG_MODE" \
  --project "$PROJECT" \
  --run_name "clay_${KEY}_lr${LR}" \
  > "$LOG_DIR/clay_images_noaug_${KEY}.log" 2>&1

echo "✅ Finished: key=$KEY (lr=$LR)"
