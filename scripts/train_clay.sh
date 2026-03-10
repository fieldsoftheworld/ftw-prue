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
# ==========================================================

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
LOG_DIR="$WORK_DIR/logs"
CONFIG_PATH="$WORK_DIR/configs/release/3_class/vit.yaml"
PROJECT="FTW-gfm"

mkdir -p "$LOG_DIR"

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
# Data pipeline (Clay)
# ------------------------------
DATA_ARGS="
  --data.input_type images_noaug
  --data.preprocessing clay
  --data.dict_kwargs.metadata_path $WORK_DIR/configs/metadata.yaml
"

# ------------------------------
# Launch training
# ------------------------------
python -m ftw_tools.cli model fit \
  --config "$CONFIG_PATH" -- \
  --model.model gfm \
  --model.backbone clay \
  --model.weights "$WEIGHTS_PATH" \
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
