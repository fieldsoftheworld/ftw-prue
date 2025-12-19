#!/bin/bash

# ==============================
# ARGUMENTS
# ==============================
export EXP_NAME=${1:-clay}           # clay / terrafm / terramind / dinov3
export INPUT_TYPE=${2:-images_noaug} # images_noaug | features

# If input_type=features → 3rd arg is FEAT_ROOT
if [ "$INPUT_TYPE" = "features" ]; then
    export FEAT_ROOT=${3:?❌ ERROR: features mode requires 3rd argument = feat_root path}
    export LOG_MODE=${4:-disabled}
else
    export FEAT_ROOT=""
    export LOG_MODE=${3:-disabled}
fi

export WORK_DIR="$(pwd)"
export LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

export CONFIG_PATH="$WORK_DIR/configs/release/3_class/vit.yaml"
export PROJECT="FTW-gfm"

echo "🚀 Running EXP=$EXP_NAME | INPUT=$INPUT_TYPE | LOG_MODE=$LOG_MODE"
echo "Working directory: $WORK_DIR"
echo "=============================="

# ==============================
# Pretrained encoder weight lookup
# ==============================
ENCODER_DIR="$WORK_DIR/gfm_ckpts/encoders"

case "$EXP_NAME" in
    clay)      WEIGHTS_PATH="$ENCODER_DIR/clay-v1.5.ckpt" ;;
    terrafm)   WEIGHTS_PATH="$ENCODER_DIR/TerraFM-B.pth" ;;
    dinov3)    WEIGHTS_PATH="$ENCODER_DIR/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" ;;
    terramind) WEIGHTS_PATH="null" ;;  # TerraMind uses no ckpt
    *)         WEIGHTS_PATH="null" ;;
esac

echo "🧠 Encoder weights: $WEIGHTS_PATH"

# ==============================
# Model config lookup
# ==============================
case $EXP_NAME in
  clay)      hidden_dim=1024; patch=8;  input_size=256 ;;
  croma)     hidden_dim=768;  patch=8;  input_size=120 ;;
  decur)     hidden_dim=384;  patch=16; input_size=224 ;;
  dofa)      hidden_dim=1024; patch=16; input_size=224 ;;
  dinov3)    hidden_dim=1024; patch=16; input_size=224 ;;
  galileo)   hidden_dim=768;  patch=4;  input_size=256 ;;
  prithvi)   hidden_dim=1024; patch=16; input_size=224 ;;
  satlas)    hidden_dim=768;  patch=16; input_size=256 ;;
  softcon)   hidden_dim=384;  patch=14; input_size=224 ;;
  terrafm)   hidden_dim=768;  patch=16; input_size=224 ;;
  terramind) hidden_dim=768;  patch=16; input_size=224 ;;
  *)
    echo "❌ Unknown EXP_NAME=$EXP_NAME"
    exit 1 ;;
esac

# ==============================
# Metadata / data pipeline setup
# ==============================
if [ "$INPUT_TYPE" = "features" ]; then

    MODEL_TYPE="pretrained"
    BACKBONE="null"

    DATA_ARGS="
      --data.input_type features
      --data.dict_kwargs.feat_root $FEAT_ROOT
      --data.dict_kwargs.metadata_path null
      --data.preprocessing null
    "

else

    MODEL_TYPE="gfm"
    BACKBONE="$EXP_NAME"

    if [ "$EXP_NAME" = "clay" ]; then
        DATA_ARGS="
          --data.input_type images_noaug
          --data.preprocessing clay
          --data.dict_kwargs.metadata_path $WORK_DIR/configs/metadata.yaml
        "
    else
        DATA_ARGS="
          --data.input_type images_noaug
          --data.preprocessing $EXP_NAME
          --data.dict_kwargs.metadata_path null
        "
    fi

fi

# ==============================
# Launch Training
# ==============================
python -m ftw_tools.cli model fit \
  --config "$CONFIG_PATH" -- \
  --model.model "$MODEL_TYPE" \
  --model.backbone "$BACKBONE" \
  --model.weights "$WEIGHTS_PATH" \
  --model.model_kwargs.hidden_dim "$hidden_dim" \
  --model.model_kwargs.patch_size "$patch" \
  --model.model_kwargs.original_input_size "$input_size" \
  $DATA_ARGS \
  --log_mode "$LOG_MODE" \
  --project "$PROJECT" \
  --run_name "$EXP_NAME" \
  > "$LOG_DIR/${EXP_NAME}_${INPUT_TYPE}.log" 2>&1

echo "✅ Finished: $EXP_NAME with $INPUT_TYPE (log_mode=$LOG_MODE)"

# ==========================================================
# EXAMPLE USAGE
# ==========================================================
# Standard training w/o wandb
#   ./train_gfm.sh clay images_noaug
#
# Training with features
#   ./train_gfm.sh terrafm features /path/to/precomputed_feats/terrafm
#
# Enable wandb logging online:
#   ./train_gfm.sh clay images_noaug online
#   ./train_gfm.sh terrafm features /path/to/feats online
# ==========================================================
