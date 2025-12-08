#!/bin/bash
set -e

# ==============================
# EXPERIMENT SELECTION
# ==============================
export EXP_NAME=${1:-clay}               # clay / terrafm / terramind / dinov3
export INPUT_TYPE=${2:-images_noaug}     # images_noaug | features
export FEAT_ROOT=${3:-""}                # required if input type = features

export WORK_DIR="$(pwd)"
export LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"
export CONFIG_PATH="$WORK_DIR/configs/release/3_class/vit.yaml"
export PROJECT="FTW-gfm"
ENV_NAME="ftw-prue"


echo "🚀 Running: EXP=$EXP_NAME   INPUT=$INPUT_TYPE"
echo "Working directory: $WORK_DIR"
echo "=============================="

# ==============================
# Model config lookup
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
    exit 1 ;;
esac

# ==============================
# Metadata / backbone behavior
# ==============================
if [ "$INPUT_TYPE" = "features" ]; then
    MODEL_TYPE="pretrained"
    BACKBONE="null"

    # require feat_root
    if [ -z "$FEAT_ROOT" ]; then
        echo "❌ ERROR: input_type=features requires feat_root path."
        exit 1
    fi

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
  --model.model_kwargs.hidden_dim "$hidden_dim" \
  --model.model_kwargs.patch_size "$patch" \
  --model.model_kwargs.original_input_size "$input_size" \
  $DATA_ARGS \
  --log_mode disabled \
  --project "$PROJECT" \
  --run_name "$EXP_NAME" \
  > "$LOG_DIR/${EXP_NAME}_${INPUT_TYPE}.log" 2>&1

echo "✅ Finished: $EXP_NAME with $INPUT_TYPE"


# ==============================
# Example usage:

# cd ftw-prue
# ./train_gfm.sh clay images_noaug
# ./train_gfm.sh clay features /path-to/precomputed_feats/clay

# ==============================