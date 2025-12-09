#!/bin/bash
set -e

# =========================================
# ARGUMENTS (NEW ORDER)
# =========================================
# 1. model filter: "all" (default), or specific model name
export MODEL_FILTER=${1:-all}

# Normalize: treat "all" as no filter
if [[ "$MODEL_FILTER" == "all" ]]; then
    MODEL_FILTER=""
fi

# 2. experiment type (main | supp)
export EXPR_TYPE=${2:-main}

# 3. input type (features | images_noaug)
export INPUT_TYPE=${3:-features}

# 4. feature root required only for features mode
if [[ "$INPUT_TYPE" == "features" ]]; then
    export FEAT_ROOT_BASE=${4:?❌ ERROR: features mode requires 4th argument = feat_root_base}
else
    export FEAT_ROOT_BASE=""
fi

# Default split = test unless overridden externally
export COUNTRY_SPLIT=${COUNTRY_SPLIT:-test}

GPU=0

echo "========================================="
echo "🚀 GFM Evaluation"
echo " Model filter : ${MODEL_FILTER:-ALL}"
echo " Expr type    : $EXPR_TYPE"
echo " Input type   : $INPUT_TYPE"
echo " Country split: $COUNTRY_SPLIT"
echo " GPU          : $GPU"
echo " Feat root    : ${FEAT_ROOT_BASE:-NONE}"
echo "========================================="


# =========================================
# Pretrained encoder weight lookup (MIRRORED FROM TRAIN SCRIPT)
# =========================================
ENCODER_DIR="gfm_ckpts/encoders"

get_encoder_ckpt() {
    local name="$1"
    case "$name" in
        clay)      echo "$ENCODER_DIR/clay-v1.5.ckpt" ;;
        terrafm)   echo "$ENCODER_DIR/TerraFM-B.pth" ;;
        dinov3)    echo "$ENCODER_DIR/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" ;;
        terramind) echo "null" ;;  # No ckpt
        *)         echo "null" ;;
    esac
}


# =========================================
# Country list
# =========================================
FULL_DATA_COUNTRIES=(
  "austria" "belgium" "cambodia" "corsica" "croatia"
  "denmark" "estonia" "finland" "france" "germany"
  "latvia" "lithuania" "luxembourg" "netherlands"
  "slovakia" "slovenia" "south_africa" "spain"
  "sweden" "vietnam" "portugal"
)

# =========================================
# Decoder checkpoint map
# =========================================
DECODER_DIR="gfm_ckpts/decoders/$EXPR_TYPE"

declare -A ckpt_map=(
  [softcon]="$DECODER_DIR/softcon/last.ckpt"
  [prithvi]="$DECODER_DIR/prithvi/last.ckpt"
  [dinov3]="$DECODER_DIR/dinov3/last.ckpt"
  [dofa]="$DECODER_DIR/dofa/last.ckpt"
  [decur]="$DECODER_DIR/decur/last.ckpt"
  [croma]="$DECODER_DIR/croma/last.ckpt"
  [clay]="$DECODER_DIR/clay/last.ckpt"
  [terrafm]="$DECODER_DIR/terrafm/last.ckpt"
  [galileo]="$DECODER_DIR/galileo/last.ckpt"
  [satlas]="$DECODER_DIR/satlas/last.ckpt"
  [terramind]="$DECODER_DIR/terramind/last.ckpt"
)

mkdir -p logs


# =========================================
# Model Loop
# =========================================
for MODEL_NAME in "${!ckpt_map[@]}"; do

  # optional model filter
  if [[ -n "$MODEL_FILTER" && "$MODEL_FILTER" != "$MODEL_NAME" ]]; then
      continue
  fi

  CKPT_PATH="${ckpt_map[$MODEL_NAME]}"

  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "⚠️ Skipping $MODEL_NAME — missing checkpoint: $CKPT_PATH"
    continue
  fi

  # Encoder weights for this backbone
  ENCODER_CKPT_PATH=$(get_encoder_ckpt "$MODEL_NAME")

  echo "→ Encoder ckpt for $MODEL_NAME = $ENCODER_CKPT_PATH"

  mkdir -p "results/$MODEL_NAME"
  LOG_FILE="logs/${MODEL_NAME}_${EXPR_TYPE}_${INPUT_TYPE}.log"

  echo "======================================" | tee -a "$LOG_FILE"
  echo "📌 Evaluating model: $MODEL_NAME" | tee -a "$LOG_FILE"
  echo "Checkpoint: $CKPT_PATH" | tee -a "$LOG_FILE"
  echo "======================================" | tee -a "$LOG_FILE"

  model_start=$(date +%s)

  for COUNTRY_NAME in "${FULL_DATA_COUNTRIES[@]}"; do
    echo "--> Country: $COUNTRY_NAME" | tee -a "$LOG_FILE"
    country_start=$(date +%s)

    if [[ "$INPUT_TYPE" == "features" ]]; then

      python -m ftw_tools.cli model test \
        --model "$CKPT_PATH" \
        --countries "$COUNTRY_NAME" \
        --test_split "$COUNTRY_SPLIT" \
        --input_type "features" \
        --dir ./data/ftw \
        --gpu "$GPU" \
        --feat_root "$FEAT_ROOT_BASE/$MODEL_NAME" \
        --encoder_ckpt_path "$ENCODER_CKPT_PATH" \
        --backbone "$MODEL_NAME" \
        --model_predicts_3_classes --test_on_3_classes \
        --out results/$MODEL_NAME/${MODEL_NAME}_${COUNTRY_NAME}_${EXPR_TYPE}.json \
        2>&1 | tee -a "$LOG_FILE"

    else

      python -m ftw_tools.cli model test \
        --model "$CKPT_PATH" \
        --backbone "$MODEL_NAME" \
        --encoder_ckpt_path "$ENCODER_CKPT_PATH" \
        --countries "$COUNTRY_NAME" \
        --test_split "$COUNTRY_SPLIT" \
        --input_type "images_noaug" \
        --dir ./data/ftw \
        --gpu "$GPU" \
        --model_predicts_3_classes --test_on_3_classes \
        --out results/$MODEL_NAME/${MODEL_NAME}_${COUNTRY_NAME}_${EXPR_TYPE}.json \
        2>&1 | tee -a "$LOG_FILE"
    fi

    country_end=$(date +%s)
    runtime_country=$((country_end - country_start))
    printf "⏱️  %s | %s => %dm %ds\n" "$MODEL_NAME" "$COUNTRY_NAME" \
      $((runtime_country / 60)) $((runtime_country % 60)) | tee -a "$LOG_FILE"
  done

  model_end=$(date +%s)
  runtime_model=$((model_end - model_start))
  printf "✅ Finished %s in %dm %ds\n\n" "$MODEL_NAME" \
    $((runtime_model / 60)) $((runtime_model % 60)) | tee -a "$LOG_FILE"

done

echo "✨ All evaluations complete!"


# ==========================================================
# EXAMPLE USAGE
# ==========================================================
# Evaluate ALL models on FEATURES:
#   ./eval_gfm.sh all main features /path/to/feat_root
#
# Evaluate ONE model (e.g., clay) on FEATURES:
#   ./eval_gfm.sh clay main features /path/to/feat_root
#
# Evaluate ALL models on IMAGES:
#   ./eval_gfm.sh all main images_noaug
#
# Evaluate ONE model (e.g., dinov3) on IMAGES:
#   ./eval_gfm.sh dinov3 main images_noaug
#
# Arg order:
#   1. model_filter   (all | clay | terrafm | ...)
#   2. experiment     (main | supp)
#   3. input_type     (features | images_noaug)
#   4. feat_root_base (required only when input_type=features)
# ==========================================================
