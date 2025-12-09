#!/bin/bash
set -e

# =========================================
# ARGUMENTS
# =========================================
export EXPR_TYPE=${1:-main}          # main | supp
export INPUT_TYPE=${2:-features}     # features | images_noaug

# If features → 3rd = FEAT_ROOT_BASE, 4th = MODEL_FILTER
if [[ "$INPUT_TYPE" == "features" ]]; then
    export FEAT_ROOT_BASE=${3:?❌ ERROR: features mode requires 3rd argument = feat_root_base}
    export MODEL_FILTER=${4:-""}
else
    export FEAT_ROOT_BASE=""
    export MODEL_FILTER=${3:-""}
fi

# Default split = test unless overridden externally
export COUNTRY_SPLIT=${COUNTRY_SPLIT:-test}

GPU=0

echo "========================================="
echo "🚀 GFM Evaluation"
echo " Expr type    : $EXPR_TYPE"
echo " Input type   : $INPUT_TYPE"
echo " Country split: $COUNTRY_SPLIT"
echo " GPU          : $GPU"
echo " Model filter : ${MODEL_FILTER:-ALL}"
echo " Feat root    : ${FEAT_ROOT_BASE:-NONE}"
echo "========================================="

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

  # optional filter
  if [[ -n "$MODEL_FILTER" && "$MODEL_FILTER" != "$MODEL_NAME" ]]; then
      continue
  fi

  CKPT_PATH="${ckpt_map[$MODEL_NAME]}"

  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "⚠️ Skipping $MODEL_NAME — missing checkpoint: $CKPT_PATH"
    continue
  fi

  # Only create results directory if we're actually evaluating this model
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
        --out results/$MODEL_NAME/${MODEL_NAME}_${COUNTRY_NAME}_${EXPR_TYPE}.json \
        --model_predicts_3_classes --test_on_3_classes \
        2>&1 | tee -a "$LOG_FILE"

    else

      python -m ftw_tools.cli model test \
        --model "$CKPT_PATH" \
        --countries "$COUNTRY_NAME" \
        --test_split "$COUNTRY_SPLIT" \
        --input_type "images_noaug" \
        --dir ./data/ftw \
        --gpu "$GPU" \
        --out results/$MODEL_NAME/${MODEL_NAME}_${COUNTRY_NAME}_${EXPR_TYPE}.json \
        --model_predicts_3_classes --test_on_3_classes \
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
# USAGE EXAMPLES
# ==========================================================
# All models, features:
#   ./eval_gfm.sh main features /path/to/precomputed_feats
#
# Single model (e.g., clay), features:
#   ./eval_gfm.sh main features /path/to/precomputed_feats clay
#
# All models, images:
#   ./eval_gfm.sh main images_noaug
#
# Single model (e.g., dinov3), images:
#   ./eval_gfm.sh main images_noaug dinov3
# ==========================================================
