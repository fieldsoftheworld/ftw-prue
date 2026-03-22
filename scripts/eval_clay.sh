#!/bin/bash

# =========================================
# BASIC SETUP
# =========================================
# Resolve repo root (parent of scripts/) so Clay's "src" package is importable
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="${REPO_ROOT}/pretrained/models/clay:${PYTHONPATH:-}"

AGGREGATE_SCRIPT="scripts/aggregate.py"
RESULT_DIR_BASE="./results"

MODEL_NAME="clay-finetuned"
EXPR_TYPE=${1:-main}
COUNTRY_SPLIT=${COUNTRY_SPLIT:-test}
GPU=0

echo "🚀 Clay Full-Model Evaluation | Expr: $EXPR_TYPE | Input: images_noaug"
echo "   COCO metrics: prue_eval.eval_gfms → pycocotools COCOeval (segm); no prue_eval.evaluator"

# =========================================
# Clay FULL checkpoint (encoder + decoder)
# =========================================
CKPT_PATH="${CLAY_CKPT_PATH:?Set CLAY_CKPT_PATH to the Clay finetuned checkpoint}"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "❌ Clay checkpoint not found: $CKPT_PATH"
  exit 1
fi

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

mkdir -p logs "results/$MODEL_NAME"
LOG_FILE="logs/${MODEL_NAME}_${EXPR_TYPE}_images_noaug.log"

# =========================================
# Main evaluation loop
# =========================================
echo "📌 Evaluating Clay full model" | tee -a "$LOG_FILE"
model_start=$(date +%s)

for COUNTRY_NAME in "${FULL_DATA_COUNTRIES[@]}"; do
  country_start=$(date +%s)

  python -m prue_eval.eval_gfms \
    --model "$CKPT_PATH" \
    --backbone clay \
    --countries "$COUNTRY_NAME" \
    --test_split "$COUNTRY_SPLIT" \
    --input_type "images_noaug" \
    --dir "${FTW_DATA_DIR:-./data/ftw}" \
    --gpu "$GPU" \
    --model_predicts_3_classes \
    --test_on_3_classes \
    --out "results/$MODEL_NAME/${MODEL_NAME}_${COUNTRY_NAME}_${EXPR_TYPE}.json" \
    2>&1 | tee -a "$LOG_FILE"

  country_end=$(date +%s)
  runtime_country=$((country_end - country_start))
  printf "⏱️  Clay | %s => %dm %ds\n" "$COUNTRY_NAME" \
    $((runtime_country / 60)) $((runtime_country % 60)) | tee -a "$LOG_FILE"
done

model_end=$(date +%s)
runtime_model=$((model_end - model_start))
printf "✅ Finished Clay in %dm %ds\n\n" \
  $((runtime_model / 60)) $((runtime_model % 60)) | tee -a "$LOG_FILE"

# =========================================
# Aggregation
# =========================================
echo "📊 Running aggregation..."
python "$AGGREGATE_SCRIPT" \
  --model "$MODEL_NAME" \
  --result_dir "$RESULT_DIR_BASE" \
  --expr "$EXPR_TYPE"

echo "✨ Clay evaluation complete!"
