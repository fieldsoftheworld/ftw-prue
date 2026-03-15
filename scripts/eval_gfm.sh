#!/bin/bash

# Resolve repo root so Clay's "src" package is importable (for clay / clay finetuned)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/pretrained/models/clay:${PYTHONPATH:-}"

AGGREGATE_SCRIPT="scripts/aggregate.py"
RESULT_DIR_BASE="./results"

export MODEL_FILTER=${1:-all}
if [[ "$MODEL_FILTER" == "all" ]]; then
    MODEL_FILTER=""
fi

export EXPR_TYPE=${2:-main}
export INPUT_TYPE=${3:-features}

if [[ "$INPUT_TYPE" == "features" ]]; then
    export FEAT_ROOT_BASE=${4:?ERROR: features mode requires 4th argument = feat_root_base}
else
    export FEAT_ROOT_BASE=""
fi

export COUNTRY_SPLIT=${COUNTRY_SPLIT:-test}

GPU=0

echo "GFM Evaluation | Model: ${MODEL_FILTER:-ALL} | Expr: $EXPR_TYPE | Input: $INPUT_TYPE"

ENCODER_DIR="gfm_ckpts/encoders"

get_encoder_ckpt() {
    local name="$1"
    case "$name" in
        clay)      echo "$ENCODER_DIR/clay-v1.5.ckpt" ;;
        terrafm)   echo "$ENCODER_DIR/TerraFM-B.pth" ;;
        dinov3)    echo "$ENCODER_DIR/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" ;;
        terramind) echo "null" ;;
        *)         echo "null" ;;
    esac
}

FULL_DATA_COUNTRIES=(
  "austria" "belgium" "cambodia" "corsica" "croatia"
  "denmark" "estonia" "finland" "france" "germany"
  "latvia" "lithuania" "luxembourg" "netherlands"
  "slovakia" "slovenia" "south_africa" "spain"
  "sweden" "vietnam" "portugal"
)

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

run_aggregation() {
    local model_name="$1"
    
    echo "Running aggregation for $model_name..."
    
    python "$AGGREGATE_SCRIPT" \
        --model "$model_name" \
        --result_dir "$RESULT_DIR_BASE" \
        --expr "$EXPR_TYPE"
}

for MODEL_NAME in "${!ckpt_map[@]}"; do

  if [[ -n "$MODEL_FILTER" && "$MODEL_FILTER" != "$MODEL_NAME" ]]; then
      continue
  fi

  CKPT_PATH="${ckpt_map[$MODEL_NAME]}"

  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Skipping $MODEL_NAME — missing checkpoint: $CKPT_PATH"
    continue
  fi

  ENCODER_CKPT_PATH=$(get_encoder_ckpt "$MODEL_NAME")

  mkdir -p "results/$MODEL_NAME"
  LOG_FILE="logs/${MODEL_NAME}_${EXPR_TYPE}_${INPUT_TYPE}.log"

  echo "Evaluating $MODEL_NAME" | tee -a "$LOG_FILE"
  model_start=$(date +%s)

  for COUNTRY_NAME in "${FULL_DATA_COUNTRIES[@]}"; do
    country_start=$(date +%s)

    DATA_DIR="${FTW_DATA_DIR:-./data/ftw}"
    if [[ "$INPUT_TYPE" == "features" ]]; then
      python -m ftw_tools.cli model test \
        --model "$CKPT_PATH" --countries "$COUNTRY_NAME" --test_split "$COUNTRY_SPLIT" \
        --input_type "features" --dir "$DATA_DIR" --gpu "$GPU" \
        --feat_root "$FEAT_ROOT_BASE/$MODEL_NAME" --encoder_ckpt_path "$ENCODER_CKPT_PATH" \
        --backbone "$MODEL_NAME" --model_predicts_3_classes --test_on_3_classes \
        --out results/$MODEL_NAME/${MODEL_NAME}_${COUNTRY_NAME}_${EXPR_TYPE}.json \
        2>&1 | tee -a "$LOG_FILE"

    else
      python -m ftw_tools.cli model test \
        --model "$CKPT_PATH" --backbone "$MODEL_NAME" --encoder_ckpt_path "$ENCODER_CKPT_PATH" \
        --countries "$COUNTRY_NAME" --test_split "$COUNTRY_SPLIT" \
        --input_type "images_noaug" --dir "$DATA_DIR" --gpu "$GPU" \
        --model_predicts_3_classes --test_on_3_classes \
        --out results/$MODEL_NAME/${MODEL_NAME}_${COUNTRY_NAME}_${EXPR_TYPE}.json \
        2>&1 | tee -a "$LOG_FILE"
    fi

    country_end=$(date +%s)
    runtime_country=$((country_end - country_start))
    printf "%s | %s => %dm %ds\n" "$MODEL_NAME" "$COUNTRY_NAME" \
      $((runtime_country / 60)) $((runtime_country % 60)) | tee -a "$LOG_FILE"
  done

  model_end=$(date +%s)
  runtime_model=$((model_end - model_start))
  printf "Finished %s in %dm %ds\n\n" "$MODEL_NAME" \
    $((runtime_model / 60)) $((runtime_model % 60)) | tee -a "$LOG_FILE"
    
  run_aggregation "$MODEL_NAME"

done

echo "All evaluations complete!"