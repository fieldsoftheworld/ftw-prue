#!/bin/bash

# List of countries
FULL_DATA_COUNTRIES=(
  "austria"
  "belgium"
  "cambodia"
  "corsica"
  "croatia"
  "denmark"
  "estonia"
  "finland"
  "france"
  "germany"
  "latvia"
  "lithuania"
  "luxembourg"
  "netherlands"
  "slovakia"
  "slovenia"
  "south_africa"
  "spain"
  "sweden"
  "vietnam"
  "portugal"
)

# Define model dictionary (associative array)
declare -A model_dict=(
  [softcon]="kqvwszui"
  [prithvi]="7wdd3xb1"
  [dinov3]="b1dnr4uo"
  [dofa]="sjax08yt"
  [decur]="pv20m92t"
  [croma]="11zt5mr4"
  [clay]="85p9vk4c"
  [terrafm]="kh04zjm6"
  [galileo]="y7au2hz7"
  [satlas]="k7jkdsud"
  [terramind]="1tso7vvj"
)

# Loop over models and countries
mkdir -p logs results

for MODEL_NAME in "${!model_dict[@]}"; do
  MODEL_ID=${model_dict[$MODEL_NAME]}
  LOG_FILE="logs/${MODEL_NAME}.log"

  echo "======================================" | tee -a "$LOG_FILE"
  echo "Running model: $MODEL_NAME (ID: $MODEL_ID)" | tee -a "$LOG_FILE"
  echo "======================================" | tee -a "$LOG_FILE"

  model_start=$(date +%s)

  for COUNTRY_NAME in "${FULL_DATA_COUNTRIES[@]}"; do
    echo "----> Testing on country: $COUNTRY_NAME" | tee -a "$LOG_FILE"

    country_start=$(date +%s)

    python -m ftw_tools.cli model test \
      --model /projects/benq/ckpts/FTW-pretrained-ablation-final/${MODEL_ID}/checkpoints/last.ckpt \
      --countries ${COUNTRY_NAME} \
      --test_split test \
      --input_type "features" \
      --dir /projects/benq/ftw-data/data/ftw \
      --gpu 0 \
      --feat_root /projects/benq/ftw-data/precomputed_feats/${MODEL_NAME} \
      --out results/${MODEL_NAME}_${COUNTRY_NAME}.json \
      --model_predicts_3_classes --test_on_3_classes \
      2>&1 | tee -a "$LOG_FILE"

    country_end=$(date +%s)
    runtime_country=$((country_end - country_start))
    printf "⏱️  Time for %s (%s): %dm %ds\n\n" "$COUNTRY_NAME" "$MODEL_NAME" $((runtime_country / 60)) $((runtime_country % 60)) | tee -a "$LOG_FILE"
  done

  model_end=$(date +%s)
  runtime_model=$((model_end - model_start))
  printf "✅ Total time for model %s: %dm %ds\n\n" "$MODEL_NAME" $((runtime_model / 60)) $((runtime_model % 60)) | tee -a "$LOG_FILE"
done

echo "All experiments completed!"