#!/usr/bin/env bash
set -euo pipefail

### --------------------------
### Default values
### --------------------------
MODEL="sam"
DATA_DIR="data"
COUNTRIES="all"
MODEL_WEIGHTS="checkpoint/checkpoint.ckpt"
EVAL_BASE_DIR="eval_outputs"       # User can override
SPLIT="test"
METRICS="pixel object coco"

### --------------------------
### Argument parser
### --------------------------
usage() {
    echo "Usage: $0 [--data_dir PATH] [--countries LIST] [--eval_base_dir PATH] [--model_weights PATH] [--model NAME] [--split NAME]"
    echo
    echo "Defaults:"
    echo "  --data_dir         $DATA_DIR"
    echo "  --countries        $COUNTRIES"
    echo "  --eval_base_dir    $EVAL_BASE_DIR"
    echo "  --model_weights    $MODEL_WEIGHTS"
    echo "  --model            $MODEL"
    echo "  --split            $SPLIT"
    exit 1
}

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir) DATA_DIR="$2"; shift ;;
        --countries) COUNTRIES="$2"; shift ;;
        --eval_base_dir) EVAL_BASE_DIR="$2"; shift ;;
        --model_weights) MODEL_WEIGHTS="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --split) SPLIT="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done

### --------------------------
### Derived paths
### --------------------------
MODEL_DETECTIONS_DIR="${EVAL_BASE_DIR}/model_detections"
RESULTS_DIR="${EVAL_BASE_DIR}/results"

# Convert space-separated countries to a hyphen-joined token for the filename
IFS=' ' read -r -a COUNTRY_LIST <<< "$COUNTRIES"

if [ "${#COUNTRY_LIST[@]}" -gt 1 ]; then
    COUNTRY_TOKEN=$(printf "%s-" "${COUNTRY_LIST[@]}")
    COUNTRY_TOKEN="${COUNTRY_TOKEN%-}"   # remove trailing hyphen
else
    COUNTRY_TOKEN="${COUNTRY_LIST[0]}"
fi

DETECTIONS_FILE="${MODEL_DETECTIONS_DIR}/${MODEL}_detections_${COUNTRY_TOKEN}.pkl"


mkdir -p "$MODEL_DETECTIONS_DIR" "$RESULTS_DIR"

echo "--------------------------------------"
echo "Running full evaluation pipeline"
echo "--------------------------------------"
echo "Model:             $MODEL"
echo "Data dir:          $DATA_DIR"
echo "Countries:         $COUNTRIES"
echo "Eval base dir:     $EVAL_BASE_DIR"
echo "Model weights:     $MODEL_WEIGHTS"
echo "Split:             $SPLIT"
echo "Metrics:           $METRICS"
echo "--------------------------------------"
echo

### --------------------------
### Step 1: Model inference
### --------------------------
echo "Step 1: Running model inference..."
python scripts/run_model_inference.py \
    --model "$MODEL" \
    --data_dir "$DATA_DIR" \
    --model_weights "$MODEL_WEIGHTS" \
    --output_dir "$MODEL_DETECTIONS_DIR" \
    --countries "$COUNTRIES" \
    --split "$SPLIT"

echo "Model inference complete."
echo "Detections saved to: $DETECTIONS_FILE"
echo

### --------------------------
### Step 2: Evaluation
### --------------------------
echo "Step 2: Running evaluation..."
python scripts/evaluate_by_country.py \
    --model_detections "{\"$MODEL\": \"$DETECTIONS_FILE\"}" \
    --output_dir "$RESULTS_DIR" \
    --data_dir "$DATA_DIR" \
    --countries "$COUNTRIES" \
    --metrics $METRICS

echo "Evaluation complete."
echo "Results saved in: $RESULTS_DIR"
echo
echo "Pipeline finished successfully."
