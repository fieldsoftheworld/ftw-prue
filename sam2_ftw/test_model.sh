#!/bin/bash
# Quick test script for SAM-2 FTW model

CHECKPOINT_PATH="${1:-logs/sam2-ftw-3-class/checkpoints/last.ckpt}"
DATA_DIR="${2:-/u/gmuhawenayo/datasets/FTW-Dataset/ftw}"
OUTPUT_FILE="${3:-sam2_ftw/test_results.json}"
GPU="${4:--1}"  # -1 for CPU, 0 for GPU

echo "Testing SAM-2 model..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data directory: $DATA_DIR"
echo "Output: $OUTPUT_FILE"
echo "GPU: $GPU"
echo ""

python -m ftw_tools.cli model test \
  --model "$CHECKPOINT_PATH" \
  --countries france \
  --test_split test \
  --input_type images \
  --temporal_options sam2 \
  --dir "$DATA_DIR" \
  --gpu "$GPU" \
  --out "$OUTPUT_FILE"

echo ""
echo "Test complete! Results saved to: $OUTPUT_FILE"

