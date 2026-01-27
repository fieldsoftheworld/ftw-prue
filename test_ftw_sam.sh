#!/bin/bash

#SBATCH --job-name=ftw_sam2
#SBATCH --output=logs/jobs/ftw_sam2_%j.out
#SBATCH --error=logs/jobs/ftw_sam2_%j.err
#SBATCH --time=0-2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmuhawenayo@asu.edu

# Activate environment
source ~/.bashrc
mamba activate ftw

cd /u/gmuhawenayo/projects/PRUE-CVPR/ftw-prue

# Configuration
export MODEL_PATH="/u/gmuhawenayo/projects/PRUE-CVPR/ftw-prue/logs/sam2-ftw-rebuttal/FTW-project/5294stag/checkpoints/last.ckpt"
export DATA_DIR="/u/gmuhawenayo/datasets/FTW-Dataset/ftw"
export GPU=0
export TEST_SPLIT="test"
export OUTPUT_DIR="sam2_ftw/results"

TRAIN_COUNTRIES=(
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
  "portugal"
  "slovakia"
  "slovenia"
  "south_africa"
  "spain"
  "sweden"
  "vietnam"
)

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/jobs
LOG_FILE="sam2_ftw/test_all_countries_${SLURM_JOB_ID}.log"
SUMMARY_CSV="$OUTPUT_DIR/sam2_all_countries_summary.csv"
COUNTRY_LOG_DIR="$OUTPUT_DIR/country_logs"
mkdir -p "$COUNTRY_LOG_DIR"

echo "SAM-2 Testing on All Training Countries" | tee "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Model: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Data directory: $DATA_DIR" | tee -a "$LOG_FILE"
echo "GPU: $GPU" | tee -a "$LOG_FILE"
echo "Test split: $TEST_SPLIT" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Initialize summary CSV header
echo "country,pixel_iou,pixel_precision,pixel_recall,object_precision,object_recall,object_f1" > "$SUMMARY_CSV"

model_start=$(date +%s)

for COUNTRY in "${TRAIN_COUNTRIES[@]}"; do
  country_start=$(date +%s)
  
  echo "Testing on $COUNTRY..." | tee -a "$LOG_FILE"
  
  OUTPUT_FILE="$OUTPUT_DIR/sam2_${COUNTRY}_${TEST_SPLIT}.json"
  COUNTRY_LOG="$COUNTRY_LOG_DIR/${COUNTRY}.log"
  
  # Run test and capture output
  python -m ftw_tools.cli model test \
    --model "$MODEL_PATH" \
    --countries "$COUNTRY" \
    --test_split "$TEST_SPLIT" \
    --input_type images \
    --temporal_options sam2 \
    --dir "$DATA_DIR" \
    --gpu "$GPU" \
    --out "$OUTPUT_FILE" \
    2>&1 | tee "$COUNTRY_LOG" | tee -a "$LOG_FILE"
  
  # Parse metrics from log output
  if grep -q "Pixel IoU (crop):" "$COUNTRY_LOG"; then
    PIXEL_IOU=$(grep "Pixel IoU (crop):" "$COUNTRY_LOG" | awk '{print $NF}')
    PIXEL_PREC=$(grep "Pixel Precision (crop):" "$COUNTRY_LOG" | awk '{print $NF}')
    PIXEL_RECALL=$(grep "Pixel Recall (crop):" "$COUNTRY_LOG" | awk '{print $NF}')
    OBJ_PREC=$(grep "Object Precision:" "$COUNTRY_LOG" | awk '{print $NF}')
    OBJ_RECALL=$(grep "Object Recall:" "$COUNTRY_LOG" | awk '{print $NF}')
    OBJ_F1=$(grep "Object F1:" "$COUNTRY_LOG" | awk '{print $NF}')
    
    if [ -n "$PIXEL_IOU" ] && [ -n "$PIXEL_PREC" ] && [ -n "$PIXEL_RECALL" ] && \
       [ -n "$OBJ_PREC" ] && [ -n "$OBJ_RECALL" ] && [ -n "$OBJ_F1" ]; then
      echo "$COUNTRY,$PIXEL_IOU,$PIXEL_PREC,$PIXEL_RECALL,$OBJ_PREC,$OBJ_RECALL,$OBJ_F1" >> "$SUMMARY_CSV"
      echo "OK: Parsed metrics for $COUNTRY" | tee -a "$LOG_FILE"
    else
      echo "WARN: Failed to parse all metrics for $COUNTRY" | tee -a "$LOG_FILE"
    fi
  else
    echo "WARN: No metrics found in output for $COUNTRY (may have failed)" | tee -a "$LOG_FILE"
  fi
  
  country_end=$(date +%s)
  runtime_country=$((country_end - country_start))
  printf "Completed %s in %dm %ds\n" "$COUNTRY" \
    $((runtime_country / 60)) $((runtime_country % 60)) | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
done

model_end=$(date +%s)
runtime_model=$((model_end - model_start))

# Calculate averages
echo "Calculating averages..." | tee -a "$LOG_FILE"

python3 << EOF
import csv
import numpy as np
import os

summary_file = "$SUMMARY_CSV"
avg_file = "$OUTPUT_DIR/sam2_all_countries_average.csv"

if not os.path.exists(summary_file) or os.path.getsize(summary_file) <= 20:
    print("No valid results found. Cannot calculate averages.")
    exit(1)

metrics = {
    'pixel_iou': [],
    'pixel_precision': [],
    'pixel_recall': [],
    'object_precision': [],
    'object_recall': [],
    'object_f1': []
}

with open(summary_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            metrics['pixel_iou'].append(float(row['pixel_iou']))
            metrics['pixel_precision'].append(float(row['pixel_precision']))
            metrics['pixel_recall'].append(float(row['pixel_recall']))
            metrics['object_precision'].append(float(row['object_precision']))
            metrics['object_recall'].append(float(row['object_recall']))
            metrics['object_f1'].append(float(row['object_f1']))
        except (ValueError, KeyError):
            print(f"Warning: Skipping invalid row: {row}")
            continue

if len(metrics['pixel_iou']) == 0:
    print("No valid metrics found. Cannot calculate averages.")
    exit(1)

avg_pixel_iou = np.mean(metrics['pixel_iou'])
avg_pixel_precision = np.mean(metrics['pixel_precision'])
avg_pixel_recall = np.mean(metrics['pixel_recall'])
avg_object_precision = np.mean(metrics['object_precision'])
avg_object_recall = np.mean(metrics['object_recall'])
avg_object_f1 = np.mean(metrics['object_f1'])

std_pixel_iou = np.std(metrics['pixel_iou'])
std_pixel_precision = np.std(metrics['pixel_precision'])
std_pixel_recall = np.std(metrics['pixel_recall'])
std_object_precision = np.std(metrics['object_precision'])
std_object_recall = np.std(metrics['object_recall'])
std_object_f1 = np.std(metrics['object_f1'])

with open(avg_file, 'w') as f:
    f.write("metric,mean,std\n")
    f.write(f"pixel_iou,{avg_pixel_iou:.4f},{std_pixel_iou:.4f}\n")
    f.write(f"pixel_precision,{avg_pixel_precision:.4f},{std_pixel_precision:.4f}\n")
    f.write(f"pixel_recall,{avg_pixel_recall:.4f},{std_pixel_recall:.4f}\n")
    f.write(f"object_precision,{avg_object_precision:.4f},{std_object_precision:.4f}\n")
    f.write(f"object_recall,{avg_object_recall:.4f},{std_object_recall:.4f}\n")
    f.write(f"object_f1,{avg_object_f1:.4f},{std_object_f1:.4f}\n")

print(f"\nAverage Results Across {len(metrics['pixel_iou'])} Countries:")
print(f"Pixel IoU (crop):        {avg_pixel_iou:.4f} ± {std_pixel_iou:.4f}")
print(f"Pixel Precision (crop):  {avg_pixel_precision:.4f} ± {std_pixel_precision:.4f}")
print(f"Pixel Recall (crop):     {avg_pixel_recall:.4f} ± {std_pixel_recall:.4f}")
print(f"Object Precision:        {avg_object_precision:.4f} ± {std_object_precision:.4f}")
print(f"Object Recall:           {avg_object_recall:.4f} ± {std_object_recall:.4f}")
print(f"Object F1:               {avg_object_f1:.4f} ± {std_object_f1:.4f}")
print(f"\nAverages saved to: {avg_file}")
EOF

printf "All testing completed in %dm %ds\n" \
  $((runtime_model / 60)) $((runtime_model % 60)) | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Summary CSV: $SUMMARY_CSV"
echo "Average results: $OUTPUT_DIR/sam2_all_countries_average.csv"
echo "Log file: $LOG_FILE"