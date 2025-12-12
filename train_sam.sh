#!/usr/bin/env bash
set -euo pipefail

### --------------------------
### Default values
### --------------------------
CONFIG_FILE="configs/sam/e2e_config.yaml"

### --------------------------
### Argument parser
### --------------------------
usage() {
    echo "Usage: $0 [--config PATH]"
    echo
    echo "Defaults:"
    echo "  --config    $CONFIG_FILE"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG_FILE="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done

### --------------------------
### Run training
### --------------------------
echo "--------------------------------------"
echo "Starting SAM training"
echo "--------------------------------------"
echo "Using config: $CONFIG_FILE"
echo

python src/models/sam/train.py "$CONFIG_FILE"

echo
echo "Training complete."
