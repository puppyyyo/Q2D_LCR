#!/bin/bash

# Usage:
# bash run_ablation.sh <model> <crime_type> <dataset_version_base>

if [ $# -lt 3 ]; then
    echo "[ERROR] Usage: $0 <model> <crime_type> <dataset_version_base>"
    echo "Example: $0 m3 forgery v2"
    exit 1
fi

MODEL=$1
CRIME_TYPE=$2
VERSION_BASE=$3

SPLIT_DIR="data_num_ablation"
SIZES=(50 100 200 300 400 500 600)

for SIZE in "${SIZES[@]}"; do
    VERSION="${VERSION_BASE}_${SIZE}"
    echo "[INFO] Running fine-tuning for version: ${VERSION}"
    bash scripts/Finetuning_LICT.sh "$MODEL" "$CRIME_TYPE" "$VERSION" "$SPLIT_DIR"
done

echo "[INFO] All ablation runs completed."
