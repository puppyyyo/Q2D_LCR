#!/bin/bash
# Script: run.sh
# Program:
#       Process legal judgments into LICT format.
#       Supports two dataset versions:
#         - v1: W/O CSA, uses `format_by_rand_filter.py`
#         - v2: W/ CSA, uses `format_by_rand_vaild.py`
# Usage:
#       bash task/lict_refactor/run.sh <crime_type> <dataset_version: v1|v2>
# Example:
#       bash task/lict_refactor/run.sh larceny v1       # Process without CSA
#       bash task/lict_refactor/run.sh larceny v2       # Process with CSA
# History:
#       2025/05/29  First release

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "[ERROR] Usage: bash $0 <crime_type> <dataset_version: v1|v2>"
    exit 1
fi

CRIME_TYPE=$1
VERSION=$2
SCRIPT_DIR="task/lict_data_construct"

# Select formatter by version
case "$VERSION" in
    v1) FORMAT_SCRIPT="format_by_rand_filter.py" ;;
    v2) FORMAT_SCRIPT="format_by_rand_vaild.py" ;;
    *)
        echo "[ERROR] Invalid dataset_version: $VERSION. Use 'v1' or 'v2'."
        exit 1
        ;;
esac

# Run processing
python "${SCRIPT_DIR}/construct_chunks.py" --crime_type "$CRIME_TYPE" --version "$VERSION"
python "${SCRIPT_DIR}/${FORMAT_SCRIPT}" --crime_type "$CRIME_TYPE" --version "$VERSION"