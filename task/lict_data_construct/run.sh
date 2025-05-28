#!/bin/bash
# Program:
#       This script automates the processing of crime dataset versions.
#       It sequentilly runs `1.run_construct_dataset.py` and `2.run_format_data.py`.
# Usage:
#       bash run.sh <crime_type> <version>
# Example:
#       bash run.sh larceny v1
# History:
#       2025/05/12  First realse

set -e

export PYTHONPATH=$(pwd)

if [ $# -ne 2 ]; then
    echo "Usage: $0 <crime_type> <version>"
    echo "Example: $0 larceny v1"
    exit 1
fi

CRIME_TYPE=$1
VERSION=$2

echo "Processing crime_type: $CRIME_TYPE, version: $VERSION"

# Construct script mapping
SCRIPT_DIR="task/lict_data_construct"
CONSTRUCT_SCRIPT="$SCRIPT_DIR/1.construct_dataset_${VERSION}.py"

# Format script mapping
case $VERSION in
    "v1") FORMAT_SCRIPT="$SCRIPT_DIR/2.format_dataset_random_from_filter.py" ;;
    "v2") FORMAT_SCRIPT="$SCRIPT_DIR/2.format_dataset_random_from_vaild.py" ;;
    *) echo "Error: Unknown version '$VERSION'"; exit 1 ;;
esac


if [ ! -f "$CONSTRUCT_SCRIPT" ]; then
    echo "Error: $CONSTRUCT_SCRIPT not found!"
    exit 1
fi

if [ ! -f "$FORMAT_SCRIPT" ]; then
    echo "Error: $FORMAT_SCRIPT not found!"
    exit 1
fi

# 執行 Construct Dataset
echo "Running $CONSTRUCT_SCRIPT..."
python "$CONSTRUCT_SCRIPT" --crime_type "$CRIME_TYPE" --version "$VERSION"

# 執行 Format Dataset
echo "Running $FORMAT_SCRIPT..."
python "$FORMAT_SCRIPT" --crime_type "$CRIME_TYPE" --version "$VERSION"

echo "$CRIME_TYPE $VERSION process complete!"
