#!/bin/bash
# Program:
#       This script is for fine-tuning the BGE model for legal vase retrieval.
# Usage:
#       bash 2.Finetuning.sh <model> <crime_type> <dataset_version> <split>
# Example:
#       bash 2.Finetuning.sh base larceny v1 full
# History:
#       2025/05/28  First realse

set -e
set -u
set -o pipefail

if [ $# -lt 4 ]; then
    echo "[ERROR] Usage: $0 <model> <crime_type> <dataset_version> <split>"
    echo "Example: $0 base larceny v1 full"
    exit 1
fi

MODEL=$1
CRIME_TYPE=$2
DATASET_VERSION=$3
SPLIT=$4

case "$MODEL" in
    "base")
        MODEL_TYPE="base"
        MODEL_NAME="BAAI/bge-base-zh-v1.5"
        TRAIN_MODULE="FlagEmbedding.finetune.embedder.encoder_only.base"
        QUERY_MAX_LEN="512"
        PASSAGE_MAX_LEN="512"
        ;;
    "large")
        MODEL_TYPE="large"
        MODEL_NAME="BAAI/bge-large-zh-v1.5"
        TRAIN_MODULE="FlagEmbedding.finetune.embedder.encoder_only.base"
        QUERY_MAX_LEN="512"
        PASSAGE_MAX_LEN="512"
        ;;
    "m3")
        MODEL_TYPE="m3"
        MODEL_NAME="BAAI/bge-m3"
        TRAIN_MODULE="FlagEmbedding.finetune.embedder.encoder_only.m3"
        QUERY_MAX_LEN="512"
        PASSAGE_MAX_LEN="512"
        ;;
    *)
        echo "[ERROR] Unsupported model type '$MODEL'. Please choose 'base', 'large', or 'm3'."
        exit 1
        ;;
esac

if ! command -v torchrun &> /dev/null; then
    echo "[ERROR] torchrun not found! Please install torch first."
    exit 1
fi

CACHE_DIR="./cache/model"
TRAIN_DATA="./dataset/lict_ft_data/${CRIME_TYPE}/${SPLIT}/${CRIME_TYPE}_judgment_${DATASET_VERSION}.json"
CACHE_PATH="./cache/data"
OUTPUT_DIR="./models/lict/${SPLIT}/${CRIME_TYPE}-${MODEL_TYPE}-lict_${DATASET_VERSION}"
DEEPSPEED_CONFIG="ds_config/ds_stage0.json"

EPOCHS="50"
BATCH_SIZE="16"
GRADIENT_ACCUMULATION_STEPS="1"
LEARNING_RATE="1e-5"
WARMUP_RATIO="0.1"
TRAIN_GROUP_SIZE="8"
NUM_GPUS=$(nvidia-smi -L | wc -l)
NPROC_PER_NODE=${NUM_GPUS:-2}

LOG_DIR="logs/lict/${CRIME_TYPE}/${SPLIT}"
STDOUT_LOG="${LOG_DIR}/${MODEL_TYPE}-${DATASET_VERSION}-train_stdout.log"
STDERR_LOG="${LOG_DIR}/${MODEL_TYPE}-${DATASET_VERSION}-train_stderr.log"

mkdir -p "$LOG_DIR"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "[ERROR] Training data file not found: $TRAIN_DATA"
    exit 1
fi

torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    -m "$TRAIN_MODULE" \
    --model_name_or_path "$MODEL_NAME" \
    --cache_dir "$CACHE_DIR" \
    --train_data "$TRAIN_DATA" \
    --cache_path "$CACHE_PATH" \
    --train_group_size "$TRAIN_GROUP_SIZE" \
    --query_max_len "$QUERY_MAX_LEN" \
    --passage_max_len "$PASSAGE_MAX_LEN" \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --learning_rate "$LEARNING_RATE" \
    --fp16 \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --dataloader_drop_last True \
    --warmup_ratio "$WARMUP_RATIO" \
    --gradient_checkpointing \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --logging_steps 100 \
    --save_strategy epoch \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
    --report_to tensorboard \
    > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)

echo "[INFO] Fine-tuning completed successfully! Logs saved in $LOG_DIR."

# # Step 1: Add .gitignore to model directory
# echo "checkpoint*/" > "${OUTPUT_DIR}/.gitignore"
# echo "*.ckpt" >> "${OUTPUT_DIR}/.gitignore"
# echo "[INFO] .gitignore added to ${OUTPUT_DIR}"

# # Step 2: Upload model to Hugging Face
# if ! command -v huggingface-cli &> /dev/null; then
#     echo "[ERROR] huggingface-cli not found! Please install it with 'pip install huggingface_hub'."
#     exit 1
# fi

# echo "[INFO] Uploading model to Hugging Face..."
# huggingface-cli upload "$OUTPUT_DIR" --repo-type model
# echo "[INFO] Model upload completed!"