#!/bin/bash
# Program:
#       This script is for fine-tuning the BGE model for legal vase retrieval.
# Usage:
#       bash 2.2.Finetuning_lcaet.sh <model> <crime_type> <lcaet_type> <split>
# Example:
#       bash 2.2.Finetuning_lcaet.sh m3 larceny full
# History:
#       2025/05/28  First realse

set -e
set -u
set -o pipefail

if [ $# -lt 4 ]; then
    echo "[ERROR] Usage: $0 <model> <crime_type> <lcaet_type> <split>"
    exit 1
fi

MODEL=$1
CRIME_TYPE=$2
LCAET_TYPE=$3
SPLIT=$4

case "$MODEL" in
    "base")
        MODEL_TYPE="base"

        # 1-stage, like paragraph-level and sentence-level
        MODEL_NAME="./models/lict/${SPLIT}/${CRIME_TYPE}-${MODEL_TYPE}-lict_v2"

        TRAIN_MODULE="FlagEmbedding.finetune.embedder.encoder_only.base"
        QUERY_MAX_LEN="512"
        PASSAGE_MAX_LEN="512"
        ;;
    "large")
        MODEL_TYPE="large"

        # 1-stage, like paragraph-level and sentence-level
        MODEL_NAME="./models/lict/${SPLIT}/${CRIME_TYPE}-${MODEL_TYPE}-lict_v2"

        TRAIN_MODULE="FlagEmbedding.finetune.embedder.encoder_only.base"
        QUERY_MAX_LEN="512"
        PASSAGE_MAX_LEN="512"
        ;;
    "m3")
        MODEL_TYPE="m3"      

        # 1-stage, like paragraph-level and sentence-level
        MODEL_NAME="./models/lict/${SPLIT}/${CRIME_TYPE}-${MODEL_TYPE}-lict_v2"

        TRAIN_MODULE="FlagEmbedding.finetune.embedder.encoder_only.m3"
        QUERY_MAX_LEN="2048"
        PASSAGE_MAX_LEN="2048"
        ;;
    *)
        echo "[ERROR] Unsupported model type '$MODEL'. Please choose 'base', 'large' or 'm3'."
        exit 1
        ;;
esac

if ! command -v torchrun &> /dev/null; then
    echo "[ERROR] torchrun not found! Please install torch first."
    exit 1
fi

CACHE_DIR="./cache/model"
TRAIN_DATA="./dataset/lcaet_ft_data/${LCAET_TYPE}/${SPLIT}/${CRIME_TYPE}_judgment.json"
CACHE_PATH="./cache/data"

# 1-stage, like paragraph-level and sentence-level
OUTPUT_DIR="./models/lcaet/${LCAET_TYPE}/${SPLIT}/${CRIME_TYPE}-${MODEL_TYPE}-lict_v2"
DEEPSPEED_CONFIG="ds_config/ds_stage1.json"

EPOCHS="3"
BATCH_SIZE="8"
GRADIENT_ACCUMULATION_STEPS="1"
LEARNING_RATE="1e-5"
WARMUP_RATIO="0.1"
TRAIN_GROUP_SIZE="2"
QUERY_MAX_LEN=${QUERY_MAX_LEN}
PASSAGE_MAX_LEN=${PASSAGE_MAX_LEN}
NUM_GPUS=$(nvidia-smi -L | wc -l)
NPROC_PER_NODE=${NUM_GPUS:-2}

# 1-stage, like paragraph-level and sentence-level
LOG_DIR="logs/lcaet/${CRIME_TYPE}/${SPLIT}/${LCAET_TYPE}"

STDOUT_LOG="${LOG_DIR}/${MODEL_TYPE}-ICT_v2-train_stdout.log"
STDERR_LOG="${LOG_DIR}/${MODEL_TYPE}-ICT_v2-train_stderr.log"

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
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --dataloader_drop_last True \
    --warmup_ratio "$WARMUP_RATIO" \
    --gradient_checkpointing \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --logging_steps 100 \
    --save_strategy epoch \
    --negatives_cross_device \
    --newerature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
    > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)

echo "[INFO] Fine-tuning completed successfully! Logs saved in $LOG_DIR."
