#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/models/Qwen3-VL-Embedding-2B}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/dataset/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output/qwen3-vl-emb-lora}"
RUN_NAME=${RUN_NAME:-qwen3-vl-emb-$(date +%Y%m%d-%H%M%S)}
export WANDB_PROJECT=${WANDB_PROJECT:-qwen3-embedding}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_NAME}
export TMPDIR=/tmp
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
NPROC_PER_NODE=1 \
INFONCE_TEMPERATURE=0.1 \
swift sft \
    --model $MODEL_PATH \
    --task_type embedding \
    --dataset $DATASET_PATH \
    --loss_type infonce \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --padding_free true \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --split_dataset_ratio 0.02 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --max_length 8192 \
    --output_dir $OUTPUT_DIR \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file true \
    --dataloader_drop_last true \
    --deepspeed zero2
