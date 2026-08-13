#!/usr/bin/env bash
# Full-parameter dialect SFT. Set DATA (and optionally MODEL / OUTPUT_DIR).
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}"

MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
DATA="${DATA:?Set DATA to the SFT jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/sft}"

NPROC_PER_NODE="${NPROC_PER_NODE}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift sft \
    --model "${MODEL}" \
    --dataset "${DATA}" \
    --split_dataset_ratio 0.001 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 96 \
    --per_device_eval_batch_size 96 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --padding_free true \
    --eval_steps 500 \
    --save_steps 500 \
    --max_length 1024 \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 8 \
    --attn_impl flash_attention_2 \
    --deepspeed zero2
