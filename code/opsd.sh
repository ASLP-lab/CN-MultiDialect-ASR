#!/usr/bin/env bash
# Vanilla OPSD (GKD) on the SFT checkpoint. Set MODEL and DATA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}"

MODEL="${MODEL:?Set MODEL to the SFT checkpoint}"
DATA="${DATA:?Set DATA to the SFT jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/opsd}"
export OPSD_DATA_PATH="${DATA}"

NPROC_PER_NODE="${NPROC_PER_NODE}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model "${MODEL}" \
    --teacher_model "${MODEL}" \
    --tuner_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size "${NPROC_PER_NODE}" \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_max_num_seqs 64 \
    --vllm_max_model_len 8192 \
    --sleep_level 0 \
    --external_plugins "${SCRIPT_DIR}/plugins/opsd_plugin.py" \
    --dataset asr-opsd \
    --lmbda 1.0 \
    --beta 0.5 \
    --temperature 0.8 \
    --top_p 0.9 \
    --sft_alpha 0 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --steps_per_generation 1 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --padding_free true \
    --save_steps 200 \
    --eval_steps 100 \
    --logging_steps 10 \
    --output_dir "${OUTPUT_DIR}" \
    --max_length 8192 \
    --max_completion_length 2048 \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --attn_impl flash_attention_2 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 8 \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --vit_gradient_checkpointing true
