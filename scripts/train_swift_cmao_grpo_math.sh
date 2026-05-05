#!/usr/bin/env bash
set -xeuo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SWIFT_DIR=${SWIFT_DIR:-"${ROOT_DIR}/third_party/ms-swift"}
CMAO_PLUGIN=${CMAO_PLUGIN:-"${ROOT_DIR}/integrations/swift/cmao_plugin.py"}

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${SWIFT_DIR}:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
TRAIN_FILE=${TRAIN_FILE:-"${ROOT_DIR}/data/swift_math500/train.jsonl"}
VAL_FILE=${VAL_FILE:-"${ROOT_DIR}/data/swift_math500/val.jsonl"}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TUNER_TYPE=${TUNER_TYPE:-${TRAIN_TYPE:-lora}}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
TARGET_MODULES=${TARGET_MODULES:-all-linear}

NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1536}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
LOGGING_STEPS=${LOGGING_STEPS:-1}
EVAL_STRATEGY=${EVAL_STRATEGY:-no}
WARMUP_RATIO=${WARMUP_RATIO:-0.0}
TEMPERATURE=${TEMPERATURE:-1.0}

USE_VLLM=${USE_VLLM:-false}
VLLM_MODE=${VLLM_MODE:-colocate}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.35}
SLEEP_LEVEL=${SLEEP_LEVEL:-1}

CMAO_LAMBDA_ANS=${CMAO_LAMBDA_ANS:-1.0}
CMAO_LAMBDA_QUAL=${CMAO_LAMBDA_QUAL:-0.4}
CMAO_LAMBDA_MODE=${CMAO_LAMBDA_MODE:-0.1}
CMAO_QUALITY_SIGNAL=${CMAO_QUALITY_SIGNAL:-raw}
CMAO_QUALITY_MARGIN=${CMAO_QUALITY_MARGIN:-0.05}
CMAO_CONCISE_TOKEN_CAP=${CMAO_CONCISE_TOKEN_CAP:-512}

export CMAO_LAMBDA_ANS
export CMAO_LAMBDA_QUAL
export CMAO_LAMBDA_MODE
export CMAO_QUALITY_SIGNAL
export CMAO_QUALITY_MARGIN
export CMAO_CONCISE_TOKEN_CAP

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_9b_math500_swift_cmao_grpo_$(date +%Y%m%d_%H%M)}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT_DIR}/outputs/swift/${EXPERIMENT_NAME}"}

CMD=(
    swift rlhf
    --rlhf_type grpo
    --model "${MODEL_PATH}"
    --use_hf true
    --dataset "${TRAIN_FILE}"
    --val_dataset "${VAL_FILE}"
    --external_plugins "${CMAO_PLUGIN}"
    --reward_funcs cmao
    --tuner_type "${TUNER_TYPE}"
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --target_modules "${TARGET_MODULES}"
    --torch_dtype bfloat16
    --max_length "${MAX_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --num_generations "${NUM_GENERATIONS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --max_steps "${MAX_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit 2
    --logging_steps "${LOGGING_STEPS}"
    --eval_strategy "${EVAL_STRATEGY}"
    --warmup_ratio "${WARMUP_RATIO}"
    --temperature "${TEMPERATURE}"
    --output_dir "${OUTPUT_DIR}"
    --log_completions true
    --report_to none
    --use_vllm "${USE_VLLM}"
)

if [[ "${USE_VLLM}" == "true" ]]; then
    CMD+=(
        --vllm_mode "${VLLM_MODE}"
        --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
        --sleep_level "${SLEEP_LEVEL}"
        --offload_model true
        --offload_optimizer true
        --gc_collect_after_offload true
    )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE="${NPROC_PER_NODE}" "${CMD[@]}" "$@"
