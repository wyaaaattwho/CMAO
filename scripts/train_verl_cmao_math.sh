#!/usr/bin/env bash
set -xeuo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VERL_DIR=${VERL_DIR:-"${ROOT_DIR}/third_party/verl"}
CMAO_MANAGER=${CMAO_MANAGER:-"${ROOT_DIR}/integrations/verl/cmao_reward_manager.py"}

export PYTHONPATH="${ROOT_DIR}/src:${VERL_DIR}:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
TRAIN_FILE=${TRAIN_FILE:-"${ROOT_DIR}/data/verl_math500/train.parquet"}
VAL_FILE=${VAL_FILE:-"${ROOT_DIR}/data/verl_math500/test.parquet"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
INFER_BACKEND=${INFER_BACKEND:-vllm}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1536}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:--1}

CMAO_LAMBDA_ANS=${CMAO_LAMBDA_ANS:-1.0}
CMAO_LAMBDA_QUAL=${CMAO_LAMBDA_QUAL:-0.4}
CMAO_LAMBDA_MODE=${CMAO_LAMBDA_MODE:-0.1}
CMAO_QUALITY_SIGNAL=${CMAO_QUALITY_SIGNAL:-raw}
CMAO_QUALITY_MARGIN=${CMAO_QUALITY_MARGIN:-0.05}
CMAO_CONCISE_TOKEN_CAP=${CMAO_CONCISE_TOKEN_CAP:-512}

LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']"}

PROJECT_NAME=${PROJECT_NAME:-cmao_verl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_9b_math500_cmao_$(date +%Y%m%d_%H%M)}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT_DIR}/outputs/verl/${EXPERIMENT_NAME}"}
LOGGER=${LOGGER:-'["console"]'}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}"
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.num_workers=1
    reward.reward_model.enable=False
    reward.reward_manager.source=importlib
    reward.reward_manager.name=CMAORewardManager
    reward.reward_manager.module.path="${CMAO_MANAGER}"
    +reward.reward_kwargs.lambda_ans=${CMAO_LAMBDA_ANS}
    +reward.reward_kwargs.lambda_qual=${CMAO_LAMBDA_QUAL}
    +reward.reward_kwargs.lambda_mode=${CMAO_LAMBDA_MODE}
    +reward.reward_kwargs.quality_signal=${CMAO_QUALITY_SIGNAL}
    +reward.reward_kwargs.quality_pairwise_margin=${CMAO_QUALITY_MARGIN}
    +reward.reward_kwargs.quality_correct_only=True
    +reward.reward_kwargs.concise_token_cap=${CMAO_CONCISE_TOKEN_CAP}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger="${LOGGER}"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.default_local_dir="${OUTPUT_DIR}"
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
