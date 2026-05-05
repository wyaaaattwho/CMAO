#!/usr/bin/env bash
set -xeuo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export CMAO_LAMBDA_ANS=${CMAO_LAMBDA_ANS:-1.0}
export CMAO_LAMBDA_QUAL=${CMAO_LAMBDA_QUAL:-0.4}
export CMAO_LAMBDA_MODE=${CMAO_LAMBDA_MODE:-0.1}
export CMAO_QUALITY_SIGNAL=${CMAO_QUALITY_SIGNAL:-raw}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_9b_math500_swift_cmao_$(date +%Y%m%d_%H%M)}
export EXPERIMENT_NAME

exec bash "${ROOT_DIR}/scripts/train_swift_cmao_grpo_math.sh" "$@"
