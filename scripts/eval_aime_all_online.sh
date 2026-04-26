#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_NAME="${DATASET_NAME:-math-500}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
DATASET_LIMIT="${DATASET_LIMIT:-}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-}"
DATASET_TAG="${DATASET_TAG:-math500test}"

GROUP_SIZE="${GROUP_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
DO_SAMPLE="${DO_SAMPLE:-true}"
FORCE_MERGE="${FORCE_MERGE:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
AUTO_SUMMARY="${AUTO_SUMMARY:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-}"

# name|type|source
# type=adapter: source is adapter checkpoint dir
# type=base: source is model id/path
RUNS=(
  "math500_online_grpo_1.5b_lora|adapter|outputs/train/math500_online_grpo_1.5b_lora/checkpoint-final"
  "math500_online_cmao_1.5b_lora|adapter|outputs/train/math500_online_cmao_1.5b_lora/checkpoint-final"
  "math500_online_grpo_7b_lora|adapter|outputs/train/math500_online_grpo_7b_lora/checkpoint-final"
  "math500_online_cmao_7b_lora|adapter|outputs/train/math500_online_cmao_7b_lora/checkpoint-final"
  "qwen2.5-math-1.5b-base|base|Qwen/Qwen2.5-Math-1.5B-Instruct"
  "qwen2.5-math-7b-base|base|Qwen/Qwen2.5-Math-7B-Instruct"
)

usage() {
  cat <<EOF
Usage: bash scripts/eval_aime_all_online.sh [options]

Options:
  --dataset-name NAME           Dataset name (default: math-500)
  --dataset-split SPLIT         Dataset split (default: test)
  --dataset-limit N             Limit number of problems (default: all)
  --dataset-config-name NAME    Optional HF dataset config name
  --dataset-tag TAG             Output suffix tag (default: math500test)
  --group-size N                Sampling group size (default: 8)
  --temperature F               Sampling temperature (default: 0.7)
  --top-p F                     Sampling top-p (default: 0.95)
  --max-new-tokens N            Max new tokens (default: 2048)
  --do-sample BOOL              true/false (default: true)
  --gpu-ids IDS                 Comma-separated GPU ids for round-robin (default: 0,1,2,3)
  --max-parallel N              Max concurrent runs (default: number of GPUs)
  --force-merge 0|1             Force re-merge adapters (default: 0)
  --auto-summary 0|1            Run summary after eval (default: 1)
  --python-bin BIN              Python executable (default: python)
  -h, --help                    Show this help message

Example:
  bash scripts/eval_aime_all_online.sh --dataset-name math-500 --dataset-split test --dataset-limit 500 --dataset-tag math500test
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --dataset-split)
      DATASET_SPLIT="$2"
      shift 2
      ;;
    --dataset-limit)
      DATASET_LIMIT="$2"
      shift 2
      ;;
    --dataset-config-name)
      DATASET_CONFIG_NAME="$2"
      shift 2
      ;;
    --dataset-tag)
      DATASET_TAG="$2"
      shift 2
      ;;
    --group-size)
      GROUP_SIZE="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --do-sample)
      DO_SAMPLE="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --force-merge)
      FORCE_MERGE="$2"
      shift 2
      ;;
    --auto-summary)
      AUTO_SUMMARY="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p outputs/eval outputs/eval/configs outputs/eval/logs outputs/merged

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ ${#GPU_ARRAY[@]} -lt 1 ]]; then
  echo "[ERROR] GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi
if [[ -z "$MAX_PARALLEL" ]]; then
  MAX_PARALLEL="${#GPU_ARRAY[@]}"
fi
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "[ERROR] MAX_PARALLEL must be >= 1." >&2
  exit 1
fi

run_cmd() {
  local cmd=("$@")
  printf '[CMD]'
  for token in "${cmd[@]}"; do
    printf ' %q' "$token"
  done
  printf '\n'
  "${cmd[@]}"
}

build_config() {
  local config_path="$1"
  local model_name="$2"

  {
    echo "{"
    echo "  \"model\": {"
    echo "    \"name\": \"${model_name}\""
    echo "  },"
    echo "  \"dataset\": {"
    echo "    \"name\": \"${DATASET_NAME}\"," 
    echo "    \"split\": \"${DATASET_SPLIT}\"$( [[ -n "$DATASET_LIMIT" ]] && printf "," )"
    if [[ -n "$DATASET_LIMIT" ]]; then
      echo "    \"limit\": ${DATASET_LIMIT}$( [[ -n "$DATASET_CONFIG_NAME" ]] && printf "," )"
    fi
    if [[ -n "$DATASET_CONFIG_NAME" ]]; then
      echo "    \"config_name\": \"${DATASET_CONFIG_NAME}\""
    fi
    echo "  },"
    echo "  \"sampling\": {"
    echo "    \"group_size\": ${GROUP_SIZE},"
    echo "    \"temperature\": ${TEMPERATURE},"
    echo "    \"top_p\": ${TOP_P},"
    echo "    \"max_new_tokens\": ${MAX_NEW_TOKENS},"
    echo "    \"do_sample\": ${DO_SAMPLE}"
    echo "  }"
    echo "}"
  } > "$config_path"
}

run_one() {
  local name="$1"
  local run_type="$2"
  local source="$3"
  local gpu_id="$4"

  local model_name
  if [[ "$run_type" == "adapter" ]]; then
    local adapter_dir="$source"
    local merged_dir="outputs/merged/${name}"
    if [[ ! -d "$adapter_dir" ]]; then
      echo "[ERROR][$name] Adapter checkpoint not found: $adapter_dir" >&2
      return 1
    fi
    if [[ "$FORCE_MERGE" == "1" || ! -d "$merged_dir" ]]; then
      echo "[INFO][$name] Merging LoRA"
      run_cmd "$PYTHON_BIN" scripts/merge_lora.py --adapter "$adapter_dir" --output "$merged_dir"
    else
      echo "[INFO][$name] Reusing merged model: $merged_dir"
    fi
    model_name="$merged_dir"
  elif [[ "$run_type" == "base" ]]; then
    model_name="$source"
    echo "[INFO][$name] Using base model: $model_name"
  else
    echo "[ERROR][$name] Unknown run type: $run_type" >&2
    return 1
  fi

  local config_path="outputs/eval/configs/${name}_${DATASET_TAG}.json"
  build_config "$config_path" "$model_name"

  local sample_out="outputs/eval/${name}_${DATASET_TAG}_samples.json"
  local score_out="outputs/eval/${name}_${DATASET_TAG}_scores.json"
  local adv_out="outputs/eval/${name}_${DATASET_TAG}_advantages.json"
  local report_out="outputs/eval/${name}_${DATASET_TAG}_report.json"
  local report_pretty_out="outputs/eval/${name}_${DATASET_TAG}_report_pretty.json"
  local case_prefix="outputs/eval/${name}_${DATASET_TAG}_analysis"

  echo "[INFO][$name] Using GPU ${gpu_id}"
  echo "[INFO][$name] Sampling on dataset ${DATASET_NAME}/${DATASET_SPLIT}"
  run_cmd env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -m cmao.cli sample --config "$config_path" --output "$sample_out"

  echo "[INFO][$name] Scoring"
  run_cmd "$PYTHON_BIN" -m cmao.cli score --input "$sample_out" --output "$score_out" --config configs/scoring/default.json

  echo "[INFO][$name] Computing advantage"
  run_cmd "$PYTHON_BIN" -m cmao.cli advantage --input "$score_out" --output "$adv_out" --config configs/scoring/default.json

  echo "[INFO][$name] Rerank evaluation"
  run_cmd "$PYTHON_BIN" -m cmao.cli rerank_eval --input "$adv_out" --output "$report_out"

  echo "[INFO][$name] Building pretty report"
  run_cmd "$PYTHON_BIN" -m cmao.cli report --input "$adv_out" --output "$report_pretty_out"

  echo "[INFO][$name] Exporting case analysis"
  run_cmd "$PYTHON_BIN" -m cmao.cli analyze_cases --input "$adv_out" --output-prefix "$case_prefix"

  echo "[DONE][$name] Completed on GPU ${gpu_id}"
}

failed=0
declare -a pids=()
declare -a names=()

wait_for_slot() {
  while true; do
    local running
    running=$(jobs -rp | wc -l | tr -d ' ')
    if [[ "$running" -lt "$MAX_PARALLEL" ]]; then
      break
    fi
    sleep 1
  done
}

for i in "${!RUNS[@]}"; do
  IFS='|' read -r name run_type source <<< "${RUNS[$i]}"
  gpu_index=$((i % ${#GPU_ARRAY[@]}))
  gpu_id="${GPU_ARRAY[$gpu_index]}"
  log_path="outputs/eval/logs/${name}_${DATASET_TAG}.log"

  wait_for_slot
  echo "[INFO] Launching $name on GPU ${gpu_id}. Log: ${log_path}"
  (
    run_one "$name" "$run_type" "$source" "$gpu_id"
  ) >"$log_path" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  if wait "$pid"; then
    echo "[OK] $name finished successfully"
  else
    echo "[FAIL] $name failed. Check log: outputs/eval/logs/${name}_${DATASET_TAG}.log" >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[DONE] Completed with failures. See outputs/eval/logs/*_${DATASET_TAG}.log" >&2
  exit 1
fi

if [[ "$AUTO_SUMMARY" == "1" ]]; then
  echo "[INFO] Generating comparison summary tables for ${DATASET_TAG}"
  run_cmd "$PYTHON_BIN" scripts/summarize_aime_eval.py \
    --eval-dir outputs/eval \
    --train-dir outputs/train \
    --report-suffix "_${DATASET_TAG}_report.json" \
    --analysis-suffix "_${DATASET_TAG}_analysis_summary.json" \
    --output-prefix "outputs/eval/${DATASET_TAG}_comparison"
fi

echo "[DONE] Evaluation finished for 6 models on ${DATASET_NAME}/${DATASET_SPLIT} (GPU round-robin)."
