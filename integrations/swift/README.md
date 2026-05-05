# CMAO on ms-swift

This integration replaces the experimental `verl` route with Qwen's recommended
Swift ecosystem path. It uses Swift's GRPO entrypoint and registers CMAO as a
custom ORM reward plugin.

## Files

- `integrations/swift/cmao_plugin.py`: Swift reward plugin registered as
  `cmao`.
- `scripts/prepare_swift_math_data.py`: converts supported CMAO math datasets
  into Swift messages JSONL.
- `scripts/train_swift_cmao_grpo_math.sh`: Qwen3.5 GRPO launcher.

## Data

```bash
python scripts/prepare_swift_math_data.py \
  --dataset-name math-500 \
  --train-split test \
  --val-split test \
  --train-output data/swift_math500/train.jsonl \
  --val-output data/swift_math500/val.jsonl
```

Each row keeps Swift's `messages` input and extra reward columns:

```json
{
  "messages": [{"role": "user", "content": "Problem ..."}],
  "solution": "42",
  "gold_answer": "42",
  "problem_id": "math-500-0",
  "source": "math-500"
}
```

## Training

Use a clean Swift environment rather than the old `verl` debugging environment.

```bash
pip install -e '.[swift]'
pip install -e third_party/ms-swift
python scripts/prepare_swift_math_data.py

MODEL_PATH=Qwen/Qwen3.5-9B \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
USE_VLLM=false \
bash scripts/train_swift_cmao_math.sh
```

For an answer-only GRPO baseline:

```bash
MODEL_PATH=Qwen/Qwen3.5-9B \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
USE_VLLM=false \
bash scripts/train_swift_grpo_math.sh
```

The old `scripts/train_verl_cmao_math.sh` and `scripts/train_verl_grpo_math.sh`
names are kept as compatibility wrappers, but they now call the Swift launcher
and do not import or execute verl.

The default launchers start with Swift's PyTorch/Transformers backend
(`USE_VLLM=false`) to avoid the CUDA IPC issues seen in the `verl` rollout path.
After the base path is stable, enable Swift colocated vLLM with:

```bash
USE_VLLM=true \
VLLM_GPU_MEMORY_UTILIZATION=0.35 \
bash scripts/train_swift_cmao_grpo_math.sh
```

## Reward

The plugin returns raw scalar rewards:

```text
r_i = lambda_ans * correct_i
    + lambda_qual * quality_i
    + lambda_mode * quality_i * -log p(mode_i | correct group)
```

Swift/TRL GRPO then performs group-relative normalization. Configure reward
weights with environment variables:

- `CMAO_LAMBDA_ANS`
- `CMAO_LAMBDA_QUAL`
- `CMAO_LAMBDA_MODE`
- `CMAO_QUALITY_SIGNAL=raw|pairwise`
- `CMAO_QUALITY_MARGIN`
- `CMAO_CONCISE_TOKEN_CAP`
