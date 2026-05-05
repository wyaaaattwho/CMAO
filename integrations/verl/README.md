# CMAO on verl

This integration keeps `third_party/verl` unchanged and injects CMAO through verl's
external reward-manager interface.

## Why verl

verl is a mature open-source RL training stack with GRPO/DAPO support, FSDP and
Megatron backends, and vLLM/SGLang rollout engines. That makes it a better base
for stable large-model training than the small local online GRPO loop.

## CMAO Reward Shape

`CMAORewardManager` buffers all rollouts sharing the same verl `uid`, then scores
the whole prompt group:

```text
r_i = lambda_ans  * correct_i
    + lambda_qual * quality_i
    + lambda_mode * quality_i * -log(p(mode_i | correct group))
```

Only correct answers contribute quality and mode rewards when
`quality_correct_only=True`. verl then applies its own GRPO group normalization by
`uid`, so the manager returns raw scalar rewards rather than pre-normalized CMAO
advantages.

Set `CMAO_QUALITY_SIGNAL=pairwise` to use the legacy pairwise quality preference
signal instead of raw quality.

## Data

Prepare MATH-500 in verl parquet format:

```bash
python scripts/prepare_verl_math_data.py \
  --dataset-name math-500 \
  --train-split test \
  --val-split test \
  --limit 500
```

## Training

```bash
bash scripts/train_verl_cmao_math.sh
```

Useful overrides:

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/test.parquet \
CMAO_LAMBDA_ANS=1.0 \
CMAO_LAMBDA_QUAL=0.4 \
CMAO_LAMBDA_MODE=0.1 \
ROLLOUT_N=16 \
bash scripts/train_verl_cmao_math.sh
```

For pure GRPO on the same stack:

```bash
CMAO_LAMBDA_QUAL=0 CMAO_LAMBDA_MODE=0 bash scripts/train_verl_cmao_math.sh
```
