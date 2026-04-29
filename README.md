# CMAO

Correct-Mode Advantage Optimization experiments for math reasoning.

## Online GRPO

Online GRPO training follows Hugging Face TRL GRPO parameter semantics. CMAO only changes the scalar reward used to compute group-relative advantages; the policy update uses the same GRPO loss.

Training configs use these fields:

- `sampling.num_generations`: number of completions sampled per prompt.
- `sampling.max_completion_length`: maximum generated completion length.
- `training.generation_batch_size`: total completions in one generation batch.
- `training.per_device_train_batch_size`: completion batch size for policy forward/backward.
- `training.gradient_accumulation_steps`: gradient accumulation steps. This implementation requires `generation_batch_size = per_device_train_batch_size * gradient_accumulation_steps`.
- `training.max_steps`: outer generation/update steps.
- `training.num_iterations`: TRL GRPO `mu`, the number of optimization iterations per generation batch.
- `training.epsilon`: clipping range.
- `training.beta`: reference-model KL penalty. If `beta = 0`, no reference model is loaded.
- `training.loss_type`: GRPO loss normalization. The default training configs use `dapo`.

CMAO reward construction:

```text
r_i = lambda_ans * r_ans_i
    + lambda_qual * r_qual_i
    + lambda_mode * r_mode_i

A_i = (r_i - mean(group_rewards)) / std(group_rewards)
```

The resulting `A_i` is used in the GRPO clipped objective:

```text
L = - sum_t mask_t * min(r_t A_i, clip(r_t, 1-epsilon, 1+epsilon) A_i) / sum_t mask_t
    + beta * KL
```

Example:

```bash
cmao train_online_grpo --config configs/training/math500_online_cmao_qwen35_9b_lora.json
```
