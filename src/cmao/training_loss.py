from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossBreakdown:
    policy_loss: float
    kl_term: float
    total_loss: float
    clip_fraction: float


def build_total_advantage(
    a_ans,
    a_qual,
    a_mode,
    lambda_ans: float = 1.0,
    lambda_qual: float = 1.0,
    lambda_mode: float = 1.0,
):
    return lambda_ans * a_ans + lambda_qual * a_qual + lambda_mode * a_mode


def cmao_clipped_policy_loss(
    current_logprobs,
    old_logprobs,
    advantages,
    kl_values=None,
    response_mask=None,
    clip_range: float = 0.2,
    kl_coef: float = 0.0,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for CMAO training loss.") from exc

    log_ratio = current_logprobs - old_logprobs
    if response_mask is not None:
        active_mask = response_mask.to(dtype=torch.bool, device=log_ratio.device)
        log_ratio = torch.where(active_mask, log_ratio, torch.zeros_like(log_ratio))
    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    objective_advantages = advantages
    while objective_advantages.ndim < ratio.ndim:
        objective_advantages = objective_advantages.unsqueeze(-1)

    unclipped_objective = ratio * objective_advantages
    clipped_objective = clipped_ratio * objective_advantages
    policy_objective = torch.minimum(unclipped_objective, clipped_objective)
    if response_mask is None:
        policy_loss = -policy_objective.mean()
    else:
        mask = response_mask.to(dtype=policy_objective.dtype, device=policy_objective.device)
        token_counts = mask.sum(dim=-1)
        valid_samples = token_counts > 0
        per_sample_policy_loss = -(policy_objective * mask).sum(dim=-1) / token_counts.clamp_min(1.0)
        if valid_samples.any():
            policy_loss = per_sample_policy_loss[valid_samples].mean()
        else:
            policy_loss = (current_logprobs * 0.0).sum()

    if kl_values is None:
        kl_term = torch.zeros((), device=current_logprobs.device, dtype=current_logprobs.dtype)
    elif response_mask is None:
        kl_term = kl_values.mean()
    else:
        mask = response_mask.to(dtype=kl_values.dtype, device=kl_values.device)
        token_counts = mask.sum(dim=-1)
        valid_samples = token_counts > 0
        per_sample_kl = (kl_values * mask).sum(dim=-1) / token_counts.clamp_min(1.0)
        if valid_samples.any():
            kl_term = per_sample_kl[valid_samples].mean()
        else:
            kl_term = torch.zeros((), device=current_logprobs.device, dtype=current_logprobs.dtype)
    total_loss = policy_loss + kl_coef * kl_term

    clipped_tokens = ((ratio - clipped_ratio).abs() > 1e-8).float()
    if response_mask is None:
        clip_fraction = clipped_tokens.mean()
    else:
        mask = response_mask.to(dtype=clipped_tokens.dtype, device=clipped_tokens.device)
        clip_fraction = (clipped_tokens * mask).sum() / mask.sum().clamp_min(1.0)
    return total_loss, LossBreakdown(
        policy_loss=float(policy_loss.detach().item()),
        kl_term=float(kl_term.detach().item()),
        total_loss=float(total_loss.detach().item()),
        clip_fraction=float(clip_fraction.detach().item()),
    )
