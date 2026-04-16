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
    clip_range: float = 0.2,
    kl_coef: float = 0.0,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for CMAO training loss.") from exc

    log_ratio = current_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    policy_objective = torch.minimum(unclipped_objective, clipped_objective)
    policy_loss = -policy_objective.mean()

    if kl_values is None:
        kl_term = torch.zeros((), device=current_logprobs.device, dtype=current_logprobs.dtype)
    else:
        kl_term = kl_values.mean()
    total_loss = policy_loss + kl_coef * kl_term

    clip_fraction = ((ratio - clipped_ratio).abs() > 1e-8).float().mean()
    return total_loss, LossBreakdown(
        policy_loss=float(policy_loss.detach().item()),
        kl_term=float(kl_term.detach().item()),
        total_loss=float(total_loss.detach().item()),
        clip_fraction=float(clip_fraction.detach().item()),
    )
