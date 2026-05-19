from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossBreakdown:
    policy_loss: float
    kl_term: float
    total_loss: float
    clip_ratio_region_mean: float
    clip_ratio_low_mean: float
    clip_ratio_high_mean: float

    @property
    def clip_fraction(self) -> float:
        return self.clip_ratio_region_mean


@dataclass
class DCSPOBreakdown:
    policy_loss: float
    total_loss: float
    weight_mean: float
    weight_min: float
    weight_max: float
    delta_phi_mean: float
    delta_phi_clipped_fraction: float


def cmao_clipped_policy_loss(
    current_logprobs,
    old_logprobs,
    advantages,
    kl_values=None,
    response_mask=None,
    epsilon: float = 0.2,
    beta: float = 0.0,
    loss_type: str = "dapo",
    max_completion_length: int | None = None,
    clip_range: float | None = None,
    kl_coef: float | None = None,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for CMAO training loss.") from exc

    if clip_range is not None:
        epsilon = clip_range
    if kl_coef is not None:
        beta = kl_coef

    log_ratio = current_logprobs - old_logprobs
    if response_mask is not None:
        active_mask = response_mask.to(dtype=torch.bool, device=log_ratio.device)
        log_ratio = torch.where(active_mask, log_ratio, torch.zeros_like(log_ratio))
    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    objective_advantages = advantages
    while objective_advantages.ndim < ratio.ndim:
        objective_advantages = objective_advantages.unsqueeze(-1)

    unclipped_objective = ratio * objective_advantages
    clipped_objective = clipped_ratio * objective_advantages
    policy_objective = torch.minimum(unclipped_objective, clipped_objective)
    normalized_loss_type = str(loss_type or "dapo").lower()
    if response_mask is None:
        policy_loss = -policy_objective.mean()
    else:
        mask = response_mask.to(dtype=policy_objective.dtype, device=policy_objective.device)
        if normalized_loss_type in {"dapo", "bnpo"}:
            policy_loss = -(policy_objective * mask).sum() / mask.sum().clamp_min(1.0)
        elif normalized_loss_type == "dr_grpo":
            normalizer = float(max_completion_length or int(mask.shape[-1]) or 1)
            policy_loss = -(policy_objective * mask).sum() / (
                mask.shape[0] * max(normalizer, 1.0)
            )
        elif normalized_loss_type == "grpo":
            token_counts = mask.sum(dim=-1)
            valid_samples = token_counts > 0
            per_sample_policy_loss = -(policy_objective * mask).sum(dim=-1) / token_counts.clamp_min(1.0)
            if valid_samples.any():
                policy_loss = per_sample_policy_loss[valid_samples].mean()
            else:
                policy_loss = (current_logprobs * 0.0).sum()
        else:
            raise ValueError(f"Unsupported GRPO loss_type: {loss_type}")

    if kl_values is None:
        kl_term = torch.zeros((), device=current_logprobs.device, dtype=current_logprobs.dtype)
    elif response_mask is None:
        kl_term = kl_values.mean()
    else:
        mask = response_mask.to(dtype=kl_values.dtype, device=kl_values.device)
        kl_term = (kl_values * mask).sum() / mask.sum().clamp_min(1.0)
    total_loss = policy_loss + beta * kl_term

    low_clipped_tokens = (ratio < 1.0 - epsilon).float()
    high_clipped_tokens = (ratio > 1.0 + epsilon).float()
    clipped_tokens = torch.maximum(low_clipped_tokens, high_clipped_tokens)
    if response_mask is None:
        clip_fraction = clipped_tokens.mean()
        clip_low = low_clipped_tokens.mean()
        clip_high = high_clipped_tokens.mean()
    else:
        mask = response_mask.to(dtype=clipped_tokens.dtype, device=clipped_tokens.device)
        denominator = mask.sum().clamp_min(1.0)
        clip_fraction = (clipped_tokens * mask).sum() / denominator
        clip_low = (low_clipped_tokens * mask).sum() / denominator
        clip_high = (high_clipped_tokens * mask).sum() / denominator
    return total_loss, LossBreakdown(
        policy_loss=float(policy_loss.detach().item()),
        kl_term=float(kl_term.detach().item()),
        total_loss=float(total_loss.detach().item()),
        clip_ratio_region_mean=float(clip_fraction.detach().item()),
        clip_ratio_low_mean=float(clip_low.detach().item()),
        clip_ratio_high_mean=float(clip_high.detach().item()),
    )


def dcspo_weighted_sft_loss(
    current_logprobs,
    current_entropies,
    reference_logprobs,
    reference_entropies,
    utilities=None,
    response_mask=None,
    clip_range: float = 2.0,
    normalize_by: str = "tokens",
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for DCSPO training loss.") from exc

    current_phi = current_logprobs + current_entropies
    reference_phi = reference_logprobs + reference_entropies
    delta_phi = current_phi - reference_phi.detach()
    clipped_delta_phi = delta_phi.clamp(min=-float(clip_range), max=float(clip_range))
    rho = torch.exp(clipped_delta_phi)

    weights = rho
    if utilities is not None:
        sample_utilities = utilities.to(dtype=weights.dtype, device=weights.device)
        while sample_utilities.ndim < weights.ndim:
            sample_utilities = sample_utilities.unsqueeze(-1)
        weights = weights * sample_utilities
    weights = weights.detach()

    objective = weights * current_logprobs
    if response_mask is None:
        mask = torch.ones_like(current_logprobs, dtype=current_logprobs.dtype)
    else:
        mask = response_mask.to(dtype=current_logprobs.dtype, device=current_logprobs.device)

    masked_objective = objective * mask
    normalized = str(normalize_by or "tokens").lower()
    if normalized == "tokens":
        policy_loss = -masked_objective.sum() / mask.sum().clamp_min(1.0)
    elif normalized == "weighted_tokens":
        policy_loss = -masked_objective.sum() / (weights * mask).sum().clamp_min(1e-8)
    elif normalized == "batch":
        policy_loss = -masked_objective.sum() / max(1, int(current_logprobs.shape[0]))
    else:
        raise ValueError(f"Unsupported DCSPO normalize_by: {normalize_by}")

    active = mask > 0
    if active.any():
        active_weights = weights[active]
        active_delta = delta_phi.detach()[active]
        clipped_fraction = (
            (active_delta.abs() > float(clip_range)).to(dtype=current_logprobs.dtype).mean()
        )
        weight_mean = active_weights.mean()
        weight_min = active_weights.min()
        weight_max = active_weights.max()
        delta_mean = active_delta.mean()
    else:
        zero = (current_logprobs * 0.0).sum().detach()
        clipped_fraction = zero
        weight_mean = zero
        weight_min = zero
        weight_max = zero
        delta_mean = zero

    return policy_loss, DCSPOBreakdown(
        policy_loss=float(policy_loss.detach().item()),
        total_loss=float(policy_loss.detach().item()),
        weight_mean=float(weight_mean.detach().item()),
        weight_min=float(weight_min.detach().item()),
        weight_max=float(weight_max.detach().item()),
        delta_phi_mean=float(delta_mean.detach().item()),
        delta_phi_clipped_fraction=float(clipped_fraction.detach().item()),
    )
