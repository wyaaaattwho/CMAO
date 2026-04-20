from __future__ import annotations

import math
from collections import Counter

from .types import AdvantageBundle, ScoredGroup, ScoredSample


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], epsilon: float) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) + epsilon


class CMAOComputer:
    def __init__(
        self,
        lambda_ans: float = 1.0,
        lambda_qual: float = 1.0,
        lambda_mode: float = 1.0,
        quality_pairwise_margin: float = 0.2,
        epsilon: float = 1e-8,
    ) -> None:
        self.lambda_ans = lambda_ans
        self.lambda_qual = lambda_qual
        self.lambda_mode = lambda_mode
        self.quality_pairwise_margin = quality_pairwise_margin
        self.epsilon = epsilon

    def compute_group(self, group: ScoredGroup) -> ScoredGroup:
        if not group.scored_samples:
            return group

        correctness = [1.0 if item.score.answer_correct else 0.0 for item in group.scored_samples]
        quality_raw = [
            item.score.quality_score if math.isfinite(item.score.quality_score) else 0.0
            for item in group.scored_samples
        ]
        quality_signal = [0.0 for _ in group.scored_samples]

        correct_indices = [idx for idx, value in enumerate(correctness) if value > 0.0]
        if len(correct_indices) >= 2:
            for offset, left_index in enumerate(correct_indices):
                for right_index in correct_indices[offset + 1 :]:
                    left_quality = quality_raw[left_index]
                    right_quality = quality_raw[right_index]
                    if left_quality - right_quality > self.quality_pairwise_margin:
                        quality_signal[left_index] += 1.0
                        quality_signal[right_index] -= 1.0
                    elif right_quality - left_quality > self.quality_pairwise_margin:
                        quality_signal[left_index] -= 1.0
                        quality_signal[right_index] += 1.0
            scale = float(max(1, len(correct_indices) - 1))
            quality_signal = [value / scale for value in quality_signal]

        mode_bonus = [0.0 for _ in group.scored_samples]
        if correct_indices:
            mode_labels = [str(group.scored_samples[idx].score.mode_label) for idx in correct_indices]
            counts = Counter(mode_labels)
            total = len(correct_indices)
            for idx in correct_indices:
                mode = str(group.scored_samples[idx].score.mode_label)
                mode_probability = counts[mode] / total
                safe_probability = max(mode_probability, self.epsilon)
                mode_bonus[idx] = quality_raw[idx] * (-math.log(safe_probability))

        # GRPO style: combine scalar rewards first, then normalize once per group.
        total_rewards = []
        for idx in range(len(group.scored_samples)):
            reward = (
                self.lambda_ans * correctness[idx]
                + self.lambda_qual * quality_signal[idx]
                + self.lambda_mode * mode_bonus[idx]
            )
            total_rewards.append(reward if math.isfinite(reward) else 0.0)

        mean_r = _mean(total_rewards)
        std_r = _std(total_rewards, self.epsilon)

        updated_samples: list[ScoredSample] = []
        for idx, item in enumerate(group.scored_samples):
            a_total = (total_rewards[idx] - mean_r) / std_r if std_r > 0.0 else 0.0
            updated_samples.append(
                ScoredSample(
                    sample=item.sample,
                    score=item.score,
                    advantage=AdvantageBundle(
                        a_ans=correctness[idx],
                        a_qual=quality_signal[idx],
                        a_mode=mode_bonus[idx],
                        a_total=a_total,
                    ),
                )
            )
        return ScoredGroup(problem=group.problem, scored_samples=updated_samples, metadata=group.metadata)
