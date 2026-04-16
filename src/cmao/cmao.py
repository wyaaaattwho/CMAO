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
        correctness = [1.0 if item.score.answer_correct else 0.0 for item in group.scored_samples]
        quality = [item.score.quality_score for item in group.scored_samples]
        correct_indices = [idx for idx, value in enumerate(correctness) if value > 0.0]

        mean_ans = _mean(correctness)
        std_ans = _std(correctness, self.epsilon)
        a_ans = [(value - mean_ans) / std_ans for value in correctness]

        a_qual = [0.0 for _ in group.scored_samples]
        if len(correct_indices) >= 2:
            denom = max(1, len(correct_indices) - 1)
            for idx in correct_indices:
                preference_sum = 0
                for other_idx in correct_indices:
                    if idx == other_idx:
                        continue
                    diff = quality[idx] - quality[other_idx]
                    if diff > self.quality_pairwise_margin:
                        preference_sum += 1
                    elif diff < -self.quality_pairwise_margin:
                        preference_sum -= 1
                a_qual[idx] = max(-1.0, min(1.0, preference_sum / denom))

        mode_bonus = [0.0 for _ in group.scored_samples]
        if correct_indices:
            counts = Counter(group.scored_samples[idx].score.mode_label for idx in correct_indices)
            total = len(correct_indices)
            for idx in correct_indices:
                mode = group.scored_samples[idx].score.mode_label
                mode_probability = counts[mode] / total
                mode_bonus[idx] = quality[idx] * (-math.log(mode_probability))

        mean_mode = _mean(mode_bonus)
        std_mode = _std(mode_bonus, self.epsilon)
        a_mode = [(value - mean_mode) / std_mode if std_mode > 0 else 0.0 for value in mode_bonus]

        updated_samples: list[ScoredSample] = []
        for idx, item in enumerate(group.scored_samples):
            total_advantage = (
                self.lambda_ans * a_ans[idx]
                + self.lambda_qual * a_qual[idx]
                + self.lambda_mode * a_mode[idx]
            )
            updated_samples.append(
                ScoredSample(
                    sample=item.sample,
                    score=item.score,
                    advantage=AdvantageBundle(
                        a_ans=a_ans[idx],
                        a_qual=a_qual[idx],
                        a_mode=a_mode[idx],
                        a_total=total_advantage,
                    ),
                )
            )
        return ScoredGroup(problem=group.problem, scored_samples=updated_samples, metadata=group.metadata)
