from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from .answer_judge import normalize_math_text
from .types import ScoredGroup, ScoredSample


def _score_for_strategy(scored_samples: list[ScoredSample], item: ScoredSample, strategy: str) -> float:
    if strategy == "greedy":
        return 1.0 if item is scored_samples[0] else 0.0
    if strategy == "quality":
        return item.score.quality_score
    if strategy == "a_total":
        return item.advantage.a_total if item.advantage else item.score.quality_score
    if strategy == "a_total_without_mode":
        if item.advantage:
            return item.advantage.a_ans + item.advantage.a_qual
        return item.score.quality_score
    if strategy == "quality_only_correct_samples":
        return item.score.quality_score if item.score.answer_correct else float("-inf")
    if strategy == "majority_vote":
        counts = Counter(
            normalize_math_text(candidate.sample.final_answer or candidate.sample.raw_text)
            for candidate in scored_samples
        )
        answer_key = normalize_math_text(item.sample.final_answer or item.sample.raw_text)
        return counts[answer_key] + item.score.quality_score * 1e-3
    raise ValueError(f"Unknown strategy: {strategy}")


def _select_best(scored_samples: list[ScoredSample], strategy: str) -> ScoredSample:
    if strategy == "greedy":
        return scored_samples[0]
    return max(scored_samples, key=lambda item: _score_for_strategy(scored_samples, item, strategy))


def _partition_name(group: ScoredGroup) -> str:
    correctness_values = [item.score.answer_correct for item in group.scored_samples]
    if correctness_values and all(correctness_values):
        return "all_correct"
    if any(correctness_values):
        return "partially_correct"
    return "all_incorrect"


def _quality_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_quality = sum(values) / len(values)
    return sum((value - mean_quality) ** 2 for value in values) / len(values)


def _empty_metrics() -> dict[str, float | int]:
    return {"correct": 0, "total": 0}


def _finalize_metrics(metrics: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "accuracy": values["correct"] / values["total"] if values["total"] else 0.0,
            "correct": values["correct"],
            "total": values["total"],
        }
        for name, values in metrics.items()
    }


def _build_pass_at_k(
    groups: list[ScoredGroup],
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, dict[str, float | int]]]]:
    max_group_size = max((len(group.scored_samples) for group in groups), default=0)
    if max_group_size <= 0:
        return {}, {"all_correct": {}, "partially_correct": {}, "all_incorrect": {}}

    overall = {k: {"correct": 0, "total": 0} for k in range(1, max_group_size + 1)}
    per_subset = {
        "all_correct": {k: {"correct": 0, "total": 0} for k in range(1, max_group_size + 1)},
        "partially_correct": {k: {"correct": 0, "total": 0} for k in range(1, max_group_size + 1)},
        "all_incorrect": {k: {"correct": 0, "total": 0} for k in range(1, max_group_size + 1)},
    }

    for group in groups:
        partition = _partition_name(group)
        for k in range(1, max_group_size + 1):
            window = group.scored_samples[:k]
            passed = any(item.score.answer_correct for item in window)
            overall[k]["total"] += 1
            per_subset[partition][k]["total"] += 1
            if passed:
                overall[k]["correct"] += 1
                per_subset[partition][k]["correct"] += 1

    overall_report = {
        str(k): {
            "k": k,
            "pass_rate": values["correct"] / values["total"] if values["total"] else 0.0,
            "correct": values["correct"],
            "total": values["total"],
        }
        for k, values in overall.items()
    }
    per_subset_report = {
        subset: {
            str(k): {
                "k": k,
                "pass_rate": values["correct"] / values["total"] if values["total"] else 0.0,
                "correct": values["correct"],
                "total": values["total"],
            }
            for k, values in subset_values.items()
        }
        for subset, subset_values in per_subset.items()
    }
    return overall_report, per_subset_report


def _ablation_score(sample: ScoredSample, ablation: str) -> float:
    subscores = sample.score.quality_subscores
    if ablation == "drop_local_check":
        weights = {"format": 0.2, "structure": 0.2, "self_verify": 0.15, "concise": 0.1}
    elif ablation == "drop_self_verify":
        weights = {"format": 0.2, "local_check": 0.35, "structure": 0.2, "concise": 0.1}
    elif ablation == "format_structure_only":
        weights = {"format": 0.5, "structure": 0.5}
    else:
        raise ValueError(f"Unknown ablation: {ablation}")
    numerator = sum(subscores.get(key, 0.0) * value for key, value in weights.items())
    denominator = sum(weights.values()) or 1.0
    return numerator / denominator


def _build_ablation_report(groups: list[ScoredGroup]) -> dict[str, dict[str, float | int]]:
    ablations = ("drop_local_check", "drop_self_verify", "format_structure_only")
    metrics = {name: {"correct": 0, "total": 0} for name in ablations}
    for group in groups:
        for ablation in ablations:
            winner = max(group.scored_samples, key=lambda item: _ablation_score(item, ablation))
            metrics[ablation]["total"] += 1
            if winner.score.answer_correct:
                metrics[ablation]["correct"] += 1
    return _finalize_metrics(metrics)


def build_report(groups: Iterable[ScoredGroup]) -> dict:
    group_list = list(groups)
    pass_at_k, per_subset_pass_at_k = _build_pass_at_k(group_list)
    strategies = (
        "greedy",
        "majority_vote",
        "quality",
        "a_total",
        "a_total_without_mode",
        "quality_only_correct_samples",
    )
    metrics = {strategy: {"correct": 0, "total": 0} for strategy in strategies}
    subset_metrics = {
        "all_correct": {strategy: {"correct": 0, "total": 0} for strategy in strategies},
        "partially_correct": {strategy: {"correct": 0, "total": 0} for strategy in strategies},
        "all_incorrect": {strategy: {"correct": 0, "total": 0} for strategy in strategies},
    }
    mode_counter = Counter()
    all_correct_group_count = 0
    all_correct_quality_variance = []
    empty_extraction_count = 0
    placeholder_extraction_count = 0
    nonempty_incorrect_count = 0
    total_samples = 0
    dataset_counter = Counter()
    group_diagnostics = []

    for group in group_list:
        dataset_counter[group.problem.source] += 1
        correctness_values = [item.score.answer_correct for item in group.scored_samples]
        partition = _partition_name(group)
        if correctness_values and all(correctness_values):
            all_correct_group_count += 1
            quality_scores = [item.score.quality_score for item in group.scored_samples]
            variance = _quality_variance(quality_scores)
            all_correct_quality_variance.append(variance)

        for item in group.scored_samples:
            total_samples += 1
            if item.score.answer_correct:
                mode_counter[item.score.mode_label] += 1
            if item.score.answer_extraction.get("empty_prediction"):
                empty_extraction_count += 1
            if item.score.answer_extraction.get("placeholder_prediction"):
                placeholder_extraction_count += 1
            if item.score.answer_judgment.get("nonempty_incorrect"):
                nonempty_incorrect_count += 1

        winners = {}
        for strategy in strategies:
            winner = _select_best(group.scored_samples, strategy)
            winners[strategy] = {
                "sample_id": winner.sample.sample_id,
                "final_answer": winner.sample.final_answer,
                "answer_correct": winner.score.answer_correct,
                "score": _score_for_strategy(group.scored_samples, winner, strategy),
            }
            metrics[strategy]["total"] += 1
            subset_metrics[partition][strategy]["total"] += 1
            if winner.score.answer_correct:
                metrics[strategy]["correct"] += 1
                subset_metrics[partition][strategy]["correct"] += 1

        quality_sorted = sorted(
            group.scored_samples,
            key=lambda item: item.score.quality_score,
            reverse=True,
        )
        mode_distribution = Counter(item.score.mode_label for item in group.scored_samples)
        answer_distribution = Counter(
            normalize_math_text(item.sample.final_answer or item.sample.raw_text)
            for item in group.scored_samples
        )
        group_diagnostics.append(
            {
                "problem_id": group.problem.id,
                "dataset_name": group.problem.source,
                "report_partition": partition,
                "correct_sample_count": sum(1 for value in correctness_values if value),
                "total_sample_count": len(group.scored_samples),
                "extraction_issue_count": sum(
                    1
                    for item in group.scored_samples
                    if item.score.answer_extraction.get("empty_prediction")
                    or item.score.answer_extraction.get("placeholder_prediction")
                ),
                "placeholder_answer_count": sum(
                    1
                    for item in group.scored_samples
                    if item.score.answer_extraction.get("placeholder_prediction")
                ),
                "answer_distribution": dict(answer_distribution),
                "mode_distribution": dict(mode_distribution),
                "quality_ranking": [
                    {
                        "sample_id": item.sample.sample_id,
                        "quality_score": item.score.quality_score,
                        "mode_label": item.score.mode_label,
                        "answer_correct": item.score.answer_correct,
                    }
                    for item in quality_sorted
                ],
                "strategy_winners": winners,
            }
        )

    return {
        "strategies": _finalize_metrics(metrics),
        "pass_at_k": pass_at_k,
        "per_subset_pass_at_k": per_subset_pass_at_k,
        "per_subset_strategies": {
            subset: _finalize_metrics(strategy_metrics)
            for subset, strategy_metrics in subset_metrics.items()
        },
        "all_correct_group_count": all_correct_group_count,
        "avg_all_correct_quality_variance": (
            sum(all_correct_quality_variance) / len(all_correct_quality_variance)
            if all_correct_quality_variance
            else 0.0
        ),
        "correct_mode_distribution": dict(mode_counter),
        "empty_extraction_rate": empty_extraction_count / total_samples if total_samples else 0.0,
        "placeholder_extraction_rate": placeholder_extraction_count / total_samples if total_samples else 0.0,
        "nonempty_incorrect_rate": nonempty_incorrect_count / total_samples if total_samples else 0.0,
        "dataset_breakdown": dict(dataset_counter),
        "quality_ablations": _build_ablation_report(group_list),
        "group_diagnostics": group_diagnostics,
    }
