from __future__ import annotations

from collections import Counter
from pathlib import Path

from .io_utils import save_json, save_jsonl
from .reporter import _partition_name, _select_best
from .types import ScoredGroup


def _build_case_record(case_type: str, group: ScoredGroup, payload: dict) -> dict:
    return {
        "case_type": case_type,
        "problem_id": group.problem.id,
        "dataset_name": group.problem.source,
        "prompt": group.problem.prompt,
        "gold_answer": group.problem.gold_answer,
        **payload,
    }


def build_case_records(groups: list[ScoredGroup]) -> list[dict]:
    cases: list[dict] = []
    for group in groups:
        greedy = _select_best(group.scored_samples, "greedy")
        quality = _select_best(group.scored_samples, "quality")
        majority = _select_best(group.scored_samples, "majority_vote")
        a_total = _select_best(group.scored_samples, "a_total")

        if (not greedy.score.answer_correct) and quality.score.answer_correct:
            cases.append(
                _build_case_record(
                    "greedy_wrong_quality_right",
                    group,
                    {
                        "selected_samples": {
                            "greedy": greedy.to_dict(),
                            "quality": quality.to_dict(),
                        }
                    },
                )
            )
        if (not quality.score.answer_correct) and majority.score.answer_correct:
            cases.append(
                _build_case_record(
                    "quality_wrong_majority_right",
                    group,
                    {
                        "selected_samples": {
                            "quality": quality.to_dict(),
                            "majority_vote": majority.to_dict(),
                        },
                        "answer_distribution": dict(
                            Counter(item.sample.final_answer for item in group.scored_samples)
                        ),
                    },
                )
            )
        placeholder_samples = [
            item.to_dict()
            for item in group.scored_samples
            if item.score.answer_extraction.get("placeholder_prediction")
            or item.score.answer_extraction.get("empty_prediction")
        ]
        if placeholder_samples:
            cases.append(
                _build_case_record(
                    "placeholder_extraction_failure",
                    group,
                    {
                        "samples": placeholder_samples,
                    },
                )
            )
        if _partition_name(group) == "all_correct":
            sorted_samples = sorted(
                group.scored_samples,
                key=lambda item: item.score.quality_score,
                reverse=True,
            )
            quality_scores = [item.score.quality_score for item in sorted_samples]
            variance = 0.0
            if quality_scores:
                mean_quality = sum(quality_scores) / len(quality_scores)
                variance = sum((value - mean_quality) ** 2 for value in quality_scores) / len(quality_scores)
            if variance > 1e-4:
                cases.append(
                    _build_case_record(
                        "all_correct_quality_spread",
                        group,
                        {
                            "quality_variance": variance,
                            "samples": [
                                sorted_samples[0].to_dict(),
                                sorted_samples[-1].to_dict(),
                            ],
                        },
                    )
                )
        correct_samples = [item for item in group.scored_samples if item.score.answer_correct]
        if correct_samples:
            mode_counts = Counter(item.score.mode_label for item in correct_samples)
            total_correct = len(correct_samples)
            rare_candidates = []
            for item in correct_samples:
                freq = mode_counts[item.score.mode_label] / total_correct
                rarity_score = item.score.quality_score * (-__import__("math").log(freq))
                rare_candidates.append((rarity_score, item, freq))
            rare_candidates.sort(key=lambda item: item[0], reverse=True)
            best_rarity, best_item, freq = rare_candidates[0]
            if best_rarity > 0:
                cases.append(
                    _build_case_record(
                        "rare_mode_high_quality",
                        group,
                        {
                            "selected_samples": {
                                "rare_mode_candidate": best_item.to_dict(),
                                "a_total": a_total.to_dict(),
                            },
                            "mode_frequency": freq,
                            "rarity_score": best_rarity,
                        },
                    )
                )
    return cases


def analyze_cases(groups: list[ScoredGroup], output_prefix: str) -> dict:
    cases = build_case_records(groups)
    case_path = f"{output_prefix}_cases.jsonl"
    summary_path = f"{output_prefix}_summary.json"
    summary = {
        "total_groups": len(groups),
        "total_cases": len(cases),
        "case_counts": dict(Counter(case["case_type"] for case in cases)),
        "case_file": case_path,
    }
    save_jsonl(case_path, cases)
    save_json(summary_path, summary)
    return {"case_path": case_path, "summary_path": summary_path, "summary": summary}
