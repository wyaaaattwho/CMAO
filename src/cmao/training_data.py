from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import load_jsonl, save_jsonl
from .train_types import PolicyTrainingRecord
from .training_quality import TrainingQualityConfig, TrainingQualityScorer
from .types import AdvantageBundle
from .types import ScoredGroup


def flatten_training_records(
    groups: list[ScoredGroup],
    quality_config: TrainingQualityConfig | None = None,
) -> list[PolicyTrainingRecord]:
    scorer = TrainingQualityScorer(config=quality_config)
    records: list[PolicyTrainingRecord] = []
    for group in groups:
        correct_count = sum(1 for item in group.scored_samples if item.score.answer_correct)
        mode_counter = Counter(
            item.score.mode_label for item in group.scored_samples if item.score.answer_correct
        )
        pairwise_a_qual, training_evidences, training_scores = scorer.pairwise_quality_advantages(group)
        for item in group.scored_samples:
            if item.advantage is None:
                raise ValueError(
                    f"Scored sample {item.sample.sample_id} is missing advantage. "
                    "Run the advantage stage before preparing training data."
                )
            sample_index = group.scored_samples.index(item)
            training_quality = training_scores[sample_index]
            pairwise_quality_advantage = pairwise_a_qual[sample_index]
            records.append(
                PolicyTrainingRecord(
                    problem_id=group.problem.id,
                    sample_id=item.sample.sample_id,
                    prompt=group.problem.prompt,
                    response_text=item.sample.raw_text or item.sample.cot_text,
                    final_answer=item.sample.final_answer,
                    answer_correct=item.score.answer_correct,
                    quality_score=training_quality,
                    quality_subscores=item.score.quality_subscores,
                    mode_label=item.score.mode_label,
                    advantage=AdvantageBundle(
                        a_ans=item.advantage.a_ans,
                        a_qual=pairwise_quality_advantage,
                        a_mode=item.advantage.a_mode,
                        a_total=item.advantage.a_ans + pairwise_quality_advantage + item.advantage.a_mode,
                    ),
                    metadata={
                        "dataset_name": group.problem.source,
                        "group_size": len(group.scored_samples),
                        "correct_count": correct_count,
                        "all_correct_group": correct_count == len(group.scored_samples),
                        "mode_frequency_in_correct": mode_counter.get(item.score.mode_label, 0),
                        "sampling_config": dict(group.metadata.get("sampling_config", {})),
                        "report_partition": group.metadata.get("report_partition", "uncomputed"),
                        "offline_quality_score": item.score.quality_score,
                        "training_quality_score": training_quality,
                        "training_quality_margin": scorer.config.pairwise_margin,
                        "training_quality_evidence": training_evidences[sample_index],
                    },
                )
            )
    return records


def summarize_training_records(records: list[PolicyTrainingRecord]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "record_count": 0,
            "correct_record_count": 0,
            "correct_rate": 0.0,
            "nonzero_a_qual_ratio": 0.0,
            "nonzero_a_mode_ratio": 0.0,
            "avg_quality_score": 0.0,
        }
    correct = sum(1 for record in records if record.answer_correct)
    nonzero_a_qual = sum(1 for record in records if abs(record.advantage.a_qual) > 0.0)
    nonzero_a_mode = sum(1 for record in records if abs(record.advantage.a_mode) > 0.0)
    return {
        "record_count": total,
        "correct_record_count": correct,
        "correct_rate": correct / total,
        "nonzero_a_qual_ratio": nonzero_a_qual / total,
        "nonzero_a_mode_ratio": nonzero_a_mode / total,
        "avg_quality_score": sum(record.quality_score for record in records) / total,
    }


def save_training_records(path: str | Path, records: list[PolicyTrainingRecord]) -> dict[str, Any]:
    save_jsonl(path, [record.to_dict() for record in records])
    return summarize_training_records(records)


def load_training_records(path: str | Path) -> list[PolicyTrainingRecord]:
    return [PolicyTrainingRecord.from_dict(payload) for payload in load_jsonl(path)]
