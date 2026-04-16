from __future__ import annotations

from pathlib import Path
from typing import Any

from .answer_judge import AnswerJudge, extract_final_answer
from .case_analysis import analyze_cases
from .cmao import CMAOComputer
from .config import load_config
from .datasets import load_problems
from .generator import SamplingConfig, TransformersGeneratorBackend
from .io_utils import load_json, save_json
from .mode_tagger import ModeTagger
from .quality_scorer import QualityScorer
from .reporter import build_report
from .trainer import run_train_online_grpo, run_train_policy
from .training_quality import TrainingQualityConfig
from .training_data import flatten_training_records, save_training_records
from .types import GroupedSamples, ScoredGroup, ScoredSample, ScoreBundle

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _load_grouped_samples(path: str | Path) -> list[GroupedSamples]:
    payload = load_json(path)
    return [GroupedSamples.from_dict(item) for item in payload["groups"]]


def _save_grouped_samples(path: str | Path, groups: list[GroupedSamples], metadata: dict[str, Any]) -> None:
    save_json(
        path,
        {
            "metadata": metadata,
            "groups": [group.to_dict() for group in groups],
        },
    )


def _load_scored_groups(path: str | Path) -> list[ScoredGroup]:
    payload = load_json(path)
    return [ScoredGroup.from_dict(item) for item in payload["groups"]]


def _save_scored_groups(path: str | Path, groups: list[ScoredGroup], metadata: dict[str, Any]) -> None:
    save_json(
        path,
        {
            "metadata": metadata,
            "groups": [group.to_dict() for group in groups],
        },
    )


def _with_progress(items: list, description: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=description, unit="problem")


def run_sample(config_path: str, output_path: str) -> None:
    config = load_config(config_path)
    problems = load_problems(
        dataset_name=config["dataset"].get("name"),
        split=config["dataset"].get("split"),
        limit=config["dataset"].get("limit"),
        path=config["dataset"].get("path"),
        config_name=config["dataset"].get("config_name"),
    )
    model_config = dict(config["model"])
    model_name = model_config.pop("name")
    backend = TransformersGeneratorBackend(model_name=model_name, **model_config)
    sampling_cfg = SamplingConfig(**config["sampling"])
    groups: list[GroupedSamples] = []
    for problem in _with_progress(problems, description="Sampling problems"):
        samples = backend.generate_group(problem, sampling_cfg, run_metadata={"config": config_path})
        for sample in samples:
            if not sample.final_answer:
                sample.final_answer = extract_final_answer(sample.raw_text)
        groups.append(GroupedSamples(problem=problem, samples=samples))
    _save_grouped_samples(output_path, groups, metadata={"config_path": config_path})


def run_score(input_path: str, output_path: str, config_path: str | None = None) -> None:
    scoring_config = load_config(config_path) if config_path else {}
    scorer = QualityScorer(
        weights=scoring_config.get("weights"),
        concise_token_cap=scoring_config.get("concise_token_cap", 320),
    )
    judge = AnswerJudge()
    mode_tagger = ModeTagger()
    grouped_samples = _load_grouped_samples(input_path)
    scored_groups: list[ScoredGroup] = []
    for group in grouped_samples:
        scored_samples: list[ScoredSample] = []
        for sample in group.samples:
            answer_info = judge.evaluate(group.problem, sample)
            sample.final_answer = answer_info["predicted_answer"]
            quality_score, subscores, quality_evidence = scorer.score(group.problem, sample)
            mode_label, mode_evidence = mode_tagger.tag_with_evidence(group.problem, sample)
            score_bundle = ScoreBundle(
                answer_correct=answer_info["answer_correct"],
                quality_score=quality_score,
                quality_subscores=subscores,
                mode_label=mode_label,
                quality_evidence=quality_evidence,
                mode_evidence=mode_evidence,
                answer_extraction=answer_info["answer_extraction"],
                answer_judgment=answer_info["judgment_details"],
            )
            scored_samples.append(ScoredSample(sample=sample, score=score_bundle))
        sampling_meta = {}
        if group.samples:
            first_meta = group.samples[0].generation_meta
            sampling_meta = {
                key: first_meta.get(key)
                for key in ("temperature", "top_p", "max_new_tokens", "model_name")
                if key in first_meta
            }
        scored_groups.append(
            ScoredGroup(
                problem=group.problem,
                scored_samples=scored_samples,
                metadata={
                    "dataset_name": group.problem.source,
                    "sampling_config": sampling_meta,
                    "report_partition": "uncomputed",
                },
            )
        )
    _save_scored_groups(output_path, scored_groups, metadata={"source": input_path, "config_path": config_path})


def run_advantage(input_path: str, output_path: str, config_path: str | None = None) -> None:
    config = load_config(config_path) if config_path else {}
    lambdas = config.get("lambdas", {})
    computer = CMAOComputer(
        lambda_ans=lambdas.get("ans", 1.0),
        lambda_qual=lambdas.get("qual", 1.0),
        lambda_mode=lambdas.get("mode", 1.0),
        quality_pairwise_margin=config.get("quality_pairwise_margin", config.get("pairwise_margin", 0.2)),
    )
    groups = _load_scored_groups(input_path)
    advantaged = [computer.compute_group(group) for group in groups]
    _save_scored_groups(output_path, advantaged, metadata={"source": input_path, "config_path": config_path})


def run_rerank_eval(input_path: str, output_path: str) -> None:
    groups = _load_scored_groups(input_path)
    report = build_report(groups)
    save_json(output_path, report)


def run_report(input_path: str) -> dict[str, Any]:
    payload = load_json(input_path)
    if "groups" in payload:
        groups = [ScoredGroup.from_dict(item) for item in payload["groups"]]
        return build_report(groups)
    return payload


def run_analyze_cases(input_path: str, output_prefix: str) -> dict[str, Any]:
    groups = _load_scored_groups(input_path)
    return analyze_cases(groups, output_prefix)


def save_report(input_path: str, output_path: str) -> dict[str, Any]:
    report = run_report(input_path)
    save_json(output_path, report)
    return report


def run_prepare_train_data(input_path: str, output_path: str) -> dict[str, Any]:
    groups = _load_scored_groups(input_path)
    records = flatten_training_records(groups, quality_config=TrainingQualityConfig())
    summary = save_training_records(output_path, records)
    save_json(
        f"{output_path}.summary.json",
        {
            "input_path": input_path,
            "output_path": output_path,
            "training_quality_mode": "correct_only_pairwise",
            "pairwise_margin": TrainingQualityConfig().pairwise_margin,
            **summary,
        },
    )
    return summary


def run_train(config_path: str, training_path: str) -> dict[str, Any]:
    return run_train_policy(config_path=config_path, training_path=training_path)


def run_train_online(config_path: str) -> dict[str, Any]:
    return run_train_online_grpo(config_path=config_path)
