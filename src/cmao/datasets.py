from __future__ import annotations

from pathlib import Path
from typing import Any

from .answer_judge import extract_gold_answer_from_gsm8k
from .io_utils import load_json, load_jsonl
from .types import ProblemRecord

HF_DATASETS = {
    "gsm8k": {"path": "openai/gsm8k", "default_split": "test", "config_name": "main"},
    "math-500": {"path": "HuggingFaceH4/MATH-500", "default_split": "test"},
    "math-lighteval": {
        "path": "DigitalLearningGmbH/MATH-lighteval",
        "default_split": "test",
    },
}


def _pick_first(record: dict[str, Any], candidates: tuple[str, ...], default: str = "") -> str:
    for key in candidates:
        value = record.get(key)
        if value is not None:
            return str(value)
    return default


def _record_to_problem(source: str, index: int, record: dict[str, Any]) -> ProblemRecord:
    prompt = _pick_first(
        record,
        (
            "prompt",
            "problem",
            "question",
            "input",
            "instruction",
            "query",
            "Question",
            "Problem",
        ),
    )
    gold = _pick_first(
        record,
        (
            "gold_answer",
            "answer",
            "solution",
            "final_answer",
            "target",
            "expected_answer",
            "reference_answer",
            "Answer",
            "Final Answer",
        ),
    )
    if source == "gsm8k":
        gold = extract_gold_answer_from_gsm8k(gold)
    metadata = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "prompt",
            "problem",
            "question",
            "input",
            "instruction",
            "query",
            "Question",
            "Problem",
            "gold_answer",
            "answer",
            "solution",
            "final_answer",
            "target",
            "expected_answer",
            "reference_answer",
            "Answer",
            "Final Answer",
        }
    }
    return ProblemRecord(
        id=str(record.get("id", f"{source}-{index}")),
        source=source,
        prompt=prompt,
        gold_answer=gold,
        metadata=metadata,
    )


def load_problems(
    dataset_name: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    path: str | None = None,
    config_name: str | None = None,
) -> list[ProblemRecord]:
    if path:
        return load_local_problems(path, dataset_name or "local", limit)
    if not dataset_name:
        raise ValueError("Either dataset_name or path must be provided.")
    return load_hf_problems(dataset_name, split=split, limit=limit, config_name=config_name)


def load_local_problems(path: str, source: str, limit: int | None = None) -> list[ProblemRecord]:
    target = Path(path)
    if target.suffix == ".jsonl":
        raw_records = load_jsonl(target)
    else:
        payload = load_json(target)
        raw_records = payload if isinstance(payload, list) else payload["records"]
    problems = [_record_to_problem(source, index, record) for index, record in enumerate(raw_records)]
    return problems[:limit] if limit else problems


def load_hf_problems(
    dataset_name: str,
    split: str | None = None,
    limit: int | None = None,
    config_name: str | None = None,
) -> list[ProblemRecord]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("datasets is required for Hugging Face dataset loading.") from exc

    # Support both predefined aliases (e.g. "gsm8k") and raw HF dataset ids
    # (e.g. "Maxwell-Jia/AIME_2024") without code changes.
    spec = HF_DATASETS.get(dataset_name, {"path": dataset_name, "default_split": "train"})
    dataset = load_dataset(
        spec["path"],
        config_name or spec.get("config_name"),
        split=split or spec["default_split"],
    )
    records = [dict(item) for item in dataset]
    problems = [_record_to_problem(dataset_name, index, record) for index, record in enumerate(records)]
    return problems[:limit] if limit else problems
