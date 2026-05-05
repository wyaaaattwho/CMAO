from __future__ import annotations

import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmao.answer_judge import AnswerJudge, extract_final_answer
from cmao.mode_tagger import ModeTagger
from cmao.quality_scorer import QualityScorer
from cmao.types import ProblemRecord, ReasoningSample

try:
    from swift.plugin import ORM, orms
except ImportError:  # Allows local syntax/unit checks without ms-swift installed.
    class ORM:  # type: ignore[no-redef]
        pass

    orms: dict[str, Any] = {}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if isinstance(completion.get("content"), str):
            return completion["content"]
        if isinstance(completion.get("text"), str):
            return completion["text"]
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(completion)


def _column_values(kwargs: dict[str, Any], key: str, n: int, default: str = "") -> list[str]:
    value = kwargs.get(key, default)
    if isinstance(value, list):
        items = value
    else:
        items = [value] * n
    if len(items) < n:
        items = items + [default] * (n - len(items))
    return ["" if item is None else str(item) for item in items[:n]]


def _messages_to_prompt(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("role") == "user":
                parts.append(str(item.get("content", "")))
            elif isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if value is None else str(value)


def _prompt_values(kwargs: dict[str, Any], n: int) -> list[str]:
    for key in ("prompt", "query", "question", "problem"):
        values = _column_values(kwargs, key, n)
        if any(values):
            return values
    messages = kwargs.get("messages", [""] * n)
    if not isinstance(messages, list):
        messages = [messages] * n
    if len(messages) < n:
        messages = messages + [""] * (n - len(messages))
    return [_messages_to_prompt(item) for item in messages[:n]]


class CMAORewardFunction(ORM):
    """CMAO reward function for ms-swift GRPO.

    The function returns raw per-completion scalar rewards. Swift/TRL GRPO owns
    group-relative normalization, so we only combine correctness, quality, and
    correct-mode diversity signals here.
    """

    def __init__(self) -> None:
        self.lambda_ans = _env_float("CMAO_LAMBDA_ANS", 1.0)
        self.lambda_qual = _env_float("CMAO_LAMBDA_QUAL", 0.4)
        self.lambda_mode = _env_float("CMAO_LAMBDA_MODE", 0.1)
        self.quality_signal = os.environ.get("CMAO_QUALITY_SIGNAL", "raw").strip().lower()
        self.quality_pairwise_margin = _env_float("CMAO_QUALITY_MARGIN", 0.05)
        self.quality_correct_only = _env_bool("CMAO_QUALITY_CORRECT_ONLY", True)
        concise_token_cap = int(_env_float("CMAO_CONCISE_TOKEN_CAP", 512))

        self.answer_judge = AnswerJudge()
        self.quality_scorer = QualityScorer(concise_token_cap=concise_token_cap)
        self.mode_tagger = ModeTagger()

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        texts = [_completion_text(item) for item in completions]
        n = len(texts)
        problem_ids = _column_values(kwargs, "problem_id", n)
        sources = _column_values(kwargs, "source", n, "swift")
        gold_answers = _column_values(kwargs, "gold_answer", n)
        solutions = _column_values(kwargs, "solution", n)
        prompts = _prompt_values(kwargs, n)

        scored: list[dict[str, Any]] = []
        for idx, text in enumerate(texts):
            gold = gold_answers[idx] or solutions[idx]
            problem = ProblemRecord(
                id=problem_ids[idx] or f"swift-{idx}",
                source=sources[idx] or "swift",
                prompt=prompts[idx],
                gold_answer=gold,
            )
            sample = ReasoningSample(
                problem_id=problem.id,
                sample_id=str(idx),
                cot_text=text,
                final_answer=extract_final_answer(text),
                raw_text=text,
            )
            answer_info = self.answer_judge.evaluate(problem, sample)
            sample.final_answer = answer_info["predicted_answer"]
            raw_quality, _, _ = self.quality_scorer.score(problem, sample)
            correct = bool(answer_info["answer_correct"])
            quality = raw_quality if (correct or not self.quality_correct_only) else 0.0
            mode_label = self.mode_tagger.tag(problem, sample)
            scored.append(
                {
                    "index": idx,
                    "problem_id": problem.id,
                    "correct": correct,
                    "quality": quality,
                    "mode_label": mode_label,
                }
            )

        rewards = [0.0 for _ in scored]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in scored:
            groups[str(item["problem_id"])].append(item)

        for group in groups.values():
            quality_rewards = self._quality_rewards(group)
            mode_rewards = self._mode_rewards(group)
            for item, quality_reward, mode_reward in zip(group, quality_rewards, mode_rewards, strict=True):
                reward = (
                    self.lambda_ans * (1.0 if item["correct"] else 0.0)
                    + self.lambda_qual * quality_reward
                    + self.lambda_mode * mode_reward
                )
                rewards[int(item["index"])] = reward if math.isfinite(reward) else 0.0
        return rewards

    def _quality_rewards(self, group: list[dict[str, Any]]) -> list[float]:
        if self.quality_signal != "pairwise":
            return [float(item["quality"]) if item["correct"] else 0.0 for item in group]

        rewards = [0.0 for _ in group]
        correct_indices = [idx for idx, item in enumerate(group) if item["correct"]]
        if len(correct_indices) < 2:
            return rewards
        for left_pos, left_idx in enumerate(correct_indices):
            for right_idx in correct_indices[left_pos + 1 :]:
                left_quality = float(group[left_idx]["quality"])
                right_quality = float(group[right_idx]["quality"])
                if left_quality - right_quality > self.quality_pairwise_margin:
                    rewards[left_idx] += 1.0
                    rewards[right_idx] -= 1.0
                elif right_quality - left_quality > self.quality_pairwise_margin:
                    rewards[left_idx] -= 1.0
                    rewards[right_idx] += 1.0
        scale = max(1, len(correct_indices) - 1)
        return [value / scale for value in rewards]

    def _mode_rewards(self, group: list[dict[str, Any]]) -> list[float]:
        rewards = [0.0 for _ in group]
        correct_indices = [idx for idx, item in enumerate(group) if item["correct"]]
        if len(correct_indices) < 2:
            return rewards
        counts = Counter(str(group[idx]["mode_label"]) for idx in correct_indices)
        total = len(correct_indices)
        for idx in correct_indices:
            probability = max(counts[str(group[idx]["mode_label"])] / total, 1e-8)
            rewards[idx] = float(group[idx]["quality"]) * (-math.log(probability))
        return rewards


orms["cmao"] = CMAORewardFunction
