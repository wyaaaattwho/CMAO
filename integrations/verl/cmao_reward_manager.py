from __future__ import annotations

import asyncio
import math
import sys
from collections import Counter
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
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class CMAORewardManager(RewardManagerBase):
    """Group-aware CMAO reward manager for verl.

    verl calls reward managers once per sampled response. CMAO needs all responses
    from the same prompt, so this manager buffers responses by verl's repeated
    `uid`, computes correctness-gated quality and mode-diversity rewards for the
    whole group, then resolves each per-response reward call.
    """

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        reward_kwargs = _as_plain_dict(_cfg_get(config.reward, "reward_kwargs", {}))
        custom_reward_cfg = _cfg_get(config.reward, "custom_reward_function", {})
        custom_kwargs = _as_plain_dict(_cfg_get(custom_reward_cfg, "reward_kwargs", {}))
        reward_kwargs.update(custom_kwargs)

        self.lambda_ans = _safe_float(reward_kwargs.get("lambda_ans", 1.0), 1.0)
        self.lambda_qual = _safe_float(reward_kwargs.get("lambda_qual", 0.4), 0.4)
        self.lambda_mode = _safe_float(reward_kwargs.get("lambda_mode", 0.1), 0.1)
        self.quality_pairwise_margin = _safe_float(reward_kwargs.get("quality_pairwise_margin", 0.05), 0.05)
        self.quality_signal = str(reward_kwargs.get("quality_signal", "raw")).lower()
        self.quality_correct_only = _safe_bool(reward_kwargs.get("quality_correct_only", True), True)
        self.timeout_seconds = _safe_float(reward_kwargs.get("group_timeout_seconds", 600.0), 600.0)
        self.fallback_singleton = _safe_bool(reward_kwargs.get("fallback_singleton_on_timeout", True), True)

        concise_token_cap = int(reward_kwargs.get("concise_token_cap", 512))
        quality_weights = reward_kwargs.get("quality_weights")
        self.answer_judge = AnswerJudge()
        self.quality_scorer = QualityScorer(weights=quality_weights, concise_token_cap=concise_token_cap)
        self.mode_tagger = ModeTagger()

        self._pending: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    def _expected_group_size(self, data: DataProto) -> int:
        rollout_cfg = self.config.actor_rollout_ref.rollout
        if data.meta_info.get("validate"):
            val_kwargs = _cfg_get(rollout_cfg, "val_kwargs", None)
            return int(_cfg_get(val_kwargs, "n", 1) or 1)
        return int(_cfg_get(rollout_cfg, "n", 1) or 1)

    def _decode_response(self, data_item) -> str:
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    def _decode_prompt(self, data_item) -> str:
        if "prompts" not in data_item.batch:
            return ""
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        return self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

    def _build_record(self, data: DataProto) -> dict[str, Any]:
        data_item = data[0]
        uid = str(data_item.non_tensor_batch.get("uid", data_item.non_tensor_batch.get("index", "unknown")))
        extra_info = _as_plain_dict(data_item.non_tensor_batch.get("extra_info", {}))
        data_source = str(data_item.non_tensor_batch.get("data_source", extra_info.get("source", "math-500")))
        reward_model = _as_plain_dict(data_item.non_tensor_batch.get("reward_model", {}))
        ground_truth = str(reward_model.get("ground_truth", ""))
        response = self._decode_response(data_item)
        prompt = self._decode_prompt(data_item)
        future = self.loop.create_future()
        return {
            "uid": uid,
            "future": future,
            "data_source": data_source,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "response": response,
            "extra_info": extra_info,
        }

    def _score_group(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored = []
        for offset, record in enumerate(records):
            problem = ProblemRecord(
                id=str(record["extra_info"].get("problem_id", record["uid"])),
                source=str(record["data_source"]),
                prompt=str(record["prompt"]),
                gold_answer=str(record["ground_truth"]),
            )
            response = str(record["response"])
            sample = ReasoningSample(
                problem_id=problem.id,
                sample_id=f"{record['uid']}-{offset}",
                cot_text=response,
                final_answer=extract_final_answer(response),
                raw_text=response,
            )
            answer_info = self.answer_judge.evaluate(problem, sample)
            sample.final_answer = answer_info["predicted_answer"]
            raw_quality, subscores, quality_evidence = self.quality_scorer.score(problem, sample)
            correct = bool(answer_info["answer_correct"])
            quality = raw_quality if (correct or not self.quality_correct_only) else 0.0
            mode_label, mode_evidence = self.mode_tagger.tag_with_evidence(problem, sample)
            scored.append(
                {
                    "correct": correct,
                    "correctness": 1.0 if correct else 0.0,
                    "quality": quality,
                    "raw_quality": raw_quality,
                    "mode_label": mode_label,
                    "final_answer": sample.final_answer,
                    "quality_subscores": subscores,
                    "quality_evidence": quality_evidence,
                    "mode_evidence": mode_evidence,
                    "answer_extraction_strategy": answer_info["answer_extraction"].get("strategy", ""),
                }
            )

        correct_indices = [idx for idx, item in enumerate(scored) if item["correct"]]
        quality_rewards = [0.0 for _ in scored]
        if self.quality_signal == "pairwise" and len(correct_indices) >= 2:
            for left_pos, left_idx in enumerate(correct_indices):
                for right_idx in correct_indices[left_pos + 1 :]:
                    left_quality = scored[left_idx]["quality"]
                    right_quality = scored[right_idx]["quality"]
                    if left_quality - right_quality > self.quality_pairwise_margin:
                        quality_rewards[left_idx] += 1.0
                        quality_rewards[right_idx] -= 1.0
                    elif right_quality - left_quality > self.quality_pairwise_margin:
                        quality_rewards[left_idx] -= 1.0
                        quality_rewards[right_idx] += 1.0
            scale = float(max(1, len(correct_indices) - 1))
            quality_rewards = [value / scale for value in quality_rewards]
        else:
            quality_rewards = [item["quality"] if item["correct"] else 0.0 for item in scored]

        mode_rewards = [0.0 for _ in scored]
        if correct_indices:
            mode_counts = Counter(str(scored[idx]["mode_label"]) for idx in correct_indices)
            total_correct = len(correct_indices)
            for idx in correct_indices:
                probability = max(mode_counts[str(scored[idx]["mode_label"])] / total_correct, 1e-8)
                mode_rewards[idx] = scored[idx]["quality"] * (-math.log(probability))

        results = []
        for item, quality_reward, mode_reward in zip(scored, quality_rewards, mode_rewards, strict=True):
            reward = (
                self.lambda_ans * item["correctness"]
                + self.lambda_qual * quality_reward
                + self.lambda_mode * mode_reward
            )
            results.append(
                {
                    "reward_score": reward,
                    "reward_extra_info": {
                        "acc": item["correctness"],
                        "cmao_reward": reward,
                        "cmao_quality": item["quality"],
                        "cmao_quality_raw": item["raw_quality"],
                        "cmao_quality_reward": quality_reward,
                        "cmao_mode_reward": mode_reward,
                        "cmao_mode_label": item["mode_label"],
                        "cmao_final_answer": item["final_answer"],
                        "cmao_answer_extraction": item["answer_extraction_strategy"],
                    },
                }
            )
        return results

    async def _finish_group(self, uid: str, records: list[dict[str, Any]]) -> None:
        results = self._score_group(records)
        for record, result in zip(records, results, strict=True):
            if not record["future"].done():
                record["future"].set_result(result)

    async def _remove_record(self, record: dict[str, Any]) -> None:
        uid = record["uid"]
        async with self._lock:
            pending = self._pending.get(uid, [])
            self._pending[uid] = [item for item in pending if item is not record]
            if not self._pending[uid]:
                self._pending.pop(uid, None)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        record = self._build_record(data)
        uid = record["uid"]
        expected = self._expected_group_size(data)

        async with self._lock:
            group = self._pending.setdefault(uid, [])
            group.append(record)
            if len(group) >= expected:
                ready = group[:expected]
                remainder = group[expected:]
                if remainder:
                    self._pending[uid] = remainder
                else:
                    self._pending.pop(uid, None)
                await self._finish_group(uid, ready)

        try:
            return await asyncio.wait_for(record["future"], timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            await self._remove_record(record)
            if not self.fallback_singleton:
                raise
            return self._score_group([record])[0]
