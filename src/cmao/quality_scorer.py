from __future__ import annotations

import re
from collections import Counter

from .answer_judge import (
    answers_equivalent,
    extract_final_answer,
    is_placeholder_answer,
    normalize_math_text,
    try_parse_numeric_value,
)
from .types import ProblemRecord, ReasoningSample

SAFE_SYMBOLIC_PATTERN = re.compile(r"^[A-Za-z0-9_\(\)\[\]\{\}\^\+\-\*/\\\., ]+$")


class QualityScorer:
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        concise_token_cap: int = 320,
    ) -> None:
        self.weights = weights or {
            "format": 0.20,
            "local_check": 0.35,
            "structure": 0.20,
            "self_verify": 0.15,
            "concise": 0.10,
        }
        self.concise_token_cap = concise_token_cap

    def score(
        self,
        problem: ProblemRecord,
        sample: ReasoningSample,
    ) -> tuple[float, dict[str, float], dict[str, object]]:
        cot_text = sample.cot_text or sample.raw_text
        final_answer = sample.final_answer or extract_final_answer(sample.raw_text)
        format_score, format_evidence = self._format_score(final_answer, sample.raw_text)
        local_check_score, local_check_evidence = self._local_check_score(cot_text)
        structure_score, structure_evidence = self._structure_score(cot_text)
        self_verify_score, self_verify_evidence = self._self_verify_score(cot_text)
        concise_score, concise_evidence = self._concise_score(cot_text)
        answer_consistency = self._answer_consistency(final_answer, cot_text)
        reasoning_redundancy = self._reasoning_redundancy(cot_text)
        subscores = {
            "format": format_score,
            "local_check": local_check_score,
            "structure": structure_score,
            "self_verify": self_verify_score,
            "concise": concise_score,
        }
        total = sum(self.weights[key] * value for key, value in subscores.items())
        evidence = {
            "subscore_evidence": {
                "format": format_evidence,
                "local_check": local_check_evidence,
                "structure": structure_evidence,
                "self_verify": self_verify_evidence,
                "concise": concise_evidence,
            },
            "candidate_features": {
                "answer_consistency": answer_consistency,
                "reasoning_redundancy": reasoning_redundancy,
            },
            "applied_weights": dict(self.weights),
        }
        return max(0.0, min(1.0, total)), subscores, evidence

    def _format_score(self, final_answer: str, raw_text: str) -> tuple[float, dict[str, object]]:
        if not final_answer or is_placeholder_answer(final_answer):
            return 0.0, {"has_final_answer": False, "signals": [], "placeholder_only": True}
        score = 0.7
        lowered = raw_text.lower()
        signals: list[str] = []
        if "\\boxed{" in raw_text:
            score += 0.2
            signals.append("boxed")
        if "final answer" in lowered or "therefore" in lowered:
            score += 0.1
            signals.append("explicit_conclusion")
        return min(1.0, score), {
            "has_final_answer": True,
            "signals": signals,
        }

    def _local_check_score(self, cot_text: str) -> tuple[float, dict[str, object]]:
        equations = []
        checked_lines: list[dict[str, object]] = []
        ignored_errors = 0
        for line in cot_text.splitlines():
            cleaned = line.strip()
            if cleaned.count("=") < 1:
                continue
            segments = [part.strip() for part in cleaned.split("=") if part.strip()]
            if len(segments) < 2:
                continue
            segment_validities = []
            for left, right in zip(segments, segments[1:]):
                try:
                    left_value = try_parse_numeric_value(left)
                    right_value = try_parse_numeric_value(right)
                    if left_value is not None and right_value is not None:
                        is_valid = abs(left_value - right_value) < 1e-6
                        equations.append(is_valid)
                        segment_validities.append({"left": left, "right": right, "valid": is_valid})
                        continue
                    if self._is_safe_symbolic_pair(left, right) and answers_equivalent(left, right):
                        equations.append(True)
                        segment_validities.append({"left": left, "right": right, "valid": True})
                except Exception:
                    # Keep scoring resilient to malformed expressions in individual segments.
                    ignored_errors += 1
                    continue
            if segment_validities:
                checked_lines.append({"line": cleaned, "segments": segment_validities})
        if not equations:
            return 0.5, {
                "checked_equations": 0,
                "valid_equations": 0,
                "ignored_errors": ignored_errors,
                "lines": [],
            }
        score = sum(1.0 for item in equations if item) / len(equations)
        return score, {
            "checked_equations": len(equations),
            "valid_equations": sum(1 for item in equations if item),
            "ignored_errors": ignored_errors,
            "lines": checked_lines,
        }

    def _structure_score(self, cot_text: str) -> tuple[float, dict[str, object]]:
        lines = [line.strip() for line in cot_text.splitlines() if line.strip()]
        if not lines:
            return 0.0, {"line_count": 0, "step_markers": 0, "unique_ratio": 0.0}
        unique_ratio = len(set(lines)) / len(lines)
        repeated_penalty = 1.0 if unique_ratio > 0.75 else 0.75 * unique_ratio
        step_markers = sum(
            1 for line in lines if re.match(r"^(step\s+\d+|[\-\*\d]+\.)", line.lower())
        )
        marker_bonus = min(0.2, 0.05 * step_markers)
        line_target = 1.0 if 2 <= len(lines) <= 12 else max(0.4, 1.0 - abs(len(lines) - 7) * 0.08)
        token_counts = Counter(line.lower() for line in lines)
        dominant_ratio = max(token_counts.values()) / len(lines)
        diversity_bonus = max(0.0, 1.0 - max(0.0, dominant_ratio - 0.34))
        score = max(
            0.0,
            min(1.0, 0.5 * repeated_penalty + 0.2 * line_target + 0.2 * diversity_bonus + marker_bonus),
        )
        return score, {
            "line_count": len(lines),
            "step_markers": step_markers,
            "unique_ratio": unique_ratio,
            "dominant_ratio": dominant_ratio,
        }

    def _self_verify_score(self, cot_text: str) -> tuple[float, dict[str, object]]:
        lowered = cot_text.lower()
        keywords = ("check", "verify", "substitute", "plug back", "sanity", "confirm")
        hits = [keyword for keyword in keywords if keyword in lowered]
        has_equation = "=" in cot_text or "\\boxed{" in cot_text
        return (1.0 if hits and has_equation else 0.0), {"hits": hits, "has_equation_context": has_equation}

    def _concise_score(self, cot_text: str) -> tuple[float, dict[str, object]]:
        tokens = cot_text.split()
        if not tokens:
            return 0.0, {"token_count": 0, "overflow": 0}
        count = len(tokens)
        if count <= self.concise_token_cap:
            return 1.0, {"token_count": count, "overflow": 0}
        overflow = count - self.concise_token_cap
        return max(0.0, 1.0 - overflow / self.concise_token_cap), {
            "token_count": count,
            "overflow": overflow,
        }

    def _answer_consistency(self, final_answer: str, cot_text: str) -> float:
        if not final_answer.strip() or is_placeholder_answer(final_answer):
            return 0.0
        answer_norm = final_answer.strip()
        lowered = cot_text.lower()
        if answer_norm in cot_text or answer_norm.lower() in lowered:
            return 1.0
        return 0.5 if extract_final_answer(cot_text).strip() else 0.0

    def _reasoning_redundancy(self, cot_text: str) -> float:
        lines = [line.strip().lower() for line in cot_text.splitlines() if line.strip()]
        if len(lines) <= 1:
            return 0.0
        unique_ratio = len(set(lines)) / len(lines)
        return max(0.0, 1.0 - unique_ratio)

    def _is_safe_symbolic_pair(self, left: str, right: str) -> bool:
        left_norm = normalize_math_text(left)
        right_norm = normalize_math_text(right)
        if not left_norm or not right_norm:
            return False
        if len(left_norm) > 80 or len(right_norm) > 80:
            return False
        if left_norm.count("(") + right_norm.count("(") > 8:
            return False
        if "[" in left_norm or "]" in left_norm or "[" in right_norm or "]" in right_norm:
            return False
        if not SAFE_SYMBOLIC_PATTERN.match(left_norm):
            return False
        if not SAFE_SYMBOLIC_PATTERN.match(right_norm):
            return False
        return True
