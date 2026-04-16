from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .answer_judge import answers_equivalent, extract_final_answer_with_evidence
from .types import ProblemRecord, ReasoningSample, ScoredGroup

LINE_EQUATION_PATTERN = re.compile(r"^\s*([^=\n]+?)\s*=\s*([^=\n]+?)\s*$")
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class TrainingQualityConfig:
    pairwise_margin: float = 0.2
    max_tail_lines: int = 4
    max_equation_checks: int = 6
    contradiction_clip: float = 1.0


class TrainingQualityScorer:
    def __init__(self, config: TrainingQualityConfig | None = None) -> None:
        self.config = config or TrainingQualityConfig()

    def score(
        self,
        problem: ProblemRecord,
        sample: ReasoningSample,
        *,
        answer_correct: bool,
    ) -> tuple[float, dict[str, float], dict[str, Any]]:
        lines = [line.strip() for line in sample.raw_text.splitlines() if line.strip()]
        final_answer = sample.final_answer.strip()

        final_support, final_support_evidence = self._final_support(lines, final_answer)
        consistency, consistency_evidence = self._consistency(lines, final_answer)
        step_validity, step_validity_evidence = self._step_validity(lines)
        contradiction_penalty, contradiction_evidence = self._contradiction_penalty(lines, final_answer)

        quality_score = max(
            0.0,
            min(
                1.0,
                0.40 * final_support
                + 0.30 * consistency
                + 0.20 * step_validity
                - 0.10 * contradiction_penalty,
            ),
        )

        subscores = {
            "final_support": final_support,
            "consistency": consistency,
            "step_validity": step_validity,
            "contradiction_penalty": contradiction_penalty,
        }
        evidence = {
            "answer_correct": answer_correct,
            "final_support": final_support_evidence,
            "consistency": consistency_evidence,
            "step_validity": step_validity_evidence,
            "contradiction_penalty": contradiction_evidence,
        }
        return quality_score, subscores, evidence

    def pairwise_quality_advantages(self, group: ScoredGroup) -> tuple[list[float], list[dict[str, Any]], list[float]]:
        training_scores: list[float] = []
        evidences: list[dict[str, Any]] = []
        for item in group.scored_samples:
            score, _, evidence = self.score(
                group.problem,
                item.sample,
                answer_correct=item.score.answer_correct,
            )
            training_scores.append(score)
            evidences.append(evidence)

        pairwise_advantages = [0.0 for _ in group.scored_samples]
        correct_indices = [idx for idx, item in enumerate(group.scored_samples) if item.score.answer_correct]
        if len(correct_indices) < 2:
            return pairwise_advantages, evidences, training_scores

        margin = self.config.pairwise_margin
        denom = max(1, len(correct_indices) - 1)
        for idx in correct_indices:
            wins = 0
            losses = 0
            for other_idx in correct_indices:
                if idx == other_idx:
                    continue
                diff = training_scores[idx] - training_scores[other_idx]
                if diff > margin:
                    wins += 1
                elif diff < -margin:
                    losses += 1
            pairwise_advantages[idx] = max(-1.0, min(1.0, (wins - losses) / denom))
        return pairwise_advantages, evidences, training_scores

    def _final_support(self, lines: list[str], final_answer: str) -> tuple[float, dict[str, Any]]:
        if not final_answer:
            return 0.0, {"matched": False, "reason": "empty_final_answer"}
        tail_lines = lines[-self.config.max_tail_lines :]
        equation_hits = []
        explicit_hits = []
        for line in tail_lines:
            equation_match = LINE_EQUATION_PATTERN.match(line)
            if equation_match and answers_equivalent(equation_match.group(2).strip(), final_answer):
                equation_hits.append(line)
                continue
            if self._line_supports_answer(line, final_answer):
                explicit_hits.append(line)
        if equation_hits:
            return 1.0, {"matched": True, "support_type": "equation", "lines": equation_hits}
        if explicit_hits:
            return 0.6, {"matched": True, "support_type": "explicit", "lines": explicit_hits}
        all_hits = [line for line in lines if self._line_supports_answer(line, final_answer)]
        if all_hits:
            return 0.3, {"matched": True, "support_type": "early_only", "lines": all_hits[-2:]}
        return 0.0, {"matched": False, "reason": "no_supporting_line"}

    def _consistency(self, lines: list[str], final_answer: str) -> tuple[float, dict[str, Any]]:
        candidates = self._extract_candidate_answers(lines)
        if not final_answer:
            return 0.0, {"candidate_count": len(candidates), "conflicts": []}
        conflicting = [candidate for candidate in candidates if not answers_equivalent(candidate, final_answer)]
        if not conflicting:
            return 1.0, {"candidate_count": len(candidates), "conflicts": []}
        if len(conflicting) == len(candidates):
            return 0.0, {"candidate_count": len(candidates), "conflicts": conflicting[:4]}
        score = max(0.0, 1.0 - len(conflicting) / max(1, len(candidates)))
        return score, {"candidate_count": len(candidates), "conflicts": conflicting[:4]}

    def _step_validity(self, lines: list[str]) -> tuple[float, dict[str, Any]]:
        checked = 0
        valid = 0
        checked_examples: list[dict[str, Any]] = []
        for line in lines:
            if checked >= self.config.max_equation_checks:
                break
            match = LINE_EQUATION_PATTERN.match(line)
            if not match:
                continue
            left = match.group(1).strip()
            right = match.group(2).strip()
            if len(left) > 64 or len(right) > 64:
                continue
            if not self._looks_mathy(left) or not self._looks_mathy(right):
                continue
            checked += 1
            equivalent = answers_equivalent(left, right)
            if equivalent:
                valid += 1
            checked_examples.append({"left": left, "right": right, "equivalent": equivalent})
        if checked == 0:
            return 0.5, {"checked_equations": 0, "valid_equations": 0, "examples": []}
        return valid / checked, {
            "checked_equations": checked,
            "valid_equations": valid,
            "examples": checked_examples,
        }

    def _contradiction_penalty(self, lines: list[str], final_answer: str) -> tuple[float, dict[str, Any]]:
        conflicting = []
        candidates = self._extract_candidate_answers(lines)
        for candidate in candidates:
            if final_answer and not answers_equivalent(candidate, final_answer):
                conflicting.append(candidate)

        tail_numbers = []
        for line in lines[-self.config.max_tail_lines :]:
            tail_numbers.extend(NUMBER_PATTERN.findall(line))
        distinct_tail_numbers: list[str] = []
        for number in tail_numbers:
            if all(not answers_equivalent(number, seen) for seen in distinct_tail_numbers):
                distinct_tail_numbers.append(number)
        numeric_conflict = 1 if len(distinct_tail_numbers) >= 2 else 0

        raw_penalty = len(conflicting) / max(1, len(candidates)) + numeric_conflict
        clipped = max(0.0, min(self.config.contradiction_clip, raw_penalty))
        return clipped, {
            "conflicting_candidates": conflicting[:4],
            "distinct_tail_numbers": distinct_tail_numbers[:4],
            "raw_penalty": raw_penalty,
        }

    def _line_supports_answer(self, line: str, final_answer: str) -> bool:
        extracted, _ = extract_final_answer_with_evidence(line)
        if extracted and answers_equivalent(extracted, final_answer):
            return True
        equation_match = LINE_EQUATION_PATTERN.match(line)
        if equation_match and answers_equivalent(equation_match.group(2).strip(), final_answer):
            return True
        return final_answer in line

    def _extract_candidate_answers(self, lines: list[str]) -> list[str]:
        candidates: list[str] = []
        for line in lines[-(self.config.max_tail_lines + 2) :]:
            extracted, evidence = extract_final_answer_with_evidence(line)
            if extracted and evidence["strategy"] != "trailing_number":
                candidates.append(extracted)
                continue
            match = LINE_EQUATION_PATTERN.match(line)
            if match:
                candidates.append(match.group(2).strip())
        unique: list[str] = []
        for candidate in candidates:
            if all(not answers_equivalent(candidate, existing) for existing in unique):
                unique.append(candidate)
        return unique

    @staticmethod
    def _looks_mathy(text: str) -> bool:
        return any(char.isdigit() for char in text) or "=" in text or any(op in text for op in "+-*/()")
