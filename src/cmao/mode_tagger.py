from __future__ import annotations

import re

from .types import ProblemRecord, ReasoningSample


class ModeTagger:
    labels = (
        "tool_integrated",
        "case_split",
        "backsolve_or_check",
        "enumeration_or_counting",
        "equation_manipulation",
        "direct_arithmetic",
        "other_math",
    )

    def tag(self, problem: ProblemRecord, sample: ReasoningSample) -> str:
        return self.tag_with_evidence(problem, sample)[0]

    def tag_with_evidence(
        self,
        problem: ProblemRecord,
        sample: ReasoningSample,
    ) -> tuple[str, dict[str, object]]:
        text = f"{problem.prompt}\n{sample.cot_text or sample.raw_text}".lower()
        checks = (
            ("tool_integrated", self._is_tool_integrated),
            ("case_split", self._is_case_split),
            ("backsolve_or_check", self._is_backsolve_or_check),
            ("enumeration_or_counting", self._is_enumeration_or_counting),
            ("equation_manipulation", self._is_equation_manipulation),
            ("direct_arithmetic", self._is_direct_arithmetic),
        )
        for label, fn in checks:
            matched, reason = fn(text)
            if matched:
                return label, {"matched_rule": reason, "confidence": 1.0, "label": label}
        return "other_math", {
            "matched_rule": "fallback_other_math",
            "confidence": 0.25,
            "label": "other_math",
        }

    def _is_tool_integrated(self, text: str) -> tuple[bool, str]:
        patterns = (
            (r"\bpython\b.*\bcompute\b", "python_compute"),
            (r"\brun\b.*\bcode\b", "run_code"),
            (r"\buse\b.*\bcalculator\b", "use_calculator"),
            (r"\bscript\b", "script_reference"),
        )
        for pattern, reason in patterns:
            if re.search(pattern, text):
                return True, reason
        return False, ""

    def _is_case_split(self, text: str) -> tuple[bool, str]:
        patterns = (
            (r"\bcase\s+\d", "numbered_case_split"),
            (r"\bconsider\s+case", "consider_case"),
            (r"\bif\b.+\belse\b", "if_else_branch"),
        )
        for pattern, reason in patterns:
            if re.search(pattern, text):
                return True, reason
        return False, ""

    def _is_backsolve_or_check(self, text: str) -> tuple[bool, str]:
        keywords = ("check", "verify", "substitute", "plug back", "sanity")
        for keyword in keywords:
            if keyword in text:
                return True, f"keyword:{keyword}"
        return False, ""

    def _is_enumeration_or_counting(self, text: str) -> tuple[bool, str]:
        patterns = (
            (r"\bcount\b", "count_keyword"),
            (r"\benumerate\b", "enumerate_keyword"),
            (r"\bways\b", "ways_keyword"),
            (r"\b1\.", "ordered_enumeration"),
            (r"\bfirst\b", "first_keyword"),
        )
        for pattern, reason in patterns:
            if re.search(pattern, text):
                return True, reason
        return False, ""

    def _is_equation_manipulation(self, text: str) -> tuple[bool, str]:
        if text.count("=") >= 2:
            return True, "multiple_equations"
        if "solve for" in text:
            return True, "solve_for_keyword"
        return False, ""

    def _is_direct_arithmetic(self, text: str) -> tuple[bool, str]:
        if re.search(r"\d+\s*[\+\-\*/]\s*\d+", text):
            return True, "simple_arithmetic_pattern"
        return False, ""
