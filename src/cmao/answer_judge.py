from __future__ import annotations

import ast
import math
import re
from fractions import Fraction
from typing import Any

from .types import ProblemRecord, ReasoningSample

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_PATTERNS = [
    re.compile(r"final answer\s*[:：]\s*(.+)", re.IGNORECASE),
    re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE),
]
TRAILING_NUMBER_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")
INLINE_EQUATION_PATTERN = re.compile(r"([A-Za-z0-9\\\{\}\[\]\(\)\^\+\-\*/\., ]+\s=\s[^=]+)$")
CONCLUSION_PATTERNS = [
    re.compile(r"^(?:therefore|thus|so|hence)[,:\s]+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:the answer is|thus the answer is|therefore the answer is)\s+(.+)$", re.IGNORECASE),
]
LATEX_JUNK = [r"\left", r"\right", "$", ",", " "]
SIMPLE_EXPR_PATTERN = re.compile(r"^[0-9\.\-\+\*/\(\)]+$")
PLACEHOLDER_PREFIXES = (
    "the final answer is",
    "final answer",
    "answer",
    "therefore",
    "thus",
    "hence",
    "so",
)


def strip_latex_noise(text: str) -> str:
    result = text.strip()
    for token in LATEX_JUNK:
        result = result.replace(token, "")
    result = re.sub(r"\\text\{([^{}]*)\}", r"\1", result)
    result = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", result)
    result = re.sub(r"\\,", "", result)
    return result.strip(" .")


def latex_fraction_to_plain(text: str) -> str:
    result = text
    while "\\frac" in result:
        result = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", result)
        if "\\frac" in result:
            break
    return result


def extract_boxed_answer(text: str) -> str:
    matches = BOXED_PATTERN.findall(text)
    return matches[-1].strip() if matches else ""


def is_placeholder_answer(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    normalized = normalized.strip(" .:;,!-")
    if not normalized:
        return True
    return normalized in PLACEHOLDER_PREFIXES


def _clean_extracted_answer(text: str) -> str:
    cleaned = text.strip().strip(" .")
    cleaned = re.sub(r"^[\-\*\u2022]\s*", "", cleaned)
    if is_placeholder_answer(cleaned):
        return ""
    return cleaned


def extract_final_answer_with_evidence(text: str) -> tuple[str, dict[str, Any]]:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed, {"strategy": "boxed", "matched_text": boxed}

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        for pattern in FINAL_PATTERNS:
            match = pattern.search(line)
            if match:
                extracted = _clean_extracted_answer(match.group(1))
                if extracted:
                    return extracted, {"strategy": "explicit_answer", "matched_text": line}
    for line in reversed(lines):
        for pattern in CONCLUSION_PATTERNS:
            match = pattern.search(line)
            if match:
                extracted = _clean_extracted_answer(match.group(1))
                if extracted:
                    return extracted, {"strategy": "conclusion_line", "matched_text": line}
    for line in reversed(lines):
        equation_match = INLINE_EQUATION_PATTERN.search(line)
        if equation_match:
            extracted = _clean_extracted_answer(equation_match.group(1))
            if extracted:
                return extracted, {"strategy": "equation_tail", "matched_text": line}
    for line in reversed(lines):
        numbers = TRAILING_NUMBER_PATTERN.findall(line)
        if numbers:
            return numbers[-1], {"strategy": "trailing_number", "matched_text": line}
    fallback = _clean_extracted_answer(lines[-1] if lines else text.strip())
    if fallback:
        return fallback, {"strategy": "fallback_last_line", "matched_text": fallback}
    return "", {"strategy": "placeholder_only", "matched_text": lines[-1] if lines else text.strip()}


def extract_final_answer(text: str) -> str:
    return extract_final_answer_with_evidence(text)[0]


def normalize_math_text(text: str) -> str:
    normalized = latex_fraction_to_plain(strip_latex_noise(text))
    normalized = normalized.replace("{", "(").replace("}", ")")
    normalized = normalized.replace("^", "**")
    normalized = normalized.replace("\\cdot", "*")
    normalized = normalized.replace("\\times", "*")
    normalized = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", normalized)
    normalized = normalized.replace("−", "-")
    normalized = re.sub(r"\\boxed\((.+)\)", r"\1", normalized)
    normalized = re.sub(r"^(?:thefinalansweris|finalansweris|answeris|theansweris)[:=]?", "", normalized)
    normalized = re.sub(r"[=:]$", "", normalized)
    return normalized.strip()


class _NumericEvaluator(ast.NodeVisitor):
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
    )

    def visit(self, node: ast.AST) -> float:
        if not isinstance(node, self.allowed_nodes):
            raise ValueError(f"Unsupported node: {type(node).__name__}")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported binary operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError("Unsupported unary operator")

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            try:
                return float(node.value)
            except OverflowError as exc:
                raise ValueError("Numeric constant is too large to convert to float") from exc
        raise ValueError("Unsupported constant")


def try_parse_numeric_value(text: str) -> float | None:
    candidate = normalize_math_text(text)
    candidate = candidate.replace("%", "")
    if not candidate:
        return None
    if candidate.endswith("."):
        candidate = candidate[:-1]
    if "/" in candidate and SIMPLE_EXPR_PATTERN.match(candidate):
        try:
            return float(Fraction(candidate))
        except (ValueError, ZeroDivisionError):
            pass
    if SIMPLE_EXPR_PATTERN.match(candidate):
        try:
            node = ast.parse(candidate, mode="eval")
            return float(_NumericEvaluator().visit(node))
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            pass
    try:
        return float(candidate)
    except (TypeError, ValueError, OverflowError):
        return None


def answers_equivalent(predicted: str, gold: str) -> bool:
    if is_placeholder_answer(predicted):
        return False
    pred_value = try_parse_numeric_value(predicted)
    gold_value = try_parse_numeric_value(gold)
    if pred_value is not None and gold_value is not None:
        return math.isclose(pred_value, gold_value, rel_tol=1e-6, abs_tol=1e-6)

    pred_norm = normalize_math_text(predicted)
    gold_norm = normalize_math_text(gold)
    if pred_norm == gold_norm:
        return True

    try:
        import sympy  # type: ignore
    except ImportError:
        return False

    try:
        pred_expr = sympy.sympify(pred_norm)
        gold_expr = sympy.sympify(gold_norm)
        return bool(sympy.simplify(pred_expr - gold_expr) == 0)
    except Exception:
        return False


def extract_gold_answer_from_gsm8k(solution: str) -> str:
    marker = "####"
    if marker not in solution:
        return solution.strip()
    return solution.split(marker)[-1].strip()


def extract_gold_answer_from_math500(solution: str) -> str:
    extracted, _ = extract_final_answer_with_evidence(solution)
    return extracted or solution.strip()


class AnswerJudge:
    def evaluate(self, problem: ProblemRecord, sample: ReasoningSample) -> dict[str, Any]:
        provided = sample.final_answer.strip()
        extraction = {"strategy": "provided_final_answer", "matched_text": sample.final_answer}
        predicted = provided
        if (not predicted) or is_placeholder_answer(predicted):
            predicted, extraction = extract_final_answer_with_evidence(sample.raw_text)
        gold = problem.gold_answer
        if problem.source == "gsm8k":
            gold = extract_gold_answer_from_gsm8k(gold)
        elif problem.source == "math-500":
            gold = extract_gold_answer_from_math500(gold)
        is_correct = answers_equivalent(predicted, gold)
        normalized_predicted = normalize_math_text(predicted)
        normalized_gold = normalize_math_text(gold)
        extraction_empty = not bool(predicted.strip())
        extraction_placeholder = is_placeholder_answer(predicted)
        return {
            "answer_correct": is_correct,
            "predicted_answer": predicted,
            "gold_answer": gold,
            "normalized_predicted": normalized_predicted,
            "normalized_gold": normalized_gold,
            "answer_extraction": {
                **extraction,
                "empty_prediction": extraction_empty,
                "placeholder_prediction": extraction_placeholder,
                "dataset": problem.source,
            },
            "judgment_details": {
                "dataset": problem.source,
                "equivalent_after_normalization": normalized_predicted == normalized_gold,
                "post_extraction_mismatch": bool(predicted.strip()) and not is_correct,
                "nonempty_incorrect": bool(predicted.strip()) and not extraction_placeholder and not is_correct,
            },
        }
