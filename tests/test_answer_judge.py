from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmao.answer_judge import (
    AnswerJudge,
    answers_equivalent,
    extract_final_answer,
    extract_final_answer_with_evidence,
    is_placeholder_answer,
)
from cmao.types import ProblemRecord, ReasoningSample


class AnswerJudgeTest(unittest.TestCase):
    def test_extracts_boxed_answer(self) -> None:
        text = "We solve it carefully.\nThus the result is \\boxed{42}."
        self.assertEqual(extract_final_answer(text), "42")

    def test_numeric_equivalence_handles_fraction(self) -> None:
        self.assertTrue(answers_equivalent("\\frac{1}{2}", "0.5"))

    def test_gsm8k_answer_is_marked_correct(self) -> None:
        judge = AnswerJudge()
        problem = ProblemRecord(
            id="gsm8k-1",
            source="gsm8k",
            prompt="What is 20 + 22?",
            gold_answer="Reasoning here #### 42",
        )
        sample = ReasoningSample(
            problem_id="gsm8k-1",
            sample_id="s1",
            cot_text="20 + 22 = 42\nFinal Answer: 42",
            final_answer="42",
            raw_text="20 + 22 = 42\nFinal Answer: 42",
        )
        result = judge.evaluate(problem, sample)
        self.assertTrue(result["answer_correct"])

    def test_huge_integer_does_not_crash_equivalence_check(self) -> None:
        huge = "9" * 1000
        self.assertFalse(answers_equivalent(huge, "42"))

    def test_explicit_answer_beats_trailing_number(self) -> None:
        text = "We try some numbers 1, 2, 3.\nFinal Answer: 17\nResidual check 999"
        answer, evidence = extract_final_answer_with_evidence(text)
        self.assertEqual(answer, "17")
        self.assertEqual(evidence["strategy"], "explicit_answer")

    def test_conclusion_line_beats_tail_number(self) -> None:
        text = "Work omitted.\nTherefore, x = 12.\nSome token 999"
        answer, evidence = extract_final_answer_with_evidence(text)
        self.assertEqual(answer, "x = 12")
        self.assertEqual(evidence["strategy"], "conclusion_line")

    def test_placeholder_answer_is_detected(self) -> None:
        self.assertTrue(is_placeholder_answer("The final answer is:"))

    def test_placeholder_final_answer_falls_back_to_raw_text(self) -> None:
        judge = AnswerJudge()
        problem = ProblemRecord(
            id="math-1",
            source="math-500",
            prompt="Solve x+1=4",
            gold_answer="x = 3",
        )
        sample = ReasoningSample(
            problem_id="math-1",
            sample_id="s1",
            cot_text="Reasoning\nTherefore, x = 3",
            final_answer="The final answer is:",
            raw_text="Reasoning\nTherefore, x = 3",
        )
        result = judge.evaluate(problem, sample)
        self.assertEqual(result["predicted_answer"], "x = 3")
        self.assertEqual(result["answer_extraction"]["strategy"], "conclusion_line")
        self.assertTrue(result["answer_correct"])

    def test_equation_tail_extraction_for_math_answer(self) -> None:
        text = "Some derivation.\nHence, the solution is x = 7/2"
        answer, evidence = extract_final_answer_with_evidence(text)
        self.assertEqual(answer, "the solution is x = 7/2")
        self.assertIn(evidence["strategy"], {"conclusion_line", "equation_tail"})


if __name__ == "__main__":
    unittest.main()
