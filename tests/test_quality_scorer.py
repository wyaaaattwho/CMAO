from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmao.quality_scorer import QualityScorer
from cmao.types import ProblemRecord, ReasoningSample


class QualityScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = ProblemRecord(
            id="p1",
            source="gsm8k",
            prompt="What is 2 + 3?",
            gold_answer="5",
        )
        self.scorer = QualityScorer()

    def test_scores_well_structured_checked_answer_higher(self) -> None:
        strong = ReasoningSample(
            problem_id="p1",
            sample_id="s1",
            cot_text="Step 1. Compute 2 + 3 = 5.\nStep 2. Check: 5 is consistent.\nFinal Answer: 5",
            final_answer="5",
            raw_text="Step 1. Compute 2 + 3 = 5.\nStep 2. Check: 5 is consistent.\nFinal Answer: 5",
        )
        weak = ReasoningSample(
            problem_id="p1",
            sample_id="s2",
            cot_text="5",
            final_answer="5",
            raw_text="5",
        )
        strong_score, _, strong_evidence = self.scorer.score(self.problem, strong)
        weak_score, _, _ = self.scorer.score(self.problem, weak)
        self.assertGreater(strong_score, weak_score)
        self.assertIn("candidate_features", strong_evidence)
        self.assertIn("subscore_evidence", strong_evidence)

    def test_quality_evidence_matches_local_check(self) -> None:
        sample = ReasoningSample(
            problem_id="p1",
            sample_id="s3",
            cot_text="2 + 3 = 5\nFinal Answer: 5",
            final_answer="5",
            raw_text="2 + 3 = 5\nFinal Answer: 5",
        )
        _, subscores, evidence = self.scorer.score(self.problem, sample)
        self.assertGreater(subscores["local_check"], 0.9)
        local = evidence["subscore_evidence"]["local_check"]
        self.assertEqual(local["checked_equations"], 1)
        self.assertEqual(local["valid_equations"], 1)

    def test_placeholder_final_answer_gets_zero_format_score(self) -> None:
        sample = ReasoningSample(
            problem_id="p1",
            sample_id="s4",
            cot_text="The final answer is:",
            final_answer="The final answer is:",
            raw_text="The final answer is:",
        )
        _, subscores, evidence = self.scorer.score(self.problem, sample)
        self.assertEqual(subscores["format"], 0.0)
        self.assertTrue(evidence["subscore_evidence"]["format"]["placeholder_only"])

    def test_chain_equation_counts_for_local_check(self) -> None:
        sample = ReasoningSample(
            problem_id="p1",
            sample_id="s5",
            cot_text="2 + 3 = 5 = 10/2\nFinal Answer: 5",
            final_answer="5",
            raw_text="2 + 3 = 5 = 10/2\nFinal Answer: 5",
        )
        _, subscores, evidence = self.scorer.score(self.problem, sample)
        self.assertGreater(subscores["local_check"], 0.9)
        self.assertEqual(evidence["subscore_evidence"]["local_check"]["checked_equations"], 2)

    def test_safe_symbolic_guard_accepts_short_expressions(self) -> None:
        self.assertTrue(self.scorer._is_safe_symbolic_pair("x + 1", "2"))
        self.assertFalse(self.scorer._is_safe_symbolic_pair("[" * 90, "2"))


if __name__ == "__main__":
    unittest.main()
