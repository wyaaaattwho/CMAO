from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmao.mode_tagger import ModeTagger
from cmao.types import ProblemRecord, ReasoningSample


class ModeTaggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = ProblemRecord(
            id="p1",
            source="gsm8k",
            prompt="Solve the problem.",
            gold_answer="5",
        )
        self.tagger = ModeTagger()

    def test_detects_case_split(self) -> None:
        sample = ReasoningSample(
            problem_id="p1",
            sample_id="s1",
            cot_text="Case 1: x is even.\nCase 2: x is odd.",
            final_answer="5",
            raw_text="Case 1: x is even.\nCase 2: x is odd.",
        )
        self.assertEqual(self.tagger.tag(self.problem, sample), "case_split")

    def test_detects_backsolve_before_equation(self) -> None:
        sample = ReasoningSample(
            problem_id="p1",
            sample_id="s2",
            cot_text="We solve x + 2 = 5, then check by substitution.",
            final_answer="3",
            raw_text="We solve x + 2 = 5, then check by substitution.",
        )
        self.assertEqual(self.tagger.tag(self.problem, sample), "backsolve_or_check")

    def test_tightened_tool_integrated_rule_avoids_plain_python_mentions(self) -> None:
        sample = ReasoningSample(
            problem_id="p1",
            sample_id="s3",
            cot_text="This reminds me of python, but I check by substitution.",
            final_answer="3",
            raw_text="This reminds me of python, but I check by substitution.",
        )
        label, evidence = self.tagger.tag_with_evidence(self.problem, sample)
        self.assertEqual(label, "backsolve_or_check")
        self.assertEqual(evidence["matched_rule"], "keyword:check")

    def test_mode_evidence_contains_rule_name(self) -> None:
        sample = ReasoningSample(
            problem_id="p1",
            sample_id="s4",
            cot_text="Case 1: do this.\nCase 2: do that.",
            final_answer="5",
            raw_text="Case 1: do this.\nCase 2: do that.",
        )
        _, evidence = self.tagger.tag_with_evidence(self.problem, sample)
        self.assertIn("matched_rule", evidence)


if __name__ == "__main__":
    unittest.main()
