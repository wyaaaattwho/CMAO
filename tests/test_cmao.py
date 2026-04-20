from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmao.cmao import CMAOComputer
from cmao.reporter import build_report
from cmao.types import AdvantageBundle, ProblemRecord, ReasoningSample, ScoreBundle, ScoredGroup, ScoredSample


class CMAOComputerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = ProblemRecord(
            id="p1",
            source="gsm8k",
            prompt="Prompt",
            gold_answer="10",
        )

    def _sample(self, sample_id: str, correct: bool, quality: float, mode: str) -> ScoredSample:
        return ScoredSample(
            sample=ReasoningSample(
                problem_id="p1",
                sample_id=sample_id,
                cot_text="Reasoning",
                final_answer="10",
                raw_text="Reasoning",
            ),
            score=ScoreBundle(
                answer_correct=correct,
                quality_score=quality,
                quality_subscores={},
                mode_label=mode,
                answer_extraction={},
                answer_judgment={},
            ),
            advantage=None,
        )

    def test_all_correct_group_gets_non_zero_quality_advantage(self) -> None:
        group = ScoredGroup(
            problem=self.problem,
            scored_samples=[
                self._sample("s1", True, 0.90, "equation_manipulation"),
                self._sample("s2", True, 0.55, "equation_manipulation"),
                self._sample("s3", True, 0.72, "case_split"),
            ],
        )
        updated = CMAOComputer().compute_group(group)
        qual_values = [item.advantage.a_qual for item in updated.scored_samples]
        self.assertEqual(qual_values, [0.5, -0.5, 0.0])

    def test_incorrect_samples_get_zero_quality_advantage(self) -> None:
        group = ScoredGroup(
            problem=self.problem,
            scored_samples=[
                self._sample("s1", True, 0.9, "equation_manipulation"),
                self._sample("s2", False, 0.8, "equation_manipulation"),
            ],
        )
        updated = CMAOComputer().compute_group(group)
        self.assertEqual(updated.scored_samples[1].advantage.a_qual, 0.0)

    def test_report_contains_subset_and_ablation_metrics(self) -> None:
        group = ScoredGroup(
            problem=self.problem,
            scored_samples=[
                self._sample("s1", True, 0.9, "equation_manipulation"),
                self._sample("s2", True, 0.6, "case_split"),
                self._sample("s3", False, 0.4, "other_math"),
            ],
        )
        updated = CMAOComputer().compute_group(group)
        report = build_report([updated])
        self.assertIn("per_subset_strategies", report)
        self.assertIn("quality_ablations", report)
        self.assertIn("group_diagnostics", report)
        self.assertEqual(report["per_subset_strategies"]["partially_correct"]["greedy"]["total"], 1)

    def test_report_contains_refined_extraction_metrics(self) -> None:
        broken = self._sample("s4", False, 0.2, "other_math")
        broken.score.answer_extraction = {"empty_prediction": False, "placeholder_prediction": True}
        broken.score.answer_judgment = {"nonempty_incorrect": False}
        report = build_report([ScoredGroup(problem=self.problem, scored_samples=[broken])])
        self.assertIn("empty_extraction_rate", report)
        self.assertIn("placeholder_extraction_rate", report)
        self.assertIn("nonempty_incorrect_rate", report)
        self.assertGreater(report["placeholder_extraction_rate"], 0.0)

    def test_report_contains_pass_at_k_metrics(self) -> None:
        group = ScoredGroup(
            problem=self.problem,
            scored_samples=[
                self._sample("s1", False, 0.9, "equation_manipulation"),
                self._sample("s2", True, 0.6, "case_split"),
                self._sample("s3", False, 0.4, "other_math"),
            ],
        )
        report = build_report([CMAOComputer().compute_group(group)])
        self.assertIn("pass_at_k", report)
        self.assertIn("per_subset_pass_at_k", report)
        self.assertEqual(report["pass_at_k"]["1"]["correct"], 0)
        self.assertEqual(report["pass_at_k"]["2"]["correct"], 1)
        self.assertEqual(
            report["per_subset_pass_at_k"]["partially_correct"]["2"]["pass_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
