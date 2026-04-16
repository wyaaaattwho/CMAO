import tempfile
import unittest
from pathlib import Path

from cmao.training_data import (
    flatten_training_records,
    load_training_records,
    save_training_records,
    summarize_training_records,
)
from cmao.types import (
    AdvantageBundle,
    ProblemRecord,
    ReasoningSample,
    ScoreBundle,
    ScoredGroup,
    ScoredSample,
)


class TrainingDataTest(unittest.TestCase):
    def _build_group(self) -> ScoredGroup:
        problem = ProblemRecord(
            id="gsm8k-0",
            source="gsm8k",
            prompt="What is 1+1?",
            gold_answer="2",
        )
        sample0 = ScoredSample(
            sample=ReasoningSample(
                problem_id="gsm8k-0",
                sample_id="gsm8k-0-sample-0",
                cot_text="1+1=2",
                final_answer="2",
                raw_text="1+1=2\nFinal Answer: 2",
            ),
            score=ScoreBundle(
                answer_correct=True,
                quality_score=0.9,
                quality_subscores={"format": 1.0},
                mode_label="direct_arithmetic",
            ),
            advantage=AdvantageBundle(a_ans=1.0, a_qual=0.5, a_mode=0.2, a_total=1.7),
        )
        sample1 = ScoredSample(
            sample=ReasoningSample(
                problem_id="gsm8k-0",
                sample_id="gsm8k-0-sample-1",
                cot_text="1+1=3",
                final_answer="3",
                raw_text="1+1=3\nFinal Answer: 3",
            ),
            score=ScoreBundle(
                answer_correct=False,
                quality_score=0.1,
                quality_subscores={"format": 0.0},
                mode_label="direct_arithmetic",
            ),
            advantage=AdvantageBundle(a_ans=-1.0, a_qual=0.0, a_mode=0.0, a_total=-1.0),
        )
        return ScoredGroup(
            problem=problem,
            scored_samples=[sample0, sample1],
            metadata={"sampling_config": {"temperature": 0.7}},
        )

    def test_flatten_training_records_preserves_advantages(self) -> None:
        records = flatten_training_records([self._build_group()])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].advantage.a_ans, 1.0)
        self.assertEqual(records[1].advantage.a_ans, -1.0)
        self.assertEqual(records[0].advantage.a_qual, 0.0)
        self.assertTrue(records[0].metadata["mode_frequency_in_correct"] >= 1)

    def test_summary_counts_nonzero_advantages(self) -> None:
        records = flatten_training_records([self._build_group()])
        summary = summarize_training_records(records)
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["correct_record_count"], 1)
        self.assertAlmostEqual(summary["nonzero_a_qual_ratio"], 0.0)

    def test_save_and_load_training_records(self) -> None:
        records = flatten_training_records([self._build_group()])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.jsonl"
            save_training_records(path, records)
            loaded = load_training_records(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].sample_id, "gsm8k-0-sample-0")

    def test_flatten_training_records_uses_pairwise_training_quality(self) -> None:
        records = flatten_training_records([self._build_group()])
        self.assertAlmostEqual(records[0].advantage.a_ans, 1.0)
        self.assertEqual(records[0].advantage.a_qual, 0.0)
        self.assertIn("training_quality_score", records[0].metadata)


if __name__ == "__main__":
    unittest.main()
