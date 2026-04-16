from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmao.case_analysis import build_case_records
from cmao.types import AdvantageBundle, ProblemRecord, ReasoningSample, ScoreBundle, ScoredGroup, ScoredSample


class CaseAnalysisTest(unittest.TestCase):
    def _sample(self, sample_id: str, answer: str, correct: bool, quality: float, mode: str, total: float) -> ScoredSample:
        return ScoredSample(
            sample=ReasoningSample(
                problem_id="p1",
                sample_id=sample_id,
                cot_text=f"Reasoning for {answer}",
                final_answer=answer,
                raw_text=f"Reasoning for {answer}\nFinal Answer: {answer}",
            ),
            score=ScoreBundle(
                answer_correct=correct,
                quality_score=quality,
                quality_subscores={"format": quality, "structure": quality},
                mode_label=mode,
            ),
            advantage=AdvantageBundle(a_ans=0.0, a_qual=0.0, a_mode=0.0, a_total=total),
        )

    def test_build_case_records_detects_greedy_quality_case(self) -> None:
        group = ScoredGroup(
            problem=ProblemRecord(id="p1", source="gsm8k", prompt="Prompt", gold_answer="42"),
            scored_samples=[
                self._sample("s1", "41", False, 0.1, "other_math", -1.0),
                self._sample("s2", "42", True, 0.9, "case_split", 0.5),
                self._sample("s3", "42", True, 0.7, "equation_manipulation", 0.2),
            ],
        )
        records = build_case_records([group])
        case_types = {record["case_type"] for record in records}
        self.assertIn("greedy_wrong_quality_right", case_types)


if __name__ == "__main__":
    unittest.main()
