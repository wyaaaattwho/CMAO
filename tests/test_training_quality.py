import unittest

from cmao.training_quality import TrainingQualityScorer
from cmao.types import (
    AdvantageBundle,
    ProblemRecord,
    ReasoningSample,
    ScoreBundle,
    ScoredGroup,
    ScoredSample,
)


class TrainingQualityTest(unittest.TestCase):
    def test_final_support_scores_supported_answer_higher(self) -> None:
        scorer = TrainingQualityScorer()
        problem = ProblemRecord(id="p0", source="gsm8k", prompt="Q", gold_answer="8")
        strong = ReasoningSample(
            problem_id="p0",
            sample_id="s0",
            cot_text="",
            final_answer="8",
            raw_text="We compute 5+3=8.\nTherefore, the answer is 8.",
        )
        weak = ReasoningSample(
            problem_id="p0",
            sample_id="s1",
            cot_text="",
            final_answer="8",
            raw_text="We inspect the problem.\nThis seems plausible.\nFinal Answer: 8",
        )
        strong_score, _, _ = scorer.score(problem, strong, answer_correct=True)
        weak_score, _, _ = scorer.score(problem, weak, answer_correct=True)
        self.assertGreater(strong_score, weak_score)

    def test_pairwise_quality_advantage_only_uses_correct_samples(self) -> None:
        scorer = TrainingQualityScorer()
        problem = ProblemRecord(id="p0", source="gsm8k", prompt="Q", gold_answer="8")
        group = ScoredGroup(
            problem=problem,
            scored_samples=[
                ScoredSample(
                    sample=ReasoningSample(
                        problem_id="p0",
                        sample_id="s0",
                        cot_text="",
                        final_answer="8",
                        raw_text="2+6=8\nTherefore, the answer is 8.",
                    ),
                    score=ScoreBundle(answer_correct=True, quality_score=0.0, mode_label="direct_arithmetic"),
                    advantage=AdvantageBundle(a_ans=1.0, a_qual=0.0, a_mode=0.0, a_total=1.0),
                ),
                ScoredSample(
                    sample=ReasoningSample(
                        problem_id="p0",
                        sample_id="s1",
                        cot_text="",
                        final_answer="8",
                        raw_text="We guess that 8 looks plausible.\nNo further derivation is needed.\nFinal Answer: 8",
                    ),
                    score=ScoreBundle(answer_correct=True, quality_score=0.0, mode_label="direct_arithmetic"),
                    advantage=AdvantageBundle(a_ans=1.0, a_qual=0.0, a_mode=0.0, a_total=1.0),
                ),
                ScoredSample(
                    sample=ReasoningSample(
                        problem_id="p0",
                        sample_id="s2",
                        cot_text="",
                        final_answer="7",
                        raw_text="Final Answer: 7",
                    ),
                    score=ScoreBundle(answer_correct=False, quality_score=0.0, mode_label="direct_arithmetic"),
                    advantage=AdvantageBundle(a_ans=-1.0, a_qual=0.0, a_mode=0.0, a_total=-1.0),
                ),
            ],
        )
        pairwise_advantages, _, _ = scorer.pairwise_quality_advantages(group)
        self.assertGreater(pairwise_advantages[0], pairwise_advantages[1])
        self.assertEqual(pairwise_advantages[2], 0.0)
