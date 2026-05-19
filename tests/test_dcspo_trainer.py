import tempfile
import unittest
from pathlib import Path

from cmao.dcspo_trainer import load_dcspo_examples
from cmao.io_utils import save_json


class DCSPODataTest(unittest.TestCase):
    def test_loads_scored_groups_with_correct_quality_utility(self) -> None:
        payload = {
            "groups": [
                {
                    "problem": {"id": "p1", "source": "unit", "prompt": "What is 1+1?", "gold_answer": "2", "metadata": {}},
                    "scored_samples": [
                        {
                            "sample": {"problem_id": "p1", "sample_id": "s1", "cot_text": "", "final_answer": "2", "raw_text": "Final Answer: 2", "generation_meta": {}},
                            "score": {"answer_correct": True, "quality_score": 0.75},
                        },
                        {
                            "sample": {"problem_id": "p1", "sample_id": "s2", "cot_text": "", "final_answer": "3", "raw_text": "Final Answer: 3", "generation_meta": {}},
                            "score": {"answer_correct": False, "quality_score": 0.9},
                        },
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.json"
            save_json(path, payload)
            examples = load_dcspo_examples(path, utility_field="correct_quality")
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].utility, 0.75)
        self.assertEqual(examples[1].utility, 0.0)


if __name__ == "__main__":
    unittest.main()
