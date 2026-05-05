from __future__ import annotations

import unittest

from integrations.swift.cmao_plugin import CMAORewardFunction, orms


class SwiftIntegrationTest(unittest.TestCase):
    def test_cmao_reward_plugin_scores_correct_completion_higher(self) -> None:
        reward = CMAORewardFunction()
        values = reward(
            [
                "Compute 40 + 2 = 42. Final Answer: 42",
                "Compute 40 + 2 = 41. Final Answer: 41",
            ],
            problem_id=["p1", "p1"],
            source=["unit", "unit"],
            prompt=["What is 40+2?", "What is 40+2?"],
            gold_answer=["42", "42"],
        )
        self.assertEqual(len(values), 2)
        self.assertGreater(values[0], values[1])

    def test_plugin_registers_swift_orm_name(self) -> None:
        self.assertIs(orms["cmao"], CMAORewardFunction)


if __name__ == "__main__":
    unittest.main()
