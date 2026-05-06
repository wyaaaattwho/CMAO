import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_training_metrics import load_metrics, metric_xy_series


class PlotTrainingMetricsTest(unittest.TestCase):

    def test_loads_swift_jsonl_log_and_infers_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.log"
            path.write_text(
                "\n".join(
                    [
                        "[INFO:swift] warmup",
                        json.dumps(
                            {
                                "global_step/max_steps": "1/100",
                                "reward": 0.25,
                                "completions/clipped_ratio": 1.0,
                            }
                        ),
                        json.dumps({"global_step/max_steps": "2/100", "reward": 0.5}),
                    ]
                ),
                encoding="utf-8",
            )

            records = load_metrics(path)

        self.assertEqual([record["step"] for record in records], [1, 2])
        self.assertEqual([record["iteration"] for record in records], [1, 2])
        self.assertEqual([record["reward"] for record in records], [0.25, 0.5])

    def test_ignores_embedded_summary_when_step_logs_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.log"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"global_step/max_steps": "1/100", "reward": 0.25}),
                        json.dumps({"global_step/max_steps": "2/100", "reward": 0.5}),
                        json.dumps(
                            {
                                "global_step": 2,
                                "log_history": [
                                    {"step": 1, "reward": 0.25},
                                    {"step": 2, "reward": 0.5},
                                ],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            records = load_metrics(path)
            x_values, y_values = metric_xy_series(records, "reward", "step")

        self.assertEqual(len(records), 2)
        self.assertEqual(x_values, [1.0, 2.0])
        self.assertEqual(y_values, [0.25, 0.5])

    def test_loads_swift_log_history_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trainer_state.json"
            path.write_text(
                json.dumps(
                    {
                        "global_step": 2,
                        "log_history": [
                            {"step": 1, "reward": 0.25},
                            {"step": 2, "reward": 0.5},
                            {"train_loss": 0.1},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            records = load_metrics(path)
            x_values, y_values = metric_xy_series(records, "reward", "step")

        self.assertEqual(x_values, [1.0, 2.0])
        self.assertEqual(y_values, [0.25, 0.5])

    def test_metric_xy_series_sorts_and_deduplicates_steps(self):
        records = [
            {"step": 2, "reward": 0.5},
            {"step": 1, "reward": 0.25},
            {"step": 2, "reward": 0.75},
        ]

        x_values, y_values = metric_xy_series(records, "reward", "step")

        self.assertEqual(x_values, [1.0, 2.0])
        self.assertEqual(y_values, [0.25, 0.75])


if __name__ == "__main__":
    unittest.main()
