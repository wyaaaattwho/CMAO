import unittest

from cmao.trainer import online_grpo_config_from_dict


class OnlineGRPOConfigTest(unittest.TestCase):
    def test_online_config_reads_rollout_and_update_settings(self) -> None:
        config = online_grpo_config_from_dict(
            {
                "model": {"name": "test-model", "trust_remote_code": False},
                "dataset": {"name": "gsm8k", "split": "train", "limit": 8},
                "sampling": {"group_size": 6, "max_new_tokens": 128, "temperature": 0.8},
                "training": {
                    "output_dir": "outputs/test-online",
                    "rollout_batch_size": 2,
                    "mini_batch_size": 3,
                    "num_iterations": 5,
                    "update_epochs": 2,
                    "max_grad_norm": 0.7,
                },
                "cmao": {"lambda_ans": 1.0, "lambda_qual": 0.4, "lambda_mode": 0.1},
            }
        )
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.dataset_name, "gsm8k")
        self.assertEqual(config.rollout_batch_size, 2)
        self.assertEqual(config.group_size, 6)
        self.assertEqual(config.mini_batch_size, 3)
        self.assertEqual(config.num_iterations, 5)
        self.assertEqual(config.update_epochs, 2)
        self.assertAlmostEqual(config.max_grad_norm, 0.7)
        self.assertEqual(config.max_new_tokens, 128)
        self.assertAlmostEqual(config.lambda_qual, 0.4)


if __name__ == "__main__":
    unittest.main()
