import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from cmao.training_loss import build_total_advantage, cmao_clipped_policy_loss


@unittest.skipIf(torch is None, "torch is not installed in the current environment")
class TrainingLossTest(unittest.TestCase):
    def test_build_total_advantage_combines_all_terms(self) -> None:
        a_ans = torch.tensor([1.0, -1.0])
        a_qual = torch.tensor([0.5, 0.0])
        a_mode = torch.tensor([0.2, 0.0])
        total = build_total_advantage(a_ans, a_qual, a_mode, 1.0, 0.5, 0.1)
        self.assertTrue(torch.allclose(total, torch.tensor([1.27, -1.0])))

    def test_clipped_policy_loss_returns_breakdown(self) -> None:
        current = torch.tensor([0.0, 0.2], dtype=torch.float32)
        old = torch.tensor([0.0, 0.0], dtype=torch.float32)
        advantages = torch.tensor([1.0, -0.5], dtype=torch.float32)
        kl_values = torch.tensor([0.05, 0.10], dtype=torch.float32)
        loss, breakdown = cmao_clipped_policy_loss(
            current_logprobs=current,
            old_logprobs=old,
            advantages=advantages,
            kl_values=kl_values,
            clip_range=0.2,
            kl_coef=0.1,
        )
        self.assertIsInstance(breakdown.total_loss, float)
        self.assertGreaterEqual(breakdown.clip_fraction, 0.0)
        self.assertEqual(loss.ndim, 0)
        self.assertGreaterEqual(breakdown.kl_term, 0.0)


if __name__ == "__main__":
    unittest.main()
