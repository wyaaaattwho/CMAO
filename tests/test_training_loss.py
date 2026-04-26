import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from cmao.training_loss import cmao_clipped_policy_loss


@unittest.skipIf(torch is None, "torch is not installed in the current environment")
class TrainingLossTest(unittest.TestCase):
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

    def test_clipped_policy_loss_is_masked_per_completion_token(self) -> None:
        current = torch.tensor([[10.0, 0.2, 0.1]], dtype=torch.float32)
        old = torch.zeros_like(current)
        advantages = torch.tensor([1.0], dtype=torch.float32)
        response_mask = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32)
        loss, breakdown = cmao_clipped_policy_loss(
            current_logprobs=current,
            old_logprobs=old,
            advantages=advantages,
            response_mask=response_mask,
            clip_range=0.2,
        )
        expected = -torch.tensor([(1.2 + torch.exp(torch.tensor(0.1))) / 2.0])
        self.assertTrue(torch.allclose(loss, expected, atol=1e-5))
        self.assertEqual(breakdown.clip_fraction, 0.5)

    def test_masked_prompt_tokens_cannot_create_nan_ratio(self) -> None:
        current = torch.tensor([[1000.0, 0.0]], dtype=torch.float32)
        old = torch.zeros_like(current)
        advantages = torch.tensor([1.0], dtype=torch.float32)
        response_mask = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        loss, breakdown = cmao_clipped_policy_loss(
            current_logprobs=current,
            old_logprobs=old,
            advantages=advantages,
            response_mask=response_mask,
            clip_range=0.2,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.tensor(breakdown.total_loss)))


if __name__ == "__main__":
    unittest.main()
