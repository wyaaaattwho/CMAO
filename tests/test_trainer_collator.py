import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from cmao.trainer import _completion_mask_from_generated_ids, _forward_response_stats, _sampled_token_kl


@unittest.skipIf(torch is None, "torch is not installed in the current environment")
class TrainerCollatorTest(unittest.TestCase):
    def test_forward_response_stats_masks_only_completion_tokens(self) -> None:
        class _UniformModel:
            def __call__(self, input_ids, attention_mask):
                vocab_size = 8
                logits = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    vocab_size,
                    dtype=torch.float32,
                )
                return type("Output", (), {"logits": logits})()

        input_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        stats = _forward_response_stats(_UniformModel(), input_ids, attention_mask, prompt_lengths=[2])
        self.assertTrue(torch.equal(stats["response_mask"], torch.tensor([[0.0, 1.0, 0.0]])))
        expected_logprob = -torch.log(torch.tensor(8.0))
        self.assertTrue(torch.allclose(stats["token_logprobs"][0, 1], expected_logprob))

    def test_sampled_token_kl_is_zero_when_policies_match(self) -> None:
        current = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
        reference = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(_sampled_token_kl(current, reference), torch.zeros_like(current)))

    def test_completion_mask_stops_at_first_generated_eos(self) -> None:
        input_ids = torch.tensor([10, 11, 20, 21, 2, 2], dtype=torch.long)
        mask, active_length = _completion_mask_from_generated_ids(input_ids, prompt_length=2, eos_token_id=2)
        self.assertEqual(active_length, 5)
        self.assertTrue(torch.equal(mask, torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
