import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from cmao.train_types import PolicyTrainingRecord
from cmao.trainer import CMAOCollator, _completion_mask_from_generated_ids, _forward_response_stats, _sampled_token_kl
from cmao.types import AdvantageBundle


class _FakeTokenizer:
    def __call__(
        self,
        texts,
        padding=False,
        truncation=True,
        max_length=2048,
        return_tensors=None,
        add_special_tokens=False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        encoded = []
        for text in texts:
            token_ids = [max(1, min(255, ord(ch))) for ch in text[:max_length]]
            encoded.append(token_ids or [1])
        if not padding:
            return {"input_ids": encoded}
        max_len = max(len(item) for item in encoded)
        padded = [item + [0] * (max_len - len(item)) for item in encoded]
        mask = [[1] * len(item) + [0] * (max_len - len(item)) for item in encoded]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": mask}


@unittest.skipIf(torch is None, "torch is not installed in the current environment")
class TrainerCollatorTest(unittest.TestCase):
    def test_collator_preserves_float_advantages(self) -> None:
        collator = CMAOCollator(_FakeTokenizer(), max_length=32)
        batch = collator(
            [
                PolicyTrainingRecord(
                    problem_id="p0",
                    sample_id="s0",
                    prompt="Q",
                    response_text="A",
                    final_answer="2",
                    answer_correct=True,
                    quality_score=0.801,
                    mode_label="direct_arithmetic",
                    advantage=AdvantageBundle(a_ans=1.0, a_qual=0.577, a_mode=0.125, a_total=1.702),
                )
            ]
        )
        self.assertEqual(batch["a_ans"].dtype, torch.float32)
        self.assertAlmostEqual(float(batch["a_qual"][0].item()), 0.577, places=5)
        self.assertAlmostEqual(float(batch["a_mode"][0].item()), 0.125, places=5)
        self.assertAlmostEqual(float(batch["quality_score"][0].item()), 0.801, places=5)

    def test_collator_treats_prompt_separator_as_prefix(self) -> None:
        collator = CMAOCollator(_FakeTokenizer(), max_length=32)
        batch = collator(
            [
                PolicyTrainingRecord(
                    problem_id="p0",
                    sample_id="s0",
                    prompt="Q",
                    response_text="A",
                    final_answer="A",
                    answer_correct=True,
                    quality_score=1.0,
                    mode_label="direct_arithmetic",
                    advantage=AdvantageBundle(a_ans=1.0, a_qual=0.0, a_mode=0.0, a_total=1.0),
                )
            ]
        )
        self.assertEqual(batch["prompt_lengths"], [2])

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
