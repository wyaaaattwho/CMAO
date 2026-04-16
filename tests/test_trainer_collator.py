import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from cmao.train_types import PolicyTrainingRecord
from cmao.trainer import CMAOCollator
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


if __name__ == "__main__":
    unittest.main()
