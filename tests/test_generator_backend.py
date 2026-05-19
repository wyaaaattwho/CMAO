import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cmao.generator import TransformersGeneratorBackend


class GeneratorBackendTest(unittest.TestCase):
    def test_local_peft_adapter_loads_base_model(self) -> None:
        calls = {}

        class _Tokenizer:
            pad_token = None
            eos_token = "</s>"

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls["tokenizer"] = name
                return _Tokenizer()

        class _AutoModel:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls["base_model"] = name
                return object()

        class _PeftConfig:
            base_model_name_or_path = "base/model"

            @staticmethod
            def from_pretrained(name):
                calls["peft_config"] = name
                return _PeftConfig()

        class _PeftModel:
            @staticmethod
            def from_pretrained(model, name):
                calls["adapter"] = name
                return type("Model", (), {"eval": lambda self: None})()

        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp)
            (adapter_dir / "adapter_config.json").write_text("{}")
            (adapter_dir / "tokenizer_config.json").write_text("{}")
            with patch.dict(
                "sys.modules",
                {
                    "torch": object(),
                    "transformers": type("Transformers", (), {"AutoModelForCausalLM": _AutoModel, "AutoTokenizer": _AutoTokenizer}),
                    "peft": type("Peft", (), {"PeftConfig": _PeftConfig, "PeftModel": _PeftModel}),
                },
            ):
                backend = TransformersGeneratorBackend(str(adapter_dir))
        self.assertEqual(calls["tokenizer"], str(adapter_dir))
        self.assertEqual(calls["base_model"], "base/model")
        self.assertIn("adapter", calls)
        self.assertEqual(backend.model_name, str(adapter_dir))

    def test_outputs_checkpoint_missing_path_raises_clear_error(self) -> None:
        with patch.dict(
            "sys.modules",
            {
                "torch": object(),
                "transformers": type("Transformers", (), {"AutoModelForCausalLM": object, "AutoTokenizer": object}),
            },
        ):
            with self.assertRaises(FileNotFoundError):
                TransformersGeneratorBackend("outputs/train/missing/checkpoint-final")


if __name__ == "__main__":
    unittest.main()
