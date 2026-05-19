from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer_judge import extract_final_answer
from .types import ProblemRecord, ReasoningSample


@dataclass
class SamplingConfig:
    group_size: int = 8
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 1024
    do_sample: bool = True


class GeneratorBackend:
    def generate_group(
        self,
        problem: ProblemRecord,
        sampling_cfg: SamplingConfig,
        run_metadata: dict[str, Any] | None = None,
    ) -> list[ReasoningSample]:
        raise NotImplementedError




def _resolve_local_model_path(model_name: str) -> Path | None:
    raw_path = Path(model_name).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _looks_like_local_model_path(model_name: str) -> bool:
    return (
        model_name.startswith(("/", "./", "../", "~", "outputs/", "data/"))
        or "/checkpoint-" in model_name
        or model_name.endswith("checkpoint-final")
    )

def format_chat_prompt(tokenizer, prompt: str, *, enable_thinking: bool = False) -> str:
    messages = [{"role": "user", "content": prompt.strip()}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt.strip()


class TransformersGeneratorBackend(GeneratorBackend):
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for sampling.") from exc

        self._torch = torch
        local_model_path = _resolve_local_model_path(model_name)
        if local_model_path is None and _looks_like_local_model_path(model_name):
            raise FileNotFoundError(
                f"Local model path does not exist: {model_name}. "
                "Use an absolute path or run from the repository root."
            )
        model_path = local_model_path or Path(model_name)
        adapter_config_path = model_path / "adapter_config.json"
        if local_model_path is not None and adapter_config_path.exists():
            try:
                from peft import PeftConfig, PeftModel  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Evaluating a LoRA adapter checkpoint requires peft.") from exc
            peft_config = PeftConfig.from_pretrained(str(model_path))
            base_model_name = peft_config.base_model_name_or_path
            tokenizer_source = str(model_path) if (model_path / "tokenizer_config.json").exists() else base_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            self.model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            model_source = str(local_model_path) if local_model_path is not None else model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        self.model.eval()
        self.model_name = model_name

    def generate_group(
        self,
        problem: ProblemRecord,
        sampling_cfg: SamplingConfig,
        run_metadata: dict[str, Any] | None = None,
    ) -> list[ReasoningSample]:
        prompt = format_chat_prompt(self.tokenizer, problem.prompt, enable_thinking=False)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **encoded,
            max_new_tokens=sampling_cfg.max_new_tokens,
            do_sample=sampling_cfg.do_sample,
            temperature=sampling_cfg.temperature,
            top_p=sampling_cfg.top_p,
            num_return_sequences=sampling_cfg.group_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        samples: list[ReasoningSample] = []
        for index, output_ids in enumerate(outputs):
            decoded = self.tokenizer.decode(output_ids[encoded["input_ids"].shape[1] :], skip_special_tokens=True)
            final_answer = extract_final_answer(decoded)
            cot_text = decoded
            samples.append(
                ReasoningSample(
                    problem_id=problem.id,
                    sample_id=f"{problem.id}-sample-{index}",
                    cot_text=cot_text,
                    final_answer=final_answer,
                    raw_text=decoded,
                    generation_meta={
                        "model_name": self.model_name,
                        "temperature": sampling_cfg.temperature,
                        "top_p": sampling_cfg.top_p,
                        "max_new_tokens": sampling_cfg.max_new_tokens,
                        "run_metadata": run_metadata or {},
                    },
                )
            )
        return samples
