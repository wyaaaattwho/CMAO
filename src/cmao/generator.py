from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .answer_judge import extract_final_answer
from .types import ProblemRecord, ReasoningSample


@dataclass
class SamplingConfig:
    group_size: int = 8
    temperature: float = 0.7
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
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
        prompt = problem.prompt.strip()
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

