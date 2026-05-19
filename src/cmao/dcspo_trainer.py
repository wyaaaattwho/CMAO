from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .generator import format_chat_prompt
from .io_utils import load_json, load_jsonl, save_json
from .training_loss import dcspo_weighted_sft_loss
from .types import GroupedSamples, ScoredGroup

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class DCSPOExample:
    prompt: str
    completion: str
    utility: float = 1.0
    sample_id: str = ""


@dataclass
class DCSPOConfig:
    model_name: str
    output_dir: str
    data_path: str
    reference_model_name: str | None = None
    utility_field: str = "constant"
    max_examples: int | None = None
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    max_steps: int = 0
    max_length: int = 4096
    clip_range: float = 2.0
    normalize_by: str = "tokens"
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 1
    save_steps: int = 0
    seed: int = 42
    trust_remote_code: bool = True
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    lora_autocast_adapter_dtype: bool = False


def dcspo_config_from_dict(config: dict[str, Any]) -> DCSPOConfig:
    model_cfg = dict(config.get("model", {}))
    data_cfg = dict(config.get("data", config.get("dataset", {})))
    train_cfg = dict(config.get("training", {}))
    dcspo_cfg = dict(config.get("dcspo", {}))
    lora_cfg = dict(config.get("lora", {}))
    return DCSPOConfig(
        model_name=model_cfg["name"],
        reference_model_name=model_cfg.get("reference_name") or dcspo_cfg.get("reference_model_name"),
        output_dir=train_cfg["output_dir"],
        data_path=data_cfg.get("path") or data_cfg["offline_path"],
        utility_field=dcspo_cfg.get("utility_field", data_cfg.get("utility_field", "constant")),
        max_examples=data_cfg.get("limit"),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_epochs=train_cfg.get("num_epochs", 1),
        max_steps=train_cfg.get("max_steps", 0),
        max_length=train_cfg.get("max_length", 4096),
        clip_range=dcspo_cfg.get("clip_range", train_cfg.get("clip_range", 2.0)),
        normalize_by=dcspo_cfg.get("normalize_by", "tokens"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 1),
        save_steps=train_cfg.get("save_steps", 0),
        seed=train_cfg.get("seed", 42),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        lora_enabled=lora_cfg.get("enabled", True),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        lora_target_modules=lora_cfg.get("target_modules"),
        lora_autocast_adapter_dtype=lora_cfg.get("autocast_adapter_dtype", False),
    )


def _pick_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def _utility_from_score(score: Any, utility_field: str) -> float:
    mode = str(utility_field or "constant").lower()
    if mode in {"constant", "one", "sft"}:
        return 1.0
    if mode in {"correct", "correctness", "answer_correct"}:
        return 1.0 if bool(getattr(score, "answer_correct", False)) else 0.0
    if mode in {"quality", "quality_score"}:
        return max(0.0, float(getattr(score, "quality_score", 0.0)))
    if mode in {"correct_quality", "quality_if_correct"}:
        if not bool(getattr(score, "answer_correct", False)):
            return 0.0
        return max(0.0, float(getattr(score, "quality_score", 0.0)))
    return 1.0


def load_dcspo_examples(path: str | Path, utility_field: str = "constant", limit: int | None = None) -> list[DCSPOExample]:
    target = Path(path)
    payload: Any
    if target.suffix == ".jsonl":
        payload = load_jsonl(target)
    else:
        payload = load_json(target)

    examples: list[DCSPOExample] = []
    if isinstance(payload, dict) and "groups" in payload:
        for group_payload in payload["groups"]:
            if "scored_samples" in group_payload:
                group = ScoredGroup.from_dict(group_payload)
                for item in group.scored_samples:
                    examples.append(
                        DCSPOExample(
                            prompt=group.problem.prompt,
                            completion=item.sample.raw_text or item.sample.cot_text,
                            utility=_utility_from_score(item.score, utility_field),
                            sample_id=item.sample.sample_id,
                        )
                    )
            else:
                group = GroupedSamples.from_dict(group_payload)
                for sample in group.samples:
                    examples.append(
                        DCSPOExample(
                            prompt=group.problem.prompt,
                            completion=sample.raw_text or sample.cot_text,
                            utility=1.0,
                            sample_id=sample.sample_id,
                        )
                    )
    else:
        records = payload if isinstance(payload, list) else payload.get("records", [])
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            prompt = _pick_text(record, ("prompt", "query", "question", "problem", "instruction", "input"))
            completion = _pick_text(record, ("completion", "response", "output", "answer", "text", "solution"))
            if not prompt or not completion:
                continue
            utility = float(record.get("utility", record.get("score", 1.0)))
            examples.append(DCSPOExample(prompt=prompt, completion=completion, utility=max(0.0, utility), sample_id=str(record.get("id", index))))

    examples = [item for item in examples if item.prompt and item.completion and math.isfinite(item.utility)]
    return examples[:limit] if limit else examples


class OfflineDCSPOTrainer:
    def __init__(self, config: DCSPOConfig) -> None:
        self.config = config
        try:
            import torch
            from accelerate import Accelerator
            from accelerate import DistributedDataParallelKwargs
            from accelerate.utils import set_seed
            from torch.optim import AdamW
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DCSPO training requires accelerate, torch, and transformers.") from exc

        self.torch = torch
        self.Accelerator = Accelerator
        self.DistributedDataParallelKwargs = DistributedDataParallelKwargs
        self.set_seed = set_seed
        self.AdamW = AdamW
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer

        self.examples = load_dcspo_examples(config.data_path, config.utility_field, config.max_examples)
        if not self.examples:
            raise ValueError("DCSPO training requires at least one offline prompt/completion example.")

        ddp_kwargs = self.DistributedDataParallelKwargs(broadcast_buffers=False)
        self.accelerator = Accelerator(mixed_precision="bf16" if config.bf16 else "no", kwargs_handlers=[ddp_kwargs])
        self.set_seed(config.seed + int(self.accelerator.process_index))

        self.tokenizer = self.AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._build_trainable_model()
        self.reference_model = self._build_reference_model()
        self._disable_dropout(self.model)
        self._disable_dropout(self.reference_model)
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad = False

        self.optimizer = self.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.model, self.reference_model, self.optimizer = self.accelerator.prepare(self.model, self.reference_model, self.optimizer)
        self.output_dir = Path(config.output_dir)
        self.metrics_path = self.output_dir / "dcspo_metrics.jsonl"

    def _model_load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"trust_remote_code": self.config.trust_remote_code}
        if self.config.bf16:
            kwargs["torch_dtype"] = self.torch.bfloat16
        return kwargs

    def _build_trainable_model(self):
        model = self.AutoModelForCausalLM.from_pretrained(self.config.model_name, **self._model_load_kwargs())
        if hasattr(model, "config"):
            model.config.use_cache = False
        if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        if not self.config.lora_enabled:
            return model
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LoRA DCSPO training requires peft.") from exc
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.lora_target_modules,
        )
        try:
            model = get_peft_model(model, lora_config, autocast_adapter_dtype=self.config.lora_autocast_adapter_dtype)
        except TypeError:
            model = get_peft_model(model, lora_config)
        if self.config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        return model

    def _build_reference_model(self):
        name = self.config.reference_model_name or self.config.model_name
        model = self.AutoModelForCausalLM.from_pretrained(name, **self._model_load_kwargs())
        if hasattr(model, "config"):
            model.config.use_cache = False
        return model

    def _disable_dropout(self, model) -> None:
        for module in model.modules():
            if isinstance(module, self.torch.nn.Dropout):
                module.p = 0.0

    def _encode_example(self, example: DCSPOExample) -> dict[str, Any]:
        prompt_text = format_chat_prompt(self.tokenizer, example.prompt, enable_thinking=False)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(example.completion, add_special_tokens=False)["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            completion_ids = completion_ids + [int(self.tokenizer.eos_token_id)]
        max_completion = max(1, self.config.max_length - len(prompt_ids))
        completion_ids = completion_ids[:max_completion]
        input_ids = prompt_ids + completion_ids
        if len(input_ids) < 2:
            input_ids = input_ids + [self.tokenizer.eos_token_id or self.tokenizer.pad_token_id]
        target_length = len(input_ids) - 1
        response_mask = [0.0] * target_length
        for target_index in range(max(len(prompt_ids) - 1, 0), target_length):
            response_mask[target_index] = 1.0
        return {"input_ids": input_ids, "response_mask": response_mask, "utility": example.utility}

    def _collate(self, examples: list[DCSPOExample]) -> dict[str, Any]:
        encoded = [self._encode_example(item) for item in examples]
        pad_token_id = int(self.tokenizer.pad_token_id)
        max_length = max(len(item["input_ids"]) for item in encoded)
        input_ids = []
        attention_mask = []
        response_mask = []
        utilities = []
        for item in encoded:
            pad = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
            response_mask.append(item["response_mask"] + [0.0] * pad)
            utilities.append(float(item["utility"]))
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long, device=self.accelerator.device),
            "attention_mask": self.torch.tensor(attention_mask, dtype=self.torch.long, device=self.accelerator.device),
            "response_mask": self.torch.tensor(response_mask, dtype=self.torch.float32, device=self.accelerator.device)[:, : max_length - 1],
            "utilities": self.torch.tensor(utilities, dtype=self.torch.float32, device=self.accelerator.device),
        }

    def _forward_distribution_stats(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
        token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        probs = log_probs.exp()
        entropies = -(probs * log_probs).sum(dim=-1)
        return {"token_logprobs": token_logprobs, "entropies": entropies}

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        if not self.accelerator.is_main_process:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def train(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        global_step = 0
        backward_steps = 0
        self.optimizer.zero_grad()
        total_updates = self.config.max_steps if self.config.max_steps > 0 else self.config.num_epochs * math.ceil(len(self.examples) / self.config.per_device_train_batch_size)
        iterator = range(total_updates)
        progress = tqdm(iterator, desc="DCSPO", unit="step", disable=not self.accelerator.is_main_process) if tqdm is not None else iterator
        for step in progress:
            start = (step * self.config.per_device_train_batch_size) % len(self.examples)
            batch_examples = [self.examples[(start + offset) % len(self.examples)] for offset in range(self.config.per_device_train_batch_size)]
            batch = self._collate(batch_examples)
            current = self._forward_distribution_stats(self.model, batch["input_ids"], batch["attention_mask"])
            with self.torch.no_grad():
                reference = self._forward_distribution_stats(self.reference_model, batch["input_ids"], batch["attention_mask"])
            loss, breakdown = dcspo_weighted_sft_loss(
                current_logprobs=current["token_logprobs"],
                current_entropies=current["entropies"],
                reference_logprobs=reference["token_logprobs"],
                reference_entropies=reference["entropies"],
                utilities=batch["utilities"],
                response_mask=batch["response_mask"],
                clip_range=self.config.clip_range,
                normalize_by=self.config.normalize_by,
            )
            if not self.torch.isfinite(loss):
                self.optimizer.zero_grad()
                continue
            self.accelerator.backward(loss / self.config.gradient_accumulation_steps)
            backward_steps += 1
            record = {
                "step": step + 1,
                "optimizer_step": global_step,
                "loss": breakdown.total_loss,
                "policy_loss": breakdown.policy_loss,
                "dcspo/weight_mean": breakdown.weight_mean,
                "dcspo/weight_min": breakdown.weight_min,
                "dcspo/weight_max": breakdown.weight_max,
                "dcspo/delta_phi_mean": breakdown.delta_phi_mean,
                "dcspo/clipped_fraction": breakdown.delta_phi_clipped_fraction,
                "utility_mean": float(batch["utilities"].mean().detach().item()),
                "active_tokens": float(batch["response_mask"].sum().detach().item()),
            }
            if backward_steps % self.config.gradient_accumulation_steps == 0:
                if self.config.max_grad_norm and self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1
                record["optimizer_step"] = global_step
                if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"checkpoint-step-{global_step}")
            history.append(record)
            self._append_jsonl(record)
            if tqdm is not None and (step + 1) % self.config.logging_steps == 0:
                progress.set_postfix({"loss": f"{record['loss']:.4f}", "w": f"{record['dcspo/weight_mean']:.3f}", "clip": f"{record['dcspo/clipped_fraction']:.3f}"}, refresh=False)
            del batch, current, reference, loss
        if tqdm is not None:
            progress.close()
        self._save_checkpoint("checkpoint-final")
        summary = {
            "training_mode": "dcspo",
            "model_name": self.config.model_name,
            "reference_model_name": self.config.reference_model_name or self.config.model_name,
            "output_dir": str(self.output_dir),
            "data_path": self.config.data_path,
            "example_count": len(self.examples),
            "optimizer_step": global_step,
            "clip_range": self.config.clip_range,
            "utility_field": self.config.utility_field,
            "history": history,
        }
        save_json(self.output_dir / "training_summary.json", summary)
        return summary

    def _save_checkpoint(self, checkpoint_name: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self.output_dir / checkpoint_name
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)


def run_train_dcspo(config_path: str) -> dict[str, Any]:
    config = dcspo_config_from_dict(load_config(config_path))
    trainer = OfflineDCSPOTrainer(config)
    return trainer.train()
