from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .io_utils import save_json
from .train_types import PolicyTrainingRecord
from .training_data import load_training_records
from .training_loss import build_total_advantage, cmao_clipped_policy_loss

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class TrainingConfig:
    model_name: str
    output_dir: str
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    max_steps: int | None = None
    max_length: int = 2048
    clip_range: float = 0.2
    kl_coef: float = 0.02
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42
    trust_remote_code: bool = True
    lambda_ans: float = 1.0
    lambda_qual: float = 0.5
    lambda_mode: float = 0.1
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    old_policy_source: str = "reference"


def training_config_from_dict(config: dict[str, Any]) -> TrainingConfig:
    model_cfg = dict(config.get("model", {}))
    train_cfg = dict(config.get("training", {}))
    cmao_cfg = dict(config.get("cmao", {}))
    lora_cfg = dict(config.get("lora", {}))
    return TrainingConfig(
        model_name=model_cfg["name"],
        output_dir=train_cfg["output_dir"],
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        batch_size=train_cfg.get("batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_epochs=train_cfg.get("num_epochs", 1),
        max_steps=train_cfg.get("max_steps"),
        max_length=train_cfg.get("max_length", 2048),
        clip_range=train_cfg.get("clip_range", 0.2),
        kl_coef=train_cfg.get("kl_coef", 0.02),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        seed=train_cfg.get("seed", 42),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        lambda_ans=cmao_cfg.get("lambda_ans", 1.0),
        lambda_qual=cmao_cfg.get("lambda_qual", 0.5),
        lambda_mode=cmao_cfg.get("lambda_mode", 0.1),
        lora_enabled=lora_cfg.get("enabled", True),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        lora_target_modules=lora_cfg.get("target_modules"),
        old_policy_source=cmao_cfg.get("old_policy_source", "reference"),
    )


class PolicyTrainingDataset:
    def __init__(self, records: list[PolicyTrainingRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PolicyTrainingRecord:
        return self.records[index]


class CMAOCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, records: list[PolicyTrainingRecord]) -> dict[str, Any]:
        prompts = [record.prompt for record in records]
        responses = [record.response_text for record in records]
        full_texts = [prompt.rstrip() + "\n" + response.lstrip() for prompt, response in zip(prompts, responses)]
        prompt_token_ids = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )["input_ids"]
        encoded = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_lengths = [min(len(token_ids), self.max_length - 1) for token_ids in prompt_token_ids]
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "prompt_lengths": prompt_lengths,
            "a_ans": encoded["attention_mask"].new_tensor(
                [record.advantage.a_ans for record in records], dtype=self._float_dtype()
            ),
            "a_qual": encoded["attention_mask"].new_tensor(
                [record.advantage.a_qual for record in records], dtype=self._float_dtype()
            ),
            "a_mode": encoded["attention_mask"].new_tensor(
                [record.advantage.a_mode for record in records], dtype=self._float_dtype()
            ),
            "correct": encoded["attention_mask"].new_tensor(
                [1.0 if record.answer_correct else 0.0 for record in records],
                dtype=self._float_dtype(),
            ),
            "quality_score": encoded["attention_mask"].new_tensor(
                [record.quality_score for record in records], dtype=self._float_dtype()
            ),
        }

    @staticmethod
    def _float_dtype():
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for CMAO training.") from exc
        return torch.float32


def _forward_response_stats(model, input_ids, attention_mask, prompt_lengths):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for CMAO training.") from exc

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:].float()

    full_log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = full_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    response_masks = []
    sequence_logprobs = []
    for index in range(input_ids.shape[0]):
        response_start = max(prompt_lengths[index] - 1, 0)
        response_mask = target_mask[index].clone()
        if response_start > 0:
            response_mask[:response_start] = 0.0
        response_masks.append(response_mask)
        denom = response_mask.sum().clamp_min(1.0)
        sequence_logprobs.append((token_logprobs[index] * response_mask).sum() / denom)
    return {
        "sequence_logprobs": torch.stack(sequence_logprobs),
        "response_mask": torch.stack(response_masks),
        "full_log_probs": full_log_probs,
    }


def _standard_kl_per_sample(current_log_probs, reference_log_probs, response_mask):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for CMAO training.") from exc

    current_probs = current_log_probs.exp()
    token_kl = (current_probs * (current_log_probs - reference_log_probs)).sum(dim=-1)
    denom = response_mask.sum(dim=-1).clamp_min(1.0)
    return (token_kl * response_mask).sum(dim=-1) / denom


class CMAOTrainer:
    def __init__(self, config: TrainingConfig, training_path: str | Path) -> None:
        self.config = config
        self.training_path = str(training_path)
        try:
            import torch
            from accelerate import Accelerator
            from accelerate.utils import set_seed
            from torch.optim import AdamW
            from torch.utils.data import DataLoader
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Training requires accelerate, torch, and transformers. "
                "Install the training dependencies before running this command."
            ) from exc

        self.torch = torch
        self.Accelerator = Accelerator
        self.set_seed = set_seed
        self.AdamW = AdamW
        self.DataLoader = DataLoader
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer

        self.records = load_training_records(training_path)
        self.dataset = PolicyTrainingDataset(self.records)
        self.accelerator = Accelerator(mixed_precision="bf16" if config.bf16 else "no")
        self.set_seed(config.seed)

        self.tokenizer = self.AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._build_trainable_model()
        self.reference_model = self._build_reference_model()
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad = False

        self.optimizer = self.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.collator = CMAOCollator(self.tokenizer, max_length=config.max_length)
        self.dataloader = self.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )
        self.model, self.reference_model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model,
            self.reference_model,
            self.optimizer,
            self.dataloader,
        )

    def _build_trainable_model(self):
        model = self.AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if not self.config.lora_enabled:
            return model
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LoRA training requires peft.") from exc
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.lora_target_modules,
        )
        return get_peft_model(model, lora_config)

    def _build_reference_model(self):
        return self.AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

    def train(self) -> dict[str, Any]:
        self.model.train()
        global_step = 0
        optimizer_step = 0
        history: list[dict[str, Any]] = []
        for epoch in range(self.config.num_epochs):
            progress = self._progress_wrapper(epoch)
            for batch_index, batch in enumerate(progress, start=1):
                current_stats = _forward_response_stats(
                    self.model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["prompt_lengths"],
                )
                with self.torch.no_grad():
                    reference_stats = _forward_response_stats(
                        self.reference_model,
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["prompt_lengths"],
                    )
                    old_logprobs = reference_stats["sequence_logprobs"]
                advantages = build_total_advantage(
                    batch["a_ans"],
                    batch["a_qual"],
                    batch["a_mode"],
                    lambda_ans=self.config.lambda_ans,
                    lambda_qual=self.config.lambda_qual,
                    lambda_mode=self.config.lambda_mode,
                )
                kl_values = _standard_kl_per_sample(
                    current_log_probs=current_stats["full_log_probs"],
                    reference_log_probs=reference_stats["full_log_probs"],
                    response_mask=current_stats["response_mask"],
                )
                loss, breakdown = cmao_clipped_policy_loss(
                    current_logprobs=current_stats["sequence_logprobs"],
                    old_logprobs=old_logprobs,
                    advantages=advantages,
                    kl_values=kl_values,
                    clip_range=self.config.clip_range,
                    kl_coef=self.config.kl_coef,
                )
                global_step += 1
                scaled_loss = loss / self.config.gradient_accumulation_steps
                self.accelerator.backward(scaled_loss)

                if batch_index % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    optimizer_step += 1

                if global_step % self.config.logging_steps == 0:
                    record = {
                        "step": global_step,
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "loss": breakdown.total_loss,
                        "policy_loss": breakdown.policy_loss,
                        "kl": breakdown.kl_term,
                        "clip_fraction": breakdown.clip_fraction,
                        "a_ans_mean": float(batch["a_ans"].mean().item()),
                        "a_qual_mean": float(batch["a_qual"].mean().item()),
                        "a_mode_mean": float(batch["a_mode"].mean().item()),
                        "correct_ratio": float(batch["correct"].mean().item()),
                        "quality_score_mean": float(batch["quality_score"].mean().item()),
                        "active_a_qual_ratio": float((batch["a_qual"].abs() > 0).float().mean().item()),
                        "active_a_mode_ratio": float((batch["a_mode"].abs() > 0).float().mean().item()),
                    }
                    history.append(record)
                    if tqdm is not None:
                        progress.set_postfix(
                            loss=f"{record['loss']:.4f}",
                            kl=f"{record['kl']:.4f}",
                            aq=f"{record['a_qual_mean']:.3f}",
                            am=f"{record['a_mode_mean']:.3f}",
                        )
                if optimizer_step > 0 and optimizer_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"checkpoint-step-{optimizer_step}")
                if self.config.max_steps and global_step >= self.config.max_steps:
                    if batch_index % self.config.gradient_accumulation_steps != 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        optimizer_step += 1
                    if tqdm is not None:
                        progress.close()
                    return self._finalize(history, global_step, optimizer_step)
            if batch_index % self.config.gradient_accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                optimizer_step += 1
            if tqdm is not None:
                progress.close()
        return self._finalize(history, global_step, optimizer_step)

    def _progress_wrapper(self, epoch: int):
        if tqdm is None:
            return self.dataloader
        total = len(self.dataloader)
        return tqdm(
            self.dataloader,
            total=total,
            desc=f"Training epoch {epoch + 1}/{self.config.num_epochs}",
            unit="batch",
        )

    def _save_checkpoint(self, checkpoint_name: str) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        checkpoint_dir = output_dir / checkpoint_name
        unwrapped.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

    def _finalize(self, history: list[dict[str, Any]], global_step: int, optimizer_step: int) -> dict[str, Any]:
        self._save_checkpoint("checkpoint-final")
        output_dir = Path(self.config.output_dir)
        summary = {
            "training_path": self.training_path,
            "model_name": self.config.model_name,
            "output_dir": str(output_dir),
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "lambda_ans": self.config.lambda_ans,
            "lambda_qual": self.config.lambda_qual,
            "lambda_mode": self.config.lambda_mode,
            "history": history,
        }
        save_json(output_dir / "training_summary.json", summary)
        return summary


def run_train_policy(config_path: str, training_path: str) -> dict[str, Any]:
    config = training_config_from_dict(load_config(config_path))
    trainer = CMAOTrainer(config=config, training_path=training_path)
    return trainer.train()
