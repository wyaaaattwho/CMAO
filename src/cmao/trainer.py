from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer_judge import AnswerJudge, extract_final_answer
from .cmao import CMAOComputer
from .config import load_config
from .datasets import load_problems
from .io_utils import ensure_parent, save_json
from .mode_tagger import ModeTagger
from .quality_scorer import QualityScorer
from .train_types import PolicyTrainingRecord
from .training_data import load_training_records
from .training_loss import build_total_advantage, cmao_clipped_policy_loss
from .types import ReasoningSample, ScoreBundle, ScoredGroup, ScoredSample

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
    max_grad_norm: float = 1.0
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


@dataclass
class OnlineGRPOConfig:
    model_name: str
    output_dir: str
    dataset_name: str | None = None
    dataset_split: str | None = None
    dataset_limit: int | None = None
    dataset_path: str | None = None
    dataset_config_name: str | None = None
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    rollout_batch_size: int = 1
    group_size: int = 4
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_iterations: int = 1
    update_epochs: int = 1
    max_new_tokens: int = 512
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    clip_range: float = 0.2
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0
    bf16: bool = True
    logging_steps: int = 1
    save_steps: int = 50
    seed: int = 42
    trust_remote_code: bool = True
    lambda_ans: float = 1.0
    lambda_qual: float = 0.5
    lambda_mode: float = 0.1
    quality_pairwise_margin: float = 0.2
    concise_token_cap: int = 320
    quality_weights: dict[str, float] | None = None
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    save_rollout_log: bool = True


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
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
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


def online_grpo_config_from_dict(config: dict[str, Any]) -> OnlineGRPOConfig:
    model_cfg = dict(config.get("model", {}))
    dataset_cfg = dict(config.get("dataset", {}))
    train_cfg = dict(config.get("training", {}))
    sampling_cfg = dict(config.get("sampling", {}))
    cmao_cfg = dict(config.get("cmao", {}))
    scoring_cfg = dict(config.get("scoring", {}))
    lora_cfg = dict(config.get("lora", {}))
    return OnlineGRPOConfig(
        model_name=model_cfg["name"],
        output_dir=train_cfg["output_dir"],
        dataset_name=dataset_cfg.get("name"),
        dataset_split=dataset_cfg.get("split"),
        dataset_limit=dataset_cfg.get("limit"),
        dataset_path=dataset_cfg.get("path"),
        dataset_config_name=dataset_cfg.get("config_name"),
        learning_rate=train_cfg.get("learning_rate", 1e-6),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        rollout_batch_size=train_cfg.get("rollout_batch_size", train_cfg.get("batch_size", 1)),
        group_size=sampling_cfg.get("group_size", train_cfg.get("group_size", 4)),
        mini_batch_size=train_cfg.get("mini_batch_size", sampling_cfg.get("group_size", 4)),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_iterations=train_cfg.get("num_iterations", train_cfg.get("max_steps", 1)),
        update_epochs=train_cfg.get("update_epochs", 1),
        max_new_tokens=sampling_cfg.get("max_new_tokens", 512),
        max_length=train_cfg.get("max_length", 2048),
        temperature=sampling_cfg.get("temperature", 0.7),
        top_p=sampling_cfg.get("top_p", 0.95),
        do_sample=sampling_cfg.get("do_sample", True),
        clip_range=train_cfg.get("clip_range", 0.2),
        kl_coef=train_cfg.get("kl_coef", 0.02),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 1),
        save_steps=train_cfg.get("save_steps", 50),
        seed=train_cfg.get("seed", 42),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        lambda_ans=cmao_cfg.get("lambda_ans", 1.0),
        lambda_qual=cmao_cfg.get("lambda_qual", 0.5),
        lambda_mode=cmao_cfg.get("lambda_mode", 0.1),
        quality_pairwise_margin=cmao_cfg.get(
            "quality_pairwise_margin",
            scoring_cfg.get("quality_pairwise_margin", 0.2),
        ),
        concise_token_cap=scoring_cfg.get("concise_token_cap", 320),
        quality_weights=scoring_cfg.get("weights"),
        lora_enabled=lora_cfg.get("enabled", True),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        lora_target_modules=lora_cfg.get("target_modules"),
        save_rollout_log=train_cfg.get("save_rollout_log", True),
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
        prefix_texts = [prompt.rstrip() + "\n" for prompt in prompts]
        full_texts = [prefix + response.lstrip() for prefix, response in zip(prefix_texts, responses)]
        prefix_token_ids = self.tokenizer(
            prefix_texts,
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
        prompt_lengths = [min(len(token_ids), self.max_length) for token_ids in prefix_token_ids]
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
    for index in range(input_ids.shape[0]):
        response_start = max(prompt_lengths[index] - 1, 0)
        response_mask = target_mask[index].clone()
        if response_start > 0:
            response_mask[:response_start] = 0.0
        response_masks.append(response_mask)
    return {
        "token_logprobs": token_logprobs,
        "response_mask": torch.stack(response_masks),
    }


def _sampled_token_kl(current_token_logprobs, reference_token_logprobs):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for CMAO training.") from exc

    log_ratio_ref_current = (reference_token_logprobs - current_token_logprobs).clamp(min=-20.0, max=20.0)
    return torch.exp(log_ratio_ref_current) - log_ratio_ref_current - 1.0


def _completion_mask_from_generated_ids(input_ids, prompt_length: int, eos_token_id: int | None):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for online GRPO training.") from exc

    target_length = max(int(input_ids.shape[0]) - 1, 0)
    response_mask = torch.zeros(target_length, dtype=torch.float32, device=input_ids.device)
    if prompt_length >= int(input_ids.shape[0]):
        return response_mask, int(input_ids.shape[0])

    completion_end = int(input_ids.shape[0]) - 1
    if eos_token_id is not None:
        for token_index in range(prompt_length, int(input_ids.shape[0])):
            if int(input_ids[token_index].item()) == int(eos_token_id):
                completion_end = token_index
                break

    for token_index in range(prompt_length, completion_end + 1):
        target_index = token_index - 1
        if 0 <= target_index < target_length:
            response_mask[target_index] = 1.0
    return response_mask, completion_end + 1


def _pad_1d_tensors(tensors, pad_value: int | float):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for online GRPO training.") from exc

    if not tensors:
        raise ValueError("Cannot pad an empty tensor list.")
    max_length = max(int(tensor.shape[0]) for tensor in tensors)
    padded = []
    for tensor in tensors:
        if int(tensor.shape[0]) == max_length:
            padded.append(tensor)
            continue
        pad_shape = (max_length - int(tensor.shape[0]),)
        pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        padded.append(torch.cat([tensor, pad], dim=0))
    return torch.stack(padded)


@dataclass
class OnlineRolloutBatch:
    input_ids: Any
    attention_mask: Any
    response_mask: Any
    old_token_logprobs: Any
    a_ans: Any
    a_qual: Any
    a_mode: Any
    correct: Any
    quality_score: Any
    groups: list[ScoredGroup]
    diagnostics: dict[str, Any]


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
        if config.old_policy_source != "reference":
            raise ValueError(
                "Only old_policy_source='reference' is supported by this offline trainer. "
                "Standard GRPO needs frozen behavior-policy logprobs for sampled completions; "
                "store those in PolicyTrainingRecord before using another old policy source."
            )
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
                advantages = build_total_advantage(
                    batch["a_ans"],
                    batch["a_qual"],
                    batch["a_mode"],
                    lambda_ans=self.config.lambda_ans,
                    lambda_qual=self.config.lambda_qual,
                    lambda_mode=self.config.lambda_mode,
                )
                kl_values = _sampled_token_kl(
                    current_token_logprobs=current_stats["token_logprobs"],
                    reference_token_logprobs=reference_stats["token_logprobs"],
                )
                loss, breakdown = cmao_clipped_policy_loss(
                    current_logprobs=current_stats["token_logprobs"],
                    old_logprobs=reference_stats["token_logprobs"],
                    advantages=advantages,
                    kl_values=kl_values,
                    response_mask=current_stats["response_mask"],
                    clip_range=self.config.clip_range,
                    kl_coef=self.config.kl_coef,
                )
                global_step += 1
                scaled_loss = loss / self.config.gradient_accumulation_steps
                self.accelerator.backward(scaled_loss)

                if batch_index % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm and self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
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
                        if self.config.max_grad_norm and self.config.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        optimizer_step += 1
                    if tqdm is not None:
                        progress.close()
                    return self._finalize(history, global_step, optimizer_step)
            if batch_index % self.config.gradient_accumulation_steps != 0:
                if self.config.max_grad_norm and self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
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


class OnlineGRPOTrainer:
    def __init__(self, config: OnlineGRPOConfig) -> None:
        self.config = config
        try:
            import torch
            from accelerate import Accelerator
            from accelerate.utils import set_seed
            from torch.optim import AdamW
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Online GRPO training requires accelerate, torch, and transformers. "
                "Install the training dependencies before running this command."
            ) from exc

        self.torch = torch
        self.Accelerator = Accelerator
        self.set_seed = set_seed
        self.AdamW = AdamW
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer

        self.problems = load_problems(
            dataset_name=config.dataset_name,
            split=config.dataset_split,
            limit=config.dataset_limit,
            path=config.dataset_path,
            config_name=config.dataset_config_name,
        )
        if not self.problems:
            raise ValueError("Online GRPO training requires at least one problem.")

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
        self.model, self.reference_model, self.optimizer = self.accelerator.prepare(
            self.model,
            self.reference_model,
            self.optimizer,
        )

        self.answer_judge = AnswerJudge()
        self.quality_scorer = QualityScorer(
            weights=config.quality_weights,
            concise_token_cap=config.concise_token_cap,
        )
        self.mode_tagger = ModeTagger()
        self.advantage_computer = CMAOComputer(
            lambda_ans=config.lambda_ans,
            lambda_qual=config.lambda_qual,
            lambda_mode=config.lambda_mode,
            quality_pairwise_margin=config.quality_pairwise_margin,
        )
        self.output_dir = Path(config.output_dir)
        self.metrics_path = self.output_dir / "online_metrics.jsonl"
        self.rollout_log_path = self.output_dir / "online_rollouts.jsonl"

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        if not self.accelerator.is_main_process:
            return
        target = ensure_parent(path)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

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
        history: list[dict[str, Any]] = []
        optimizer_step = 0
        problem_cursor = 0
        iterations = range(self.config.num_iterations)
        progress = (
            tqdm(
                iterations,
                total=self.config.num_iterations,
                desc="Online GRPO",
                unit="iter",
                disable=not self.accelerator.is_main_process,
            )
            if tqdm is not None
            else iterations
        )
        for iteration in progress:
            selected = []
            for _ in range(self.config.rollout_batch_size):
                selected.append(self.problems[problem_cursor % len(self.problems)])
                problem_cursor += 1
            rollout = self._collect_rollout(selected, rollout_step=iteration)
            update_summary = self._update_from_rollout(rollout)
            optimizer_step += update_summary["optimizer_steps"]

            record = {
                "iteration": iteration + 1,
                "optimizer_step": optimizer_step,
                "sample_count": int(rollout.input_ids.shape[0]),
                "correct_ratio": float(rollout.correct.mean().item()),
                "correct_count": int(rollout.correct.sum().item()),
                "quality_score_mean": float(rollout.quality_score.mean().item()),
                "a_ans_mean": float(rollout.a_ans.mean().item()),
                "a_qual_mean": float(rollout.a_qual.mean().item()),
                "a_mode_mean": float(rollout.a_mode.mean().item()),
                "a_total_abs_mean": float(
                    build_total_advantage(
                        rollout.a_ans,
                        rollout.a_qual,
                        rollout.a_mode,
                        lambda_ans=self.config.lambda_ans,
                        lambda_qual=self.config.lambda_qual,
                        lambda_mode=self.config.lambda_mode,
                    )
                    .abs()
                    .mean()
                    .item()
                ),
                "nonzero_advantage_ratio": float(
                    (
                        build_total_advantage(
                            rollout.a_ans,
                            rollout.a_qual,
                            rollout.a_mode,
                            lambda_ans=self.config.lambda_ans,
                            lambda_qual=self.config.lambda_qual,
                            lambda_mode=self.config.lambda_mode,
                        ).abs()
                        > 1e-8
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "response_tokens_mean": rollout.diagnostics["response_tokens_mean"],
                "problem_ids": rollout.diagnostics["problem_ids"],
                **update_summary,
            }
            history.append(record)
            self._append_jsonl(self.metrics_path, record)
            if self.config.save_rollout_log:
                self._append_jsonl(
                    self.rollout_log_path,
                    {
                        "iteration": iteration + 1,
                        "optimizer_step": optimizer_step,
                        **rollout.diagnostics,
                    },
                )

            if tqdm is not None and (iteration + 1) % self.config.logging_steps == 0:
                progress.set_postfix(
                    {
                        "loss": f"{record['loss']:.4f}",
                        "kl": f"{record['kl']:.4f}",
                        "clip": f"{record['clip_fraction']:.3f}",
                        "correct": f"{record['correct_count']}/{record['sample_count']}",
                        "adv": f"{record['nonzero_advantage_ratio']:.3f}",
                        "tok": f"{record['response_tokens_mean']:.1f}",
                        "pid": ",".join(record["problem_ids"]),
                    },
                    refresh=False,
                )
            if self.config.save_steps > 0 and optimizer_step > 0 and optimizer_step % self.config.save_steps == 0:
                self._save_checkpoint(f"checkpoint-step-{optimizer_step}")
        if tqdm is not None:
            progress.close()
        return self._finalize(history, rollout_step=self.config.num_iterations, optimizer_step=optimizer_step)

    def _collect_rollout(self, problems, rollout_step: int) -> OnlineRolloutBatch:
        self.model.eval()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        input_sequences = []
        attention_masks = []
        response_masks = []
        scored_groups: list[ScoredGroup] = []
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        with self.torch.no_grad():
            for problem in problems:
                prefix_text = problem.prompt.rstrip() + "\n"
                encoded = self.tokenizer(
                    prefix_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(1, self.config.max_length - self.config.max_new_tokens),
                )
                encoded = {key: value.to(self.accelerator.device) for key, value in encoded.items()}
                prompt_length = int(encoded["input_ids"].shape[1])
                generated = unwrapped_model.generate(
                    **encoded,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    num_return_sequences=self.config.group_size,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    remove_invalid_values=True,
                    renormalize_logits=True,
                )

                scored_samples: list[ScoredSample] = []
                for sample_index, output_ids in enumerate(generated):
                    response_mask, active_length = _completion_mask_from_generated_ids(
                        output_ids,
                        prompt_length=prompt_length,
                        eos_token_id=eos_token_id,
                    )
                    active_ids = output_ids[:active_length]
                    completion_end = active_length
                    if eos_token_id is not None and active_length > prompt_length:
                        if int(output_ids[active_length - 1].item()) == int(eos_token_id):
                            completion_end = active_length - 1
                    completion_ids = output_ids[prompt_length:completion_end]
                    raw_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                    final_answer = extract_final_answer(raw_text)
                    sample = ReasoningSample(
                        problem_id=problem.id,
                        sample_id=f"{problem.id}-online-{rollout_step}-{sample_index}",
                        cot_text=raw_text,
                        final_answer=final_answer,
                        raw_text=raw_text,
                        generation_meta={
                            "model_name": self.config.model_name,
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p,
                            "max_new_tokens": self.config.max_new_tokens,
                            "rollout_step": rollout_step,
                        },
                    )
                    answer_info = self.answer_judge.evaluate(problem, sample)
                    sample.final_answer = answer_info["predicted_answer"]
                    quality_score, subscores, quality_evidence = self.quality_scorer.score(problem, sample)
                    mode_label, mode_evidence = self.mode_tagger.tag_with_evidence(problem, sample)
                    scored_samples.append(
                        ScoredSample(
                            sample=sample,
                            score=ScoreBundle(
                                answer_correct=answer_info["answer_correct"],
                                quality_score=quality_score,
                                quality_subscores=subscores,
                                mode_label=mode_label,
                                quality_evidence=quality_evidence,
                                mode_evidence=mode_evidence,
                                answer_extraction=answer_info["answer_extraction"],
                                answer_judgment=answer_info["judgment_details"],
                            ),
                        )
                    )
                    input_sequences.append(active_ids)
                    attention_masks.append(self.torch.ones_like(active_ids, dtype=self.torch.long))
                    response_masks.append(response_mask[: max(active_length - 1, 0)])

                scored_group = ScoredGroup(problem=problem, scored_samples=scored_samples)
                scored_groups.append(self.advantage_computer.compute_group(scored_group))

            input_ids = _pad_1d_tensors(input_sequences, pad_token_id)
            attention_mask = _pad_1d_tensors(attention_masks, 0)
            response_mask = _pad_1d_tensors(response_masks, 0.0)
            old_stats = _forward_response_stats(
                self.model,
                input_ids,
                attention_mask,
                prompt_lengths=[0 for _ in range(int(input_ids.shape[0]))],
            )
            old_token_logprobs = old_stats["token_logprobs"].detach()

        advantages = []
        correct = []
        quality_scores = []
        rollout_groups = []
        for group in scored_groups:
            group_correct = sum(1 for item in group.scored_samples if item.score.answer_correct)
            group_records = []
            for item in group.scored_samples:
                if item.advantage is None:
                    raise RuntimeError("Online rollout scoring did not produce advantages.")
                advantages.append(item.advantage)
                correct.append(1.0 if item.score.answer_correct else 0.0)
                quality_scores.append(item.score.quality_score)
                group_records.append(
                    {
                        "sample_id": item.sample.sample_id,
                        "final_answer": item.sample.final_answer,
                        "answer_correct": item.score.answer_correct,
                        "quality_score": item.score.quality_score,
                        "mode_label": item.score.mode_label,
                        "a_ans": item.advantage.a_ans,
                        "a_qual": item.advantage.a_qual,
                        "a_mode": item.advantage.a_mode,
                        "a_total": item.advantage.a_total,
                        "text_preview": item.sample.raw_text[:500],
                    }
                )
            rollout_groups.append(
                {
                    "problem_id": group.problem.id,
                    "source": group.problem.source,
                    "gold_answer": group.problem.gold_answer,
                    "correct_count": group_correct,
                    "group_size": len(group.scored_samples),
                    "samples": group_records,
                }
            )

        a_total_values = [
            self.config.lambda_ans * adv.a_ans
            + self.config.lambda_qual * adv.a_qual
            + self.config.lambda_mode * adv.a_mode
            for adv in advantages
        ]
        diagnostics = {
            "problem_ids": [group.problem.id for group in scored_groups],
            "group_correct_counts": [group["correct_count"] for group in rollout_groups],
            "group_sizes": [group["group_size"] for group in rollout_groups],
            "response_tokens_mean": float(response_mask.sum(dim=-1).mean().item()),
            "response_tokens_max": int(response_mask.sum(dim=-1).max().item()),
            "nonzero_advantage_count": sum(1 for value in a_total_values if abs(value) > 1e-8),
            "a_total_abs_mean": sum(abs(value) for value in a_total_values) / max(1, len(a_total_values)),
            "groups": rollout_groups,
        }

        return OnlineRolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            old_token_logprobs=old_token_logprobs,
            a_ans=input_ids.new_tensor([adv.a_ans for adv in advantages], dtype=self.torch.float32),
            a_qual=input_ids.new_tensor([adv.a_qual for adv in advantages], dtype=self.torch.float32),
            a_mode=input_ids.new_tensor([adv.a_mode for adv in advantages], dtype=self.torch.float32),
            correct=input_ids.new_tensor(correct, dtype=self.torch.float32),
            quality_score=input_ids.new_tensor(quality_scores, dtype=self.torch.float32),
            groups=scored_groups,
            diagnostics=diagnostics,
        )

    def _update_from_rollout(self, rollout: OnlineRolloutBatch) -> dict[str, Any]:
        self.model.train()
        num_samples = int(rollout.input_ids.shape[0])
        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl = 0.0
        total_clip_fraction = 0.0
        backward_steps = 0
        optimizer_steps = 0
        self.optimizer.zero_grad()

        for _ in range(self.config.update_epochs):
            permutation = self.torch.randperm(num_samples, device=rollout.input_ids.device)
            for start in range(0, num_samples, self.config.mini_batch_size):
                indices = permutation[start : start + self.config.mini_batch_size]
                current_stats = _forward_response_stats(
                    self.model,
                    rollout.input_ids[indices],
                    rollout.attention_mask[indices],
                    prompt_lengths=[0 for _ in range(int(indices.shape[0]))],
                )
                with self.torch.no_grad():
                    reference_stats = _forward_response_stats(
                        self.reference_model,
                        rollout.input_ids[indices],
                        rollout.attention_mask[indices],
                        prompt_lengths=[0 for _ in range(int(indices.shape[0]))],
                    )
                advantages = build_total_advantage(
                    rollout.a_ans[indices],
                    rollout.a_qual[indices],
                    rollout.a_mode[indices],
                    lambda_ans=self.config.lambda_ans,
                    lambda_qual=self.config.lambda_qual,
                    lambda_mode=self.config.lambda_mode,
                )
                kl_values = _sampled_token_kl(
                    current_token_logprobs=current_stats["token_logprobs"],
                    reference_token_logprobs=reference_stats["token_logprobs"],
                )
                loss, breakdown = cmao_clipped_policy_loss(
                    current_logprobs=current_stats["token_logprobs"],
                    old_logprobs=rollout.old_token_logprobs[indices],
                    advantages=advantages,
                    kl_values=kl_values,
                    response_mask=rollout.response_mask[indices],
                    clip_range=self.config.clip_range,
                    kl_coef=self.config.kl_coef,
                )
                scaled_loss = loss / self.config.gradient_accumulation_steps
                if not self.torch.isfinite(loss):
                    self.optimizer.zero_grad()
                    continue
                self.accelerator.backward(scaled_loss)
                backward_steps += 1
                total_loss += breakdown.total_loss
                total_policy_loss += breakdown.policy_loss
                total_kl += breakdown.kl_term
                total_clip_fraction += breakdown.clip_fraction

                if backward_steps % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm and self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    optimizer_steps += 1

        if backward_steps % self.config.gradient_accumulation_steps != 0:
            if self.config.max_grad_norm and self.config.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            optimizer_steps += 1

        denom = max(1, backward_steps)
        return {
            "loss": total_loss / denom,
            "policy_loss": total_policy_loss / denom,
            "kl": total_kl / denom,
            "clip_fraction": total_clip_fraction / denom,
            "backward_steps": backward_steps,
            "optimizer_steps": optimizer_steps,
        }

    def _save_checkpoint(self, checkpoint_name: str) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        checkpoint_dir = output_dir / checkpoint_name
        unwrapped.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

    def _finalize(self, history: list[dict[str, Any]], rollout_step: int, optimizer_step: int) -> dict[str, Any]:
        self._save_checkpoint("checkpoint-final")
        output_dir = Path(self.config.output_dir)
        summary = {
            "training_mode": "online_grpo",
            "model_name": self.config.model_name,
            "output_dir": str(output_dir),
            "rollout_step": rollout_step,
            "optimizer_step": optimizer_step,
            "group_size": self.config.group_size,
            "rollout_batch_size": self.config.rollout_batch_size,
            "update_epochs": self.config.update_epochs,
            "lambda_ans": self.config.lambda_ans,
            "lambda_qual": self.config.lambda_qual,
            "lambda_mode": self.config.lambda_mode,
            "history": history,
        }
        save_json(output_dir / "training_summary.json", summary)
        return summary


def run_train_online_grpo(config_path: str) -> dict[str, Any]:
    config = online_grpo_config_from_dict(load_config(config_path))
    trainer = OnlineGRPOTrainer(config=config)
    return trainer.train()
