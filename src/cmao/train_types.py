from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import AdvantageBundle


@dataclass
class PolicyTrainingRecord:
    problem_id: str
    sample_id: str
    prompt: str
    response_text: str
    final_answer: str
    answer_correct: bool
    quality_score: float
    quality_subscores: dict[str, float] = field(default_factory=dict)
    mode_label: str = "other_math"
    advantage: AdvantageBundle = field(
        default_factory=lambda: AdvantageBundle(a_ans=0.0, a_qual=0.0, a_mode=0.0, a_total=0.0)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyTrainingRecord":
        return cls(
            problem_id=str(payload["problem_id"]),
            sample_id=str(payload["sample_id"]),
            prompt=str(payload["prompt"]),
            response_text=str(payload["response_text"]),
            final_answer=str(payload.get("final_answer", "")),
            answer_correct=bool(payload["answer_correct"]),
            quality_score=float(payload["quality_score"]),
            quality_subscores={
                str(key): float(value)
                for key, value in payload.get("quality_subscores", {}).items()
            },
            mode_label=str(payload.get("mode_label", "other_math")),
            advantage=AdvantageBundle.from_dict(payload["advantage"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "response_text": self.response_text,
            "final_answer": self.final_answer,
            "answer_correct": self.answer_correct,
            "quality_score": self.quality_score,
            "quality_subscores": self.quality_subscores,
            "mode_label": self.mode_label,
            "advantage": self.advantage.to_dict(),
            "metadata": self.metadata,
        }
