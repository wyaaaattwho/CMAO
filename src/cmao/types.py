from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProblemRecord:
    id: str
    source: str
    prompt: str
    gold_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProblemRecord":
        return cls(
            id=str(payload["id"]),
            source=str(payload["source"]),
            prompt=str(payload["prompt"]),
            gold_answer=str(payload["gold_answer"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "prompt": self.prompt,
            "gold_answer": self.gold_answer,
            "metadata": self.metadata,
        }


@dataclass
class ReasoningSample:
    problem_id: str
    sample_id: str
    cot_text: str
    final_answer: str
    raw_text: str
    generation_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReasoningSample":
        return cls(
            problem_id=str(payload["problem_id"]),
            sample_id=str(payload["sample_id"]),
            cot_text=str(payload.get("cot_text", "")),
            final_answer=str(payload.get("final_answer", "")),
            raw_text=str(payload.get("raw_text", "")),
            generation_meta=dict(payload.get("generation_meta", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "sample_id": self.sample_id,
            "cot_text": self.cot_text,
            "final_answer": self.final_answer,
            "raw_text": self.raw_text,
            "generation_meta": self.generation_meta,
        }


@dataclass
class ScoreBundle:
    answer_correct: bool
    quality_score: float
    quality_subscores: dict[str, float] = field(default_factory=dict)
    mode_label: str = "other_math"
    quality_evidence: dict[str, Any] = field(default_factory=dict)
    mode_evidence: dict[str, Any] = field(default_factory=dict)
    answer_extraction: dict[str, Any] = field(default_factory=dict)
    answer_judgment: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoreBundle":
        return cls(
            answer_correct=bool(payload["answer_correct"]),
            quality_score=float(payload["quality_score"]),
            quality_subscores={
                str(key): float(value)
                for key, value in payload.get("quality_subscores", {}).items()
            },
            mode_label=str(payload.get("mode_label", "other_math")),
            quality_evidence=dict(payload.get("quality_evidence", {})),
            mode_evidence=dict(payload.get("mode_evidence", {})),
            answer_extraction=dict(payload.get("answer_extraction", {})),
            answer_judgment=dict(payload.get("answer_judgment", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_correct": self.answer_correct,
            "quality_score": self.quality_score,
            "quality_subscores": self.quality_subscores,
            "mode_label": self.mode_label,
            "quality_evidence": self.quality_evidence,
            "mode_evidence": self.mode_evidence,
            "answer_extraction": self.answer_extraction,
            "answer_judgment": self.answer_judgment,
        }


@dataclass
class AdvantageBundle:
    a_ans: float
    a_qual: float
    a_mode: float
    a_total: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdvantageBundle":
        return cls(
            a_ans=float(payload["a_ans"]),
            a_qual=float(payload["a_qual"]),
            a_mode=float(payload["a_mode"]),
            a_total=float(payload["a_total"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_ans": self.a_ans,
            "a_qual": self.a_qual,
            "a_mode": self.a_mode,
            "a_total": self.a_total,
        }


@dataclass
class GroupedSamples:
    problem: ProblemRecord
    samples: list[ReasoningSample]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroupedSamples":
        return cls(
            problem=ProblemRecord.from_dict(payload["problem"]),
            samples=[ReasoningSample.from_dict(item) for item in payload.get("samples", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass
class ScoredSample:
    sample: ReasoningSample
    score: ScoreBundle
    advantage: AdvantageBundle | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoredSample":
        advantage_payload = payload.get("advantage")
        return cls(
            sample=ReasoningSample.from_dict(payload["sample"]),
            score=ScoreBundle.from_dict(payload["score"]),
            advantage=AdvantageBundle.from_dict(advantage_payload) if advantage_payload else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "score": self.score.to_dict(),
            "advantage": self.advantage.to_dict() if self.advantage else None,
        }


@dataclass
class ScoredGroup:
    problem: ProblemRecord
    scored_samples: list[ScoredSample]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoredGroup":
        return cls(
            problem=ProblemRecord.from_dict(payload["problem"]),
            scored_samples=[
                ScoredSample.from_dict(item) for item in payload.get("scored_samples", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "scored_samples": [item.to_dict() for item in self.scored_samples],
            "metadata": self.metadata,
        }
