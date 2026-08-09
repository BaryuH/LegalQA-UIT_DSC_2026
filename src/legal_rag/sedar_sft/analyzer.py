"""SS-12 Requirement Analyzer — rule-first, no gold, no final prose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Requirement:
    requirement_id: str
    type: str
    required: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "type": self.type,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class RequirementAnalysis:
    question_type: str
    temporal_mode: str
    answer_shape: str
    requirements: tuple[Requirement, ...]
    uncertain_fields: tuple[str, ...]
    llm_fallback_needed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "temporal_mode": self.temporal_mode,
            "answer_shape": self.answer_shape,
            "requirements": [item.as_dict() for item in self.requirements],
            "uncertain_fields": list(self.uncertain_fields),
            "llm_fallback_needed": self.llm_fallback_needed,
        }


def analyze_requirements(question: str) -> RequirementAnalysis:
    """Rule-first requirement decomposition (framework §14)."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Analyzer question must be non-blank")
    text = question.casefold()
    requirements: list[Requirement] = [
        Requirement("R1", "legal_basis", True),
    ]
    uncertain: list[str] = []

    if any(token in text for token in ("trước đây", "trước kia")):
        temporal = "historical"
    elif any(token in text for token in ("hiện nay", "hiện hành", "hiện tại")):
        temporal = "current"
    else:
        temporal = "unknown"
        uncertain.append("temporal_mode")

    if any(token in text for token in ("cao nhất", "tối đa")):
        question_type = "maximum"
        answer_shape = "scalar"
        requirements.append(Requirement("R2", "maximum_value", True))
    elif "tối thiểu" in text:
        question_type = "minimum"
        answer_shape = "scalar"
        requirements.append(Requirement("R2", "minimum_value", True))
    elif any(token in text for token in ("trách nhiệm", "bao gồm", "những gì", "gồm những")):
        question_type = "responsibility_list"
        answer_shape = "enumeration"
        requirements.append(Requirement("R2", "legal_duty", True))
    elif "điều kiện" in text:
        question_type = "conditions"
        answer_shape = "enumeration"
        requirements.append(Requirement("R2", "conditions", True))
    elif "khi nào" in text:
        question_type = "time_condition"
        answer_shape = "conditions"
        requirements.append(Requirement("R2", "trigger", True))
    else:
        question_type = "general"
        answer_shape = "prose"
        uncertain.append("question_type")

    return RequirementAnalysis(
        question_type=question_type,
        temporal_mode=temporal,
        answer_shape=answer_shape,
        requirements=tuple(requirements),
        uncertain_fields=tuple(uncertain),
        llm_fallback_needed=bool(uncertain),
    )


__all__ = ["Requirement", "RequirementAnalysis", "analyze_requirements"]
