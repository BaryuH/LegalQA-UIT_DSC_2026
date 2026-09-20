"""Question/evidence prompts for generative SFT and inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..schemas import PackedEvidence


class GenerativePromptError(ValueError):
    """Raised when a prompt would violate the answer-free inference boundary."""


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
    version: str
    sha256: str


class GenerativePromptBuilder:
    """Render one shared question/evidence prefix with or without a target."""

    def __init__(self, train_template: str, inference_template: str, *, version: str):
        if not train_template.strip() or not inference_template.strip():
            raise GenerativePromptError("Prompt templates must be non-blank")
        if "{question}" not in train_template or "{evidence}" not in train_template:
            raise GenerativePromptError(
                "Training prompt must contain {question} and {evidence}"
            )
        if (
            "{question}" not in inference_template
            or "{evidence}" not in inference_template
        ):
            raise GenerativePromptError(
                "Inference prompt must contain {question} and {evidence}"
            )
        if "{target}" in inference_template or "{answer}" in inference_template:
            raise GenerativePromptError(
                "Inference prompt must not contain a target field"
            )
        self.train_template = train_template
        self.inference_template = inference_template
        self.version = version
        self.train_sha256 = hashlib.sha256(train_template.encode("utf-8")).hexdigest()
        self.inference_sha256 = hashlib.sha256(
            inference_template.encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_files(
        cls,
        train_path: str | Path,
        inference_path: str | Path,
        *,
        version: str,
    ) -> GenerativePromptBuilder:
        return cls(
            Path(train_path).read_text(encoding="utf-8"),
            Path(inference_path).read_text(encoding="utf-8"),
            version=version,
        )

    @staticmethod
    def _validate_question(question: str) -> None:
        if not isinstance(question, str) or not question.strip():
            raise GenerativePromptError("Prompt question must be a non-blank string")

    @staticmethod
    def _validate_evidence(evidence: PackedEvidence) -> None:
        if not isinstance(evidence, PackedEvidence):
            raise TypeError("Prompt evidence must be PackedEvidence")
        if not evidence.rendered_text.strip() or not evidence.included_ids:
            raise GenerativePromptError("Prompt evidence must contain packed chunks")

    def build_inference(
        self, question: str, evidence: PackedEvidence
    ) -> RenderedPrompt:
        """Build a prompt from question and packed evidence only."""

        self._validate_question(question)
        self._validate_evidence(evidence)
        text = self.inference_template.format(
            question=question.strip(), evidence=evidence.rendered_text
        )
        return RenderedPrompt(
            text=text,
            version=self.version,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def build_training(
        self,
        question: str,
        evidence: PackedEvidence,
        target_answer: str,
    ) -> tuple[RenderedPrompt, str]:
        """Build the shared prefix and return the target separately."""

        self._validate_question(question)
        self._validate_evidence(evidence)
        if not isinstance(target_answer, str) or not target_answer.strip():
            raise GenerativePromptError("Training target must be a non-blank string")
        text = self.train_template.format(
            question=question.strip(), evidence=evidence.rendered_text
        )
        return (
            RenderedPrompt(
                text=text,
                version=self.version,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            ),
            target_answer.strip(),
        )


__all__ = ["GenerativePromptBuilder", "GenerativePromptError", "RenderedPrompt"]
