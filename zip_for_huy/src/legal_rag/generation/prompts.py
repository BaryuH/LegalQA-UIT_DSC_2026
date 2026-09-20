"""Versioned, inference-safe prompt templates for Direct and RAG generation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ..config import PromptsSection
from ..schemas import PackedEvidence

_VERSION_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")


class PromptTemplateError(ValueError):
    """Raised when a configured prompt template cannot be used safely."""


@dataclass(frozen=True, slots=True)
class PromptMetadata:
    """Stable identity for one prompt template, without prompt contents."""

    name: str
    version: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        """Return JSON-safe prompt identity metadata."""

        return {
            "prompt_name": self.name,
            "prompt_version": self.version,
            "prompt_sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class PromptRender:
    """Rendered prompt plus the identity of the template that produced it."""

    text: str
    metadata: PromptMetadata


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """UTF-8 template content and its content-addressed identity."""

    name: str
    version: str
    text: str
    sha256: str

    @property
    def metadata(self) -> PromptMetadata:
        """Return the metadata safe to attach to an inference artifact."""

        return PromptMetadata(
            name=self.name,
            version=self.version,
            sha256=self.sha256,
        )


def load_prompt_template(
    path: str | Path,
    *,
    name: str,
    version: str,
) -> PromptTemplate:
    """Load one exact UTF-8 template and hash its original bytes."""

    if not isinstance(name, str) or not name.strip():
        raise PromptTemplateError("Prompt template name must be non-blank")
    if not isinstance(version, str) or not version.strip():
        raise PromptTemplateError("Prompt template version must be non-blank")
    template_path = Path(path)
    try:
        raw = template_path.read_bytes()
    except OSError as exc:
        raise PromptTemplateError(
            f"Prompt template is not readable: {template_path}"
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromptTemplateError(
            f"Prompt template is not valid UTF-8: {template_path}"
        ) from exc
    if not text.strip():
        raise PromptTemplateError(f"Prompt template is blank: {template_path}")
    return PromptTemplate(
        name=name,
        version=version,
        text=text,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _template_path(prompt_dir: str | Path, name: str, version: str) -> Path:
    """Resolve a safe ``<name>_<version-token>.txt`` template path."""

    if version.startswith(f"{name}-"):
        token = version[len(name) + 1 :]
    elif version.startswith(f"{name}_"):
        token = version[len(name) + 1 :]
    else:
        token = version
    if _VERSION_TOKEN.fullmatch(token) is None:
        raise PromptTemplateError(
            f"Prompt template version has an invalid file token: {version!r}"
        )
    return Path(prompt_dir) / f"{name}_{token}.txt"


@dataclass(frozen=True, slots=True)
class PromptBuilder:
    """Build Direct or RAG prompts from explicitly loaded versioned templates."""

    direct_template: PromptTemplate
    rag_template: PromptTemplate

    @classmethod
    def from_directory(
        cls,
        prompt_dir: str | Path,
        *,
        direct_version: str = "direct-v1",
        rag_version: str = "rag-v1",
    ) -> Self:
        """Load the configured Direct and RAG templates from one directory."""

        return cls(
            direct_template=load_prompt_template(
                _template_path(prompt_dir, "direct", direct_version),
                name="direct",
                version=direct_version,
            ),
            rag_template=load_prompt_template(
                _template_path(prompt_dir, "rag", rag_version),
                name="rag",
                version=rag_version,
            ),
        )

    @classmethod
    def from_config(
        cls,
        config: PromptsSection,
        *,
        prompt_dir: str | Path,
    ) -> Self:
        """Load template versions from the typed prompt configuration section."""

        return cls.from_directory(
            prompt_dir,
            direct_version=config.direct_version,
            rag_version=config.rag_version,
        )

    def build_direct(self, question: str) -> PromptRender:
        """Render Direct using only the question string."""

        _validate_question(question)
        return _render(self.direct_template, question, evidence=None)

    def build_rag(self, question: str, evidence: PackedEvidence) -> PromptRender:
        """Render RAG using only the question and packed legal evidence."""

        _validate_question(question)
        if not isinstance(evidence, PackedEvidence):
            raise TypeError("evidence must be a PackedEvidence instance")
        return _render(self.rag_template, question, evidence=evidence)


def _validate_question(question: str) -> None:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not question.strip():
        raise ValueError("question must be a non-blank string")


def _render(
    template: PromptTemplate,
    question: str,
    *,
    evidence: PackedEvidence | None,
) -> PromptRender:
    text = template.text
    if "{question}" not in text:
        raise PromptTemplateError(
            f"{template.name} prompt template must contain {{question}}"
        )
    if evidence is None:
        if "{evidence}" in text:
            raise PromptTemplateError(
                "Direct prompt template must not contain {evidence}"
            )
    elif "{evidence}" not in text:
        raise PromptTemplateError("RAG prompt template must contain {evidence}")

    rendered = text.replace("{question}", question)
    if evidence is not None:
        rendered = rendered.replace("{evidence}", evidence.rendered_text)
    if not rendered.strip():  # pragma: no cover - guarded by template loading
        raise PromptTemplateError("Rendered prompt must not be blank")
    return PromptRender(text=rendered, metadata=template.metadata)


__all__ = [
    "PromptBuilder",
    "PromptMetadata",
    "PromptRender",
    "PromptTemplate",
    "PromptTemplateError",
    "load_prompt_template",
]
