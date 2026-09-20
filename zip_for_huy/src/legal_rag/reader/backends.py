"""Offline mock and local-only Transformers extractive-reader backends."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path

from ..config import ReaderSection
from .types import ExtractiveReader, ReaderSpan


class ReaderUnavailableError(RuntimeError):
    """Raised when the configured local checkpoint cannot be used."""


class MockExtractiveReader:
    """Deterministic offline backend used by unit and E2E fixture tests."""

    def __init__(
        self,
        scorer: Callable[[str, str, str], ReaderSpan],
        *,
        model: str = "mock-extractive-reader",
        model_version: str = "offline-v1",
    ) -> None:
        self._scorer = scorer
        self._model = model
        self._model_version = model_version

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(
        self,
        *,
        question: str,
        context: str,
        case_id: str,
    ) -> ReaderSpan:
        return self._scorer(question, context, case_id)


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ReaderUnavailableError(f"Reader checkpoint directory is empty: {path}")
    for candidate in files:
        digest.update(candidate.relative_to(path).as_posix().encode("utf-8"))
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Validate local checkpoint provenance without downloading any model."""

    if not checkpoint_path.is_dir():
        raise ReaderUnavailableError(
            f"Reader checkpoint directory not found: {checkpoint_path}"
        )
    if not manifest_path.is_file():
        raise ReaderUnavailableError(
            f"Reader checkpoint manifest not found: {manifest_path}"
        )
    try:
        payload: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReaderUnavailableError(
            f"Invalid reader checkpoint manifest: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ReaderUnavailableError("Reader checkpoint manifest must be an object")
    for field in ("model", "model_version", "checkpoint_sha256"):
        if not isinstance(payload.get(field), str) or not str(payload[field]).strip():
            raise ReaderUnavailableError(
                f"Reader checkpoint manifest missing non-blank {field!r}"
            )
    actual_hash = _directory_hash(checkpoint_path)
    if payload["checkpoint_sha256"] != actual_hash:
        raise ReaderUnavailableError(
            "Reader checkpoint hash does not match checkpoint manifest"
        )
    return payload


class TransformersExtractiveReader:
    """Question-answering backend that loads one local checkpoint fail-closed."""

    def __init__(
        self,
        checkpoint_path: Path,
        manifest: Mapping[str, object],
        settings: ReaderSection,
    ) -> None:
        if not settings.local_files_only:  # pragma: no cover - Literal guard
            raise ReaderUnavailableError("Reader model downloads are forbidden")
        try:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        except ImportError as exc:
            raise ReaderUnavailableError(
                "Reader runtime requires torch and transformers"
            ) from exc
        selected_device = settings.device
        if selected_device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        if selected_device == "cuda" and not torch.cuda.is_available():
            raise ReaderUnavailableError("Reader device cuda requested but unavailable")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                local_files_only=True,
                use_fast=True,
            )
            self._qa_model = AutoModelForQuestionAnswering.from_pretrained(
                checkpoint_path,
                local_files_only=True,
            ).to(selected_device)
        except (OSError, ValueError) as exc:
            raise ReaderUnavailableError(
                f"Unable to load local reader checkpoint: {checkpoint_path}"
            ) from exc
        self._qa_model.eval()
        self._torch = torch
        self._device = selected_device
        self._settings = settings
        self._model = str(manifest["model"])
        self._model_version = str(manifest["model_version"])

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(
        self,
        *,
        question: str,
        context: str,
        case_id: str,
    ) -> ReaderSpan:
        del case_id
        encoded = self._tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self._settings.max_seq_length,
            stride=self._settings.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
        )
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        model_inputs = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            outputs = self._qa_model(**model_inputs)

        best_score = float("-inf")
        best_start: int | None = None
        best_end: int | None = None
        no_answer_score = float("-inf")
        for feature_index in range(len(outputs.start_logits)):
            sequence_ids = encoded.sequence_ids(feature_index)
            start_logits = outputs.start_logits[feature_index]
            end_logits = outputs.end_logits[feature_index]
            no_answer_score = max(
                no_answer_score,
                float(start_logits[0].item() + end_logits[0].item()),
            )
            token_count = len(sequence_ids)
            for start_token in range(token_count):
                if sequence_ids[start_token] != 1:
                    continue
                max_end = min(
                    token_count,
                    start_token + self._settings.max_answer_length,
                )
                for end_token in range(start_token, max_end):
                    if sequence_ids[end_token] != 1:
                        break
                    score = float(
                        start_logits[start_token].item() + end_logits[end_token].item()
                    )
                    if score > best_score:
                        start_offset = int(offsets[feature_index][start_token][0])
                        end_offset = int(offsets[feature_index][end_token][1])
                        if end_offset <= start_offset:
                            continue
                        best_score = score
                        best_start = start_offset
                        best_end = end_offset
        if best_start is None or best_end is None or no_answer_score >= best_score:
            return ReaderSpan(
                answer="",
                confidence=no_answer_score,
                impossible=True,
            )
        return ReaderSpan(
            answer=context[best_start:best_end],
            confidence=best_score,
            start_char=best_start,
            end_char=best_end,
        )


def create_local_reader(
    settings: ReaderSection,
    *,
    repo_root: Path,
) -> ExtractiveReader:
    checkpoint_path = (repo_root / settings.checkpoint_path).resolve()
    manifest_path = (repo_root / settings.checkpoint_manifest_path).resolve()
    manifest = validate_checkpoint(checkpoint_path, manifest_path)
    return TransformersExtractiveReader(checkpoint_path, manifest, settings)


__all__ = [
    "MockExtractiveReader",
    "ReaderUnavailableError",
    "TransformersExtractiveReader",
    "create_local_reader",
    "validate_checkpoint",
]
