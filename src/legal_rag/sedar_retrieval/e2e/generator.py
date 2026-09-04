"""Load frozen SEDAR-SFT generators for TASK 20 end-to-end runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from legal_rag.finetuned_reader.checkpoint import ValidatedCheckpoint
from legal_rag.finetuned_reader.inference import (
    FineTunedReaderGenerator,
    GenerativeReaderError,
    TransformersCausalBackend,
    load_finetuned_reader_generator,
)
from legal_rag.finetuned_reader.prompting import GenerativePromptBuilder
from legal_rag.sedar_sft.checkpoint import validate_sedar_checkpoint


def _validated_checkpoint_from_sedar(
    checkpoint_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    require_sedar_extras: bool = False,
) -> ValidatedCheckpoint:
    sedar = validate_sedar_checkpoint(
        checkpoint_dir,
        manifest_path=manifest_path,
        require_sedar_extras=require_sedar_extras,
    )
    return ValidatedCheckpoint(
        checkpoint_dir=sedar.checkpoint_dir,
        manifest_path=sedar.manifest_path,
        manifest=sedar.manifest,
        manifest_hash=sedar.manifest_hash,
        adapter_hash=sedar.adapter_hash,
    )


def load_sedar_sft_generator(
    checkpoint_dir: str | Path,
    *,
    checkpoint_manifest: str | Path | None = None,
    prompt_builder: GenerativePromptBuilder,
    max_new_tokens: int,
    stop_sequences: Sequence[str] = (),
    device: str = "auto",
    load_in_4bit: bool | None = None,
    require_sedar_extras: bool = False,
    no_repeat_ngram_size: int | None = None,
    repetition_penalty: float | None = None,
) -> FineTunedReaderGenerator:
    """Validate a SEDAR-SFT checkpoint and load the local generative backend."""

    if max_new_tokens <= 0:
        raise GenerativeReaderError("max_new_tokens must be positive")
    checkpoint = _validated_checkpoint_from_sedar(
        checkpoint_dir,
        manifest_path=checkpoint_manifest,
        require_sedar_extras=require_sedar_extras,
    )
    manifest = checkpoint.manifest
    selected_load_in_4bit = load_in_4bit
    if selected_load_in_4bit is None:
        quantization = manifest.get("quantization")
        if isinstance(quantization, dict):
            selected_load_in_4bit = bool(quantization.get("load_in_4bit", False))
        else:
            selected_load_in_4bit = (
                str(manifest.get("adapter_type", "")).lower() == "qlora"
            )
    backend = TransformersCausalBackend.from_checkpoint(
        checkpoint,
        device=device,
        load_in_4bit=selected_load_in_4bit,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
    )
    return FineTunedReaderGenerator(
        backend=backend,
        prompt_builder=prompt_builder,
        checkpoint=checkpoint,
        max_new_tokens=max_new_tokens,
        stop_sequences=tuple(stop_sequences),
    )


def load_generative_reader_generator(
    checkpoint_dir: str | Path,
    *,
    checkpoint_manifest: str | Path | None = None,
    prompt_builder: GenerativePromptBuilder,
    max_new_tokens: int,
    stop_sequences: Sequence[str] = (),
    device: str = "auto",
    load_in_4bit: bool | None = None,
    profile: str | None = None,
    no_repeat_ngram_size: int | None = None,
    repetition_penalty: float | None = None,
) -> FineTunedReaderGenerator:
    """Load either a finetuned_reader or sedar_sft generative checkpoint."""

    root = Path(checkpoint_dir)
    manifest_path = (
        Path(checkpoint_manifest)
        if checkpoint_manifest is not None
        else root / "checkpoint_manifest.json"
    )
    payload_profile = profile
    if payload_profile is None and manifest_path.is_file():
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload_profile = str(payload.get("profile", "")).strip() or None
    if payload_profile == "sedar_sft":
        return load_sedar_sft_generator(
            checkpoint_dir,
            checkpoint_manifest=checkpoint_manifest,
            prompt_builder=prompt_builder,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
            device=device,
            load_in_4bit=load_in_4bit,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )
    return load_finetuned_reader_generator(
        str(checkpoint_dir),
        checkpoint_manifest=str(checkpoint_manifest) if checkpoint_manifest else None,
        prompt_builder=prompt_builder,
        max_new_tokens=max_new_tokens,
        stop_sequences=stop_sequences,
        device=device,
        load_in_4bit=bool(load_in_4bit),
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
    )


__all__ = [
    "load_generative_reader_generator",
    "load_sedar_sft_generator",
]
