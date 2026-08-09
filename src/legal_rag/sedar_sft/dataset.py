"""Deterministic SEDAR-SFT dataset builder over frozen B2 (SS-05)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ProjectConfig, load_config
from ..finetuned_reader.contracts import ExcludedExample, SFTExample
from ..finetuned_reader.dataset import (
    DatasetBuildError,
    DatasetBuildResult,
    build_sft_dataset_from_config,
)
from .contracts import SEDAR_METHOD, SEDAR_TRAINING_ROLE, to_sedar_example_dict


@dataclass(frozen=True, slots=True)
class SedarDatasetBuildResult:
    """Dataset build with SEDAR identity overlays."""

    underlying: DatasetBuildResult
    sedar_examples: tuple[dict[str, Any], ...]
    output_dir: Path
    manifest_path: Path

    @property
    def example_count(self) -> int:
        return len(self.sedar_examples)


def remap_examples_to_sedar_contract(
    examples: tuple[SFTExample, ...],
) -> tuple[dict[str, Any], ...]:
    """Attach SEDAR example_id / training_role without mutating gold/retrieval."""

    return tuple(to_sedar_example_dict(example) for example in examples)


def _write_sedar_overlay(
    output_dir: Path,
    *,
    underlying: DatasetBuildResult,
    sedar_examples: tuple[dict[str, Any], ...],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "sedar_examples.jsonl"
    with examples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sedar_examples:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "sedar_sft.ss05.dataset.v1",
        "method": SEDAR_METHOD,
        "training_role": SEDAR_TRAINING_ROLE,
        "example_count": len(sedar_examples),
        "underlying_dataset_dir": underlying.output_dir.as_posix(),
        "underlying_manifest": (
            (underlying.output_dir / "manifest.json").as_posix()
            if (underlying.output_dir / "manifest.json").is_file()
            else None
        ),
        "gpu_deferred": [
            "SS-05G CUDA reranker throughput/equivalence",
            "full effective-train retrieval build on server",
        ],
    }
    manifest_path = output_dir / "sedar_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def build_sedar_sft_dataset_from_config(
    config: ProjectConfig | str | Path,
    *,
    repo_root: str | Path,
    max_examples: int | None = None,
) -> SedarDatasetBuildResult:
    """Build frozen-B2 SFT examples then emit SEDAR-contract JSONL.

    Gold is attached only after question-only retrieval inside the shared
    ``finetuned_reader`` builder. Full-corpus builds and GPU reranker
    optimization remain operator/server responsibilities (SS-05G).
    """

    root = Path(repo_root).resolve()
    project = (
        config
        if isinstance(config, ProjectConfig)
        else load_config(root / config if not Path(config).is_absolute() else config)
    )
    if project.project.profile != "sedar_sft":
        raise DatasetBuildError(
            "SEDAR dataset build requires project.profile='sedar_sft'"
        )
    if project.finetuned_reader is None:
        raise DatasetBuildError("SEDAR dataset build requires finetuned_reader settings")

    underlying = build_sft_dataset_from_config(
        project,
        repo_root=root,
        max_examples=max_examples,
    )
    sedar_examples = remap_examples_to_sedar_contract(underlying.examples)
    settings = project.finetuned_reader
    overlay_dir = root / settings.output.dataset_root / f"{settings.dataset_version}-sedar"
    if max_examples is not None:
        overlay_dir = overlay_dir / f"smoke-{max_examples}"
    manifest_path = _write_sedar_overlay(
        overlay_dir, underlying=underlying, sedar_examples=sedar_examples
    )
    return SedarDatasetBuildResult(
        underlying=underlying,
        sedar_examples=sedar_examples,
        output_dir=overlay_dir,
        manifest_path=manifest_path,
    )


__all__ = [
    "ExcludedExample",
    "SedarDatasetBuildResult",
    "build_sedar_sft_dataset_from_config",
    "remap_examples_to_sedar_contract",
]
