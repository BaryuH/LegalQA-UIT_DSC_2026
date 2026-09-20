"""Branch 5 — build an MBR self-distillation SFT dataset (Finkelstein & Freitag, 2024).

Idea: run the expensive MBR selection *once, offline, on train* to produce a
target answer per question, then fine-tune the reader to emit that answer with
plain greedy decoding.  At serve time you keep single-pass latency and the frozen
``temperature=0.0`` contract while inheriting MBR-level quality.

Selection modes:
- ``self`` (default): target = MBR-selected candidate.  Gold is never read, so
  this is pure self-distillation on ``train.json`` questions.
- ``oracle``: target = the candidate with the highest gold METEOR.  This *reads
  gold* and is therefore an approved training task (AGENTS.md invariant 4); the
  mode is recorded in every record and in the manifest.

The produced record carries only ``question`` / optional ``evidence`` /
``target`` — the exact fields a downstream SFT collator consumes.  Gold text is
never copied into the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import grounding, metrics, selection
from .data import build_prompt, load_dataset, load_prompt_template
from .generation import HFCandidateGenerator, ModelConfig, SamplingConfig
from .runner import _resolve
from .utilities import LexicalUtility

SelectMode = Literal["self", "oracle"]


@dataclass(frozen=True, slots=True)
class DistillConfig:
    data_path: str
    prompt_template: str
    output_path: str
    model: ModelConfig
    sampling: SamplingConfig
    evidence_path: str | None = None
    limit: int | None = None
    utility_metric: str = "meteor"
    prune: bool = True
    select_mode: SelectMode = "self"


def _select_target(
    gold: str,
    question: str,
    cands: list[str],
    evidence: str | None,
    cfg: DistillConfig,
) -> tuple[int, str]:
    if cfg.select_mode == "oracle":
        scores = np.array([metrics.meteor_exact(gold, c) for c in cands])
        return int(np.argmax(scores)), "oracle"
    matrix = LexicalUtility(cfg.utility_metric).pairwise(cands)
    if cfg.prune:
        keep, _reasons, _mode = grounding.keep_mask(cands, evidence)
        sel = selection.mbr_pruned(matrix, keep)
        if sel.index >= 0:
            return sel.index, "self_pruned"
    sel = selection.mbr(matrix)
    return sel.index, "self"


def build_distill_dataset(cfg: DistillConfig) -> Path:
    dataset = load_dataset(
        _resolve(cfg.data_path),
        evidence_path=_resolve(cfg.evidence_path),
        limit=cfg.limit,
    )
    template = load_prompt_template(_resolve(cfg.prompt_template))
    generator = HFCandidateGenerator(cfg.model)
    generator.load()

    out_path = _resolve(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for case in dataset.cases:
            prompt = build_prompt(template, case)
            cset = generator.generate(case.id, prompt, cfg.sampling)
            cands = list(cset.candidates)
            if not cands:
                continue
            index, mode = _select_target(
                case.gold, case.question, cands, case.evidence, cfg
            )
            record: dict[str, Any] = {
                "id": case.id,
                "question": case.question,
                "evidence": case.evidence,
                "target": cands[index],
                "selection_mode": mode,
                "num_candidates": len(cands),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return out_path


__all__ = ["DistillConfig", "SelectMode", "build_distill_dataset"]
