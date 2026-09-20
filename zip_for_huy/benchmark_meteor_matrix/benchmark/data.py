"""Dataset loading and prompt construction.

Contract compliance:
- Gold answers are loaded into :class:`Case.gold` but are *only* read by the
  evaluator.  They never enter the prompt, generation, utility, or selection
  paths (AGENTS.md invariants 4/5).
- Optional per-question evidence is read from a side JSONL keyed by id; it is the
  only retrieval-derived text allowed into the prompt.  Absent evidence means a
  closed-book run and is recorded as such.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    question: str
    gold: str  # evaluation only
    evidence: str | None = None


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    cases: tuple[Case, ...] = field(default_factory=tuple)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for case in self.cases:
            digest.update(case.id.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()[:16]


def _load_evidence(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    evidence: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        evidence[str(row["id"])] = str(row.get("evidence", ""))
    return evidence


def load_dataset(
    data_path: str | Path,
    *,
    evidence_path: str | Path | None = None,
    limit: int | None = None,
) -> Dataset:
    """Load a ``{id: {question, answer}}`` split into :class:`Case` records."""

    path = Path(data_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    evidence = _load_evidence(Path(evidence_path) if evidence_path else None)
    cases: list[Case] = []
    for identifier, record in raw.items():
        question = str(record.get("question", "")).strip()
        gold = str(record.get("answer", "")).strip()
        if not question or not gold:
            continue
        cases.append(
            Case(
                id=str(identifier),
                question=question,
                gold=gold,
                evidence=evidence.get(str(identifier)),
            )
        )
        if limit and len(cases) >= limit:
            break
    return Dataset(name=path.stem, cases=tuple(cases))


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_prompt(template: str, case: Case) -> str:
    """Fill the RAG/infer template with question and evidence.

    Closed-book runs pass an explicit sentinel so the model is never handed a
    dangling ``{evidence}`` placeholder.
    """

    evidence = case.evidence or "(không có trích đoạn; trả lời theo hiểu biết pháp lý chung)"
    return template.replace("{question}", case.question).replace("{evidence}", evidence)


def batched(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


__all__ = [
    "Case",
    "Dataset",
    "batched",
    "build_prompt",
    "load_dataset",
    "load_prompt_template",
]
