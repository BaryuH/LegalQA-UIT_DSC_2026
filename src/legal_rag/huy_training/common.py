"""Shared fail-closed contracts for embedding and reranker fine-tuning."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

HardwareName = Literal["rtx4090_24gb", "a100_24gb", "a100_40gb", "a100_80gb"]
TaskName = Literal["embedding", "reranker"]
TuneMode = Literal["lora", "full"]


@dataclass(frozen=True, slots=True)
class TrainingPair:
    """One query with one positive and audited negatives."""

    query_id: str
    query: str
    positive: str
    negatives: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Conservative starting point; a smoke probe must confirm peak VRAM."""

    name: HardwareName
    task: TaskName
    tune_mode: TuneMode
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    max_query_tokens: int
    max_negatives: int
    learning_rate: float
    epochs: int
    lora_rank: int
    lora_alpha: int
    gradient_checkpointing: bool = True
    dtype: Literal["bf16"] = "bf16"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROFILES: dict[tuple[HardwareName, TaskName], HardwareProfile] = {
    ("rtx4090_24gb", "embedding"): HardwareProfile(
        name="rtx4090_24gb",
        task="embedding",
        tune_mode="lora",
        micro_batch_size=1,
        gradient_accumulation_steps=32,
        max_length=2048,
        max_query_tokens=256,
        max_negatives=4,
        learning_rate=2e-5,
        epochs=1,
        lora_rank=16,
        lora_alpha=32,
    ),
    ("a100_24gb", "embedding"): HardwareProfile(
        name="a100_24gb",
        task="embedding",
        tune_mode="lora",
        micro_batch_size=1,
        gradient_accumulation_steps=32,
        max_length=2048,
        max_query_tokens=256,
        max_negatives=4,
        learning_rate=2e-5,
        epochs=1,
        lora_rank=16,
        lora_alpha=32,
    ),
    ("a100_40gb", "embedding"): HardwareProfile(
        name="a100_40gb",
        task="embedding",
        tune_mode="lora",
        micro_batch_size=2,
        gradient_accumulation_steps=16,
        max_length=2048,
        max_query_tokens=256,
        max_negatives=8,
        learning_rate=2e-5,
        epochs=1,
        lora_rank=16,
        lora_alpha=32,
    ),
    ("a100_80gb", "embedding"): HardwareProfile(
        name="a100_80gb",
        task="embedding",
        tune_mode="full",
        micro_batch_size=4,
        gradient_accumulation_steps=8,
        max_length=2048,
        max_query_tokens=256,
        max_negatives=10,
        learning_rate=1e-5,
        epochs=1,
        lora_rank=16,
        lora_alpha=32,
    ),
    ("rtx4090_24gb", "reranker"): HardwareProfile(
        name="rtx4090_24gb",
        task="reranker",
        tune_mode="lora",
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        max_length=2304,
        max_query_tokens=256,
        max_negatives=10,
        learning_rate=2e-5,
        epochs=2,
        lora_rank=16,
        lora_alpha=32,
    ),
    ("a100_24gb", "reranker"): HardwareProfile(
        name="a100_24gb",
        task="reranker",
        tune_mode="lora",
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        max_length=2304,
        max_query_tokens=256,
        max_negatives=10,
        learning_rate=2e-5,
        epochs=2,
        lora_rank=16,
        lora_alpha=32,
    ),
    ("a100_40gb", "reranker"): HardwareProfile(
        name="a100_40gb",
        task="reranker",
        tune_mode="lora",
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        max_length=2304,
        max_query_tokens=256,
        max_negatives=10,
        learning_rate=2e-5,
        epochs=2,
        lora_rank=16,
        lora_alpha=32,
    ),
    ("a100_80gb", "reranker"): HardwareProfile(
        name="a100_80gb",
        task="reranker",
        tune_mode="full",
        micro_batch_size=4,
        gradient_accumulation_steps=4,
        max_length=2304,
        max_query_tokens=256,
        max_negatives=10,
        learning_rate=1e-5,
        epochs=2,
        lora_rank=16,
        lora_alpha=32,
    ),
}


def get_profile(name: HardwareName, task: TaskName) -> HardwareProfile:
    """Return a versioned conservative profile."""

    return _PROFILES[(name, task)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_pairs(
    path: Path,
    *,
    source_split: str,
    max_negatives: int,
) -> tuple[TrainingPair, ...]:
    """Load audited train pairs and reject public/private/warmup use."""

    if source_split != "train":
        raise ValueError("Fine-tuning is allowed only with source_split='train'")
    if max_negatives <= 0:
        raise ValueError("max_negatives must be positive")
    if not path.is_file():
        raise ValueError(f"Training pairs do not exist: {path}")
    audit_path = path.with_name("audit.json")
    if not audit_path.is_file():
        raise ValueError(f"Required negative audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError(f"Negative audit did not pass: {audit.get('gate_failures')}")

    rows: list[TrainingPair] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            query_id = str(raw.get("query_id") or "").strip()
            query = str(raw.get("query") or "").strip()
            positive = str(raw.get("positive") or "").strip()
            negatives = tuple(
                str(item).strip()
                for item in (raw.get("negatives") or [])
                if str(item).strip()
            )[:max_negatives]
            if not query_id or not query or not positive or not negatives:
                raise ValueError(f"Invalid training row at line {line_number}")
            if query_id in seen:
                raise ValueError(f"Duplicate query_id in training pairs: {query_id}")
            seen.add(query_id)
            rows.append(TrainingPair(query_id, query, positive, negatives))
    if len(rows) < 2:
        raise ValueError("At least two query groups are required")
    return tuple(sorted(rows, key=lambda item: item.query_id))


def split_pairs(
    pairs: tuple[TrainingPair, ...], *, dev_fraction: float, seed: int
) -> tuple[tuple[TrainingPair, ...], tuple[TrainingPair, ...]]:
    """Deterministic query-level split."""

    if not 0.0 < dev_fraction < 0.5:
        raise ValueError("dev_fraction must be in (0, 0.5)")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    dev_count = max(1, math.ceil(len(shuffled) * dev_fraction))
    dev = tuple(sorted(shuffled[:dev_count], key=lambda item: item.query_id))
    train = tuple(sorted(shuffled[dev_count:], key=lambda item: item.query_id))
    if not train:
        raise ValueError("Train split is empty")
    return train, dev


def reciprocal_rank(positive_score: float, negative_scores: list[float]) -> float:
    """Tie-stable reciprocal rank with the positive ordered after equal scores."""

    rank = 1 + sum(score >= positive_score for score in negative_scores)
    return 1.0 / rank


def discover_lora_targets(model: Any) -> tuple[str, ...]:
    """Find attention projection names for XLM-R/BGE or Qwen-like models."""

    suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    candidates = (
        ("query", "key", "value"),
        ("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    for candidate in candidates:
        if all(name in suffixes for name in candidate):
            return candidate
    raise ValueError("Could not discover supported LoRA attention target modules")


def count_parameters(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
