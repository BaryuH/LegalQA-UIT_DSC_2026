"""Validated, config-driven settings for the Vietnamese Legal RAG pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


DeviceSetting = Literal["auto", "cpu", "cuda"]


class DataSection(BaseModel):
    """Paths and split configuration."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str = "data"
    question_path: str = "data/warmup.json"
    selected_contexts_path: str = "data/selected-contexts.zip"
    split: str = "warmup"


class ChunkingSection(BaseModel):
    """Legal-text chunking parameters."""

    model_config = ConfigDict(extra="forbid")

    max_chars: int = Field(default=1200, gt=0)
    overlap_chars: int = Field(default=200, ge=0)
    min_chars: int = Field(default=100, gt=0)
    num_workers: int = Field(default=30, gt=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be less than max_chars")
        return self


class BM25Section(BaseModel):
    """BM25 retrieval parameters."""

    model_config = ConfigDict(extra="forbid")

    top_n: int = Field(default=30, gt=0)
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    num_workers: int = Field(default=30, gt=0)


class DenseSection(BaseModel):
    """Dense retrieval (vietlegal-e5) parameters."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "mainguyen9/vietlegal-e5"
    top_n: int = Field(default=30, gt=0)
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = Field(default=128, gt=0)


class HybridSection(BaseModel):
    """Hybrid retrieval fusion parameters."""

    model_config = ConfigDict(extra="forbid")

    # Reciprocal Rank Fusion constant
    rrf_k: int = Field(default=60, gt=0)
    top_n: int = Field(default=30, gt=0)


class RerankerSection(BaseModel):
    """ViRanker reranking parameters."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model_name: str = "namdp-ptit/ViRanker"
    top_k: int = Field(default=3, gt=0)
    batch_size: int = Field(default=64, gt=0)


class GeneratorSection(BaseModel):
    """LLM generator parameters."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2"
    max_new_tokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    do_sample: bool = False
    dtype: str = "auto"


class EvidenceSection(BaseModel):
    """Evidence packing parameters."""

    model_config = ConfigDict(extra="forbid")

    max_total_chars: int = Field(default=4000, gt=0)


class EvaluationSection(BaseModel):
    """Evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    primary_metric: str = "meteor"
    secondary_metric: str = "rouge_l"


class RuntimeSection(BaseModel):
    """Runtime and environment settings."""

    model_config = ConfigDict(extra="forbid")

    cache_dir: str = "cache_nhan"
    outputs_dir: str = "outputs_nhan"
    device: DeviceSetting = "auto"
    seed: int = 42
    fail_fast: bool = True


class PipelineConfig(BaseModel):
    """Top-level validated pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    data: DataSection = DataSection()
    chunking: ChunkingSection = ChunkingSection()
    bm25: BM25Section = BM25Section()
    dense: DenseSection = DenseSection()
    hybrid: HybridSection = HybridSection()
    reranker: RerankerSection = RerankerSection()
    generator: GeneratorSection = GeneratorSection()
    evidence: EvidenceSection = EvidenceSection()
    evaluation: EvaluationSection = EvaluationSection()
    runtime: RuntimeSection = RuntimeSection()

    def fingerprint(self) -> str:
        """Content-addressed config fingerprint for reproducibility."""
        raw = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML configuration file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")
    return PipelineConfig.model_validate(raw)
