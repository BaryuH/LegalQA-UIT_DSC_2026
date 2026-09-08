"""Main RAG pipeline orchestrator.

Flow: Question → BM25 + Dense → Hybrid RRF → ViReranker → Evidence Pack → Qwen3 Generator → Prediction

All components are wired through validated PipelineConfig.
Gold answers NEVER enter any inference path.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunker import LegalChunk, ParentChunk, chunk_corpus
from .config import PipelineConfig
from .data_loader import (
    InferenceQuestion,
    load_gold_answers,
    load_legal_contexts,
    load_questions,
)
from .evidence import PackedEvidence, pack_evidence
from .generator import GenerationResult, GeneratorError, LegalGenerator
from .reranker import RerankHit, RerankerError, ViReranker
from .retriever_bm25 import BM25Retriever
from .retriever_dense import DenseRetriever
from .retriever_hybrid import HybridHit, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full pipeline run result."""

    predictions: dict[str, str]
    errors: list[dict[str, str]]
    total_questions: int
    successful: int
    failed: int
    total_latency_ms: float


def _set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _resolve_device_and_optimize(setting: str) -> str:
    """Resolve device and optimize PyTorch runtime for RTX 3090 (Ampere) and AMD EPYC (30 cores)."""
    import os

    # Pin OpenMP and MKL thread pools to 30 CPU cores
    os.environ["OMP_NUM_THREADS"] = "30"
    os.environ["MKL_NUM_THREADS"] = "30"
    os.environ["OPENBLAS_NUM_THREADS"] = "30"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "30"
    os.environ["NUMEXPR_NUM_THREADS"] = "30"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    resolved = "cuda" if setting in ("auto", "cuda") else "cpu"
    try:
        import torch

        # Limit PyTorch intra-op threads to 30 CPU cores (prevents cross-socket NUMA penalty)
        torch.set_num_threads(30)

        if torch.cuda.is_available() and resolved == "cuda":
            # Enable TensorFloat-32 (TF32) on Ampere architecture (RTX 3090)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            logger.info(
                "Hardware optimization enabled: RTX 3090 (TF32 + bfloat16), AMD EPYC (30 threads)."
            )
            return "cuda"
    except ImportError:
        pass
    return resolved


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the full RAG pipeline.

    Steps:
    1. Load questions (inference-safe, no gold answers)
    2. Load legal contexts from ZIP (no extraction)
    3. Chunk documents
    4. Build BM25 + Dense indexes
    5. For each question: retrieve → rerank → pack evidence → generate
    6. Collect predictions
    """
    t_start = time.perf_counter()

    _set_seed(config.runtime.seed)

    # Ensure output directories exist
    outputs_dir = Path(config.runtime.outputs_dir)
    cache_dir = Path(config.runtime.cache_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device_and_optimize(config.runtime.device)

    # --- Step 1: Load questions ---
    logger.info("Loading questions from %s...", config.data.question_path)
    questions = load_questions(
        config.data.question_path, split=config.data.split
    )
    logger.info("Loaded %d questions (split=%s).", len(questions), config.data.split)

    # --- Step 2: Load legal contexts ---
    logger.info(
        "Loading legal contexts from %s...",
        config.data.selected_contexts_path,
    )
    documents = load_legal_contexts(config.data.selected_contexts_path)
    logger.info("Loaded %d legal documents.", len(documents))

    # --- Step 3: Chunk documents ---
    logger.info("Chunking %d documents (clause-level & parent-child)...", len(documents))
    corpus_result = chunk_corpus(
        documents,
        max_chars=config.chunking.max_chars,
        overlap_chars=config.chunking.overlap_chars,
        min_chars=config.chunking.min_chars,
        num_workers=config.chunking.num_workers,
    )
    chunks = corpus_result.chunks
    parents_by_id = corpus_result.parents_by_id
    chunks_by_id = corpus_result.chunks_by_id
    logger.info(
        "Created %d child chunks and %d parent documents from %d documents.",
        len(chunks),
        len(corpus_result.parents),
        len(documents),
    )

    logger.info("Building BM25 retriever (multi-core & caching enabled)...")
    bm25_retriever = BM25Retriever(
        chunks,
        k1=config.bm25.k1,
        b=config.bm25.b,
        cache_dir=str(cache_dir),
        num_workers=config.bm25.num_workers,
    )

    logger.info("Building dense retriever (vietlegal-e5)...")
    dense_retriever = DenseRetriever(
        chunks,
        model_name=config.dense.model_name,
        query_prefix=config.dense.query_prefix,
        passage_prefix=config.dense.passage_prefix,
        batch_size=config.dense.batch_size,
        device=device,
        cache_dir=str(cache_dir),
    )

    # --- Step 5: Load reranker ---
    reranker: ViReranker | None = None
    if config.reranker.enabled:
        logger.info("Loading reranker (ViRanker)...")
        reranker = ViReranker(
            model_name=config.reranker.model_name,
            device=device,
            batch_size=config.reranker.batch_size,
        )

    # --- Step 6: Load generator ---
    logger.info("Loading generator (Qwen3-1.7B)...")
    generator = LegalGenerator(
        model_name=config.generator.model_name,
        max_new_tokens=config.generator.max_new_tokens,
        temperature=config.generator.temperature,
        do_sample=config.generator.do_sample,
        dtype=config.generator.dtype,
        device=device,
    )

    # --- Step 7: Process each question ---
    predictions: dict[str, str] = {}
    errors: list[dict[str, str]] = []

    for i, question in enumerate(questions):
        qid = question.id
        logger.info(
            "[%d/%d] Processing question %s...", i + 1, len(questions), qid
        )

        try:
            # Retrieve
            bm25_hits = bm25_retriever.retrieve(
                question.question, top_n=config.bm25.top_n
            )
            dense_hits = dense_retriever.retrieve(
                question.question, top_n=config.dense.top_n
            )

            # Hybrid fusion
            hybrid_hits = reciprocal_rank_fusion(
                bm25_hits,
                dense_hits,
                k=config.hybrid.rrf_k,
                top_n=config.hybrid.top_n,
            )

            # Rerank
            if reranker is not None and hybrid_hits:
                reranked = reranker.rerank(
                    question.question,
                    hybrid_hits,
                    chunks_by_id,
                    top_k=config.reranker.top_k,
                )
            else:
                # Convert hybrid hits to RerankHit format (no reranker)
                reranked = [
                    RerankHit(
                        chunk_id=h.chunk_id,
                        document_id=h.document_id,
                        rerank_score=h.rrf_score,
                        original_rrf_score=h.rrf_score,
                        rank=h.rank,
                    )
                    for h in hybrid_hits[: config.reranker.top_k]
                ]

            # Pack evidence using Small-to-Big (Parent Document)
            packed = pack_evidence(
                reranked,
                chunks_by_id,
                parents_by_id=parents_by_id,
                max_total_chars=config.evidence.max_total_chars,
            )

            # Generate answer
            result = generator.generate(
                question_id=qid,
                question=question.question,
                evidence_text=packed.rendered_text,
            )

            predictions[qid] = result.answer
            logger.info(
                "[%d/%d] Done (%.0fms, %d chars answer)",
                i + 1,
                len(questions),
                result.latency_ms,
                len(result.answer),
            )

        except (GeneratorError, RerankerError) as exc:
            error_entry = {
                "question_id": qid,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(error_entry)
            logger.error(
                "[%d/%d] FAILED question %s: %s",
                i + 1,
                len(questions),
                qid,
                exc,
            )
            if config.runtime.fail_fast:
                raise

        except Exception as exc:
            error_entry = {
                "question_id": qid,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(error_entry)
            logger.error(
                "[%d/%d] UNEXPECTED ERROR question %s: %s",
                i + 1,
                len(questions),
                qid,
                exc,
            )
            if config.runtime.fail_fast:
                raise

    total_latency = (time.perf_counter() - t_start) * 1000

    result = PipelineResult(
        predictions=predictions,
        errors=errors,
        total_questions=len(questions),
        successful=len(predictions),
        failed=len(errors),
        total_latency_ms=total_latency,
    )

    # --- Save predictions ---
    pred_path = outputs_dir / f"predictions_{config.data.split}.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for qid, answer in predictions.items():
            fh.write(
                json.dumps(
                    {"id": qid, "answer": answer},
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("Predictions saved to %s", pred_path)

    # --- Save errors if any ---
    if errors:
        err_path = outputs_dir / f"errors_{config.data.split}.json"
        with err_path.open("w", encoding="utf-8") as fh:
            json.dump(errors, fh, ensure_ascii=False, indent=2)
        logger.info("Errors saved to %s", err_path)

    logger.info(
        "Pipeline complete: %d/%d succeeded, %d failed, %.1fs total",
        result.successful,
        result.total_questions,
        result.failed,
        total_latency / 1000,
    )

    return result
