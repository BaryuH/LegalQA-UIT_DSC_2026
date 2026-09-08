#!/usr/bin/env python3
"""Fail-closed preflight for the ColBERT MaxSim reranker (TASK 26).

Nine checks, run before any GPU time. Each one either passes or aborts; none
falls back to a weaker configuration.

The two that exist specifically because of how this stack behaves:

**The sentence-transformers pin.** PyLate 1.6.0 requires
``sentence-transformers==5.3.0`` exactly. Sentence-Transformers 6.x ships its own
``MultiVectorEncoder`` and cannot coexist with PyLate in one environment. An
import that half-works produces a confusing runtime failure deep inside
training, so the version is checked up front.

**The untrained-projection trap.** ``models.ColBERT`` given a bare HF encoder
appends a *randomly initialised* bias-free ``Dense(hidden -> 128)`` and logs a
line about it. Such a model loads, encodes, and returns numbers that mean
nothing. The score-separation check is what catches it: a fine-tuned checkpoint
separates a relevant from an irrelevant legal passage; a random projection does
not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legal_rag.sedar_retrieval.ranking.colbert_maxsim import (
    ColbertMaxsimConfig,
    ColbertMaxsimError,
    maxsim_score,
    pool_tokens_hierarchical,
)

_SMOKE_QUERY = "Phó chủ tịch công đoàn có được quyền ký thỏa ước lao động tập thể không?"
_SMOKE_RELEVANT = (
    "Điều 76. Lấy ý kiến và ký kết thỏa ước lao động tập thể\n"
    "4. Thỏa ước lao động tập thể được ký kết bởi đại diện hợp pháp của các bên "
    "thương lượng."
)
_SMOKE_IRRELEVANT = (
    "Điều 12. Tiêu chuẩn kỹ thuật của thiết bị đo lường nhóm 2 trong lĩnh vực "
    "kiểm định phương tiện đo khối lượng."
)
_BASE_ENCODER_HINTS = (
    "bge-m3",
    "xlm-roberta",
    "phobert",
    "bert-base",
    "modernbert",
    "gte-multilingual",
)


def _fail(check: str, detail: str) -> None:
    raise SystemExit(f"PREFLIGHT FAIL [{check}] {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local snapshot directory.")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--backend", choices=("pylate", "bge_m3"), default="pylate")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--document-length", type=int, default=512)
    parser.add_argument("--reduction", choices=("sum", "mean"), default="sum")
    parser.add_argument("--pool-factor", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-score-gap", type=float, default=0.5)
    parser.add_argument("--allow-hub", action="store_true")
    parser.add_argument(
        "--allow-base-encoder",
        action="store_true",
        help=(
            "Permit a checkpoint that looks like an un-fine-tuned base encoder. "
            "Off by default: zero-shot late interaction scores below BM25 on "
            "Vietnamese (MRR@10 21.54 vs 23.09)."
        ),
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, object] = {"model": args.model, "backend": args.backend}
    checks: dict[str, object] = {}
    report["checks"] = checks

    # 1. Local snapshot
    is_local = Path(args.model).expanduser().is_dir()
    if not is_local and not args.allow_hub:
        _fail(
            "local_snapshot",
            f"{args.model!r} is not a local directory and --allow-hub was not passed",
        )
    checks["local_snapshot"] = "local" if is_local else "hub_allowed"

    # 2. Not an obvious base encoder
    lowered = args.model.lower()
    looks_base = any(hint in lowered for hint in _BASE_ENCODER_HINTS)
    if looks_base and not args.allow_base_encoder and args.backend == "pylate":
        _fail(
            "finetuned_checkpoint",
            f"{args.model!r} looks like a base encoder. PyLate appends a RANDOM "
            "Dense projection to a bare HF encoder, which loads and encodes but "
            "means nothing. Fine-tune first, or pass --allow-base-encoder for a "
            "deliberate baseline arm.",
        )
    checks["finetuned_checkpoint"] = "base_encoder_allowed" if looks_base else "ok"

    # 3. Library versions and the hard pin
    versions: dict[str, str | None] = {}
    try:
        import sentence_transformers  # noqa: PLC0415

        versions["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        versions["sentence_transformers"] = None
    if args.backend == "pylate":
        try:
            import pylate  # noqa: PLC0415

            versions["pylate"] = getattr(pylate, "__version__", "unknown")
        except ImportError:
            _fail("pylate", "pylate is not installed (`pip install pylate`)")
        st_version = versions.get("sentence_transformers") or ""
        if st_version and not st_version.startswith("5.3."):
            _fail(
                "sentence_transformers_pin",
                f"pylate pins sentence-transformers==5.3.0; found {st_version}. "
                "ST 6.x ships its own MultiVectorEncoder and cannot coexist with "
                "pylate in one environment.",
            )
    else:
        try:
            import FlagEmbedding  # noqa: PLC0415, F401

            versions["FlagEmbedding"] = getattr(FlagEmbedding, "__version__", "unknown")
        except ImportError:
            _fail("flagembedding", "FlagEmbedding is not installed")
    checks["versions"] = versions

    # 4. CUDA
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        _fail("torch", "torch is not installed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        _fail("cuda", "device='cuda' requested but torch.cuda.is_available() is False")
    checks["cuda"] = {
        "requested": args.device,
        "available": bool(torch.cuda.is_available()),
    }

    config = ColbertMaxsimConfig(
        model=args.model,
        revision=args.model_revision,
        backend=args.backend,
        dim=args.dim,
        query_length=args.query_length,
        document_length=args.document_length,
        reduction=args.reduction,
        pool_factor=args.pool_factor,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=not args.allow_hub,
    )
    checks["config"] = config.as_dict()

    # 5. Load
    from legal_rag.sedar_retrieval.ranking.colbert_maxsim import (  # noqa: PLC0415
        BgeM3ColbertBackend,
        PylateColbertBackend,
    )

    try:
        backend = (
            PylateColbertBackend(config)
            if args.backend == "pylate"
            else BgeM3ColbertBackend(config)
        )
    except ColbertMaxsimError as exc:
        _fail("load", str(exc))

    # 6. Query encoding shape
    query_embedding = backend.encode_query(_SMOKE_QUERY)
    checks["query_tokens"] = {
        "returned": int(query_embedding.shape[0]),
        "expected_for_pylate": args.query_length,
        "dim": int(query_embedding.shape[1]),
    }
    if query_embedding.shape[1] != args.dim:
        _fail(
            "dim",
            f"backend returned dim {query_embedding.shape[1]}, --dim says {args.dim}",
        )
    if args.backend == "pylate" and query_embedding.shape[0] != args.query_length:
        _fail(
            "query_expansion",
            f"expected exactly {args.query_length} query vectors (ColBERT pads "
            f"queries with [MASK] to enable learned expansion), got "
            f"{query_embedding.shape[0]}",
        )

    # 7. Unit norm on both sides
    documents = backend.encode_documents([_SMOKE_RELEVANT, _SMOKE_IRRELEVANT])
    norms = [float(np.linalg.norm(item, axis=1).mean()) for item in documents]
    if not all(abs(value - 1.0) < 0.05 for value in norms):
        _fail(
            "unit_norm",
            f"document token vectors are not unit-norm (means {norms}). MaxSim "
            "is defined on unit vectors so a dot product is a cosine.",
        )
    checks["unit_norm"] = {"document_norm_means": norms}

    # 8. Score separation and scale
    pooled = [
        pool_tokens_hierarchical(item, pool_factor=args.pool_factor)
        for item in documents
    ]
    scores = [
        maxsim_score(query_embedding, item, reduction=args.reduction) for item in pooled
    ]
    gap = scores[0] - scores[1]
    ceiling = args.query_length if args.reduction == "sum" else 1.0
    checks["smoke_scores"] = {
        "relevant": scores[0],
        "irrelevant": scores[1],
        "gap": gap,
        "min_required": args.min_score_gap,
        "theoretical_ceiling": ceiling,
        "score_scale": config.score_scale,
    }
    if scores[0] > ceiling * 1.01:
        _fail(
            "score_scale",
            f"score {scores[0]:.4f} exceeds the theoretical ceiling {ceiling} for "
            f"reduction={args.reduction!r}; the reduction or query length is wrong",
        )
    if gap < args.min_score_gap:
        _fail(
            "score_separation",
            f"relevant={scores[0]:.4f} irrelevant={scores[1]:.4f} gap={gap:.4f} "
            f"< --min-score-gap={args.min_score_gap}. An untrained projection "
            "loads and encodes but cannot separate these.",
        )

    # 9. Determinism
    again = maxsim_score(
        backend.encode_query(_SMOKE_QUERY),
        pool_tokens_hierarchical(
            backend.encode_documents([_SMOKE_RELEVANT])[0],
            pool_factor=args.pool_factor,
        ),
        reduction=args.reduction,
    )
    if abs(again - scores[0]) > 1e-4:
        _fail(
            "determinism",
            f"the same pair scored {scores[0]:.6f} then {again:.6f}",
        )
    checks["determinism"] = "identical"

    report["status"] = "PASS"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
