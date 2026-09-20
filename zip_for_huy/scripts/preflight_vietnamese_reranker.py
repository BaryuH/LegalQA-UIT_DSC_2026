#!/usr/bin/env python3
"""Fail-closed preflight for the AITeamVN/Vietnamese_Reranker deployment.

Runs before any GPU time is spent. Every check either passes or aborts; none of
them fall back to a weaker configuration, because a silent fallback is what
turns a model change into an unexplained metric change.

Checks, and what each one is guarding against:

1. Local snapshot present            - no hub download mid-run.
2. CUDA available when requested     - no silent CPU scoring at 3.5 docs/s.
3. Tokenizer window >= 2304          - the model card's 256 + 2048 budget; the
                                       repo's existing cross-encoder default is
                                       512, which would truncate the majority of
                                       corpus v4 units (median 932 chars).
4. Single-label head                 - the score must be one logit per pair.
5. Query survives truncation         - only_second must not eat the question.
6. Score separation on a smoke pair  - a relevant/irrelevant pair must not tie.
7. Determinism                       - the same input twice, identical scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.ranking.vietnamese_reranker import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    VietnameseRerankerConfig,
    VietnameseRerankerError,
    VietnameseRerankerScorer,
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


def _fail(check: str, detail: str) -> None:
    raise SystemExit(f"PREFLIGHT FAIL [{check}] {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Repo id or, preferred on the server, a local snapshot directory.",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-query-tokens", type=int, default=256)
    parser.add_argument("--max-passage-tokens", type=int, default=2048)
    parser.add_argument(
        "--allow-hub",
        action="store_true",
        help="Permit hub resolution. Off by default: runs must be reproducible.",
    )
    parser.add_argument("--min-score-gap", type=float, default=0.5)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, object] = {"model": args.model, "checks": {}}
    checks: dict[str, object] = report["checks"]  # type: ignore[assignment]

    # 1. Snapshot
    is_local = Path(args.model).expanduser().is_dir()
    if not is_local and not args.allow_hub:
        _fail(
            "local_snapshot",
            f"{args.model!r} is not a local directory and --allow-hub was not "
            "passed. Stage the snapshot on the server first.",
        )
    checks["local_snapshot"] = "local" if is_local else "hub_allowed"

    config = VietnameseRerankerConfig(
        model=args.model,
        revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        max_query_tokens=args.max_query_tokens,
        max_passage_tokens=args.max_passage_tokens,
        max_length=args.max_length,
        batch_size=args.batch_size,
        local_files_only=not args.allow_hub,
    )
    checks["config"] = config.as_dict()

    # 2. CUDA
    try:
        import torch
    except ImportError:
        _fail("torch", "torch is not installed in this environment")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        _fail("cuda", "device='cuda' requested but torch.cuda.is_available() is False")
    checks["cuda"] = {
        "requested": args.device,
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }

    try:
        scorer = VietnameseRerankerScorer(config)
    except VietnameseRerankerError as exc:
        _fail("load", str(exc))

    # 3. Tokenizer window
    model_max = int(getattr(scorer.tokenizer, "model_max_length", 0) or 0)
    if model_max and model_max < args.max_length:
        _fail(
            "tokenizer_window",
            f"tokenizer.model_max_length={model_max} < --max-length={args.max_length}. "
            "The model card gives query 256 + passage 2048 = 2304.",
        )
    checks["tokenizer_window"] = {"model_max_length": model_max, "used": args.max_length}

    # 4. Single-label head
    num_labels = int(getattr(getattr(scorer.model, "config", None), "num_labels", 0) or 0)
    if num_labels != 1:
        _fail(
            "num_labels",
            f"expected a single-logit reranker head, found num_labels={num_labels}",
        )
    checks["num_labels"] = num_labels

    # 5. Query survives truncation
    long_query = (_SMOKE_QUERY + " ") * 20
    clipped = scorer._truncate_query(long_query)
    if not clipped.strip():
        _fail("query_truncation", "the query was truncated to nothing")
    checks["query_truncation"] = {
        "input_chars": len(long_query),
        "kept_chars": len(clipped),
    }

    # 6. Score separation
    scores = list(scorer(_SMOKE_QUERY, [_SMOKE_RELEVANT, _SMOKE_IRRELEVANT]))
    gap = scores[0] - scores[1]
    checks["smoke_scores"] = {
        "relevant": scores[0],
        "irrelevant": scores[1],
        "gap": gap,
        "min_required": args.min_score_gap,
    }
    if gap < args.min_score_gap:
        _fail(
            "score_separation",
            f"relevant={scores[0]:.4f} irrelevant={scores[1]:.4f} gap={gap:.4f} "
            f"< --min-score-gap={args.min_score_gap}. A saturated or untrained "
            "head cannot carry a cutoff threshold.",
        )

    # 7. Determinism
    again = list(scorer(_SMOKE_QUERY, [_SMOKE_RELEVANT, _SMOKE_IRRELEVANT]))
    if again != scores:
        _fail(
            "determinism",
            f"the same input scored differently on a second pass: {scores} vs {again}",
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
