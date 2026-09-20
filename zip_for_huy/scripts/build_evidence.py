#!/usr/bin/env python3
"""Build evidence.jsonl from questions via BM25 retrieval and evidence packing.

Complies with AGENTS.md:
- Question text and retrieved evidence ONLY (no gold answers enter retrieval).
- Fully provenance-preserving through repo pack_evidence.
- Supports deterministic held-out dev slicing from train.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure repo root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from legal_rag.config import load_config  # noqa: E402
from legal_rag.evidence import (  # noqa: E402
    deduplicate_retrieved_chunks,
    pack_evidence,
)
from legal_rag.pipeline import prepare_bm25_index_from_config  # noqa: E402
from legal_rag.retrieval.bm25 import retrieve_bm25  # noqa: E402
from legal_rag.retrieval.reranker import (  # noqa: E402
    RerankResult,
    create_reranker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bm25_rag.yaml"),
        help="Path to pipeline configuration (e.g. configs/bm25_rag.yaml).",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/train.json"),
        help="Source questions JSON file.",
    )
    parser.add_argument(
        "--output-evidence",
        type=Path,
        required=True,
        help="Path to output evidence.jsonl.",
    )
    parser.add_argument(
        "--output-questions",
        type=Path,
        default=None,
        help="Optional path to output sliced questions JSON.",
    )
    parser.add_argument(
        "--split-holdout",
        type=int,
        default=1500,
        help="Number of held-out questions to slice (e.g. 1500). 0 disables.",
    )
    parser.add_argument(
        "--slice-mode",
        choices=("last", "random"),
        default="last",
        help="How to select holdout slice: 'last' takes last N, 'random' shuffles.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic holdout slicing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of questions to process.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval rough top_n override (default from config).",
    )
    parser.add_argument(
        "--evidence-top-k",
        type=int,
        default=None,
        help="Evidence top_k override (default from config).",
    )
    return parser.parse_args()


def load_and_slice_questions(
    questions_path: Path,
    *,
    holdout_size: int,
    slice_mode: str,
    seed: int,
    limit: int | None = None,
) -> tuple[dict[str, dict[str, str]], list[tuple[str, str]]]:
    raw = json.loads(questions_path.read_text(encoding="utf-8"))
    items = list(raw.items())

    if 0 < holdout_size < len(items):
        if slice_mode == "last":
            items = items[-holdout_size:]
        elif slice_mode == "random":
            rng = random.Random(seed)
            shuffled = list(items)
            rng.shuffle(shuffled)
            items = shuffled[:holdout_size]
            items.sort(key=lambda x: str(x[0]))

    if limit and limit > 0:
        items = items[:limit]

    selected_dict = {qid: data for qid, data in items}
    query_pairs = [
        (str(qid), str(data.get("question", "")).strip()) for qid, data in items
    ]
    return selected_dict, query_pairs


def main() -> int:
    args = parse_args()
    print(f"[build_evidence] Loading config from {args.config}")
    cfg = load_config(args.config)

    print(f"[build_evidence] Loading questions from {args.questions}")
    sliced_dict, query_pairs = load_and_slice_questions(
        args.questions,
        holdout_size=args.split_holdout,
        slice_mode=args.slice_mode,
        seed=args.seed,
        limit=args.limit,
    )
    n_queries = len(query_pairs)
    print(
        f"[build_evidence] Selected {n_queries} questions "
        f"(slice_mode={args.slice_mode}, holdout={args.split_holdout})"
    )

    if args.output_questions:
        args.output_questions.parent.mkdir(parents=True, exist_ok=True)
        args.output_questions.write_text(
            json.dumps(sliced_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[build_evidence] Wrote {len(sliced_dict)} questions")

    print("[build_evidence] Preparing BM25 index & chunks...")
    prep = prepare_bm25_index_from_config(args.config, repo_root=_REPO_ROOT)
    chunks = (
        prep.chunk_lookup
        if prep.chunk_lookup is not None
        else {c.chunk_id: c for c in prep.chunks}
    )
    documents = prep.documents
    index = prep.index

    rough_top_n = args.top_k or cfg.retrieval.rough_top_n
    evidence_top_k = args.evidence_top_k or cfg.evidence.evidence_top_k

    reranker = None
    if cfg.reranker.enabled and cfg.retrieval.strategy == "bm25_rerank":
        model_name = cfg.reranker.model
        print(f"[build_evidence] Initializing reranker: {model_name}")
        reranker = create_reranker(cfg.reranker)

    import torch

    from legal_rag.retrieval.bm25 import (
        build_bm25_cuda_query_cache,
        build_bm25_query_cache,
    )

    use_cuda = torch.cuda.is_available()
    query_cache = None
    cuda_query_cache = None

    if use_cuda:
        try:
            print("[build_evidence] Building BM25 CUDA query cache...")
            all_queries = [text for _, text in query_pairs if text]
            query_cache = build_bm25_query_cache(index, all_queries)
            cuda_query_cache = build_bm25_cuda_query_cache(query_cache)
            print("[build_evidence] CUDA query cache ready!")
        except Exception as e:
            print(f"[build_evidence] CUDA query cache fallback to CPU: {e}")
            use_cuda = False

    try:
        from tqdm import tqdm

        iterator = tqdm(query_pairs, desc="Retrieving & packing evidence")
    except ImportError:
        iterator = query_pairs

    args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.output_evidence.open("w", encoding="utf-8") as handle:
        for qid, question_text in iterator:
            if not question_text:
                continue

            if use_cuda and query_cache is not None:
                raw_hits = retrieve_bm25(
                    index,
                    question_text,
                    top_k=rough_top_n,
                    query_cache=query_cache,
                    backend="cuda",
                    cuda_query_cache=cuda_query_cache,
                )
            else:
                raw_hits = retrieve_bm25(index, question_text, top_k=rough_top_n)
            if not raw_hits:
                row = {"id": qid, "evidence": "(không có trích đoạn phù hợp)"}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            ordered_hits = raw_hits
            if reranker is not None:
                candidate_texts = {
                    hit.chunk_id: chunks[hit.chunk_id].retrieval_text
                    for hit in raw_hits
                    if hit.chunk_id in chunks
                }
                rerank_res = reranker.rerank(
                    question_text, raw_hits, candidate_texts=candidate_texts
                )
                if isinstance(rerank_res, RerankResult) and rerank_res.used:
                    ordered_hits = rerank_res.hits

            dedup = deduplicate_retrieved_chunks(ordered_hits, chunks)
            selected_hits = dedup.kept_hits[:evidence_top_k]

            try:
                packed = pack_evidence(
                    selected_hits,
                    chunks,
                    max_total_chars=cfg.evidence.max_total_chars,
                    max_chunks_per_document=cfg.evidence.max_chunks_per_document,
                    documents=documents,
                )
                evidence_text = packed.rendered_text
            except Exception as e:
                evidence_text = f"(lỗi trích xuất bằng chứng: {e})"

            row = {"id": qid, "evidence": evidence_text}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[build_evidence] Completed! Evidence written to {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
