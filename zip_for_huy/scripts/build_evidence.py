#!/usr/bin/env python3
"""Build evidence.jsonl from questions via BM25 retrieval and evidence packing.

Complies with AGENTS.md:
- Question text and retrieved evidence ONLY (no gold answers enter retrieval).
- Fully provenance-preserving through repo pack_evidence.
- Supports deterministic held-out dev slicing from train.json.
- Supports semantic cross-encoder reranking with AITeamVN/Vietnamese_Reranker.
"""

from __future__ import annotations

import argparse
import gc
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
    rerank_hits_with_citations,
    select_dynamic_evidence_hits,
)
from legal_rag.pipeline import prepare_bm25_index_from_config  # noqa: E402
from legal_rag.retrieval.bm25 import retrieve_bm25  # noqa: E402
from legal_rag.schemas import RetrievalHit  # noqa: E402
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
        help="Retrieval rough top_n override (default: 125 candidates when reranker is active).",
    )
    parser.add_argument(
        "--evidence-top-k",
        type=int,
        default=None,
        help="Evidence top_k override (default from config).",
    )
    parser.add_argument(
        "--reranker",
        choices=("none", "aiteamvn"),
        default="none",
        help="Semantic reranker: 'aiteamvn' uses AITeamVN/Vietnamese_Reranker.",
    )
    parser.add_argument(
        "--reranker-adapter",
        type=Path,
        default=None,
        help="Optional path to finetuned LoRA reranker adapter checkpoint directory.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2304,
        help="Max sequence length for CrossEncoder (default: 2304 = 256 query + 2048 passage).",
    )
    parser.add_argument(
        "--citation-boost",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply explicit legal citation matching boost (Phần 2b) to reranked hits (default: False; regressed in dev200).",
    )
    parser.add_argument(
        "--citation-weight",
        type=float,
        default=1.0,
        help="Weight multiplier for citation match bonus (default: 1.0).",
    )
    parser.add_argument(
        "--dynamic-k",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dynamic score-based evidence selection (min-k to max-k) instead of fixed top-k.",
    )
    parser.add_argument(
        "--min-evidence-k",
        type=int,
        default=2,
        help="Minimum number of evidence chunks to keep when dynamic-k is enabled (default: 2).",
    )
    parser.add_argument(
        "--max-evidence-k",
        type=int,
        default=6,
        help="Maximum number of evidence chunks to keep when dynamic-k is enabled (default: 6).",
    )
    parser.add_argument(
        "--margin-top1",
        type=float,
        default=3.0,
        help="Score margin relative to Top 1 rerank score (default: 3.0).",
    )
    parser.add_argument(
        "--margin-top2",
        type=float,
        default=2.0,
        help="Score margin relative to Top 2 rerank score (default: 2.0).",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=None,
        help="Optional path to directory containing index.faiss and passage_metadata.jsonl for hybrid retrieval.",
    )
    parser.add_argument(
        "--dense-model",
        default="AITeamVN/Vietnamese_Embedding_v2",
        help="Dense embedding model ID (default: AITeamVN/Vietnamese_Embedding_v2).",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.25,
        help="BM25 weight in RRF fusion (default: 0.25, champion weighting).",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
        help="Dense weight in RRF fusion (default: 1.0, champion weighting).",
    )
    parser.add_argument(
        "--passages",
        type=Path,
        default=Path("/mnt/G/sedar-legalqa/artifacts/sedar_retrieval/views/parser_blankline_20260901/passages_r2a.jsonl"),
        help="Path to canonical passages_r2a.jsonl for hybrid dense chunk mapping.",
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

    if args.dense_index is not None and args.passages and args.passages.exists():
        print(f"[build_evidence] Loading canonical passages from {args.passages} for dense chunk lookup...")
        from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl, passage_to_legal_chunk
        passages_list = load_passages_jsonl(str(args.passages))
        chunks = dict(chunks)
        for p in passages_list:
            c = passage_to_legal_chunk(p, body_source="raw_text")
            chunks[c.chunk_id] = c
        print(f"[build_evidence] Loaded {len(passages_list)} canonical passages into chunk lookup.")

    # Default rough_top_n: if reranker is used, pull 125 candidates for reranking
    default_rough = 125 if args.reranker == "aiteamvn" else cfg.retrieval.rough_top_n
    rough_top_n = args.top_k or default_rough
    evidence_top_k = args.evidence_top_k or cfg.evidence.evidence_top_k

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_encoder = None
    if args.reranker == "aiteamvn":
        if args.reranker_adapter is not None:
            adapter_path = Path(args.reranker_adapter).resolve()
            print(
                f"[build_evidence] Loading finetuned Vietnamese_Reranker adapter from {adapter_path} "
                f"on {device} (max_length={args.max_length})..."
            )

            class FinetunedRerankerWrapper:
                def __init__(self, adapter_dir: Path, dev: str, max_len: int) -> None:
                    import torch
                    from peft import PeftModel
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer

                    self.device = dev
                    self.max_length = max_len
                    self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        "AITeamVN/Vietnamese_Reranker",
                        revision="f536976248403314225d7fdfdbc87f0e9516a54e",
                        torch_dtype=torch.bfloat16,
                    ).to(dev)
                    self.model = PeftModel.from_pretrained(base_model, str(adapter_dir)).to(dev)
                    self.model.eval()

                def predict(
                    self,
                    pairs: list[tuple[str, str]],
                    batch_size: int = 32,
                    show_progress_bar: bool = False,
                ) -> list[float]:
                    import torch

                    scores: list[float] = []
                    for i in range(0, len(pairs), batch_size):
                        batch = pairs[i : i + batch_size]
                        queries = [q for q, p in batch]
                        passages = [p for q, p in batch]
                        encoded = self.tokenizer(
                            queries,
                            passages,
                            padding=True,
                            truncation="only_second",
                            max_length=self.max_length,
                            return_tensors="pt",
                        ).to(self.device)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = self.model(**encoded, return_dict=True).logits.view(-1)
                            scores.extend([float(v) for v in logits.float().cpu()])
                    return scores

            cross_encoder = FinetunedRerankerWrapper(adapter_path, device, args.max_length)
        else:
            from sentence_transformers import CrossEncoder

            print(
                f"[build_evidence] Loading base AITeamVN/Vietnamese_Reranker on {device} "
                f"(max_length={args.max_length})..."
            )
            cross_encoder = CrossEncoder(
                "AITeamVN/Vietnamese_Reranker",
                device=device,
                max_length=args.max_length,
            )
    reranker = None
    if (
        args.reranker == "none"
        and cfg.reranker.enabled
        and cfg.retrieval.strategy == "bm25_rerank"
    ):
        model_name = cfg.reranker.model
        print(f"[build_evidence] Initializing reranker: {model_name}")
        reranker = create_reranker(cfg.reranker)

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

    dense_hits_by_qid = None
    if args.dense_index is not None:
        from legal_rag.sedar_retrieval.retrieval.dense import (
            SentenceTransformerEncoder,
            load_dense_index,
            normalize_embedding_matrix,
            search_dense_index,
        )

        dense_dir = args.dense_index.resolve()
        print(f"[build_evidence] Loading dense FAISS index from {dense_dir}...")
        loaded_dense = load_dense_index(dense_dir)
        print(f"[build_evidence] Loading dense encoder {args.dense_model} on {device}...")
        dtype_dense = (
            "bf16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "fp16"
        )
        dense_encoder = SentenceTransformerEncoder(
            model=args.dense_model, device=device, dtype=dtype_dense
        )
        print(f"[build_evidence] Encoding {len(query_pairs)} queries for dense retrieval...")
        query_texts = [text for _, text in query_pairs]
        q_vectors = dense_encoder.encode(query_texts, batch_size=32)
        q_vectors = normalize_embedding_matrix(q_vectors)
        dense_results = search_dense_index(
            loaded_dense.index, loaded_dense.passage_ids, q_vectors, top_k=rough_top_n
        )
        dense_hits_by_qid = {
            qid: hits for (qid, _), hits in zip(query_pairs, dense_results)
        }
        print(
            f"[build_evidence] Dense retrieval complete! Will fuse BM25 + Dense v2 via RRF "
            f"(bm25={args.bm25_weight}, dense={args.dense_weight})."
        )
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

            if dense_hits_by_qid is not None and qid in dense_hits_by_qid:
                d_hits = dense_hits_by_qid[qid]
                fused_scores: dict[str, float] = {}
                for hit in raw_hits:
                    fused_scores[hit.chunk_id] = fused_scores.get(
                        hit.chunk_id, 0.0
                    ) + args.bm25_weight / (60.0 + hit.rank)
                for d_hit in d_hits:
                    fused_scores[d_hit.passage_id] = fused_scores.get(
                        d_hit.passage_id, 0.0
                    ) + args.dense_weight / (60.0 + d_hit.rank)
                sorted_fused = sorted(
                    fused_scores.items(), key=lambda it: (-it[1], it[0])
                )[:rough_top_n]
                bm25_hit_map = {h.chunk_id: h for h in raw_hits}
                fused_hits: list[RetrievalHit] = []
                for rank, (cid, fused_score) in enumerate(sorted_fused, start=1):
                    if cid in bm25_hit_map:
                        fused_hits.append(
                            bm25_hit_map[cid].model_copy(
                                update={"rank": rank, "bm25_score": float(fused_score)}
                            )
                        )
                    elif cid in chunks:
                        chunk = chunks[cid]
                        fused_hits.append(
                            RetrievalHit(
                                chunk_id=cid,
                                document_id=chunk.document_id,
                                source_path=chunk.source_path,
                                source_member=chunk.source_member,
                                section_label=chunk.section_label,
                                start_offset=chunk.start_offset,
                                end_offset=chunk.end_offset,
                                rank=rank,
                                bm25_score=float(fused_score),
                            )
                        )
                raw_hits = tuple(fused_hits)
            if not raw_hits:
                row = {"id": qid, "evidence": "(không có trích đoạn phù hợp)"}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            ordered_hits = raw_hits
            if cross_encoder is not None:
                valid_hits = [h for h in raw_hits if h.chunk_id in chunks]
                passages = [chunks[h.chunk_id].raw_text for h in valid_hits]
                if passages:
                    scores = cross_encoder.predict(
                        [(question_text, p) for p in passages],
                        batch_size=32,
                        show_progress_bar=False,
                    )
                    scored = sorted(
                        zip(scores, valid_hits, strict=True),
                        key=lambda item: float(item[0]),
                        reverse=True,
                    )
                    ordered_hits = tuple(
                        hit.model_copy(
                            update={"rerank_score": float(score), "rank": rank}
                        )
                        for rank, (score, hit) in enumerate(scored, start=1)
                    )
            elif reranker is not None:
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

            if args.citation_boost:
                ordered_hits = rerank_hits_with_citations(
                    ordered_hits,
                    chunks,
                    question_text,
                    documents=documents,
                    citation_weight=args.citation_weight,
                )

            dedup = deduplicate_retrieved_chunks(ordered_hits, chunks)
            if args.dynamic_k:
                selected_hits = select_dynamic_evidence_hits(
                    dedup.kept_hits,
                    min_k=args.min_evidence_k,
                    max_k=args.max_evidence_k,
                    margin_top1=args.margin_top1,
                    margin_top2=args.margin_top2,
                )
            else:
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

    if cross_encoder is not None:
        del cross_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[build_evidence] Completed! Evidence written to {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
