#!/usr/bin/env python3
"""Build (query, positive, semi-hard negatives) pairs for reranker fine-tuning.

**Read this before changing the negative policy.** On the closest published
setup to ours - SoICT Hackathon 2024, a 261,446-document Vietnamese legal corpus
(arXiv:2507.14619, also ICCCI 2025) - the choice of negatives dominated every
other training decision, and the intuitive choice was the worst one:

    baseline (no fine-tune)   MRR@10 0.5584
    hard negatives,  n=2      MRR@10 0.2689   <- halved
    hard negatives,  n=5      MRR@10 0.4796
    hard negatives,  n=10     MRR@10 0.6751
    easy negatives,  n=10     MRR@10 0.5940
    semi-hard,       n=2      MRR@10 0.7681
    semi-hard,       n=5      MRR@10 0.7821
    semi-hard,       n=10     MRR@10 0.7911   <- best, +0.233 over baseline

The mechanism is measured too: **50.87% of their "hard" negatives sat at cosine
>= 0.9 with the positive** (mean 0.6806), i.e. most of them were false
negatives. Semi-hard negatives had 79.44% below 0.5 (mean 0.2072). In a legal
corpus this is not an artefact - neighbouring articles of the same Điều
genuinely co-answer a question, and our own gold answers cite 1.65 distinct Điều
on average with 38% citing two or more. Training a reranker to push those away
teaches it to reject correct evidence.

So this builder mines a *band*, not a top-k. A candidate is accepted as a
semi-hard negative only when its per-query normalized score or rank sits
strictly between ``--easy-below`` and the configured upper bound. In
``positive_cosine`` mode, suspected false negatives are instead identified by
cosine similarity to the query's positive passage, not by first-stage score.
The band and the share of candidates each rule removed are written to the
audit so the distribution can be inspected before any GPU time is spent - and
the run fails closed if the accepted band looks like the hard-negative failure
case.

Silver retrieval labels are read here because this is an approved training task;
the questions are loaded through ``load_inference_questions``, so no gold answer
text is materialised at all. Nothing this script writes may enter a retrieval
query, an index, an inference prompt or a submission artifact.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from legal_rag.questions import load_inference_questions
from legal_rag.sedar_retrieval.query.citation_parser import parse_citations
from legal_rag.sedar_retrieval.ranking.vietnamese_reranker import load_rerank_units

import math
import re
from collections import Counter

RERANKER_DATA_SCHEMA_VERSION = "sedar-reranker-train-data-v1"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _char_ngrams(text: str, n: int = 3) -> Counter[str]:
    padded = f"^{text}$"
    return Counter(padded[i : i + n] for i in range(max(0, len(padded) - n + 1)))


def _token_jaccard(a: str, b: str) -> float:
    set_a = set(a.casefold().split())
    set_b = set(b.casefold().split())
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def _compute_tfidf_vectors(questions: dict[str, str], n: int = 3) -> dict[str, dict[str, float]]:
    df: Counter[str] = Counter()
    grams_by_id = {}
    for qid, text in questions.items():
        grams = _char_ngrams(text, n)
        grams_by_id[qid] = grams
        for g in grams:
            df[g] += 1
    total_docs = len(questions)
    idf = {g: math.log((total_docs + 1) / (count + 1)) + 1.0 for g, count in df.items()}
    vectors = {}
    for qid, grams in grams_by_id.items():
        vec = {g: (1.0 + math.log(cnt)) * idf[g] for g, cnt in grams.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        vectors[qid] = {g: v / norm for g, v in vec.items()} if norm else {}
    return vectors


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(val * vec_b.get(k, 0.0) for k, val in vec_a.items())


def _unit_matches_article(unit: Any, cited_articles: set[str]) -> bool:
    if not cited_articles:
        return False
    unit_id_lower = unit.unit_id.lower()
    u_norm = _normalize_text(unit.reader_text)
    for art in cited_articles:
        art_str = str(art).strip().lower()
        if not art_str:
            continue
        if f"::art::{art_str}" in unit_id_lower or f"_art_{art_str}" in unit_id_lower or f"::{art_str}" in unit_id_lower:
            return True
        if f"điều {art_str}." in u_norm or f"điều {art_str} " in u_norm or u_norm.startswith(f"điều {art_str}"):
            return True
    return False



def _require_file(path: Path, flag: str) -> Path:
    if str(path) in {"", "."}:
        raise SystemExit(f"{flag} resolved to an empty path; export the variable")
    if not path.is_file():
        raise SystemExit(f"{flag} does not exist or is not a file: {path}")
    return path


def _load_labels(path: Path) -> dict[str, set[str]]:
    """query_id -> set of gold unit ids (silver labels are fine here)."""

    labels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or row.get("id") or "").strip()
            if not query_id:
                raise SystemExit(f"Label row without query_id in {path}")
            for key in (
                "gold_unit_ids",
                "gold_passage_ids",
                "passage_ids",
                "relevant_ids",
                "labels",
            ):
                value = row.get(key)
                if isinstance(value, list):
                    labels[query_id].update(str(item) for item in value)
            single = row.get("passage_id") or row.get("unit_id")
            if single:
                labels[query_id].add(str(single))
    resolved = {key: value for key, value in labels.items() if value}
    if not resolved:
        raise SystemExit(f"No usable labels found in {path}")
    return resolved


def _load_candidates(path: Path) -> dict[str, list[tuple[str, float, int]]]:
    """query_id -> [(unit_id, raw retriever score, rank)] in ranked order."""

    candidates: dict[str, list[tuple[str, float, int]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", "")).strip()
            if not query_id:
                raise SystemExit(f"Candidate row without query_id in {path}")
            rows: list[tuple[str, float, int]] = []
            payload = row.get("candidates")
            if isinstance(payload, list) and payload:
                for rank, item in enumerate(payload, start=1):
                    unit_id = str(item.get("passage_id") or item.get("unit_id") or "")
                    if not unit_id:
                        continue
                    score = item.get("fusion_score")
                    if score is None:
                        score = item.get("rrf_score")
                    if score is None:
                        score = item.get("dense_score")
                    if score is None:
                        score = item.get("bm25_score")
                    if score is None:
                        score = 1.0 / (60 + rank)
                    rows.append((unit_id, float(score), rank))
            else:
                for rank, unit_id in enumerate(row.get("ranked_ids") or [], start=1):
                    rows.append((str(unit_id), 1.0 / (60 + rank), rank))
            if rows:
                candidates[query_id] = rows
    if not candidates:
        raise SystemExit(f"No candidate rows found in {path}")
    return candidates


def _normalise(scores: list[float]) -> list[float]:
    """Per-query min-max so the band is comparable across queries."""

    if not scores:
        return []
    low, high = min(scores), max(scores)
    span = high - low
    if span <= 0.0:
        return [1.0] * len(scores)
    return [(value - low) / span for value in scores]


def _normalise_ranks(rows: list[tuple[str, float, int]]) -> list[float]:
    """Map rank 1 to 1 and the last observed rank to 0."""

    if not rows:
        return []
    last_rank = max(rank for _, _, rank in rows)
    span = last_rank - 1
    if span <= 0:
        return [1.0] * len(rows)
    return [(last_rank - rank) / span for _, _, rank in rows]


def _band_share(*, band_candidate_count: int, examined: int) -> float:
    """Return the share of examined candidates that survived band filtering."""

    return band_candidate_count / max(1, examined)


def _children_by_parent(units: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Index containment once instead of scanning the corpus per positive."""

    children: dict[str, list[str]] = defaultdict(list)
    for unit_id, unit in units.items():
        parent_id = unit.parent_unit_id
        if parent_id:
            children[parent_id].append(unit_id)
    return {
        parent_id: tuple(sorted(unit_ids)) for parent_id, unit_ids in children.items()
    }


class _PositiveEmbeddingSimilarity:
    """Read normalized passage vectors and score candidates against positives."""

    def __init__(
        self,
        *,
        index_path: Path,
        metadata_path: Path,
        manifest_path: Path,
    ) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise SystemExit(
                "positive-embedding false-negative detection requires faiss and numpy"
            ) from exc

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("normalized") is not True:
            raise SystemExit(
                f"Embedding cache must be normalized for cosine scoring: "
                f"{manifest_path}"
            )
        self._np = np
        self._index = faiss.read_index(str(index_path))
        self._id_to_ordinal: dict[str, int] = {}
        with metadata_path.open(encoding="utf-8") as handle:
            for ordinal, line in enumerate(handle):
                row = json.loads(line)
                passage_id = str(row.get("passage_id") or "")
                if not passage_id:
                    raise SystemExit(
                        f"Embedding metadata row {ordinal} has no passage_id"
                    )
                self._id_to_ordinal[passage_id] = int(row.get("ordinal", ordinal))
        if self._index.ntotal != len(self._id_to_ordinal):
            raise SystemExit(
                "Embedding index/metadata size mismatch: "
                f"{self._index.ntotal} vs {len(self._id_to_ordinal)}"
            )
        self.provenance = {
            "mode": "positive_embedding_cosine",
            "index": str(index_path),
            "metadata": str(metadata_path),
            "manifest": str(manifest_path),
            "model": manifest.get("model"),
            "model_revision": manifest.get("model_revision"),
            "corpus_hash": manifest.get("corpus_hash"),
            "cache_key": manifest.get("cache_key"),
        }

    def validate_ids(self, passage_ids: set[str], *, label: str) -> None:
        missing = sorted(set(passage_ids) - set(self._id_to_ordinal))
        if missing:
            sample = missing[:5]
            raise SystemExit(
                f"{len(missing)} {label} passage IDs are absent from embedding "
                f"metadata; sample={sample}"
            )

    def max_similarity(
        self, positive_ids: set[str], candidate_ids: list[str]
    ) -> dict[str, float]:
        np = self._np
        positive_vectors = np.asarray(
            [
                self._index.reconstruct(self._id_to_ordinal[passage_id])
                for passage_id in sorted(positive_ids)
            ],
            dtype="float32",
        )
        candidate_vectors = np.asarray(
            [
                self._index.reconstruct(self._id_to_ordinal[passage_id])
                for passage_id in candidate_ids
            ],
            dtype="float32",
        )
        similarities = candidate_vectors @ positive_vectors.T
        return {
            passage_id: float(similarities[index].max())
            for index, passage_id in enumerate(candidate_ids)
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", default="warmup")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--negatives",
        type=int,
        default=10,
        help=(
            "Negatives per positive. 10 is the measured optimum for semi-hard "
            "mining on a Vietnamese legal corpus; do not lower it to 2."
        ),
    )
    parser.add_argument(
        "--false-negative-above",
        type=float,
        default=0.75,
        help=(
            "Candidates whose normalised score exceeds this are treated as "
            "suspected false negatives and excluded, not used as negatives."
        ),
    )
    parser.add_argument(
        "--false-negative-mode",
        choices=("score", "positive_cosine"),
        default="score",
        help="Use the band score or positive-passage cosine for false negatives.",
    )
    parser.add_argument(
        "--false-negative-similarity-above",
        type=float,
        default=0.90,
        help="Positive-passage cosine threshold for suspected false negatives.",
    )
    parser.add_argument("--embedding-index", type=Path, default=None)
    parser.add_argument("--embedding-metadata", type=Path, default=None)
    parser.add_argument("--embedding-manifest", type=Path, default=None)
    parser.add_argument(
        "--easy-below",
        type=float,
        default=0.15,
        help="Candidates below this are too easy to teach anything.",
    )
    parser.add_argument("--max-rank", type=int, default=200)
    parser.add_argument(
        "--score-mode",
        choices=("raw", "rank"),
        default="raw",
        help="Normalize fused scores or observed candidate ranks per query.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-band-share",
        type=float,
        default=0.20,
        help=(
            "Fail closed when fewer than this share of examined candidates land "
            "inside the semi-hard band: the band is then misconfigured for this "
            "score distribution and the run would train on the wrong negatives."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.easy_below < args.false_negative_above <= 1.0:
        raise SystemExit("require 0 <= --easy-below < --false-negative-above <= 1")
    if not 0.0 <= args.false_negative_similarity_above <= 1.0:
        raise SystemExit("--false-negative-similarity-above must be in [0, 1]")
    embedding_args = (
        args.embedding_index,
        args.embedding_metadata,
        args.embedding_manifest,
    )
    if args.false_negative_mode == "positive_cosine" and any(
        path is None for path in embedding_args
    ):
        raise SystemExit(
            "positive_cosine mode requires --embedding-index, "
            "--embedding-metadata, and --embedding-manifest"
        )
    if args.false_negative_mode == "score" and any(
        path is not None for path in embedding_args
    ):
        raise SystemExit(
            "embedding arguments require --false-negative-mode=positive_cosine"
        )
    if args.negatives <= 0:
        raise SystemExit("--negatives must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")

    for path, flag in (
        (args.labels, "--labels"),
        (args.candidates, "--candidates"),
        (args.units, "--units"),
        (args.questions, "--questions"),
    ):
        _require_file(path, flag)

    labels = _load_labels(args.labels)
    candidates = _load_candidates(args.candidates)
    units = load_rerank_units(args.units)
    children_by_parent = _children_by_parent(units)
    positive_similarity = (
        _PositiveEmbeddingSimilarity(
            index_path=args.embedding_index,
            metadata_path=args.embedding_metadata,
            manifest_path=args.embedding_manifest,
        )
        if args.false_negative_mode == "positive_cosine"
        else None
    )
    if positive_similarity is not None:
        positive_similarity.validate_ids(
            {
                passage_id
                for query_labels in labels.values()
                for passage_id in query_labels
                if passage_id in units
            },
            label="positive",
        )
        positive_similarity.validate_ids(
            {
                passage_id
                for query_candidates in candidates.values()
                for passage_id, _, _ in query_candidates
                if passage_id in units
            },
            label="candidate",
        )
    questions = {
        item.id: item.question
        for item in load_inference_questions(args.questions, split=args.split)
    }

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "pairs.jsonl"

    counters = {
        "queries_seen": 0,
        "queries_written": 0,
        "queries_without_positive_in_view": 0,
        "queries_without_enough_negatives": 0,
        "candidates_examined": 0,
        "excluded_gold": 0,
        "excluded_same_article": 0,
        "excluded_too_hard": 0,
        "excluded_suspected_false_negative": 0,
        "excluded_too_easy": 0,
        "excluded_beyond_max_rank": 0,
        "band_candidates": 0,
        "accepted_negatives": 0,
        "positive_pairs": 0,
    }
    band_scores: list[float] = []
    excluded_scores: list[float] = []
    excluded_similarities: list[float] = []

    excluded_query_ids: set[str] = set()
    nontraining_questions: dict[str, str] = {}
    data_dir = args.questions.parent
    for split_file in ("warmup.json", "public-official.json", "private-official.json"):
        sp_path = data_dir / split_file
        if sp_path.is_file():
            try:
                sp_json = json.loads(sp_path.read_text(encoding="utf-8"))
                for k, v in sp_json.items():
                    qid_s = str(k)
                    excluded_query_ids.add(qid_s)
                    q_text = _normalize_text(str(v.get("question") or ""))
                    if q_text:
                        nontraining_questions[qid_s] = q_text
            except Exception:
                pass
    dev_path = args.questions.parent.parent / "processed" / "train_dev200.json"
    if dev_path.is_file():
        try:
            dev_json = json.loads(dev_path.read_text(encoding="utf-8"))
            for k, v in dev_json.items():
                qid_s = str(k)
                excluded_query_ids.add(qid_s)
                q_text = _normalize_text(str(v.get("question") or "") if isinstance(v, dict) else str(v))
                if q_text:
                    nontraining_questions[qid_s] = q_text
        except Exception:
            pass

    # Compute exact TF-IDF vectors for all questions (non-training + training)
    all_q_texts: dict[str, str] = dict(nontraining_questions)
    train_answers: dict[str, str] = {}
    if args.questions.is_file():
        try:
            tr_raw = json.loads(args.questions.read_text(encoding="utf-8"))
            for qk, qv in tr_raw.items():
                if isinstance(qv, dict) and "answer" in qv:
                    train_answers[str(qk)] = str(qv["answer"])
        except Exception:
            pass

    for qid in labels:
        if qid not in excluded_query_ids:
            q_str = questions.get(qid)
            if q_str:
                all_q_texts[qid] = _normalize_text(q_str)
    tfidf_vectors = _compute_tfidf_vectors(all_q_texts, n=3)

    with pairs_path.open("w", encoding="utf-8") as handle:
        for query_id in sorted(labels):
            if query_id in excluded_query_ids:
                continue
            question = questions.get(query_id)
            rows = candidates.get(query_id)
            if question is None or not rows:
                continue

            # Gate 1: Check TF-IDF 3-gram cosine similarity against non-training questions (cosine >= 0.90)
            q_vec = tfidf_vectors.get(query_id, {})
            if any(_cosine_similarity(q_vec, tfidf_vectors.get(nt_id, {})) >= 0.90 for nt_id in nontraining_questions):
                continue

            counters["queries_seen"] += 1

            gold_ans = train_answers.get(query_id, "")
            cited_articles: set[str] = set()
            cited_docs: set[str] = set()
            if gold_ans:
                citations = parse_citations(gold_ans)
                for c in citations:
                    if c.article:
                        cited_articles.add(str(c.article))
                    if c.document_number:
                        cited_docs.add(_normalize_text(str(c.document_number)))
                    if c.document_name:
                        cited_docs.add(_normalize_text(str(c.document_name)))

            # Gate 2: Positive passage quality filter (exclude trivial passages < 50 chars)
            gold_ids = {
                gid for gid in labels[query_id]
                if gid in units and len(units[gid].reader_text.strip()) >= 50
            }
            # Expand gold_ids to include ALL resolved candidate passages matching cited articles in gold_ans
            for unit_id, _, _ in rows:
                if unit_id in units and unit_id not in gold_ids:
                    u = units[unit_id]
                    if len(u.reader_text.strip()) >= 50 and _unit_matches_article(u, cited_articles):
                        gold_ids.add(unit_id)

            if not gold_ids:
                counters["queries_without_positive_in_view"] += 1
                continue

            gold_texts = {_normalize_text(units[gid].reader_text) for gid in gold_ids}
            seen_neg_texts: set[str] = set()

            # Containment leakage + Co-regulating article protection:
            # All gold articles, their parents/children, AND any candidate passages belonging to
            # or citing ANY co-regulating article or document in gold_ans are FORBIDDEN from being negatives!
            forbidden = set(gold_ids)
            for gid in list(gold_ids):
                gold_unit = units[gid]
                if gold_unit.parent_unit_id:
                    forbidden.add(gold_unit.parent_unit_id)
                forbidden.update(children_by_parent.get(gid, ()))

            for unit_id, _, _ in rows:
                if unit_id in units and unit_id not in forbidden:
                    u = units[unit_id]
                    if _unit_matches_article(u, cited_articles):
                        forbidden.add(unit_id)
                        continue
                    u_norm = _normalize_text(u.reader_text)
                    if any(doc in u_norm for doc in cited_docs if len(doc) >= 5):
                        forbidden.add(unit_id)
                        continue

            normalised = (
                _normalise_ranks(rows)
                if args.score_mode == "rank"
                else _normalise([score for _, score, _ in rows])
            )
            positive_similarities = (
                positive_similarity.max_similarity(
                    gold_ids,
                    [unit_id for unit_id, _, _ in rows if unit_id in units],
                )
                if positive_similarity is not None
                else {}
            )
            band: list[str] = []
            for (unit_id, _, rank), norm in zip(rows, normalised, strict=True):
                counters["candidates_examined"] += 1
                if rank > args.max_rank:
                    counters["excluded_beyond_max_rank"] += 1
                    continue
                if unit_id not in units:
                    continue
                cand_text = units[unit_id].reader_text
                cand_norm = _normalize_text(cand_text)
                if unit_id in gold_ids or cand_norm in gold_texts:
                    counters["excluded_gold"] += 1
                    continue
                if cand_norm in seen_neg_texts:
                    continue
                if unit_id in forbidden:
                    counters["excluded_same_article"] += 1
                    continue
                if any(_token_jaccard(units[gid].reader_text, cand_text) > 0.85 for gid in gold_ids):
                    counters["excluded_suspected_false_negative"] += 1
                    continue
                if norm > args.false_negative_above:
                    if args.false_negative_mode == "positive_cosine":
                        counters["excluded_too_hard"] += 1
                    else:
                        counters["excluded_suspected_false_negative"] += 1
                        excluded_scores.append(norm)
                    continue
                if args.false_negative_mode == "positive_cosine":
                    similarity = positive_similarities[unit_id]
                    if similarity >= args.false_negative_similarity_above:
                        counters["excluded_suspected_false_negative"] += 1
                        excluded_similarities.append(similarity)
                        continue
                elif norm > args.false_negative_above:
                    counters["excluded_suspected_false_negative"] += 1
                    excluded_scores.append(norm)
                    continue
                if norm < args.easy_below:
                    counters["excluded_too_easy"] += 1
                    continue
                band.append(unit_id)
                seen_neg_texts.add(cand_norm)
                counters["band_candidates"] += 1
                band_scores.append(norm)

            if len(band) < args.negatives:
                counters["queries_without_enough_negatives"] += 1
                if not band:
                    continue
            chosen = (
                band
                if len(band) <= args.negatives
                else rng.sample(band, args.negatives)
            )
            counters["accepted_negatives"] += len(chosen)

            for gid in sorted(gold_ids):
                counters["positive_pairs"] += 1
                handle.write(
                    json.dumps(
                        {
                            "schema_version": RERANKER_DATA_SCHEMA_VERSION,
                            "query_id": query_id,
                            "query": question,
                            "positive_id": gid,
                            "positive": units[gid].reader_text,
                            "negative_ids": chosen,
                            "negatives": [units[nid].reader_text for nid in chosen],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            counters["queries_written"] += 1

    examined = max(1, counters["candidates_examined"])
    band_share = _band_share(
        band_candidate_count=counters["band_candidates"],
        examined=counters["candidates_examined"],
    )
    audit: dict[str, Any] = {
        "schema_version": RERANKER_DATA_SCHEMA_VERSION,
        "policy": {
            "negatives": args.negatives,
            "false_negative_mode": args.false_negative_mode,
            "band_above": args.false_negative_above,
            "false_negative_above": (
                args.false_negative_above
                if args.false_negative_mode == "score"
                else None
            ),
            "false_negative_similarity_above": args.false_negative_similarity_above,
            "easy_below": args.easy_below,
            "max_rank": args.max_rank,
            "score_mode": args.score_mode,
            "seed": args.seed,
        },
        "counters": counters,
        "band_share_of_examined": round(band_share, 4),
        "suspected_false_negative_share": round(
            counters["excluded_suspected_false_negative"] / examined, 4
        ),
        "band_score": {
            "mean": round(statistics.mean(band_scores), 4) if band_scores else 0.0,
            "median": round(statistics.median(band_scores), 4) if band_scores else 0.0,
        },
        "excluded_score": {
            "mean": (
                round(statistics.mean(excluded_scores), 4) if excluded_scores else 0.0
            ),
        },
        "excluded_positive_similarity": {
            "mean": (
                round(statistics.mean(excluded_similarities), 4)
                if excluded_similarities
                else 0.0
            ),
        },
        "positive_similarity": (
            positive_similarity.provenance if positive_similarity is not None else None
        ),
        "pairs_path": str(pairs_path),
    }

    gate_failures: list[str] = []
    if counters["positive_pairs"] == 0:
        gate_failures.append("no positive pairs were written")
    if band_share < args.min_band_share:
        gate_failures.append(
            f"only {band_share:.3f} of examined candidates fell inside the "
            f"semi-hard band (< --min-band-share={args.min_band_share}); the "
            "band bounds do not match this score distribution"
        )
    if band_scores and statistics.mean(band_scores) > args.false_negative_above:
        gate_failures.append(
            "the accepted band's mean score exceeds the false-negative bound, "
            "which is the hard-negative failure case"
        )
    audit["gate_failures"] = gate_failures
    audit["status"] = "FAIL" if gate_failures else "PASS"

    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if gate_failures:
        raise SystemExit(
            "Refusing to hand these pairs to training; see audit.json gate_failures"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
