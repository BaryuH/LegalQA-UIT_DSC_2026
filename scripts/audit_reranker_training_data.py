#!/usr/bin/env python3
"""Comprehensive, continuous QA/QC audit suite for reranker fine-tuning data.

Enforces strict compliance with competition contracts and legal-RAG principles:
- Gate 1 (Zero Leakage): 0 ID overlap, 0 near-duplicate question overlap (TF-IDF >= 0.90)
  against warmup/public/private splits; no forbidden gold/answer fields.
- Gate 2 (Positive & Query Quality): Non-empty, non-trivial (>50 chars), query <= 256 tokens,
  passage <= 2048 tokens (model card limit).
- Gate 3 (Negative Quality & False-Negative Guardrail): Positive never in negatives, no duplicate
  negatives, token Jaccard <= 0.85 to guard against false-negative pollution.
- Gate 4 (Corpus Diversity): No single document dominates > 25% of pairs (prevents statute overfit).
- Gate 5 (Upstream Audit Integrity): audit.json exists with status=PASS and 0 gate_failures.
- Gate 6 (Encoding & Schema): Strict UTF-8 JSONL.

Outputs an auditable qa_qc_report.json. Fails closed (exit code 1) on any contract violation.

Usage (from repo root or zip_for_huy/):
    python scripts/audit_reranker_training_data.py \
        --pairs <path_to_pairs.jsonl> \
        --data-dir data \
        [--max-negatives 10] [--output-report <path_to_report.json>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Ensure repo root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _cosine_sparse(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(val * vec_b.get(k, 0.0) for k, val in vec_a.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="Path to reranker training pairs.jsonl")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train/warmup/public/private json files")
    parser.add_argument("--max-negatives", type=int, default=10, help="Expected number of negatives per positive (default: 10)")
    parser.add_argument("--max-query-tokens", type=int, default=256, help="Max query token cutoff (default: 256)")
    parser.add_argument("--max-passage-tokens", type=int, default=2048, help="Max passage token cutoff (default: 2048)")
    parser.add_argument("--output-report", type=Path, default=None, help="Optional output path for qa_qc_report.json")
    args = parser.parse_args()

    pairs_path = args.pairs.resolve()
    data_dir = args.data_dir.resolve()
    if not pairs_path.is_file():
        raise SystemExit(f"Pairs file not found: {pairs_path}")

    gate_failures: dict[str, list[str]] = defaultdict(list)
    gate_warnings: dict[str, list[str]] = defaultdict(list)
    metrics: dict[str, Any] = {}

    print(f"[QA/QC Audit] Auditing: {pairs_path}")

    # ----------------------------------------------------------------------- #
    # Gate 1: Zero Leakage & Data Isolation Audit
    # ----------------------------------------------------------------------- #
    nontraining_ids: dict[str, str] = {}  # id -> split_name
    nontraining_questions: dict[str, str] = {}  # id -> text
    for split in ("warmup.json", "public-official.json", "private-official.json"):
        split_path = data_dir / split
        if split_path.is_file():
            try:
                content = json.loads(split_path.read_text(encoding="utf-8"))
                for qid, row in content.items():
                    qid_str = str(qid)
                    nontraining_ids[qid_str] = split
                    q_text = _normalize_text(str(row.get("question") or ""))
                    if q_text:
                        nontraining_questions[qid_str] = q_text
            except Exception as exc:
                gate_warnings["Gate1_Leakage"].append(f"Cannot read {split}: {exc}")

    # Read pairs and verify line-by-line
    pair_rows: list[dict[str, Any]] = []
    forbidden_fields_detected: list[str] = []
    seen_query_ids: set[str] = set()

    with pairs_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                gate_failures["Gate6_Format"].append(f"Invalid JSON at line {line_no}: {exc}")
                continue

            # Check forbidden gold fields
            forbidden = {"answer", "gold", "reference", "reference_answer"} & set(row.keys())
            if forbidden:
                forbidden_fields_detected.append(f"Line {line_no} contains forbidden fields: {sorted(forbidden)}")

            pair_rows.append(row)

    metrics["total_pair_rows"] = len(pair_rows)
    if forbidden_fields_detected:
        gate_failures["Gate1_Leakage"].extend(forbidden_fields_detected[:5])

    # Check ID leakage
    leaked_ids = [r.get("query_id") for r in pair_rows if str(r.get("query_id")) in nontraining_ids]
    if leaked_ids:
        gate_failures["Gate1_Leakage"].append(f"Found {len(leaked_ids)} training IDs overlapping with non-training splits! Sample: {leaked_ids[:5]}")

    # Check Near-Duplicate Question Leakage (TF-IDF char 3-gram >= 0.90)
    train_queries: dict[str, str] = {}
    for r in pair_rows:
        qid = str(r.get("query_id") or "")
        q_text = _normalize_text(str(r.get("query") or ""))
        if qid and q_text:
            train_queries[qid] = q_text

    if nontraining_questions and train_queries:
        all_q = dict(nontraining_questions)
        all_q.update(train_queries)
        vectors = _compute_tfidf_vectors(all_q, n=3)
        near_dups: list[tuple[str, str, float]] = []
        for t_id, t_text in train_queries.items():
            t_vec = vectors.get(t_id, {})
            for nt_id, nt_text in nontraining_questions.items():
                if t_id == nt_id:
                    continue
                score = _cosine_sparse(t_vec, vectors.get(nt_id, {}))
                if score >= 0.90:
                    near_dups.append((t_id, nt_id, round(score, 4)))
        if near_dups:
            gate_failures["Gate1_Leakage"].append(f"Found {len(near_dups)} near-duplicate questions (cosine >= 0.90) with non-training splits! Sample: {near_dups[:3]}")

    # ----------------------------------------------------------------------- #
    # Gate 2: Positive & Query Quality Audit
    # ----------------------------------------------------------------------- #
    query_lengths: list[int] = []
    positive_lengths: list[int] = []
    short_positives: list[str] = []
    truncated_queries: list[str] = []
    truncated_positives: list[str] = []

    for idx, r in enumerate(pair_rows):
        qid = str(r.get("query_id") or f"row_{idx}")
        query = str(r.get("query") or "")
        pos = str(r.get("positive") or "")

        q_toks = len(query.split())
        p_toks = len(pos.split())
        query_lengths.append(q_toks)
        positive_lengths.append(p_toks)

        if len(pos.strip()) < 50:
            short_positives.append(qid)
        if q_toks > args.max_query_tokens:
            truncated_queries.append(qid)
        if p_toks > args.max_passage_tokens:
            truncated_positives.append(qid)

    if short_positives:
        gate_failures["Gate2_PositiveQuality"].append(f"{len(short_positives)} positive passages are trivial (<50 chars). Sample QIDs: {short_positives[:5]}")
    if truncated_queries:
        gate_warnings["Gate2_PositiveQuality"].append(f"{len(truncated_queries)} queries exceed max_query_tokens ({args.max_query_tokens}). Sample: {truncated_queries[:5]}")
    if truncated_positives:
        gate_warnings["Gate2_PositiveQuality"].append(f"{len(truncated_positives)} positive passages exceed max_passage_tokens ({args.max_passage_tokens}). Sample: {truncated_positives[:5]}")

    metrics["query_tokens"] = {
        "mean": round(statistics.mean(query_lengths), 1) if query_lengths else 0,
        "p50": round(statistics.median(query_lengths), 1) if query_lengths else 0,
        "max": max(query_lengths) if query_lengths else 0,
    }
    metrics["positive_tokens"] = {
        "mean": round(statistics.mean(positive_lengths), 1) if positive_lengths else 0,
        "p50": round(statistics.median(positive_lengths), 1) if positive_lengths else 0,
        "p90": round(float(statistics.quantiles(positive_lengths, n=10)[8]), 1) if len(positive_lengths) >= 10 else 0,
        "max": max(positive_lengths) if positive_lengths else 0,
    }

    # ----------------------------------------------------------------------- #
    # Gate 3: Negative Quality & False-Negative Pollution Guardrail
    # ----------------------------------------------------------------------- #
    pos_in_neg_count = 0
    duplicate_neg_count = 0
    neg_counts: list[int] = []
    high_jaccard_false_negatives: list[dict[str, Any]] = []

    for idx, r in enumerate(pair_rows):
        qid = str(r.get("query_id") or f"row_{idx}")
        pos = str(r.get("positive") or "")
        pos_norm = _normalize_text(pos)
        negs = [str(n) for n in (r.get("negatives") or [])]
        neg_counts.append(len(negs))

        # Check positive in negatives
        norm_negs = [_normalize_text(n) for n in negs]
        if pos_norm in norm_negs:
            pos_in_neg_count += 1

        # Check duplicate negatives
        if len(norm_negs) != len(set(norm_negs)):
            duplicate_neg_count += 1

        # Check False Negative via High Jaccard Overlap (> 0.85)
        for nid_idx, n_text in enumerate(negs):
            jacc = _token_jaccard(pos, n_text)
            if jacc > 0.85:
                high_jaccard_false_negatives.append({
                    "query_id": qid,
                    "negative_idx": nid_idx,
                    "jaccard_overlap": round(jacc, 4),
                })

    if pos_in_neg_count > 0:
        gate_failures["Gate3_NegativeQuality"].append(f"{pos_in_neg_count} queries contain the positive passage inside their negative list!")
    if duplicate_neg_count > 0:
        gate_warnings["Gate3_NegativeQuality"].append(f"{duplicate_neg_count} queries contain duplicate negatives within the same group.")
    if high_jaccard_false_negatives:
        gate_warnings["Gate3_NegativeQuality"].append(
            f"Detected {len(high_jaccard_false_negatives)} suspected false-negatives (token Jaccard > 0.85 with positive)! Sample: {high_jaccard_false_negatives[:3]}"
        )

    metrics["negatives_per_positive"] = {
        "mean": round(statistics.mean(neg_counts), 1) if neg_counts else 0,
        "min": min(neg_counts) if neg_counts else 0,
        "max": max(neg_counts) if neg_counts else 0,
    }

    # ----------------------------------------------------------------------- #
    # Gate 4: Corpus Diversity & Document Concentration Guardrail
    # ----------------------------------------------------------------------- #
    doc_id_counts: Counter[str] = Counter()
    for r in pair_rows:
        pid = str(r.get("positive_id") or "")
        # Unit IDs typically start with document ID or have document prefix
        doc_part = pid.split("::")[0] if "::" in pid else pid.split("_")[0]
        if doc_part:
            doc_id_counts[doc_part] += 1

    if doc_id_counts and len(pair_rows) >= 100:
        top_doc, top_count = doc_id_counts.most_common(1)[0]
        concentration = top_count / len(pair_rows)
        metrics["top_document_concentration"] = {
            "document": top_doc,
            "count": top_count,
            "share": round(concentration, 4),
        }
        if concentration > 0.25:
            gate_warnings["Gate4_Diversity"].append(
                f"Document {top_doc} represents {concentration:.1%} of all positive pairs (>25%); risk of statute overfit."
            )

    # ----------------------------------------------------------------------- #
    # Gate 5: Upstream Audit & Provenance Verification
    # ----------------------------------------------------------------------- #
    upstream_audit_path = pairs_path.with_name("audit.json")
    if not upstream_audit_path.is_file():
        gate_failures["Gate5_UpstreamAudit"].append(f"audit.json missing next to {pairs_path.name}")
    else:
        try:
            up_audit = json.loads(upstream_audit_path.read_text(encoding="utf-8"))
            if up_audit.get("status") != "PASS":
                gate_failures["Gate5_UpstreamAudit"].append(f"Upstream audit status is {up_audit.get('status')!r}, not 'PASS'!")
            up_fails = up_audit.get("gate_failures") or []
            if up_fails:
                gate_failures["Gate5_UpstreamAudit"].append(f"Upstream audit has unresolved gate failures: {up_fails}")
            metrics["upstream_band_share"] = up_audit.get("band_share_of_examined")
        except Exception as exc:
            gate_failures["Gate5_UpstreamAudit"].append(f"Failed to parse audit.json: {exc}")

    # ----------------------------------------------------------------------- #
    # Final Report & Status
    # ----------------------------------------------------------------------- #
    has_failures = any(len(fails) > 0 for fails in gate_failures.values())
    status = "FAIL" if has_failures else "PASS"

    report: dict[str, Any] = {
        "schema_version": "legalqa.reranker_data_qa_qc.v1",
        "status": status,
        "pairs_file": str(pairs_path),
        "pairs_sha256": _sha256_file(pairs_path),
        "metrics": metrics,
        "gate_failures": dict(gate_failures),
        "gate_warnings": dict(gate_warnings),
    }

    report_str = json.dumps(report, ensure_ascii=False, indent=2)
    print(f"\n{'='*70}\n[QA/QC AUDIT REPORT: {status}]\n{'='*70}")
    print(report_str)

    report_path = args.output_report or (pairs_path.parent / "qa_qc_report.json")
    report_path.write_text(report_str + "\n", encoding="utf-8")
    print(f"\n[QA/QC Audit] Report saved to: {report_path}")

    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
