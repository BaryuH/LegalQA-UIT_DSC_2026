#!/usr/bin/env python3
"""Audit train-to-warmup near-duplicate questions without exposing answers."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from legal_rag.finetuned_reader.split_remediation import normalize_question_text
from legal_rag.questions import load_inference_questions
from legal_rag.splits import SplitName

SCHEMA_VERSION = "sedar-question-overlap-audit-v1"


def _load_question_texts(path: Path, split: str) -> dict[str, str]:
    return {
        item.id: normalize_question_text(item.question)
        for item in load_inference_questions(path, split=cast(SplitName, split))
    }


def _char_ngrams(text: str, ngram_size: int) -> Counter[str]:
    padded = f"^{text}$"
    return Counter(
        padded[index : index + ngram_size]
        for index in range(max(0, len(padded) - ngram_size + 1))
    )


def _tfidf_vectors(
    questions: Mapping[str, str], idf: Mapping[str, float], ngram_size: int
) -> dict[str, dict[str, float]]:
    vectors: dict[str, dict[str, float]] = {}
    for question_id, text in questions.items():
        counts = _char_ngrams(text, ngram_size)
        vector = {
            gram: (1.0 + math.log(count)) * idf[gram] for gram, count in counts.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        vectors[question_id] = (
            {gram: weight / norm for gram, weight in vector.items()} if norm else {}
        )
    return vectors


def audit_questions(
    train_questions: Mapping[str, str],
    warmup_questions: Mapping[str, str],
    *,
    threshold: float = 0.90,
    ngram_size: int = 3,
    top_samples: int = 20,
) -> dict[str, Any]:
    """Return content-free exact and near-duplicate overlap diagnostics."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if ngram_size < 1:
        raise ValueError("ngram_size must be positive")
    if top_samples < 0:
        raise ValueError("top_samples must be non-negative")

    all_questions = [*train_questions.values(), *warmup_questions.values()]
    document_frequency: Counter[str] = Counter()
    for text in all_questions:
        document_frequency.update(_char_ngrams(text, ngram_size))
    document_count = max(1, len(all_questions))
    idf = {
        gram: math.log((1.0 + document_count) / (1.0 + frequency)) + 1.0
        for gram, frequency in document_frequency.items()
    }

    train_vectors = _tfidf_vectors(train_questions, idf, ngram_size)
    warmup_vectors = _tfidf_vectors(warmup_questions, idf, ngram_size)
    inverted: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for warmup_id, vector in warmup_vectors.items():
        for gram, weight in vector.items():
            inverted[gram].append((warmup_id, weight))

    near_pairs: list[tuple[float, str, str]] = []
    for train_id, vector in train_vectors.items():
        scores: defaultdict[str, float] = defaultdict(float)
        for gram, train_weight in vector.items():
            for warmup_id, warmup_weight in inverted.get(gram, ()):
                scores[warmup_id] += train_weight * warmup_weight
        near_pairs.extend(
            (score, train_id, warmup_id)
            for warmup_id, score in scores.items()
            if score >= threshold
        )

    near_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    near_train_ids = sorted({item[1] for item in near_pairs})
    exact_pairs = sum(
        1
        for train_text in train_questions.values()
        for warmup_text in warmup_questions.values()
        if train_text == warmup_text
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "metric": "character_tfidf_cosine",
        "ngram_size": ngram_size,
        "threshold": threshold,
        "train_count": len(train_questions),
        "warmup_count": len(warmup_questions),
        "exact_normalized_pair_count": exact_pairs,
        "near_duplicate_pair_count": len(near_pairs),
        "train_ids_with_near_duplicate": len(near_train_ids),
        "near_duplicate_train_ids": near_train_ids,
        "top_matches": [
            {"cosine": round(score, 6), "train_id": train_id, "warmup_id": warmup_id}
            for score, train_id, warmup_id in near_pairs[:top_samples]
        ],
        "content_policy": "ids_and_scores_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--top-samples", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite artifact: {args.output}")
    if not args.train.is_file() or not args.warmup.is_file():
        raise SystemExit("--train and --warmup must be files")

    report = audit_questions(
        _load_question_texts(args.train, "train"),
        _load_question_texts(args.warmup, "warmup"),
        threshold=args.threshold,
        ngram_size=args.ngram_size,
        top_samples=args.top_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
