#!/usr/bin/env python3
"""Fail-closed preflight for Huy embedding/reranker training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from legal_rag.huy_training.common import load_training_pairs, sha256_file


def _expected_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        hashes[relative.strip()] = digest
    return hashes


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _nontraining_questions(root: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    questions: set[str] = set()
    for filename in ("warmup.json", "public-official.json", "private-official.json"):
        path = root / "data" / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw_id, row in payload.items():
            ids.add(str(raw_id))
            question = str(row.get("question") or "")
            if question.strip():
                questions.add(_normalize_question(question))
    return ids, questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--max-negatives", type=int, default=10)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    root = args.package_root.resolve()
    failures: list[str] = []
    hash_file = root / "DATA_SHA256SUMS.txt"
    if not hash_file.is_file():
        failures.append("DATA_SHA256SUMS.txt missing")
    else:
        for relative, expected in _expected_hashes(hash_file).items():
            candidate = root / relative
            if not candidate.is_file():
                failures.append(f"missing data file: {relative}")
            elif sha256_file(candidate) != expected:
                failures.append(f"data hash mismatch: {relative}")

    try:
        pairs = load_training_pairs(
            args.pairs, source_split="train", max_negatives=args.max_negatives
        )
    except (ValueError, json.JSONDecodeError) as exc:
        failures.append(str(exc))
        pairs = ()

    nontraining_ids, nontraining_questions = _nontraining_questions(root)
    overlap_ids = sorted(
        pair.query_id for pair in pairs if pair.query_id in nontraining_ids
    )
    overlap_questions = sorted(
        pair.query_id
        for pair in pairs
        if _normalize_question(pair.query) in nontraining_questions
    )
    if overlap_ids:
        failures.append(f"non-training ID overlap count={len(overlap_ids)}")
    if overlap_questions:
        failures.append(
            f"normalized non-training question overlap count={len(overlap_questions)}"
        )

    if args.pairs.is_file():
        with args.pairs.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                forbidden = {"answer", "gold", "reference"}.intersection(row)
                if forbidden:
                    failures.append(
                        f"forbidden gold-like fields at line {line_number}: "
                        f"{sorted(forbidden)}"
                    )

    cuda_report: dict[str, object] = {"required": args.require_cuda}
    if args.require_cuda:
        try:
            import torch

            cuda_report.update(
                {
                    "available": torch.cuda.is_available(),
                    "bf16_supported": (
                        torch.cuda.is_bf16_supported()
                        if torch.cuda.is_available()
                        else False
                    ),
                    "device": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                    "total_memory": (
                        torch.cuda.get_device_properties(0).total_memory
                        if torch.cuda.is_available()
                        else 0
                    ),
                }
            )
            if not cuda_report["available"]:
                failures.append("CUDA unavailable")
            elif not cuda_report["bf16_supported"]:
                failures.append("CUDA device does not support BF16")
        except ImportError as exc:
            failures.append(f"torch missing: {exc}")

    report = {
        "schema_version": "huy.training_preflight.v1",
        "status": "PASS" if not failures else "FAIL",
        "pair_count": len(pairs),
        "nontraining_id_overlap_count": len(overlap_ids),
        "nontraining_question_overlap_count": len(overlap_questions),
        "pairs_sha256": sha256_file(args.pairs) if args.pairs.is_file() else None,
        "cuda": cuda_report,
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
