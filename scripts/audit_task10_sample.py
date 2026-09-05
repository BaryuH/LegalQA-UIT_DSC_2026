#!/usr/bin/env python3
"""Assisted audit for TASK10 hard-negative audit samples."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_passage_id(passage_id: str) -> dict[str, str | None]:
    parts = passage_id.split("::")
    out: dict[str, str | None] = {
        "document_id": None,
        "article": None,
        "clause": None,
        "level": "unknown",
    }
    if not parts:
        return out
    out["document_id"] = parts[0]
    index = 1
    while index + 1 < len(parts):
        key, value = parts[index], parts[index + 1]
        if key == "art":
            out["article"] = value
        elif key == "cl":
            out["clause"] = value
        index += 2
    if out["clause"] is not None:
        out["level"] = "clause"
    elif out["article"] is not None:
        out["level"] = "article"
    else:
        out["level"] = "document"
    return out


def has_explicit_citation(query: str) -> bool:
    return bool(re.search(r"\b(điều|khoản|điểm)\s*\d+", query, flags=re.IGNORECASE))


def audit_row(row: dict[str, object]) -> dict[str, object]:
    positive_id = str(row["positive_passage_id"])
    negative_id = str(row["negative_passage_id"])
    category = str(row["negative_category"])
    flags = tuple(row.get("false_negative_flags") or [])
    query = str(row["query"])

    positive = parse_passage_id(positive_id)
    negative = parse_passage_id(negative_id)
    note_parts: list[str] = []

    same_document = positive["document_id"] == negative["document_id"]
    same_article = (
        same_document
        and positive["article"] is not None
        and negative["article"] is not None
        and positive["article"] == negative["article"]
    )
    explicit_citation = has_explicit_citation(query)

    manual_false_negative = False
    manual_valid = True

    if (
        positive["level"] == "clause"
        and negative["level"] == "article"
        and same_article
    ):
        manual_false_negative = True
        manual_valid = False
        note_parts.append(
            "Negative là passage cấp Điều; positive là Khoản cụ thể nên "
            "passage Điều thường chứa nội dung positive."
        )
    elif (
        category == "B"
        and same_article
        and positive["level"] == "clause"
        and negative["level"] == "clause"
    ):
        if positive["clause"] != negative["clause"]:
            note_parts.append(
                "Cùng Điều nhưng khác Khoản; hard negative hợp lệ cho câu hỏi "
                "trỏ đúng một khoản."
            )
        else:
            manual_false_negative = True
            manual_valid = False
            note_parts.append("Cùng Điều và cùng Khoản; không phải negative.")
    elif category == "A" and same_document and not same_article:
        note_parts.append("Khác Điều trong cùng văn bản; negative cấu trúc hợp lệ.")
    elif category == "C" and not same_document:
        note_parts.append("Khác văn bản; chỉ trùng từ ngữ chung, negative hợp lệ.")
    elif (
        category == "B"
        and same_article
        and explicit_citation
        and "citation_overlap" in flags
    ):
        note_parts.append(
            "Có citation overlap nhưng negative là khoản khác trong cùng Điều; "
            "vẫn là hard negative hợp lệ."
        )
    elif category == "A" and same_document and "high_lexical_agreement" in flags:
        note_parts.append(
            "Lexical overlap cao do cùng domain/văn bản nhưng khác Điều; vẫn hợp lệ."
        )
    else:
        note_parts.append("Không thấy dấu hiệu false negative rõ từ cấu trúc passage.")

    if row.get("potential_false_negative") and not manual_false_negative:
        note_parts.append(
            "Machine flag quá nhạy; assisted audit đánh giá vẫn là negative hợp lệ."
        )
    if not row.get("potential_false_negative") and manual_false_negative:
        note_parts.append(
            "Machine không flag nhưng assisted audit nghi ngờ false negative."
        )

    audited = dict(row)
    audited["manual_negative_is_valid"] = manual_valid
    audited["manual_false_negative"] = manual_false_negative
    audited["reviewer_note"] = " ".join(note_parts)
    audited["assisted_audit_basis"] = (
        "passage_id_structure_and_query_citation_heuristic"
    )
    return audited


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audited = [audit_row(row) for row in rows]

    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "fn": 0, "valid": 0}
    )
    for row in audited:
        category = str(row["negative_category"])
        by_category[category]["total"] += 1
        by_category[category]["fn"] += int(row["manual_false_negative"])
        by_category[category]["valid"] += int(row["manual_negative_is_valid"])

    fn_rate = sum(1 for row in audited if row["manual_false_negative"]) / len(audited)
    valid_rate = sum(1 for row in audited if row["manual_negative_is_valid"]) / len(
        audited
    )
    machine_fn_rate = sum(
        1 for row in audited if row.get("potential_false_negative")
    ) / len(audited)
    summary = {
        "rows": len(audited),
        "assisted_audit_fn_rate": round(fn_rate, 4),
        "assisted_audit_valid_rate": round(valid_rate, 4),
        "machine_potential_fn_rate": round(machine_fn_rate, 4),
        "target_fn_rate": 0.03,
        "passes_target_assisted_audit": fn_rate <= 0.03,
        "by_category": dict(by_category),
        "notes": [
            "Assisted audit without passage text; parent-article vs "
            "clause-level rule applied.",
            "Category B same-article different-clause treated as valid hard negatives.",
            "Recommend spot-check 20 rows with passage text before promotion.",
        ],
    }

    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in audited) + "\n",
        encoding="utf-8",
    )
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
