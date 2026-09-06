"""Cut degenerate repetition out of finished answers, after generation.

Suppressing repetition during decoding was measured and rejected: at
--no-repeat-ngram-size 6 the model lost 0.127 METEOR and started inserting
spaces inside words and citing other documents, because Vietnamese legal prose
repeats n-grams obligatorily. Trimming afterwards is a different operation. It
cannot corrupt wording or citations, since it only ever deletes a suffix.

The pathology is a RUN, not a repeat. A legal answer legitimately quotes a
list and then restates it, so a line appearing twice is normal. A line
appearing a third time is not, and that is where this cuts.

The suffix is dropped from that point, then rolled back to the last sentence
boundary so the answer does not end mid-clause. Nothing is rewritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_LEADING_MARKER = re.compile(
    r"^\s*(?:\d+\s*[.)]|[a-zA-Zđ]\s*[.)]|[-+*•]|[a-z]\.\d+(?:\.\d+)*\s*[.)]?)\s*"
)
_SENTENCE_END = re.compile(r"[.;:!?]\s*$")


def _normalise(line: str) -> str:
    """Strip list markers and spacing so a renumbered repeat still matches."""

    stripped = _LEADING_MARKER.sub("", line)
    return " ".join(stripped.split()).casefold()


def trim(answer: str, *, allowed_repeats: int = 2, min_chars: int = 24) -> str:
    """Return the answer with its first degenerate run and everything after removed."""

    lines = answer.splitlines()
    seen: Counter[str] = Counter()
    cut = None
    for index, line in enumerate(lines):
        key = _normalise(line)
        if len(key) < min_chars:
            continue  # short connectives repeat harmlessly
        seen[key] += 1
        if seen[key] > allowed_repeats:
            cut = index
            break
    if cut is None:
        return answer

    kept = lines[:cut]
    while kept and not _SENTENCE_END.search(kept[-1]):
        kept.pop()
    if not kept:
        kept = lines[:cut]
    return "\n".join(kept).rstrip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--allowed-repeats", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.output.exists() and not args.force:
        raise SystemExit(f"Output already exists: {args.output}")
    if args.allowed_repeats < 1:
        raise SystemExit("--allowed-repeats must be at least 1")

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = 0
    removed_chars: list[int] = []
    for row in rows:
        field = "answer" if "answer" in row else "prediction"
        original = str(row.get(field) or "")
        if not original:
            continue
        trimmed = trim(
            original,
            allowed_repeats=args.allowed_repeats,
            min_chars=args.min_chars,
        )
        if trimmed != original:
            changed += 1
            removed_chars.append(len(original) - len(trimmed))
            row[field] = trimmed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    removed_chars.sort()
    median = removed_chars[len(removed_chars) // 2] if removed_chars else 0
    print(json.dumps({
        "cases": len(rows),
        "trimmed": changed,
        "trimmed_share": round(changed / len(rows), 4) if rows else 0.0,
        "removed_chars_median": median,
        "removed_chars_max": removed_chars[-1] if removed_chars else 0,
        "output": args.output.as_posix(),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
