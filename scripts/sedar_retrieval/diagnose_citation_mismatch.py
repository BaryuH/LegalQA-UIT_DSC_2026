"""Split wrong-document answers into a retrieval fault and a reader fault.

Roughly half of the sampled failures reproduce the right article text under
the wrong document. Two very different causes produce that, and they need
opposite fixes, so guessing between them is not acceptable:

  group A - the pack DID contain a passage from the document the reference
            cites, and the answer still named another one. The evidence was
            there and the reader did not use its identity. Fix lives in the
            evidence header and the prompt.

  group B - the pack contained no passage from that document. The reader was
            faithful to what it was given; the wrong document arrived from
            retrieval, most likely a near-duplicate twin. Fix lives in the
            corpus and the candidate selection.

Group membership is decided by silver labels and packed passage IDs, never by
parsing a document number out of the answer, so it does not inherit the
citation parser's blind spots.

Evaluation-only. Prints counts and case IDs; never writes answer text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from legal_rag.evaluation.error_classifier import (  # noqa: E402
    _document_identity_keys,
)
from legal_rag.sedar_retrieval.eval.ensemble_metrics import (  # noqa: E402
    load_relevance_labels,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import (  # noqa: E402
    load_passages_jsonl,
)


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _ids(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(x for x in value.replace(";", " ").split() if x)
    return tuple(str(x) for x in value)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--references", required=True, type=Path)
    ap.add_argument("--retrieval", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--passages", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="JSON: {group: [ids]}")
    args = ap.parse_args(argv)

    passages = {p.passage_id: p for p in load_passages_jsonl(str(args.passages))}
    labels = load_relevance_labels(args.labels)

    def _load_answers(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        text = path.read_text(encoding="utf-8")
        if text.lstrip().startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(x) for x in text.splitlines() if x.strip()]
        for row in rows:
            out[str(row["id"])] = str(row.get("answer") or row.get("prediction") or "")
        return out

    predictions = _load_answers(args.predictions)
    references = _load_answers(args.references)
    packed = {
        str(row["id"]): _ids(row.get("packed_chunk_ids"))
        for row in _jsonl(args.retrieval)
    }

    groups: dict[str, list[str]] = {
        "A_pack_co_van_ban_dung__reader_van_dan_sai": [],
        "B_pack_khong_co_van_ban_dung": [],
        "C_khong_co_nhan_bac": [],
    }
    mismatch = 0
    for case_id, prediction in sorted(predictions.items()):
        reference = references.get(case_id, "")
        pred_docs = _document_identity_keys(prediction)
        ref_docs = _document_identity_keys(reference)
        if not pred_docs or not ref_docs or (pred_docs & ref_docs):
            continue
        mismatch += 1

        relevant, provenance = labels.get(case_id, (frozenset(), None))
        if provenance != "silver" or not relevant:
            groups["C_khong_co_nhan_bac"].append(case_id)
            continue
        gold_documents = {
            passages[pid].document_id for pid in relevant if pid in passages
        }
        packed_documents = {
            passages[pid].document_id
            for pid in packed.get(case_id, ())
            if pid in passages
        }
        key = (
            "A_pack_co_van_ban_dung__reader_van_dan_sai"
            if gold_documents & packed_documents
            else "B_pack_khong_co_van_ban_dung"
        )
        groups[key].append(case_id)

    total = len(predictions)
    print(f"tong case            : {total}")
    print(f"dan sai van ban      : {mismatch} ({mismatch/total:.1%})\n")
    for name, ids in groups.items():
        share = len(ids) / mismatch if mismatch else 0.0
        print(f"{name:44s} {len(ids):4d}  ({share:.1%} cua so dan sai)")
    decidable = (
        len(groups["A_pack_co_van_ban_dung__reader_van_dan_sai"])
        + len(groups["B_pack_khong_co_van_ban_dung"])
    )
    if decidable:
        a = len(groups["A_pack_co_van_ban_dung__reader_van_dan_sai"])
        print(f"\ntrong so case phan xu duoc ({decidable}): "
              f"loi READER {a/decidable:.1%} | loi RETRIEVAL {1-a/decidable:.1%}")
    print("\nvai ID moi nhom de soi tay:")
    for name, ids in groups.items():
        print(f"  {name}: {', '.join(ids[:8])}")
    if args.out:
        args.out.write_text(
            json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nda ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
