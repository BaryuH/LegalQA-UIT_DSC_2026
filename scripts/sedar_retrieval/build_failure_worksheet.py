"""Build a blind review worksheet for the worst-scoring cases.

The bottom quartile holds 42% of the total METEOR deficit at a mean of
0.24, and no taxonomy code explains it: its worst cases are spread across
RETRIEVAL_MISS, UNDER_SPECIFIED, WRONG_ARTICLE_CITATION and
RERANKING_REGRESSION alike. Whatever is actually destroying those answers is
not yet named, so the only way forward is to read them.

Two properties matter, both learned the hard way.

Blind: the classifier's error_type, reason_code and confidence are excluded,
so a reviewer forms a judgement before seeing the machine's.

With evidence TEXT, not just passage IDs. A previous review inferred "gold was
never retrieved" from ID prefixes alone and got five of sixty cases wrong,
because IDs cannot show what a passage says. Every packed passage is rendered
in full here.

The output contains reference answers. It is an evaluation-only artifact:
keep it out of git and out of memory-bank, and record only IDs and reason
codes downstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl


def _answers(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    rows = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(x) for x in text.splitlines() if x.strip()]
    )
    return {
        str(r["id"]): str(r.get("answer") or r.get("prediction") or "")
        for r in rows
    }


def _ids(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(x for x in value.replace(";", " ").split() if x)
    return tuple(str(x) for x in value)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--references", required=True, type=Path)
    ap.add_argument("--retrieval", required=True, type=Path)
    ap.add_argument("--passages", required=True, type=Path)
    ap.add_argument("--labels", type=Path, default=None,
                    help="Silver labels; marks which packed passage is gold.")
    ap.add_argument("--metric", default="meteor")
    ap.add_argument("--worst", type=int, default=20)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--passage-chars", type=int, default=1500)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    per_case = {
        str(r["id"]): r
        for r in payload["per_case"]
        if r.get(args.metric) is not None
    }
    if args.ids:
        selected = [i for i in args.ids if i in per_case]
    else:
        selected = sorted(per_case, key=lambda c: per_case[c][args.metric])
        selected = selected[: args.worst]

    passages = {p.passage_id: p for p in load_passages_jsonl(str(args.passages))}
    predictions = _answers(args.predictions)
    references = _answers(args.references)
    retrieval = {
        str(row["id"]): row
        for row in (
            json.loads(line)
            for line in args.retrieval.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    gold: dict[str, set[str]] = {}
    if args.labels:
        from legal_rag.sedar_retrieval.eval.ensemble_metrics import (
            load_relevance_labels,
        )
        for case_id, (relevant, provenance) in load_relevance_labels(
            args.labels
        ).items():
            if provenance == "silver":
                gold[str(case_id)] = set(relevant)

    rows = []
    for case_id in selected:
        trace = retrieval.get(case_id, {})
        packed = _ids(trace.get("packed_chunk_ids"))
        gold_ids = gold.get(case_id, set())
        gold_documents = {
            passages[p].document_id for p in gold_ids if p in passages
        }
        blocks = []
        for rank, pid in enumerate(packed, start=1):
            p = passages.get(pid)
            body = (getattr(p, "raw_text", "") or "") if p else ""
            blocks.append({
                "rank": rank,
                "passage_id": pid,
                "document_id": getattr(p, "document_id", None) if p else None,
                "article_number": getattr(p, "article_number", None) if p else None,
                "is_gold_passage": pid in gold_ids,
                "is_gold_document": bool(
                    p is not None and p.document_id in gold_documents
                ),
                "text": body[: args.passage_chars],
                "text_truncated": len(body) > args.passage_chars,
            })
        rows.append({
            "id": case_id,
            args.metric: per_case[case_id][args.metric],
            "rouge_l": per_case[case_id].get("rouge_l"),
            "retrieval_status": trace.get("status"),
            "n_packed": len(packed),
            "n_gold_labelled": len(gold_ids),
            "gold_in_pack": any(b["is_gold_passage"] for b in blocks),
            "gold_document_in_pack": any(b["is_gold_document"] for b in blocks),
            "evidence": blocks,
            "prediction": predictions.get(case_id, ""),
            "reference": references.get(case_id, ""),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with_gold = sum(1 for r in rows if r["gold_in_pack"])
    with_gold_doc = sum(1 for r in rows if r["gold_document_in_pack"])
    print(json.dumps({
        "cases": len(rows),
        "gold_passage_in_pack": with_gold,
        "gold_document_in_pack": with_gold_doc,
        "labelled": sum(1 for r in rows if r["n_gold_labelled"]),
        "out": args.out.as_posix(),
        "note": "chua reference - artifact eval-only, khong dua vao git",
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
