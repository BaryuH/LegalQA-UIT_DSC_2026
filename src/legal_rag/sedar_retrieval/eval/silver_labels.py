"""Silver retrieval labels from evaluation-only gold answers (TASK 02 support).

Gold answers are used solely to build evaluation labels. They must never enter
retrieval indexes, queries, or generator prompts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from legal_rag.sedar_retrieval.query.citation_parser import parse_citations

_ARTICLE_IN_ANSWER = re.compile(
    r"điều\s+(?P<article>\d+[A-Za-z]?)",
    flags=re.IGNORECASE | re.UNICODE,
)


def build_silver_labels_from_answers(
    *,
    questions_path: Path,
    passages_path: Path,
    output_path: Path,
    limit: int = 0,
) -> dict[str, int]:
    """Map warmup/train answers' cited article numbers onto passage IDs."""

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    # passages: article_number -> passage_ids (article-level preferred)
    article_index: dict[str, list[str]] = {}
    for line in passages_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("retrieval_level") != "article":
            continue
        number = row.get("article_number")
        if not number:
            continue
        article_index.setdefault(str(number), []).append(str(row["passage_id"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_labeled = 0
    n_unlabeled = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, (qid, payload) in enumerate(sorted(questions.items(), key=lambda x: x[0])):
            if limit and index >= limit:
                break
            answer = str(payload.get("answer") or "")
            articles = {m.group("article") for m in _ARTICLE_IN_ANSWER.finditer(answer)}
            # Also parse structured citations.
            for mention in parse_citations(answer):
                if mention.article:
                    articles.add(mention.article)
            relevant: list[str] = []
            for article in sorted(articles):
                relevant.extend(article_index.get(article, [])[:3])
            provenance = "silver" if relevant else "unlabeled"
            if relevant:
                n_labeled += 1
            else:
                n_unlabeled += 1
            handle.write(
                json.dumps(
                    {
                        "query_id": str(qid),
                        "relevant_ids": relevant,
                        "provenance": provenance,
                        "note": "evaluation_only_silver_from_answer_citations",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {"labeled": n_labeled, "unlabeled": n_unlabeled, "total": n_labeled + n_unlabeled}
