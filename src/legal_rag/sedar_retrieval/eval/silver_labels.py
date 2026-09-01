"""Document-scoped silver labels from evaluation-only reference answers."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from legal_rag.sedar_retrieval.query.citation_parser import (
    CitationMention,
    parse_citations,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import (
    load_passages_jsonl,
)

SILVER_LABEL_SCHEMA_VERSION = "sedar-silver-labels-v2"
_YEAR_TOKEN = re.compile(r"^(?:19|20)\d{2}$")


@dataclass(frozen=True, slots=True)
class _DocumentScope:
    """Passage IDs and aliases belonging to one legal document."""

    document_id: str
    folded_name: str
    aliases: tuple[str, ...]
    article_ids: Mapping[str, tuple[str, ...]]


def _fold(value: str) -> str:
    """Fold text for identifier matching without changing stored source text."""

    decomposed = unicodedata.normalize("NFKD", value).casefold().replace("đ", "d")
    return "".join(character for character in decomposed if character.isalnum())


def _article_key(value: str) -> str:
    return value.casefold().strip()


def _document_aliases(document_name: str) -> tuple[str, ...]:
    """Build conservative name aliases, including prefixes through a year."""

    raw_tokens = tuple(token for token in re.split(r"[-_\s]+", document_name) if token)
    aliases = {_fold(document_name)}
    folded_name_without_suffix = re.sub(r"\d{4,}$", "", _fold(document_name))
    if len(folded_name_without_suffix) >= 8:
        aliases.add(folded_name_without_suffix)

    for index, token in enumerate(raw_tokens):
        if not _YEAR_TOKEN.fullmatch(_fold(token)):
            continue
        for end in range(index + 1, min(len(raw_tokens), index + 5) + 1):
            alias = _fold("".join(raw_tokens[:end]))
            if len(alias) >= 8:
                aliases.add(alias)
    return tuple(
        sorted((item for item in aliases if item), key=lambda item: (-len(item), item))
    )


def _build_document_scopes(passages_path: Path) -> tuple[_DocumentScope, ...]:
    """Group article-level passage IDs by document identity."""

    passages = load_passages_jsonl(str(passages_path))
    seen_passage_ids: set[str] = set()
    names: dict[str, str] = {}
    all_article_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    preferred_article_ids: dict[tuple[str, str], list[str]] = defaultdict(list)

    for passage in passages:
        if passage.passage_id in seen_passage_ids:
            raise ValueError(f"Duplicate passage_id: {passage.passage_id}")
        seen_passage_ids.add(passage.passage_id)
        document_id = str(passage.document_id)
        names.setdefault(document_id, passage.document_name or document_id)
        if passage.article_number is None:
            continue
        key = (document_id, _article_key(passage.article_number))
        all_article_ids[key].append(passage.passage_id)
        if passage.retrieval_level == "article":
            preferred_article_ids[key].append(passage.passage_id)

    grouped: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
    for key, passage_ids in all_article_ids.items():
        document_id, article = key
        preferred = preferred_article_ids.get(key)
        grouped[document_id][article] = tuple(preferred or passage_ids[:1])

    scopes = tuple(
        _DocumentScope(
            document_id=document_id,
            folded_name=_fold(names[document_id]),
            aliases=_document_aliases(names[document_id]),
            article_ids=dict(sorted(grouped.get(document_id, {}).items())),
        )
        for document_id in sorted(names)
    )
    if not scopes:
        raise ValueError(f"Passage corpus is empty: {passages_path}")
    return scopes


def _document_ids_for_number(
    document_number: str | None,
    scopes: tuple[_DocumentScope, ...],
) -> frozenset[str]:
    if not document_number:
        return frozenset()
    folded_number = _fold(document_number)
    if not folded_number:
        return frozenset()
    return frozenset(
        scope.document_id for scope in scopes if folded_number in scope.folded_name
    )


def _document_ids_for_text(
    text: str,
    scopes: tuple[_DocumentScope, ...],
) -> frozenset[str]:
    folded_text = _fold(text)
    if not folded_text:
        return frozenset()
    return frozenset(
        scope.document_id
        for scope in scopes
        if any(alias in folded_text for alias in scope.aliases)
    )


def _segment_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Return a punctuation-bounded citation segment."""

    separators = ".!?;\n"
    left = max(
        (text.rfind(separator, 0, start) for separator in separators), default=-1
    )
    right_candidates = [
        position
        for separator in separators
        for position in (text.find(separator, end),)
        if position >= 0
    ]
    right = min(right_candidates, default=len(text))
    return left + 1, right


def _resolve_article_document(
    *,
    answer: str,
    article_start: int,
    article_end: int,
    document_mentions: tuple[CitationMention, ...],
    scopes: tuple[_DocumentScope, ...],
) -> tuple[str | None, str]:
    """Resolve one article mention to one document, or fail closed."""

    segment_start, segment_end = _segment_bounds(
        answer,
        article_start,
        article_end,
    )
    local_distances: dict[str, int] = {}
    local_number_mention_count = 0
    for mention in document_mentions:
        mention_start = mention.start
        mention_end = mention.end
        if mention_end <= segment_start or mention_start >= segment_end:
            continue
        local_number_mention_count += 1
        candidates = _document_ids_for_number(mention.document_number, scopes)
        distance = min(
            abs(article_start - mention_end),
            abs(mention_start - article_end),
        )
        for document_id in candidates:
            previous = local_distances.get(document_id)
            local_distances[document_id] = (
                distance if previous is None else min(previous, distance)
            )

    if local_distances:
        best_distance = min(local_distances.values())
        best_documents = {
            document_id
            for document_id, distance in local_distances.items()
            if distance == best_distance
        }
        if len(best_documents) == 1:
            return next(iter(best_documents)), "resolved_document_number"
        return None, "ambiguous_document_number"

    segment = answer[segment_start:segment_end]
    if local_number_mention_count:
        return None, "document_number_not_in_corpus"
    alias_candidates = _document_ids_for_text(segment, scopes)
    if len(alias_candidates) == 1:
        return next(iter(alias_candidates)), "resolved_document_name"
    if len(alias_candidates) > 1:
        return None, "ambiguous_document_name"

    global_candidates = {
        document_id
        for mention in document_mentions
        for document_id in _document_ids_for_number(
            mention.document_number,
            scopes,
        )
    }
    if not global_candidates:
        global_candidates = set(_document_ids_for_text(answer, scopes))
    if len(global_candidates) == 1:
        return next(iter(global_candidates)), "resolved_global_document"
    if len(global_candidates) > 1:
        return None, "ambiguous_document_scope"
    return None, "document_identity_not_found"


def _build_label_row(
    *,
    query_id: str,
    answer: str,
    scopes: tuple[_DocumentScope, ...],
) -> tuple[dict[str, object], int, int]:
    citations = parse_citations(answer)
    article_mentions = tuple(item for item in citations if item.article)
    document_mentions = tuple(item for item in citations if item.document_number)
    scopes_by_id = {scope.document_id: scope for scope in scopes}
    resolved_scopes: list[dict[str, str]] = []
    relevant_ids: list[str] = []
    unresolved_count = 0
    resolution_reasons: Counter[str] = Counter()

    for article in article_mentions:
        document_id, resolution = _resolve_article_document(
            answer=answer,
            article_start=article.start,
            article_end=article.end,
            document_mentions=document_mentions,
            scopes=scopes,
        )
        article_number = str(article.article)
        if document_id is None:
            unresolved_count += 1
            resolution_reasons[resolution] += 1
            continue
        scope = scopes_by_id.get(document_id)
        article_ids = (
            scope.article_ids.get(_article_key(article_number), ())
            if scope is not None
            else ()
        )
        if not article_ids:
            unresolved_count += 1
            resolution_reasons["article_not_in_passages"] += 1
            continue
        resolution_reasons[resolution] += 1
        resolved_scopes.append(
            {
                "document_id": document_id,
                "article_number": article_number,
                "resolution": resolution,
            }
        )
        for passage_id in article_ids:
            if passage_id not in relevant_ids:
                relevant_ids.append(passage_id)

    if not article_mentions:
        resolution_reasons["no_article_citation"] += 1

    citation_summary = {
        "article_mentions": len(article_mentions),
        "document_mentions": len(document_mentions),
        "resolved_article_mentions": len(resolved_scopes),
        "unresolved_article_mentions": unresolved_count,
    }
    serialized_reasons = dict(sorted(resolution_reasons.items()))
    fully_resolved = (
        bool(article_mentions) and unresolved_count == 0 and bool(relevant_ids)
    )
    if fully_resolved:
        resolved_row: dict[str, object] = {
            "schema_version": SILVER_LABEL_SCHEMA_VERSION,
            "query_id": query_id,
            "relevant_ids": relevant_ids,
            "provenance": "silver",
            "resolution_status": "resolved",
            "resolved_scopes": resolved_scopes,
            "citation_summary": citation_summary,
            "resolution_reasons": serialized_reasons,
            "note": "evaluation_only_silver_document_article_scoped",
        }
        return resolved_row, len(article_mentions), 0

    note = "no_article_citation" if not article_mentions else "ambiguous_document_scope"
    unresolved_row: dict[str, object] = {
        "schema_version": SILVER_LABEL_SCHEMA_VERSION,
        "query_id": query_id,
        "relevant_ids": [],
        "provenance": "unlabeled",
        "resolution_status": "unresolved",
        "resolved_scopes": resolved_scopes,
        "citation_summary": citation_summary,
        "resolution_reasons": serialized_reasons,
        "note": note,
    }
    return (
        unresolved_row,
        len(resolved_scopes),
        unresolved_count or len(article_mentions),
    )


def build_silver_labels_from_answers(
    *,
    questions_path: Path,
    passages_path: Path,
    output_path: Path,
    limit: int = 0,
) -> dict[str, object]:
    """Build document-scoped silver labels from evaluation-only answers.

    A citation is labeled only when its document identity and article number
    resolve to a passage in the same document. Ambiguous citations are marked
    ``unlabeled`` instead of selecting arbitrary passages with the same article
    number elsewhere in the corpus.
    """

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, Mapping):
        raise ValueError("Questions artifact must be a JSON object")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    scopes = _build_document_scopes(passages_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_labeled = 0
    n_unlabeled = 0
    resolved_citations = 0
    ambiguous_citations = 0
    resolution_reason_counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for index, (qid, payload) in enumerate(
            sorted(questions.items(), key=lambda x: x[0])
        ):
            if limit and index >= limit:
                break
            if not isinstance(payload, Mapping):
                raise ValueError(f"Question record must be an object: {qid}")
            answer = str(payload.get("answer") or "")
            row, resolved_count, ambiguous_count = _build_label_row(
                query_id=str(qid),
                answer=answer,
                scopes=scopes,
            )
            if row["provenance"] == "silver":
                n_labeled += 1
            else:
                n_unlabeled += 1
            resolved_citations += resolved_count
            ambiguous_citations += ambiguous_count
            raw_reasons = row.get("resolution_reasons", {})
            if isinstance(raw_reasons, Mapping):
                for reason, count in raw_reasons.items():
                    resolution_reason_counts[str(reason)] += int(count)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "labeled": n_labeled,
        "unlabeled": n_unlabeled,
        "total": n_labeled + n_unlabeled,
        "resolved_citations": resolved_citations,
        "ambiguous_citations": ambiguous_citations,
        "resolution_reason_counts": dict(sorted(resolution_reason_counts.items())),
        "schema_version": SILVER_LABEL_SCHEMA_VERSION,
    }
