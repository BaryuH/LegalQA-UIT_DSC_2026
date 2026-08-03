import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from legal_rag.contexts import ContextLoadError, load_contexts


def _context(
    document_id: int | str, passage: str, *, link: str | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": document_id,
        "name": f"Văn bản {document_id}",
        "passage": passage,
    }
    if link is not None:
        payload["link"] = link
    return payload


def _write_zip(path: Path, members: list[tuple[str, object]]) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        for member, payload in members:
            if isinstance(payload, str):
                archive.writestr(member, payload)
            else:
                archive.writestr(
                    member,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                )


def test_zip_loader_is_deterministic_preserves_provenance_and_warns(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "selected-contexts.zip"
    _write_zip(
        archive_path,
        [
            ("README.txt", "ignored"),
            ("nested/context_10.json", _context(10, "Điều 10.")),
            (
                "context_2.json",
                _context(
                    2,
                    "  Điều 2.\r\nNội dung nguyên bản.  ",
                    link="https://example.test/context-2",
                ),
            ),
        ],
    )
    before = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    corpus = load_contexts(archive_path)

    after = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert before == after
    assert [document.id for document in corpus] == ["10", "2"]
    assert corpus.documents[1].passage == "  Điều 2.\r\nNội dung nguyên bản.  "
    assert corpus.documents[1].link == "https://example.test/context-2"
    assert corpus.documents[0].source_path == str(archive_path)
    assert corpus.documents[0].source_member == "nested/context_10.json"
    with zipfile.ZipFile(archive_path) as archive:
        raw_member = archive.read("nested/context_10.json")
    assert corpus.documents[0].content_hash == hashlib.sha256(raw_member).hexdigest()
    assert len(corpus.warnings) == 1
    warning = corpus.warnings[0]
    assert warning.code == "UNRELATED_FILE"
    assert warning.path == f"{archive_path}::README.txt"
    assert warning.as_dict()["source_member"] == "README.txt"
    assert not (tmp_path / "nested").exists()


def test_directory_loader_uses_file_provenance_and_optional_link(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "contexts"
    nested.mkdir()
    (nested / "context_b.json").write_text(
        json.dumps(_context("b", "Passage B"), ensure_ascii=False), encoding="utf-8"
    )
    (nested / "context_a.json").write_text(
        json.dumps(_context("a", "Passage A"), ensure_ascii=False), encoding="utf-8"
    )
    (nested / "notes.md").write_text("ignored", encoding="utf-8")

    corpus = load_contexts(nested)

    assert [document.id for document in corpus.documents] == ["a", "b"]
    assert corpus.documents[0].source_path == str(nested / "context_a.json")
    assert corpus.documents[0].source_member is None
    assert corpus.documents[0].link is None
    assert corpus.warnings[0].path == str(nested / "notes.md")


def test_duplicate_document_ids_fail_with_both_locations(tmp_path: Path) -> None:
    archive_path = tmp_path / "contexts.zip"
    _write_zip(
        archive_path,
        [
            ("context_first.json", _context(7, "Một.")),
            ("context_second.json", _context("7", "Hai.")),
        ],
    )

    with pytest.raises(ContextLoadError, match="duplicate document ID '7'") as exc_info:
        load_contexts(archive_path)

    message = str(exc_info.value)
    assert "context_first.json" in message
    assert "context_second.json" in message


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "   ", "field 'id' must not be blank"),
        ("name", "\n", "field 'name' must not be blank"),
        ("passage", "\t", "field 'passage' must not be blank"),
        ("link", 123, "optional field 'link' must be a string"),
    ],
)
def test_document_fields_are_validated(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _context("1", "Nội dung")
    payload[field] = value
    archive_path = tmp_path / "contexts.zip"
    _write_zip(archive_path, [("context_1.json", payload)])

    with pytest.raises(ContextLoadError, match=message):
        load_contexts(archive_path)


def test_invalid_context_json_is_not_silently_warned(tmp_path: Path) -> None:
    archive_path = tmp_path / "contexts.zip"
    _write_zip(archive_path, [("context_bad.json", "{not-json")])

    with pytest.raises(ContextLoadError, match="context_bad.json.*invalid JSON"):
        load_contexts(archive_path)


def test_empty_corpus_fails_even_when_only_unrelated_files_exist(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.txt").write_text("not a corpus", encoding="utf-8")

    with pytest.raises(ContextLoadError, match=r"no context_\*\.json documents"):
        load_contexts(tmp_path)
