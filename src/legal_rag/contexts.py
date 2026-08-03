"""Read-only loader for the selected legal-context corpus.

The competition corpus is documented as a directory or ``selected-contexts.zip``
containing one ``context_*.json`` object per legal document. This module reads
those sources in place and never extracts or rewrites them. Each source file is
read and validated independently, so temporary memory is bounded by the largest
source record plus the validated documents returned to the caller.
"""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from .schemas import LegalDocument


class ContextLoadError(ValueError):
    """Raised when a selected-context source cannot satisfy its schema."""


@dataclass(frozen=True, slots=True)
class ContextLoadWarning:
    """Structured diagnostic for a source member ignored by the loader."""

    code: str
    source_path: str
    message: str
    source_member: str | None = None

    @property
    def path(self) -> str:
        """Return a human-readable source location for reports."""

        if self.source_member is None:
            return self.source_path
        return f"{self.source_path}::{self.source_member}"

    def as_dict(self) -> dict[str, str | None]:
        """Serialize the warning without losing archive-member provenance."""

        return {
            "code": self.code,
            "source_path": self.source_path,
            "source_member": self.source_member,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ContextCorpus:
    """Validated documents and non-fatal diagnostics from one corpus source."""

    documents: tuple[LegalDocument, ...]
    warnings: tuple[ContextLoadWarning, ...]
    source_path: str

    def __iter__(self) -> Iterator[LegalDocument]:
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(
        self, index: int | slice
    ) -> LegalDocument | tuple[LegalDocument, ...]:
        return self.documents[index]


ContextLoadResult = ContextCorpus


class _ObjectPairs(list[tuple[str, Any]]):
    """Retain JSON object pairs so duplicate keys cannot be silently lost."""


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> _ObjectPairs:
    return _ObjectPairs(pairs)


def _convert_json(value: Any, location: str) -> Any:
    if isinstance(value, _ObjectPairs):
        result: dict[str, Any] = {}
        for key, item in value:
            if key in result:
                raise ContextLoadError(
                    f"{location}: duplicate JSON object field {key!r}"
                )
            result[key] = _convert_json(item, location)
        return result
    if isinstance(value, list):
        return [_convert_json(item, location) for item in value]
    return value


def _is_context_filename(filename: str) -> bool:
    """Match the documented ``context_*.json`` corpus members."""

    basename = PurePosixPath(filename).name
    return basename.startswith("context_") and basename.casefold().endswith(".json")


def _location(source_path: str, source_member: str | None) -> str:
    if source_member is None:
        return source_path
    return f"{source_path}::{source_member}"


def _unrelated_warning(
    source_path: str, source_member: str | None
) -> ContextLoadWarning:
    return ContextLoadWarning(
        code="UNRELATED_FILE",
        source_path=source_path,
        source_member=source_member,
        message="Ignored file that does not match context_*.json",
    )


def _validate_document_fields(
    payload: Any, source_path: str, source_member: str | None
) -> dict[str, Any]:
    location = _location(source_path, source_member)
    if not isinstance(payload, dict):
        raise ContextLoadError(
            f"{location}: top-level schema must be a JSON object; "
            f"got {type(payload).__name__}"
        )

    expected_fields = {"id", "name", "passage", "link"}
    required_fields = {"id", "name", "passage"}
    missing_fields = required_fields - set(payload)
    if missing_fields:
        raise ContextLoadError(
            f"{location}: missing required field(s): "
            f"{', '.join(sorted(missing_fields))}"
        )
    extra_fields = set(payload) - expected_fields
    if extra_fields:
        raise ContextLoadError(
            f"{location}: unexpected field(s): {', '.join(sorted(extra_fields))}"
        )

    document_id = payload["id"]
    if isinstance(document_id, bool) or not isinstance(document_id, (str, int)):
        raise ContextLoadError(
            f"{location}: field 'id' must be a non-blank string or integer"
        )
    if isinstance(document_id, str) and not document_id.strip():
        raise ContextLoadError(f"{location}: field 'id' must not be blank")

    for field_name in ("name", "passage"):
        value = payload[field_name]
        if not isinstance(value, str):
            raise ContextLoadError(f"{location}: field {field_name!r} must be a string")
        if not value.strip():
            raise ContextLoadError(
                f"{location}: field {field_name!r} must not be blank"
            )

    link = payload.get("link")
    if link is not None and not isinstance(link, str):
        raise ContextLoadError(
            f"{location}: optional field 'link' must be a string when present"
        )
    if isinstance(link, str) and not link.strip():
        raise ContextLoadError(f"{location}: optional field 'link' must not be blank")

    return payload


def _build_document(
    payload: Any,
    source_path: str,
    source_member: str | None,
    content_hash: str,
) -> LegalDocument:
    fields = _validate_document_fields(payload, source_path, source_member)
    location = _location(source_path, source_member)
    try:
        return LegalDocument(
            id=fields["id"],
            name=fields["name"],
            link=fields.get("link"),
            passage=fields["passage"],
            source_path=source_path,
            source_member=source_member,
            content_hash=content_hash,
        )
    except ValidationError as exc:
        raise ContextLoadError(f"{location}: invalid legal document: {exc}") from exc


def _read_document(
    content: bytes, source_path: str, source_member: str | None
) -> LegalDocument:
    location = _location(source_path, source_member)
    try:
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_object_pairs_hook
        )
        converted = _convert_json(payload, location)
    except ContextLoadError:
        raise
    except UnicodeDecodeError as exc:
        raise ContextLoadError(f"{location}: source must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ContextLoadError(f"{location}: invalid JSON: {exc}") from exc
    return _build_document(
        converted,
        source_path,
        source_member,
        content_hash=sha256(content).hexdigest(),
    )


def _directory_files(root: Path) -> Iterator[Path]:
    """Yield directory files in deterministic order without collecting the tree."""

    for current, directories, filenames in os.walk(root):
        directories.sort()
        for filename in sorted(filenames):
            yield Path(current) / filename


def _load_directory(
    root: Path,
    documents: list[LegalDocument],
    warnings: list[ContextLoadWarning],
) -> None:
    for file_path in _directory_files(root):
        if not _is_context_filename(file_path.name):
            warnings.append(_unrelated_warning(str(file_path), None))
            continue
        try:
            content = file_path.read_bytes()
        except OSError as exc:
            raise ContextLoadError(f"{file_path}: cannot read source: {exc}") from exc
        documents.append(_read_document(content, str(file_path), None))


def _load_zip(
    archive_path: Path,
    documents: list[LegalDocument],
    warnings: list[ContextLoadWarning],
) -> None:
    source_path = str(archive_path)
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            entries = sorted(archive.infolist(), key=lambda info: info.filename)
            for info in entries:
                if info.is_dir():
                    continue
                member = info.filename
                if not _is_context_filename(member):
                    warnings.append(_unrelated_warning(source_path, member))
                    continue
                try:
                    content = archive.read(info)
                except OSError as exc:
                    location = _location(source_path, member)
                    raise ContextLoadError(
                        f"{location}: cannot read source: {exc}"
                    ) from exc
                documents.append(_read_document(content, source_path, member))
    except ContextLoadError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise ContextLoadError(f"{source_path}: cannot read ZIP source: {exc}") from exc


def load_contexts(path: str | Path) -> ContextCorpus:
    """Load selected contexts from a directory or ZIP archive.

    Only files named ``context_*.json`` are parsed. All other files are ignored
    and returned as structured ``UNRELATED_FILE`` warnings. Documents are sorted
    by canonical string ID; duplicate IDs and an empty corpus are errors. The
    source path is read-only and is never extracted or rewritten.
    """

    source = Path(path)
    if not source.exists():
        raise ContextLoadError(f"Source corpus missing: {source}")
    if not source.is_dir() and source.suffix.casefold() != ".zip":
        raise ContextLoadError(
            f"Source corpus must be a directory or ZIP archive: {source}"
        )

    documents: list[LegalDocument] = []
    warnings: list[ContextLoadWarning] = []
    if source.is_dir():
        _load_directory(source, documents, warnings)
    else:
        _load_zip(source, documents, warnings)

    seen_locations: dict[str, str] = {}
    for document in documents:
        location = _location(document.source_path, document.source_member)
        previous = seen_locations.get(document.id)
        if previous is not None:
            raise ContextLoadError(
                f"{location}: duplicate document ID {document.id!r}; "
                f"first seen at {previous}"
            )
        seen_locations[document.id] = location

    if not documents:
        raise ContextLoadError(
            f"{source}: no context_*.json documents found in selected corpus"
        )

    documents.sort(
        key=lambda document: (
            document.id,
            document.source_path,
            document.source_member or "",
        )
    )
    warnings.sort(key=lambda item: item.path)
    return ContextCorpus(
        documents=tuple(documents),
        warnings=tuple(warnings),
        source_path=str(source),
    )


def load_selected_contexts(path: str | Path) -> tuple[LegalDocument, ...]:
    """Return only documents for callers using the original loader API."""

    return load_contexts(path).documents
