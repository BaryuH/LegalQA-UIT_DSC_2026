"""Official submission builder per SUBMISSION_CONTRACT.

Creates submission.zip containing exactly one root submission.json
with format: {"question_id": {"answer": "..."}}
UTF-8, ensure_ascii=False. No internal metadata fields emitted.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


class SubmissionError(ValueError):
    """Raised when submission validation fails."""


def create_submission(
    predictions: dict[str, str],
    question_ids: list[str],
    output_path: str | Path,
) -> Path:
    """Create a valid submission.zip from predictions.

    Args:
        predictions: {question_id: answer_text}
        question_ids: Expected question IDs from the dataset (ordering source).
        output_path: Path for the output submission.zip.

    Returns:
        Path to the created submission.zip.
    """
    out = Path(output_path)

    # Validate: no missing, no extra, no duplicates
    pred_ids = set(predictions.keys())
    expected_ids = set(question_ids)

    missing = expected_ids - pred_ids
    extra = pred_ids - expected_ids

    if missing:
        raise SubmissionError(
            f"Missing predictions for {len(missing)} IDs: "
            f"{sorted(missing)[:10]}..."
        )
    if extra:
        raise SubmissionError(
            f"Extra predictions for {len(extra)} IDs: "
            f"{sorted(extra)[:10]}..."
        )

    # Build payload in dataset input order
    payload: dict[str, dict[str, str]] = {}
    empty_count = 0
    for qid in question_ids:
        answer = predictions[qid]
        if not answer.strip():
            empty_count += 1
            logger.warning("Empty answer for question ID: %s", qid)
        payload[qid] = {"answer": answer}

    if empty_count > 0:
        logger.warning(
            "%d/%d predictions have empty answers", empty_count, len(question_ids)
        )

    # Serialize JSON
    json_bytes = json.dumps(
        payload, ensure_ascii=False, indent=None
    ).encode("utf-8")

    # Validate by re-parsing
    reparsed = json.loads(json_bytes.decode("utf-8"))
    if reparsed != payload:
        raise SubmissionError("JSON round-trip validation failed")

    # Write ZIP with exactly one root member: submission.json
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SubmissionError(f"Output path already exists: {out}")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json_bytes)

    # Validate the written ZIP
    _validate_submission_zip(out)

    logger.info(
        "Submission created: %s (%d predictions, %d bytes)",
        out,
        len(payload),
        out.stat().st_size,
    )
    return out


def _validate_submission_zip(path: Path) -> None:
    """Validate the written submission ZIP per SUBMISSION_CONTRACT."""
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if names != ["submission.json"]:
            raise SubmissionError(
                f"ZIP must contain exactly 'submission.json', got: {names}"
            )

        raw = zf.read("submission.json")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SubmissionError(f"Invalid JSON in submission.json: {exc}") from exc

        if not isinstance(payload, dict):
            raise SubmissionError("submission.json top-level must be a JSON object")

        for qid, value in payload.items():
            if not isinstance(qid, str):
                raise SubmissionError(f"Question ID must be string: {qid!r}")
            if not isinstance(value, dict):
                raise SubmissionError(
                    f"Value for {qid!r} must be an object, got {type(value).__name__}"
                )
            if set(value.keys()) != {"answer"}:
                raise SubmissionError(
                    f"Value for {qid!r} must have exactly 'answer' field, "
                    f"got: {sorted(value.keys())}"
                )
            if not isinstance(value["answer"], str):
                raise SubmissionError(
                    f"Answer for {qid!r} must be a string"
                )
