# Official submission contract

This repository uses the fixed `SUBMISSION-P0` contract. The official artifact is
always a file named `submission.zip` and is created by the dedicated serializer;
retrieval, generation, prompt, and evaluator code do not write submission fields.

## ZIP boundary

The ZIP must contain exactly one root member named `submission.json`:

```text
submission.zip
└── submission.json
```

Nested paths, directories, `__MACOSX`, `.DS_Store`, duplicate members, and any
other metadata are invalid. The packager writes a fixed timestamp and validates
the archive before replacing the final path. Existing final files are never
overwritten.

## JSON boundary

`submission.json` is UTF-8 JSON written with `ensure_ascii=False`. Its top-level
value is an object mapping question IDs to objects with exactly one field:

```json
{
  "007": {"answer": "Câu trả lời tiếng Việt"}
}
```

Answers must be strings. Empty strings are schema-valid by default and produce a
warning; a caller can explicitly enable the reject-empty policy. Numbers,
`null`, booleans, arrays, direct string values, missing `answer`, and extra fields
fail validation. No answer text is rewritten or generated during serialization.

## ID and ordering rules

- Expected IDs come from the inference/submission question dataset, never from
  the prediction artifact.
- A raw integer ID becomes `str(int)`.
- A raw string ID is preserved exactly, including leading zeroes; strings are
  never stripped or parsed as numbers.
- Duplicate, missing, and extra IDs fail closed.
- JSON object insertion order is the dataset input order and is preserved in the
  writer and validator.

The question-ID loader reads only ID keys/fields. It does not use question text,
reference answers, evidence, scores, model metadata, or private evaluator data.

## Typed failure codes

The JSON boundary reports:

`SUBMISSION_FILE_NOT_FOUND`, `SUBMISSION_WRONG_FILENAME`,
`SUBMISSION_INVALID_UTF8`, `SUBMISSION_INVALID_JSON`,
`SUBMISSION_TOP_LEVEL_NOT_OBJECT`, `SUBMISSION_INVALID_QUESTION_ID`,
`SUBMISSION_DUPLICATE_QUESTION_ID`, `SUBMISSION_MISSING_QUESTION_ID`,
`SUBMISSION_EXTRA_QUESTION_ID`, `SUBMISSION_VALUE_NOT_OBJECT`,
`SUBMISSION_MISSING_ANSWER`, `SUBMISSION_ANSWER_NOT_STRING`,
`SUBMISSION_EXTRA_FIELDS`, and `SUBMISSION_EMPTY_ANSWER`.

The ZIP boundary additionally reports `SUBMISSION_ZIP_INVALID`,
`SUBMISSION_ZIP_WRONG_LAYOUT`, `SUBMISSION_ZIP_EXTRA_MEMBER`, and
`SUBMISSION_ZIP_MISSING_JSON`.

## Commands

Official profiles fix `format: object_by_question_id`, `answer_field: answer`,
`encoding: utf-8`, `ensure_ascii: false`, `forbid_extra_fields: true`,
`require_full_coverage: true`, `reject_extra_ids: true`,
`reject_duplicate_ids: true`, and `reject_non_string_answers: true`.

Create the final artifact from an inference prediction JSONL and the question
dataset:

```bash
python -m legal_rag.cli create-submission \
  --predictions outputs/<run>/predictions.jsonl \
  --questions data/public-official.json \
  --output submission.zip
```

Validate without rewriting:

```bash
python -m legal_rag.cli validate-submission \
  --submission submission.zip \
  --questions data/public-official.json
```

Creation and validation are non-zero on any contract failure. Unit tests use
synthetic offline fixtures only; no model, provider, private reference, or
network access is needed.
