# Reproducibility

## Source-data manifest gate

Before every experiment, run this command from the repository root:

```bash
python scripts/verify_data_manifest.py
```

The command must pass before any experiment starts. It verifies the deterministic
SHA256 manifest at `artifacts/data-baseline/manifest.json` and fails with a non-zero
exit code when a source file is missing, extra, or has a different size/SHA256.

The current source scope is:

- every file recursively under `data/` (currently `train.json`, `warmup.json`,
  `public-official.json`, and `selected-contexts.zip`);
- the root-level `selected-contexts.zip` when it exists as a separate copy.

Paths containing `cache/`, `output/`, or `outputs/` are excluded from hashing. The
manifest stores each included file's repository-relative POSIX path, byte size, and
lowercase SHA256 digest. Hashing reads files in binary chunks and never copies,
extracts, normalizes, or rewrites source data or archives.

## Per-run artifact contract

The default run ID is a UTC timestamp followed by the configured split and method:

```text
outputs/<timestamp>_<split>_<method>/
```

`legal_rag.artifacts.RunManager` reserves the directory and writes each artifact
through a temporary file followed by an atomic replace. Existing run directories and
artifacts are never overwritten. JSON object keys and JSONL records use deterministic
ordering.

Every run records:

- `config.json`: a redacted configuration snapshot and `config_hash`;
- `environment.json`: Python/platform details, installed package versions, seed,
  Git commit, and dirty state;
- `run_summary.json`: model/prompt, data-manifest, index/chunk fingerprints and
  artifact references;
- `predictions.jsonl`, `generation.jsonl`, `errors.jsonl`, and `retrieval.jsonl`
  (for retrieval methods only);
- `metrics.json` and `submission.json`: evaluation status and the internal run
  placeholder. The official final artifact is produced separately as
  `submission.zip` by `create-submission`; it contains only root `submission.json`.
- `artifacts/experiments.jsonl`: one append-only I1 registry row with run identity,
  Git/config/split fingerprints, model/seed/prompt, metric slots, error rate,
  mean generation latency, reranker fallback rate, and a hash of the completed
  run artifacts. `meteor` and `rouge_l` are `null` until an approved evaluation
  artifact is available.

Inference artifacts contain no gold/reference answers or chain-of-thought. A provider
or reranker fallback is represented by its existing structured status/reason rather
than hidden by the run manager.

## Official submission artifact

Build and validate the final package from inference predictions and the question
dataset. The dataset supplies expected IDs and their order; prediction IDs are not
used to define coverage:

```bash
python -m legal_rag.cli create-submission \
  --split public \
  --predictions outputs/<run>/predictions.jsonl \
  --questions data/public-official.json \
  --output submission.zip
python -m legal_rag.cli validate-submission \
  --split public \
  --submission submission.zip \
  --questions data/public-official.json
```

For a complete batch, `run --package-submission` calls the same dedicated
serializer after inference and fails if the prediction artifact is incomplete.

The writer uses UTF-8 `ensure_ascii=False`, writes no gold or internal metadata,
preserves dataset order, rejects missing/extra/duplicate IDs, and requires the
exact ZIP layout `submission.zip/submission.json`. See
[`SUBMISSION_CONTRACT.md`](SUBMISSION_CONTRACT.md) for typed diagnostics.

Case-level failure isolation and the stable error categories are defined in
[`ERROR_TAXONOMY.md`](ERROR_TAXONOMY.md). The configured `runtime.fail_fast` controls
whether a recorded case failure stops the batch; it never suppresses the error
artifact.

## Evaluation-only error reports

After an approved evaluation has produced `metrics.json`, generate deterministic
worst-first Markdown and CSV reports with:

```bash
python scripts/generate_error_report.py \
  --predictions outputs/<run_id>/predictions.jsonl \
  --references <approved-reference-file> \
  --metrics <evaluation-metrics.json> \
  --retrieval outputs/<run_id>/retrieval.jsonl \
  --markdown reports/<run_id>-errors.md \
  --csv reports/<run_id>-errors.csv
```

The metrics artifact must declare
`reference_role: evaluation_reference_only`. Manual labels can be supplied with
`--error-types <case-id-to-taxonomy-json>`. A private report is rejected by
default and requires the explicit `--allow-private` flag; the report is never an
inference artifact.

## Updating the baseline intentionally

Only after an approved source-data change, regenerate the baseline explicitly:

```bash
python scripts/verify_data_manifest.py --write
```

Review the resulting manifest diff. Do not use `--write` as a substitute for the
verification gate, and do not run it after an unapproved data change.

## Related invariants

- `data/` is read-only source data.
- Private test data is never used for tuning.
- Cache and output artifacts are not source inputs and are not part of this manifest.
- The manifest is a provenance artifact, not a production data loader.
- The selected-context loader reads ZIP members directly or walks a directory;
  it never permanently extracts into `data/` and retains source file/member
  provenance on every `LegalDocument`.
- LLM client metadata records only configured provider/model and operational
  counters; prompts, references, secrets, and chain-of-thought are not logged.
- The offline mock can use exact prompt fixtures or a SHA256 prompt hash and
  performs retry tests without network calls, model downloads, or sleeping.
- Direct and RAG templates are loaded from `configs/prompts/` as exact UTF-8
  bytes. Each rendered prompt carries only its prompt name, configured version,
  and SHA256 template hash; the builder accepts no whole question record or gold.
- Answer postprocessing stores both `raw_answer` and `cleaned_answer`. Cleaning
  is limited to line-ending normalization, outer whitespace, one/two outer
  technical wrappers, and the two exact approved prefixes; legal content is not
  rewritten.

## OpenAI-compatible local/API adapter

The configured `openai` provider targets an OpenAI-compatible
`/chat/completions` endpoint. The adapter uses only the environment variable
named by `generation.api_key_env`; no API key is accepted in YAML or Python
configuration. A local endpoint can omit `api_key_env`.

```yaml
generation:
  provider: openai
  model: local-chat-model
  base_url: http://localhost:8000/v1
  api_key_env: OPENAI_API_KEY
  temperature: 0.0
  max_output_chars: 1200
  retries: 3
  timeout_seconds: 30.0
  backoff_seconds: 0.5
```

Only timeouts, connection resets, HTTP 429, and HTTP 5xx responses are retried.
Backoff is exponential and request/response byte sizes are retained as
operational metadata. Prompt and response bodies are not logged by default.
