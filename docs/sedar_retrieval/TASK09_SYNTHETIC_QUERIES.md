# SEDAR Retrieval v3 — TASK 09 Synthetic Queries

TASK 09 creates training-only Vietnamese legal queries from canonical R2a
passages.  It never reads question answers, warmup references, public data, or
the frozen SEDAR-SFT reader.

The frozen machine query-type IDs are `direct`, `citizen_paraphrase`, `scenario`,
`citation_free`, and `condition_exception`.

## Local scaffold

The deterministic template backend validates the schema, document split, and
filters without downloading a generator model.  Its output is a scaffold and
must not be promoted as a quality result without manual audit.

```bash
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"

VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"
OUT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/training/task09_template_smoke"

python scripts/sedar_retrieval/generate_synthetic_queries.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --output-dir "$OUT" \
  --backend template \
  --target-accepted 100 \
  --max-attempts 500
```

## Server pilot

Use an explicitly approved causal generator model.  Do not use the frozen
SEDAR-SFT reader as a mutable generator.  Pin the generator revision.

```bash
GENERATOR_MODEL="REPLACE_WITH_APPROVED_CAUSAL_MODEL"
GENERATOR_REVISION="REPLACE_WITH_VERIFIED_GENERATOR_COMMIT"

OUT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/training/task09_pilot_10k"

python scripts/sedar_retrieval/generate_synthetic_queries.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --output-dir "$OUT" \
  --backend transformers \
  --generator-model "$GENERATOR_MODEL" \
  --generator-revision "$GENERATOR_REVISION" \
  --device cuda \
  --dtype bf16 \
  --target-accepted 10000 \
  --max-attempts 30000 \
  --seed 42 \
  --validation-fraction 0.1 \
  --test-fraction 0.1 \
  --local-files-only
```

The generator uses only `reader_text` from the selected passage and writes:

- `synthetic_queries.jsonl`
- `rejections.jsonl`
- `generation_errors.jsonl`
- `audit.json`
- `manifest.json`

Each record contains the required passage/document provenance, query type,
generator revision, prompt version, source hash, quality flags, and raw
generation.  Source documents are deterministically partitioned into disjoint
train/validation/test groups; only the configured `train` group is generated.
For a previously curated holdout, pass `--validation-documents` and
`--test-documents` with JSON lists or JSONL document-ID manifests.  Manifest
IDs must exist in the passage corpus and the two holdouts must not overlap.

Model reasoning is removed before `raw_generation` is persisted.  Provider
failures and rejected attempts are written as metadata-only artifacts; prompts,
answers, and rejected model output are not written.

## Exit gate

The manifest remains `NEEDS_MANUAL_AUDIT` after generation.  Audit at least 300
records with stratification over query type and source document.  Scale to
50k/100k only when:

- accepted count is 10k;
- supportable queries ≥95%;
- valid Vietnamese ≥98%;
- wrong citation ≤1%;
- validation/test leakage is zero.

If any threshold fails, fix the prompt or filters and regenerate a new run.
Do not overwrite or silently promote the previous run.
