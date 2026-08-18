# TASK 10 — Hard / semi-hard negatives

TASK 10 mines training negatives for the accepted TASK 09 synthetic queries.
The miner reads only the synthetic query, its positive passage ID, canonical
passage metadata, and ranked retrieval candidates.  It never reads answers or
the private test set.

The implementation is:

```text
src/legal_rag/sedar_retrieval/training/hard_negatives.py
scripts/sedar_retrieval/mine_hard_negatives.py
tests/sedar_retrieval/test_hard_negatives.py
```

## Candidate sources

The CLI accepts either or both of these sources:

1. `--bm25-cache-root`: retrieve each synthetic query directly from a
   corpus-fingerprinted BM25 index.
2. `--candidate-file`: one or more ranked JSONL files.  The file follows the
   existing retrieval contract:

```json
{"query_id":"syn-...", "ranked_ids":["passage-1"], "scores":[
  {"passage_id":"passage-1","dense":0.81,"rank":1,"source":"dense"}
]}
```

The existing TASK 08 RRF output is also accepted:

```json
{"query_id":"syn-...", "ranked_ids":["passage-1"], "candidates":[
  {"passage_id":"passage-1","bm25_score":4.1,"dense_score":0.81,
   "rrf_score":0.03,"fused_rank":1}
]}
```

For fused candidates, `mined_from` is `rrf` and `mining_sources` preserves
the contributing sources.  A dense contribution takes precedence for the E
taxonomy when no structural A–D classification applies.

For TASK 07 dense candidates, the `query_id` must be the TASK 09
`synthetic_id`.  A candidate with an unknown canonical passage ID is not
silently retained: it is counted as unresolved and cannot satisfy the
minimum-negative contract.

## Server command

Use a new output directory for each run:

```bash
python scripts/sedar_retrieval/mine_hard_negatives.py \
  --synthetic "$PILOT_OUT/synthetic_queries.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --bm25-cache-root "$INDEXES/bm25_r2a" \
  --candidate-file "$DENSE_SYNTHETIC_CANDIDATES" \
  --output-dir "$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/training/task10_hn_<run>" \
  --top-k 150 \
  --min-negatives 2 \
  --max-negatives 5 \
  --seed 42
```

`--candidate-file` is optional when the BM25 source is sufficient.  The
current local scaffold does not load a dense model or fall back to CPU; dense
candidate generation remains an explicit server-side step.

For a smoke run, use `--limit 20`.  `--force` is required to overwrite an
existing output directory artifact.

## Taxonomy and safety checks

Each selected negative is resolved back to a canonical passage and receives:

```text
A same-law wrong article
B same-article wrong clause
C similar term, wrong applicability
D stale/repealed passage only when status metadata is explicit
E dense-mined
F BM25-mined
```

Structural categories take precedence over E/F.  E/F records retain
`mined_from` and fused records retain `mining_sources`, so the retrieval
source is never lost.  The miner:

- excludes the positive passage ID;
- deduplicates candidate IDs;
- selects two to five negatives per accepted query;
- records rank, score, category, source, source hash, and hardness score;
- flags citation overlap, same-article overlap, high lexical agreement,
  high dense agreement, and reference overlap as potential false negatives;
- writes a deterministic pair-level audit sample.

The hardness score is a machine diagnostic combining query/candidate lexical
overlap, hierarchy proximity, and candidate rank.  It is compared with a
deterministic random-corpus baseline; this is not a semantic correctness
proof.

## Artifacts and exit gate

The output directory contains:

```text
hard_negatives.jsonl   # one query and 2–5 explicit resolved negatives
rejections.jsonl       # query IDs that did not reach the minimum
audit_sample.jsonl     # up to 300 pair-level rows with blank manual fields
manifest.json          # hashes, config, sources, metrics, and gate diagnostics
```

The script returns status `NEEDS_MANUAL_AUDIT` when the machine checks pass.
That status is intentional: TASK 10 is not formally passed until the audit
sample confirms that all pairs are valid negatives and the estimated
false-negative rate is at most 3%.  Unknown candidates, positive leakage,
incomplete query coverage, or a hardness distribution no harder than random
produce `FAIL` and exit code 2.

Because TASK 09 was provisionally waived without its manual audit, TASK 10
outputs based on those records remain provisional and must not be promoted to
the retriever-training entry gate until both audits are completed.
