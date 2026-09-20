# benchmark_meteor_matrix

Answer-selection benchmark for the Vietnamese Legal RAG-QA task.

## Status: MBR dropped, production selector locked in

The benchmark started from the pairwise-METEOR-matrix idea (Minimum Bayes Risk
consensus decoding). It was evaluated end-to-end on a grounded, length-fixed dev
slice (Qwen3-8B, AITeamVN reranker evidence, detailed prompt) and **dropped**:

- Selection headroom is tiny: oracle 0.4772 vs random 0.4126 (~0.065).
- **Every strategy has length_residual <= 0** — none picks better *content* than
  its length predicts. MBR ties `longest` (ns, p~0.27) and never beats it.
- The real levers were **retrieval (reranker, +0.22 oracle)** and **generation
  length/completeness**, not the selection algorithm.

So the O(N^2) matrix, MBR variants, embedding/ensemble/reranker utilities, CBMBR,
and MBR self-distillation were removed. What ships is a cheap O(N) selector.

## Production selector

`benchmark/production_selector.py::select_final_answer(candidates, evidence)`:
1. drop refusals and candidates asserting legal ids/dates absent from evidence;
2. return the longest survivor (METEOR is recall-weighted → length maximises
   coverage of the gold answer);
3. explicit fallback to global longest if the gate empties (never silent).

Self-contained (only `grounding` + `metrics`) so the main RAG pipeline can import
it directly.

## Layout

```
benchmark/
  repo.py              bridge to legal_rag (official METEOR/ROUGE-L)
  metrics.py           deterministic METEOR/ROUGE-L gold scoring (+ NLTK cross-check)
  grounding.py         grounding + refusal gate
  selection.py         production selector + reference baselines
  production_selector.py  select_final_answer for the main pipeline
  generation.py        single-GPU HF candidate sampler (8-bit, epsilon sampling)
  data.py              split loader (gold isolated to eval) + prompt builder
  manifest.py          fingerprints + versions
  runner.py            end-to-end orchestration
config/                default.yaml, train_dev200.yaml (recommended), train_dev500.yaml
scripts/               run_benchmark.py, bootstrap_significance.py, length_controlled_analysis.py
```

## Running

From `zip_for_huy/` on the GPU box (Qwen3-8B fits a 4090/A100-24GB in 8-bit):

```bash
pip install nltk   # optional official METEOR cross-check

python benchmark_meteor_matrix/scripts/run_benchmark.py \
    --config benchmark_meteor_matrix/config/train_dev200.yaml
```

Outputs in `outputs/<run_name>/`:
- `final_answers.jsonl` — the answer the pipeline ships (production selector);
- `per_case.jsonl` — candidates, selections, per-strategy gold scores;
- `summary.json` — per-strategy METEOR/ROUGE-L, oracle/random, health, lengths;
- `manifest.json` — model, sampling, dataset fingerprint, versions, git commit;
- `candidates.jsonl` — resumable candidate cache.

Prerequisites (built by the retrieval pipeline): a dev slice
`processed/train_dev200.json` and reranked evidence
`processed/train_dev200_rerank_evidence.jsonl` (`{id, evidence}`).

## Analysis helpers

```bash
# real baseline is 'longest', not greedy 'first'
python benchmark_meteor_matrix/scripts/bootstrap_significance.py \
    --run-dir outputs/<run> --baseline longest --iters 10000

# isolate content selection from the length confound
python benchmark_meteor_matrix/scripts/length_controlled_analysis.py \
    --run-dir outputs/<run>
```

`summary.json` is the standing yardstick for the C+B work (retrieval reranker,
evidence packing, prompt/length): measure each change as a paired METEOR delta.

## Contract notes

- `data/` is read-only; gold is loaded into `Case.gold` and read only by the
  scorer, never by prompt/generation/selection.
- Closed-book vs grounded is recorded as `grounding_mode` (no silent fallback).
- 4090 vs A100 differ only in `dtype: auto` (fp16 vs bf16); `load_in_8bit` fits
  Qwen3-8B on 24 GB.
