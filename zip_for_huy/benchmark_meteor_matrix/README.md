# benchmark_meteor_matrix

Answer-selection benchmark for the Vietnamese Legal RAG-QA task, built around the
**pairwise METEOR matrix** idea: sample several candidate answers, score every
candidate against every other candidate with METEOR, and pick the one with the
highest total. That rule is exactly **Minimum Bayes Risk (MBR) consensus
decoding** with METEOR as the utility. This folder turns the idea into a
runnable, config-driven benchmark on a single GPU (4090 or A100-24GB) and
implements the 2022→2026 research line on top of it.

## Why this works (offline pre-study)

On 500 real `warmup.json` gold answers with the repo METEOR, over simulated
candidate pools:

- **Direction matters.** `M[i][j] = METEOR(ref=i, hyp=j)`; the MBR estimate of
  candidate `j` is the **column** mean. Column-MBR closes ~97% of the oracle gap;
  the row direction is *worse than random* (it collapses to picking short
  answers). Encoded in `selection.column_mean`.
- **The payoff is robustness to hallucination.** Without contamination, "pick the
  longest" ties MBR. With a fraction ρ of confident-but-wrong candidates,
  `longest` collapses (0.80→0.27) while column-MBR holds ~0.82 up to ρ=0.4 and
  only breaks once the wrong cluster becomes the majority (ρ≈0.5).
- **Cheap utility is enough.** The repo exact-token METEOR selects as well as
  official NLTK METEOR (0.891 vs 0.889), so the matrix uses the fast in-repo
  metric.
- **Naive O(N) aggregation is *not* free.** Reference aggregation degrades under
  contamination (0.82→0.62 at ρ=0.5) because it averages contaminants into the
  pseudo-reference. Efficiency must come *after* a grounding prune, or via
  contamination-robust centroid clustering (CBMBR).

These numbers motivated the design; the benchmark reproduces them with a real
generator instead of simulated pools.

## Development roadmap (2022 → 2026), and where each branch lives

| Branch | Idea | Code |
| --- | --- | --- |
| 0. Base MBR | METEOR matrix, argmax column sum | `selection.mbr` |
| 1. Better utility | neural/semantic utility beats lexical metric (Freitag 2022) | `utilities.EmbeddingUtility`, `selection.mbr` on its matrix |
| 2. Ensemble utility | blend metrics to fight reward hacking (WMT24) | `utilities.EnsembleUtility` |
| 3. Break O(N²) | reference aggregation / centroid CBMBR | `utilities.*.aggregate`, `selection.aggregate`, `selection.cbmbr` |
| 4. QE-MBR | weight pseudo-refs by a quality estimate; grounding prune | `utilities.RerankerPrior`, `selection.mbr_weighted`, `selection.mbr_pruned`, `grounding` |
| 5. Self-distillation | run MBR offline on train, fine-tune to greedy it (Finkelstein & Freitag 2024) | `distill.build_distill_dataset` |

Selected references: Eikema & Aziz 2022 (sampling-based MBR); Freitag et al. 2022
(neural-metric MBR, TACL); Freitag et al. 2023 (epsilon sampling); Bertsch et al.
2023 (MBR = reranking = self-consistency); Cheng & Vlachos 2023 (confidence
pruning); Vamvas & Sennrich 2024 (reference aggregation, `zurichnlp/mbr`);
Deguchi et al. 2024 (CBMBR / MBRS, EMNLP demo); Finkelstein & Freitag 2024 (MBR &
QE finetuning, arXiv:2309.10966); Tomani et al. 2024 (quality-aware self-estimation,
arXiv:2310.06707).

## Layout

```
benchmark/
  repo.py         bridge to legal_rag (official METEOR/ROUGE-L)
  metrics.py      fast exact METEOR (O(P+R)) + gold scoring (+ NLTK cross-check)
  utilities.py    MBR utilities: lexical, embedding, ensemble, QE prior
  selection.py    strategies: mbr, mbr_row, mbr_pruned, mbr_weighted, aggregate, cbmbr, baselines
  grounding.py    hallucination prune (unsupported legal id / date vs evidence)
  generation.py   single-GPU HF candidate sampler (epsilon sampling)
  data.py         split loader (gold isolated to eval) + prompt builder
  manifest.py     fingerprints + versions
  runner.py       end-to-end orchestration
  distill.py      MBR self-distillation dataset builder
config/           default.yaml (benchmark), distill.yaml (branch 5)
scripts/          run_benchmark.py, build_distill_dataset.py
```

## Running

From `zip_for_huy/` on the GPU box:

```bash
pip install nltk            # optional: official METEOR cross-check
# transformers / accelerate / sentence-transformers / bitsandbytes already in requirements.txt

python benchmark_meteor_matrix/scripts/run_benchmark.py \
    --config benchmark_meteor_matrix/config/default.yaml
```

Outputs land in `outputs/<run_name>/`: `per_case.jsonl` (candidates, all
selections, per-strategy gold scores), `summary.json` (per-strategy METEOR /
ROUGE-L and fraction of oracle gap closed), `manifest.json` (model, sampling,
utility/metric versions, dataset fingerprint, git commit), and
`candidates.jsonl` (reusable candidate cache).

Branch 5 dataset:

```bash
python benchmark_meteor_matrix/scripts/build_distill_dataset.py \
    --config benchmark_meteor_matrix/config/distill.yaml
```

## Contract notes

- `data/` is read-only; gold is loaded into `Case.gold` and read **only** by the
  evaluator, never by prompt/generation/utility/selection.
- Closed-book vs grounded pruning is recorded as `grounding_mode` in the manifest
  (no silent fallback).
- Self-distillation `select_mode: self` never reads gold; `oracle` reads gold to
  pick the target and is an approved training task, flagged per record.
- 4090 vs A100 differ only in `dtype: auto` (fp16 vs bf16); use `load_in_4bit`
  for 14B checkpoints on 24 GB.
- Not yet executed on GPU here — this commit is the implementation; run it on the
  4090/A100 instance to produce real numbers.
```
