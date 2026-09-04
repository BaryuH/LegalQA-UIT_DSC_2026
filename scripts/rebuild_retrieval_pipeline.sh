#!/usr/bin/env bash
# ============================================================================
# SEDAR Retrieval Pipeline — Full Rebuild after Parser Blank-Line Fix
# ============================================================================
#
# Run on: GPU server (r6639@131, RTX 4090 24GB)
# Pre-requisites:
#   - Git HEAD has the parser fixes (5ea21a7 or later)
#   - conda activate legalqa
#   - Python 3.11+
#
# Rebuilds retrieval artifacts against the parser_blankline_20260901
# corpus/views. Corpus, views and document-scoped labels must already exist.
# It does not train LTR and does not run the reader.
#
# Usage:
#   chmod +x scripts/rebuild_retrieval_pipeline.sh
#   ./scripts/rebuild_retrieval_pipeline.sh
#
# Offline Qwen snapshot (host has no outbound network):
#   export QWEN_MODEL_PATH=/path/to/models--Qwen--Qwen3-Embedding-4B/snapshots/<sha>
#   ./scripts/rebuild_retrieval_pipeline.sh
#   QWEN_MODEL stays the logical model name, so the manifest and the index
#   cache fingerprint are unchanged; only the loader reads from the directory.
#
# Resume a partially completed run (skips finished steps, keeps the same dir):
#   export RUN_TAG=rebuild_parser_blankline_20260901_20260902_101500
#   ./scripts/rebuild_retrieval_pipeline.sh
#
# Force a full rebuild of every step:
#   SKIP_EXISTING=0 ./scripts/rebuild_retrieval_pipeline.sh
#
# Dense memory controls (safe defaults for a 24 GB GPU):
#   DENSE_BATCH_SIZE=1 DENSE_MAX_SEQ_LENGTH=4096 \
#     ./scripts/rebuild_retrieval_pipeline.sh
#
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

WORK_ROOT="${WORK_ROOT:-/mnt/G/sedar-legalqa}"
PROJ_ROOT="${PROJECT_ROOT:-/mnt/G/LegalQA-UIT_DSC_2026}"
ART_ROOT="${ART_ROOT:-${WORK_ROOT}/artifacts/sedar_retrieval}"

# Corpus paths (already built on server)
CORPUS_TAG="${CORPUS_TAG:-parser_blankline_20260901}"
NODES="${ART_ROOT}/canonical/${CORPUS_TAG}/nodes.jsonl"
PASSAGES_R2A="${ART_ROOT}/views/${CORPUS_TAG}/passages_r2a.jsonl"

# Label paths (already built on server)
SILVER_LABELS="${SILVER_LABELS:-${ART_ROOT}/eval/silver_r2a_warmup500_${CORPUS_TAG}.jsonl}"

# Questions
QUESTIONS="${PROJ_ROOT}/data/warmup.json"

# Qwen model.
#   QWEN_MODEL      logical model name — goes into the manifest, the index cache
#                   fingerprint and the dense source-name contract. Do not
#                   change it to a filesystem path: run_dense_retrieval.py
#                   validates it against the 'dense' source contract.
#   QWEN_MODEL_PATH optional local snapshot directory used only for loading.
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-Embedding-4B}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${QWEN_MODEL_DIR:-}}"
QWEN_REVISION="${QWEN_REVISION:-5cf2132abc99cad020ac570b19d031efec650f2b}"
QWEN_LOCAL_FILES_ONLY="${QWEN_LOCAL_FILES_ONLY:-1}"

# Existing clean-warmup champion used as the recall-audit metadata anchor.
CHAMPION_RUN="${CHAMPION_RUN:-${WORK_ROOT}/outputs/task20/task20_ltr_clean460_repro_20260830}"
CHAMPION_EVAL="${CHAMPION_EVAL:-${WORK_ROOT}/artifacts/sedar_sft/validation/eval/val01_task20_ltr_clean460_repro_20260830}"

# Output tag for this run. Override RUN_TAG to resume an existing run dir.
RUN_TAG="${RUN_TAG:-rebuild_${CORPUS_TAG}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ART_ROOT}/runs/${RUN_TAG}"

# Skip a step when its output already exists (resume-friendly). 0 = redo all.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Retrieval parameters
BM25_TOP_K="${BM25_TOP_K:-500}"
DENSE_TOP_K="${DENSE_TOP_K:-500}"
# BM25@500 ∪ Qwen@500 can contain up to 1000 unique IDs.
FUSION_CAP="${FUSION_CAP:-1000}"
# Keep 4 in the cutoff list: it is the evidence-pack cutoff and the reader's
# upper bound. Every earlier report omitted it.
EVAL_CUTOFFS="${EVAL_CUTOFFS:-4,10,20,50,100,500}"
AUDIT_CUTOFFS="${AUDIT_CUTOFFS:-4,10,20,50,100,200,500}"
# Qwen3-Embedding-4B can exhaust a 24 GB GPU with batch 8 at 8192 tokens.
# Keep the full sequence length by default and lower the batch first; callers
# can lower DENSE_MAX_SEQ_LENGTH if a long passage still exceeds the budget.
DENSE_BATCH_SIZE="${DENSE_BATCH_SIZE:-1}"
DENSE_QUERY_BATCH_SIZE="${DENSE_QUERY_BATCH_SIZE:-32}"
DENSE_MAX_SEQ_LENGTH="${DENSE_MAX_SEQ_LENGTH:-8192}"

for dense_setting in \
    "${DENSE_BATCH_SIZE}" \
    "${DENSE_QUERY_BATCH_SIZE}" \
    "${DENSE_MAX_SEQ_LENGTH}"; do
    if ! [[ "${dense_setting}" =~ ^[1-9][0-9]*$ ]]; then
        echo "FATAL: dense batch sizes and max sequence length must be positive integers"
        exit 1
    fi
done

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/src"

mkdir -p "${RUN_DIR}"

# Log this run into its own directory as well as stdout.
LOG_FILE="${RUN_DIR}/rebuild_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

DENSE_LOCAL_ONLY_ARGS=()
if [[ "${QWEN_LOCAL_FILES_ONLY}" == "1" ]]; then
    DENSE_LOCAL_ONLY_ARGS+=(--local-files-only)
elif [[ "${QWEN_LOCAL_FILES_ONLY}" != "0" ]]; then
    echo "FATAL: QWEN_LOCAL_FILES_ONLY must be 0 or 1"
    exit 1
fi

QWEN_PATH_ARGS=()
if [[ -n "${QWEN_MODEL_PATH}" ]]; then
    if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
        echo "FATAL: QWEN_MODEL_PATH is not a directory: ${QWEN_MODEL_PATH}"
        exit 1
    fi
    QWEN_PATH_ARGS+=(--model-path "${QWEN_MODEL_PATH}")
fi

# Guard against the historical mistake of pointing QWEN_MODEL at a directory:
# the manifest and the dense source contract both key on the logical name.
if [[ "${QWEN_MODEL}" == /* || -d "${QWEN_MODEL}" ]]; then
    echo "FATAL: QWEN_MODEL must be the logical model name, not a path."
    echo "       Use QWEN_MODEL_PATH for a local snapshot directory."
    exit 1
fi

step_done () {   # $1 = path that proves the step finished
    [[ "${SKIP_EXISTING}" == "1" && -e "$1" ]]
}

# ── Preflight checks ──────────────────────────────────────────────────────

echo "================================================================"
echo "PREFLIGHT CHECKS"
echo "================================================================"

echo "Git HEAD:   $(git rev-parse --short HEAD)"
echo "Python:     $(python --version 2>&1)"
echo "CUDA:       $(python -c 'import torch; print(f"available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}")' 2>&1)"
echo "Corpus:     ${PASSAGES_R2A}"
echo "Labels:     ${SILVER_LABELS}"
echo "Qwen name:  ${QWEN_MODEL}"
echo "Qwen path:  ${QWEN_MODEL_PATH:-<hub cache>}"
echo "Revision:   ${QWEN_REVISION}"
echo "Champion:   ${CHAMPION_RUN}"
echo "Run tag:    ${RUN_TAG}"
echo "Run dir:    ${RUN_DIR}"
echo "Skip done:  ${SKIP_EXISTING}"
echo "Dense batch: ${DENSE_BATCH_SIZE}"
echo "Dense query batch: ${DENSE_QUERY_BATCH_SIZE}"
echo "Dense max sequence: ${DENSE_MAX_SEQ_LENGTH}"
echo "Log:        ${LOG_FILE}"
echo ""

python - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"FATAL: Python 3.11+ required, got {sys.version}")
PY

# Verify critical files exist
for f in \
    "${NODES}" \
    "${PASSAGES_R2A}" \
    "${SILVER_LABELS}" \
    "${QUESTIONS}" \
    "${CHAMPION_RUN}/config.json" \
    "${CHAMPION_RUN}/run_summary.json" \
    "${CHAMPION_RUN}/retrieval.jsonl" \
    "${CHAMPION_RUN}/predictions.jsonl" \
    "${CHAMPION_EVAL}/metrics.json"; do
    if [ ! -f "$f" ]; then
        echo "FATAL: Missing file: $f"
        exit 1
    fi
done
echo "All input files verified."

# Fail fast on the encoder BEFORE the expensive index build. Loading the model
# is where the offline snapshot problem shows up; discovering that after a long
# BM25 build wastes a lot of time.
echo ""
echo "Preflight: dense encoder smoke load"
QWEN_MODEL="${QWEN_MODEL}" \
QWEN_MODEL_PATH="${QWEN_MODEL_PATH}" \
QWEN_REVISION="${QWEN_REVISION}" \
QWEN_LOCAL_FILES_ONLY="${QWEN_LOCAL_FILES_ONLY}" \
DENSE_MAX_SEQ_LENGTH="${DENSE_MAX_SEQ_LENGTH}" \
python - <<'PY'
import os
import sys

from sentence_transformers import SentenceTransformer

name = os.environ["QWEN_MODEL"]
path = os.environ.get("QWEN_MODEL_PATH", "").strip()
revision = os.environ["QWEN_REVISION"]
local_only = os.environ["QWEN_LOCAL_FILES_ONLY"] == "1"

kwargs = {"device": "cuda", "trust_remote_code": True, "local_files_only": local_only}
if path:
    target = path
    print(f"loading local snapshot: {target}")
else:
    target = name
    kwargs["revision"] = revision
    print(f"loading via hub cache: {target}@{revision}")

try:
    model = SentenceTransformer(target, **kwargs)
except Exception as exc:  # noqa: BLE001 - preflight must report any failure
    print(f"FATAL: dense encoder failed to load: {type(exc).__name__}: {exc}")
    print("")
    print("If the hub-cache path failed, inventory the FUNCTIONAL files in the")
    print("snapshot (config.json, config_sentence_transformers.json,")
    print("modules.json, sentence_bert_config.json, 1_Pooling/config.json,")
    print("tokenizer*, model*.safetensors) and then set QWEN_MODEL_PATH to that")
    print("directory. '.gitattributes' and 'generation_config.json' are NOT")
    print("required to encode: an embedding model never calls .generate().")
    sys.exit(1)

model.max_seq_length = int(os.environ["DENSE_MAX_SEQ_LENGTH"])
vectors = model.encode(
    ["Điều 76 quy định về hợp đồng lao động"],
    batch_size=1,
    convert_to_numpy=True,
)
if vectors.shape[0] != 1 or float(vectors.std()) <= 0.0:
    raise SystemExit(f"FATAL: degenerate embedding, shape={vectors.shape}")
print(f"encoder OK, dim={vectors.shape[1]}")
PY
echo ""

# Save run metadata
cat > "${RUN_DIR}/run_config.json" <<METAEOF
{
  "run_tag": "${RUN_TAG}",
  "corpus_tag": "${CORPUS_TAG}",
  "git_commit": "$(git rev-parse HEAD)",
  "passages_r2a": "${PASSAGES_R2A}",
  "silver_labels": "${SILVER_LABELS}",
  "depth": {
    "bm25_top_k": ${BM25_TOP_K},
    "dense_top_k": ${DENSE_TOP_K},
    "fusion_union_cap": ${FUSION_CAP},
    "eval_cutoffs": "${EVAL_CUTOFFS}",
    "audit_cutoffs": "${AUDIT_CUTOFFS}"
  },
  "qwen_model": "${QWEN_MODEL}",
  "qwen_model_path": "${QWEN_MODEL_PATH}",
  "qwen_revision": "${QWEN_REVISION}",
  "champion_run": "${CHAMPION_RUN}",
  "champion_eval": "${CHAMPION_EVAL}",
  "dense_batch_size": ${DENSE_BATCH_SIZE},
  "dense_query_batch_size": ${DENSE_QUERY_BATCH_SIZE},
  "dense_max_seq_length": ${DENSE_MAX_SEQ_LENGTH},
  "skip_existing": ${SKIP_EXISTING},
  "timestamp": "$(date -Iseconds)"
}
METAEOF

# ── Step 1: Rebuild BM25 Index ────────────────────────────────────────────

echo "================================================================"
echo "STEP 1/7: Rebuild BM25 Index"
echo "================================================================"

BM25_CACHE="${ART_ROOT}/indexes/bm25"
BM25_MANIFEST="${RUN_DIR}/bm25_index_manifest.json"

if step_done "${BM25_MANIFEST}"; then
    echo "SKIP: ${BM25_MANIFEST} already exists"
else
    python scripts/sedar_retrieval/build_bm25_index.py \
        --passages "${PASSAGES_R2A}" \
        --cache-root "${BM25_CACHE}" \
        --manifest-out "${BM25_MANIFEST}" \
        --smoke-query "Điều 76" \
        --top-k 10
fi

echo "BM25 index manifest: ${BM25_MANIFEST}"
echo ""

# ── Step 2: Rebuild Qwen Dense Index ──────────────────────────────────────

echo "================================================================"
echo "STEP 2/7: Rebuild Qwen Dense Index"
echo "================================================================"

DENSE_OUTPUT="${RUN_DIR}/dense_index"

if step_done "${DENSE_OUTPUT}/manifest.json"; then
    echo "SKIP: ${DENSE_OUTPUT}/manifest.json already exists"
else
    python scripts/sedar_retrieval/build_dense_index.py \
        --passages "${PASSAGES_R2A}" \
        --output-dir "${DENSE_OUTPUT}" \
        --model "${QWEN_MODEL}" \
        --model-revision "${QWEN_REVISION}" \
        "${QWEN_PATH_ARGS[@]+"${QWEN_PATH_ARGS[@]}"}" \
        --input-format qwen_instruction \
        --device cuda \
        --dtype bf16 \
        --batch-size "${DENSE_BATCH_SIZE}" \
        --shard-size 4096 \
        --max-seq-length "${DENSE_MAX_SEQ_LENGTH}" \
        --smoke-query "Điều 76" \
        --top-k 10 \
        "${DENSE_LOCAL_ONLY_ARGS[@]+"${DENSE_LOCAL_ONLY_ARGS[@]}"}" \
        --force
fi

echo "Dense index: ${DENSE_OUTPUT}"
echo ""

# ── Step 3: Export BM25 Top-500 Rankings ──────────────────────────────────

echo "================================================================"
echo "STEP 3/7: Export BM25 Top-${BM25_TOP_K} Rankings"
echo "================================================================"

BM25_RETRIEVAL="${RUN_DIR}/bm25_warmup_top${BM25_TOP_K}.jsonl"

if step_done "${BM25_RETRIEVAL}"; then
    echo "SKIP: ${BM25_RETRIEVAL} already exists"
else
    python scripts/sedar_retrieval/run_bm25_retrieval.py \
        --passages "${PASSAGES_R2A}" \
        --cache-root "${BM25_CACHE}" \
        --questions "${QUESTIONS}" \
        --split warmup \
        --top-k "${BM25_TOP_K}" \
        --output "${BM25_RETRIEVAL}"
fi

echo "BM25 retrieval: ${BM25_RETRIEVAL}"
echo ""

# ── Step 4: Export Qwen Dense Top-500 Rankings ────────────────────────────

echo "================================================================"
echo "STEP 4/7: Export Qwen Dense Top-${DENSE_TOP_K} Rankings"
echo "================================================================"

DENSE_RETRIEVAL="${RUN_DIR}/qwen_warmup_top${DENSE_TOP_K}.jsonl"

if step_done "${DENSE_RETRIEVAL}"; then
    echo "SKIP: ${DENSE_RETRIEVAL} already exists"
else
    python scripts/sedar_retrieval/run_dense_retrieval.py \
        --index-dir "${DENSE_OUTPUT}" \
        --passages "${PASSAGES_R2A}" \
        --questions "${QUESTIONS}" \
        --split warmup \
        --source-name dense \
        --device cuda \
        --batch-size "${DENSE_QUERY_BATCH_SIZE}" \
        --top-k "${DENSE_TOP_K}" \
        "${QWEN_PATH_ARGS[@]+"${QWEN_PATH_ARGS[@]}"}" \
        "${DENSE_LOCAL_ONLY_ARGS[@]+"${DENSE_LOCAL_ONLY_ARGS[@]}"}" \
        --output "${DENSE_RETRIEVAL}"
fi

# Alias so downstream docs/commands that expect dense_* find the same file.
DENSE_ALIAS="${RUN_DIR}/dense_warmup_top${DENSE_TOP_K}.jsonl"
if [ ! -e "${DENSE_ALIAS}" ]; then
    ln -s "$(basename "${DENSE_RETRIEVAL}")" "${DENSE_ALIAS}"
fi

echo "Dense retrieval: ${DENSE_RETRIEVAL}"
echo "Dense alias:     ${DENSE_ALIAS}"
echo ""

# ── Step 4b: Validate ranking files before fusion ─────────────────────────

echo "================================================================"
echo "STEP 4b/7: Validate ranking alignment"
echo "================================================================"

BM25_RETRIEVAL="${BM25_RETRIEVAL}" \
DENSE_RETRIEVAL="${DENSE_RETRIEVAL}" \
BM25_TOP_K="${BM25_TOP_K}" \
DENSE_TOP_K="${DENSE_TOP_K}" \
python - <<'PY'
import json
import os
import sys


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in rows:
                raise SystemExit(f"FATAL: duplicate query_id {query_id!r} in {path}")
            ranked = [str(x) for x in row["ranked_ids"]]
            if len(set(ranked)) != len(ranked):
                raise SystemExit(
                    f"FATAL: duplicate passage_id inside query {query_id!r} in {path}"
                )
            rows[query_id] = ranked
    if not rows:
        raise SystemExit(f"FATAL: no rows in {path}")
    return rows


bm25 = load(os.environ["BM25_RETRIEVAL"])
dense = load(os.environ["DENSE_RETRIEVAL"])
bm25_k = int(os.environ["BM25_TOP_K"])
dense_k = int(os.environ["DENSE_TOP_K"])

if set(bm25) != set(dense):
    only_bm25 = sorted(set(bm25) - set(dense))[:5]
    only_dense = sorted(set(dense) - set(bm25))[:5]
    raise SystemExit(
        "FATAL: BM25/dense query sets differ. "
        f"bm25-only e.g. {only_bm25}; dense-only e.g. {only_dense}"
    )

short_bm25 = sum(1 for ranked in bm25.values() if len(ranked) < bm25_k)
short_dense = sum(1 for ranked in dense.values() if len(ranked) < dense_k)
print(
    json.dumps(
        {
            "queries": len(bm25),
            "bm25_queries_below_top_k": short_bm25,
            "dense_queries_below_top_k": short_dense,
            "recall_at_top_k_is_lower_bound": bool(short_bm25 or short_dense),
        },
        ensure_ascii=False,
    )
)
if short_bm25 or short_dense:
    print(
        "WARNING: some queries returned fewer than top-k results; recall at the "
        "deepest cutoff is a lower bound, not an exact value.",
        file=sys.stderr,
    )
PY
echo ""

# ── Step 5: Fuse Candidates (Union + Weighted RRF) ───────────────────────

echo "================================================================"
echo "STEP 5/7: Fuse Candidates"
echo "================================================================"

# 5a: Pure union (for recall audit)
FUSED_UNION="${RUN_DIR}/fused_union_top${FUSION_CAP}.jsonl"

if step_done "${FUSED_UNION}"; then
    echo "SKIP: ${FUSED_UNION} already exists"
else
    python scripts/sedar_retrieval/fuse_candidates.py \
        --bm25 "${BM25_RETRIEVAL}" \
        --dense "${DENSE_RETRIEVAL}" \
        --fusion-method union \
        --union-cap "${FUSION_CAP}" \
        --output "${FUSED_UNION}" \
        --force
fi

echo "Fused union: ${FUSED_UNION}"

# 5b: Weighted RRF (for downstream LTR/reader)
#     Weights are 1.0/1.0 here as the historical control. Dense recall is
#     materially higher than BM25, so the weighting itself is a tuning target
#     — see Phase 5 in docs/sedar_retrieval/RETRIEVAL_FIX_PLAN.md.
FUSED_RRF="${RUN_DIR}/fused_rrf_top${FUSION_CAP}.jsonl"

if step_done "${FUSED_RRF}"; then
    echo "SKIP: ${FUSED_RRF} already exists"
else
    python scripts/sedar_retrieval/fuse_candidates.py \
        --bm25 "${BM25_RETRIEVAL}" \
        --dense "${DENSE_RETRIEVAL}" \
        --fusion-method weighted_rrf \
        --rrf-k 60 \
        --union-cap "${FUSION_CAP}" \
        --bm25-weight 1.0 \
        --dense-weight 1.0 \
        --output "${FUSED_RRF}" \
        --force
fi

echo "Fused RRF: ${FUSED_RRF}"
echo ""

# ── Step 6: Evaluate Retrieval Metrics ────────────────────────────────────

echo "================================================================"
echo "STEP 6/7: Evaluate Retrieval Metrics (level-aware)"
echo "================================================================"
echo "--passages is mandatory: without it every level recall silently reports"
echo "0.0 and only exact passage_id matches count. Read article_recall_at and"
echo "the article_expanded bundle, not the top-level exact recall."
echo ""

eval_one () {   # $1 = ranking jsonl, $2 = output metrics json, $3 = label
    echo "-- ${3}"
    python scripts/sedar_retrieval/eval_retrieval.py \
        --pred "$1" \
        --labels "${SILVER_LABELS}" \
        --passages "${PASSAGES_R2A}" \
        --cutoffs "${EVAL_CUTOFFS}" \
        --output "$2" \
        --force
}

BM25_METRICS="${RUN_DIR}/metrics_bm25.json"
DENSE_METRICS="${RUN_DIR}/metrics_dense.json"
UNION_METRICS="${RUN_DIR}/metrics_fused_union.json"
RRF_METRICS="${RUN_DIR}/metrics_fused_rrf.json"

eval_one "${BM25_RETRIEVAL}" "${BM25_METRICS}"  "BM25"
eval_one "${DENSE_RETRIEVAL}" "${DENSE_METRICS}" "Qwen dense"
eval_one "${FUSED_UNION}"    "${UNION_METRICS}" "Fused union"
eval_one "${FUSED_RRF}"      "${RRF_METRICS}"   "Fused RRF"
echo ""

# ── Step 7: Recall Audit ─────────────────────────────────────────────────

echo "================================================================"
echo "STEP 7/7: Retrieval Recall Audit"
echo "================================================================"

RECALL_AUDIT="${RUN_DIR}/recall_audit.json"

python scripts/sedar_retrieval/audit_retrieval_recall.py \
    --run-dir "${CHAMPION_RUN}" \
    --metrics "${CHAMPION_EVAL}/metrics.json" \
    --bm25 "${BM25_RETRIEVAL}" \
    --qwen "${DENSE_RETRIEVAL}" \
    --labels "${SILVER_LABELS}" \
    --passages "${PASSAGES_R2A}" \
    --cutoffs "${AUDIT_CUTOFFS}" \
    --scope-anchor-only \
    --output "${RECALL_AUDIT}" \
    --force

echo "Recall audit: ${RECALL_AUDIT}"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────

echo "================================================================"
echo "REBUILD COMPLETE"
echo "================================================================"
echo ""
echo "Run directory: ${RUN_DIR}"
echo ""
echo "Key outputs:"
echo "  BM25 index manifest:  ${BM25_MANIFEST}"
echo "  Dense index:          ${DENSE_OUTPUT}/manifest.json"
echo "  BM25 retrieval:       ${BM25_RETRIEVAL}"
echo "  Dense retrieval:      ${DENSE_RETRIEVAL}"
echo "  Fused union:          ${FUSED_UNION}"
echo "  Fused RRF:            ${FUSED_RRF}"
echo "  BM25 metrics:         ${BM25_METRICS}"
echo "  Dense metrics:        ${DENSE_METRICS}"
echo "  Union metrics:        ${UNION_METRICS}"
echo "  RRF metrics:          ${RRF_METRICS}"
echo "  Recall audit:         ${RECALL_AUDIT}"
echo "  Log:                  ${LOG_FILE}"
echo ""

echo "── Baseline table (article-level, the decision-grade numbers) ──"
RUN_DIR="${RUN_DIR}" python - <<'PY'
import json
import os
import pathlib

run_dir = pathlib.Path(os.environ["RUN_DIR"])
systems = [
    ("BM25", "metrics_bm25.json"),
    ("Qwen dense", "metrics_dense.json"),
    ("Fused union", "metrics_fused_union.json"),
    ("Fused RRF", "metrics_fused_rrf.json"),
]
cuts = ["4", "10", "20", "100", "500"]
header = f"{'system':<14}" + "".join(f"{'art@' + c:>10}" for c in cuts)
header += f"{'MRR@10*':>10}{'nDCG@10*':>10}"
print(header)
print("-" * len(header))
for label, name in systems:
    path = run_dir / name
    if not path.exists():
        print(f"{label:<14}{'(missing)':>10}")
        continue
    m = json.loads(path.read_text(encoding="utf-8"))
    art = m.get("article_recall_at", {})
    expanded = m.get("article_expanded", {})
    row = f"{label:<14}"
    for c in cuts:
        value = art.get(c)
        row += f"{value:>10.4f}" if isinstance(value, (int, float)) else f"{'-':>10}"
    for key in ("mrr_at_10", "ndcg_at_10"):
        value = expanded.get(key)
        row += f"{value:>10.4f}" if isinstance(value, (int, float)) else f"{'-':>10}"
    print(row)
print("")
print("* MRR/nDCG come from the article_expanded bundle: evaluate_retrieval only")
print("  ever scores mrr/ndcg against relevant_ids, so the expansion is what")
print("  makes them article-level. Top-level mrr/ndcg stay exact-passage.")
print("")
print("Read the gap between art@4 and art@500: that is the entire headroom of")
print("the rank + pack stages, and it is the number the fix plan is built on.")
PY
echo ""
echo "Next: docs/sedar_retrieval/RETRIEVAL_FIX_PLAN.md — Phase 1 gate."
