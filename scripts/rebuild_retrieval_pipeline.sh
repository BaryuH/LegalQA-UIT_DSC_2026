#!/usr/bin/env bash
# ============================================================================
# SEDAR Retrieval Pipeline — Full Rebuild after Parser Blank-Line Fix
# ============================================================================
#
# Run on: GPU server (r6639@131, RTX 4090 24GB)
# Pre-requisites:
#   - Git HEAD is 5ea21a7 (or later with parser fixes)
#   - conda activate legalqa
#   - Python 3.11+
#
# This script rebuilds all retrieval artifacts against the new
# parser_blankline_20260901 corpus/views. It does NOT rebuild the corpus
# or views (those are already done). It starts from step 3 in the
# activeContext ordering.
#
# Usage:
#   chmod +x rebuild_retrieval_pipeline.sh
#   ./rebuild_retrieval_pipeline.sh 2>&1 | tee rebuild_$(date +%Y%m%d_%H%M%S).log
#
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

PROJ_ROOT="/mnt/G/LegalQA-UIT_DSC_2026"
ART_ROOT="/mnt/G/sedar-legalqa/artifacts/sedar_retrieval"

# Corpus paths (already built on server)
CORPUS_TAG="parser_blankline_20260901"
NODES="${ART_ROOT}/canonical/${CORPUS_TAG}/nodes.jsonl"
PASSAGES_R2A="${ART_ROOT}/views/${CORPUS_TAG}/passages_r2a.jsonl"

# Label paths (already built on server)
SILVER_LABELS="${ART_ROOT}/eval/silver_r2a_warmup500_${CORPUS_TAG}.jsonl"

# Clean warmup manifest (for recall audit)
CLEAN_MANIFEST="${PROJ_ROOT}/artifacts/sedar_sft/validation/clean_warmup_manifest.json"

# Questions
QUESTIONS="${PROJ_ROOT}/data/warmup.json"

# Qwen model
QWEN_MODEL="Qwen/Qwen3-Embedding-4B"
QWEN_REVISION="7568a60f24a415e3597a74e423728272c929eb0b"

# Output tag for this run
RUN_TAG="rebuild_${CORPUS_TAG}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ART_ROOT}/runs/${RUN_TAG}"

# Retrieval parameters
BM25_TOP_K=500
DENSE_TOP_K=500
FUSION_CAP=500
LTR_TOP_K=150

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/src"

# ── Preflight checks ──────────────────────────────────────────────────────

echo "================================================================"
echo "PREFLIGHT CHECKS"
echo "================================================================"

echo "Git HEAD: $(git rev-parse --short HEAD)"
echo "Python:   $(python --version 2>&1)"
echo "CUDA:     $(python -c 'import torch; print(f"available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}")' 2>&1)"
echo "Corpus:   ${PASSAGES_R2A}"
echo "Labels:   ${SILVER_LABELS}"
echo "Run tag:  ${RUN_TAG}"
echo ""

# Verify critical files exist
for f in "${NODES}" "${PASSAGES_R2A}" "${SILVER_LABELS}" "${QUESTIONS}"; do
    if [ ! -f "$f" ]; then
        echo "FATAL: Missing file: $f"
        exit 1
    fi
done
echo "All input files verified."

# Create run directory
mkdir -p "${RUN_DIR}"
echo "Run directory: ${RUN_DIR}"
echo ""

# Save run metadata
cat > "${RUN_DIR}/run_config.json" <<METAEOF
{
  "run_tag": "${RUN_TAG}",
  "corpus_tag": "${CORPUS_TAG}",
  "git_commit": "$(git rev-parse HEAD)",
  "passages_r2a": "${PASSAGES_R2A}",
  "silver_labels": "${SILVER_LABELS}",
  "bm25_top_k": ${BM25_TOP_K},
  "dense_top_k": ${DENSE_TOP_K},
  "fusion_cap": ${FUSION_CAP},
  "qwen_model": "${QWEN_MODEL}",
  "qwen_revision": "${QWEN_REVISION}",
  "timestamp": "$(date -Iseconds)"
}
METAEOF

# ── Step 1: Rebuild BM25 Index ────────────────────────────────────────────

echo "================================================================"
echo "STEP 1/7: Rebuild BM25 Index"
echo "================================================================"

BM25_CACHE="${ART_ROOT}/indexes/bm25"
BM25_MANIFEST="${RUN_DIR}/bm25_index_manifest.json"

python scripts/sedar_retrieval/build_bm25_index.py \
    --passages "${PASSAGES_R2A}" \
    --cache-root "${BM25_CACHE}" \
    --manifest-out "${BM25_MANIFEST}" \
    --smoke-query "Điều 76" \
    --top-k 10

echo "BM25 index manifest: ${BM25_MANIFEST}"
echo ""

# ── Step 2: Rebuild Qwen Dense Index ──────────────────────────────────────

echo "================================================================"
echo "STEP 2/7: Rebuild Qwen Dense Index"
echo "================================================================"

DENSE_OUTPUT="${RUN_DIR}/dense_index"

python scripts/sedar_retrieval/build_dense_index.py \
    --passages "${PASSAGES_R2A}" \
    --output-dir "${DENSE_OUTPUT}" \
    --model "${QWEN_MODEL}" \
    --model-revision "${QWEN_REVISION}" \
    --input-format qwen_instruction \
    --device cuda \
    --dtype bf16 \
    --batch-size 8 \
    --shard-size 4096 \
    --max-seq-length 8192 \
    --smoke-query "Điều 76" \
    --top-k 10 \
    --force

echo "Dense index: ${DENSE_OUTPUT}"
echo ""

# ── Step 3: Export BM25 Top-500 Rankings ──────────────────────────────────

echo "================================================================"
echo "STEP 3/7: Export BM25 Top-${BM25_TOP_K} Rankings"
echo "================================================================"

BM25_RETRIEVAL="${RUN_DIR}/bm25_warmup_top${BM25_TOP_K}.jsonl"

python scripts/sedar_retrieval/run_bm25_retrieval.py \
    --passages "${PASSAGES_R2A}" \
    --cache-root "${BM25_CACHE}" \
    --questions "${QUESTIONS}" \
    --split warmup \
    --top-k "${BM25_TOP_K}" \
    --output "${BM25_RETRIEVAL}"

echo "BM25 retrieval: ${BM25_RETRIEVAL}"
echo ""

# ── Step 4: Export Qwen Dense Top-500 Rankings ────────────────────────────

echo "================================================================"
echo "STEP 4/7: Export Qwen Dense Top-${DENSE_TOP_K} Rankings"
echo "================================================================"

DENSE_RETRIEVAL="${RUN_DIR}/qwen_warmup_top${DENSE_TOP_K}.jsonl"

python scripts/sedar_retrieval/run_dense_retrieval.py \
    --index-dir "${DENSE_OUTPUT}" \
    --passages "${PASSAGES_R2A}" \
    --questions "${QUESTIONS}" \
    --split warmup \
    --source-name dense \
    --device cuda \
    --batch-size 32 \
    --top-k "${DENSE_TOP_K}" \
    --output "${DENSE_RETRIEVAL}"

echo "Dense retrieval: ${DENSE_RETRIEVAL}"
echo ""

# ── Step 5: Fuse Candidates (Union + Weighted RRF) ───────────────────────

echo "================================================================"
echo "STEP 5/7: Fuse Candidates"
echo "================================================================"

# 5a: Pure union (for recall audit)
FUSED_UNION="${RUN_DIR}/fused_union_top${FUSION_CAP}.jsonl"

python scripts/sedar_retrieval/fuse_candidates.py \
    --bm25 "${BM25_RETRIEVAL}" \
    --dense "${DENSE_RETRIEVAL}" \
    --fusion-method union \
    --union-cap "${FUSION_CAP}" \
    --output "${FUSED_UNION}" \
    --force

echo "Fused union: ${FUSED_UNION}"

# 5b: Weighted RRF (for downstream LTR/reader)
FUSED_RRF="${RUN_DIR}/fused_rrf_top${FUSION_CAP}.jsonl"

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

echo "Fused RRF: ${FUSED_RRF}"
echo ""

# ── Step 6: Evaluate Retrieval Metrics ────────────────────────────────────

echo "================================================================"
echo "STEP 6/7: Evaluate Retrieval Metrics"
echo "================================================================"

# 6a: BM25-only metrics
BM25_METRICS="${RUN_DIR}/metrics_bm25.json"
python scripts/sedar_retrieval/eval_retrieval.py \
    --pred "${BM25_RETRIEVAL}" \
    --labels "${SILVER_LABELS}" \
    --output "${BM25_METRICS}"
echo "BM25 metrics: ${BM25_METRICS}"

# 6b: Dense-only metrics
DENSE_METRICS="${RUN_DIR}/metrics_dense.json"
python scripts/sedar_retrieval/eval_retrieval.py \
    --pred "${DENSE_RETRIEVAL}" \
    --labels "${SILVER_LABELS}" \
    --output "${DENSE_METRICS}"
echo "Dense metrics: ${DENSE_METRICS}"

# 6c: Fused union metrics
UNION_METRICS="${RUN_DIR}/metrics_fused_union.json"
python scripts/sedar_retrieval/eval_retrieval.py \
    --pred "${FUSED_UNION}" \
    --labels "${SILVER_LABELS}" \
    --output "${UNION_METRICS}"
echo "Union metrics: ${UNION_METRICS}"

# 6d: Fused RRF metrics
RRF_METRICS="${RUN_DIR}/metrics_fused_rrf.json"
python scripts/sedar_retrieval/eval_retrieval.py \
    --pred "${FUSED_RRF}" \
    --labels "${SILVER_LABELS}" \
    --output "${RRF_METRICS}"
echo "RRF metrics: ${RRF_METRICS}"
echo ""

# ── Step 7: Recall Audit ─────────────────────────────────────────────────

echo "================================================================"
echo "STEP 7/7: Retrieval Recall Audit"
echo "================================================================"

RECALL_AUDIT="${RUN_DIR}/recall_audit.json"

python scripts/sedar_retrieval/audit_retrieval_recall.py \
    --run-dir "${RUN_DIR}" \
    --metrics "${RRF_METRICS}" \
    --bm25 "${BM25_RETRIEVAL}" \
    --qwen "${DENSE_RETRIEVAL}" \
    --labels "${SILVER_LABELS}" \
    --passages "${PASSAGES_R2A}" \
    --cutoffs "10,20,50,100,200,500" \
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
echo ""
echo "Next steps:"
echo "  1. Compare recall curves with pre-correction baseline"
echo "  2. If recall improved: proceed to LTR retrain + frozen reader E2E"
echo "  3. If recall flat: investigate query processing or ranker issues"
echo ""

# Print metrics summary
echo "── Metrics Summary ──"
echo ""
echo "BM25:"
python -c "import json; m=json.load(open('${BM25_METRICS}')); print(f'  recall@10={m[\"recall_at\"][\"10\"]:.4f}  recall@50={m[\"recall_at\"][\"50\"]:.4f}  mrr@10={m[\"mrr_at_10\"]:.4f}')"
echo "Dense (Qwen):"
python -c "import json; m=json.load(open('${DENSE_METRICS}')); print(f'  recall@10={m[\"recall_at\"][\"10\"]:.4f}  recall@50={m[\"recall_at\"][\"50\"]:.4f}  mrr@10={m[\"mrr_at_10\"]:.4f}')"
echo "Fused Union:"
python -c "import json; m=json.load(open('${UNION_METRICS}')); print(f'  recall@10={m[\"recall_at\"][\"10\"]:.4f}  recall@50={m[\"recall_at\"][\"50\"]:.4f}  mrr@10={m[\"mrr_at_10\"]:.4f}')"
echo "Fused RRF:"
python -c "import json; m=json.load(open('${RRF_METRICS}')); print(f'  recall@10={m[\"recall_at\"][\"10\"]:.4f}  recall@50={m[\"recall_at\"][\"50\"]:.4f}  mrr@10={m[\"mrr_at_10\"]:.4f}')"
echo ""
echo "Done."
