# FTR-05 — SFT Dataset Implementation

**Status:** IMPLEMENTED; FTR-03 effective-split remediation is ready for dataset build

`src/legal_rag/finetuned_reader/dataset.py` provides deterministic, provenance-
preserving SFT construction over the frozen B2 retriever. It sorts IDs,
retrieves from `InferenceQuestion`, stores chunk/document IDs and evidence
hashes, records duplicate/blank/retrieval exclusions, and writes artifacts
outside `data/`.

`configs/finetuned_reader_train.yaml` selects the explicit,
profile-scoped `exclude_and_record` remediation
`ftr03-train-overlap-exclusion-v1`. It excludes and records 391 source train
cases whose ID or normalized question overlaps warmup/public/private, leaving
6,609 overlap-safe train cases. The builder then applies the same Unicode-safe
normalizer for intra-train deduplication (currently 14 additional records), so
the maximum pre-retrieval example count is 6,595. The default
`overlap_policy: fail` remains available for profiles without this decision.
Dataset manifests record the policy, remediation ID, and exclusion hash. No
dataset or checkpoint was generated in this task.
