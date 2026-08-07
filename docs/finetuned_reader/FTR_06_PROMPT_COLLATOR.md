# FTR-06 — Prompt and Collator

**Status:** PASS offline

The generative prompt builder has separate train/inference templates with a
shared question/evidence prefix. Inference accepts only question plus packed
evidence and rejects target placeholders. The answer-only collator masks prompt
and padding labels with `-100`, appends exactly one EOS, and raises an explicit
`TARGET_DOES_NOT_FIT` error instead of truncating a target.

Acceptance coverage is in `tests/test_finetuned_reader_generative.py` and the
offline generative self-check. No chain-of-thought field or answer is included
in inference prompts or inference artifacts.
