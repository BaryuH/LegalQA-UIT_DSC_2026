"""Answer-selection benchmark for Vietnamese Legal RAG-QA.

The MBR / pairwise-METEOR-matrix line was evaluated and dropped (never beat
``longest``, length_residual <= 0). What ships:
- ``metrics``            deterministic METEOR/ROUGE-L gold scoring
- ``grounding``          grounding + refusal gate
- ``selection``          production selector + reference baselines
- ``production_selector`` self-contained ``select_final_answer`` for the pipeline
- ``generation``         single-GPU HF candidate sampler
- ``runner``             end-to-end benchmark orchestration
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
