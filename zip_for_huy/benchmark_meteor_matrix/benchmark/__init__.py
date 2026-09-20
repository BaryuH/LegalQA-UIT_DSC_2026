"""MBR / METEOR-matrix answer-selection benchmark for Vietnamese Legal RAG-QA.

Public surface:
- ``metrics``    deterministic METEOR/ROUGE-L (utility + gold scoring)
- ``utilities``  MBR utilities: lexical, embedding, ensemble, QE prior
- ``selection``  strategies: MBR, pruned MBR, weighted MBR, aggregation, CBMBR
- ``generation`` single-GPU HF candidate sampler (epsilon sampling)
- ``runner``     end-to-end benchmark orchestration
- ``distill``    MBR self-distillation dataset builder
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
