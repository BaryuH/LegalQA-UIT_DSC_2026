"""Single point of contact with the parent ``legal_rag`` package.

The benchmark lives beside ``zip_for_huy/src`` but is not installed as part of
the package.  Importing this module makes ``legal_rag`` importable regardless of
the current working directory and re-exports the *official* metric primitives so
the benchmark never re-implements a second scoring convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

# zip_for_huy/benchmark_meteor_matrix/benchmark/repo.py -> parents[2] == zip_for_huy
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from legal_rag.evaluation.meteor import compute_meteor  # noqa: E402
from legal_rag.evaluation.normalization import normalize_text  # noqa: E402
from legal_rag.evaluation.rouge_l import compute_rouge_l  # noqa: E402

__all__ = ["compute_meteor", "compute_rouge_l", "normalize_text", "src_root"]


def src_root() -> Path:
    """Absolute path to ``zip_for_huy/src`` (for manifests)."""

    return _SRC
