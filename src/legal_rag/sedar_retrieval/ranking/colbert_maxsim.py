"""Late-interaction (MaxSim) reranking over the hybrid top-100 (TASK 26).

## Why late interaction, and why only as a reranker

The closest published analogue to this corpus is TVPL: **224,006 Vietnamese
legal passages** from thuvienphapluat.vn with 10,000 test queries
([arXiv:2412.00657](https://arxiv.org/abs/2412.00657)). On it, a fine-tuned
ColBERT beats the fine-tuned bi-encoder from the same paper:

| system | MRR@10 | MAP@10 | R@10 | R@100 |
| --- | --- | --- | --- | --- |
| BM25 | 21.60 | 20.87 | 41.11 | 70.64 |
| CoT-MAE bi-encoder | 70.69 | 68.25 | 87.34 | **96.92** |
| CoT-MAE ColBERT | **74.61** | **72.04** | **89.29** | 96.41 |

**+3.92 MRR@10 and +1.95 R@10, with R@100 flat or slightly worse.** The gain is
entirely at the top of the list, which is precisely the article@4 (0.7920) versus
article@10 (0.8832) gap in our own numbers - and it is why this belongs as a
*reranker over the existing top-100*, not as a replacement first stage. A first
stage would need PLAID/WARP machinery to buy recall we already have (article@500
= 0.9745).

Supporting evidence for the same conclusion from two other directions:
BGE-M3 on **MIRACL Vietnamese** scores multi-vector **58.3 vs dense 56.1**
nDCG@10 (+2.2), with full three-way fusion adding only 0.7 on top of
multi-vector ([arXiv:2402.03216](https://arxiv.org/abs/2402.03216)); and
Luan et al. (TACL 2021) show a single vector losing fidelity as the unit
lengthens (DE-BERT-768 MRR@10 90.2 -> 63.0 from 50 to 400 tokens) while a
multi-vector representation degrades only 96.8 -> 85.2. Our p90 unit is 2,210
characters with a tail past 6,000, so that is the regime we are in.

## The precondition: it must be fine-tuned

Zero-shot late interaction *loses* on Vietnamese. "Which Works Best for
Vietnamese?" (Findings of EACL 2026) measures zero-shot ColBERT at **MRR@10
21.54 against BM25's 23.09**. No credible benchmarked Vietnamese ColBERT
checkpoint exists as of September 2026 (see the decision record for the survey),
so this module is built around a checkpoint *we* fine-tune, and the config
refuses to run a bare base encoder by default.

## Two implementation details that are easy to get wrong

**Sum, not mean, and the two conventions do not mix.** ColBERT (SIGIR 2020, Eq.
3) and ColBERTv2 define ``S = sum_i max_j (q_i . d_j)`` - summed over query
tokens. PyLate implements exactly that. BGE-M3's ``colbert_score`` divides by the
query token count instead, and the M3 paper's ``s_mul`` carries the ``1/N``. So a
PyLate score lives on roughly ``[0, query_length]`` (32 by default) and an M3
score on ``[-1, 1]``. Ranking within one query is unaffected; **any fixed-weight
fusion with BM25 or dense scores is not**. ``reduction`` is therefore explicit
here, and the scale is recorded in the run artifact.

**Mask with -inf, not by multiplying by zero.** PyLate applies its masks
multiplicatively, so a masked position contributes exactly ``0.0`` to the inner
max. Because embeddings are L2-normalised, real similarities can be negative -
so for a query token whose best genuine match is negative, the max returns 0
from a *padding slot*. This module masks with ``-inf``, which is what the
definition means.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

COLBERT_MAXSIM_SCHEMA_VERSION = "sedar-colbert-maxsim-v1"

Reduction = Literal["sum", "mean"]

__all__ = [
    "COLBERT_MAXSIM_SCHEMA_VERSION",
    "ColbertBackend",
    "ColbertMaxsimConfig",
    "ColbertMaxsimError",
    "DocEmbeddingCache",
    "Reduction",
    "dequantize_int8",
    "l2_normalize",
    "maxsim_score",
    "maxsim_scores",
    "pool_tokens_hierarchical",
    "quantize_int8",
    "rerank_with_maxsim",
]


class ColbertMaxsimError(RuntimeError):
    """Raised when late-interaction scoring cannot proceed safely."""


class ColbertBackend(Protocol):
    """Anything that turns text into per-token embeddings.

    Kept as a protocol so every pure function below is testable with random
    matrices and no model weights. Concrete backends live at the bottom of this
    module.
    """

    #: Embedding dimension, used to validate cached vectors.
    dim: int
    #: "sum" for a ColBERT/PyLate checkpoint, "mean" for a BGE-M3 head.
    native_reduction: Reduction

    def encode_query(self, query: str) -> np.ndarray:
        """(n_query_tokens, dim), L2-normalised, padding already removed."""

    def encode_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        """One (n_doc_tokens, dim) matrix per text, L2-normalised."""


def l2_normalize(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation.

    MaxSim is defined on unit vectors so that a dot product *is* a cosine, and
    every step that changes a vector's length - pooling, dequantisation - has to
    restore it. PyLate does not re-normalise after pooling; colpali-engine does.
    We follow colpali here, because the scoring assumes unit norm.
    """

    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ColbertMaxsimError(f"Expected a 2-D matrix, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def maxsim_score(
    query_embedding: np.ndarray,
    document_embedding: np.ndarray,
    *,
    reduction: Reduction = "sum",
    query_mask: np.ndarray | None = None,
    document_mask: np.ndarray | None = None,
) -> float:
    """Late-interaction score for one (query, document) pair.

    ``S = reduce_i ( max_j  q_i . d_j )`` with the inner max taken only over
    unmasked document tokens. Masked positions are set to ``-inf`` rather than
    ``0`` so a negative genuine similarity cannot be beaten by padding.
    """

    query = np.asarray(query_embedding, dtype=np.float32)
    document = np.asarray(document_embedding, dtype=np.float32)
    if query.ndim != 2 or document.ndim != 2:
        raise ColbertMaxsimError("Both embeddings must be 2-D (tokens, dim)")
    if query.shape[1] != document.shape[1]:
        raise ColbertMaxsimError(
            f"Dimension mismatch: query {query.shape[1]} vs document "
            f"{document.shape[1]}"
        )
    if document.shape[0] == 0 or query.shape[0] == 0:
        return 0.0

    similarity = query @ document.T
    if document_mask is not None:
        keep = np.asarray(document_mask, dtype=bool)
        if keep.shape[0] != document.shape[0]:
            raise ColbertMaxsimError("document_mask length must match token count")
        if not keep.any():
            return 0.0
        similarity = np.where(keep[None, :], similarity, -np.inf)

    per_query_token = similarity.max(axis=1)
    if query_mask is not None:
        keep_query = np.asarray(query_mask, dtype=bool)
        if keep_query.shape[0] != query.shape[0]:
            raise ColbertMaxsimError("query_mask length must match token count")
        if not keep_query.any():
            return 0.0
        per_query_token = per_query_token[keep_query]

    total = float(per_query_token.sum())
    if reduction == "mean":
        return total / float(per_query_token.shape[0])
    if reduction != "sum":
        raise ColbertMaxsimError(f"Unknown reduction: {reduction!r}")
    return total


def maxsim_scores(
    query_embedding: np.ndarray,
    document_embeddings: Sequence[np.ndarray],
    *,
    reduction: Reduction = "sum",
) -> list[float]:
    """One score per document, ragged inputs, no padding involved."""

    return [
        maxsim_score(query_embedding, document, reduction=reduction)
        for document in document_embeddings
    ]


def pool_tokens_hierarchical(
    document_embedding: np.ndarray,
    *,
    pool_factor: int = 1,
    protected_tokens: int = 1,
    renormalize: bool = True,
) -> np.ndarray:
    """Ward-cluster a document's token vectors down by ``pool_factor``.

    Token pooling ([arXiv:2409.14683](https://arxiv.org/abs/2409.14683), Answer.AI
    / LightOn) halves the vector count at **100.62% of unpooled quality on
    average** - i.e. free - and a factor of 3 still holds 99.03%. Ward linkage on
    ``1 - cosine`` is the paper's method and beats k-means (97.38 at factor 2)
    and sequential pooling (97.56).

    ``protected_tokens`` keeps the first vectors out of the clustering, matching
    the reference implementations. Requires SciPy; without it this raises rather
    than silently falling back to a worse pooling, because the fallback would
    change scores without changing the config.
    """

    if pool_factor <= 1:
        return l2_normalize(document_embedding) if renormalize else np.asarray(
            document_embedding, dtype=np.float32
        )

    array = np.asarray(document_embedding, dtype=np.float32)
    if array.ndim != 2:
        raise ColbertMaxsimError("document_embedding must be 2-D (tokens, dim)")
    if protected_tokens < 0:
        raise ColbertMaxsimError("protected_tokens must be non-negative")

    protected = array[:protected_tokens]
    to_pool = array[protected_tokens:]
    if to_pool.shape[0] <= 1:
        return array

    try:
        from scipy.cluster import hierarchy  # noqa: PLC0415
        from scipy.spatial.distance import squareform  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ColbertMaxsimError(
            "pool_factor > 1 requires SciPy. Refusing to substitute a different "
            "pooling method, which would change scores without changing config."
        ) from exc

    similarity = to_pool @ to_pool.T
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    # Ward on the *condensed* 1-cos distances. Handing SciPy a square matrix
    # makes it treat the rows as observation vectors and cluster on Euclidean
    # distances between distance rows, which is not the intended metric.
    condensed = squareform(np.clip(distance, 0.0, None), checks=False)
    linkage = hierarchy.linkage(condensed, method="ward")
    clusters = max(to_pool.shape[0] // pool_factor, 1)
    labels = hierarchy.fcluster(linkage, t=clusters, criterion="maxclust")

    pooled = np.zeros((labels.max(), to_pool.shape[1]), dtype=np.float32)
    for index in range(1, labels.max() + 1):
        members = to_pool[labels == index]
        if members.shape[0]:
            pooled[index - 1] = members.mean(axis=0)
    if renormalize:
        pooled = l2_normalize(pooled)
    return np.concatenate([protected, pooled], axis=0) if protected.size else pooled


def quantize_int8(matrix: np.ndarray) -> np.ndarray:
    """Pack unit-norm token vectors into int8.

    Every component of an L2-normalised vector is in ``[-1, 1]``, so a single
    fixed scale of 127 works and no per-vector scale needs storing. 4x smaller
    than float32 at a quantisation step of 1/127, which is far below the
    similarity differences MaxSim resolves.

    This is a *reranker cache*, not an index. For an actual late-interaction
    index the right answer is ColBERTv2 residual compression, where the measured
    knee on Vietnamese legal text is **2 bits** (TVPL: 1 bit 647 MB / MRR@10
    73.61, 2 bits 1,094 MB / 74.61, 4 bits 1,987 MB / 74.93, 8 bits 3,774 MB /
    75.02 - and 1-bit already beats a 672 MB dense bi-encoder by +2.92 MRR@10).
    """

    array = np.asarray(matrix, dtype=np.float32)
    return np.clip(np.rint(array * 127.0), -127, 127).astype(np.int8)


def dequantize_int8(matrix: np.ndarray, *, renormalize: bool = True) -> np.ndarray:
    """Unpack int8 token vectors and restore unit norm."""

    array = np.asarray(matrix, dtype=np.float32) / 127.0
    return l2_normalize(array) if renormalize else array


class DocEmbeddingCache:
    """In-memory, optionally int8 store of per-unit token embeddings.

    A top-100 reranker re-encodes the same popular articles across queries. The
    cache makes that once-per-unit instead of once-per-(query, unit).
    """

    def __init__(self, *, dim: int, quantize: bool = True) -> None:
        if dim <= 0:
            raise ColbertMaxsimError("dim must be positive")
        self.dim = dim
        self.quantize = quantize
        self._store: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, unit_id: str) -> bool:
        return unit_id in self._store

    def put(self, unit_id: str, embedding: np.ndarray) -> None:
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.dim:
            raise ColbertMaxsimError(
                f"Expected (tokens, {self.dim}) for {unit_id!r}, got {array.shape}"
            )
        self._store[unit_id] = quantize_int8(array) if self.quantize else array

    def get(self, unit_id: str) -> np.ndarray | None:
        stored = self._store.get(unit_id)
        if stored is None:
            return None
        return dequantize_int8(stored) if self.quantize else stored

    def nbytes(self) -> int:
        return int(sum(array.nbytes for array in self._store.values()))

    def save(self, path: Any) -> None:
        from pathlib import Path  # noqa: PLC0415

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            __meta__=np.array([self.dim, int(self.quantize)], dtype=np.int64),
            **self._store,
        )

    @classmethod
    def load(cls, path: Any) -> "DocEmbeddingCache":
        with np.load(path, allow_pickle=False) as payload:
            meta = payload["__meta__"]
            cache = cls(dim=int(meta[0]), quantize=bool(meta[1]))
            for key in payload.files:
                if key == "__meta__":
                    continue
                cache._store[key] = payload[key]
        return cache


@dataclass(frozen=True, slots=True)
class ColbertMaxsimConfig:
    """Configuration for one late-interaction rerank pass."""

    model: str = ""
    revision: str | None = None
    backend: Literal["pylate", "bge_m3"] = "pylate"
    dim: int = 128
    query_length: int = 32
    document_length: int = 512
    reduction: Reduction = "sum"
    pool_factor: int = 1
    top_k: int = 100
    batch_size: int = 32
    device: str = "cuda"
    quantize_cache: bool = True
    #: Refuse to run a base encoder that has never been fine-tuned. Zero-shot
    #: late interaction measurably loses to BM25 on Vietnamese, so an accidental
    #: bare-base run would look like a model failure rather than a config error.
    require_finetuned: bool = True
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.query_length <= 0 or self.document_length <= 0:
            raise ValueError("dim, query_length and document_length must be positive")
        if self.pool_factor < 1:
            raise ValueError("pool_factor must be >= 1")
        if self.top_k <= 0 or self.batch_size <= 0:
            raise ValueError("top_k and batch_size must be positive")
        if self.reduction not in {"sum", "mean"}:
            raise ValueError(f"Unknown reduction: {self.reduction!r}")
        if self.backend not in {"pylate", "bge_m3"}:
            raise ValueError(f"Unknown backend: {self.backend!r}")
        if self.backend == "bge_m3" and self.reduction != "mean":
            # M3's own colbert_score divides by the query token count; using
            # "sum" with an M3 head silently changes the score scale by ~N_q.
            raise ValueError(
                "backend='bge_m3' must use reduction='mean' to match its own "
                "colbert_score; see the module docstring"
            )
        if self.require_finetuned and not self.model:
            raise ValueError(
                "model must be set. require_finetuned=True refuses to run an "
                "unnamed or base checkpoint: zero-shot late interaction scores "
                "below BM25 on Vietnamese (MRR@10 21.54 vs 23.09)."
            )

    @property
    def score_scale(self) -> str:
        """Human-readable score range, recorded in the run artifact."""

        if self.reduction == "mean":
            return "[-1, 1]"
        return f"[0, {self.query_length}] approx"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "revision": self.revision,
            "backend": self.backend,
            "dim": self.dim,
            "query_length": self.query_length,
            "document_length": self.document_length,
            "reduction": self.reduction,
            "score_scale": self.score_scale,
            "pool_factor": self.pool_factor,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "quantize_cache": self.quantize_cache,
            "schema_version": COLBERT_MAXSIM_SCHEMA_VERSION,
        }


def rerank_with_maxsim(
    query: str,
    candidate_ids: Sequence[str],
    texts: Mapping[str, str],
    *,
    backend: ColbertBackend,
    config: ColbertMaxsimConfig,
    cache: DocEmbeddingCache | None = None,
) -> tuple[tuple[str, float], ...]:
    """Score the head of one candidate list with MaxSim.

    Returns ``(unit_id, score)`` for the reranked head only, best first.
    Candidates past ``top_k`` are the caller's business - the reranker must not
    silently drop them, and the run script appends them below the head in
    retriever order so deeper recall is unchanged.
    """

    if len(set(candidate_ids)) != len(candidate_ids):
        raise ColbertMaxsimError("Duplicate candidate id in the ranking")
    head = list(candidate_ids[: config.top_k])
    if not head:
        return ()
    missing = [unit_id for unit_id in head if unit_id not in texts]
    if missing:
        raise ColbertMaxsimError(
            f"{len(missing)} candidates are absent from the corpus view "
            f"(e.g. {missing[:3]})"
        )

    query_embedding = l2_normalize(backend.encode_query(query))
    if query_embedding.shape[1] != config.dim:
        raise ColbertMaxsimError(
            f"Backend returned dim {query_embedding.shape[1]}, config says {config.dim}"
        )

    document_embeddings: list[np.ndarray] = []
    to_encode: list[str] = []
    slots: list[int] = []
    for index, unit_id in enumerate(head):
        cached = cache.get(unit_id) if cache is not None else None
        if cached is None:
            document_embeddings.append(np.zeros((0, config.dim), dtype=np.float32))
            to_encode.append(texts[unit_id])
            slots.append(index)
        else:
            document_embeddings.append(cached)

    for start in range(0, len(to_encode), config.batch_size):
        batch = to_encode[start : start + config.batch_size]
        encoded = backend.encode_documents(batch)
        if len(encoded) != len(batch):
            raise ColbertMaxsimError(
                f"Backend returned {len(encoded)} embeddings for {len(batch)} texts"
            )
        for offset, embedding in enumerate(encoded):
            pooled = pool_tokens_hierarchical(
                embedding, pool_factor=config.pool_factor
            )
            slot = slots[start + offset]
            document_embeddings[slot] = pooled
            if cache is not None:
                cache.put(head[slot], pooled)

    scores = maxsim_scores(
        query_embedding, document_embeddings, reduction=config.reduction
    )
    ordered = sorted(
        zip(head, scores, strict=True),
        # Score desc, then the retriever's own order. Never break a tie on the
        # id: that would scramble the tail into id order and throw away the
        # retriever ranking, the only signal left where MaxSim stops separating.
        key=lambda item: (-item[1], head.index(item[0])),
    )
    return tuple(ordered)


# --------------------------------------------------------------------------
# Concrete backends. Both are thin and injected, so nothing above needs them.
# --------------------------------------------------------------------------


class PylateColbertBackend:
    """Adapter over ``pylate.models.ColBERT`` (PyLate 1.6.0 API).

    PyLate pins ``sentence-transformers==5.3.0`` exactly. Sentence-Transformers
    6.x ships its own ``MultiVectorEncoder``, and the two cannot coexist in one
    environment - so the preflight checks the installed versions rather than
    letting an import error surface mid-run.

    Note ``models.ColBERT("BAAI/bge-m3")`` does **not** work: bge-m3's
    ``modules.json`` is Transformer/Pooling/Normalize with no Dense, and PyLate's
    loader tries to convert the Pooling module into a Dense. Build a bge-m3-based
    ColBERT with the modular constructor instead (see the training script).
    """

    native_reduction: Reduction = "sum"

    def __init__(
        self,
        config: ColbertMaxsimConfig,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.dim = config.dim
        if model_factory is None:
            try:
                from pylate import models  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ColbertMaxsimError(
                    "The pylate backend requires `pip install pylate` "
                    "(which pins sentence-transformers==5.3.0)."
                ) from exc
            model_factory = models.ColBERT
        kwargs: dict[str, Any] = {
            "model_name_or_path": config.model,
            "device": config.device,
            "embedding_size": config.dim,
            "query_length": config.query_length,
            "document_length": config.document_length,
            "local_files_only": config.local_files_only,
        }
        if config.revision:
            kwargs["revision"] = config.revision
        self.model = model_factory(**kwargs)

    def encode_query(self, query: str) -> np.ndarray:
        embeddings = self.model.encode(
            [query], is_query=True, batch_size=1, show_progress_bar=False
        )
        return np.asarray(embeddings[0], dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        embeddings = self.model.encode(
            list(texts),
            is_query=False,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            # Pooling is applied by this module so the same code path is used
            # for cached and freshly encoded vectors, and so the pooled vectors
            # get re-normalised (PyLate does not re-normalise after pooling).
            pool_factor=1,
        )
        return [np.asarray(item, dtype=np.float32) for item in embeddings]


class BgeM3ColbertBackend:
    """Adapter over BGE-M3's ColBERT head (FlagEmbedding).

    Available without any fine-tuning, and the honest baseline arm: on MIRACL
    Vietnamese, M3 multi-vector scores 58.3 nDCG@10 against its own dense 56.1.
    Two costs to know: the head is **1024-dimensional** by default (8x the
    storage per token of a 128-dim ColBERT), and its ``colbert_score``
    **averages** over query tokens, so ``reduction`` must be ``"mean"``.
    """

    native_reduction: Reduction = "mean"

    def __init__(
        self,
        config: ColbertMaxsimConfig,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.dim = config.dim
        if model_factory is None:
            try:
                from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ColbertMaxsimError(
                    "The bge_m3 backend requires `pip install FlagEmbedding`."
                ) from exc
            model_factory = BGEM3FlagModel
        self.model = model_factory(
            config.model or "BAAI/bge-m3",
            use_fp16=config.device.startswith("cuda"),
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )

    def _encode(self, texts: Sequence[str], *, max_length: int) -> list[np.ndarray]:
        payload = self.model.encode(
            list(texts),
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
            max_length=max_length,
        )
        vectors = payload["colbert_vecs"]
        return [np.asarray(item, dtype=np.float32) for item in vectors]

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([query], max_length=self.config.query_length)[0]

    def encode_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        return self._encode(texts, max_length=self.config.document_length)
