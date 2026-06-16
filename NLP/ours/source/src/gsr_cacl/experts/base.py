"""Expert abstraction for Modular Multi-Expert Retrieval (MMER).

An Expert is an *independent* retrieval method: it owns its representation and produces
a score ``s_i(Q, D)`` for every document, and can be evaluated standalone. The learned
fusion head (``experts.fusion``) combines several experts — it never reaches inside one.

Contract
--------
``prepare(corpus, doc_metas)``         build doc-side representations once (index time).
``score_pool(qi, pool) -> np.ndarray`` raw score for each candidate doc index in ``pool``
                                       for pre-registered query ``qi`` (set via ``set_queries``).

Scores are returned RAW (each expert's natural scale); the fusion layer min-max normalises
per query so heterogeneous scales (BM25 vs cosine vs coverage) become comparable.
"""

from __future__ import annotations

import abc
from typing import Sequence

import numpy as np

from gsr_cacl.core import Document


class Expert(abc.ABC):
    name: str = "expert"
    #: True if standalone scores are meaningful over the WHOLE corpus (can seed the pool);
    #: False if the expert only *re-scores* an existing candidate pool (e.g. structural signals).
    is_retriever: bool = False

    @abc.abstractmethod
    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        ...

    @abc.abstractmethod
    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        """Register the query batch and precompute any query-side representation."""

    @abc.abstractmethod
    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        """Raw scores aligned with ``pool`` (len == len(pool))."""

    def full_scores(self, qi: int) -> np.ndarray:
        """Optional: scores over the whole corpus (only for ``is_retriever`` experts)."""
        raise NotImplementedError

    def get_candidates(self, qi: int, pool_size: int) -> list[int]:
        """Return candidate doc indices for pool seeding.

        Default: top-``pool_size`` by full_scores (argsort).
        Override in subclasses that need different selection logic (e.g.
        MetadataRetriever returns only genuinely-matching docs, ignoring pool_size).
        Only called when ``is_retriever=True`` and ``full_scores`` is implemented.
        """
        return [int(j) for j in np.argsort(-self.full_scores(qi))[:pool_size]]


def minmax(x: np.ndarray) -> np.ndarray:
    """Per-vector min-max to [0,1]; flat vector → all 0.5 (no signal)."""
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x.astype(np.float64) - lo) / (hi - lo)
