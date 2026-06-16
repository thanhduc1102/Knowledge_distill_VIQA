"""Phase-A honest retrieval signals (training-free).

Two surface-form normalisers that make the BM25 backbone robust to the failure
modes the EDA quantified, *without* using any gold metadata:

  * ``normalize`` — abbreviation / full-form unification (root cause #5).
  * ``self_query`` — alias-robust company detection from the QUESTION text only
    (root cause #4 same-company confusion), with an honest no-op when the
    question carries no company (the ~20-36% of questions the user flagged).
"""

from gsr_cacl.retrieval.normalize import (
    concept_sentinels,
    expand_bm25_tokens,
)
from gsr_cacl.retrieval.self_query import CompanyIndex

__all__ = [
    "concept_sentinels",
    "expand_bm25_tokens",
    "CompanyIndex",
]
