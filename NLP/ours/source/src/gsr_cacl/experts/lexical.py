"""LexicalExpert — sparse BM25 retrieval with abbreviation/full-form unification.

Independent method #1. Sparse exact-term IR (Robertson & Zaragoza, BM25). The only
addition over textbook BM25 is the Phase-A concept sentinel (``retrieval.normalize``):
"EBITDA" and "earnings before interest..." collapse to one token, fixing the abbreviation
mismatch (EDA #5). This is the honest backbone and the natural pool seeder.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.experts.base import Expert
from gsr_cacl.retrieval.normalize import concept_sentinels

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "was", "is", "are", "what",
         "how", "much", "many", "did", "does", "do", "as", "by", "on", "at", "year", "fiscal",
         "report", "reported", "company", "value", "during", "between", "change", "total"}


def _toks(text: str) -> list[str]:
    return [t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


class LexicalExpert(Expert):
    name = "lexical"
    is_retriever = True

    def __init__(self, abbr_expand: bool = True):
        self.abbr_expand = abbr_expand

    def _doc_tokens(self, text: str) -> list[str]:
        base = _toks(text)
        return base + concept_sentinels(text) if self.abbr_expand else base

    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([self._doc_tokens(d.page_content) for d in corpus])
        self._n = len(corpus)

    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        self._q_tokens = [self._doc_tokens(q) for q in raw_queries]
        self._cache: dict[int, np.ndarray] = {}

    def full_scores(self, qi: int) -> np.ndarray:
        s = self._cache.get(qi)
        if s is None:
            s = np.asarray(self._bm25.get_scores(self._q_tokens[qi]), dtype=np.float64)
            self._cache[qi] = s
        return s

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        return self.full_scores(qi)[np.asarray(pool, dtype=int)]
