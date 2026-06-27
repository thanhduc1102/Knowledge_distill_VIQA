"""Small dependency-light IR utilities for research diagnostics.

The main retrieval code uses richer experts and optional third-party packages.  These
helpers keep the paper-facing evaluation scripts reproducible in lean environments:
pure-Python BM25, reciprocal-rank metrics, and simple score normalisation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "in", "for", "to", "and", "or", "was", "is", "are", "were",
    "what", "how", "much", "many", "did", "does", "do", "as", "by", "on", "at", "year",
    "fiscal", "report", "reported", "company", "value", "during", "between",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def minmax(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - lo) / (hi - lo)


@dataclass
class SimpleBM25:
    docs: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.n_docs = len(self.docs)
        self.lengths = np.asarray([len(d) for d in self.docs], dtype=np.float64)
        self.avgdl = float(self.lengths.mean()) if self.n_docs else 1.0
        self.tf = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {
            t: max(0.0, math.log((self.n_docs - f + 0.5) / (f + 0.5) + 1.0))
            for t, f in df.items()
        }

    @classmethod
    def from_texts(cls, texts: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> "SimpleBM25":
        return cls([tokenize(t) for t in texts], k1=k1, b=b)

    def scores(self, query: str | Iterable[str]) -> np.ndarray:
        qtok = list(query) if not isinstance(query, str) else tokenize(query)
        out = np.zeros(self.n_docs, dtype=np.float64)
        if not qtok or self.n_docs == 0:
            return out
        qterms = set(qtok)
        for i, counts in enumerate(self.tf):
            dl = self.lengths[i] if i < len(self.lengths) else 0.0
            s = 0.0
            for t in qterms:
                f = counts.get(t, 0)
                if f <= 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1.0)) / max(denom, 1e-9)
            out[i] = s
        return out


def rank_of_gold(order: Sequence[int], gold: int) -> int:
    for r, idx in enumerate(order):
        if int(idx) == int(gold):
            return r
    return 10**9


def ranking_metrics(ranks: Sequence[int]) -> dict[str, float]:
    ranks = list(ranks)
    if not ranks:
        return {"MRR@3": 0.0, "R@1": 0.0, "R@3": 0.0, "R@5": 0.0, "NDCG@3": 0.0}
    return {
        "MRR@3": round(float(np.mean([1.0 / (r + 1) if r < 3 else 0.0 for r in ranks])), 4),
        "R@1": round(float(np.mean([r < 1 for r in ranks])), 4),
        "R@3": round(float(np.mean([r < 3 for r in ranks])), 4),
        "R@5": round(float(np.mean([r < 5 for r in ranks])), 4),
        "NDCG@3": round(float(np.mean([1.0 / np.log2(r + 2) if r < 3 else 0.0 for r in ranks])), 4),
    }
