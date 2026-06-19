"""LocalLexicalExpert (``loclex``) — pool-local IDF re-scorer (C2 / Conditional Salience).

Motivation (the core domain insight)
------------------------------------
Standard BM25 estimates IDF over the WHOLE corpus (~2.7k docs). But once the candidate
pool has been narrowed to the chunks of a *single company filing*, the discriminative
terms are those that are rare **within that company's chunks** — the specific note /
table topic ("compensation", "net operating loss carryforwards", "Asia Pacific").
Boilerplate shared by every chunk of a filing (the company name, "for the year ended
December 31, 2019", generic accounting vocabulary) carries ~0 local information yet is
still credited by global IDF, blurring the within-filing ranking.

``LocalLexicalExpert`` recomputes BM25 with **document frequency measured on the pool
itself**. A term occurring in nearly every pooled chunk → local IDF ≈ 0; a term in only
1–2 chunks → dominates. This is a pure re-scorer (``is_retriever=False``): it never seeds
the pool, it only re-ranks an existing one.

This operationalises *conditional salience*: a term's discriminativeness is defined
*relative to the retrieved cluster*, not the global corpus. Empirically (see
``docs/RESEARCH_AAAI27.md`` §2.3) loclex LOSES to global BM25 at corpus scale but WINS
inside a company-scoped pool — the signal is unlocked only by clustering.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.experts.base import Expert
from gsr_cacl.experts.lexical import _toks
from gsr_cacl.retrieval.normalize import concept_sentinels


class LocalLexicalExpert(Expert):
    """BM25 with IDF re-estimated over the candidate pool (company-local) at score time."""

    name = "loclex"
    is_retriever = False  # pure re-scorer; relies on another expert to seed the pool

    def __init__(self, abbr_expand: bool = True, k1: float = 1.5, b: float = 0.75):
        self.abbr_expand = abbr_expand
        self.k1 = k1
        self.b = b

    def _doc_tokens(self, text: str) -> list[str]:
        base = _toks(text)
        return base + concept_sentinels(text) if self.abbr_expand else base

    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        # Cache per-doc tokens once; pool-local BM25 is rebuilt per query (pools are tiny).
        self._doc_tok: list[list[str]] = [self._doc_tokens(d.page_content) for d in corpus]
        self._n = len(corpus)

    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        self._q_tok: list[list[str]] = [self._doc_tokens(q) for q in raw_queries]

    def _pool_bm25(self, qtok: list[str], pool: Sequence[int]) -> np.ndarray:
        """BM25 scores for the query over the pool, with df/idf computed on the pool only."""
        docs = [self._doc_tok[d] for d in pool]
        N = len(docs)
        if N == 0:
            return np.zeros(0, dtype=np.float64)
        lengths = np.array([len(d) for d in docs], dtype=np.float64)
        avgdl = float(lengths.mean()) if lengths.mean() > 0 else 1.0

        # Pool-local document frequency for the query terms only.
        qterms = set(qtok)
        df: dict[str, int] = {t: 0 for t in qterms}
        tf: list[dict[str, int]] = []
        for d in docs:
            counts: dict[str, int] = {}
            seen: set[str] = set()
            for tok in d:
                if tok in qterms:
                    counts[tok] = counts.get(tok, 0) + 1
                    seen.add(tok)
            tf.append(counts)
            for t in seen:
                df[t] += 1

        # Robertson-Sparck-Jones idf with +1 smoothing (>=0); pool-local.
        idf = {
            t: max(0.0, np.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0))
            for t in qterms
        }

        scores = np.zeros(N, dtype=np.float64)
        for i, counts in enumerate(tf):
            dl = lengths[i]
            s = 0.0
            for t, f in counts.items():
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                s += idf[t] * (f * (self.k1 + 1.0)) / denom if denom > 0 else 0.0
            scores[i] = s
        return scores

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        return self._pool_bm25(self._q_tok[qi], list(pool))
