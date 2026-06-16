"""LateInteractionExpert — fact-level MaxSim retrieval (ColBERT-style, sentence granularity).

Independent method #7 and the lever for the recall ceiling. A single-vector dense encoder
compresses a whole table into one point, so a document answering 3+ different questions
(context-sharing — EDA #7, 92% of FinQA docs) is blurred. Late interaction keeps a SEPARATE
vector per Fact (``concept [period] = value``) and scores a document by its *most relevant
fact*:

    score(Q, D) = max_{f ∈ facts(D)}  cos(enc(Q), enc(f))          (fact-level MaxSim)

This is the sentence-granularity analogue of ColBERT's token MaxSim — it targets the exact
line item in the right period instead of the linearised whole table, and because it scores
over the whole corpus it can pull gold documents that BM25 misses (raising pool recall).

Encoder defaults to a small, fast model (``BAAI/bge-small-en-v1.5``). ``is_retriever=True``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.datasets.gsr_document import extract_table
from gsr_cacl.experts.base import Expert
from gsr_cacl.ledger.extract import extract_ledger_from_table

_BGE_QUERY_INSTR = "Represent this sentence for searching relevant passages: "


def _doc_fact_strings(page_content: str) -> list[str]:
    tmd = extract_table(page_content)
    if not tmd:
        return []
    L = extract_ledger_from_table(tmd)
    out = []
    for f in L.facts:
        r = f.render()
        if r and r.strip():
            out.append(r)
    return out


class LateInteractionExpert(Expert):
    name = "lateint"
    is_retriever = True

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None,
                 batch_size: int = 256, topm: int = 1, max_facts: int = 80):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.topm = topm                 # average of top-m facts (1 = pure MaxSim)
        self.max_facts = max_facts

    def _load(self):
        if getattr(self, "_model", None) is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.model_name, device=dev)

    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        self._load()
        # Flatten all facts; remember which doc each fact belongs to. Docs with no facts get
        # a single fallback "fact" = the raw content head so they remain scoreable.
        fact_strings: list[str] = []
        fact_doc: list[int] = []
        for di, d in enumerate(corpus):
            facts = _doc_fact_strings(d.page_content)[: self.max_facts]
            if not facts:
                facts = [(d.page_content or "")[:200]]
            for s in facts:
                fact_strings.append(s)
                fact_doc.append(di)
        self._fact_doc = np.asarray(fact_doc, dtype=np.int64)
        self._n_docs = len(corpus)
        # Facts are appended doc-by-doc → fact_doc is contiguous; record each doc's start
        # offset so per-doc max is a fast np.maximum.reduceat instead of a slow scatter .at.
        starts = np.searchsorted(self._fact_doc, np.arange(self._n_docs))
        self._doc_start = starts.astype(np.int64)
        self._fact_emb = np.asarray(self._model.encode(
            fact_strings, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, batch_size=self.batch_size), np.float32)

    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        self._q_emb = np.asarray(self._model.encode(
            [_BGE_QUERY_INSTR + q for q in raw_queries], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False, batch_size=self.batch_size), np.float32)
        self._cache: dict[int, np.ndarray] = {}

    def full_scores(self, qi: int) -> np.ndarray:
        s = self._cache.get(qi)
        if s is None:
            sims = (self._fact_emb @ self._q_emb[qi]).astype(np.float64)   # [F]
            out = np.maximum.reduceat(sims, self._doc_start)              # per-doc max (fast)
            self._cache[qi] = out
            s = out
        return s

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        return self.full_scores(qi)[np.asarray(pool, dtype=int)]
