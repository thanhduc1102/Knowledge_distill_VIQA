"""CellExpert — fine-grained table-cell match (Fact-Ledger granularity).

Independent method #5. Row/cell-level retrieval (TableRAG / FT-RAG paradigm): each table
cell becomes a ``(row-label tokens, period)`` unit, so the query can match the exact line
item in the right period instead of a linearised whole table. Uses RAW row labels (covers
~100% of facts) rather than canonical concepts (which the small ontology maps for ~14%),
so it is the broad-coverage structural signal. Targets context-sharing (EDA #7) — a single
document serving many questions is disambiguated at the cell, not the document, level.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.datasets.gsr_document import extract_table
from gsr_cacl.experts.base import Expert
from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.ledger.numeric import extract_years

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "was", "is", "are", "what",
         "how", "much", "many", "did", "does", "do", "as", "by", "on", "at", "year", "fiscal",
         "report", "reported", "company", "value", "during", "between", "change", "total"}


def _tokset(text: str) -> set[str]:
    return {t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0


def _doc_cells(page_content: str):
    tmd = extract_table(page_content)
    if not tmd:
        return set(extract_years(page_content)), []
    L = extract_ledger_from_table(tmd)
    yrs, cells = set(), []
    for f in L.facts:
        p = None
        if f.period:
            try:
                p = int(float(str(f.period))); yrs.add(p)
            except (ValueError, TypeError):
                pass
        cells.append((_tokset(f.concept), p))
    if not yrs:
        yrs = set(extract_years(tmd))
    return yrs, cells


class CellExpert(Expert):
    name = "cell"
    is_retriever = False

    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        dc = [_doc_cells(d.page_content) for d in corpus]
        self._doc_years = [a for a, _ in dc]
        self._doc_cells = [b for _, b in dc]

    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        self._q_content = [_tokset(q) for q in raw_queries]
        self._q_years = [set(extract_years(q)) for q in raw_queries]

    def _cell_match(self, q_content: set, q_years: set, cells, doc_years: set) -> float:
        if not q_content or not cells:
            return 0.0
        best = 0.0
        for ctoks, p in cells:
            ov = _jaccard(q_content, ctoks)
            if ov <= 0:
                continue
            pf = 1.0 if (p is not None and p in q_years) else (0.6 if (q_years & doc_years) else 0.3)
            best = max(best, ov * pf)
        return best

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        qc, qy = self._q_content[qi], self._q_years[qi]
        return np.asarray(
            [self._cell_match(qc, qy, self._doc_cells[i], self._doc_years[i]) for i in pool],
            dtype=np.float64,
        )
