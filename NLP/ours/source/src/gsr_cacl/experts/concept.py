"""ConceptExpert — query-conditioned accounting-concept coverage (ontology C2/C3).

Independent method #4. Grounded in the accounting ontology (42 IFRS/GAAP canonical
concepts + 7 accounting identities, ``ontology.concepts``). Asks a query-conditioned
question — "does this table carry the CONCEPT and PERIOD the query needs?" — via
``scoring.concept_coverage`` (with derivable-concept expansion through the identities).
Re-scorer only (meaningless over the whole corpus), so ``is_retriever=False``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.experts.base import Expert
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.scoring.concept_coverage import concept_coverage_score


def _doc_concepts(page_content: str):
    # Direct text scan achieves ~94% coverage vs <5% via ledger canonical mapping.
    # Real FinQA row labels are too verbose to match the 42-concept alias list.
    concepts = concepts_in_text(page_content)
    periods = set(extract_years(page_content))
    return concepts, periods


class ConceptExpert(Expert):
    name = "concept"
    is_retriever = False

    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        dc = [_doc_concepts(d.page_content) for d in corpus]
        self._doc_concepts = [a for a, _ in dc]
        self._doc_periods = [b for _, b in dc]

    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        self._q_concepts = [concepts_in_text(q) for q in raw_queries]
        self._q_periods = [set(extract_years(q)) for q in raw_queries]

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        qc, qp = self._q_concepts[qi], self._q_periods[qi]
        return np.asarray(
            [concept_coverage_score(qc, qp, self._doc_concepts[i], self._doc_periods[i]) for i in pool],
            dtype=np.float64,
        )
