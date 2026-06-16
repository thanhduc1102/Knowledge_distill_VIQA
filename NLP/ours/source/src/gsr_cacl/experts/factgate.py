"""FactGate expert — fact-level gated multiplicative joint score (contribution C6).

Implements the gated typed late interaction from core_method.md §3.2:

    S(q, D) = max_f [ σ_m(q, f) · γ_t(q, f) · γ_e(q, D) ]

where
  σ_m — concept similarity: max(lexical jaccard, 0.7 · [canonical ontology hit])
  γ_t — period soft gate: exp(−min_year_delta / σ_t),  σ_t = 1.5 years
  γ_e — entity gate: company_match_score(q_company, doc_company), 1.0 when
         the query carries no company signal (neutral, not a penalty)

The core novelty vs. existing experts:

  * vs. CellExpert: multiplies γ_e (entity gate) into the per-fact score instead of
    leaving entity matching to a separate additive term in the fusion head.  A wrong-
    company document with perfect concept/period overlap is suppressed to near 0 here,
    but CellExpert would still give it a high cell score (entity only helps via the
    additive w_entity · entity_expert term in the learned head).

  * vs. EntityExpert: σ_m and γ_t directly condition the match on the query's target
    concept and period, not just company/year metadata.  EntityExpert has no per-fact
    resolution — it is document-level.

  * vs. ConceptExpert / GraphExpert: both are document-level coverage signals (does the
    document CONTAIN the concept?). FactGate checks concept+period+entity at the same
    FACT, so a document that has Revenue for 2020 but not 2021 is scored lower for a
    2021 Revenue query, even if doc-level concept coverage is 1.0.

Multiplicative gating (γ_e · max_f[σ_m · γ_t]) produces scores ∈ [0, 1]:
  - Right company, right concept, right period → ≈ 0.7 – 1.0
  - Right company, right concept, wrong period → 0.7 · exp(−Δyear/1.5)
  - Wrong company                              → 0 – 0.3 (suppressed by γ_e)
  - No concept/period match                   → 0.0
"""

from __future__ import annotations

import math
import re
from typing import Optional, Sequence

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.datasets.gsr_document import extract_table
from gsr_cacl.experts.base import Expert
from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.aliases import company_match_score
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.retrieval.self_query import CompanyIndex

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "the", "a", "an", "of", "in", "and", "or", "to", "is", "are", "was", "were",
    "for", "from", "with", "at", "by", "on", "as", "its", "that", "this", "what",
    "how", "which", "did", "do", "does", "have", "has", "had", "been", "be",
    "not", "no", "all", "year", "fiscal", "report", "reported", "company",
    "during", "between", "much", "many",
})
_SIG_T = 1.5   # period gate decay constant (years); exp(-1) ≈ 0.51 at 1.5y off


def _tokset(text: str) -> frozenset[str]:
    toks = _TOK.findall((text or "").lower())
    return frozenset(t for t in toks if t not in _STOP and not t.isdigit() and len(t) > 1)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Pre-extracted per-doc facts: (raw_label_tokset, concept_canonical | None, period_year | None)
_FactTuple = tuple[frozenset, Optional[str], Optional[int]]


def _extract_doc_facts(page_content: str) -> list[_FactTuple]:
    tmd = extract_table(page_content)
    if not tmd:
        return []
    try:
        ledger = extract_ledger_from_table(tmd)
    except Exception:
        return []
    facts: list[_FactTuple] = []
    for f in ledger.facts:
        label_toks = _tokset(f.concept or "")
        canon = f.concept_canonical
        period: Optional[int] = None
        if f.period is not None:
            try:
                period = int(float(str(f.period)))
            except (ValueError, TypeError):
                pass
        facts.append((label_toks, canon, period))
    return facts


class FactGateExpert(Expert):
    """Fact-level gated multiplicative joint score (concept × period × entity)."""

    name = "factgate"
    is_retriever = False

    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        self._doc_facts: list[list[_FactTuple]] = []
        self._doc_company: list[str] = []
        for d, m in zip(corpus, doc_metas):
            self._doc_company.append(str((m or {}).get("company_name", "") or "").strip())
            self._doc_facts.append(_extract_doc_facts(d.page_content))
        self._comp_index = CompanyIndex(list(doc_metas))

    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        self._q_data: list[tuple[Optional[str], set[int], frozenset, set[str]]] = []
        for q in raw_queries:
            q_company_key = self._comp_index.detect(q)   # canonical key or None (honest)
            q_years = set(extract_years(q))
            q_toks = _tokset(q)
            q_concepts = concepts_in_text(q)             # set of canonical concept strings
            self._q_data.append((q_company_key, q_years, q_toks, q_concepts))

    # ------------------------------------------------------------------
    def _entity_gate(self, q_company_key: Optional[str], di: int) -> float:
        if q_company_key is None:
            return 1.0   # query carries no company signal → neutral, never penalise
        doc_co = self._doc_company[di]
        if not doc_co:
            return 0.4   # doc has no company metadata → mild suppression
        return company_match_score(q_company_key, doc_co)

    def _fact_score(
        self,
        label_toks: frozenset,
        canon: Optional[str],
        period: Optional[int],
        q_toks: frozenset,
        q_concepts: set[str],
        q_years: set[int],
    ) -> float:
        # σ_m: lexical jaccard + canonical concept boost (raises 14% → 0.7 floor on exact hit)
        jac = _jaccard(q_toks, label_toks)
        onto_boost = 0.7 if (canon is not None and canon in q_concepts) else 0.0
        sigma_m = max(jac, onto_boost)
        if sigma_m == 0.0:
            return 0.0   # short-circuit: no concept match at all

        # γ_t: period soft gate
        if period is not None and q_years:
            min_delta = min(abs(period - y) for y in q_years)
            gamma_t = math.exp(-min_delta / _SIG_T)
        else:
            gamma_t = 1.0   # no period constraint expressed

        return sigma_m * gamma_t

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        q_company, q_years, q_toks, q_concepts = self._q_data[qi]
        out = np.zeros(len(pool), dtype=np.float64)
        for pi, di in enumerate(pool):
            gamma_e = self._entity_gate(q_company, di)
            facts = self._doc_facts[di]
            if not facts:
                continue
            best = max(
                self._fact_score(lt, canon, period, q_toks, q_concepts, q_years)
                for lt, canon, period in facts
            )
            out[pi] = gamma_e * best
        return out
