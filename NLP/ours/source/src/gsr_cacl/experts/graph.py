"""GraphExpert — structural fact-graph matching (HierFinRAG-style).

Independent method #6. Where ``ConceptExpert`` scores *set overlap* (does the table contain
the concept?), GraphExpert scores the *relational structure* a query needs, by treating each
document's Fact Ledger as a concept↔period graph:

    nodes   = canonical concepts  ∪  periods
    edges   = (concept—period)     a value exists for this concept in this period
              (concept—concept)    an accounting identity links them (the 7 IDENTITIES)
              (period—period)       consecutive fiscal years (temporal adjacency)

Query intent decides which sub-structure must be present (EDA: ~38-42% of questions are
cross-period "change/growth" and ~30-69% involve ratios/percentages):

  * **temporal** ("change / increase / from X to Y / growth") → the concept must exist across
    ≥2 periods (a concept—periodA—periodB path), covering the required years.
  * **ratio** ("margin / ratio / percentage of A to B / proportion") → ≥2 concepts co-present
    in one period (a period node joining two concept nodes).
  * **lookup** (default) → the (concept, period) cell exists.

The accounting identities are finally used *correctly*: operands are matched on ROW labels
(the gen-1 KG matched columns = years → 0% hit). A required concept absent from the table but
*derivable* from present, connected operands earns partial structural credit.

Training-free, CPU, query-conditioned. Re-scorer (``is_retriever=False``).
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
from gsr_cacl.ontology.concepts import IDENTITIES, canonical_concept, concepts_in_text
from gsr_cacl.scoring.concept_coverage import expand_derivable

_TEMPORAL = re.compile(
    r"\b(change|changed|increase|increased|decrease|decreased|grow|grew|growth|"
    r"difference|differ|decline|declined|rose|fell|gain|drop|year[- ]over[- ]year|"
    r"compared|versus|vs\.?|from\s+\d{4}\s+to\s+\d{4})\b")
_RATIO = re.compile(
    r"\b(ratio|margin|percentage|percent|proportion|share of|as a (?:percent|%)|"
    r"per\b|relative to|divided by)\b")

_IDENTITY_MAP = {tgt: [c for c, _ in ops] for tgt, ops in IDENTITIES}


def _doc_struct(page_content: str):
    """concept → set(periods) map, plus the doc's full period set.

    Strategy: text-scan for concepts (94% coverage vs <5% via ledger alias),
    then refine per-concept period assignment from the table ledger when possible.
    Falls back to assigning all doc-level periods to every detected concept.
    """
    periods = set(extract_years(page_content))
    concepts_found = concepts_in_text(page_content)
    if not concepts_found:
        return {}, periods

    # Start with all-periods assignment (safe default).
    cp: dict[str, set[int]] = {c: set(periods) for c in concepts_found}

    # Try to tighten per-concept periods via ledger row labels.
    tmd = extract_table(page_content)
    if tmd:
        L = extract_ledger_from_table(tmd)
        refined: dict[str, set[int]] = {}
        for f in L.facts:
            c = canonical_concept(f.concept)
            if c and c in cp and f.period:
                try:
                    p = int(float(str(f.period)))
                    refined.setdefault(c, set()).add(p)
                    periods.add(p)
                except (ValueError, TypeError):
                    pass
        for c, ps in refined.items():
            if ps:
                cp[c] = ps

    return cp, periods


def _derivable_periods(concept: str, cp: dict[str, set[int]]) -> set[int]:
    """Periods in which `concept` can be DERIVED: intersection of its operands' periods."""
    ops = _IDENTITY_MAP.get(concept)
    if not ops:
        return set()
    sets = [cp.get(o, set()) for o in ops]
    if not sets or any(not s for s in sets):
        return set()
    out = set(sets[0])
    for s in sets[1:]:
        out &= s
    return out


def _intent(q: str) -> str:
    if _TEMPORAL.search(q.lower()):
        return "temporal"
    if _RATIO.search(q.lower()):
        return "ratio"
    return "lookup"


class GraphExpert(Expert):
    name = "graph"
    is_retriever = False

    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        s = [_doc_struct(d.page_content) for d in corpus]
        self._cp = [a for a, _ in s]
        self._periods = [b for _, b in s]

    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        self._qc = [concepts_in_text(q) for q in raw_queries]
        self._qp = [set(extract_years(q)) for q in raw_queries]
        self._intent = [_intent(q) for q in raw_queries]

    def _score(self, qc, qp, intent, cp, doc_periods) -> float:
        if not qc and not qp:
            return 0.5
        d_concepts = set(cp)
        covered = expand_derivable(d_concepts)
        concept_cov = (len(qc & covered) / len(qc)) if qc else 0.5
        if concept_cov == 0.0:
            return 0.0

        if intent == "temporal":
            rel = 0.3
            for c in (qc or d_concepts):
                ps = cp.get(c) or _derivable_periods(c, cp)
                if not ps:
                    continue
                if qp:
                    if qp <= ps:
                        rel = max(rel, 1.0)
                    elif qp & ps:
                        rel = max(rel, 0.6)
                elif len(ps) >= 2:           # change query, no explicit years → needs ≥2 periods
                    rel = max(rel, 1.0)
        elif intent == "ratio":
            target = qc or d_concepts
            per_count: dict[int, int] = {}
            for c in target:
                for p in (cp.get(c) or _derivable_periods(c, cp)):
                    per_count[p] = per_count.get(p, 0) + 1
            best = max(per_count.values(), default=0)
            rel = 1.0 if best >= 2 else (0.7 if best == 1 else 0.3)
            if qp and not (qp & set(per_count)):
                rel *= 0.6
        else:  # lookup
            rel = (1.0 if (qp & doc_periods) else 0.3) if qp else 1.0

        return concept_cov * rel

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        qc, qp, intent = self._qc[qi], self._qp[qi], self._intent[qi]
        return np.asarray(
            [self._score(qc, qp, intent, self._cp[i], self._periods[i]) for i in pool],
            dtype=np.float64,
        )
