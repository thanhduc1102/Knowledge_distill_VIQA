"""Query-conditioned structural signal: concept + period coverage (contribution C3).

The original constraint score ``CS(G_D)`` is **query-independent** — it measures whether a
table is internally consistent, which is the same number no matter what the user asked. That
is why adding γ·CS to the retrieval score does not help ranking (and even hurts): it cannot
tell *which* document answers *this* query. C3 fixes the framing.

Given the question, we extract the canonical concept(s) it asks about (e.g. "gross margin"
→ ``GrossProfit``/``GrossMarginRatio``) and the period(s) (years). A document is scored by
how well its Fact-Ledger **covers those concepts for that period**:

    s_struct(Q, D) = concept_coverage(Q, D) · (w0 + w1 · period_match(Q, D))

This directly targets the EDA's *context-sharing* failure (one document, many questions):
two questions over the same document differ in the concept/period they need, and this signal
separates them — something a single dense vector cannot. A document also "covers" a concept
it can **derive** via an accounting identity (has Revenue + CostOfRevenue ⇒ can answer
GrossProfit), which rewards tables that are *answerable* even when the exact line-item is
implicit.
"""

from __future__ import annotations

import re

from gsr_cacl.ontology.concepts import concepts_in_text, IDENTITIES

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def query_periods(question: str) -> set[int]:
    return {int(m.group(0)) for m in _YEAR_RE.finditer(question or "")}


def query_concepts(question: str) -> set[str]:
    return concepts_in_text(question or "")


def expand_derivable(concepts: set[str]) -> set[str]:
    """Add identity targets whose operands are all present (one hop of accounting derivation)."""
    out = set(concepts)
    changed = True
    while changed:                      # tiny fixpoint (identities chain: GP→OpInc→…→NetIncome)
        changed = False
        for target, operands in IDENTITIES:
            if target not in out and all(op in out for op, _ in operands):
                out.add(target)
                changed = True
    return out


def doc_periods_from_ledger(ledger) -> set[int]:
    out: set[int] = set()
    for f in ledger.facts:
        if f.period:
            try:
                out.add(int(float(str(f.period))))
            except (ValueError, TypeError):
                pass
    return out


def concept_coverage_score(
    q_concepts: set[str],
    q_periods: set[int],
    d_concepts: set[str],
    d_periods: set[int],
    *,
    w0: float = 0.4,
    w1: float = 0.6,
    use_derivable: bool = True,
    neutral: float = 0.5,
) -> float:
    """Query-conditioned structural score in [0, 1]."""
    if not q_concepts and not q_periods:
        return neutral
    # concept term
    if q_concepts:
        covered = expand_derivable(d_concepts) if use_derivable else d_concepts
        concept_cov = len(q_concepts & covered) / len(q_concepts)
    else:
        concept_cov = neutral
    # period term
    if q_periods:
        period_match = 1.0 if (q_periods & d_periods) else 0.0
    else:
        period_match = 1.0
    return concept_cov * (w0 + w1 * period_match)
