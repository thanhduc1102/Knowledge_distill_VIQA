"""Query-aware fact selection — the bridge from KG to generator.

Given a question and the FactLedgers of the top-K retrieved documents, return the
*precise* facts most likely to answer the question, so the generator receives exact
cells instead of raw markdown. This directly addresses the user requirement: "even with
top-3 docs, retrieve and pass the exact data to the model".

Scoring is intentionally lightweight (lexical + period + entity) so it runs without a
GPU; an optional embedding scorer can be plugged in via ``concept_sim_fn``.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.numeric import extract_years

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "in", "for", "to", "and", "or", "was", "is", "are", "were",
    "what", "how", "much", "many", "did", "does", "do", "as", "by", "on", "at", "its",
    "their", "this", "that", "with", "from", "year", "fiscal", "reported", "report",
    "company", "value", "total", "between", "during", "ended", "than", "which", "be",
}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


def _lexical_overlap(q_tokens: set[str], concept: str) -> float:
    c_tokens = _tokens(concept)
    if not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / (len(c_tokens) ** 0.5)


def score_fact(
    fact: Fact,
    q_tokens: set[str],
    q_years: list[int],
    concept_sim_fn: Optional[Callable[[str], float]] = None,
) -> float:
    """Relevance score of a single fact to the query."""
    if concept_sim_fn is not None:
        concept_score = concept_sim_fn(fact.concept)
    else:
        concept_score = _lexical_overlap(q_tokens, fact.concept)

    score = concept_score
    # period gate: boost facts whose period matches a year in the question
    if q_years and fact.period:
        try:
            if int(fact.period) in q_years:
                score += 0.6
        except ValueError:
            pass
    elif not q_years:
        score += 0.05  # no temporal constraint → slight neutral boost
    # strongly prefer structured table facts over noisier narrative numbers
    if fact.source == "table":
        score += 0.25
    else:
        score -= 0.20
    return score


def select_facts(
    query: str,
    ledgers: list[FactLedger],
    top_n: int = 12,
    concept_sim_fn: Optional[Callable[[str], float]] = None,
    keep_per_doc: int = 8,
) -> list[Fact]:
    """Return the ``top_n`` most query-relevant facts across the given ledgers."""
    q_tokens = _tokens(query)
    q_years = extract_years(query)

    scored: list[tuple[float, Fact]] = []
    for ledger in ledgers:
        doc_scored = [
            (score_fact(f, q_tokens, q_years, concept_sim_fn), f)
            for f in ledger.numeric_facts()
        ]
        doc_scored.sort(key=lambda x: x[0], reverse=True)
        scored.extend(doc_scored[:keep_per_doc])

    scored.sort(key=lambda x: x[0], reverse=True)
    out, seen = [], set()
    for s, f in scored:
        key = (f.doc_id, f.concept, f.period, f.value)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
        if len(out) >= top_n:
            break
    return out


def build_evidence_block(
    query: str,
    ledgers: list[FactLedger],
    top_n: int = 12,
    concept_sim_fn: Optional[Callable[[str], float]] = None,
) -> str:
    """Render selected facts into a compact evidence block for the generator prompt."""
    facts = select_facts(query, ledgers, top_n=top_n, concept_sim_fn=concept_sim_fn)
    if not facts:
        return ""
    by_doc: dict[str, list[Fact]] = {}
    for f in facts:
        by_doc.setdefault(f.doc_id, []).append(f)

    lines = ["RELEVANT FINANCIAL FACTS (extracted from the retrieved documents):"]
    for doc_id, fs in by_doc.items():
        company = fs[0].company or ""
        sl = fs[0].scale_label
        head = f"[{doc_id}]" + (f" {company}" if company else "") + (f" (in {sl})" if sl else "")
        lines.append(head)
        for f in fs:
            lines.append("  - " + f.render())
    return "\n".join(lines)
