"""Query-aware fact selection — the bridge from KG to generator.

Given a question and the FactLedgers of the top-K retrieved documents, return the
*precise* facts most likely to answer the question, so the generator receives exact
cells instead of raw markdown. This directly addresses the user requirement: "even with
top-3 docs, retrieve and pass the exact data to the model".

Scoring stays lightweight and KG-aware (lexical + ontology + period), but it now also
raises narrative facts when the query asks for explanation/factors, and it uses column
headers / total-like labels so operand selection is closer to the gold arithmetic.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.scoring.concept_coverage import expand_derivable

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TOTAL_HINT_RE = re.compile(r"\b(total|overall|all|entire)\b", re.I)
_STOP = {
    "the", "a", "an", "of", "in", "for", "to", "and", "or", "was", "is", "are", "were",
    "what", "how", "much", "many", "did", "does", "do", "as", "by", "on", "at", "its",
    "their", "this", "that", "with", "from", "year", "fiscal", "reported", "report",
    "company", "value", "between", "during", "ended", "than", "which", "be",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

_TASK_PATTERNS = [
    ("percent_change", re.compile(r"\b(percent(?:age)? change|percentage change)\b", re.I)),
    ("average", re.compile(r"\b(average|mean)\b", re.I)),
    ("ratio", re.compile(r"\b(ratio|percent|percentage|proportion|margin|share of|fraction|portion)\b", re.I)),
    ("factor_sum", re.compile(r"\b(primarily due to|considering|factors?|because|as a result|driven by|impact of)\b", re.I)),
    ("difference", re.compile(r"\b(change|difference|increase|decrease|net change|how much (?:more|less))\b", re.I)),
    ("sum", re.compile(r"\b(sum|total|combined|together|aggregate)\b", re.I)),
]

_EXPLANATION_CUES = _TASK_PATTERNS[3][1]


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall((text or "").lower()):
        # Strip row/column suffix digits such as "facilities2" -> "facilities".
        tok = re.sub(r"(?<=\D)\d+$", "", raw)
        if not tok or tok in _STOP:
            continue
        if tok.isdigit() and not (len(tok) == 4 and 1900 <= int(tok) <= 2049):
            continue
        if len(tok) > 1:
            out.add(tok)
    return out


def _is_total_like(fact: Fact) -> bool:
    return bool(
        _TOTAL_HINT_RE.search(fact.concept or "")
        or _TOTAL_HINT_RE.search(fact.column_header or "")
        or _TOTAL_HINT_RE.search(fact.concept_canonical or "")
    )


def infer_task_type(query: str) -> str:
    """Coarse arithmetic intent for the prompt and fact selection."""
    q = query or ""
    for task, pat in _TASK_PATTERNS:
        if pat.search(q):
            return task
    return "lookup"


def task_formula(query: str, task: str) -> str:
    """Human-readable formula line for the prompt."""
    if task == "percent_change":
        return "(new - old) / abs(old)"
    if task == "difference":
        return "new - old"
    if task == "ratio":
        return "part / total"
    if task == "sum":
        return "sum(values)"
    if task == "average":
        return "sum(values) / count(values)"
    if task == "factor_sum":
        return "infer the change from the factor facts"
    return "copy the matching value exactly"


def _render_num(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.12g}"


def _render_fact_for_prompt(fact: Fact) -> str:
    header = f" | {fact.column_header}" if fact.column_header else ""
    per = f" [{fact.period}]" if fact.period else ""
    canon = f" [{fact.concept_canonical}]" if fact.concept_canonical else ""
    unit = f" {fact.unit}" if fact.unit else ""
    sc = f" (in {fact.scale_label})" if fact.scale_label else ""
    val = fact.raw_text if fact.raw_text else (f"{fact.value}" if fact.value is not None else "—")
    return f"{fact.concept}{canon}{per}{header} = {val}{unit}{sc}"


def score_fact(
    fact: Fact,
    q_tokens: set[str],
    q_years: list[int],
    q_concepts: Optional[set[str]] = None,
    concept_sim_fn: Optional[Callable[[str], float]] = None,
    narrative_sensitive: bool = False,
) -> float:
    """Relevance score of a single fact to the query."""
    q_concepts = q_concepts or set()
    q_derived = expand_derivable(q_concepts) if q_concepts else set()

    concept_tokens = _tokens(fact.concept)
    if concept_sim_fn is not None:
        concept_score = concept_sim_fn(fact.concept)
    else:
        concept_score = len(q_tokens & concept_tokens) / (len(concept_tokens) ** 0.5 if concept_tokens else 1.0)

    score = concept_score
    if concept_tokens == {"total"}:
        score -= 0.45
    header_tokens = _tokens(fact.column_header)
    if header_tokens:
        score += 0.30 * len(q_tokens & header_tokens)
        if "total" in header_tokens and ("total" in q_tokens or "percentage" in q_tokens or "percent" in q_tokens):
            score += 0.35
    if q_years:
        header_years = extract_years(fact.column_header)
        if header_years and any(y in q_years for y in header_years):
            score += 0.15
    elif not q_years:
        score += 0.05

    if q_concepts and fact.concept_canonical:
        if fact.concept_canonical in q_concepts:
            score += 1.20
        elif fact.concept_canonical in q_derived:
            score += 0.75
        elif concept_sim_fn is None:
            score += 0.05 * len(q_tokens & _tokens(fact.concept_canonical))

    if q_years and fact.period:
        try:
            if int(fact.period) in q_years:
                score += 0.15
        except ValueError:
            pass

    if _TOTAL_HINT_RE.search(fact.column_header or "") and ("total" in q_tokens or "overall" in q_tokens):
        score += 0.35

    if fact.source == "table":
        score += 0.22 if not narrative_sensitive else -1.20
    else:
        score += 0.18 if not narrative_sensitive else 1.10
        if narrative_sensitive:
            score += 0.25 * len(q_tokens & _tokens(fact.raw_text or fact.concept))
    return score


def select_facts(
    query: str,
    ledgers: list[FactLedger],
    top_n: int = 6,
    concept_sim_fn: Optional[Callable[[str], float]] = None,
    keep_per_doc: int = 8,
) -> list[Fact]:
    """Return the ``top_n`` most query-relevant facts across the given ledgers."""
    q_tokens = _tokens(query)
    q_years = extract_years(query)
    q_concepts = concepts_in_text(query)
    narrative_sensitive = bool(_EXPLANATION_CUES.search(query or ""))

    scored: list[tuple[float, Fact]] = []
    for ledger in ledgers:
        doc_scored = [
            (score_fact(f, q_tokens, q_years, q_concepts, concept_sim_fn, narrative_sensitive), f)
            for f in ledger.numeric_facts()
        ]
        doc_scored.sort(key=lambda x: x[0], reverse=True)
        scored.extend(doc_scored[:keep_per_doc])

    scored.sort(key=lambda x: x[0], reverse=True)
    out, seen = [], set()
    for s, f in scored:
        key = (f.doc_id, f.concept, f.period, f.column_header, f.value)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
        if len(out) >= top_n:
            break
    return out


def _sort_for_task(task: str, facts: list[Fact], query: str = "") -> list[Fact]:
    if not facts:
        return facts
    if task in ("difference", "percent_change"):
        def _period_key(f: Fact):
            try:
                if f.period:
                    return int(float(str(f.period)))
                yrs = extract_years((f.column_header or "") + " " + (f.concept or ""))
                if yrs:
                    return yrs[-1]
                return 10**9
            except ValueError:
                return 10**9
        return sorted(facts, key=lambda f: (_period_key(f), f.concept.lower(), f.row_idx, f.col_idx))
    if task == "ratio":
        q_tokens = _tokens(query)

        def _relevance(f: Fact) -> float:
            return len(q_tokens & _tokens(f.concept)) + 0.5 * len(q_tokens & _tokens(f.column_header))

        numerator_pool = [
            f for f in facts
            if _TOTAL_HINT_RE.search(f.column_header or "") and not _TOTAL_HINT_RE.search(f.concept or "")
        ]
        if not numerator_pool:
            numerator_pool = [f for f in facts if not _TOTAL_HINT_RE.search(f.concept or "")]
        numerator = max(numerator_pool, key=_relevance) if numerator_pool else None

        denominator_pool = [f for f in facts if _TOTAL_HINT_RE.search(f.concept or "")]
        if not denominator_pool:
            denominator_pool = [f for f in facts if _TOTAL_HINT_RE.search(f.column_header or "")]
        if numerator is not None and numerator in denominator_pool and len(denominator_pool) > 1:
            denominator_pool = [f for f in denominator_pool if f is not numerator]
        denominator = max(denominator_pool, key=_relevance) if denominator_pool else None

        ordered = [f for f in (numerator, denominator) if f is not None]
        rest = [f for f in facts if f not in ordered]
        return ordered + rest
    if task in ("sum", "average", "factor_sum"):
        return facts[:4]
    return facts


def _calculation_hint(task: str, facts: list[Fact]) -> str:
    vals = [f.value for f in facts if f.value is not None]
    if task == "difference" and len(vals) >= 2:
        old, new = vals[0], vals[1]
        return f"CALCULATION HINT: {_render_num(new)} - {_render_num(old)} = {_render_num(new - old)}"
    if task == "percent_change" and len(vals) >= 2:
        old, new = vals[0], vals[1]
        if old:
            return (
                "CALCULATION HINT: ("
                f"{_render_num(new)} - {_render_num(old)}) / abs({_render_num(old)}) = "
                f"{_render_num((new - old) / abs(old))}"
            )
    if task == "ratio" and len(vals) >= 2:
        part, total = vals[0], vals[1]
        if total:
            return f"CALCULATION HINT: {_render_num(part)} / {_render_num(total)} = {_render_num(part / total)}"
    if task == "sum" and vals:
        return f"CALCULATION HINT: sum({', '.join(_render_num(v) for v in vals)}) = {_render_num(sum(vals))}"
    if task == "average" and vals:
        return f"CALCULATION HINT: sum({', '.join(_render_num(v) for v in vals)}) / {len(vals)} = {_render_num(sum(vals) / len(vals))}"
    if task == "factor_sum":
        return ""
    return ""


def build_evidence_block(
    query: str,
    ledgers: list[FactLedger],
    top_n: int = 6,
    concept_sim_fn: Optional[Callable[[str], float]] = None,
    facts: list[Fact] | None = None,
) -> str:
    """Render selected facts into a compact evidence block for the generator prompt."""
    task = infer_task_type(query)
    formula = task_formula(query, task)
    selected = facts if facts is not None else select_facts(query, ledgers, top_n=top_n, concept_sim_fn=concept_sim_fn)
    facts = _sort_for_task(task, selected, query)
    if not facts:
        return ""

    lines = [
        "RELEVANT FINANCIAL FACTS (extracted from the retrieved documents):",
        f"TASK: {task}",
        f"FORMULA: {formula}",
    ]
    if task in ("ratio", "percent_change"):
        lines.append("PREFERRED OUTPUT SCALE: decimal fraction")

    if task in ("difference", "percent_change") and len(facts) >= 2:
        lines.append("OPERANDS:")
        lines.append("  - old: " + _render_fact_for_prompt(facts[0]))
        lines.append("  - new: " + _render_fact_for_prompt(facts[1]))
        hint = _calculation_hint(task, facts[:2])
        if hint:
            lines.append(hint)
        facts = facts[:2]
    elif task == "ratio" and len(facts) >= 2:
        lines.append("OPERANDS:")
        lines.append("  - part: " + _render_fact_for_prompt(facts[0]))
        lines.append("  - total: " + _render_fact_for_prompt(facts[1]))
        hint = _calculation_hint(task, facts[:2])
        if hint:
            lines.append(hint)
        facts = facts[:2]
    elif task in ("sum", "average"):
        lines.append("OPERANDS:")
        for i, f in enumerate(facts[:4], start=1):
            lines.append(f"  - value {i}: " + _render_fact_for_prompt(f))
        hint = _calculation_hint(task, facts[:4])
        if hint:
            lines.append(hint)
        facts = facts[:4]
    elif task == "factor_sum":
        text_facts = [f for f in facts if f.source == "text"]
        if text_facts:
            facts = text_facts
        lines.append("KEY FACTORS:")
        for i, f in enumerate(facts[:4], start=1):
            lines.append(f"  - factor {i}: " + _render_fact_for_prompt(f))
        lines.append("GUIDANCE: use the factor facts to infer the change; do not sum every number blindly.")
        facts = facts[:4]

    by_doc: dict[str, list[Fact]] = {}
    for f in facts:
        by_doc.setdefault(f.doc_id, []).append(f)

    lines.append("FOCUS FACTS:")
    for doc_id, fs in by_doc.items():
        company = fs[0].company or ""
        sl = fs[0].scale_label
        head = f"[{doc_id}]" + (f" {company}" if company else "") + (f" (in {sl})" if sl else "")
        lines.append(head)
        for f in fs:
            lines.append("  - " + _render_fact_for_prompt(f))
    return "\n".join(lines)
