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
from typing import Any, Callable, Optional

from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.numeric import extract_numbers, extract_years
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

_ADJUSTMENT_RE = re.compile(r"\b(assuming|without|excluding|absent|had there been no)\b", re.I)
_COMPARISON_RE = re.compile(
    r"\b(did|does|do|is|are|was|were|which year)\b.*"
    r"\b(exceed|greater than|higher than|more than|less than|lower than|below|above)\b",
    re.I,
)
_PERCENT_CHANGE_RE = re.compile(
    r"\b(percent(?:age)? change|percentage change|percent(?:age)? (?:increase|decrease)|"
    r"percentage (?:increase|decrease))\b",
    re.I,
)
_RATIO_RE = re.compile(r"\b(ratio|percent|percentage|proportion|margin|share of|fraction|portion)\b", re.I)
_FACTOR_RE = re.compile(r"\b(primarily due to|considering|factors?|because|as a result|driven by|impact of)\b", re.I)

_TASK_PATTERNS = [
    ("percent_change", _PERCENT_CHANGE_RE),
    ("ratio", re.compile(r"\b(as a percentage of|what percentage of|percentage of|percent of|ratio|proportion|share of|fraction|portion)\b", re.I)),
    ("adjustment", _ADJUSTMENT_RE),
    ("comparison", _COMPARISON_RE),
    ("average", re.compile(r"\b(average|mean)\b", re.I)),
    ("ratio", _RATIO_RE),
    ("factor_sum", _FACTOR_RE),
    ("difference", re.compile(r"\b(change|difference|increase|decrease|net change|how much (?:more|less))\b", re.I)),
    ("sum", re.compile(r"\b(sum|total|combined|together|aggregate)\b", re.I)),
]

_EXPLANATION_CUES = _FACTOR_RE


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
        return "new - old when boundary values exist; otherwise use listed factor facts"
    if task == "comparison":
        return "1 if the left value satisfies the comparison, else 0"
    if task == "adjustment":
        return "base value +/- explicit adjustment"
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
        if {"change", "percent"} & header_tokens and {"change", "percent", "percentage"} & q_tokens:
            score += 0.75
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

    explicit_pct = "%" in (fact.raw_text or "") or "%" in (fact.column_header or "") or "percent" in header_tokens or "percentage" in header_tokens
    if explicit_pct and {"change", "percent", "percentage"} & q_tokens:
        score += 0.9

    if fact.source == "table":
        score += 0.22 if not narrative_sensitive else -1.20
    else:
        score += -0.75 if not narrative_sensitive else 1.10
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
    direct_pool: list[Fact] = []
    for ledger in ledgers:
        doc_scored = [
            (score_fact(f, q_tokens, q_years, q_concepts, concept_sim_fn, narrative_sensitive), f)
            for f in ledger.numeric_facts()
        ]
        doc_scored.sort(key=lambda x: x[0], reverse=True)
        direct_pool.extend(f for _, f in doc_scored)
        scored.extend(doc_scored[:keep_per_doc])

    scored.sort(key=lambda x: x[0], reverse=True)
    out, seen = [], set()
    text_budget = top_n if narrative_sensitive else max(1, top_n // 3)
    text_count = 0
    deferred: list[tuple[float, Fact]] = []
    for s, f in scored:
        key = (f.doc_id, f.concept, f.period, f.column_header, f.value)
        if key in seen:
            continue
        if f.source != "table" and text_count >= text_budget:
            deferred.append((s, f))
            continue
        seen.add(key)
        out.append(f)
        if f.source != "table":
            text_count += 1
        if len(out) >= top_n:
            break
    if len(out) < top_n:
        for _, f in deferred:
            key = (f.doc_id, f.concept, f.period, f.column_header, f.value)
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
            if len(out) >= top_n:
                break
    if infer_task_type(query) == "percent_change":
        direct = _best_direct_percent_fact(query, direct_pool)
        if direct is not None and direct[1] >= 0.8:
            f = direct[0]
            key = (f.doc_id, f.concept, f.period, f.column_header, f.value)
            if key not in seen:
                out = [f] + out[: max(0, top_n - 1)]
    return out


def _sort_for_task(task: str, facts: list[Fact], query: str = "") -> list[Fact]:
    if not facts:
        return facts
    if task in ("difference", "percent_change", "factor_sum"):
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


def _fact_tokens(f: Fact) -> set[str]:
    return _tokens(" ".join([f.concept or "", f.column_header or "", f.concept_canonical or ""]))


def _semantic_fact_tokens(f: Fact) -> set[str]:
    years = {str(y) for y in extract_years(" ".join([f.concept or "", f.column_header or ""]))}
    return {t for t in _fact_tokens(f) if t not in years}


def _norm_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def _balance_like(f: Fact) -> bool:
    blob = _norm_text(" ".join([f.concept or "", f.column_header or ""]))
    return "balance" in blob and any(x in blob for x in ("december", "ending", "end", "as of"))


def _ending_balance_query(query: str) -> bool:
    q = _norm_text(query)
    return bool(re.search(r"\b(end|ending|as of|balance|december 31)\b", q))


def _query_money_numbers(query: str) -> list[float]:
    """Money-like literals from the question, with years/percentages filtered out."""
    vals: list[float] = []
    money_re = re.compile(
        r"(?:\$\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?"
        r"|\b\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)\b)",
        re.I,
    )
    for m in money_re.finditer(query or ""):
        nums = extract_numbers(m.group(0))
        vals.extend(nums)
    years = set(extract_years(query))
    out: list[float] = []
    for v in vals:
        if int(v) in years and 1900 <= int(v) <= 2049:
            continue
        if v > 0:
            out.append(v)
    return out


def _target_year_pair(query: str) -> tuple[int, int] | None:
    q = query or ""
    pats = [
        r"\bfrom\b[^?;\n]{0,80}?\b(19[5-9]\d|20[0-4]\d)\b[^?;\n]{0,80}?\bto\b[^?;\n]{0,80}?\b(19[5-9]\d|20[0-4]\d)\b",
        r"\bfrom\s+(?:the\s+end\s+of\s+|fiscal\s+year\s+|fiscal\s+)?(19[5-9]\d|20[0-4]\d)\s+(?:to|through)\s+(?:the\s+end\s+of\s+|fiscal\s+year\s+|fiscal\s+)?(19[5-9]\d|20[0-4]\d)\b",
        r"\bbetween\s+(?:the\s+fiscal\s+years?\s+)?(19[5-9]\d|20[0-4]\d)\s+and\s+(19[5-9]\d|20[0-4]\d)\b",
        r"\b(19[5-9]\d|20[0-4]\d)\s+(?:compared\s+to|versus|vs\.?)\s+(?:the\s+previous\s+year|prior\s+year)\b",
    ]
    for pat in pats:
        m = re.search(pat, q, re.I)
        if m:
            a = int(m.group(1))
            if len(m.groups()) >= 2 and m.group(2) and m.group(2).isdigit():
                b = int(m.group(2))
            else:
                b = a
                a = b - 1
            return (min(a, b), max(a, b))
    years = extract_years(q)
    if len(years) == 1 and re.search(r"\b(previous|prior)\s+year\b", q, re.I):
        return (years[0] - 1, years[0])
    return None


def _factor_like_facts(query: str, facts: list[Fact], max_facts: int = 6) -> list[Fact]:
    """Prefer rows explicitly named in a factor/bridge question over boundary totals."""
    qt = _tokens(query)
    out: list[tuple[float, Fact]] = []
    for f in facts:
        if f.value is None:
            continue
        ft = _fact_tokens(f)
        overlap = len(qt & ft)
        if f.concept_canonical == "Revenue" and not overlap:
            continue
        if f.concept_canonical == "Revenue":
            overlap -= 2
        if overlap <= 0:
            continue
        if any(tok in ft for tok in ("price", "volume", "weather", "credit", "credits", "deferral", "provision", "other", "rate")):
            overlap += 1.5
        out.append((float(overlap), f))
    out.sort(key=lambda x: x[0], reverse=True)
    chosen: list[Fact] = []
    seen = set()
    for _, f in out:
        key = (f.doc_id, f.concept, f.period, f.column_header, f.value)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(f)
        if len(chosen) >= max_facts:
            break
    return chosen


def _concept_compatible(a: Fact, b: Fact) -> bool:
    if a.concept_canonical and b.concept_canonical and a.concept_canonical == b.concept_canonical:
        return True
    ta, tb = _semantic_fact_tokens(a), _semantic_fact_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / max(len(ta | tb), 1) >= 0.35


def _query_fact_support(query: str, fact: Fact) -> float:
    qt = _tokens(query)
    ft = _fact_tokens(fact)
    score = float(len(qt & ft))
    if fact.concept_canonical and fact.concept_canonical in concepts_in_text(query):
        score += 2.0
    if _ending_balance_query(query) and _balance_like(fact):
        score += 2.0
    return score


def _period_int(f: Fact) -> int | None:
    try:
        if f.period:
            return int(float(str(f.period)))
    except (TypeError, ValueError):
        return None
    yrs = extract_years((f.column_header or "") + " " + (f.concept or ""))
    return yrs[-1] if yrs else None


def _best_temporal_pair(query: str, facts: list[Fact]) -> tuple[Fact, Fact] | None:
    explicit_pair = _target_year_pair(query)
    q_years = list(explicit_pair) if explicit_pair else sorted(set(extract_years(query)))
    by_group: dict[str, list[Fact]] = {}
    for f in facts:
        if f.value is None:
            continue
        key = f.concept_canonical or " ".join(sorted(_semantic_fact_tokens(f)))
        if key:
            by_group.setdefault(key, []).append(f)

    ending_balance = _ending_balance_query(query)

    def relevance(f: Fact) -> float:
        qt = _tokens(query)
        s = float(len(qt & _fact_tokens(f)))
        if ending_balance and _balance_like(f):
            s += 8.0
        if f.concept_canonical == "Revenue" and "revenue" in qt:
            s += 1.0
        return s

    best: tuple[float, Fact, Fact] | None = None
    for fs in by_group.values():
        if len(fs) < 2:
            continue
        if len(q_years) >= 2:
            old_year, new_year = q_years[0], q_years[-1]
            olds = [f for f in fs if _period_int(f) == old_year]
            news = [f for f in fs if _period_int(f) == new_year]
            if olds and news:
                old = max(olds, key=relevance)
                new = max(news, key=relevance)
                score = 3.0 + relevance(old) + relevance(new)
                if best is None or score > best[0]:
                    best = (score, old, new)
        else:
            fs2 = sorted(fs, key=lambda f: (_period_int(f) or 10**9, -relevance(f)))
            for old, new in zip(fs2, fs2[1:]):
                if _period_int(old) != _period_int(new):
                    score = 1.0 + relevance(old) + relevance(new)
                    if best is None or score > best[0]:
                        best = (score, old, new)
    return (best[1], best[2]) if best else None


def _best_ratio_pair(query: str, facts: list[Fact]) -> tuple[Fact, Fact, float] | None:
    qt = _tokens(query)
    q_concepts = concepts_in_text(query)
    usable = [f for f in facts if f.value is not None]
    if len(usable) < 2:
        return None
    q_years = set(extract_years(query))

    def rel(f: Fact) -> float:
        s = len(qt & _fact_tokens(f))
        if f.concept_canonical and f.concept_canonical in q_concepts:
            s += 3.0
        if q_years and _period_int(f) in q_years:
            s += 2.0
        return float(s)

    def axis_compatible(part: Fact, total: Fact) -> bool:
        ph, th = _tokens(part.column_header), _tokens(total.column_header)
        if ph and th:
            return bool(ph & th)
        return True

    def denom_score(part: Fact, f: Fact) -> float:
        s = rel(f)
        if _is_total_like(f):
            s += 3.0
        if f.concept_canonical in {"TotalAssets", "TotalDebt", "Revenue", "TotalEquity"}:
            s += 1.0
        if not axis_compatible(part, f):
            s -= 4.0
        return s

    best: tuple[float, Fact, Fact, float] | None = None
    for part in usable:
        for total in usable:
            if part is total or total.value in (None, 0):
                continue
            # Same metric across periods is usually a change task, not part/total.
            if part.concept_canonical and part.concept_canonical == total.concept_canonical:
                continue
            if (
                (part.concept or "").strip().lower() == (total.concept or "").strip().lower()
                and (part.period or "_") == (total.period or "_")
                and (part.column_header or "").strip().lower() != (total.column_header or "").strip().lower()
            ):
                continue
            score = rel(part) + denom_score(part, total)
            confidence = 0.45
            if rel(part) >= 1 and denom_score(part, total) >= 3 and axis_compatible(part, total):
                confidence = 0.85
            if part.concept_canonical in q_concepts and total.concept_canonical in q_concepts:
                confidence = 0.9
            if best is None or score > best[0]:
                best = (score, part, total, confidence)
    return (best[1], best[2], best[3]) if best else None


def _best_question_amount_ratio(query: str, facts: list[Fact]) -> tuple[float, Fact, float] | None:
    """Ratio where the numerator is explicitly stated in the question.

    Example: "What percentage of total EMEA assets did the United Kingdom's
    $6.9 million represent?"  The $6.9 is already given; the KG only needs to
    select the denominator cell and expose provenance for it.
    """
    amounts = _query_money_numbers(query)
    if not amounts:
        return None
    amount = amounts[0]
    qt = _tokens(query)
    q_years = set(extract_years(query))
    best: tuple[float, Fact] | None = None
    for f in facts:
        if f.value in (None, 0):
            continue
        if abs((f.value or 0.0) - amount) <= max(1e-6, abs(amount) * 1e-6):
            continue
        if f.unit == "%" or "%" in (f.raw_text or "") or "%" in (f.column_header or ""):
            continue
        if abs(f.value or 0.0) < abs(amount):
            continue
        ft = _fact_tokens(f)
        score = float(len(qt & ft))
        if q_years and _period_int(f) in q_years:
            score += 2.0
        if _is_total_like(f):
            score += 1.5
        if "total" in qt and not _is_total_like(f):
            score += 0.4
        if f.source == "table":
            score += 0.5
        if best is None or score > best[0]:
            best = (score, f)
    if best is None:
        return None
    confidence = 0.9 if best[0] >= 3.0 else 0.6
    return amount, best[1], confidence


def _best_direct_percent_fact(query: str, facts: list[Fact]) -> tuple[Fact, float] | None:
    """Find an already-computed percent-change fact, e.g. a `% CHANGE` column."""
    qt = _tokens(query)
    q_years = set(extract_years(query))
    candidates: list[tuple[float, Fact]] = []
    strict_terms = {"wireless", "internet", "iptv", "satellite", "medical", "interface", "asia", "pacific", "europe", "africa"}
    for f in facts:
        if f.value is None:
            continue
        header_tokens = _tokens(f.column_header)
        pct_like = "%" in (f.raw_text or "") or "%" in (f.column_header or "") or "percent" in header_tokens or "percentage" in header_tokens
        if not pct_like or abs(f.value) > 1000:
            continue
        ft = _fact_tokens(f) | _tokens(f.raw_text or "")
        score = float(len(qt & ft))
        if {"change", "percent", "percentage"} & header_tokens:
            score += 2.0
        if {"percent", "percentage"} & qt:
            score += 0.5
        if q_years and (not f.period or _period_int(f) in q_years):
            score += 0.5
        if f.source == "text":
            score += 0.5
        for term in strict_terms & qt:
            if term not in ft:
                score -= 2.5
        if ("subscriber" in qt or "subscribers" in qt) and not ({"subscriber", "subscribers"} & ft):
            score -= 2.0
        candidates.append((score, f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, fact = candidates[0]
    confidence = 0.9 if score >= 2.5 else (0.85 if score >= 2.0 and {"percent", "percentage"} & qt else 0.6)
    return fact, confidence


def _is_benchmark_fact(query: str, fact: Fact) -> bool:
    q = (query or "").lower()
    blob = " ".join([fact.concept or "", fact.column_header or ""]).lower()
    if "s&p" in q or "s & p" in q or "sp 500" in q or "s p 500" in q:
        if "s&p" in blob or "s & p" in blob or "sp 500" in blob or "s p 500" in blob or "500" in blob:
            return True
    for key in ("peer group", "index", "benchmark"):
        if key in q and key in blob:
            return True
    return False


def _best_comparison_pair(query: str, facts: list[Fact]) -> tuple[Fact, Fact, str, float] | None:
    usable = [f for f in facts if f.value is not None]
    if len(usable) < 2:
        return None
    qt = _tokens(query)
    q_years = sorted(set(extract_years(query)))
    target_year = str(q_years[-1]) if q_years else None
    company_tokens = set()
    for f in usable:
        company_tokens |= _tokens(f.company)

    def period_bonus(f: Fact) -> float:
        if target_year and str(f.period or "") == target_year:
            return 3.0
        return 0.0

    def left_score(f: Fact) -> float:
        s = float(len(qt & _fact_tokens(f))) + period_bonus(f)
        if company_tokens & _fact_tokens(f):
            s += 3.0
        if "peer" in _fact_tokens(f):
            s -= 2.0
        if _is_benchmark_fact(query, f):
            s -= 4.0
        return s

    def right_score(f: Fact) -> float:
        s = float(len(qt & _fact_tokens(f))) + period_bonus(f)
        if _is_benchmark_fact(query, f):
            s += 6.0
        if "peer" in _fact_tokens(f) and "peer" in qt:
            s += 2.0
        return s

    op = ">"
    ql = (query or "").lower()
    if re.search(r"\b(less than|lower than|below)\b", ql):
        op = "<"

    best: tuple[float, Fact, Fact] | None = None
    for left in usable:
        for right in usable:
            if left is right:
                continue
            if target_year and (str(left.period or "") != target_year or str(right.period or "") != target_year):
                continue
            if left.doc_id != right.doc_id:
                continue
            score = left_score(left) + right_score(right)
            if best is None or score > best[0]:
                best = (score, left, right)
    if best is None:
        return None
    confidence = 0.9 if best[0] >= 8.0 else 0.6
    return best[1], best[2], op, confidence


def _best_adjustment_plan(query: str, facts: list[Fact]) -> tuple[Fact, float, str, float] | None:
    amounts = _query_money_numbers(query)
    if not amounts:
        return None
    qt = _tokens(query)
    q_years = sorted(set(extract_years(query)))

    def base_score(f: Fact) -> float:
        s = float(len(qt & _fact_tokens(f)))
        if f.concept_canonical in concepts_in_text(query):
            s += 3.0
        if q_years and f.period and int(float(str(f.period))) in q_years:
            s += 2.0
        return s

    usable = [f for f in facts if f.value is not None]
    if not usable:
        return None
    base = max(usable, key=base_score)
    amount = amounts[0]
    ql = (query or "").lower()
    # Removing a gain/benefit/increase lowers the base; removing a loss/expense/charge raises it.
    sign = -1.0
    if re.search(r"\b(loss|expense|cost|charge|write[- ]?off|impairment)\b", ql) and not re.search(r"\b(gain|benefit|increase)\b", ql):
        sign = 1.0
    op = "-" if sign < 0 else "+"
    confidence = 0.85 if base_score(base) >= 2.0 else 0.55
    return base, amount, op, confidence


def _operand_dict(role: str, fact: Fact) -> dict[str, Any]:
    return {
        "role": role,
        "doc_id": fact.doc_id,
        "concept": fact.concept,
        "concept_canonical": fact.concept_canonical,
        "period": fact.period,
        "value": fact.value,
        "raw_text": fact.raw_text,
        "unit": fact.unit,
        "scale_label": fact.scale_label,
        "provenance": fact.provenance,
        "source": fact.source,
    }


def calculation_plan(query: str, facts: list[Fact]) -> dict[str, Any]:
    """Return a deterministic KG-side calculation plan for selected facts.

    This is the inference-time "LLM burden reducer": when the Fact Ledger has enough
    operands, it computes the answer and exposes the exact operand provenance. The
    generator can then copy/check the result instead of re-parsing markdown tables.
    """
    task = infer_task_type(query)
    formula = task_formula(query, task)
    ordered = _sort_for_task(task, [f for f in facts if f.value is not None], query)
    answer: float | None = None
    trace = ""
    operands: list[dict[str, Any]] = []
    confidence = 0.0

    vals = [f.value for f in ordered if f.value is not None]
    pair = _best_temporal_pair(query, ordered) if task in ("percent_change", "difference", "factor_sum", "comparison") else None
    if task == "percent_change":
        direct = _best_direct_percent_fact(query, ordered)
        if direct is not None and direct[1] >= 0.8:
            pct_f, confidence = direct
            answer = pct_f.value
            trace = f"copy {_render_num(answer)} from selected percent-change fact"
            operands = [_operand_dict("answer_fact", pct_f)]
            return {
                "task": task,
                "formula": formula,
                "answer": answer,
                "trace": trace,
                "operands": operands,
                "confidence": confidence,
            }
    if task == "percent_change" and pair and pair[0].value:
        old_f, new_f = pair
        old, new = old_f.value, new_f.value
        answer = (new - old) / abs(old)
        trace = (
            f"({_render_num(new)} - {_render_num(old)}) / "
            f"abs({_render_num(old)}) = {_render_num(answer)}"
        )
        operands = [_operand_dict("old", old_f), _operand_dict("new", new_f)]
        support = max(_query_fact_support(query, old_f), _query_fact_support(query, new_f))
        confidence = 0.9 if _concept_compatible(old_f, new_f) and support >= 1.0 else 0.55
    elif task == "difference" and pair:
        old_f, new_f = pair
        old, new = old_f.value, new_f.value
        answer = new - old
        trace = f"{_render_num(new)} - {_render_num(old)} = {_render_num(answer)}"
        operands = [_operand_dict("old", old_f), _operand_dict("new", new_f)]
        support = max(_query_fact_support(query, old_f), _query_fact_support(query, new_f))
        confidence = 0.9 if _concept_compatible(old_f, new_f) and support >= 1.0 else 0.55
    elif task == "factor_sum" and pair:
        old_f, new_f = pair
        old, new = old_f.value, new_f.value
        answer = new - old
        trace = f"{_render_num(new)} - {_render_num(old)} = {_render_num(answer)}"
        operands = [_operand_dict("old_boundary", old_f), _operand_dict("new_boundary", new_f)]
        support = max(_query_fact_support(query, old_f), _query_fact_support(query, new_f))
        confidence = 0.85 if _concept_compatible(old_f, new_f) and support >= 1.0 else 0.5
    elif task == "ratio":
        if not _query_money_numbers(query):
            direct = _best_direct_percent_fact(query, ordered)
            if direct is not None and direct[1] >= 0.8:
                pct_f, confidence = direct
                answer = pct_f.value
                trace = f"copy {_render_num(answer)} from selected percent fact"
                operands = [_operand_dict("answer_fact", pct_f)]
                return {
                    "task": task,
                    "formula": formula,
                    "answer": answer,
                    "trace": trace,
                    "operands": operands,
                    "confidence": confidence,
                }
        question_ratio = _best_question_amount_ratio(query, ordered)
        if question_ratio is not None and question_ratio[2] >= 0.8:
            amount, total_f, confidence = question_ratio
            total = total_f.value
            answer = amount / total
            trace = f"{_render_num(amount)} / {_render_num(total)} = {_render_num(answer)}"
            operands = [
                {"role": "question_part", "value": amount, "raw_text": str(amount), "provenance": "question"},
                _operand_dict("total", total_f),
            ]
            return {
                "task": task,
                "formula": formula,
                "answer": answer,
                "trace": trace,
                "operands": operands,
                "confidence": confidence,
            }
        ratio_pair = _best_ratio_pair(query, ordered)
        if ratio_pair is None:
            return {"task": task, "formula": formula, "answer": None, "trace": "", "operands": [], "confidence": 0.0}
        part_f, total_f, confidence = ratio_pair
        part, total = part_f.value, total_f.value
        answer = part / total
        trace = f"{_render_num(part)} / {_render_num(total)} = {_render_num(answer)}"
        operands = [_operand_dict("part", part_f), _operand_dict("total", total_f)]
    elif task == "comparison":
        if (
            pair
            and re.search(r"\bwhich\s+year\b", query or "", re.I)
            and re.search(r"\b(previous|prior)\s+year\b", query or "", re.I)
        ):
            old_f, new_f = pair
            op = "<" if re.search(r"\b(less than|lower than|below)\b", query or "", re.I) else ">"
            ok = (new_f.value or 0.0) < (old_f.value or 0.0) if op == "<" else (new_f.value or 0.0) > (old_f.value or 0.0)
            year = _period_int(new_f)
            if ok and year is not None:
                answer = float(year)
                trace = (
                    f"{_render_num(new_f.value)} {op} {_render_num(old_f.value)} "
                    f"=> year {year}"
                )
                operands = [_operand_dict("old", old_f), _operand_dict("new", new_f)]
                confidence = 0.85 if _concept_compatible(old_f, new_f) else 0.55
                return {
                    "task": task,
                    "formula": formula,
                    "answer": answer,
                    "trace": trace,
                    "operands": operands,
                    "confidence": confidence,
            }
        comp_pair = _best_comparison_pair(query, ordered)
        if comp_pair is None:
            return {"task": task, "formula": formula, "answer": None, "trace": "", "operands": [], "confidence": 0.0}
        if re.search(r"\bhow\s+many\s+years\b", query or "", re.I):
            return {"task": task, "formula": formula, "answer": None, "trace": "", "operands": [], "confidence": 0.0}
        left_f, right_f, op, confidence = comp_pair
        if op == ">":
            answer = 1.0 if (left_f.value or 0.0) > (right_f.value or 0.0) else 0.0
        else:
            answer = 1.0 if (left_f.value or 0.0) < (right_f.value or 0.0) else 0.0
        trace = (
            f"{_render_num(left_f.value)} {op} {_render_num(right_f.value)} "
            f"=> {_render_num(answer)}"
        )
        operands = [_operand_dict("left", left_f), _operand_dict("right", right_f)]
    elif task == "adjustment":
        adj = _best_adjustment_plan(query, ordered)
        if adj is None:
            return {"task": task, "formula": formula, "answer": None, "trace": "", "operands": [], "confidence": 0.0}
        base_f, amount, op, confidence = adj
        base = base_f.value or 0.0
        answer = base - amount if op == "-" else base + amount
        trace = f"{_render_num(base)} {op} {_render_num(amount)} = {_render_num(answer)}"
        operands = [_operand_dict("base", base_f), {"role": "question_adjustment", "value": amount, "raw_text": str(amount), "provenance": "question"}]
    elif task == "sum" and vals:
        use = ordered[:4]
        answer = sum(f.value for f in use if f.value is not None)
        trace = f"sum({', '.join(_render_num(f.value) for f in use)}) = {_render_num(answer)}"
        operands = [_operand_dict(f"value_{i}", f) for i, f in enumerate(use, start=1)]
        confidence = 0.55
    elif task == "average" and vals:
        use = ordered[:4]
        answer = sum(f.value for f in use if f.value is not None) / len(use)
        trace = (
            f"sum({', '.join(_render_num(f.value) for f in use)}) / {len(use)} = "
            f"{_render_num(answer)}"
        )
        operands = [_operand_dict(f"value_{i}", f) for i, f in enumerate(use, start=1)]
        confidence = 0.55
    elif task == "lookup" and ordered:
        answer = ordered[0].value
        trace = f"copy {_render_num(answer)} from selected fact"
        operands = [_operand_dict("answer_fact", ordered[0])]
        confidence = 0.85 if _query_fact_support(query, ordered[0]) >= 2.0 else 0.6

    return {
        "task": task,
        "formula": formula,
        "answer": answer,
        "trace": trace,
        "operands": operands,
        "confidence": confidence,
    }


def _fact_from_operand(op: dict[str, Any], facts: list[Fact]) -> Fact | None:
    for f in facts:
        if (
            f.doc_id == op.get("doc_id")
            and f.concept == op.get("concept")
            and f.period == op.get("period")
            and f.value == op.get("value")
            and f.provenance == op.get("provenance")
        ):
            return f
    return None


def _render_plan_operands(plan: dict[str, Any], facts: list[Fact]) -> tuple[list[str], list[Fact]]:
    role_labels = {
        "old": "old",
        "new": "new",
        "old_boundary": "old boundary",
        "new_boundary": "new boundary",
        "part": "part",
        "total": "total",
        "left": "left",
        "right": "right",
        "base": "base",
        "answer_fact": "answer fact",
    }
    lines = ["OPERANDS:"]
    used: list[Fact] = []
    for op in plan.get("operands") or []:
        role = str(op.get("role") or "")
        if role == "question_adjustment":
            lines.append(f"  - adjustment from question: {_render_num(op.get('value'))}")
            continue
        if role == "question_part":
            lines.append(f"  - part from question: {_render_num(op.get('value'))}")
            continue
        fact = _fact_from_operand(op, facts)
        if fact is None:
            continue
        used.append(fact)
        lines.append(f"  - {role_labels.get(role, role)}: " + _render_fact_for_prompt(fact))
    return lines, used


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

    plan = calculation_plan(query, facts[: min(len(facts), 8)])
    if task in ("difference", "percent_change", "ratio", "comparison", "adjustment") and len(facts) >= 1:
        if plan.get("confidence", 0.0) >= 0.8 and plan.get("operands"):
            operand_lines, used = _render_plan_operands(plan, facts)
            lines.extend(operand_lines)
            if plan.get("trace"):
                lines.append("CALCULATION HINT: " + str(plan["trace"]))
            if used:
                facts = used
        else:
            lines.append("KG_CALCULATION_STATUS: insufficient structured support; use focus facts only.")
            facts = facts[: min(len(facts), 4)]
    elif task in ("sum", "average"):
        lines.append("KG_CALCULATION_STATUS: no high-confidence closed-form plan for this question.")
        facts = facts[:4]
    elif task == "factor_sum":
        if plan.get("confidence", 0.0) >= 0.8 and plan.get("operands"):
            operand_lines, used = _render_plan_operands(plan, facts)
            lines.extend(operand_lines)
            lines.append("CALCULATION HINT: " + str(plan["trace"]))
            factor_facts = _factor_like_facts(query, facts, max_facts=4)
            if factor_facts:
                lines.append("SUPPORTING FACTORS:")
                for i, f in enumerate(factor_facts, start=1):
                    lines.append(f"  - factor {i}: " + _render_fact_for_prompt(f))
            facts = (used + [f for f in factor_facts if f not in used])[:6] if factor_facts else used
        else:
            factor_facts = _factor_like_facts(query, facts, max_facts=4)
            if factor_facts:
                lines.append("KEY FACTORS:")
                for i, f in enumerate(factor_facts, start=1):
                    lines.append(f"  - factor {i}: " + _render_fact_for_prompt(f))
                facts = factor_facts
            else:
                lines.append("KG_CALCULATION_STATUS: factor bridge incomplete in retrieved facts.")
                facts = facts[: min(len(facts), 4)]
            lines.append("GUIDANCE: do not invent missing factor values or sum unrelated boundary totals.")

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
