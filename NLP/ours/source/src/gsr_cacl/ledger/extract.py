"""Extract a :class:`FactLedger` from a financial document.

Orientation-aware: in T²-RAGBench tables the **row label** is the concept/line-item
(e.g. "net cash from operating activities") and the **column header** is the period or
amount-type (e.g. "2019", "amount ( in millions )"). The generic KG builder treats the
column header as the only header and loses the row label — here we recover the true
fact orientation, which is what makes the ledger usable for precise generation.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from gsr_cacl.kg.parser import parse_markdown_rows
from gsr_cacl.ledger.numeric import parse_financial_number, extract_years
from gsr_cacl.ledger.fact import Fact, FactLedger

# ----------------------------------------------------------------------
# Scale / unit detection
# ----------------------------------------------------------------------

_SCALE_PATTERNS = [
    (re.compile(r"\bin\s+billions?\b", re.I), 1e9, "billions"),
    (re.compile(r"\bin\s+millions?\b", re.I), 1e6, "millions"),
    (re.compile(r"\bin\s+thousands?\b", re.I), 1e3, "thousands"),
    (re.compile(r"\$\s*millions?\b", re.I), 1e6, "millions"),
    (re.compile(r"\$\s*thousands?\b", re.I), 1e3, "thousands"),
]


def detect_scale(text: str) -> tuple[float, str]:
    for pat, scale, label in _SCALE_PATTERNS:
        if pat.search(text or ""):
            return scale, label
    return 1.0, ""


def detect_unit(text: str) -> str:
    t = (text or "").lower()
    if "%" in t or "percent" in t or "margin" in t or "rate" in t:
        return "%"
    if "shares" in t or "share" in t:
        return "shares"
    if "$" in t or "usd" in t or "dollar" in t:
        return "USD"
    return ""


# ----------------------------------------------------------------------
# Column-role detection
# ----------------------------------------------------------------------

def _is_index_column(col_cells: list[str]) -> bool:
    """A column of sequential integers 0,1,2,... (pandas to_markdown index)."""
    vals = []
    for c in col_cells:
        v = parse_financial_number(c)
        if v is None or v != int(v):
            return False
        vals.append(int(v))
    if len(vals) < 2:
        return False
    return vals == list(range(vals[0], vals[0] + len(vals)))


def _numeric_ratio(col_cells: list[str]) -> float:
    if not col_cells:
        return 0.0
    n = sum(1 for c in col_cells if parse_financial_number(c) is not None)
    return n / len(col_cells)


def _period_from_header(header: str) -> Optional[str]:
    yrs = extract_years(header)
    if yrs:
        return str(yrs[-1])  # most specific / latest year token in the header
    return None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def extract_ledger_from_table(
    table_md: str,
    doc_id: str = "",
    meta: Optional[dict[str, Any]] = None,
    caption: str = "",
) -> FactLedger:
    """Build a FactLedger from a markdown table (orientation-aware)."""
    meta = meta or {}
    company = str(meta.get("company_name", ""))
    ledger = FactLedger(doc_id=doc_id, meta=meta, table_md=table_md)

    rows = parse_markdown_rows(table_md)
    if len(rows) < 2:
        return ledger

    header_row = rows[0]
    data_rows = rows[1:]
    n_cols = max(len(header_row), max((len(r) for r in data_rows), default=0))

    def col(j: int) -> list[str]:
        return [r[j] if j < len(r) else "" for r in data_rows]

    # Scale / unit from headers + caption
    head_blob = " ".join(header_row) + " " + caption + " " + table_md[:400]
    scale, scale_label = detect_scale(head_blob)
    ledger.scale, ledger.scale_label = scale, scale_label

    # Identify columns
    index_cols = {j for j in range(n_cols) if _is_index_column(col(j))}
    candidate_label = None
    for j in range(n_cols):
        if j in index_cols:
            continue
        if _numeric_ratio(col(j)) < 0.4:      # mostly textual → row-label column
            candidate_label = j
            break
    if candidate_label is None:
        # fall back to first non-index column
        candidate_label = next((j for j in range(n_cols) if j not in index_cols), 0)

    value_cols = [
        j for j in range(n_cols)
        if j not in index_cols and j != candidate_label and _numeric_ratio(col(j)) >= 0.4
    ]
    if not value_cols:  # degenerate: treat every non-label, non-index col as value
        value_cols = [j for j in range(n_cols) if j not in index_cols and j != candidate_label]

    # Build facts: one per (row, value-column)
    for ri, row in enumerate(data_rows):
        concept = (row[candidate_label] if candidate_label < len(row) else "").strip()
        if not concept:
            continue
        for cj in value_cols:
            raw = (row[cj] if cj < len(row) else "").strip()
            val = parse_financial_number(raw)
            if val is None:
                continue
            header = header_row[cj] if cj < len(header_row) else ""
            period = _period_from_header(header) or _period_from_header(concept)
            unit = detect_unit(header) or detect_unit(concept) or ledger.unit
            ledger.facts.append(Fact(
                concept=concept,
                value=val,
                raw_text=raw,
                period=period,
                column_header=header.strip(),
                unit=unit,
                scale=scale,
                scale_label=scale_label,
                row_idx=ri,
                col_idx=cj,
                source="table",
                doc_id=doc_id,
                company=company,
                provenance=f"{doc_id} r{ri} c{cj}" + (f" [{period}]" if period else ""),
            ))
    return ledger


def extract_table_md(context: str) -> str:
    """Fallback: pull the first markdown table block out of a context string
    (for TAT-DQA which has no dedicated `table` field)."""
    lines = (context or "").split("\n")
    block: list[str] = []
    started = False
    for ln in lines:
        s = ln.strip()
        if "|" in s:
            started = True
            block.append(s)
        elif started:
            break
    return "\n".join(block) if len(block) >= 2 else ""


def extract_ledger(
    *,
    table_md: str = "",
    context: str = "",
    doc_id: str = "",
    meta: Optional[dict[str, Any]] = None,
    add_text_facts: bool = True,
    max_text_facts: int = 40,
) -> FactLedger:
    """High-level entry: prefer the clean `table` field, fall back to context, and
    optionally add narrative numbers as low-priority text facts.
    """
    meta = meta or {}
    tmd = table_md or extract_table_md(context)
    ledger = extract_ledger_from_table(tmd, doc_id=doc_id, meta=meta, caption=context[:300])

    if add_text_facts and context:
        ledger.facts.extend(_text_number_facts(context, doc_id, meta, max_text_facts))
    return ledger


_NUM_TOKEN_RE = re.compile(r"[-+]?\$?\s?\(?\s?\d[\d,]*\.?\d*\s?\)?%?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def _looks_like_year(v: float) -> bool:
    return v == int(v) and 1950 <= int(v) <= 2049


def _text_number_facts(context: str, doc_id: str, meta: dict, limit: int) -> list[Fact]:
    """Extract numbers mentioned in narrative text, each with a TIGHT concept window
    (the preceding noun phrase) so the fact is specific rather than a whole sentence."""
    facts: list[Fact] = []
    company = str(meta.get("company_name", ""))
    # Drop every line that contains a table pipe so we never re-mine table numbers.
    text = "\n".join(ln for ln in context.split("\n") if "|" not in ln)
    seen_vals: set[float] = set()

    for m in _NUM_TOKEN_RE.finditer(text):
        v = parse_financial_number(m.group(0))
        if v is None or _looks_like_year(v):
            continue
        rounded = round(v, 4)
        if rounded in seen_vals:
            continue
        # concept = up to 8 words immediately before the number
        prefix = text[max(0, m.start() - 80):m.start()]
        words = _WORD_RE.findall(prefix)[-8:]
        concept = " ".join(words).strip()
        if len(concept) < 4:
            continue
        seen_vals.add(rounded)
        yrs = extract_years(text[max(0, m.start() - 80):m.end() + 20])
        facts.append(Fact(
            concept=concept, value=v, raw_text=m.group(0).strip(),
            period=str(yrs[0]) if yrs else None,
            source="text", doc_id=doc_id, company=company,
            provenance=f"{doc_id} text",
        ))
        if len(facts) >= limit:
            break
    return facts
