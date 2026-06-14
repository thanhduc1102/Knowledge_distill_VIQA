"""Fact-level representation of a financial document (the "Fact Ledger").

A *fact* is the atomic unit of relevance in financial QA — relevance is indexed at
the fact level, not the document level (cf. ``core_method_update.md``). Each fact is a
tuple ``(concept, entity, period, value, unit, scale, provenance)`` extracted from a
table cell (or a number mentioned in narrative text).

This ledger is the SHARED knowledge-graph substrate used by BOTH:
  * retrieval  — equation-faithful constraint scoring + fact-level signals, and
  * generation — feeding the *precise* cells (not raw markdown) to the LLM, plus a
    deterministic verifier that checks the answer against ledger cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Fact:
    """An atomic financial fact extracted from a table cell or narrative number."""

    concept: str                 # line-item / row label, e.g. "net cash from operating activities"
    value: Optional[float]       # numeric value in display units (table scale)
    concept_canonical: Optional[str] = None  # canonical IFRS/GAAP/XBRL concept (C2), if recognised
    raw_text: str = ""           # original cell text, e.g. "$ 206588"
    period: Optional[str] = None # period token (fiscal year) if identifiable
    column_header: str = ""      # raw column header the value came from
    unit: str = ""               # "USD" | "%" | "shares" | "ratio" | ""
    scale: float = 1.0           # 1 | 1e3 | 1e6 | 1e9 (from "in thousands/millions/...")
    scale_label: str = ""        # human label of scale, e.g. "millions"
    row_idx: int = -1
    col_idx: int = -1
    source: str = "table"        # "table" | "text"
    doc_id: str = ""
    company: str = ""
    provenance: str = ""         # human-readable location, e.g. "doc=finqa_ctx_1 r3 c2 [2019]"

    @property
    def value_absolute(self) -> Optional[float]:
        """Value scaled to absolute units (e.g. millions -> raw dollars)."""
        if self.value is None:
            return None
        return self.value * self.scale

    def render(self) -> str:
        """Compact one-line rendering for an LLM prompt / debugging."""
        per = f" [{self.period}]" if self.period else ""
        unit = f" {self.unit}" if self.unit else ""
        sc = f" (in {self.scale_label})" if self.scale_label else ""
        val = self.raw_text if self.raw_text else (f"{self.value}" if self.value is not None else "—")
        return f"{self.concept}{per} = {val}{unit}{sc}"


@dataclass
class FactLedger:
    """All facts extracted from one document (+ its source metadata)."""

    doc_id: str
    facts: list[Fact] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    table_md: str = ""
    scale: float = 1.0
    scale_label: str = ""
    unit: str = ""

    # ------------------------------------------------------------------
    def table_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.source == "table"]

    def numeric_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.value is not None]

    def concept_set(self) -> set[str]:
        """Canonical IFRS/GAAP concepts present in this ledger (C2/C3)."""
        return {f.concept_canonical for f in self.facts if f.concept_canonical}

    def periods(self) -> list[str]:
        seen, out = set(), []
        for f in self.facts:
            if f.period and f.period not in seen:
                seen.add(f.period)
                out.append(f.period)
        return out

    def find_value(self, concept_substr: str, period: Optional[str] = None) -> Optional[Fact]:
        """Lookup a fact by (substring of concept) + optional period — used by verifier."""
        cs = concept_substr.lower()
        best = None
        for f in self.numeric_facts():
            if cs in f.concept.lower():
                if period is None or (f.period and str(period) in str(f.period)):
                    return f
                best = best or f
        return best

    def render_block(self, max_facts: int = 60) -> str:
        """Render the ledger as a compact, LLM-friendly fact list."""
        lines = []
        if self.scale_label:
            lines.append(f"(values in {self.scale_label}{' ' + self.unit if self.unit else ''})")
        for f in self.facts[:max_facts]:
            lines.append("  - " + f.render())
        return "\n".join(lines)
