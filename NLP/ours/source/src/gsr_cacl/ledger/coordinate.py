"""Tabular Coordinate Grounding — answer a question by 2D cell addressing.

Diagnosis (scripts/research/symbolic_error_decomp.py): the symbolic answerer's accuracy is
governed by how tightly the table's 2D structure constrains operand selection. Tasks that use
BOTH axes (difference/%-change: one concept-row × two period-columns) hit 40-48% exact, while
1-axis tasks (lookup, sum) hit 8-34% — a 2-4× gap, retrieval-independent.

Breakthrough: treat every question as a coordinate-addressing problem on the table grid:

    answer cell = (row aligned to the question CONCEPT) × (column aligned to the question PERIOD)

Decomposing the hard joint fact-match into two easier 1-D alignments (row, column) and
combining them MULTIPLICATIVELY (a wrong cell must fail at least one axis) is exactly the
disentanglement the granularity/numeracy literature motivates — here justified by the table's
own structure. The Fact ledger already carries `row_idx / col_idx / column_header`, so the grid
is recoverable with no extra parsing and no training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.concepts import canonical_concept, concepts_in_text

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "the", "a", "an", "of", "in", "and", "or", "to", "is", "are", "was", "were", "for", "from",
    "with", "at", "by", "on", "as", "its", "that", "this", "what", "how", "much", "many", "did",
    "do", "does", "have", "has", "had", "be", "year", "fiscal", "report", "reported", "company",
    "during", "between", "value", "amount", "total",
})
_TOTAL_RE = re.compile(r"\b(total|overall|aggregate|combined)\b", re.I)


def _toks(s: str) -> frozenset[str]:
    return frozenset(t for t in _TOK.findall((s or "").lower())
                     if t not in _STOP and not t.isdigit() and len(t) > 1)


@dataclass
class CellHit:
    fact: Fact
    row_score: float
    col_score: float
    score: float          # row_score * col_score (∈[0,1])


class CoordinateGrounder:
    """2D row×column addressing over a document's Fact Ledger."""

    def __init__(self, ledger: FactLedger):
        self.row_label: dict[int, str] = {}
        self.col_header: dict[int, str] = {}
        self.cell: dict[tuple[int, int], Fact] = {}
        for f in ledger.numeric_facts():
            if f.row_idx is None or f.col_idx is None or f.row_idx < 0 or f.col_idx < 0:
                continue
            self.cell[(f.row_idx, f.col_idx)] = f
            self.row_label.setdefault(f.row_idx, f.concept or "")
            self.col_header.setdefault(f.col_idx, f.column_header or "")
        self._row_canon = {ri: canonical_concept(lbl) for ri, lbl in self.row_label.items()}

    # ----------------------------------------------------------------- axis scores
    def _row_score(self, ri: int, q_tokens: frozenset, q_concepts: set[str]) -> float:
        label = self.row_label.get(ri, "")
        rtoks = _toks(label)
        if not rtoks:
            return 0.0
        inter = len(q_tokens & rtoks)
        jac = inter / len(q_tokens | rtoks) if (q_tokens or rtoks) else 0.0
        containment = inter / len(rtoks)              # all row tokens present in question
        onto = 1.0 if (self._row_canon.get(ri) and self._row_canon[ri] in q_concepts) else 0.0
        return max(jac, 0.9 * containment, onto)

    def _col_score(self, cj: int, q_years: list[int]) -> float:
        header = self.col_header.get(cj, "")
        hyears = extract_years(header)
        if q_years:
            if any(y in q_years for y in hyears):
                return 1.0                            # exact period column
            if hyears:
                return 0.15                           # a different, explicit year → wrong column
            return 0.6                                # period-less value column (e.g. "amount")
        return 1.0                                    # question names no year → any value column

    # ----------------------------------------------------------------- addressing
    def address(self, question: str) -> Optional[CellHit]:
        if not self.cell:
            return None
        q_tokens = _toks(question)
        q_concepts = concepts_in_text(question)
        q_years = extract_years(question)
        best: Optional[CellHit] = None
        for (ri, cj), f in self.cell.items():
            rs = self._row_score(ri, q_tokens, q_concepts)
            if rs <= 0.0:
                continue
            cs = self._col_score(cj, q_years)
            sc = rs * cs
            if best is None or sc > best.score:
                best = CellHit(f, rs, cs, sc)
        return best

    def row_at_year(self, ri: int, year: int) -> Optional[Fact]:
        """The cell of row ``ri`` in the column whose header carries ``year``."""
        best = None
        for cj in self.col_header:
            if year in extract_years(self.col_header[cj]):
                f = self.cell.get((ri, cj))
                if f is not None:
                    return f
        return best

    def best_row(self, question: str) -> Optional[int]:
        q_tokens = _toks(question)
        q_concepts = concepts_in_text(question)
        scored = [(self._row_score(ri, q_tokens, q_concepts), ri) for ri in self.row_label]
        scored = [s for s in scored if s[0] > 0]
        return max(scored)[1] if scored else None

    def two_period_operands(self, question: str) -> Optional[tuple[Fact, Fact]]:
        """For Δ / %-change: fix the concept row, take the two requested (or latest two) periods."""
        ri = self.best_row(question)
        if ri is None:
            return None
        q_years = sorted(set(extract_years(question)))
        row_cells = [(cj, f) for (r, cj), f in self.cell.items() if r == ri]
        if len(q_years) >= 2:
            fo = self.row_at_year(ri, q_years[0])
            fn = self.row_at_year(ri, q_years[-1])
            if fo is not None and fn is not None and fo is not fn:
                return fo, fn
        # fallback: order this row's cells by column-header year, take the two extremes
        def cyr(cf):
            ys = extract_years(self.col_header.get(cf[0], ""))
            return ys[-1] if ys else cf[0]
        ordered = sorted(row_cells, key=cyr)
        if len(ordered) >= 2:
            return ordered[0][1], ordered[-1][1]
        return None


def coordinate_answer(question: str, ledger: FactLedger, task: str) -> Optional[dict]:
    """Answer via coordinate grounding. Returns {answer, operands, confidence, method} or None."""
    g = CoordinateGrounder(ledger)
    if not g.cell:
        return None

    if task in ("difference", "percent_change"):
        ops = g.two_period_operands(question)
        if ops is None:
            return None
        fo, fn = ops
        if fo.value is None or fn.value is None:
            return None
        if task == "percent_change":
            if not fo.value:
                return None
            ans = (fn.value - fo.value) / abs(fo.value)
        else:
            ans = fn.value - fo.value
        ri = g.best_row(question)
        conf = g._row_score(ri, _toks(question), concepts_in_text(question)) if ri is not None else 0.5
        return {"answer": ans, "operands": [fo, fn], "confidence": round(min(conf, 1.0), 3),
                "method": "coord_2period"}

    # lookup / single-cell tasks
    hit = g.address(question)
    if hit is None or hit.fact.value is None:
        return None
    return {"answer": hit.fact.value, "operands": [hit.fact],
            "confidence": round(hit.score, 3), "method": "coord_cell",
            "row_score": round(hit.row_score, 3), "col_score": round(hit.col_score, 3)}
