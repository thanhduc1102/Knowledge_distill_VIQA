"""FinancialFactGraph — the typed knowledge graph over a document's Fact Ledger.

This is the "complete KG" the project targets (docs/RESEARCH_AAAI27.md §4): NOT a generic
entity-relation graph (GraphRAG-style, brittle on numeric tables) but a *typed financial
fact graph* where

  * **nodes**  = facts  (concept, entity, period, value, unit/scale)  — from ``FactLedger``,
  * **edges**  = (a) ACCOUNTING-IDENTITY edges (Revenue−COGS=GrossProfit, Assets=Liab+Equity,
                     …) linking facts that should satisfy a known equation at the same period
                     — used as a *verifier* and for *transparency*, never as a ranking signal;
                 (b) TEMPORAL edges linking the same concept across periods — used for
                     multi-step (Δ / % change) reasoning and operand selection.

It serves the three KG-for-generator goals (training-free, inference-time):
  G1  disambiguate the noisy top-k: score each doc's subgraph against the question intent.
  G2  offload the LLM: hand it the exact answer facts (+ identity-derived operands), not raw
      markdown, so it neither re-parses the table nor invents numbers.
  G3  transparency: every answer carries a provenance trace (which cells, which identity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ontology.concepts import IDENTITIES


def _period_key(p: Optional[str]) -> str:
    return str(p) if p else "_"


def _abs_value(f: Fact) -> Optional[float]:
    return f.value_absolute if f.value_absolute is not None else f.value


@dataclass
class IdentityEdge:
    """An accounting identity instantiated at one period (a verifiable constraint)."""
    target: str                       # canonical concept, e.g. "GrossProfit"
    components: list[tuple[str, int]]  # [(concept, +1/-1), ...]
    period: str
    target_value: float
    lhs_value: float
    residual: float
    satisfied: bool
    facts: list[Fact] = field(default_factory=list)   # provenance: facts involved

    def explain(self) -> str:
        terms = " ".join(f"{'+' if w > 0 else '-'} {c}" for c, w in self.components).lstrip("+ ")
        status = "OK" if self.satisfied else f"VIOLATED (residual={self.residual:.4g})"
        return f"[{self.period}] {self.target} = {terms}  →  {status}"


@dataclass
class TemporalEdge:
    """Same concept observed at two periods (enables Δ / %-change reasoning)."""
    concept: str
    period_a: str
    period_b: str
    fact_a: Fact
    fact_b: Fact


class FinancialFactGraph:
    def __init__(self, ledger: FactLedger, identity_tol: float = 0.02):
        self.ledger = ledger
        self.facts: list[Fact] = list(ledger.facts)
        self.identity_tol = identity_tol
        # index: (canonical_concept, period) -> first table fact
        self._by_cp: dict[tuple[str, str], Fact] = {}
        for f in self.facts:
            if f.concept_canonical is None or f.value is None:
                continue
            key = (f.concept_canonical, _period_key(f.period))
            self._by_cp.setdefault(key, f)
        self.identity_edges: list[IdentityEdge] = self._build_identity_edges()
        self.temporal_edges: list[TemporalEdge] = self._build_temporal_edges()

    # ------------------------------------------------------------------ edges
    def _build_identity_edges(self) -> list[IdentityEdge]:
        edges: list[IdentityEdge] = []
        periods = {p for (_, p) in self._by_cp} or {"_"}
        for target, operands in IDENTITIES:
            for per in periods:
                tf = self._by_cp.get((target, per))
                if tf is None:
                    continue
                lhs, involved, n_ok = 0.0, [tf], 0
                for op, omega in operands:
                    of = self._by_cp.get((op, per))
                    if of is None:
                        continue
                    ov = _abs_value(of)
                    if ov is None:
                        continue
                    lhs += omega * ov
                    involved.append(of)
                    n_ok += 1
                if n_ok < max(2, len(operands)):
                    continue
                tv = _abs_value(tf) or 0.0
                residual = abs(lhs - tv)
                satisfied = residual <= self.identity_tol * max(abs(tv), 1.0)
                edges.append(IdentityEdge(target, operands, per, tv, lhs, residual, satisfied, involved))
        return edges

    def _build_temporal_edges(self) -> list[TemporalEdge]:
        by_concept: dict[str, list[Fact]] = {}
        for f in self.facts:
            key = f.concept_canonical or (f.concept or "").lower().strip()
            if not key or f.value is None or not f.period:
                continue
            by_concept.setdefault(key, []).append(f)
        edges: list[TemporalEdge] = []
        for key, fs in by_concept.items():
            def yr(f):
                try:
                    return int(float(str(f.period)))
                except (ValueError, TypeError):
                    return 0
            fs = sorted(fs, key=yr)
            for a, b in zip(fs, fs[1:]):
                if yr(a) and yr(b) and yr(a) != yr(b):
                    edges.append(TemporalEdge(key, str(a.period), str(b.period), a, b))
        return edges

    # ------------------------------------------------------------------ verify (G3)
    def verify(self) -> dict:
        """Internal accounting consistency — for transparency and inference-time filtering."""
        if not self.identity_edges:
            return {"score": 1.0, "n_edges": 0, "n_violated": 0, "edges": []}
        sat = [e for e in self.identity_edges if e.satisfied]
        score = sum(math.exp(-e.residual / max(abs(e.target_value), 1.0)) for e in self.identity_edges)
        return {
            "score": round(score / len(self.identity_edges), 4),
            "n_edges": len(self.identity_edges),
            "n_violated": len(self.identity_edges) - len(sat),
            "edges": [e.explain() for e in self.identity_edges],
        }

    # ------------------------------------------------------------------ lookup
    def find(self, concept_canonical: str, period: Optional[str] = None) -> Optional[Fact]:
        if period is not None:
            f = self._by_cp.get((concept_canonical, _period_key(period)))
            if f is not None:
                return f
        for (c, _), f in self._by_cp.items():
            if c == concept_canonical:
                return f
        return None

    def concept_periods(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for (c, p) in self._by_cp:
            out.setdefault(c, []).append(p)
        return out


def build_fact_graph(ledger: FactLedger) -> FinancialFactGraph:
    return FinancialFactGraph(ledger)
