"""Structure-level financial knowledge graph.

This graph complements the existing fact ledger.  The ledger says *what* facts exist;
the structure graph says *where and how* those facts live in the document/table:

Document -> Table -> Row/Column -> Fact/Cell -> Concept/Period

It is intentionally lightweight and deterministic.  The contribution target is not a
generic GraphRAG entity graph, but an auditable structure graph for financial tables and
filing snippets.  It can be used in retrieval arbitration, evidence planning, and prompt
rendering for LLM support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from gsr_cacl.kg.fact_graph import FinancialFactGraph
from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ledger.select import infer_task_type, _tokens
from gsr_cacl.ontology.concepts import concepts_in_text


@dataclass
class StructureNode:
    id: str
    type: str
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructureEdge:
    src: str
    dst: str
    type: str
    weight: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructureSupport:
    score: float
    concept_coverage: float
    period_coverage: float
    row_col_alignment: float
    temporal_affordance: float
    arithmetic_affordance: float
    evidence_paths: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class FinancialStructureGraph:
    """Typed structure graph over one document's ledger."""

    def __init__(self, ledger: FactLedger):
        self.ledger = ledger
        self.nodes: dict[str, StructureNode] = {}
        self.edges: list[StructureEdge] = []
        self._fact_node: dict[int, str] = {}
        self._build()

    def _add_node(self, node_id: str, typ: str, label: str, **attrs) -> str:
        self.nodes.setdefault(node_id, StructureNode(node_id, typ, label, dict(attrs)))
        return node_id

    def _add_edge(self, src: str, dst: str, typ: str, weight: float = 1.0, **attrs) -> None:
        self.edges.append(StructureEdge(src, dst, typ, weight, dict(attrs)))

    def _build(self) -> None:
        doc = self._add_node(f"doc:{self.ledger.doc_id}", "document", self.ledger.doc_id,
                             meta=self.ledger.meta)
        table = self._add_node(f"table:{self.ledger.doc_id}", "table", "table",
                               scale=self.ledger.scale_label, unit=self.ledger.unit)
        self._add_edge(doc, table, "has_table")

        for i, f in enumerate(self.ledger.numeric_facts()):
            row = self._add_node(f"row:{f.row_idx}", "row", f.concept or f"row {f.row_idx}",
                                 row_idx=f.row_idx)
            col_label = f.column_header or (str(f.period) if f.period else f"col {f.col_idx}")
            col = self._add_node(f"col:{f.col_idx}", "column", col_label, col_idx=f.col_idx,
                                 period=f.period)
            fact_id = self._add_node(
                f"fact:{i}", "fact", f.concept or "fact",
                value=f.value, raw_text=f.raw_text, period=f.period, row_idx=f.row_idx,
                col_idx=f.col_idx, unit=f.unit, scale=f.scale_label, source=f.source,
                provenance=f.provenance, concept_canonical=f.concept_canonical,
            )
            self._fact_node[id(f)] = fact_id
            self._add_edge(table, row, "has_row")
            self._add_edge(table, col, "has_column")
            self._add_edge(row, fact_id, "row_contains_fact")
            self._add_edge(col, fact_id, "column_contains_fact")
            if f.concept_canonical:
                c = self._add_node(f"concept:{f.concept_canonical}", "concept", f.concept_canonical)
                self._add_edge(fact_id, c, "has_concept")
            if f.period:
                p = self._add_node(f"period:{f.period}", "period", str(f.period))
                self._add_edge(fact_id, p, "has_period")

        # Add reasoning edges from the fact graph as structure-level support edges.
        fg = FinancialFactGraph(self.ledger)
        for e in fg.temporal_edges:
            a = self._fact_node.get(id(e.fact_a))
            b = self._fact_node.get(id(e.fact_b))
            if a and b:
                self._add_edge(a, b, "same_concept_temporal", concept=e.concept,
                               period_a=e.period_a, period_b=e.period_b)
        for e in fg.identity_edges:
            target_nodes = [self._fact_node.get(id(f)) for f in e.facts if f.concept_canonical == e.target]
            operand_nodes = [self._fact_node.get(id(f)) for f in e.facts if f.concept_canonical != e.target]
            for op in operand_nodes:
                for tgt in target_nodes:
                    if op and tgt:
                        self._add_edge(op, tgt, "accounting_identity_support",
                                       satisfied=e.satisfied, residual=e.residual, period=e.period)

    @property
    def fact_nodes(self) -> list[StructureNode]:
        return [n for n in self.nodes.values() if n.type == "fact"]

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self.nodes.values():
            counts[n.type] = counts.get(n.type, 0) + 1
        edge_counts: dict[str, int] = {}
        for e in self.edges:
            edge_counts[e.type] = edge_counts.get(e.type, 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            **{f"nodes_{k}": v for k, v in counts.items()},
            **{f"edges_{k}": v for k, v in edge_counts.items()},
        }


def build_structure_graph(ledger: FactLedger) -> FinancialStructureGraph:
    return FinancialStructureGraph(ledger)


def _concept_match_score(query: str, fact: StructureNode) -> float:
    q_tokens = _tokens(query)
    label_tokens = _tokens(fact.label)
    if not label_tokens:
        return 0.0
    lexical = len(q_tokens & label_tokens) / max(len(label_tokens) ** 0.5, 1.0)
    q_concepts = concepts_in_text(query)
    canon = fact.attrs.get("concept_canonical")
    onto = 1.0 if canon and canon in q_concepts else 0.0
    return max(lexical, onto)


def score_structure_support(query: str, graph: FinancialStructureGraph) -> StructureSupport:
    """Query-conditioned structural support score for one document graph."""
    facts = graph.fact_nodes
    if not facts:
        return StructureSupport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], ["no numeric facts"])

    q_years = sorted(set(extract_years(query)))
    q_concepts = concepts_in_text(query)
    task = infer_task_type(query)

    concept_hits = []
    period_hits = []
    row_col = []
    evidence_paths = []
    for f in facts:
        cscore = _concept_match_score(query, f)
        concept_hits.append(cscore)
        per = f.attrs.get("period")
        if q_years:
            try:
                period_hits.append(1.0 if per and int(float(str(per))) in q_years else 0.0)
            except (TypeError, ValueError):
                period_hits.append(0.0)
        else:
            period_hits.append(1.0)
        # Multiplicative row/column support: wrong evidence should fail one axis.
        row_col.append(cscore * period_hits[-1])

    concept_coverage = 1.0 if (q_concepts and max(concept_hits) >= 0.75) else (
        max(concept_hits) if concept_hits else 0.0
    )
    if not q_concepts:
        concept_coverage = max(concept_hits) if concept_hits else 0.5
    period_coverage = (
        len({str(y) for y in q_years} & {str(f.attrs.get("period")) for f in facts}) / max(len(q_years), 1)
        if q_years else 1.0
    )
    row_col_alignment = max(row_col) if row_col else 0.0

    temporal_edges = [e for e in graph.edges if e.type == "same_concept_temporal"]
    identity_edges = [e for e in graph.edges if e.type == "accounting_identity_support"]
    temporal_affordance = 0.0
    if task in {"difference", "percent_change", "comparison"}:
        if q_years and len(q_years) >= 2:
            for e in temporal_edges:
                ps = {str(e.attrs.get("period_a")), str(e.attrs.get("period_b"))}
                if {str(q_years[0]), str(q_years[-1])} <= ps:
                    temporal_affordance = 1.0
                    break
        else:
            temporal_affordance = min(1.0, len(temporal_edges) / 2.0)
    arithmetic_affordance = 0.0
    if task in {"sum", "ratio", "average", "factor_sum"}:
        arithmetic_affordance = min(1.0, len(facts) / 4.0)
    elif task in {"difference", "percent_change"}:
        arithmetic_affordance = temporal_affordance
    if identity_edges:
        arithmetic_affordance = max(arithmetic_affordance, min(1.0, len(identity_edges) / 3.0))

    # Query-conditioned structure score: coverage first, then affordance.
    score = (
        0.35 * concept_coverage
        + 0.25 * period_coverage
        + 0.25 * row_col_alignment
        + 0.10 * temporal_affordance
        + 0.05 * arithmetic_affordance
    )
    score = float(max(0.0, min(2.0, score)))

    best = sorted(facts, key=lambda f: _concept_match_score(query, f), reverse=True)[:4]
    for f in best:
        evidence_paths.append(
            "document -> table -> "
            f"row[{f.attrs.get('row_idx')}:{f.label}] -> "
            f"col[{f.attrs.get('col_idx')}:{f.attrs.get('period') or ''}] -> "
            f"value[{f.attrs.get('raw_text') or f.attrs.get('value')}]"
        )

    reasons = [
        f"concept_coverage={concept_coverage:.3f}",
        f"period_coverage={period_coverage:.3f}",
        f"row_col_alignment={row_col_alignment:.3f}",
    ]
    if temporal_affordance:
        reasons.append("has temporal structure for multi-period reasoning")
    if arithmetic_affordance:
        reasons.append("has arithmetic operand structure")

    return StructureSupport(
        score=score,
        concept_coverage=concept_coverage,
        period_coverage=period_coverage,
        row_col_alignment=row_col_alignment,
        temporal_affordance=temporal_affordance,
        arithmetic_affordance=arithmetic_affordance,
        evidence_paths=evidence_paths,
        reasons=reasons,
    )


def render_structure_block(query: str, graph: FinancialStructureGraph, max_paths: int = 4) -> str:
    support = score_structure_support(query, graph)
    lines = [
        "STRUCTURE_GRAPH_SUPPORT:",
        f"  score={support.score:.4f}",
        "  " + "; ".join(support.reasons),
    ]
    if support.evidence_paths:
        lines.append("STRUCTURE_EVIDENCE_PATHS:")
        for p in support.evidence_paths[:max_paths]:
            lines.append(f"  - {p}")
    return "\n".join(lines)
