"""Constraint-Aware Scoring: constraint score + entity score."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gsr_cacl.kg.data_structures import ConstraintKG


@dataclass
class ConstraintScoringResult:
    """Results from constraint scoring."""
    constraint_score: float        # 0–1 (higher = more consistent)
    violated_count: int
    total_count: int
    per_constraint_scores: list[float]


def compute_constraint_score(
    kg: ConstraintKG,
    epsilon: float = 1e-4,
) -> ConstraintScoringResult:
    """
    Compute constraint satisfaction score for a KG.

    For each accounting edge (u → v, ω):
        residual = |ω · v_u − v_v|
        score = exp(− residual / max(|v_v|, ε))

    Returns average score over all accounting edges.
    """
    acc_edges = kg.accounting_edges
    if not acc_edges:
        return ConstraintScoringResult(
            constraint_score=1.0,
            violated_count=0,
            total_count=0,
            per_constraint_scores=[],
        )

    node_map = {n.id: n for n in kg.nodes}
    scores = []
    violated = 0

    for edge in acc_edges:
        src_node = node_map.get(edge.src)
        tgt_node = node_map.get(edge.tgt)

        if src_node is None or tgt_node is None:
            continue
        if src_node.value is None or tgt_node.value is None:
            continue

        residual = abs(edge.omega * src_node.value - tgt_node.value)
        denom = max(abs(tgt_node.value), epsilon)
        edge_score = math.exp(-residual / denom)

        scores.append(edge_score)
        if edge_score < 0.5:
            violated += 1

    if not scores:
        return ConstraintScoringResult(
            constraint_score=1.0,
            violated_count=0,
            total_count=0,
            per_constraint_scores=[],
        )

    return ConstraintScoringResult(
        constraint_score=sum(scores) / len(scores),
        violated_count=violated,
        total_count=len(scores),
        per_constraint_scores=scores,
    )


def compute_entity_score(
    query_meta: dict,
    doc_meta: dict,
) -> float:
    """
    Compute entity matching score between query and document metadata.
    Returns float in [0, 1].
    """
    score = 0.0
    n_checks = 0

    q_company = str(query_meta.get("company_name", "")).lower().strip()
    d_company = str(doc_meta.get("company_name", "")).lower().strip()
    if q_company and d_company:
        n_checks += 1
        if q_company == d_company:
            score += 1.0
        elif q_company in d_company or d_company in q_company:
            score += 0.5

    q_year = str(query_meta.get("report_year", "")).strip()
    d_year = str(doc_meta.get("report_year", "")).strip()
    if q_year and d_year:
        n_checks += 1
        if q_year == d_year:
            score += 1.0
        else:
            try:
                if abs(int(q_year) - int(d_year)) <= 1:
                    score += 0.5
            except ValueError:
                pass

    q_sector = str(query_meta.get("company_sector", "")).lower().strip()
    d_sector = str(doc_meta.get("company_sector", "")).lower().strip()
    if q_sector and d_sector:
        n_checks += 1
        if q_sector == d_sector:
            score += 1.0
        elif q_sector in d_sector or d_sector in q_sector:
            score += 0.5

    if n_checks == 0:
        return 0.5
    return score / n_checks


def compute_equation_constraint_score(
    kg: ConstraintKG,
    epsilon: float = 1e-4,
) -> ConstraintScoringResult:
    """Equation-FAITHFUL constraint score (fixes the pairwise approximation, B5).

    The original ``compute_constraint_score`` scores every accounting edge ``(u→v, ω)``
    independently and averages, which silently splits a multi-operand identity such as
    ``Total = A + B + C`` into three pairwise checks. That is wrong: an identity is
    satisfied or violated *as a whole*. Here we GROUP accounting edges that share the
    same target ``v`` **and** the same ``constraint_name`` (the builder already emits one
    edge per operand of an identity, all within the same row), then evaluate the full
    equation::

        residual = | Σ_i ω_i · v_{u_i}  −  v_v |
        score    = exp( − residual / max(|v_v|, ε) )    ∈ (0, 1]

    Returns the average over all *equations* (not edges).
    """
    acc_edges = kg.accounting_edges
    if not acc_edges:
        return ConstraintScoringResult(1.0, 0, 0, [])

    node_map = {n.id: n for n in kg.nodes}

    # group operands by (target, constraint_name)
    groups: dict[tuple[str, str], list] = {}
    for e in acc_edges:
        groups.setdefault((e.tgt, e.constraint_name), []).append(e)

    scores: list[float] = []
    violated = 0
    for (tgt_id, _name), operands in groups.items():
        tgt = node_map.get(tgt_id)
        if tgt is None or tgt.value is None:
            continue
        lhs = 0.0
        n_ok = 0
        for e in operands:
            src = node_map.get(e.src)
            if src is None or src.value is None:
                continue
            lhs += e.omega * src.value
            n_ok += 1
        if n_ok == 0:
            continue
        residual = abs(lhs - tgt.value)
        edge_score = math.exp(-residual / max(abs(tgt.value), epsilon))
        scores.append(edge_score)
        if edge_score < 0.5:
            violated += 1

    if not scores:
        return ConstraintScoringResult(1.0, 0, 0, [])
    return ConstraintScoringResult(
        constraint_score=sum(scores) / len(scores),
        violated_count=violated,
        total_count=len(scores),
        per_constraint_scores=scores,
    )
