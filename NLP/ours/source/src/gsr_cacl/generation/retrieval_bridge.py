"""Retrieval → Generator bridge: a TRAINING-FREE KG layer that does the heavy lifting.

This module is the answer to the three KG-for-generator requirements (the generator LLM
is NOT trained — the KG carries the load at inference time):

  G1 — Focus the right document among the noisy top-k.
       MRR@3 hands the generator k=3 docs, ≥2 of which are noise (same company, look-alike).
       For each retrieved doc we build its :class:`FinancialFactGraph` and score how well its
       facts answer the question intent; the score *margin* yields a confidence tier
       (TRUST_TOP1 / PREFER_TOP1 / REVIEW) and a re-ordering, so the LLM is pointed at the
       evidence rather than left to guess.

  G2 — Offload the LLM.  Instead of raw markdown tables, the LLM receives the *exact* answer
       facts (concept · entity · period · value), the inferred task + formula, the selected
       OPERANDS, and a deterministic CALCULATION HINT — so it neither re-parses tables nor
       invents numbers (the hallucination / numeracy failure mode).

  G3 — Transparency.  Every pack carries a provenance trace: which cell each number came from
       (doc · row · col · period) and an accounting-identity verification of the source doc.

No model weights, no training; pure symbolic reasoning over the typed fact graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from gsr_cacl.kg.fact_graph import FinancialFactGraph
from gsr_cacl.kg.structure_graph import build_structure_graph, render_structure_block, score_structure_support
from gsr_cacl.ledger.extract import extract_ledger, extract_ledger_from_table
from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.select import (
    build_evidence_block,
    calculation_plan,
    infer_task_type,
    score_fact,
    select_facts,
    task_formula,
    _tokens,
)
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.ledger.numeric import numbers_close

# Tiers from the top-1/top-2 evidence-score margin (relative).
_TRUST = 0.45
_PREFER = 0.15
_RANK_PRIOR_WEIGHT = 4.0


@dataclass
class DocEvidence:
    doc_id: str
    company: str
    score: float
    ledger: FactLedger
    graph: FinancialFactGraph
    original_rank: int = -1
    retrieval_score: float = 0.0
    arbitration_score: float = 0.0
    structure_score: float = 0.0
    structure_paths: list[str] = field(default_factory=list)
    top_facts: list[Fact] = field(default_factory=list)
    matched_concepts: list[str] = field(default_factory=list)
    matched_years: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class EvidencePack:
    question: str
    task: str
    formula: str
    tier: str                       # TRUST_TOP1 | PREFER_TOP1 | REVIEW
    margin: float
    ranked: list[DocEvidence]       # re-ordered by KG evidence score (best first)
    evidence_block: str             # G2: LLM-ready fact block (operands + calc hint)
    provenance: list[dict]          # G3: per-fact cell trace
    verification: dict              # G3: accounting-identity check of the chosen doc
    selected_facts: list[Fact] = field(default_factory=list)
    calculation: dict[str, Any] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task, "formula": self.formula, "tier": self.tier,
            "margin": round(self.margin, 4),
            "ranked": [{
                "doc_id": d.doc_id,
                "company": d.company,
                "kg_score": round(d.score, 4),
                "arbitration_score": round(d.arbitration_score, 4),
                "structure_score": round(d.structure_score, 4),
                "retrieval_score": round(float(d.retrieval_score), 4),
                "original_rank": d.original_rank,
                "matched_concepts": d.matched_concepts,
                "matched_years": d.matched_years,
                "reasons": d.reasons,
            } for d in self.ranked],
            "calculation": self.calculation,
            "conflicts": self.conflicts,
            "evidence_block": self.evidence_block,
            "provenance": self.provenance,
            "verification": self.verification,
        }


def _ledger_for(rec: dict) -> FactLedger:
    table = rec.get("table") or ""
    meta = rec.get("meta") or rec.get("metadata") or rec.get("query_meta") or {}
    doc_id = str(rec.get("context_id") or rec.get("id") or "")
    context = rec.get("page_content", "") or ""
    try:
        if table:
            return extract_ledger_from_table(table, doc_id=doc_id, meta=meta, caption=context[:300])
        return extract_ledger(context=context, doc_id=doc_id, meta=meta)
    except Exception:
        return FactLedger(doc_id=doc_id, facts=[], meta=meta)


def _task_affordance(task: str, q_years: list[int], ledger: FactLedger,
                     graph: FinancialFactGraph) -> tuple[float, list[str]]:
    """Does the graph contain the shape of evidence the question needs?"""
    reasons: list[str] = []
    bonus = 0.0
    numeric = ledger.numeric_facts()
    if task in {"difference", "percent_change"}:
        if graph.temporal_edges:
            bonus += 0.35
            reasons.append("has temporal same-concept facts")
        if len(q_years) >= 2:
            hit_years = set()
            for f in numeric:
                if not f.period:
                    continue
                try:
                    if int(float(str(f.period))) in q_years:
                        hit_years.add(str(f.period))
                except (TypeError, ValueError):
                    pass
            if len(hit_years) >= 2:
                bonus += 0.25
                reasons.append("covers requested year pair")
    elif task == "ratio":
        by_period: dict[str, int] = {}
        for f in numeric:
            by_period[str(f.period or "_")] = by_period.get(str(f.period or "_"), 0) + 1
        if any(n >= 2 for n in by_period.values()):
            bonus += 0.25
            reasons.append("has multiple operands in a period")
    elif task in {"sum", "average"} and len(numeric) >= 2:
        bonus += 0.2
        reasons.append("has multiple numeric operands")
    if graph.identity_edges:
        sat = sum(1 for e in graph.identity_edges if e.satisfied)
        bonus += min(0.15, 0.03 * sat)
        if sat:
            reasons.append(f"{sat} accounting identity checks hold")
    return bonus, reasons


def _doc_evidence_score(
    query: str,
    graph: FinancialFactGraph,
) -> tuple[float, list[Fact], list[str], list[str], list[str]]:
    """G1: how strongly this doc's facts answer the question.

    This is deliberately query-conditioned and graph-aware. It rewards exact fact
    relevance, requested concept/period coverage, and whether the graph has the
    operand structure needed by the detected task.
    """
    ledger = graph.ledger
    q_tokens = _tokens(query)
    q_years = extract_years(query)
    q_concepts = concepts_in_text(query)
    scored = [(score_fact(f, q_tokens, q_years, q_concepts), f) for f in ledger.numeric_facts()]
    if not scored:
        return 0.0, [], [], [], ["no numeric facts extracted"]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:3]
    top_facts = [f for _, f in top]
    score = float(sum(max(s, 0.0) for s, _ in top))

    doc_concepts = ledger.concept_set()
    matched_concepts = sorted(q_concepts & doc_concepts)
    if q_concepts:
        score += 0.8 * (len(matched_concepts) / max(len(q_concepts), 1))

    doc_years = set(ledger.periods())
    matched_years = sorted(str(y) for y in q_years if str(y) in doc_years)
    if q_years:
        score += 0.35 * (len(matched_years) / max(len(q_years), 1))

    task = infer_task_type(query)
    task_bonus, reasons = _task_affordance(task, q_years, ledger, graph)
    score += task_bonus

    if matched_concepts:
        reasons.append("matches concept: " + ", ".join(matched_concepts[:3]))
    if matched_years:
        reasons.append("matches period: " + ", ".join(matched_years[:3]))
    if top_facts:
        reasons.append("top fact: " + (top_facts[0].concept or ""))

    return score, top_facts, matched_concepts, matched_years, reasons


def _provenance(facts: list[Fact]) -> list[dict]:
    out = []
    for f in facts:
        out.append({
            "concept": f.concept,
            "concept_canonical": f.concept_canonical,
            "period": f.period,
            "value": f.value,
            "raw_text": f.raw_text,
            "unit": f.unit,
            "scale_label": f.scale_label,
            "cell": f"{f.doc_id} r{f.row_idx} c{f.col_idx}"
                    + (f" [{f.period}]" if f.period else ""),
            "source": f.source,
        })
    return out


def _fact_conflicts(docs: list[DocEvidence], max_conflicts: int = 8) -> list[dict[str, Any]]:
    """Detect same concept/period facts with different values across close top-k docs."""
    by_key: dict[tuple[str, str], list[tuple[str, Fact]]] = {}
    for d in docs[:3]:
        for f in d.top_facts:
            key = (f.concept_canonical or (f.concept or "").lower(), str(f.period or "_"))
            if key[0] and f.value is not None:
                by_key.setdefault(key, []).append((d.doc_id, f))

    out: list[dict[str, Any]] = []
    for (concept, period), vals in by_key.items():
        if len(vals) < 2:
            continue
        if len({doc for doc, _ in vals}) < 2:
            continue
        base = vals[0][1].value
        if base is None:
            continue
        different = [(doc, f) for doc, f in vals[1:] if f.value is not None and not numbers_close(base, f.value)]
        if not different:
            continue
        out.append({
            "concept": concept,
            "period": period,
            "values": [{"doc_id": doc, "value": f.value, "provenance": f.provenance} for doc, f in vals],
        })
        if len(out) >= max_conflicts:
            break
    return out


def _conflict_matches_calculation(conflict: dict[str, Any], calc: dict[str, Any]) -> bool:
    keys = set()
    for op in calc.get("operands") or []:
        concept = op.get("concept_canonical") or str(op.get("concept") or "").lower()
        period = str(op.get("period") or "_")
        if concept:
            keys.add((concept, period))
    ckey = (str(conflict.get("concept") or ""), str(conflict.get("period") or "_"))
    return ckey in keys


def build_evidence_pack(question: str, retrieved: list[dict], top_n_facts: int = 6) -> EvidencePack:
    """Build the KG evidence pack from the top-k retrieved records (each: table+meta+id)."""
    docs: list[DocEvidence] = []
    for rank, rec in enumerate(retrieved, start=1):
        ledger = _ledger_for(rec)
        graph = FinancialFactGraph(ledger)
        score, top_facts, matched_concepts, matched_years, reasons = _doc_evidence_score(question, graph)
        structure_graph = build_structure_graph(ledger)
        structure_support = score_structure_support(question, structure_graph)
        score = score + 0.5 * structure_support.score
        reasons = reasons + [f"structure: {r}" for r in structure_support.reasons[:3]]
        meta = rec.get("meta") or rec.get("metadata") or {}
        arbitration_score = score + (_RANK_PRIOR_WEIGHT / max(rank, 1))
        docs.append(DocEvidence(
            doc_id=str(rec.get("context_id") or rec.get("id") or ""),
            company=str(meta.get("company_name", "") or ""),
            score=score,
            ledger=ledger,
            graph=graph,
            original_rank=rank,
            retrieval_score=float(rec.get("score", 0.0) or 0.0),
            arbitration_score=arbitration_score,
            structure_score=structure_support.score,
            structure_paths=structure_support.evidence_paths[:4],
            top_facts=top_facts,
            matched_concepts=matched_concepts,
            matched_years=matched_years,
            reasons=reasons,
        ))

    docs.sort(key=lambda d: (d.arbitration_score, d.score, d.retrieval_score, -d.original_rank), reverse=True)
    s0 = docs[0].arbitration_score if docs else 0.0
    s1 = docs[1].arbitration_score if len(docs) > 1 else 0.0
    margin = (s0 - s1) / s0 if s0 > 1e-9 else 0.0
    tier = "TRUST_TOP1" if margin >= _TRUST else ("PREFER_TOP1" if margin >= _PREFER else "REVIEW")

    # G2: evidence block. TRUST/PREFER → facts from the top doc only; REVIEW → top-2 docs.
    src_docs = docs[:1] if tier != "REVIEW" else docs[:2]
    ledgers = [d.ledger for d in src_docs]
    selected = select_facts(question, ledgers, top_n=top_n_facts)
    calc = calculation_plan(question, selected)
    evidence_block = build_evidence_block(question, ledgers, top_n=top_n_facts, facts=selected)
    conflicts = _fact_conflicts(docs)

    pack = EvidencePack(
        question=question,
        task=infer_task_type(question),
        formula=task_formula(question, infer_task_type(question)),
        tier=tier, margin=margin, ranked=docs,
        evidence_block=evidence_block,
        provenance=_provenance(selected),                 # G3
        verification=docs[0].graph.verify() if docs else {"score": 1.0, "n_edges": 0},  # G3
        selected_facts=selected,
        calculation=calc,
        conflicts=conflicts,
    )
    return pack


def render_prompt_context(pack: EvidencePack) -> str:
    """The final context string handed to a (frozen) LLM — evidence + tier guidance."""
    guide = {
        "TRUST_TOP1": "The first document is the confident match; answer from its facts below.",
        "PREFER_TOP1": "The first document is the likely match; prefer its facts, sanity-check the others.",
        "REVIEW": "Two documents are close; choose the facts whose concept+period match the question.",
    }[pack.tier]
    v = pack.verification
    vline = (f"(source accounting check: {v['n_edges']-v['n_violated']}/{v['n_edges']} identities hold)"
             if v.get("n_edges") else "")
    lines = [f"GUIDANCE [{pack.tier}]: {guide} {vline}".strip()]
    if pack.ranked:
        best = pack.ranked[0]
        lines.append(
            f"KG_SELECTED_DOC: {best.doc_id} "
            f"(arbitration_score={best.arbitration_score:.4g}, "
            f"kg_score={best.score:.4g}, original_rank={best.original_rank})"
        )
        if best.reasons:
            lines.append("KG_SELECTION_RATIONALE:")
            for r in best.reasons[:5]:
                lines.append(f"  - {r}")
        if best.structure_paths:
            lines.append("STRUCTURE_EVIDENCE_PATHS:")
            for p in best.structure_paths[:4]:
                lines.append(f"  - {p}")

    calc = pack.calculation or {}
    if calc.get("answer") is not None and float(calc.get("confidence", 0.0) or 0.0) >= 0.8:
        lines.append(f"KG_SYMBOLIC_ANSWER: {calc['answer']:.12g}")
        lines.append(f"KG_SYMBOLIC_CONFIDENCE: {float(calc.get('confidence', 0.0)):.2f}")
        if calc.get("trace"):
            lines.append("KG_CALCULATION_TRACE: " + str(calc["trace"]))
        if calc.get("operands"):
            lines.append("KG_OPERAND_PROVENANCE:")
            for op in calc["operands"][:6]:
                if op.get("provenance") == "question":
                    lines.append(f"  - {op.get('role')}: {op.get('value')} @ question")
                else:
                    lines.append(
                        "  - "
                        f"{op.get('role')}: {op.get('concept')} [{op.get('period')}] = "
                        f"{op.get('value')} @ {op.get('provenance')}"
                    )

    shown_conflicts = [
        c for c in pack.conflicts
        if pack.tier == "REVIEW" or _conflict_matches_calculation(c, calc)
    ]
    if shown_conflicts:
        lines.append("KG_CONFLICTS_IN_TOPK:")
        for c in shown_conflicts[:3]:
            vals = ", ".join(f"{v['doc_id']}={v['value']}" for v in c.get("values", [])[:3])
            lines.append(f"  - {c.get('concept')} [{c.get('period')}]: {vals}")

    lines.append(pack.evidence_block)
    return "\n".join(x for x in lines if x)
