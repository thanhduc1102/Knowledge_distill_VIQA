"""Concept-Period-Role (CPR) aware structure grounding.

Research contribution: the existing :func:`gsr_cacl.generation.verifier.verify` decides
"grounded" by asking *is this number present anywhere in the ledger* and "derivable" by
testing *every* value pair under a few ops. That value-only check **over-fires**: a wrong
answer often coincides with some unrelated cell (wrong concept / wrong period) or with an
arithmetic combination of two unrelated cells. The consequence is a large
``grounded_wrong`` bucket and a confidence signal that does not separate correct from
incorrect (FinQA AUROC ~= 0.50).

CPR grounding closes the loop with the *role-aware* evidence the generator already plans
(:func:`gsr_cacl.ledger.select.calculation_plan` tags operands old/new/part/total/...).
An answer is supported only when the supporting fact(s) are **concept-consistent** with the
question intent, **period-consistent** with the requested year(s), and play the **role** the
inferred arithmetic task requires. This is structure-level (not metadata) grounding, so it is
designed to transfer across datasets and to OOD financial corpora.

Pure function over a (union) FactLedger — no model, no annotation. Returns a graded,
calibrated support decision usable for selective answering / verify-then-reask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.numeric import extract_years, numbers_close
from gsr_cacl.ledger.select import (
    _period_int,
    _target_year_pair,
    _tokens,
    calculation_plan,
    infer_task_type,
    select_facts,
)
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.generation.verifier import _support_variants, extract_final_number


# Support levels, ordered. The point of CPR is to move concept/period-mismatched
# value coincidences out of "supported" and into "weak".
LEVEL_CONFIDENCE = {
    "cpr_grounded": 1.0,          # value matches a concept+period-consistent fact (copy/lookup)
    "cpr_derivable": 0.85,        # value = role-consistent task formula over consistent operands
    "scaled_grounded": 0.8,       # as cpr_grounded but via a unit/scale normalization
    "scaled_derivable": 0.7,
    "weak_value_only": 0.35,      # value/op present but concept or period MISMATCH -> not trusted
    "unsupported": 0.1,           # produced a number, nothing supports it
    "no_answer": 0.0,
}

_CONCEPT_OVERLAP_OK = 2          # content-token overlap that counts as concept-consistent
_CONCEPT_FRAC_OK = 0.5


@dataclass
class CPRResult:
    pred_value: Optional[float]
    level: str
    confidence: float
    supported: bool                       # cpr_grounded | cpr_derivable | scaled_*
    grounded: bool                        # exact/scaled concept+period-consistent copy
    derivable: bool                       # role-consistent formula
    value_only_grounded: bool             # what the legacy verifier would call grounded
    value_only_derivable: bool
    concept_consistency: float = 0.0
    period_consistency: float = 0.0
    role: str = ""
    answer_match: Optional[bool] = None
    explanation: str = ""
    reasons: list[str] = field(default_factory=list)


def _concept_consistency(fact: Fact, q_concepts: set[str], q_tokens: set[str]) -> float:
    """How well a fact's concept matches the question intent (0..1)."""
    if fact.concept_canonical and q_concepts and fact.concept_canonical in q_concepts:
        return 1.0
    ftoks = _tokens(fact.concept) | _tokens(fact.column_header)
    if fact.concept_canonical:
        ftoks |= _tokens(fact.concept_canonical)
    if not ftoks:
        return 0.0
    overlap = len(q_tokens & ftoks)
    if overlap >= _CONCEPT_OVERLAP_OK:
        return 1.0
    frac = overlap / max(len(ftoks), 1)
    return max(overlap / _CONCEPT_OVERLAP_OK, frac if frac >= _CONCEPT_FRAC_OK else 0.0)


def _period_consistency(fact: Fact, q_years: list[int], mismatch_floor: float = 0.0) -> float:
    """Whether a fact's period matches the requested year(s).

    ``mismatch_floor`` softens the penalty when the document's period structure is not
    reliably extractable (e.g. TAT-DQA multi-level headers): in that regime a "mismatch"
    is more likely a parsing failure than a genuine wrong period, so we should not zero it.
    """
    if not q_years:
        return 1.0
    p = _period_int(fact)
    if p is None:
        return max(0.5, mismatch_floor)  # unknown period: neutral
    return 1.0 if p in q_years else mismatch_floor


def _period_reliability(ledger: FactLedger) -> float:
    """Fraction of numeric facts with a parseable period — confidence in the period signal."""
    nf = ledger.numeric_facts()
    if not nf:
        return 0.0
    return sum(1 for f in nf if _period_int(f) is not None) / len(nf)


def _value_match_facts(value: float, ledger: FactLedger, rel_tol: float) -> list[Fact]:
    return [f for f in ledger.numeric_facts()
            if f.value is not None and numbers_close(value, f.value, rel_tol)]


def _consistent_pairs(value: float, facts: list[Fact], task: str,
                      q_years: list[int], q_concepts: set[str], q_tokens: set[str],
                      rel_tol: float, max_pairs: int = 600,
                      enforce_year: bool = True) -> Optional[tuple[Fact, Fact, str]]:
    """Find an operand pair that derives ``value`` AND is role-consistent for ``task``.

    For difference/percent_change the operands must be the *same concept across the two
    requested periods* (the temporal role). For ratio they must be a concept-compatible
    part/total. This is what makes the derivation auditable rather than a coincidence.
    """
    usable = [f for f in facts if f.value is not None]
    pairs = list(combinations(usable, 2))[:max_pairs]
    want_temporal = task in {"difference", "percent_change", "factor_sum"}
    want_ratio = task in {"ratio"}
    year_set = set(q_years)
    for a, b in pairs:
        # period roles
        pa, pb = _period_int(a), _period_int(b)
        same_concept = bool(
            (a.concept_canonical and a.concept_canonical == b.concept_canonical)
            or len(_tokens(a.concept) & _tokens(b.concept)) >= _CONCEPT_OVERLAP_OK
        )
        # at least one operand must be concept-consistent with the question
        ca = max(_concept_consistency(a, q_concepts, q_tokens), _concept_consistency(b, q_concepts, q_tokens))
        if ca < _CONCEPT_FRAC_OK:
            continue
        if want_temporal:
            if not same_concept:
                continue
            if enforce_year and year_set and not ({pa, pb} & year_set):
                continue
            if numbers_close(value, a.value - b.value, rel_tol) or numbers_close(value, b.value - a.value, rel_tol):
                return (a, b, "temporal_difference")
            if b.value != 0 and numbers_close(value, (a.value - b.value) / abs(b.value) * 100.0, rel_tol):
                return (a, b, "temporal_percent_change")
            if b.value != 0 and numbers_close(value, (a.value - b.value) / abs(b.value), rel_tol):
                return (a, b, "temporal_percent_change_frac")
        elif want_ratio:
            if same_concept:
                continue  # ratio operands should be different concepts (part vs total)
            if b.value != 0 and numbers_close(value, a.value / b.value, rel_tol):
                return (a, b, "part_over_total")
            if a.value != 0 and numbers_close(value, b.value / a.value, rel_tol):
                return (b, a, "part_over_total")
        else:
            # sum/average/other: operands should be concept-consistent siblings
            if numbers_close(value, a.value + b.value, rel_tol):
                return (a, b, "sum")
    return None


def verify_cpr(prediction: str, ledger: FactLedger, query: str = "", gold=None,
               rel_tol: float = 1e-2, selected_facts: Optional[list[Fact]] = None) -> CPRResult:
    """Concept-period-role aware verification of a generated answer."""
    value = extract_final_number(prediction)
    if value is None:
        return CPRResult(pred_value=None, level="no_answer", confidence=0.0,
                         supported=False, grounded=False, derivable=False,
                         value_only_grounded=False, value_only_derivable=False,
                         explanation="no parseable answer")

    q_tokens = _tokens(query)
    q_concepts = concepts_in_text(query)
    pair = _target_year_pair(query)
    q_years = list(pair) if pair else sorted(set(extract_years(query)))
    task = infer_task_type(query)
    # Gate the period signal by how reliably this document's periods parse. On tables with
    # noisy/multi-level headers (e.g. TAT-DQA) a period "mismatch" is usually a parse error,
    # so we neither hard-penalize nor hard-filter on year there.
    period_rel = _period_reliability(ledger)
    pfloor = 0.0 if period_rel >= 0.4 else 0.5
    enforce_year = period_rel >= 0.4

    answer_match = None
    if gold is not None:
        from gsr_cacl.ledger.numeric import number_match
        answer_match = bool(number_match(value, gold, rel_tol=rel_tol))

    reasons: list[str] = []
    best_concept = 0.0
    best_period = 0.0
    role = ""

    # ---- value-only baseline (what the legacy verifier sees) -----------------
    vo_grounded = False
    vo_derivable = False
    vo_match_count = 0
    for v, _ in _support_variants(value, query):
        m = _value_match_facts(v, ledger, rel_tol)
        if m:
            vo_grounded = True
            vo_match_count = max(vo_match_count, len(m))
            break
    if not vo_grounded:
        vals = [f.value for f in ledger.numeric_facts() if f.value is not None]
        for v, _ in _support_variants(value, query):
            for a, b in list(combinations(vals, 2))[:600]:
                if (numbers_close(v, a - b, rel_tol) or numbers_close(v, b - a, rel_tol)
                        or numbers_close(v, a + b, rel_tol)
                        or (b and numbers_close(v, a / b, rel_tol))
                        or (a and numbers_close(v, b / a, rel_tol))):
                    vo_derivable = True
                    break
            if vo_derivable:
                break

    # ---- continuous CPR scores ------------------------------------------------
    # grounded_score: value matches a fact, weighted by concept x period consistency
    # and dampened by ambiguity (a value matching many cells is less discriminative).
    grounded_score = 0.0
    grounded_scaled = False
    for v, vlabel in _support_variants(value, query):
        matches = _value_match_facts(v, ledger, rel_tol)
        if not matches:
            continue
        ambiguity = 1.0 / (len(matches) ** 0.5)
        for f in matches:
            cc = _concept_consistency(f, q_concepts, q_tokens)
            pc = _period_consistency(f, q_years, pfloor)
            s = cc * pc * ambiguity * (1.0 if vlabel == "identity" else 0.8)
            if s > grounded_score:
                grounded_score = s
                best_concept, best_period = cc, pc
                grounded_scaled = vlabel != "identity"
                role = "answer_fact"
                reasons = [f"value->concept-consistent fact '{f.concept}'"
                           f"{(' [' + str(f.period) + ']') if f.period else ''} "
                           f"(cc={cc:.2f}, pc={pc:.2f}, amb={len(matches)})"]
        break  # identity variant first; only fall through if no match

    # derivable_score: answer = role-consistent task formula over consistent operands
    derivable_score = 0.0
    facts = selected_facts or select_facts(query, [ledger], top_n=8)
    plan = calculation_plan(query, facts)
    plan_ans = plan.get("answer")
    plan_conf = float(plan.get("confidence", 0.0) or 0.0)
    if plan_ans is not None and plan_conf >= 0.5:
        for v, vlabel in _support_variants(value, query):
            if numbers_close(v, plan_ans, rel_tol):
                derivable_score = plan_conf * (1.0 if vlabel == "identity" else 0.8)
                if derivable_score > grounded_score:
                    role = "+".join(str(op.get("role")) for op in (plan.get("operands") or [])[:3])
                    reasons = [f"role-consistent plan ({plan.get('task')}, conf={plan_conf:.2f}): "
                               + str(plan.get("trace", ""))[:120]]
                break
    if derivable_score == 0.0:
        cand = _consistent_pairs(value, ledger.numeric_facts(), task, q_years,
                                 q_concepts, q_tokens, rel_tol, enforce_year=enforce_year)
        if cand is not None:
            a, b, kind = cand
            ccp = max(_concept_consistency(a, q_concepts, q_tokens),
                      _concept_consistency(b, q_concepts, q_tokens))
            pcp = max(_period_consistency(a, q_years, pfloor), _period_consistency(b, q_years, pfloor))
            derivable_score = 0.75 * ccp * pcp
            if derivable_score > grounded_score and not role:
                role = kind
                reasons = [f"role-consistent {kind}: {a.concept}={a.value}, {b.concept}={b.value}"]

    # value-only floor: a number at least present in the ledger beats a pure hallucination,
    # but a value matching many cells is weak evidence.
    value_only_floor = 0.0
    if vo_grounded:
        value_only_floor = 0.25 / (max(vo_match_count, 1) ** 0.5)
    elif vo_derivable:
        value_only_floor = 0.18
    raw_floor = 0.05  # produced a parseable number

    confidence = max(grounded_score, derivable_score, value_only_floor, raw_floor)

    # interpretable level (for diagnostics) derived from the dominant component
    if grounded_score >= 0.5 and grounded_score >= derivable_score:
        level = "scaled_grounded" if grounded_scaled else "cpr_grounded"
    elif derivable_score >= 0.5:
        level = "cpr_derivable"
    elif (vo_grounded or vo_derivable):
        level = "weak_value_only"
    elif value is not None:
        level = "unsupported"
    else:
        level = "no_answer"

    supported = confidence >= 0.5
    grounded = level in ("cpr_grounded", "scaled_grounded")
    derivable = level == "cpr_derivable"

    parts = [f"level={level}", f"conf={confidence:.2f}", f"cc={best_concept:.2f}", f"pc={best_period:.2f}"]
    if role:
        parts.append(f"role={role}")
    if gold is not None:
        parts.append(f"gold_match={answer_match}")

    return CPRResult(
        pred_value=value,
        level=level,
        confidence=confidence,
        supported=supported,
        grounded=grounded,
        derivable=derivable,
        value_only_grounded=vo_grounded,
        value_only_derivable=vo_derivable,
        concept_consistency=best_concept,
        period_consistency=best_period,
        role=role,
        answer_match=answer_match,
        explanation="; ".join(parts),
        reasons=reasons,
    )
