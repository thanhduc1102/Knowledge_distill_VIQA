"""Multi-path symbolic answering with deterministic agreement voting.

Validated mechanism (scripts/research/coordinate_eval.py): when two INDEPENDENT grounding
paths land the same value, Number-Match jumps far above either path alone (FinQA 0.31→0.46,
ConvFinQA 0.43→0.60, TAT 0.09→0.27). Agreement is therefore a strong, training-free, auditable
confidence signal — the finance-critical "knows when it is sure" property.

We run several independent deterministic answerers, cluster their answers by numeric
closeness, and return the majority value with confidence = (votes / n_paths). The KG emits a
symbolic answer only when ``votes >= min_votes``; otherwise it abstains and defers to the LLM
(handing over the clean evidence + the disagreement as an explicit uncertainty flag).

Independent paths (intentionally different signals):
  A. heuristic    — token-relevance fact selection + task arithmetic (ledger.select).
  B. coordinate   — 2-D row(concept)×column(period) cell addressing (ledger.coordinate).
  C. ontology     — canonical-concept + period direct lookup (ontology-driven).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.numeric import extract_years, numbers_close
from gsr_cacl.ledger.coordinate import coordinate_answer, CoordinateGrounder
from gsr_cacl.ledger.select import calculation_plan, select_facts, infer_task_type
from gsr_cacl.ontology.concepts import concepts_in_text


# --- path C: ontology canonical-concept lookup -------------------------------
def _ontology_answer(question: str, ledger: FactLedger, task: str) -> Optional[dict]:
    q_concepts = concepts_in_text(question)
    if not q_concepts:
        return None
    q_years = sorted(set(extract_years(question)))
    cands = [f for f in ledger.numeric_facts() if f.concept_canonical in q_concepts]
    if not cands:
        return None

    def at_year(y):
        for f in cands:
            try:
                if f.period and int(float(str(f.period))) == y:
                    return f
            except (TypeError, ValueError):
                pass
        return None

    if task in ("difference", "percent_change") and len(q_years) >= 2:
        fo, fn = at_year(q_years[0]), at_year(q_years[-1])
        if fo and fn and fo.value is not None and fn.value is not None and fo is not fn:
            if task == "percent_change":
                if not fo.value:
                    return None
                ans = (fn.value - fo.value) / abs(fo.value)
            else:
                ans = fn.value - fo.value
            return {"answer": ans, "operands": [fo, fn], "method": "ontology"}
        return None
    # lookup: pick the requested-year fact, else the first canonical fact
    f = None
    if q_years:
        f = at_year(q_years[0])
    f = f or cands[0]
    if f.value is None:
        return None
    return {"answer": f.value, "operands": [f], "method": "ontology"}


@dataclass
class MultiPathResult:
    answer: Optional[float]
    confidence: float            # votes / n_paths_that_produced_an_answer
    votes: int
    n_paths: int
    methods: list[str] = field(default_factory=list)   # methods that agreed on the winner
    operands: list[Fact] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)  # all path answers (for audit)


def multipath_answer(question: str, ledger: FactLedger, task: Optional[str] = None) -> MultiPathResult:
    task = task or infer_task_type(question)

    results: list[dict] = []
    # A. heuristic
    cur = calculation_plan(question, select_facts(question, [ledger], top_n=6))
    if cur.get("answer") is not None:
        results.append({"answer": float(cur["answer"]), "method": "heuristic",
                        "operands": cur.get("operand_facts", [])})
    # B. coordinate
    co = coordinate_answer(question, ledger, task)
    if co is not None:
        results.append({"answer": float(co["answer"]), "method": "coordinate",
                        "operands": co.get("operands", [])})
    # C. ontology
    on = _ontology_answer(question, ledger, task)
    if on is not None:
        results.append({"answer": float(on["answer"]), "method": "ontology",
                        "operands": on.get("operands", [])})

    n_paths = len(results)
    if n_paths == 0:
        return MultiPathResult(None, 0.0, 0, 0, [], [], [])

    # cluster by numeric closeness; majority value wins
    clusters: list[dict] = []
    for r in results:
        placed = False
        for c in clusters:
            if numbers_close(c["answer"], r["answer"]):
                c["methods"].append(r["method"])
                c["votes"] += 1
                if not c["operands"]:
                    c["operands"] = r["operands"]
                placed = True
                break
        if not placed:
            clusters.append({"answer": r["answer"], "votes": 1,
                             "methods": [r["method"]], "operands": r["operands"]})
    win = max(clusters, key=lambda c: c["votes"])
    return MultiPathResult(
        answer=win["answer"],
        confidence=round(win["votes"] / n_paths, 3),
        votes=win["votes"],
        n_paths=n_paths,
        methods=win["methods"],
        operands=win["operands"],
        candidates=[{"answer": r["answer"], "method": r["method"]} for r in results],
    )
