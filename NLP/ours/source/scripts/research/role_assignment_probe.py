#!/usr/bin/env python3
"""Role-assignment validation probe (closes Blind Spot #2b).

CPR's strongest component is *role* consistency, but role labels come from the
heuristic ``calculation_plan``. Reviewers ask: how accurate is that heuristic? This
probe measures it two ways, on the real selected facts per query:

  A. Plan self-accuracy (no API): when ``calculation_plan`` is confident (conf>=0.5),
     how often does its computed answer match the gold answer? (precision of the plan)
  B. Operand/operation agreement vs a strong oracle (Gemini, --oracle): does the
     heuristic select the same operand values and arithmetic operation a strong LLM
     would, given the same candidate facts?

Output per dataset: plan fire rate, plan-answer precision/recall, and (with --oracle)
operand-set F1 + operation accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match, numbers_close
from gsr_cacl.ledger.select import calculation_plan, infer_task_type, select_facts

RETR = {
    "finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl",
}
COMPUTED = {"difference", "percent_change", "ratio", "sum", "average", "factor_sum", "comparison"}

ORACLE_SYSTEM = (
    "You are a financial reasoning oracle. Given a question and a list of candidate "
    "numeric facts, identify EXACTLY which fact values are the operands and which single "
    "arithmetic operation combines them to answer the question. "
    "Reply with one JSON line: {\"op\": \"difference|percent_change|ratio|sum|average|lookup\", "
    "\"operands\": [<numbers>]}."
)


def _union_ledger(pack):
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    merged = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        merged.facts.extend(d.ledger.facts)
    return merged


def _plan_operand_values(plan):
    out = []
    for op in plan.get("operands", []) or []:
        v = op.get("value")
        if v is not None:
            out.append(float(v))
    return out


def _set_f1(pred_vals, gold_vals, tol=1e-2):
    if not pred_vals and not gold_vals:
        return 1.0
    if not pred_vals or not gold_vals:
        return 0.0
    used = [False] * len(gold_vals)
    tp = 0
    for p in pred_vals:
        for i, g in enumerate(gold_vals):
            if not used[i] and numbers_close(p, g, tol):
                used[i] = True
                tp += 1
                break
    prec = tp / len(pred_vals)
    rec = tp / len(gold_vals)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def _facts_block(facts, k=12):
    lines = []
    for f in facts[:k]:
        if f.value is None:
            continue
        per = f.period or ""
        lines.append(f"- {f.concept} [{per}] = {f.value}")
    return "\n".join(lines)


_JSON = re.compile(r"\{.*\}", re.S)


def _parse_oracle(text):
    m = _JSON.search(text or "")
    if not m:
        return None, []
    try:
        d = json.loads(m.group(0))
        op = str(d.get("op", "")).lower()
        ops = [float(x) for x in (d.get("operands") or []) if _is_num(x)]
        return op, ops
    except Exception:
        return None, []


def _is_num(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def run(ds, retr_path, sample, oracle_client=None, seed=20260628):
    import random
    recs = [json.loads(l) for l in open(retr_path) if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(recs)

    n_fire = n_total = n_plan_correct = n_computed = 0
    oracle_f1s, op_acc = [], []
    used = 0
    for rec in recs:
        if used >= sample:
            break
        query = rec.get("query") or rec.get("raw_question")
        gold = rec.get("gold")
        task = infer_task_type(query)
        if task not in COMPUTED:
            continue
        pack = build_evidence_pack(query, rec.get("retrieved", []), top_n_facts=10)
        ledger = _union_ledger(pack)
        if ledger is None:
            continue
        facts = pack.selected_facts or select_facts(query, [ledger], top_n=10)
        plan = calculation_plan(query, facts)
        n_computed += 1
        used += 1
        conf = float(plan.get("confidence", 0) or 0)
        n_total += 1
        if conf >= 0.5 and plan.get("answer") is not None:
            n_fire += 1
            if gold is not None and number_match(plan["answer"], gold):
                n_plan_correct += 1
        if oracle_client is not None:
            prompt = (f"Question: {query}\n\nCandidate facts:\n{_facts_block(facts)}\n\n"
                      "Which operands and which operation answer the question?")
            txt = oracle_client.generate(prompt, system=ORACLE_SYSTEM, temperature=0.0, max_tokens=120)
            o_op, o_vals = _parse_oracle(txt)
            h_vals = _plan_operand_values(plan)
            if o_vals:
                oracle_f1s.append(_set_f1(h_vals, o_vals))
                h_op = plan.get("task", "")
                op_acc.append(1.0 if (o_op and (o_op in h_op or h_op in o_op)) else 0.0)

    res = {
        "dataset": ds,
        "n_computed_queries": n_computed,
        "plan_fire_rate": round(n_fire / max(n_total, 1), 4),
        "plan_precision_when_fired": round(n_plan_correct / max(n_fire, 1), 4),
        "plan_answer_recall": round(n_plan_correct / max(n_total, 1), 4),
    }
    if oracle_client is not None and oracle_f1s:
        res["oracle_operand_f1"] = round(sum(oracle_f1s) / len(oracle_f1s), 4)
        res["oracle_operation_acc"] = round(sum(op_acc) / len(op_acc), 4)
        res["oracle_n"] = len(oracle_f1s)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--oracle", action="store_true", help="use Gemini oracle for operand/op agreement")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out", default="outputs/research/role_probe/report.json")
    args = ap.parse_args()

    client = None
    if args.oracle:
        from gsr_cacl.utils.gemini_client import GeminiClient
        client = GeminiClient(model=args.model)

    allres = {}
    for ds in args.datasets:
        r = run(ds, RETR[ds], args.sample, oracle_client=client)
        allres[ds] = r
        print(json.dumps(r))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(allres, indent=2))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
