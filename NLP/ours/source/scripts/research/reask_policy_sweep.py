#!/usr/bin/env python3
"""Sweep verify/reask policies using saved raw and re-ask predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.ledger.numeric import number_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw-predictions", required=True)
    ap.add_argument("--reask-predictions", required=True)
    ap.add_argument("--out", default="outputs/research/reask_policy")
    args = ap.parse_args()

    raw = [json.loads(l) for l in open(args.raw_predictions) if l.strip()]
    reask = [json.loads(l) for l in open(args.reask_predictions) if l.strip()]
    n = min(len(raw), len(reask))

    def raw_ok(i):
        gold = raw[i].get("gold")
        pred = raw[i].get("pred_value")
        return bool(number_match(pred, gold)) if gold is not None and pred is not None else False

    def final_ok(i):
        gold = raw[i].get("gold")
        pred = reask[i].get("final_answer")
        return bool(number_match(pred, gold)) if gold is not None and pred is not None else False

    policies: dict[str, list[bool]] = {}
    policies["keep_raw"] = [False] * n
    policies["reask_if_ungrounded"] = [not bool(raw[i].get("grounded", False)) for i in range(n)]
    # These are explicitly oracle/offline because saved reward includes gold-match when
    # gold labels are available.  They estimate headroom for a better annotation-free
    # verifier; they are NOT deployable policies.
    for t in (0.2, 0.5, 0.6, 0.8):
        policies[f"oracle_reward_lt_{t}"] = [float(raw[i].get("reward", 0.0) or 0.0) < t for i in range(n)]
    # Oracle upper bound: how much room exists if the verifier perfectly knew when re-ask helps.
    policies["oracle_reask_if_improves"] = [final_ok(i) and not raw_ok(i) for i in range(n)]

    out = {"dataset": args.dataset, "n": n, "policies": {}, "transitions": {}}
    raw_correct = [raw_ok(i) for i in range(n)]
    final_correct = [final_ok(i) for i in range(n)]
    for name, flags in policies.items():
        corr = [final_correct[i] if flags[i] else raw_correct[i] for i in range(n)]
        out["policies"][name] = {
            "nm": round(sum(corr) / max(n, 1), 4),
            "reask_frac": round(sum(flags) / max(n, 1), 4),
            "reask_n": int(sum(flags)),
        }
    out["transitions"] = {
        "raw_wrong_to_reask_correct": int(sum((not raw_correct[i]) and final_correct[i] for i in range(n))),
        "raw_correct_to_reask_wrong": int(sum(raw_correct[i] and not final_correct[i] and policies["reask_if_ungrounded"][i] for i in range(n))),
        "raw_correct_ungrounded": int(sum(raw_correct[i] and policies["reask_if_ungrounded"][i] for i in range(n))),
        "raw_wrong_grounded": int(sum((not raw_correct[i]) and bool(raw[i].get("grounded", False)) for i in range(n))),
    }

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / f"{args.dataset}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n=== Reask policy sweep — {args.dataset} ===")
    for k, v in out["policies"].items():
        print(f"{k:<28} NM={v['nm']:.4f} reask={v['reask_frac']:.2%}")
    print(f"transitions={out['transitions']}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
