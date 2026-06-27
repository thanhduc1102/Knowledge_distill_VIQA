"""Faithfulness and selective-risk diagnostics for financial RAG outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class FaithfulnessRecord:
    correct: bool
    grounded: bool
    reward: float = 0.0
    grounding_fraction: float = 0.0
    arithmetic_fraction: float = 0.0
    has_provenance: bool = False
    provenance_correct: Optional[bool] = None

    @property
    def confidence(self) -> float:
        base = 0.55 * float(self.grounded) + 0.25 * self.grounding_fraction
        base += 0.15 * self.arithmetic_fraction + 0.05 * float(self.has_provenance)
        if self.reward:
            base = max(base, min(float(self.reward), 1.0))
        return float(max(0.0, min(1.0, base)))


def grouped_correctness(records: Iterable[FaithfulnessRecord]) -> dict:
    recs = list(records)
    out = {"n": len(recs)}
    for name, subset in (
        ("grounded", [r for r in recs if r.grounded]),
        ("ungrounded", [r for r in recs if not r.grounded]),
    ):
        out[f"{name}_n"] = len(subset)
        out[f"{name}_accuracy"] = round(
            float(np.mean([r.correct for r in subset])) if subset else 0.0, 4
        )
    out["separation"] = round(out["grounded_accuracy"] - out["ungrounded_accuracy"], 4)
    return out


def risk_coverage_curve(records: Iterable[FaithfulnessRecord], steps: int = 20) -> list[dict]:
    recs = sorted(list(records), key=lambda r: r.confidence, reverse=True)
    if not recs:
        return []
    out = []
    n = len(recs)
    for k in range(1, steps + 1):
        cut = max(1, int(round(n * k / steps)))
        keep = recs[:cut]
        acc = float(np.mean([r.correct for r in keep]))
        out.append({
            "coverage": round(cut / n, 4),
            "accuracy": round(acc, 4),
            "risk": round(1.0 - acc, 4),
            "threshold": round(keep[-1].confidence, 4),
            "n": cut,
        })
    return out


def hallucination_proxy(records: Iterable[FaithfulnessRecord]) -> dict:
    """Ungrounded wrong answers are the conservative hallucination bucket."""
    recs = list(records)
    caught = [r for r in recs if (not r.grounded and not r.correct)]
    wrong = [r for r in recs if not r.correct]
    return {
        "ungrounded_wrong_n": len(caught),
        "wrong_n": len(wrong),
        "hallucination_catch_rate": round(len(caught) / max(len(wrong), 1), 4),
        "ungrounded_wrong_frac": round(len(caught) / max(len(recs), 1), 4),
    }


def provenance_summary(records: Iterable[FaithfulnessRecord]) -> dict:
    recs = list(records)
    with_prov = [r for r in recs if r.has_provenance]
    judged = [r for r in recs if r.provenance_correct is not None]
    return {
        "provenance_coverage": round(len(with_prov) / max(len(recs), 1), 4),
        "provenance_judged_n": len(judged),
        "provenance_precision_proxy": round(
            float(np.mean([r.provenance_correct for r in judged])) if judged else 0.0, 4
        ),
    }
