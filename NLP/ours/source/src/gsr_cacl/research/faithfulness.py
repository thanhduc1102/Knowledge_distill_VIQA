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


def auc_risk_coverage(records: Iterable[FaithfulnessRecord]) -> dict:
    """Area under the selective risk-coverage curve.

    Lower AURC is better.  We also report the full-coverage accuracy so the value is
    interpretable next to the base model's error rate.
    """
    recs = sorted(list(records), key=lambda r: r.confidence, reverse=True)
    if not recs:
        return {"aurc": 0.0, "base_accuracy": 0.0, "base_risk": 0.0}
    risks, covs = [], []
    for k in range(1, len(recs) + 1):
        acc = float(np.mean([r.correct for r in recs[:k]]))
        covs.append(k / len(recs))
        risks.append(1.0 - acc)
    area = 0.0
    for i in range(1, len(covs)):
        area += 0.5 * (risks[i - 1] + risks[i]) * (covs[i] - covs[i - 1])
    return {
        "aurc": round(float(area), 4),
        "base_accuracy": round(float(np.mean([r.correct for r in recs])), 4),
        "base_risk": round(1.0 - float(np.mean([r.correct for r in recs])), 4),
    }


def binary_auc(records: Iterable[FaithfulnessRecord]) -> float:
    """AUROC of verifier confidence as a correctness classifier."""
    recs = list(records)
    pos = [r.confidence for r in recs if r.correct]
    neg = [r.confidence for r in recs if not r.correct]
    if not pos or not neg:
        return 0.0
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return round(wins / (len(pos) * len(neg)), 4)


def bootstrap_group_gap(records: Iterable[FaithfulnessRecord], n_boot: int = 2000, seed: int = 0) -> dict:
    recs = list(records)
    rng = np.random.default_rng(seed)
    if not recs:
        return {"gap": 0.0, "ci95": [0.0, 0.0], "p_gap_le_0": 1.0}

    def gap(sample):
        g = [r.correct for r in sample if r.grounded]
        u = [r.correct for r in sample if not r.grounded]
        return (float(np.mean(g)) if g else 0.0) - (float(np.mean(u)) if u else 0.0)

    base = gap(recs)
    boots = []
    n = len(recs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(gap([recs[i] for i in idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "gap": round(base, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_gap_le_0": round(float(np.mean(np.asarray(boots) <= 0.0)), 4),
    }


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
