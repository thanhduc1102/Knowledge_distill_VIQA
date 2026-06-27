"""Weakly learned coordinate grounding.

The original coordinate grounder uses fixed row/column heuristics.  This module keeps the
same interpretable 2D decomposition but learns a tiny linear scorer from answer supervision:
candidate cells are ranked by `w · phi(question, cell)`, where features are row lexical
match, row containment, ontology match, period match, total hints, and value plausibility.

No cell labels are required.  During training, any candidate whose value matches the gold
answer is treated as positive; all other cells in the gold document are negatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gsr_cacl.ledger.coordinate import _toks
from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.numeric import extract_numbers, extract_years, number_match
from gsr_cacl.ledger.select import infer_task_type
from gsr_cacl.ontology.concepts import canonical_concept, concepts_in_text


def _safe_year(period: str | None, header: str = "") -> Optional[int]:
    if period:
        try:
            return int(float(str(period)))
        except (TypeError, ValueError):
            pass
    yrs = extract_years(header)
    return yrs[-1] if yrs else None


def cell_features(question: str, fact: Fact) -> np.ndarray:
    qt = _toks(question)
    rt = _toks(fact.concept or "")
    ht = _toks(fact.column_header or "")
    q_concepts = concepts_in_text(question)
    row_canon = canonical_concept(fact.concept or "")
    row_inter = len(qt & rt)
    row_union = len(qt | rt) or 1
    row_jaccard = row_inter / row_union
    row_contain = row_inter / max(len(rt), 1)
    header_hit = len(qt & ht) / max(len(ht), 1)
    onto = 1.0 if row_canon and row_canon in q_concepts else 0.0
    q_years = set(extract_years(question))
    fy = _safe_year(fact.period, fact.column_header)
    if q_years:
        period_exact = 1.0 if fy in q_years else 0.0
        period_near = math.exp(-min(abs(fy - y) for y in q_years) / 2.0) if fy else 0.4
    else:
        period_exact = 0.0
        period_near = 1.0
    total_hint = 1.0 if "total" in rt or "total" in ht else 0.0
    q_nums = [n for n in extract_numbers(question) if not (n == int(n) and 1900 <= int(n) <= 2049)]
    value_anchor = 0.0
    if q_nums and fact.value is not None:
        value_anchor = max(math.exp(-abs(math.log1p(abs(fact.value)) - math.log1p(abs(n)))) for n in q_nums)
    return np.asarray([
        1.0,
        row_jaccard,
        row_contain,
        header_hit,
        onto,
        period_exact,
        period_near,
        total_hint,
        value_anchor,
    ], dtype=np.float64)


@dataclass
class LearnedCoordinateModel:
    weights: np.ndarray

    @classmethod
    def default(cls) -> "LearnedCoordinateModel":
        # Conservative hand prior; train_weak can replace it.
        return cls(np.asarray([0.0, 0.6, 0.9, 0.25, 1.2, 1.0, 0.4, -0.15, 0.1]))

    def score_fact(self, question: str, fact: Fact) -> float:
        return float(cell_features(question, fact) @ self.weights)

    def answer_lookup(self, question: str, ledger: FactLedger) -> Optional[dict]:
        cands = [f for f in ledger.numeric_facts() if f.value is not None]
        if not cands:
            return None
        scored = [(self.score_fact(question, f), f) for f in cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        s, f = scored[0]
        return {"answer": f.value, "operands": [f], "confidence": float(s), "method": "learned_coord"}

    def _best_row(self, question: str, ledger: FactLedger) -> Optional[int]:
        by_row: dict[int, list[Fact]] = {}
        for f in ledger.numeric_facts():
            if f.row_idx is not None and f.row_idx >= 0:
                by_row.setdefault(f.row_idx, []).append(f)
        if not by_row:
            return None
        # Row score is the best learned cell score on that row.  This lets the learned
        # row matcher feed multi-period arithmetic instead of acting as a lookup-only
        # answerer.
        row_scores = []
        for ri, facts in by_row.items():
            row_scores.append((max(self.score_fact(question, f) for f in facts), ri))
        return max(row_scores)[1]

    @staticmethod
    def _cell_at_year(ledger: FactLedger, row_idx: int, year: int) -> Optional[Fact]:
        for f in ledger.numeric_facts():
            if f.row_idx != row_idx:
                continue
            if _safe_year(f.period, f.column_header) == year:
                return f
        return None

    def answer(self, question: str, ledger: FactLedger, task: Optional[str] = None) -> Optional[dict]:
        task = task or infer_task_type(question)
        if task in {"difference", "percent_change"}:
            years = sorted(set(extract_years(question)))
            row_idx = self._best_row(question, ledger)
            if row_idx is None:
                return None
            if len(years) >= 2:
                old = self._cell_at_year(ledger, row_idx, years[0])
                new = self._cell_at_year(ledger, row_idx, years[-1])
            else:
                row_cells = [f for f in ledger.numeric_facts() if f.row_idx == row_idx and f.value is not None]
                row_cells.sort(key=lambda f: _safe_year(f.period, f.column_header) or f.col_idx or 0)
                old, new = (row_cells[0], row_cells[-1]) if len(row_cells) >= 2 else (None, None)
            if old is None or new is None or old.value is None or new.value is None or old is new:
                return None
            if task == "percent_change":
                if old.value == 0:
                    return None
                ans = (new.value - old.value) / abs(old.value)
            else:
                ans = new.value - old.value
            conf = max(self.score_fact(question, old), self.score_fact(question, new))
            return {"answer": ans, "operands": [old, new], "confidence": float(conf),
                    "method": "learned_coord_2period"}
        return self.answer_lookup(question, ledger)


def train_weak(examples: list[tuple[str, FactLedger, object]], epochs: int = 120, lr: float = 0.1,
               l2: float = 1e-3, seed: int = 0) -> LearnedCoordinateModel:
    """Train a tiny logistic ranker from (question, gold-ledger, gold-answer)."""
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for q, ledger, gold in examples:
        for f in ledger.numeric_facts():
            if f.value is None:
                continue
            feats.append(cell_features(q, f))
            labels.append(1.0 if number_match(f.value, gold) else 0.0)
    if not feats or sum(labels) == 0:
        return LearnedCoordinateModel.default()
    X = np.vstack(feats)
    y = np.asarray(labels, dtype=np.float64)
    # Class balance: positives are sparse.
    pos_w = max(1.0, (len(y) - y.sum()) / max(y.sum(), 1.0))
    w = LearnedCoordinateModel.default().weights.copy()
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for i in order:
            z = float(X[i] @ w)
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            weight = pos_w if y[i] > 0.5 else 1.0
            grad = weight * (p - y[i]) * X[i] + l2 * w
            w -= lr * grad
    return LearnedCoordinateModel(w)
