#!/usr/bin/env python3
"""Unit tests for the MMER expert framework (training-free experts + fusion shapes).

Run: PYTHONPATH=src python tests/test_experts.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.experts.base import minmax
from gsr_cacl.experts.lexical import LexicalExpert
from gsr_cacl.experts.concept import ConceptExpert
from gsr_cacl.experts.cell import CellExpert
from gsr_cacl.experts.graph import GraphExpert, _intent
from gsr_cacl.experts.fusion import FusionData, train_fusion, rank_scores

_TBL_A = ("Apple Inc. income statement.\n\n"
          "| item | 2018 | 2019 |\n|---|---|---|\n"
          "| Revenue | 200 | 260 |\n| Cost of revenue | 120 | 160 |\n| Gross profit | 80 | 100 |\n")
_TBL_B = ("Microsoft balance sheet.\n\n"
          "| item | 2018 | 2019 |\n|---|---|---|\n"
          "| Total assets | 500 | 600 |\n| Total liabilities | 300 | 350 |\n")
CORPUS = [Document(page_content=_TBL_A, meta_data={"company_name": "Apple Inc.", "report_year": "2019"}, id="A"),
          Document(page_content=_TBL_B, meta_data={"company_name": "Microsoft", "report_year": "2019"}, id="B")]
METAS = [dict(d.meta_data) for d in CORPUS]
QUERIES = ["What was the gross profit in 2019?",            # → doc A
           "What were total assets in 2019?"]               # → doc B
GOLD = [0, 1]

PASS, FAIL = [], []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def test_minmax():
    out = minmax(np.array([1.0, 3.0, 2.0]))
    check("minmax_range", abs(out.min()) < 1e-9 and abs(out.max() - 1.0) < 1e-9)
    check("minmax_flat_neutral", np.allclose(minmax(np.array([2.0, 2.0])), 0.5))


def test_lexical_retrieves():
    ex = LexicalExpert(); ex.prepare(CORPUS, METAS); ex.set_queries(QUERIES, METAS)
    s0 = ex.score_pool(0, [0, 1])
    check("lexical_gross_profit_picks_A", s0[0] > s0[1])
    check("lexical_is_retriever_full_scores", ex.full_scores(0).shape == (2,))


def test_concept_cell():
    for E in (ConceptExpert, CellExpert):
        ex = E(); ex.prepare(CORPUS, METAS); ex.set_queries(QUERIES, METAS)
        s0, s1 = ex.score_pool(0, [0, 1]), ex.score_pool(1, [0, 1])
        check(f"{ex.name}_gross_profit_favors_A", s0[0] >= s0[1])
        check(f"{ex.name}_total_assets_favors_B", s1[1] >= s1[0])

def test_concept_coverage():
    """Regression guard: concept expert must find concepts via text scan, not ledger-only."""
    from gsr_cacl.experts.concept import _doc_concepts
    # Standard financial tables should yield ≥1 canonical concept
    tbl_income = ("| item | 2019 |\n|---|---|\n| Total revenues | 12345 |\n"
                  "| Net income | 1234 |\n| Operating income | 2345 |\n")
    tbl_balance = ("| item | 2019 |\n|---|---|\n| Total assets | 45678 |\n"
                   "| Total equity | 15000 |\n| Long-term debt | 5678 |\n")
    c1, p1 = _doc_concepts(tbl_income)
    c2, p2 = _doc_concepts(tbl_balance)
    check("concept_income_stmt_nonzero", len(c1) >= 2)
    check("concept_balance_sheet_nonzero", len(c2) >= 2)
    check("concept_revenue_detected", "Revenue" in c1 or "NetIncome" in c1)
    check("concept_assets_detected", "TotalAssets" in c2)


def test_graph_intent_and_score():
    check("intent_temporal", _intent("how did revenue change from 2018 to 2019") == "temporal")
    check("intent_ratio", _intent("what was the gross margin ratio") == "ratio")
    check("intent_lookup", _intent("what was revenue in 2019") == "lookup")
    ex = GraphExpert(); ex.prepare(CORPUS, METAS); ex.set_queries(QUERIES, METAS)
    s0 = ex.score_pool(0, [0, 1])
    check("graph_gross_profit_favors_A", s0[0] >= s0[1])


def test_fusion_learns():
    # Two experts: col0 is informative (high at gold), col1 is noise.
    feats, gold = [], []
    rng = np.random.default_rng(0)
    for _ in range(120):
        pool = 5
        g = int(rng.integers(pool))
        col0 = rng.random(pool) * 0.3; col0[g] += 0.7          # signal
        col1 = rng.random(pool)                                  # noise
        feats.append(np.stack([col0, col1], axis=1)); gold.append(g)
    data = FusionData(feats=feats, gold_pos=gold, qfeats=None, expert_names=["sig", "noise"])
    model = train_fusion("linear", data, list(range(120)), epochs=150)
    w = model.weights()
    check("fusion_learns_signal>noise", w[0] > w[1])
    order = np.argsort(-rank_scores(model, data, 0))
    check("fusion_ranks_gold_top", int(order[0]) == gold[0])


if __name__ == "__main__":
    print("=== MMER expert tests ===")
    for t in (test_minmax, test_lexical_retrieves, test_concept_cell,
              test_concept_coverage, test_graph_intent_and_score, test_fusion_learns):
        t()
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} tests passed")
    sys.exit(1 if FAIL else 0)
