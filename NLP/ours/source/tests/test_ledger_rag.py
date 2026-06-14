"""Fast, dependency-light smoke tests for the LEDGER-RAG upgrade.

Run:  python -m pytest ours/source/tests/test_ledger_rag.py -q
  or: python ours/source/tests/test_ledger_rag.py
No GPU / no network required (uses synthetic tables).
"""

from __future__ import annotations

SYNTH_TABLE = """| item | 2019 | 2018 |
| --- | --- | --- |
| revenue | $ 1000 | $ 900 |
| cost of goods sold | 600 | 550 |
| gross profit | 400 | 350 |
| net income | 250 | 210 |
"""


def test_numeric_parsing():
    from gsr_cacl.ledger.numeric import parse_financial_number, number_match
    assert parse_financial_number("$ 5,735") == 5735.0
    assert parse_financial_number("-32 ( 32 )") == -32.0
    assert parse_financial_number("(1,234)") == -1234.0
    assert parse_financial_number("['2019']") == 2019.0
    assert number_match("the answer is $206,588 thousand", ["206588.0"])
    assert number_match("94", ["94.0"])
    assert not number_match("100", ["200"])


def test_ledger_extraction():
    from gsr_cacl.ledger import extract_ledger, select_facts, build_evidence_block
    led = extract_ledger(table_md=SYNTH_TABLE, doc_id="d1",
                         meta={"company_name": "Acme", "report_year": "2019"})
    concepts = {f.concept.lower() for f in led.table_facts()}
    assert any("revenue" in c for c in concepts)
    assert any("gross profit" in c for c in concepts)
    facts = select_facts("What was the gross profit in 2019?", [led], top_n=5)
    assert facts and "gross profit" in facts[0].concept.lower()
    assert facts[0].value == 400.0
    block = build_evidence_block("gross profit 2019", [led], top_n=5)
    assert "gross profit" in block.lower()


def test_verifier():
    from gsr_cacl.ledger import extract_ledger
    from gsr_cacl.generation import verify
    led = extract_ledger(table_md=SYNTH_TABLE, doc_id="d1", meta={"company_name": "Acme"})
    vr = verify("Answer: 400", led, "gross profit 2019", gold=["400"])
    assert vr.answer_match and vr.grounded
    # derivable: revenue - cogs = 400
    vr2 = verify("Answer: 400", led, "", gold=None)
    assert vr2.grounded or vr2.derivable


def test_equation_constraint_score():
    from gsr_cacl.kg.builder import build_kg_from_markdown
    from gsr_cacl.scoring.constraint_score import compute_equation_constraint_score
    kg = build_kg_from_markdown(SYNTH_TABLE)
    res = compute_equation_constraint_score(kg)
    assert 0.0 <= res.constraint_score <= 1.0


def test_channel_aligned_negatives():
    from gsr_cacl.negative_sampler.channel_aligned import ChannelAlignedSampler, CHANNELS
    samp = ChannelAlignedSampler(seed=0)
    negs = samp.sample(SYNTH_TABLE, n_negatives=5)
    assert len(negs) >= 4
    channels = {n.channel for n in negs}
    assert channels.issubset(set(CHANNELS))
    # entity-swap keeps values, flags metadata swap
    ent = [n for n in negs if n.channel == "entity-swap"]
    assert ent and ent[0].swap_metadata
    # other negatives are real table mutations (differ from original)
    mutated = [n for n in negs if n.channel != "entity-swap"]
    assert all(n.table_md != SYNTH_TABLE for n in mutated)


def test_entity_embedder_separation():
    import torch
    from gsr_cacl.entity import HashMetadataEmbedder, entity_cosine
    from gsr_cacl.entity.supcon import SupConLoss, make_entity_labels
    metas = [{"company_name": c, "report_year": y, "company_sector": "Tech",
              "company_industry": "SW", "company_symbol": c[:3]}
             for c in ["Apple", "Microsoft", "Google"] for y in ["2018", "2019"]]
    labels = make_entity_labels(metas)
    model = HashMetadataEmbedder(embed_dim=32)
    loss_fn = SupConLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(80):
        emb = model(metas)
        loss = loss_fn(emb, labels)
        opt.zero_grad(); loss.backward(); opt.step()
    emb = model.encode(metas)
    same = float(entity_cosine(emb[0:1], emb[1:2]))      # Apple 2018 vs Apple 2019
    diff = float(entity_cosine(emb[0:1], emb[2:3]))      # Apple 2018 vs Microsoft 2018
    assert same > diff


def test_gics_canonicalization():
    from gsr_cacl.ontology import canonical_sector, sector_id, N_SECTORS
    assert canonical_sector("Financials") == "Financials"
    assert canonical_sector("Utilities") == "Utilities"
    # industry tokens resolve to their parent GICS sector
    assert canonical_sector("", "Semiconductors") == "Information Technology"
    assert canonical_sector("", "Software") == "Information Technology"
    assert canonical_sector("Telecommunications") == "Communication Services"
    assert canonical_sector("", "Oil & Gas Drilling") == "Energy"
    assert canonical_sector("garbage xyz") == "Unknown"
    assert 0 <= sector_id("Financials") < N_SECTORS


def test_company_alias_matching():
    from gsr_cacl.ontology import normalize_company, company_match, company_acronym
    assert normalize_company("American Water Works Company, Inc.") == \
           normalize_company("American Water Works")
    assert company_match("American Water Works Company, Inc.", "American Water Works")
    assert company_match("Apple", "Apple Inc.")
    assert company_acronym("American Water Works") == "aww"
    assert not company_match("Apple Inc.", "Microsoft Corporation")


def test_ontology_embedder_sector_proximity():
    import torch
    from gsr_cacl.entity import OntologyMetadataEmbedder, entity_cosine
    from gsr_cacl.entity.supcon import SupConLoss, make_entity_labels
    torch.manual_seed(0)
    sectors = {"Information Technology": ["Apple", "Microsoft", "Nvidia"],
               "Financials": ["JPMorgan", "Citigroup", "Wells Fargo"],
               "Energy": ["Exxon", "Chevron", "ConocoPhillips"]}
    metas = [{"company_name": c, "report_year": y, "company_sector": s,
              "company_industry": s, "company_symbol": c[:3]}
             for s, cs in sectors.items() for c in cs for y in ["2018", "2019"]]
    labels = make_entity_labels(metas)
    model = OntologyMetadataEmbedder(embed_dim=32)
    loss_fn = SupConLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(60):
        loss = loss_fn(model(metas), labels)
        opt.zero_grad(); loss.backward(); opt.step()
    emb = model.encode(metas)
    sec_of = [m["company_sector"] for m in metas]
    comp_of = [m["company_name"] for m in metas]
    same_sec, diff_sec = [], []
    for i in range(len(metas)):
        for j in range(i + 1, len(metas)):
            if comp_of[i] == comp_of[j]:
                continue  # ignore same-entity pairs; we test cross-company structure
            c = float(entity_cosine(emb[i:i+1], emb[j:j+1]))
            (same_sec if sec_of[i] == sec_of[j] else diff_sec).append(c)
    # same-sector companies are, on average, closer than different-sector companies
    assert sum(same_sec) / len(same_sec) > sum(diff_sec) / len(diff_sec)


def test_concept_ontology():
    from gsr_cacl.ontology import canonical_concept, concepts_in_text
    assert canonical_concept("Gross profit") == "GrossProfit"
    assert canonical_concept("Cost of goods sold") == "CostOfRevenue"
    assert canonical_concept("Total net revenue") == "Revenue"
    assert canonical_concept("net cash provided by operating activities") == "OperatingCashFlow"
    assert canonical_concept("random narrative sentence") is None
    cs = concepts_in_text("What was the gross profit and net income in 2019?")
    assert "GrossProfit" in cs and "NetIncome" in cs


def test_concept_coverage_signal():
    from gsr_cacl.ledger import extract_ledger
    from gsr_cacl.scoring.concept_coverage import (
        query_concepts, query_periods, doc_periods_from_ledger,
        concept_coverage_score, expand_derivable)
    led = extract_ledger(table_md=SYNTH_TABLE, doc_id="d1", meta={"company_name": "Acme"})
    dc, dp = led.concept_set(), doc_periods_from_ledger(led)
    assert "GrossProfit" in dc and "Revenue" in dc
    # query asking for a concept+period the doc covers scores higher than one it doesn't
    s_hit = concept_coverage_score(query_concepts("gross profit 2019"), query_periods("gross profit 2019"), dc, dp)
    s_miss = concept_coverage_score(query_concepts("capital expenditures 2019"), query_periods("capital expenditures 2019"), dc, dp)
    assert s_hit > s_miss
    # derivable: a doc with Revenue + CostOfRevenue can answer GrossProfit even if implicit
    assert "GrossProfit" in expand_derivable({"Revenue", "CostOfRevenue"})


def test_concept_equation_verifier():
    from gsr_cacl.ledger import extract_ledger
    from gsr_cacl.scoring.constraint_score import compute_concept_equation_score
    # SYNTH_TABLE: revenue 1000 - cogs 600 = gross profit 400 (consistent)
    led = extract_ledger(table_md=SYNTH_TABLE, doc_id="d1", meta={"company_name": "Acme"})
    good = compute_concept_equation_score(led)
    assert good.total_count >= 1 and good.constraint_score > 0.9
    # a value-identity channel-aligned negative breaks Revenue-COGS=GrossProfit → lower score
    from gsr_cacl.negative_sampler.channel_aligned import make_negative
    import random
    neg = make_negative(SYNTH_TABLE, "value-identity", random.Random(0))
    if neg is not None:
        led_neg = extract_ledger(table_md=neg.table_md, doc_id="d1", meta={"company_name": "Acme"})
        bad = compute_concept_equation_score(led_neg)
        assert bad.constraint_score <= good.constraint_score


def test_preference_reward_and_grpo():
    from gsr_cacl.ledger import extract_ledger
    from gsr_cacl.training.preference import ledger_reward, grpo_advantages
    led = extract_ledger(table_md=SYNTH_TABLE, doc_id="d1", meta={"company_name": "Acme"})
    r_correct = ledger_reward("Answer: 400", led, "gross profit", gold=["400"])
    r_wrong = ledger_reward("Answer: 99999", led, "gross profit", gold=["400"])
    assert r_correct > r_wrong
    adv = grpo_advantages([1.0, 0.0, 0.5])
    assert len(adv) == 3 and abs(sum(adv)) < 1e-6


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
