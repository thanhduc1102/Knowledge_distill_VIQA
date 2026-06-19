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
    vr3 = verify("Use revenue 1000 and cost 600.\n1000 - 600 = 400.\nAnswer: 400",
                 led, "gross profit 2019", gold=["400"])
    assert vr3.grounding_fraction > 0.5
    assert vr3.arithmetic_fraction == 1.0


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


def test_retrieval_bridge_arbitrates_top3_and_explains():
    from gsr_cacl.generation.retrieval_bridge import build_evidence_pack, render_prompt_context

    noisy = """| item | 2019 | 2018 |
| --- | --- | --- |
| revenue | 777 | 700 |
| net income | 50 | 40 |
"""
    correct = """| item | 2019 | 2018 |
| --- | --- | --- |
| revenue | 1000 | 900 |
| cost of goods sold | 600 | 550 |
| gross profit | 400 | 350 |
"""
    retrieved = [
        {"id": "noise_doc", "score": 0.99, "meta": {"company_name": "Acme"}, "table": noisy},
        {"id": "gold_doc", "score": 0.80, "meta": {"company_name": "Acme"}, "table": correct},
    ]
    pack = build_evidence_pack("Acme: What was the gross profit in 2019?", retrieved)
    assert pack.ranked[0].doc_id == "gold_doc"
    assert pack.calculation["answer"] == 400.0
    assert pack.provenance and pack.provenance[0]["cell"].startswith("gold_doc")
    context = render_prompt_context(pack)
    assert "KG_SELECTED_DOC: gold_doc" in context
    assert "KG_SYMBOLIC_ANSWER: 400" in context


def test_financial_rule_plans_balance_comparison_adjustment():
    from gsr_cacl.ledger import extract_ledger, build_evidence_block
    from gsr_cacl.ledger.select import calculation_plan, select_facts

    tax_table = """| item | 2015 | 2014 | 2013 |
| --- | --- | --- | --- |
| balance january 1 | $ 1171 | $ 1701 | $ 1573 |
| additions based on tax positions related to the current year | 67 | 63 | 90 |
| reductions for tax positions of prior years | -84 ( 84 ) | -220 ( 220 ) | -141 ( 141 ) |
| balance december 31 | $ 1136 | $ 1171 | $ 1701 |
"""
    tax = extract_ledger(table_md=tax_table, doc_id="tax", meta={"company_name": "Comcast"})
    q_tax = "What was the change in unrecognized tax benefits from the end of 2014 to the end of 2015?"
    tax_facts = select_facts(q_tax, [tax], top_n=8)
    tax_plan = calculation_plan(q_tax, tax_facts)
    assert tax_plan["answer"] == -35.0
    assert "balance december 31" in tax_plan["trace"].lower() or tax_plan["confidence"] >= 0.8
    tax_block = build_evidence_block(q_tax, [tax], facts=tax_facts)
    assert "balance december 31" in tax_block.lower()
    assert "additions based" not in tax_block.lower().split("FOCUS FACTS:")[0]

    return_table = """| date | altria group inc . | altria group inc . peer group | s&p 500 |
| --- | --- | --- | --- |
| december 2011 | $ 100.00 | $ 100.00 | $ 100.00 |
| december 2016 | $ 286.61 | $ 192.56 | $ 198.09 |
"""
    ret = extract_ledger(table_md=return_table, doc_id="ret", meta={"company_name": "Altria"})
    q_ret = "Did Altria Group, Inc.'s cumulative total shareholder return exceed that of the S&P 500 over the five-year period ending December 31, 2016?"
    ret_plan = calculation_plan(q_ret, select_facts(q_ret, [ret], top_n=8))
    assert ret_plan["answer"] == 1.0
    assert ret_plan["confidence"] >= 0.8

    rev_table = """| item | amount ( in millions ) |
| --- | --- |
| 2015 net revenue | $ 5829 |
| retail electric price | 289 |
| 2016 net revenue | $ 6179 |
"""
    rev = extract_ledger(table_md=rev_table, doc_id="rev", meta={"company_name": "Entergy"})
    q_adj = "Assuming there had been no sale, what would net revenue have been for 2015 without the $100 million net-of-tax gain?"
    adj_plan = calculation_plan(q_adj, select_facts(q_adj, [rev], top_n=6))
    assert adj_plan["answer"] == 5729.0
    assert adj_plan["confidence"] >= 0.8

    lease_table = """| item | operating leases | capital leases |
| --- | --- | --- |
| fiscal 2019 | $ 137.4 | $ 0.3 |
| fiscal 2020 | 115.7 | 0.2 |
| total noncancelable future lease commitments | $ 559.3 | $ 0.5 |
"""
    lease = extract_ledger(table_md=lease_table, doc_id="lease", meta={"company_name": "General Mills"})
    q_lease = "What proportion of total noncancelable future lease commitments are scheduled to be paid in fiscal year 2019?"
    lease_plan = calculation_plan(q_lease, select_facts(q_lease, [lease], top_n=8))
    assert round(lease_plan["answer"], 6) == round(137.4 / 559.3, 6)
    assert lease_plan["confidence"] >= 0.8

    stock_table = """| item | 12/31/2010 | 12/31/2011 | 12/31/2014 |
| --- | --- | --- | --- |
| hum | $ 125 | $ 201 | $ 342 |
| s&p 500 | $ 115 | $ 117 | $ 205 |
"""
    stock = extract_ledger(table_md=stock_table, doc_id="stock", meta={"company_name": "Humana"})
    q_stock = "What was the percent change in Humana's stock price from $125 at the end of 2010 to $201 at the end of 2011, for the five years ended December 31, 2014?"
    stock_plan = calculation_plan(q_stock, select_facts(q_stock, [stock], top_n=8))
    assert round(stock_plan["answer"], 6) == 0.608


def test_tat_context_bridge_periods_question_ratio_and_direct_percent():
    from gsr_cacl.generation.generator import ExtractiveGenerator
    from gsr_cacl.generation.retrieval_bridge import build_evidence_pack, render_prompt_context

    spirent_context = """3. Operating segments continued

|                | 2019 $ million | 2018 $ million |
|----------------|---------------|---------------|
| Americas       | 196.9         | 184.6         |
| Asia Pacific   | 7.4           | 4.4           |
| Europe, Middle East and Africa | 11.5 | 5.1 |
| **Total**      | **215.8**     | **194.1**     |

Europe, Middle East and Africa includes United Kingdom non-current assets of $6.9 million (2018 $2.0 million).
"""
    spirent_doc = {
        "id": "spirent",
        "meta": {"company_name": "spirent-communications-plc"},
        "table": "",
        "page_content": spirent_context,
    }
    q_year = (
        "spirent-communications-plc: In which year did the non-current assets in the "
        "Asia Pacific region exceed those of the previous year?"
    )
    year_pack = build_evidence_pack(q_year, [spirent_doc], top_n_facts=6)
    assert year_pack.calculation["answer"] == 2019.0
    assert any(op.get("concept") == "Asia Pacific" for op in year_pack.calculation["operands"])
    assert any(f.period == "2019" for f in year_pack.selected_facts)

    q_ratio = (
        "spirent-communications-plc: What percentage of the total non-current assets "
        "in Europe, Middle East and Africa did the United Kingdom's $6.9 million "
        "non-current assets represent in 2019?"
    )
    ratio_pack = build_evidence_pack(q_ratio, [spirent_doc], top_n_facts=6)
    assert round(ratio_pack.calculation["answer"], 6) == round(6.9 / 11.5, 6)
    assert ratio_pack.calculation["operands"][0]["provenance"] == "question"
    assert ExtractiveGenerator().generate(
        q_ratio,
        render_prompt_context(ratio_pack),
        facts=ratio_pack.selected_facts,
    ) == "Answer: 0.6"

    bce_context = """At the end of 2019, BCE retail customer connections totaled 18,983,510, and were comprised of the following:

- 9,957,962 wireless subscribers, up 3.6% compared to 2018.

|       | 2019 | 2018 | $ CHANGE | % CHANGE |
|-------|------|------|----------|----------|
| Bell Wireless | 9,142 | 8,818 | 324 | 3.7% |
"""
    bce_doc = {
        "id": "bce",
        "meta": {"company_name": "bce-inc"},
        "table": "",
        "page_content": bce_context,
    }
    q_pct = "bce-inc: What percentage change occurred in the number of BCE retail wireless subscribers from 2018 to 2019?"
    pct_pack = build_evidence_pack(q_pct, [bce_doc], top_n_facts=6)
    assert pct_pack.calculation["answer"] == 3.6
    assert pct_pack.calculation["operands"][0]["source"] == "text"


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
