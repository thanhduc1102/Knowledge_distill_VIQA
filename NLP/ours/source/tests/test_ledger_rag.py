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
