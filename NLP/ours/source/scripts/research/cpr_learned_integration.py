"""Does learned operand attribution improve CPR reliability? (operand-learning payoff)

Leak-safe 5-fold CV over the strong-generator queries:
  * train an operand scorer on the gold-doc distant labels of the TRAIN-fold queries;
  * for each TEST-fold query, score the UNION (retrieved) ledger facts, keep the top-m,
    and run verify_cpr on that restricted ledger;
  * compare AUROC of CPR-on-restricted-ledger vs CPR-on-full-ledger vs value-only.

The scorer never trains on the query it scores; features use no answer info. Hypothesis:
restricting to learned operands removes coincidental value matches → sharper CPR.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import extract_final_number
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.extract import extract_ledger_from_table, extract_ledger
from gsr_cacl.ledger.numeric import number_match, numbers_close, parse_financial_number, extract_years
from gsr_cacl.research.derivation import derivation_depth
from gsr_cacl.research.cpr_verifier import verify_cpr
from gsr_cacl.ledger.select import _tokens
from gsr_cacl.ontology.concepts import concepts_in_text

RETR = {"finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
        "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
        "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl"}
TRAIN_SRC = "golddoc"  # set by --train-src


def _gold_value(g):
    if g is None:
        return None
    if isinstance(g, list):
        for x in g:
            v = parse_financial_number(str(x))
            if v is not None:
                return v
        return None
    return parse_financial_number(str(g))


def _union(pack):
    if not pack.ranked:
        return None
    m = FactLedger(doc_id="u", facts=list(pack.ranked[0].ledger.facts), meta=dict(pack.ranked[0].ledger.meta))
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


def _gold_doc_ledger(rec):
    gid = str(rec.get("ground_truth_id") or "")
    for d in rec.get("retrieved", []):
        if str(d.get("context_id") or d.get("id")) == gid:
            tbl = d.get("table"); ctx = d.get("page_content") or d.get("context") or ""
            meta = d.get("meta") or d.get("metadata") or {}
            return (extract_ledger_from_table(tbl, doc_id=gid, meta=meta, caption=ctx[:200])
                    if tbl else extract_ledger(context=ctx, doc_id=gid, meta=meta))
    return None


def _cc(fact, q_concepts, q_tokens):
    if fact.concept_canonical and q_concepts and fact.concept_canonical in q_concepts:
        return 1.0
    ftoks = _tokens(fact.concept) | _tokens(fact.column_header or "")
    return min(1.0, len(q_tokens & ftoks) / 2.0) if ftoks else 0.0


def _pint(f):
    try:
        return int(str(f.period)[:4]) if f.period else None
    except (ValueError, TypeError):
        return None


def featurize(query, facts, embedder):
    q_tokens = _tokens(query); q_concepts = concepts_in_text(query); q_years = set(extract_years(query))
    fstrs = [f"{f.concept} {f.period or ''}".strip() for f in facts]
    qe = embedder.encode([query], normalize_embeddings=True)[0]
    fe = embedder.encode(fstrs, normalize_embeddings=True) if fstrs else np.zeros((0, qe.shape[0]))
    vals = [abs(f.value) for f in facts if f.value is not None]; mx = max(vals) if vals else 1.0
    rows = []
    for i, f in enumerate(facts):
        cos = float(np.dot(qe, fe[i])) if len(fe) else 0.0
        p = _pint(f); pm = 1.0 if (p is not None and p in q_years) else (0.5 if p is None else 0.0)
        v = abs(f.value) if f.value is not None else 0.0
        rows.append([cos, _cc(f, q_concepts, q_tokens), pm,
                     np.log1p(v) / np.log1p(mx) if mx > 0 else 0.0,
                     1.0 if ("total" in (f.concept or "").lower()) else 0.0, 1.0])
    return np.array(rows) if rows else np.zeros((0, 6))


def _auroc(s, y):
    pos = sum(y); neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(s)), key=lambda i: s[i]); r = [0.0] * len(s); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        a = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = a
        i = j + 1
    return (sum(r[i] for i in range(len(s)) if y[i]) - pos * (pos + 1) / 2) / (pos * neg)


def run(ds, embedder, topm=8, folds=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    retr = {str(r.get("query_id")): r for r in (json.loads(l) for l in open(RETR[ds]) if l.strip())}
    preds = [json.loads(l) for l in open(f"outputs/research/gemini_gen/{ds}_predictions.jsonl") if l.strip()]

    items = []  # per query: train feats (gold-doc) + y, inference (union ledger, raw, gold, correct)
    for p in preds:
        rec = retr.get(str(p.get("query_id")))
        if not rec or not rec.get("retrieved"):
            continue
        q = p.get("query") or p.get("question"); gold = p.get("gold"); raw = p.get("raw_pred") or ""
        correct = bool(p["raw_correct"]) if p.get("raw_correct") is not None else bool(number_match(raw, gold))
        pack = build_evidence_pack(q, rec["retrieved"], top_n_facts=8); uni = _union(pack)
        if uni is None:
            continue
        uni_facts = [f for f in uni.numeric_facts() if f.value is not None]
        gv = _gold_value(rec.get("gold"))
        # training signal: choose source ledger (gold-doc clean, or union in-distribution)
        src_facts = uni_facts if TRAIN_SRC == "union" else (
            [f for f in (_gold_doc_ledger(rec) or FactLedger("g", [], {})).numeric_facts() if f.value is not None])
        Xtr = np.zeros((0, 6)); ytr = np.zeros((0,), dtype=int)
        if gv is not None and len(src_facts) >= 2:
            vals = [f.value for f in src_facts]
            depth, ops = derivation_depth(gv, vals, max_ops=3, return_operands=True)
            if depth is not None:
                opv = [v for v in vals if numbers_close(gv, v, 1e-2)] if depth == "grounded" else list(ops)
                y = np.array([1 if any(numbers_close(f.value, o, 1e-2) for o in opv) else 0 for f in src_facts])
                if y.sum() > 0:
                    Xtr = featurize(q, src_facts, embedder); ytr = y
        items.append({"q": q, "gold": gold, "raw": raw, "correct": correct,
                      "uni_facts": uni_facts, "uni_meta": dict(uni.meta),
                      "pack_sel": pack.selected_facts, "Xtr": Xtr, "ytr": ytr})

    n = len(items)
    idx = list(range(n)); random.Random(seed).shuffle(idx); fold = {idx[i]: i % folds for i in range(n)}
    base_cpr, restr_cpr, soft_cpr, vonly, corr = [], [], [], [], []
    for f in range(folds):
        tr = [i for i in range(n) if fold[i] != f and items[i]["ytr"].sum() > 0]
        te = [i for i in range(n) if fold[i] == f]
        if len(tr) < 10:
            continue
        X = np.vstack([items[i]["Xtr"] for i in tr]); Y = np.concatenate([items[i]["ytr"] for i in tr])
        if len(set(Y)) < 2:
            continue
        clf = LogisticRegression(max_iter=600, C=1.0, class_weight="balanced").fit(X, Y)
        for i in te:
            it = items[i]; facts = it["uni_facts"]
            full = FactLedger(doc_id="u", facts=facts, meta=it["uni_meta"])
            c_full = verify_cpr(it["raw"], full, it["q"], gold=it["gold"], selected_facts=it["pack_sel"])
            if facts:
                Xu = featurize(it["q"], facts, embedder)
                sc = clf.predict_proba(Xu)[:, 1]
                keep = [facts[j] for j in np.argsort(-sc)[:topm]]
                wmap = {id(facts[j]): float(sc[j]) for j in range(len(facts))}
            else:
                keep = facts; wmap = {}
            # (a) HARD restrict
            restr = FactLedger(doc_id="r", facts=keep, meta=it["uni_meta"])
            c_restr = verify_cpr(it["raw"], restr, it["q"], gold=it["gold"], selected_facts=it["pack_sel"])
            # (b) SOFT weight on full ledger (down-weight low-attribution, never drop)
            c_soft = verify_cpr(it["raw"], full, it["q"], gold=it["gold"], selected_facts=it["pack_sel"],
                                fact_weight_fn=lambda f, _w=wmap: _w.get(id(f), 0.5))
            base_cpr.append(c_full.confidence); restr_cpr.append(c_restr.confidence)
            soft_cpr.append(c_soft.confidence)
            vonly.append(1.0 if c_full.value_only_grounded else 0.0); corr.append(it["correct"])
    return {"dataset": ds, "n_eval": len(corr), "topm": topm,
            "auroc_cpr_full": round(_auroc(base_cpr, corr), 4),
            "auroc_cpr_learned_soft": round(_auroc(soft_cpr, corr), 4),
            "auroc_cpr_learned_restricted_hard": round(_auroc(restr_cpr, corr), 4),
            "auroc_value_only": round(_auroc(vonly, corr), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--train-src", choices=["golddoc", "union"], default="golddoc")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/research/cpr_learned/report.json")
    args = ap.parse_args()
    global TRAIN_SRC
    TRAIN_SRC = args.train_src
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(args.model, device=args.device)
    allr = {}
    for ds in args.datasets:
        r = run(ds, emb, topm=args.topm); allr[ds] = r
        print(json.dumps(r), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allr, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
