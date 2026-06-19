"""FactLevelExpert (``factlevel``) — C2: disentangled, gated, fact-level late interaction.

The retrieval breakthrough of LEDGER (docs/RESEARCH_AAAI27.md §C2). It generalises the
training-free FactGate (C6) along three axes that FactGate lacks, to attack the
within-(company,year) residual where pool-local IDF (loclex) plateaus:

  1. **Semantic concept stream** σ_m^sem = ReLU(cos(q_concept_emb, c_f)).  Pure lexical
     jaccard misses badly here — only ~6% of FinQA fact labels map to the 42-concept
     ontology, and table row-labels are paraphrased ("redemption premium" vs
     "extinguishment of debt").  A short-string concept encoder recovers the match,
     directly addressing the *granularity dilemma* (Dense Retrievers Can Fail, EMNLP'25).

  2. **Reasoning-pair term** (§5.5).  Many questions need TWO facts — the same concept at
     two periods (e.g. "% change in revenue from 2018 to 2019").  FactGate scores a single
     best fact; FactLevel additionally rewards a document that contains the same concept at
     BOTH required periods, i.e. that is actually *able to answer* the multi-step question.

  3. **Disentangled magnitude stream** ρ.  When the query carries an explicit numeric
     anchor, fact magnitude (kept as a separate scalar, never fused into the text vector —
     Numeracy Gap, EACL'26) gates the match.  Defaults to 1 when the question asks *for* a
     number (the common case), so it never penalises.

Score:  S(q,d) = γ_e(q,d) · max( best_single_fact ,  best_reasoning_pair )
with γ_e a multiplicative entity gate (≈1 inside a company-scoped pool, decisive otherwise).
Training-free re-scorer (``is_retriever=False``); the learned fusion head decides its weight.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Sequence

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.datasets.gsr_document import extract_table
from gsr_cacl.experts.base import Expert
from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.ledger.numeric import extract_years, parse_financial_number
from gsr_cacl.ontology.aliases import company_match_score
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.retrieval.self_query import CompanyIndex

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "the", "a", "an", "of", "in", "and", "or", "to", "is", "are", "was", "were",
    "for", "from", "with", "at", "by", "on", "as", "its", "that", "this", "what",
    "how", "which", "did", "do", "does", "have", "has", "had", "been", "be",
    "not", "no", "all", "year", "fiscal", "report", "reported", "company",
    "during", "between", "much", "many",
})
_SIG_T = 1.5            # period gate decay (years)
_SEM_FLOOR = 0.35       # cos below this contributes nothing (relu-with-threshold)


def _tokset(text: str) -> frozenset[str]:
    toks = _TOK.findall((text or "").lower())
    return frozenset(t for t in toks if t not in _STOP and not t.isdigit() and len(t) > 1)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _Fact:
    __slots__ = ("toks", "canon", "period", "mag", "emb_idx")

    def __init__(self, toks, canon, period, mag, emb_idx):
        self.toks = toks; self.canon = canon; self.period = period
        self.mag = mag; self.emb_idx = emb_idx


class FactLevelExpert(Expert):
    name = "factlevel"
    is_retriever = False

    def __init__(self, use_embed: bool = True,
                 embed_model: str = "BAAI/bge-small-en-v1.5",
                 device: str | None = None):
        self.use_embed = use_embed
        self.embed_model = embed_model
        self.device = device
        self._model = None

    # ------------------------------------------------------------------ encoder
    def _load(self):
        if not self.use_embed or self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.embed_model, device=dev)
        except Exception as e:  # graceful fallback to lexical-only
            print(f"[factlevel] embed disabled ({e}); lexical σ_m only", flush=True)
            self.use_embed = False
            self._model = None

    def _encode(self, strings: list[str]) -> Optional[np.ndarray]:
        if not self.use_embed or self._model is None or not strings:
            return None
        return np.asarray(self._model.encode(
            strings, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, batch_size=256), dtype=np.float32)

    # ------------------------------------------------------------------ index
    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        self._load()
        self._doc_facts: list[list[_Fact]] = []
        self._doc_company: list[str] = []
        label_strings: list[str] = []      # flat list of distinct-ish fact labels to embed
        for d, m in zip(corpus, doc_metas):
            self._doc_company.append(str((m or {}).get("company_name", "") or "").strip())
            facts: list[_Fact] = []
            tmd = extract_table(d.page_content)
            ledger = None
            if tmd:
                try:
                    ledger = extract_ledger_from_table(tmd)
                except Exception:
                    ledger = None
            if ledger is not None:
                for f in ledger.facts:
                    label = f.concept or ""
                    toks = _tokset(label)
                    period = None
                    if f.period is not None:
                        try:
                            period = int(float(str(f.period)))
                        except (ValueError, TypeError):
                            period = None
                    mag = None
                    if f.value is not None:
                        v = f.value * (f.scale or 1.0)
                        mag = math.copysign(math.log10(1.0 + abs(v)), v) if v else 0.0
                    emb_idx = len(label_strings) if (self.use_embed and label) else -1
                    if emb_idx >= 0:
                        label_strings.append(label)
                    facts.append(_Fact(toks, f.concept_canonical, period, mag, emb_idx))
            self._doc_facts.append(facts)
        # Embed all fact labels once
        self._label_emb = self._encode(label_strings) if label_strings else None
        self._comp_index = CompanyIndex(list(doc_metas))

    # ------------------------------------------------------------------ queries
    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        self._q_company = [self._comp_index.detect(q) for q in raw_queries]
        self._q_years = [sorted(set(extract_years(q))) for q in raw_queries]
        self._q_toks = [_tokset(q) for q in raw_queries]
        self._q_concepts = [concepts_in_text(q) for q in raw_queries]
        # numeric anchor in the query (rare): magnitude of any explicit number
        self._q_mag = []
        for q in raw_queries:
            nums = [parse_financial_number(t) for t in _TOK.findall(q)]
            nums = [n for n in nums if n is not None and abs(n) >= 100]  # ignore years/small ints
            self._q_mag.append(
                math.copysign(math.log10(1.0 + abs(nums[0])), nums[0]) if nums else None)
        self._q_emb = self._encode([q for q in raw_queries]) if self.use_embed else None

    # ------------------------------------------------------------------ scoring
    def _entity_gate(self, qi: int, di: int) -> float:
        qk = self._q_company[qi]
        if qk is None:
            return 1.0
        doc_co = self._doc_company[di]
        if not doc_co:
            return 0.4
        return company_match_score(qk, doc_co)

    def _sigma_m(self, qi: int, f: _Fact) -> float:
        jac = _jaccard(self._q_toks[qi], f.toks)
        onto = 0.7 if (f.canon is not None and f.canon in self._q_concepts[qi]) else 0.0
        sem = 0.0
        if (self._q_emb is not None and self._label_emb is not None and f.emb_idx >= 0):
            cos = float(self._label_emb[f.emb_idx] @ self._q_emb[qi])
            sem = max(0.0, (cos - _SEM_FLOOR) / (1.0 - _SEM_FLOOR))  # rescale [floor,1]→[0,1]
        return max(jac, onto, sem)

    def _gamma_t(self, qi: int, period: Optional[int]) -> float:
        yrs = self._q_years[qi]
        if period is None or not yrs:
            return 1.0
        return math.exp(-min(abs(period - y) for y in yrs) / _SIG_T)

    def _rho(self, qi: int, f: _Fact) -> float:
        qm = self._q_mag[qi]
        if qm is None or f.mag is None:
            return 1.0   # no numeric anchor → neutral
        return math.exp(-abs(qm - f.mag) / 1.0)

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        yrs = self._q_years[qi]
        multi = len(yrs) >= 2
        out = np.zeros(len(pool), dtype=np.float64)
        for pi, di in enumerate(pool):
            facts = self._doc_facts[di]
            if not facts:
                continue
            ge = self._entity_gate(qi, di)
            if ge == 0.0:
                continue
            # per-fact score s_f = σ_m · γ_t · ρ
            best_single = 0.0
            scored = []   # (sigma_m, period, s_f) for reasoning pairs
            for f in facts:
                sm = self._sigma_m(qi, f)
                if sm == 0.0:
                    continue
                sf = sm * self._gamma_t(qi, f.period) * self._rho(qi, f)
                if sf > best_single:
                    best_single = sf
                if multi:
                    scored.append((sm, f.period))
            best = best_single
            # reasoning-pair: same-ish concept at the two requested periods
            if multi and scored:
                y0, y1 = yrs[0], yrs[-1]
                # best σ_m whose period is near each requested year
                def near(y):
                    vals = [sm * math.exp(-abs((p if p is not None else y) - y) / _SIG_T)
                            for sm, p in scored]
                    return max(vals) if vals else 0.0
                pair = near(y0) * near(y1)
                if pair > best:
                    best = pair
            out[pi] = ge * best
        return out
