"""LedgerRetrieval — the corrected, metadata-aware retrieval flow (fixes B1/B2/B3/B5).

Differences from the original ``GSRRetrieval``:
  * B1: no randomly-initialized GAT at inference — the structural signal is the
    deterministic *equation-faithful* constraint score (trainable GAT remains optional).
  * B2: a SINGLE embedding function is used (no train/eval encoder mismatch).
  * B3: the entity signal is ``cos(e_Q, e_D)`` from a TRAINED metadata embedder, not
    string matching.
  * SOTA signal: metadata-aware candidate construction — the candidate set is the UNION of
    the dense top-N AND all same-company(-year) documents, so the gold document is reachable
    even when its text similarity is low (the EDA-confirmed failure mode). Soft fallback
    keeps recall when the metadata mask is empty.

Score:  s(Q,D) = α·cos_text(Q,D) + β·cos(e_Q,e_D) + γ·CS_eq(G_D)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from gsr_cacl.core import Document
from gsr_cacl.kg.builder import build_kg_from_markdown
from gsr_cacl.scoring.constraint_score import compute_equation_constraint_score
from gsr_cacl.datasets.gsr_document import extract_table


class LedgerRetrieval:
    def __init__(
        self,
        corpus: list[Document],
        embedding_function: Any,
        entity_embedder: Optional[torch.nn.Module] = None,
        tables: Optional[list[str]] = None,
        top_k: int = 3,
        alpha: float = 1.0,
        beta: float = 0.6,
        gamma: float = 0.2,
        candidate_mult: int = 10,
        year_window: int = 1,
        use_metadata_filter: bool = True,
        alias_match: bool = False,
        device: str | None = None,
    ):
        self.corpus = corpus
        self.embedding_function = embedding_function
        self.entity_embedder = entity_embedder
        self.top_k = top_k
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.candidate_mult = candidate_mult
        self.year_window = year_window
        self.use_metadata_filter = use_metadata_filter
        # E2: when True, company keys are canonicalised (alias/suffix robust) and a fuzzy
        # fallback scan recovers acronym/ticker variants. Default False = exact-match baseline.
        self.alias_match = alias_match
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._build_faiss_index()
        self._build_metadata_index()
        self._build_constraint_cache(tables)
        self._build_entity_cache()

    # ------------------------------------------------------------------ indexing
    def _build_faiss_index(self) -> None:
        import faiss
        texts = [d.page_content for d in self.corpus]
        if hasattr(self.embedding_function, "embed_documents"):
            embs = self.embedding_function.embed_documents(texts)
        else:
            embs = [self.embedding_function.embed_query(t) for t in texts]
        self.doc_text_embeds = np.asarray(embs, dtype="float32")
        faiss.normalize_L2(self.doc_text_embeds)
        self.faiss_index = faiss.IndexFlatIP(self.doc_text_embeds.shape[1])
        self.faiss_index.add(self.doc_text_embeds)

    def _build_metadata_index(self) -> None:
        from gsr_cacl.ontology import normalize_company

        self.company_to_idx: dict[str, list[int]] = {}
        self.doc_years: list[Optional[int]] = []
        for i, d in enumerate(self.corpus):
            raw = str(d.meta_data.get("company_name", "")).lower().strip()
            key = normalize_company(raw) if self.alias_match else raw
            self.company_to_idx.setdefault(key, []).append(i)
            try:
                self.doc_years.append(int(float(str(d.meta_data.get("report_year", "")).strip())))
            except (ValueError, TypeError):
                self.doc_years.append(None)
        # unique company surfaces (for the fuzzy fallback used only under alias_match)
        self._company_keys = list(self.company_to_idx.keys())

    def _company_matches(self, query_company: str) -> list[int]:
        """Corpus indices whose company matches ``query_company`` (alias-aware if enabled)."""
        from gsr_cacl.ontology import normalize_company, company_match

        if not self.alias_match:
            return self.company_to_idx.get(query_company.lower().strip(), [])
        key = normalize_company(query_company)
        hits = list(self.company_to_idx.get(key, []))
        if hits:
            return hits
        # fuzzy fallback: acronym/ticker/word-order variants (corpus is small → cheap)
        for ck in self._company_keys:
            if ck and company_match(key, ck):
                hits.extend(self.company_to_idx[ck])
        return hits

    def _build_constraint_cache(self, tables: Optional[list[str]]) -> None:
        self.cs_cache = np.ones(len(self.corpus), dtype="float32")
        for i, d in enumerate(tqdm(self.corpus, desc="Equation-CS cache", leave=False)):
            tmd = tables[i] if tables else extract_table(d.page_content)
            if not tmd:
                continue
            kg = build_kg_from_markdown(tmd)
            self.cs_cache[i] = compute_equation_constraint_score(kg).constraint_score

    def _build_entity_cache(self) -> None:
        if self.entity_embedder is None:
            self.doc_entity_embeds = None
            return
        metas = [d.meta_data for d in self.corpus]
        with torch.no_grad():
            self.doc_entity_embeds = self.entity_embedder.encode(metas).to(self.device)

    # ------------------------------------------------------------------ retrieval
    def _candidate_indices(self, q_vec: np.ndarray, query_meta: Optional[dict]) -> list[int]:
        k = min(self.top_k * self.candidate_mult, len(self.corpus))
        _, idx = self.faiss_index.search(q_vec, k)
        cands = {int(i) for i in idx[0] if int(i) >= 0}

        if self.use_metadata_filter and query_meta:
            company = str(query_meta.get("company_name", "")).strip()
            try:
                qyear = int(float(str(query_meta.get("report_year", "")).strip()))
            except (ValueError, TypeError):
                qyear = None
            for j in self._company_matches(company):
                if qyear is None or self.doc_years[j] is None or \
                        abs(self.doc_years[j] - qyear) <= self.year_window:
                    cands.add(j)
        return list(cands)

    def _rank_candidates(self, query: str, query_meta: Optional[dict] = None):
        import faiss
        q_vec = np.asarray(self.embedding_function.embed_query(query), dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q_vec)

        cands = self._candidate_indices(q_vec, query_meta)
        if not cands:
            return []

        cand_text = self.doc_text_embeds[cands]                       # [C, d]
        s_text = (cand_text @ q_vec[0]).astype("float32")            # cosine (normalized)
        s_struct = self.cs_cache[cands]

        if self.doc_entity_embeds is not None and query_meta is not None:
            with torch.no_grad():
                e_q = self.entity_embedder.encode([query_meta]).to(self.device)[0]
                e_d = self.doc_entity_embeds[cands]
                s_entity = (e_d @ e_q).detach().cpu().numpy().astype("float32")
        else:
            s_entity = np.zeros(len(cands), dtype="float32")

        final = self.alpha * s_text + self.beta * s_entity + self.gamma * s_struct
        order = np.argsort(-final)[: self.top_k]
        return [(self.corpus[cands[o]], float(final[o])) for o in order]

    def retrieve(self, query: str, query_meta: Optional[dict] = None) -> list[Document]:
        return [d for d, _ in self._rank_candidates(query, query_meta)]

    def retrieve_with_scores(self, query: str, query_meta: Optional[dict] = None):
        return self._rank_candidates(query, query_meta)

    def retrieve_batch(self, queries, queries_meta=None, return_scores=False):
        queries_meta = queries_meta or [None] * len(queries)
        out = []
        for q, m in tqdm(zip(queries, queries_meta), total=len(queries), desc="LedgerRetrieval"):
            r = self._rank_candidates(q, m)
            out.append(r if return_scores else [d for d, _ in r])
        return out
