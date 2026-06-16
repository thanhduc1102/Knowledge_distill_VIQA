"""EntityExpert — ontology metadata embedding with supervised-contrastive metric learning.

Independent method #3. Combines:
  * E1 GICS sector taxonomy + E2 company-alias canonicalisation (``ontology`` + the
    ``OntologyMetadataEmbedder`` architecture: sector/industry/company/symbol/year features),
  * SupCon metric learning (Khosla et al. 2020): same-(company,year) documents are pulled
    together, others pushed apart → a metadata space where the gold document sits near the
    query's self-queried entity.

Honesty: the QUERY-side metadata is read from the question (``retrieval.self_query`` for
company, regex for year) — never from the gold ``company_name`` field. Document-side metadata
is corpus-side and known at index time (not a leak). When the question names no company and
no year, the query vector carries no entity signal and the per-query min-max collapses it to
a neutral 0.5 (cannot hurt fusion).
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.experts.base import Expert
from gsr_cacl.entity.encoder import OntologyMetadataEmbedder
from gsr_cacl.entity.supcon import SupConLoss, make_entity_labels
from gsr_cacl.retrieval.self_query import CompanyIndex
from gsr_cacl.ledger.numeric import extract_years


class EntityExpert(Expert):
    name = "entity"
    is_retriever = False

    def __init__(self, embed_dim: int = 128, epochs: int = 12, batch: int = 256,
                 lr: float = 1e-3, device: str = "cpu"):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.device = device

    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        import torch

        self._metas = list(doc_metas)
        self._comp_index = CompanyIndex(doc_metas)
        self.emb = OntologyMetadataEmbedder(embed_dim=self.embed_dim).to(self.device)

        # SupCon warm-up over corpus metadata (same-(company,year) → positives).
        labels = make_entity_labels(self._metas).to(self.device)
        crit = SupConLoss(temperature=0.1)
        opt = torch.optim.AdamW(self.emb.parameters(), lr=self.lr)
        n = len(self._metas)
        self.emb.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for b in range(0, n, self.batch):
                idx = perm[b:b + self.batch].tolist()
                e = self.emb([self._metas[i] for i in idx])
                loss = crit(e, labels[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        self.emb.eval()
        with torch.no_grad():
            self._doc_emb = self.emb(self._metas).cpu().numpy().astype(np.float64)  # [N, d], L2-normed

    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        import torch

        # Honest self-queried meta: company from the question, year from the question.
        q_metas = []
        for q in raw_queries:
            comp = self._comp_index.detect(q) or ""
            yrs = extract_years(q)
            q_metas.append({
                "company_name": comp,
                "report_year": str(yrs[0]) if yrs else "",
                "company_sector": "", "company_industry": "", "company_symbol": "",
            })
        with torch.no_grad():
            self._q_emb = self.emb(q_metas).cpu().numpy().astype(np.float64)  # [Q, d]
        # has-signal flag: no company AND no year → entity vector is uninformative
        self._has_sig = [bool(self._comp_index.detect(q) or extract_years(q)) for q in raw_queries]

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        if not self._has_sig[qi]:
            return np.zeros(len(pool), dtype=np.float64)  # neutral; min-max → 0.5
        qv = self._q_emb[qi]
        return self._doc_emb[np.asarray(pool, dtype=int)] @ qv  # cosine (both L2-normed)
