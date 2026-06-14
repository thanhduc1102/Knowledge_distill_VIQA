"""Trained metadata/entity embedding (the contribution-1 idea, done properly).

The paper claims a metadata signal learned with Supervised Contrastive Learning, whose
cosine becomes a retrieval score combined with text + constraint. In the current code,
the "entity score" is just exact string matching (bug B3). This module replaces that with
a real, trainable ``e = EntityEncoder(metadata)`` whose ``cos(e_Q, e_D)`` is a genuine,
learnable signal.

``HashMetadataEmbedder`` is self-contained (no backbone needed) so it trains from scratch
in seconds/minutes — ideal for the entity SupCon stage. It exploits the RICH metadata the
dataset actually provides (company, sector, industry, symbol, year), not just 3 fields.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsr_cacl.ontology import normalize_company, sector_id, N_SECTORS

_CAT_FIELDS = ["company_name", "company_sector", "company_industry", "company_symbol"]


def _hash_bucket(text: str, n_buckets: int) -> int:
    h = 0
    for c in str(text).lower().strip():
        h = (h * 131 + ord(c)) % n_buckets
    return h


def _year_features(year_str: str) -> list[float]:
    try:
        y = int(float(str(year_str).strip()))
    except (ValueError, TypeError):
        return [0.0, 0.0, 0.0]
    norm = (y - 2010) / 20.0
    return [norm, math.sin(y / 3.0), math.cos(y / 3.0)]


class HashMetadataEmbedder(nn.Module):
    """Self-contained entity embedder: hash categorical fields + year features → L2 vec."""

    def __init__(self, embed_dim: int = 128, n_buckets: int = 4096, field_dim: int = 32):
        super().__init__()
        self.n_buckets = n_buckets
        self.cat_fields = _CAT_FIELDS
        self.embeddings = nn.ModuleDict({
            f: nn.Embedding(n_buckets, field_dim) for f in self.cat_fields
        })
        in_dim = field_dim * len(self.cat_fields) + 3  # +3 year features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def featurize(self, metas: list[dict[str, Any]], device) -> tuple[dict, torch.Tensor]:
        cat_idx = {
            f: torch.tensor([_hash_bucket(m.get(f, ""), self.n_buckets) for m in metas],
                            dtype=torch.long, device=device)
            for f in self.cat_fields
        }
        year_feats = torch.tensor(
            [_year_features(m.get("report_year", "")) for m in metas],
            dtype=torch.float32, device=device,
        )
        return cat_idx, year_feats

    def forward(self, metas: list[dict[str, Any]]) -> torch.Tensor:
        device = next(self.parameters()).device
        cat_idx, year_feats = self.featurize(metas, device)
        parts = [self.embeddings[f](cat_idx[f]) for f in self.cat_fields]
        parts.append(year_feats)
        x = torch.cat(parts, dim=-1)
        e = self.norm(self.mlp(x))
        return F.normalize(e, p=2, dim=-1)

    @torch.no_grad()
    def encode(self, metas: list[dict[str, Any]]) -> torch.Tensor:
        self.eval()
        return self.forward(metas)


def entity_cosine(e_q: torch.Tensor, e_d: torch.Tensor) -> torch.Tensor:
    """cos(e_Q, e_D) → s_entity ∈ [-1, 1]; both inputs assumed L2-normalized."""
    return (e_q * e_d).sum(dim=-1)


class OntologyMetadataEmbedder(nn.Module):
    """Ontology-grounded entity embedder (contributions E1 + E2).

    Two structural priors are baked into the architecture so the learned space carries
    *economic* structure instead of being a near-deterministic hash of the label:

      * **E1 — GICS hierarchy.** ``company_sector``/``company_industry`` are canonicalised to
        one of the 11 GICS sectors (see :mod:`gsr_cacl.ontology.gics`) and embedded in a small
        **shared** table. Because every company in a sector reuses the same sector vector, and
        the industry vector is *nested under* its sector (``ind ← ind + W·sector``), companies
        in the same sector end up closer than companies in different sectors — useful for
        sector-level queries and cross-company generalisation.
      * **E2 — alias canonicalisation.** The company string is normalised
        (:func:`gsr_cacl.ontology.aliases.normalize_company`) *before* hashing, so
        ``"American Water Works"`` and ``"American Water Works Company, Inc."`` map to the
        same bucket — robust to legal-suffix / word-order noise.

    Drop-in compatible with :class:`HashMetadataEmbedder` (same ``forward``/``encode`` API),
    so it can be swapped in retrieval/CACL without touching the call sites.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_company_buckets: int = 4096,
        n_industry_buckets: int = 512,
        sector_dim: int = 16,
        industry_dim: int = 24,
        company_dim: int = 48,
        symbol_dim: int = 16,
        n_symbol_buckets: int = 1024,
    ):
        super().__init__()
        self.n_company_buckets = n_company_buckets
        self.n_industry_buckets = n_industry_buckets
        self.n_symbol_buckets = n_symbol_buckets

        self.sector_emb = nn.Embedding(N_SECTORS, sector_dim)
        self.industry_emb = nn.Embedding(n_industry_buckets, industry_dim)
        self.company_emb = nn.Embedding(n_company_buckets, company_dim)
        self.symbol_emb = nn.Embedding(n_symbol_buckets, symbol_dim)
        # hierarchy injection: sector → industry subspace
        self.sector_to_ind = nn.Linear(sector_dim, industry_dim)

        in_dim = sector_dim + industry_dim + company_dim + symbol_dim + 3  # +3 year feats
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    # -- featurisation -----------------------------------------------------
    def _ids(self, metas: list[dict[str, Any]], device):
        sec = torch.tensor(
            [sector_id(m.get("company_sector", ""), m.get("company_industry", "")) for m in metas],
            dtype=torch.long, device=device,
        )
        ind = torch.tensor(
            [_hash_bucket(str(m.get("company_industry", "")), self.n_industry_buckets) for m in metas],
            dtype=torch.long, device=device,
        )
        comp = torch.tensor(
            [_hash_bucket(normalize_company(m.get("company_name", "")), self.n_company_buckets) for m in metas],
            dtype=torch.long, device=device,
        )
        sym = torch.tensor(
            [_hash_bucket(str(m.get("company_symbol", "")), self.n_symbol_buckets) for m in metas],
            dtype=torch.long, device=device,
        )
        year = torch.tensor(
            [_year_features(m.get("report_year", "")) for m in metas],
            dtype=torch.float32, device=device,
        )
        return sec, ind, comp, sym, year

    def forward(self, metas: list[dict[str, Any]]) -> torch.Tensor:
        device = next(self.parameters()).device
        sec, ind, comp, sym, year = self._ids(metas, device)
        sector_v = self.sector_emb(sec)
        industry_v = self.industry_emb(ind) + self.sector_to_ind(sector_v)   # nested under sector
        company_v = self.company_emb(comp)
        symbol_v = self.symbol_emb(sym)
        x = torch.cat([sector_v, industry_v, company_v, symbol_v, year], dim=-1)
        e = self.norm(self.mlp(x))
        return F.normalize(e, p=2, dim=-1)

    @torch.no_grad()
    def encode(self, metas: list[dict[str, Any]]) -> torch.Tensor:
        self.eval()
        return self.forward(metas)


def build_entity_embedder(kind: str = "hash", **kwargs) -> nn.Module:
    """Factory: ``"hash"`` → :class:`HashMetadataEmbedder` (baseline, unchanged),
    ``"ontology"`` → :class:`OntologyMetadataEmbedder` (E1+E2)."""
    if kind == "ontology":
        return OntologyMetadataEmbedder(**kwargs)
    if kind == "hash":
        return HashMetadataEmbedder(**kwargs)
    raise ValueError(f"unknown entity embedder kind: {kind!r} (use 'hash' or 'ontology')")
