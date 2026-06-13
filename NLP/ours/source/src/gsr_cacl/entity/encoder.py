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
