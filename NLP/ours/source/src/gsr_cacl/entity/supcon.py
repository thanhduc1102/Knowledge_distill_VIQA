"""Supervised Contrastive Loss for the entity-embedding stage (Khosla et al., 2020).

Positives = samples sharing the same entity label (company, or company+year). This pulls
documents/queries of the SAME company-year together and pushes different entities apart —
directly attacking the "Entity Conflation / lexical-overlap illusion" failure mode that
the EDA quantified (same-company similarity is 2.8× higher than cross-company).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_entity_labels(metas: list[dict], by: str = "company_year") -> torch.Tensor:
    """Map metadata dicts to integer labels for SupCon positives."""
    keys = []
    for m in metas:
        if by == "company":
            k = str(m.get("company_name", "")).lower().strip()
        else:  # company_year
            k = (str(m.get("company_name", "")).lower().strip() + "|" +
                 str(m.get("report_year", "")).strip())
        keys.append(k)
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    return torch.tensor([uniq[k] for k in keys], dtype=torch.long)


class SupConLoss(nn.Module):
    """Supervised contrastive loss over L2-normalized embeddings."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        labels = labels.to(device).view(-1, 1)
        B = embeddings.shape[0]

        sim = embeddings @ embeddings.t() / self.temperature
        # numerical stability
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # mask out self-comparisons
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = (labels == labels.t()) & ~self_mask

        exp_sim = torch.exp(sim) * (~self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if valid.sum() == 0:
            return embeddings.sum() * 0.0  # no positives in batch → zero (keep grad graph)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_counts[valid]
        return -mean_log_prob_pos.mean()
