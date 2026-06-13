"""Stand-alone trainer for the entity/metadata embedder (SupCon stage).

Trains :class:`HashMetadataEmbedder` so that ``cos(e_Q, e_D)`` is high iff query and
document refer to the same entity-period. Self-contained and fast (CPU-friendly).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from gsr_cacl.entity.encoder import HashMetadataEmbedder
from gsr_cacl.entity.supcon import SupConLoss, make_entity_labels

logger = logging.getLogger(__name__)


def train_entity_embedder(
    metas: list[dict[str, Any]],
    embed_dim: int = 128,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    label_by: str = "company_year",
    temperature: float = 0.1,
    device: str | None = None,
    seed: int = 42,
) -> HashMetadataEmbedder:
    """Train and return a HashMetadataEmbedder via SupCon over entity labels."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = HashMetadataEmbedder(embed_dim=embed_dim).to(device)
    loss_fn = SupConLoss(temperature=temperature)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    labels_all = make_entity_labels(metas, by=label_by)
    n = len(metas)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_metas = [metas[j] for j in idx.tolist()]
            batch_labels = labels_all[idx]
            emb = model(batch_metas)
            loss = loss_fn(emb, batch_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        logger.info(f"[entity] epoch {ep + 1}/{epochs} loss={total / max(nb, 1):.4f}")
    model.eval()
    return model
