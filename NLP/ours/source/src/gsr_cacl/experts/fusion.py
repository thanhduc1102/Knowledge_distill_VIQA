"""Learned fusion heads for MMER + listwise InfoNCE training.

Three independent combiners over the per-query expert-score matrix ``F[qi] ∈ R^{pool×k}``
(each column min-max normalised per query). All are trained the same way (listwise InfoNCE
over the candidate pool — gold vs in-pool distractors, the natural hard negatives), so they
are directly comparable:

  * ``LinearFusion`` — ``s = Σ softplus(w_i)·s_i``. The generation-1 ``JointScorer`` α/β/γ
    generalised to k experts (global weights).
  * ``MLPFusion`` — ``s = MLP([s_1..s_k])`` shared across candidates. The "simple MLP that
    combines criteria" — nonlinear interactions among experts (e.g. concept AND period).
  * ``GateFusion`` — ``s = Σ softmax(MLP(φ(Q)))_i · s_i``. Query-conditioned mixture-of-experts:
    the weights depend on per-query discriminativeness features φ(Q), *learning* the manual
    discriminative-gating rule from Phase A instead of hand-setting it.

Pure PyTorch; trains in seconds on CPU (inputs are tiny score matrices, not text).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionData:
    feats: list[np.ndarray]          # per query: [pool, k] min-max-normalised expert scores
    gold_pos: list[int]              # per query: gold index within pool, or -1 if absent
    qfeats: Optional[np.ndarray]     # [Q, f] query discriminativeness features (for GateFusion)
    expert_names: list[str]


class LinearFusion(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(k))     # softplus(0)=0.69 → all experts on

    def forward(self, s: torch.Tensor, qf: torch.Tensor | None = None) -> torch.Tensor:
        return s @ F.softplus(self.w)             # [pool]

    def weights(self) -> np.ndarray:
        return F.softplus(self.w).detach().cpu().numpy()


class MLPFusion(nn.Module):
    def __init__(self, k: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, s: torch.Tensor, qf: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(s).squeeze(-1)            # [pool]


class GateFusion(nn.Module):
    """Query-conditioned mixture: weights = softmax(MLP(query features))."""

    def __init__(self, k: int, f: int, hidden: int = 16):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(f, hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, s: torch.Tensor, qf: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.gate(qf), dim=-1)  # [k]
        return s @ w                              # [pool]


def _make(kind: str, k: int, f: int) -> nn.Module:
    if kind == "linear":
        return LinearFusion(k)
    if kind == "mlp":
        return MLPFusion(k)
    if kind == "gate":
        return GateFusion(k, f)
    raise ValueError(kind)


def train_fusion(kind: str, data: FusionData, train_idx: Sequence[int],
                 epochs: int = 200, lr: float = 0.05, tau: float = 0.1,
                 device: str = "cpu", seed: int = 0) -> nn.Module:
    """Listwise InfoNCE: maximise gold's softmax prob within its candidate pool.

    Vectorized: pools are padded to max_pool and processed in one batched forward
    pass per epoch.  Padding positions are masked with -inf before cross-entropy,
    so they cannot contribute to the softmax denominator.  ~10-50× faster than the
    prior per-query Python loop, with identical gradients.
    """
    torch.manual_seed(seed)
    k = data.feats[0].shape[1]
    f = data.qfeats.shape[1] if data.qfeats is not None else 1
    model = _make(kind, k, f).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    examples = [i for i in train_idx if data.gold_pos[i] >= 0 and data.feats[i].shape[0] > 1]
    N = len(examples)
    max_pool = max(data.feats[i].shape[0] for i in examples)

    # Pre-build padded feature tensor [N, max_pool, k] once; reuse each epoch.
    padded = torch.zeros(N, max_pool, k, dtype=torch.float32, device=device)
    pad_mask = torch.ones(N, max_pool, dtype=torch.bool, device=device)   # True = padding
    gold_t = torch.zeros(N, dtype=torch.long, device=device)
    for j, i in enumerate(examples):
        p = data.feats[i].shape[0]
        padded[j, :p] = torch.tensor(data.feats[i], dtype=torch.float32)
        pad_mask[j, :p] = False
        gold_t[j] = data.gold_pos[i]

    qf_t: Optional[torch.Tensor] = None
    if data.qfeats is not None:
        qf_t = torch.tensor(
            data.qfeats[np.asarray(examples)], dtype=torch.float32, device=device
        )  # [N, f]

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        opt.zero_grad()

        # Batched forward: compute scores for all queries at once.
        #   LinearFusion:  padded @ w → [N, max_pool]
        #   MLPFusion:     mlp applied per row → [N, max_pool]
        #   GateFusion:    per-query gate weights → [N, max_pool]
        s_perm = padded[perm]          # [N, max_pool, k]
        qf_perm = qf_t[perm] if qf_t is not None else None   # [N, f] or None

        if isinstance(model, LinearFusion):
            w = F.softplus(model.w)     # [k]
            scores = s_perm @ w         # [N, max_pool]
        elif isinstance(model, MLPFusion):
            # apply MLP row-wise: reshape to [N*max_pool, k] → [N*max_pool, 1] → [N, max_pool]
            scores = model.net(s_perm.view(N * max_pool, k)).view(N, max_pool)
        elif isinstance(model, GateFusion):
            w_q = torch.softmax(model.gate(qf_perm), dim=-1)  # [N, k]
            scores = (s_perm * w_q.unsqueeze(1)).sum(-1)       # [N, max_pool]
        else:
            # fallback: per-query loop (for custom subclasses)
            scores = torch.stack([model(s_perm[j], qf_perm[j] if qf_perm is not None else None)
                                   for j in range(N)])

        scores = scores / tau
        scores.masked_fill_(pad_mask[perm], float("-inf"))
        loss = F.cross_entropy(scores, gold_t[perm])
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def rank_scores(model: nn.Module, data: FusionData, qi: int, device: str = "cpu") -> np.ndarray:
    s = torch.tensor(data.feats[qi], dtype=torch.float32, device=device)
    qf = (torch.tensor(data.qfeats[qi], dtype=torch.float32, device=device)
          if data.qfeats is not None else None)
    return model(s, qf).cpu().numpy()
