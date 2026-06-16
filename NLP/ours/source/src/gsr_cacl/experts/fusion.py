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
    """Listwise InfoNCE: maximise gold's softmax prob within its candidate pool."""
    torch.manual_seed(seed)
    k = data.feats[0].shape[1]
    f = data.qfeats.shape[1] if data.qfeats is not None else 1
    model = _make(kind, k, f).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    examples = [i for i in train_idx if data.gold_pos[i] >= 0 and data.feats[i].shape[0] > 1]
    feats = {i: torch.tensor(data.feats[i], dtype=torch.float32, device=device) for i in examples}
    qf = (torch.tensor(data.qfeats, dtype=torch.float32, device=device)
          if data.qfeats is not None else None)

    model.train()
    for _ in range(epochs):
        perm = np.random.permutation(examples)
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        for i in perm:
            qfi = qf[i] if qf is not None else None
            scores = model(feats[i], qfi) / tau                 # [pool]
            loss = loss + F.cross_entropy(scores.unsqueeze(0),
                                          torch.tensor([data.gold_pos[i]], device=device))
        (loss / len(perm)).backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def rank_scores(model: nn.Module, data: FusionData, qi: int, device: str = "cpu") -> np.ndarray:
    s = torch.tensor(data.feats[qi], dtype=torch.float32, device=device)
    qf = (torch.tensor(data.qfeats[qi], dtype=torch.float32, device=device)
          if data.qfeats is not None else None)
    return model(s, qf).cpu().numpy()
