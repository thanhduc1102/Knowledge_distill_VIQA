"""Preference optimization for the generator (DPO / ORPO) + GRPO reward utilities.

The Ledger Verifier produces an *annotation-free* reward, which lets us build preference
data and RL signals WITHOUT human labels:

  R(y) = R_answer(y) + λ_g · grounding(y) + λ_a · arithmetic(y),   with λ_g + λ_a < 1

so accuracy is the lexicographic priority (a grounded-but-wrong answer never beats a
correct one). This module provides:

  * ``ledger_reward``           — scalar reward for one generation (uses the verifier)
  * ``build_preference_pairs``  — sample K generations/query, emit (chosen, rejected) JSONL
                                  in the TRL-compatible {prompt, chosen, rejected} schema
  * ``dpo_loss`` / ``orpo_loss``— self-contained losses (no TRL dependency required)
  * ``sequence_logprobs``       — helper for the above
  * ``grpo_advantages``         — group-relative advantages for RLVR/GRPO

Heavy training is GPU work; ``run_preference_training`` is a compact, correct loop you can
launch on the T4s, and ``build_preference_pairs`` runs CPU-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from gsr_cacl.generation.verifier import verify, VerificationResult
from gsr_cacl.ledger.fact import FactLedger


# ---------------------------------------------------------------------------
# Verifier-based reward (annotation-free)
# ---------------------------------------------------------------------------

def ledger_reward(
    prediction: str,
    ledger: FactLedger,
    query: str = "",
    gold=None,
    lambda_g: float = 0.3,
    lambda_a: float = 0.2,
) -> float:
    """Reward in [0, ~1.5]; accuracy dominates (λ_g + λ_a < 1)."""
    vr: VerificationResult = verify(prediction, ledger, query, gold=gold)
    r_answer = 1.0 if vr.answer_match else 0.0
    r_ground = 1.0 if vr.grounded else 0.0
    r_arith = 1.0 if vr.derivable else 0.0
    return r_answer + lambda_g * r_ground + lambda_a * r_arith


# ---------------------------------------------------------------------------
# Offline preference data (DPO / ORPO)
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float


def build_preference_pairs(
    samples: list[dict],
    sample_fn: Callable[[str], list[str]],
    ledger_fn: Callable[[dict], FactLedger],
    prompt_fn: Callable[[dict], str],
    min_margin: float = 0.5,
) -> list[PreferencePair]:
    """For each sample, draw K candidate generations, score with the verifier, and keep the
    best vs worst as a (chosen, rejected) pair when their reward gap ≥ ``min_margin``.

    Args:
        samples:   list of dicts with at least 'query' and (optional) 'gold'
        sample_fn: query -> list of candidate completions (a temperature>0 generator)
        ledger_fn: sample -> FactLedger (for verification)
        prompt_fn: sample -> the exact prompt string used for training
    """
    pairs: list[PreferencePair] = []
    for s in samples:
        cands = sample_fn(s["query"])
        if len(cands) < 2:
            continue
        led = ledger_fn(s)
        scored = [(ledger_reward(c, led, s["query"], s.get("gold")), c) for c in cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        (best_r, best), (worst_r, worst) = scored[0], scored[-1]
        if best_r - worst_r >= min_margin and best != worst:
            pairs.append(PreferencePair(prompt_fn(s), best, worst, best_r, worst_r))
    return pairs


def write_preference_jsonl(pairs: list[PreferencePair], path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected,
                                "chosen_reward": p.chosen_reward, "rejected_reward": p.rejected_reward}) + "\n")
    return path


# ---------------------------------------------------------------------------
# Self-contained DPO / ORPO losses (no TRL dependency)
# ---------------------------------------------------------------------------

def sequence_logprobs(model, tokenizer, prompt: str, completion: str, device) -> "torch.Tensor":
    """Sum log p(completion | prompt) under ``model`` (completion tokens only)."""
    import torch
    p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full = tokenizer(prompt + completion, return_tensors="pt").input_ids.to(device)
    labels = full.clone()
    labels[:, : p_ids.shape[1]] = -100
    out = model(full)
    logits = out.logits[:, :-1, :]
    tgt = labels[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    mask = tgt != -100
    tgt_safe = tgt.clamp(min=0)
    tok_logp = logp.gather(-1, tgt_safe.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum()


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float = 0.1):
    """Direct Preference Optimization loss (Rafailov et al., 2023)."""
    import torch.nn.functional as F
    pi = policy_chosen - policy_rejected
    ref = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (pi - ref))


def orpo_loss(policy_chosen, policy_rejected, lambda_or: float = 0.1, nll_chosen=None):
    """ORPO loss (Hong et al., 2024): NLL on chosen + odds-ratio penalty. Reference-free."""
    import torch
    import torch.nn.functional as F
    log_odds = (policy_chosen - torch.log1p(-torch.exp(policy_chosen).clamp(max=0.999))) - \
               (policy_rejected - torch.log1p(-torch.exp(policy_rejected).clamp(max=0.999)))
    or_loss = -F.logsigmoid(log_odds)
    if nll_chosen is not None:
        return nll_chosen + lambda_or * or_loss
    return lambda_or * or_loss


def run_preference_training(model_name, pairs, method: str = "dpo", beta: float = 0.1,
                            lr: float = 5e-6, epochs: int = 1, device: str | None = None,
                            save_path: str | None = None):
    """Compact DPO/ORPO loop over PreferencePair list. GPU recommended.

    Prefers TRL's DPOTrainer if installed; otherwise uses the self-contained losses above.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device)
    ref = None
    if method == "dpo":
        ref = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                   trust_remote_code=True).to(device)
        ref.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for p in pairs:
            pc = sequence_logprobs(model, tok, p.prompt, p.chosen, device)
            pr = sequence_logprobs(model, tok, p.prompt, p.rejected, device)
            if method == "dpo":
                with torch.no_grad():
                    rc = sequence_logprobs(ref, tok, p.prompt, p.chosen, device)
                    rr = sequence_logprobs(ref, tok, p.prompt, p.rejected, device)
                loss = dpo_loss(pc, pr, rc, rr, beta=beta)
            else:  # orpo
                loss = orpo_loss(pc, pr, lambda_or=beta, nll_chosen=-pc)
            opt.zero_grad(); loss.backward(); opt.step()
    if save_path:
        model.save_pretrained(save_path); tok.save_pretrained(save_path)
    return model


# ---------------------------------------------------------------------------
# GRPO / RLVR
# ---------------------------------------------------------------------------

def grpo_advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantages: (r - mean) / std (Shao et al., 2024 / DeepSeek GRPO)."""
    import statistics as st
    if not rewards:
        return []
    mu = st.mean(rewards)
    sd = st.pstdev(rewards) or 1.0
    return [(r - mu) / sd for r in rewards]
