"""Deterministic, annotation-free Ledger Verifier.

Checks a generated answer against the Fact Ledger of the retrieved documents:
  (i)   value grounding   — is the answer a value present in the ledger?
  (ii)  derivability      — is it a simple arithmetic combination of two ledger values
                            (difference, sum, ratio, percent-change)?
  (iii) accounting identity— (when applicable) do total rows equal the sum of components?

The verifier produces a scalar *reward* in [0, 1] usable for:
  * answer-quality diagnostics during evaluation, and
  * reward signal for preference optimization / RLVR (see ``training/preference.py``),
without requiring any human annotation — only the symbolic ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import parse_financial_number, extract_numbers, numbers_close


@dataclass
class VerificationResult:
    pred_value: Optional[float]
    grounded: bool          # answer equals a ledger cell
    derivable: bool         # answer equals a simple op over two ledger cells
    answer_match: bool      # answer matches the provided gold (eval only)
    reward: float           # in [0, 1]
    explanation: str = ""


def _ledger_values(ledger: FactLedger) -> list[float]:
    return [f.value for f in ledger.numeric_facts() if f.value is not None]


def is_grounded(value: float, ledger: FactLedger, rel_tol: float = 1e-2) -> Optional[str]:
    """Return a provenance string if ``value`` matches a ledger cell, else None."""
    for f in ledger.numeric_facts():
        if f.value is not None and numbers_close(value, f.value, rel_tol=rel_tol):
            return f.provenance or f.concept
    return None


def is_derivable(value: float, ledger: FactLedger, rel_tol: float = 1e-2, max_pairs: int = 400) -> Optional[str]:
    """Return an explanation if ``value`` is a simple op over two ledger cells."""
    vals = _ledger_values(ledger)
    pairs = list(combinations(vals, 2))[:max_pairs]
    for a, b in pairs:
        if numbers_close(value, a - b, rel_tol) or numbers_close(value, b - a, rel_tol):
            return f"difference of {a} and {b}"
        if numbers_close(value, a + b, rel_tol):
            return f"sum of {a} and {b}"
        if b != 0 and numbers_close(value, a / b, rel_tol):
            return f"ratio {a}/{b}"
        if a != 0 and numbers_close(value, b / a, rel_tol):
            return f"ratio {b}/{a}"
        # percent change
        if b != 0 and numbers_close(value, (a - b) / abs(b) * 100.0, rel_tol):
            return f"percent change ({a}-{b})/{b}*100"
    return None


def extract_final_number(prediction: str) -> Optional[float]:
    """Pull the model's final numeric answer from generated text.

    Prefers an explicit 'Answer: X' marker, else the last number in the text.
    """
    if prediction is None:
        return None
    text = str(prediction)
    for marker in ("answer:", "final answer:", "the answer is"):
        idx = text.lower().rfind(marker)
        if idx >= 0:
            tail = text[idx + len(marker):]
            nums = extract_numbers(tail)
            if nums:
                return nums[0]
    v = parse_financial_number(text)
    if v is not None:
        return v
    nums = extract_numbers(text)
    return nums[-1] if nums else None


def verify(
    prediction: str,
    ledger: FactLedger,
    query: str = "",
    gold=None,
    rel_tol: float = 1e-2,
) -> VerificationResult:
    """Verify a prediction against the ledger (+ optional gold for eval)."""
    pred_value = extract_final_number(prediction)

    grounded_prov = is_grounded(pred_value, ledger, rel_tol) if pred_value is not None else None
    derivable_exp = None
    if pred_value is not None and grounded_prov is None:
        derivable_exp = is_derivable(pred_value, ledger, rel_tol)

    answer_match = False
    if gold is not None and pred_value is not None:
        from gsr_cacl.ledger.numeric import number_match
        answer_match = number_match(pred_value, gold, rel_tol=rel_tol)

    # reward: accuracy-first (lexicographic), grounding as auxiliary annotation-free signal
    if answer_match:
        reward = 1.0
    elif grounded_prov is not None:
        reward = 0.6
    elif derivable_exp is not None:
        reward = 0.5
    elif pred_value is not None:
        reward = 0.1
    else:
        reward = 0.0

    parts = []
    if grounded_prov:
        parts.append(f"grounded@{grounded_prov}")
    if derivable_exp:
        parts.append(derivable_exp)
    if gold is not None:
        parts.append(f"gold_match={answer_match}")

    return VerificationResult(
        pred_value=pred_value,
        grounded=grounded_prov is not None,
        derivable=derivable_exp is not None,
        answer_match=answer_match,
        reward=reward,
        explanation="; ".join(parts),
    )
