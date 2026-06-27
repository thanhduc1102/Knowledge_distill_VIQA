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

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional
import re

from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import parse_financial_number, extract_numbers, numbers_close


@dataclass
class StepVerification:
    """One deterministic check over a generated reasoning step."""
    text: str
    numbers: list[float] = field(default_factory=list)
    grounded: int = 0
    total_numbers: int = 0
    arithmetic_ok: Optional[bool] = None
    explanation: str = ""


@dataclass
class VerificationResult:
    pred_value: Optional[float]
    grounded: bool          # answer equals a ledger cell
    derivable: bool         # answer equals a simple op over two ledger cells
    answer_match: bool      # answer matches the provided gold (eval only)
    reward: float           # in [0, 1]
    explanation: str = ""
    grounding_fraction: float = 0.0
    arithmetic_fraction: float = 0.0
    normalization: str = "identity"
    step_checks: list[StepVerification] = field(default_factory=list)


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


_PERCENT_QUERY = re.compile(r"\b(percent|percentage|rate|ratio|margin|proportion|share|fraction)\b|%", re.I)


def _support_variants(value: float, query: str = "") -> list[tuple[float, str]]:
    """Candidate display-unit normalizations for annotation-free support checks.

    Number-Match in this project tolerates common finance answer-scale drift
    (percent display units and thousands/millions). The verifier must be at least
    aware of those variants; otherwise it will re-ask answers that evaluation already
    treats as correct. The returned label is kept in the explanation so scaled support
    can be calibrated lower than exact cell support downstream.
    """
    if value is None:
        return []

    out: list[tuple[float, str]] = [(value, "identity")]
    seen = {round(float(value), 12)}

    def add(v: float, label: str) -> None:
        key = round(float(v), 12)
        if key not in seen:
            seen.add(key)
            out.append((v, label))

    if _PERCENT_QUERY.search(query or ""):
        add(value / 100.0, "percent_display_to_decimal")
        add(value * 100.0, "percent_decimal_to_display")

    # Table units often differ from answer units: values may be in thousands or
    # millions while the answer is requested as raw units, or vice versa.
    for factor in (1000.0, 1_000_000.0):
        add(value / factor, f"scale_down_{factor:g}")
        add(value * factor, f"scale_up_{factor:g}")

    return out


def _grounding_support(value: float, ledger: FactLedger, query: str,
                       rel_tol: float) -> tuple[Optional[str], str]:
    for v, label in _support_variants(value, query):
        prov = is_grounded(v, ledger, rel_tol)
        if prov is not None:
            return prov, label
    return None, "identity"


def _derivation_support(value: float, ledger: FactLedger, query: str,
                        rel_tol: float) -> tuple[Optional[str], str]:
    for v, label in _support_variants(value, query):
        exp = is_derivable(v, ledger, rel_tol)
        if exp is not None:
            return exp, label
    return None, "identity"


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


def _is_year_like(value: float) -> bool:
    return value == int(value) and 1900 <= int(value) <= 2049


def _split_steps(prediction: str) -> list[str]:
    """Small, model-agnostic segmentation for process checks."""
    if prediction is None:
        return []
    rough = []
    for line in str(prediction).replace("=>", "=").splitlines():
        rough.extend(part.strip() for part in line.split(";"))
    return [s for s in rough if s]


def _arith_matches(step: str, nums: list[float], rel_tol: float) -> tuple[Optional[bool], str]:
    """Check common financial arithmetic forms in one generated step.

    This intentionally stays deterministic and conservative. It verifies only statements
    with an explicit equality marker, e.g. ``1000 - 900 = 100`` or
    ``(1000 - 900) / abs(900) = 0.111``. Unrecognised prose returns ``None``.
    """
    if "=" not in step or len(nums) < 3:
        return None, ""

    lhs_txt = step.split("=", 1)[0].lower()
    target = nums[-1]
    args = nums[:-1]
    # The generic number regex treats the subtraction operator in "1000 - 600" as the
    # sign of the second literal. For expression checking, the operator already carries
    # that sign, so normalise the second operand back to its magnitude.
    if "-" in lhs_txt and len(args) >= 2 and args[1] < 0:
        args = [args[0], abs(args[1]), *args[2:]]

    if "sum" in lhs_txt or "+" in lhs_txt:
        expected = sum(args)
        return numbers_close(target, expected, rel_tol), f"sum={expected}"

    if "/" in lhs_txt and "-" in lhs_txt and len(args) >= 2:
        new, old = args[0], args[1]
        if old == 0:
            return False, "division by zero"
        expected = (new - old) / abs(old)
        expected_pct = expected * 100.0
        ok = numbers_close(target, expected, rel_tol) or numbers_close(target, expected_pct, rel_tol)
        return ok, f"percent_change={expected}"

    if "/" in lhs_txt and len(args) >= 2:
        a, b = args[0], args[1]
        if b == 0:
            return False, "division by zero"
        expected = a / b
        return numbers_close(target, expected, rel_tol), f"ratio={expected}"

    if "*" in lhs_txt and len(args) >= 2:
        expected = args[0] * args[1]
        return numbers_close(target, expected, rel_tol), f"product={expected}"

    if "-" in lhs_txt and len(args) >= 2:
        expected = args[0] - args[1]
        return numbers_close(target, expected, rel_tol), f"difference={expected}"

    return None, ""


def verify_process(prediction: str, ledger: FactLedger, rel_tol: float = 1e-2) -> list[StepVerification]:
    """Verify generated intermediate steps against the ledger.

    Returns per-step grounding and arithmetic checks. Period years are ignored as values,
    because ``2019`` is usually a coordinate rather than an answer operand.
    """
    checks: list[StepVerification] = []
    for step in _split_steps(prediction):
        nums = [n for n in extract_numbers(step) if not _is_year_like(n)]
        if not nums and "=" not in step:
            continue
        grounded = sum(
            1 for n in nums
            if is_grounded(n, ledger, rel_tol) is not None
            or (n < 0 and is_grounded(abs(n), ledger, rel_tol) is not None)
        )
        ar_ok, ar_exp = _arith_matches(step, nums, rel_tol)
        checks.append(StepVerification(
            text=step,
            numbers=nums,
            grounded=grounded,
            total_numbers=len(nums),
            arithmetic_ok=ar_ok,
            explanation=ar_exp,
        ))
    return checks


def verify(
    prediction: str,
    ledger: FactLedger,
    query: str = "",
    gold=None,
    rel_tol: float = 1e-2,
) -> VerificationResult:
    """Verify a prediction against the ledger (+ optional gold for eval)."""
    pred_value = extract_final_number(prediction)
    step_checks = verify_process(prediction, ledger, rel_tol)

    grounded_prov = None
    normalization = "identity"
    if pred_value is not None:
        grounded_prov, normalization = _grounding_support(pred_value, ledger, query, rel_tol)
    derivable_exp = None
    if pred_value is not None and grounded_prov is None:
        derivable_exp, normalization = _derivation_support(pred_value, ledger, query, rel_tol)

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

    total_step_nums = sum(s.total_numbers for s in step_checks)
    grounded_step_nums = sum(s.grounded for s in step_checks)
    grounding_fraction = grounded_step_nums / total_step_nums if total_step_nums else (
        1.0 if grounded_prov is not None else 0.0
    )
    arith_steps = [s for s in step_checks if s.arithmetic_ok is not None]
    arithmetic_fraction = (
        sum(1 for s in arith_steps if s.arithmetic_ok) / len(arith_steps)
        if arith_steps else (1.0 if derivable_exp is not None else 0.0)
    )

    parts = []
    if grounded_prov:
        parts.append(f"grounded@{grounded_prov}")
    if derivable_exp:
        parts.append(derivable_exp)
    if normalization != "identity":
        parts.append(f"normalization={normalization}")
    if gold is not None:
        parts.append(f"gold_match={answer_match}")
    if step_checks:
        parts.append(
            f"process_grounding={grounding_fraction:.2f}; "
            f"process_arithmetic={arithmetic_fraction:.2f}"
        )

    return VerificationResult(
        pred_value=pred_value,
        grounded=grounded_prov is not None,
        derivable=derivable_exp is not None,
        answer_match=answer_match,
        reward=reward,
        explanation="; ".join(parts),
        grounding_fraction=grounding_fraction,
        arithmetic_fraction=arithmetic_fraction,
        normalization=normalization,
        step_checks=step_checks,
    )
