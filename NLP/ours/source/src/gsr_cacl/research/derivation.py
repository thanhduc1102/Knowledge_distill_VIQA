"""Bounded multi-operand derivability for the Fact Ledger.

The legacy verifier certifies an answer only if it is a value cell (grounded) or a *two*-operand
op (a±b, a/b, a*b). Measured consequence: the *auditable ceiling* on FinQA is only ~0.47 — most
FinQA answers are multi-operand computations (sum of 3 line items, ratio of a difference, part
over a sum-total) that two operands cannot reconstruct.

`derivation_depth` extends certification to **three** operands over the common financial patterns,
with a hard bound on the operand set so the O(n^3) search stays cheap. It returns the minimal
depth at which the value becomes reachable: "grounded" < "2op" < "3op" < None.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

from gsr_cacl.ledger.numeric import numbers_close


def _two_op(value: float, vals: list[float], rel_tol: float) -> bool:
    for a, b in combinations(vals, 2):
        if (numbers_close(value, a + b, rel_tol)
                or numbers_close(value, a - b, rel_tol) or numbers_close(value, b - a, rel_tol)
                or (b and numbers_close(value, a / b, rel_tol))
                or (a and numbers_close(value, b / a, rel_tol))
                or numbers_close(value, a * b, rel_tol)):
            return True
    return False


def _three_op(value: float, vals: list[float], rel_tol: float):
    """Common 3-operand financial forms. Returns the operand triple (a,b,c) if found, else None."""
    # Precompute pairwise combinations (a op b) once, then combine with a third cell c.
    pair_results: list[tuple[float, float, float]] = []  # (result, a, b)
    for a, b in combinations(vals, 2):
        pair_results.append((a + b, a, b))
        pair_results.append((a - b, a, b))
        pair_results.append((b - a, a, b))
        if b:
            pair_results.append((a / b, a, b))
        if a:
            pair_results.append((b / a, a, b))
    for r, a, b in pair_results:
        for c in vals:
            if c == a or c == b:
                continue
            if (numbers_close(value, r + c, rel_tol)
                    or numbers_close(value, r - c, rel_tol)
                    or numbers_close(value, c - r, rel_tol)
                    or (c and numbers_close(value, r / c, rel_tol))
                    or (r and numbers_close(value, c / r, rel_tol))):
                return (a, b, c)
    return None


def derivation_depth(value: Optional[float], vals: list[float], rel_tol: float = 1e-2,
                     max_vals: int = 16, max_ops: int = 3, return_operands: bool = False):
    """Minimal certification depth for ``value`` over ledger ``vals`` (bounded search).

    With ``return_operands`` returns ``(depth, operands)`` so the caller can type-check the path
    (e.g. CPR checks whether the 3-op operands are concept/period consistent with the query).
    """
    def _ret(depth, ops=()):
        return (depth, list(ops)) if return_operands else depth
    if value is None or not vals:
        return _ret(None)
    if any(numbers_close(value, v, rel_tol) for v in vals):
        return _ret("grounded")
    if len(vals) > max_vals:
        target = abs(value) if value else 1.0
        ranked = sorted(vals, key=lambda v: (abs(abs(v) - target), -abs(v)))
        vals = ranked[:max_vals]
    if _two_op(value, vals, rel_tol):
        return _ret("2op")
    if max_ops >= 3:
        triple = _three_op(value, vals, rel_tol)
        if triple is not None:
            return _ret("3op", triple)
    return _ret(None)
