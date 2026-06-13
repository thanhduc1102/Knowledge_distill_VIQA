"""Generation metrics — Number-Match (the T²-RAGBench leaderboard metric)."""

from __future__ import annotations

from gsr_cacl.ledger.numeric import number_match


def compute_number_match(predictions, golds, rel_tol: float = 1e-2) -> dict:
    """Number-Match accuracy over a list of predictions vs golds.

    Args:
        predictions: list of generated answer strings/numbers
        golds:       list of gold answers; each may be a value or a list of acceptable values
    Returns dict with overall accuracy + counts.
    """
    assert len(predictions) == len(golds), "predictions/golds length mismatch"
    correct = sum(1 for p, g in zip(predictions, golds) if number_match(p, g, rel_tol=rel_tol))
    n = len(predictions)
    return {
        "number_match": correct / n if n else 0.0,
        "correct": correct,
        "total": n,
    }
