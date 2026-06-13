"""Robust numeric parsing & matching for T²-RAGBench financial tables/answers.

The dataset uses several quirky encodings that the generic ``kg.parser.parse_number``
does not fully cover, e.g.::

    "-32 ( 32 )"   -> -32.0      # negative with parenthetical repeat
    "$ 5,735"      -> 5735.0
    "(1,234)"      -> -1234.0
    "12.3%"        -> 12.3       # NOTE: percentages kept in *display* units here
    "206588.0"     ->  206588.0
    "['2019']"     ->  2019.0    # list-as-string gold answers (TAT-DQA)

These helpers are shared by the Fact Ledger, the generation metrics (Number-Match)
and the deterministic verifier so all three agree on what a "number" is.
"""

from __future__ import annotations

import re
from typing import Optional

_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_NUM_RE = re.compile(r"[-+]?\$?\s?\(?\s?[-+]?\d[\d,]*\.?\d*\s?\)?%?")


def parse_financial_number(cell: str) -> Optional[float]:
    """Parse a single financial value from a messy cell/answer string.

    Returns the number in *display units* (no million/percent rescaling) so that
    a parsed table cell can be compared directly against the gold ``program_answer``,
    which is also expressed in display units.
    """
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None

    # list-as-string: "['2019']" / "['12.3']"
    if s.startswith("[") and s.endswith("]"):
        inner = s.strip("[]").replace("'", "").replace('"', "").split(",")
        if inner:
            return parse_financial_number(inner[0])

    # Drop a trailing parenthetical repeat used for negatives: "-32 ( 32 )" / "32 (32)".
    # The look-behind requires a digit before the parenthesis so a standalone "(1,234)"
    # (accounting-negative form) is NOT stripped here.
    s = re.sub(r"(?<=\d)\s*\(\s*[\d,\.]+\s*\)\s*$", "", s).strip()

    neg = False
    # Whole value wrapped in parentheses → negative accounting convention
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    pct = s.endswith("%")
    if pct:
        s = s[:-1].strip()

    # Strip currency + thousands separators + stray spaces
    s = re.sub(r"[\$€£¥,]", "", s).strip()
    s = s.replace(" ", "")

    if s.startswith("-"):
        neg = True
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]

    try:
        val = float(s)
    except ValueError:
        return None
    if neg:
        val = -val
    # percentages are returned as-is (display units); callers decide tolerance
    return val


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric literals from a free-text string (for verifier grounding)."""
    out: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        v = parse_financial_number(m.group(0))
        if v is not None:
            out.append(v)
    return out


def extract_years(text: str) -> list[int]:
    """Extract 4-digit fiscal years (1950-2049) from a string."""
    return [int(y) for y in _YEAR_RE.findall(text or "")]


def numbers_close(a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-6) -> bool:
    """Relative-tolerance match used by Number-Match (leaderboard uses rel_tol=1e-2).

    Also treats scale-of-1000 confusions leniently is *not* done here on purpose —
    Number-Match must be strict; scale handling belongs in the generator prompt.
    """
    if a is None or b is None:
        return False
    if abs(a - b) <= abs_tol:
        return True
    denom = max(abs(a), abs(b), abs_tol)
    return abs(a - b) / denom <= rel_tol


def number_match(pred: str | float | None, golds, rel_tol: float = 1e-2) -> bool:
    """True if the predicted answer numerically matches ANY gold answer.

    ``golds`` may be a single value or a list (e.g. [program_answer, original_answer]).
    Robust to: extra text around the number, $/%/commas, parenthetical negatives,
    and list-as-string golds.
    """
    if golds is None:
        return False
    if not isinstance(golds, (list, tuple)):
        golds = [golds]

    # Parse the predicted number: try whole-string first, else the last number in text
    pred_val = parse_financial_number(pred) if not isinstance(pred, float) else pred
    if pred_val is None:
        nums = extract_numbers(str(pred)) if pred is not None else []
        if not nums:
            return False
        candidates = nums  # try every number that appears in the prediction
    else:
        candidates = [pred_val]

    gold_vals = [parse_financial_number(g) for g in golds]
    gold_vals = [g for g in gold_vals if g is not None]
    if not gold_vals:
        return False

    for c in candidates:
        for g in gold_vals:
            if numbers_close(c, g, rel_tol=rel_tol):
                return True
            # tolerate millions<->units annotation drift (e.g. 3.8 vs 380, 94 vs 94)
            for factor in (100.0, 1000.0, 1e6):
                if numbers_close(c * factor, g, rel_tol=rel_tol) or numbers_close(c, g * factor, rel_tol=rel_tol):
                    return True
    return False
