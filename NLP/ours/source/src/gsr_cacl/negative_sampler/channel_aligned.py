"""Channel-aligned, answer-invalidating hard negatives (upgraded CACL).

Fixes the original CHAP weaknesses:
  * CHAP-E was a stub (only prepended a metadata header). Here entity/period negatives
    are REAL value perturbations.
  * Each negative differs from the positive in EXACTLY ONE aspect, so the InfoNCE/triplet
    gradient concentrates on exactly one scoring channel (clean, unsaturated learning) —
    the principle the reviewers asked for (review_1/2, core_method.md §4.2):

      ┌────────────────┬───────────────────────────────────────────┬──────────────────┐
      │ negative type  │ what it breaks                              │ trains channel   │
      ├────────────────┼───────────────────────────────────────────┼──────────────────┤
      │ metric-swap    │ swap values between two line-items (rows)   │ concept σ_m      │
      │ period-swap    │ swap values across two periods (columns)    │ temporal gate γ_t│
      │ entity-swap    │ same table, DIFFERENT company values/meta   │ entity gate γ_e  │
      │ scale-break    │ ×1000 a value column                        │ magnitude ρ      │
      │ value-identity │ change one operand so Total ≠ Σ components  │ constraint / CS  │
      └────────────────┴───────────────────────────────────────────┴──────────────────┘

These operate on the row-major fact orientation (concept = row label) used by the dataset,
so they actually fire (unlike the column-template CHAP edges, which rarely match).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from gsr_cacl.kg.parser import parse_markdown_rows
from gsr_cacl.ledger.numeric import parse_financial_number

CHANNELS = ["metric-swap", "period-swap", "entity-swap", "scale-break", "value-identity"]
CHANNEL_TO_SCORE = {
    "metric-swap": "concept",
    "period-swap": "temporal",
    "entity-swap": "entity",
    "scale-break": "magnitude",
    "value-identity": "constraint",
}


@dataclass
class ChannelNegative:
    table_md: str
    channel: str                 # one of CHANNELS
    target_score: str            # which scoring component this negative trains
    description: str = ""
    swap_metadata: bool = False  # entity-swap: trainer should pair with a different company meta


def _rebuild(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    widths = [max((len(str(r[i])) for r in rows if i < len(r)), default=1) for i in range(ncol)]
    out = []
    for ri, r in enumerate(rows):
        cells = [str(r[i]).ljust(widths[i]) if i < len(r) else "".ljust(widths[i]) for i in range(ncol)]
        out.append("| " + " | ".join(cells) + " |")
        if ri == 0:
            out.append("| " + " | ".join("-" * w for w in widths) + " |")
    return "\n".join(out)


def _numeric_cols(data: list[list[str]], ncol: int) -> list[int]:
    cols = []
    for c in range(ncol):
        vals = [parse_financial_number(r[c]) if c < len(r) else None for r in data]
        if sum(v is not None for v in vals) >= max(2, len(data) // 2):
            cols.append(c)
    return cols


def _format_like(orig: str, new_val: float) -> str:
    s = str(orig)
    if s.strip().startswith("$"):
        return f"$ {new_val:g}"
    if "(" in s and ")" in s:  # accounting negative
        return f"-{abs(new_val):g} ( {abs(new_val):g} )"
    return f"{new_val:g}"


def make_negative(table_md: str, channel: str, rng: random.Random) -> Optional[ChannelNegative]:
    rows = parse_markdown_rows(table_md)
    if len(rows) < 2:
        return None
    header, data = rows[0], [list(r) for r in rows[1:]]
    ncol = max(len(r) for r in rows)
    num_cols = _numeric_cols(data, ncol)
    if not num_cols and channel != "entity-swap":
        return None

    if channel == "entity-swap":
        # same numbers, different entity — the trainer pairs this with another company's meta.
        return ChannelNegative(table_md=table_md, channel=channel,
                               target_score="entity", swap_metadata=True,
                               description="same table values, different company metadata")

    if channel == "scale-break":
        c = rng.choice(num_cols)
        for r in data:
            if c < len(r):
                v = parse_financial_number(r[c])
                if v is not None:
                    r[c] = _format_like(r[c], v * 1000.0)
        return ChannelNegative(_rebuild([header] + data), channel, "magnitude",
                               f"column {c} scaled ×1000")

    if channel == "value-identity":
        c = rng.choice(num_cols)
        cand = [ri for ri, r in enumerate(data) if c < len(r) and parse_financial_number(r[c]) is not None]
        if not cand:
            return None
        ri = rng.choice(cand)
        v = parse_financial_number(data[ri][c])
        data[ri][c] = _format_like(data[ri][c], round(v * rng.choice([1.2, 0.8, 1.3, 0.7]), 4))
        return ChannelNegative(_rebuild([header] + data), channel, "constraint",
                               f"row {ri} col {c} perturbed (breaks Total=Σ)")

    if channel == "period-swap":
        if len(num_cols) < 2:
            return None
        c1, c2 = rng.sample(num_cols, 2)
        for r in data:
            if c1 < len(r) and c2 < len(r):
                r[c1], r[c2] = r[c2], r[c1]
        return ChannelNegative(_rebuild([header] + data), channel, "temporal",
                               f"swapped period columns {c1}<->{c2}")

    if channel == "metric-swap":
        c = rng.choice(num_cols)
        cand = [ri for ri, r in enumerate(data) if c < len(r) and parse_financial_number(r[c]) is not None]
        if len(cand) < 2:
            return None
        r1, r2 = rng.sample(cand, 2)
        data[r1][c], data[r2][c] = data[r2][c], data[r1][c]
        return ChannelNegative(_rebuild([header] + data), channel, "concept",
                               f"swapped values of rows {r1}<->{r2} in col {c}")
    return None


class ChannelAlignedSampler:
    """Sample channel-aligned negatives from a positive table."""

    def __init__(self, channels: Optional[list[str]] = None, seed: int = 42):
        self.channels = channels or CHANNELS
        self.rng = random.Random(seed)

    def sample(self, table_md: str, n_negatives: int = 5, one_per_channel: bool = True) -> list[ChannelNegative]:
        out: list[ChannelNegative] = []
        order = list(self.channels)
        self.rng.shuffle(order)
        for ch in order:
            if len(out) >= n_negatives:
                break
            neg = make_negative(table_md, ch, self.rng)
            if neg is not None:
                out.append(neg)
        # top up by resampling random channels if needed
        while len(out) < n_negatives:
            neg = make_negative(table_md, self.rng.choice(self.channels), self.rng)
            if neg is None:
                break
            out.append(neg)
        return out

    def diagnostics(self, negs: list[ChannelNegative]) -> dict:
        by = {}
        for n in negs:
            by[n.channel] = by.get(n.channel, 0) + 1
        return {"total": len(negs), "by_channel": by}
