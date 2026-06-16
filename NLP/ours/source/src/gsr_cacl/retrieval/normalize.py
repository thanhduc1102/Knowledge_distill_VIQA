"""Abbreviation / full-form unification for the BM25 backbone (Phase A).

Why
---
EDA root cause #5: query and context use *different surface forms* of the same
financial concept — query "GAAP" vs context "generally accepted accounting
principles"; TAT-DQA alone has 1,514 unique abbreviations while FinQA/ConvFinQA
contexts are full-form-normalised. BM25 needs literal token overlap, so these
forms simply do not match and the gold document is missed.

Fix (honest, training-free)
---------------------------
For every known concept we attach a single **sentinel token** (e.g. ``cct_ebitda``)
to any text in which *either* the abbreviation *or* the full form appears. Because
the sentinel is identical regardless of surface form, BM25 now matches "EBITDA" in
the query against "earnings before interest, taxes, ..." in the document. This is a
symmetric normalisation applied to BOTH queries and documents at tokenisation time —
no gold metadata, no learning.

The sentinels are *added* to the token stream (never replace the original tokens),
so they can only help recall and never remove an existing lexical match.
"""

from __future__ import annotations

import re

# (sentinel, abbreviation surface(s), full-form regex)
# Full-form regex is matched case-insensitively on the lower-cased text. Abbreviation
# surfaces are matched as whole tokens (word-boundary), tolerant of '&', '/', '.'.
_CONCEPTS: list[tuple[str, list[str], str]] = [
    ("cct_ebitda", ["ebitda"], r"earnings before interest,? taxes,? depreciation,? and amortization"),
    ("cct_ebit", ["ebit"], r"earnings before interest and taxes"),
    ("cct_eps", ["eps"], r"earnings per share"),
    ("cct_roe", ["roe"], r"return on (?:average )?equity"),
    ("cct_roa", ["roa"], r"return on assets"),
    ("cct_roi", ["roi"], r"return on investment"),
    ("cct_roic", ["roic"], r"return on invested capital"),
    ("cct_gaap", ["gaap"], r"generally accepted accounting principles"),
    ("cct_ifrs", ["ifrs"], r"international financial reporting standards"),
    ("cct_sga", ["sg&a", "sga", "sg a"], r"selling,? general,? and administrative"),
    ("cct_rnd", ["r&d", "r d", "rnd"], r"research and development"),
    ("cct_ppe", ["pp&e", "ppe", "pp e"], r"property,? plant,? and equipment"),
    ("cct_cogs", ["cogs"], r"cost of goods sold|cost of (?:sales|revenue)|cost of products sold"),
    ("cct_capex", ["capex"], r"capital expenditures?"),
    ("cct_oci", ["oci"], r"other comprehensive income"),
    ("cct_aoci", ["aoci"], r"accumulated other comprehensive income"),
    ("cct_nol", ["nol"], r"net operating loss(?:es)?"),
    ("cct_fcf", ["fcf"], r"free cash flow"),
    ("cct_ocf", ["ocf"], r"(?:net cash (?:provided by|from)|cash (?:provided by|from)) operating activities"),
    ("cct_da", ["d&a", "d a"], r"depreciation and amortization"),
    ("cct_wacc", ["wacc"], r"weighted average cost of capital"),
    ("cct_nav", ["nav"], r"net asset value"),
    ("cct_dcf", ["dcf"], r"discounted cash flow"),
    ("cct_npv", ["npv"], r"net present value"),
    ("cct_ltd", ["ltd debt"], r"long[- ]term (?:debt|borrowings?)"),
    ("cct_ar", ["a/r"], r"accounts receivable|trade receivables"),
    ("cct_ap", ["a/p"], r"accounts payable"),
    ("cct_fy", ["fy"], r"fiscal year"),
    ("cct_yoy", ["yoy", "y/y"], r"year[- ]over[- ]year"),
]

# Pre-compile full-form regexes once.
_FULLFORM_RE: list[tuple[str, re.Pattern]] = [
    (sent, re.compile(full, re.IGNORECASE)) for sent, _abbr, full in _CONCEPTS
]
# Abbreviation surface set → sentinel. Surfaces are normalised to a compact key
# (strip spaces/&/.// and lower-case) so the messy table forms collapse together.
_ABBR_KEY: dict[str, str] = {}
for _sent, _abbrs, _full in _CONCEPTS:
    for _a in _abbrs:
        _ABBR_KEY[re.sub(r"[^a-z0-9]", "", _a.lower())] = _sent

_TOK = re.compile(r"[a-z0-9]+")
# Keep '&', '/', '.' attached so "r&d", "sg&a", "a/r", "u.s." survive as one chunk
# for abbreviation detection before they are split for plain BM25 tokens.
_CHUNK = re.compile(r"[a-z0-9][a-z0-9&/.]*")


def concept_sentinels(text: str) -> list[str]:
    """Sentinel tokens for every concept whose abbreviation OR full form appears.

    Symmetric: the same sentinel is produced for "EBITDA" and for the spelled-out
    "earnings before interest, taxes, depreciation and amortization".
    """
    if not text:
        return []
    low = text.lower()
    out: set[str] = set()
    # Full-form hits.
    for sent, rgx in _FULLFORM_RE:
        if rgx.search(low):
            out.add(sent)
    # Abbreviation hits (whole chunks, '&'/'/'/'.' preserved then normalised).
    for chunk in _CHUNK.findall(low):
        key = re.sub(r"[^a-z0-9]", "", chunk)
        sent = _ABBR_KEY.get(key)
        if sent:
            out.add(sent)
    return sorted(out)


def expand_bm25_tokens(base_tokens: list[str], text: str) -> list[str]:
    """Append concept sentinels to an existing BM25 token list (never removes)."""
    return base_tokens + concept_sentinels(text)
