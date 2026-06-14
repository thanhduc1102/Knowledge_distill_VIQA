"""Company-name canonicalisation + fuzzy matching (contribution E2).

Why this exists
---------------
T²-RAGBench questions are reformulated to *contain* the company name in free text, while
the structured ``company_name`` metadata field may use a different surface form. The same
entity appears as e.g. ``"American Water Works"``, ``"American Water Works Company, Inc."``,
``"AWK"`` (ticker), or ``"AWCC"`` (a subsidiary acronym seen inside the table). Exact
lower-case equality (the current metadata filter) therefore *misses* the gold document
whenever the surfaces differ — a silent recall ceiling.

This module provides:
  * ``normalize_company`` — strip legal suffixes / punctuation / stopwords → canonical key
  * ``company_acronym``   — initialism of the significant words ("AWK"-style)
  * ``company_match``     — boolean match robust to suffix/word-order/acronym variation
  * ``company_match_score`` — graded [0,1] similarity for ranking

Pure-Python, deterministic, no dependencies.
"""

from __future__ import annotations

import re

# Legal/structural suffix tokens that do not identify the entity.
_LEGAL_SUFFIXES: set[str] = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "ltd", "limited", "plc", "llc", "lp", "llp", "lllp", "group", "holding", "holdings",
    "sa", "ag", "nv", "se", "spa", "kgaa", "gmbh", "pte", "bhd", "ab", "oyj", "asa",
    "the", "and", "of", "trust", "fund", "partners",
}
# Common subsidiary / financing-arm acronym suffixes seen in tables (kept for matching).
_NOISE = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def _tokens(name: str) -> list[str]:
    t = str(name or "").lower()
    t = t.replace("&", " and ")
    t = _NOISE.sub(" ", t)
    return [w for w in _WS.sub(" ", t).strip().split(" ") if w]


def significant_tokens(name: str) -> list[str]:
    """Tokens that identify the entity (legal suffixes / stopwords removed)."""
    toks = [w for w in _tokens(name) if w not in _LEGAL_SUFFIXES]
    return toks or _tokens(name)  # never return empty if there was any content


def normalize_company(name: str) -> str:
    """Canonical key for a company name: significant tokens, sorted-insensitive join."""
    return " ".join(significant_tokens(name))


def company_acronym(name: str) -> str:
    """Initialism of the significant tokens, e.g. 'American Water Works' → 'aww'."""
    toks = significant_tokens(name)
    if len(toks) < 2:
        return ""
    return "".join(w[0] for w in toks if w)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def company_match_score(a: str, b: str) -> float:
    """Graded similarity in [0, 1] between two company surface forms."""
    ca, cb = normalize_company(a), normalize_company(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    ta, tb = set(ca.split(" ")), set(cb.split(" "))
    # one is a token-subset of the other (e.g. "apple" ⊂ "apple computer")
    if ta <= tb or tb <= ta:
        return 0.9
    # acronym match (one side is a ticker / initialism of the other)
    aca, acb = company_acronym(a), company_acronym(b)
    short = cb if len(cb) <= len(ca) else ca
    long_ac = aca if short is cb else acb
    if short and long_ac and short.replace(" ", "") == long_ac:
        return 0.85
    return _jaccard(ta, tb)


def company_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """True when two surface forms refer to the same company (suffix/acronym robust)."""
    return company_match_score(a, b) >= threshold
