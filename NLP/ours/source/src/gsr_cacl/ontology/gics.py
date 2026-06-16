"""GICS-style sector/industry ontology for the entity channel (contribution E1).

Why this exists
---------------
The original entity signal hashes ``company_sector`` / ``company_industry`` as opaque
strings, so two companies in the *same* economic sector share **no** representational
structure. The dataset, however, mixes granularities: some rows say ``"Financials"``
(a GICS *sector*) while others say ``"Semiconductors"`` or ``"Software"`` (GICS
*sub-industries* under Information Technology). Without canonicalisation the embedder
treats these as unrelated tokens.

This module maps any sector/industry string onto the **11 canonical GICS sectors** and
exposes the ``sector → industry`` hierarchy so the entity embedder can place economically
related companies near each other (useful for sector-level queries and for cross-company
generalisation). It is a pure lookup — no dependencies, deterministic, CPU-only.

Reference: MSCI/S&P Global Industry Classification Standard (GICS), 11 sectors.
"""

from __future__ import annotations

import re

# The 11 canonical GICS sectors (index 0 reserved for UNKNOWN).
GICS_SECTORS: list[str] = [
    "Unknown",
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]
SECTOR_TO_ID: dict[str, int] = {s: i for i, s in enumerate(GICS_SECTORS)}
N_SECTORS = len(GICS_SECTORS)

# Keyword → canonical sector. Order matters only for readability; matching is substring
# based on word boundaries so "semiconductors" → Information Technology, etc. Keys are
# lower-cased; multi-word keys are matched as substrings of the normalised input.
_SECTOR_KEYWORDS: list[tuple[str, str]] = [
    # Energy
    ("energy", "Energy"), ("oil", "Energy"), ("gas", "Energy"), ("petroleum", "Energy"),
    ("coal", "Energy"), ("drilling", "Energy"),
    # Materials
    ("materials", "Materials"), ("chemical", "Materials"), ("metal", "Materials"),
    ("mining", "Materials"), ("steel", "Materials"), ("paper", "Materials"),
    ("construction material", "Materials"),
    # Industrials
    ("industrials", "Industrials"), ("industrial", "Industrials"), ("aerospace", "Industrials"),
    ("defense", "Industrials"), ("machinery", "Industrials"), ("airline", "Industrials"),
    ("transportation", "Industrials"), ("logistics", "Industrials"), ("railroad", "Industrials"),
    ("capital goods", "Industrials"), ("engineering", "Industrials"),
    # Consumer Discretionary
    ("consumer discretionary", "Consumer Discretionary"), ("retail", "Consumer Discretionary"),
    ("automobile", "Consumer Discretionary"), ("auto", "Consumer Discretionary"),
    ("apparel", "Consumer Discretionary"), ("hotel", "Consumer Discretionary"),
    ("restaurant", "Consumer Discretionary"), ("leisure", "Consumer Discretionary"),
    ("media", "Communication Services"),  # GICS 2018 moved media → Comm. Services
    # Consumer Staples
    ("consumer staples", "Consumer Staples"), ("food", "Consumer Staples"),
    ("beverage", "Consumer Staples"), ("tobacco", "Consumer Staples"),
    ("household product", "Consumer Staples"), ("grocery", "Consumer Staples"),
    # Health Care
    ("health care", "Health Care"), ("healthcare", "Health Care"), ("pharmaceutical", "Health Care"),
    ("biotech", "Health Care"), ("medical", "Health Care"), ("life science", "Health Care"),
    ("hospital", "Health Care"),
    # Financials
    ("financials", "Financials"), ("financial", "Financials"), ("bank", "Financials"),
    ("insurance", "Financials"), ("capital markets", "Financials"), ("asset management", "Financials"),
    ("diversified financial", "Financials"),
    # Information Technology
    ("information technology", "Information Technology"), ("semiconductor", "Information Technology"),
    ("software", "Information Technology"), ("hardware", "Information Technology"),
    ("technology", "Information Technology"), ("it services", "Information Technology"),
    ("electronic", "Information Technology"),
    # Communication Services
    ("communication services", "Communication Services"), ("telecommunication", "Communication Services"),
    ("telecom", "Communication Services"), ("wireless", "Communication Services"),
    ("entertainment", "Communication Services"), ("interactive media", "Communication Services"),
    ("publishing", "Communication Services"),
    # Utilities
    ("utilities", "Utilities"), ("utility", "Utilities"), ("electric", "Utilities"),
    ("water", "Utilities"), ("power", "Utilities"),
    # Real Estate
    ("real estate", "Real Estate"), ("reit", "Real Estate"), ("property", "Real Estate"),
]

_WS = re.compile(r"\s+")
_NOISE = re.compile(r"[^a-z0-9 ]+")

# Precompiled per-keyword patterns (longest keyword first, matching on word boundaries).
_SECTOR_KW_COMPILED: list[tuple[re.Pattern, str, int]] = sorted(
    [(re.compile(rf"\b{re.escape(kw)}"), canon, len(kw)) for kw, canon in _SECTOR_KEYWORDS],
    key=lambda t: t[2], reverse=True,
)

# Precompiled normalised canonical sector names (for fast exact match).
_NORM_GICS_SECTORS: set[str] = set()


def _norm(text: str) -> str:
    t = str(text or "").lower().strip()
    t = t.replace("&", " and ")
    t = _NOISE.sub(" ", t)
    return _WS.sub(" ", t).strip()


# Populate normalised canonical names after _norm is defined.
_NORM_GICS_SECTORS = {_norm(s): s for s in GICS_SECTORS}


def canonical_sector(sector: str = "", industry: str = "") -> str:
    """Map (sector, industry) strings onto one of the 11 GICS sectors.

    Tries the explicit ``sector`` field first, then falls back to ``industry`` (which in
    this dataset frequently carries the more specific, classifiable token). Returns
    ``"Unknown"`` when nothing matches.
    """
    for field in (sector, industry):
        n = _norm(field)
        if not n:
            continue
        # exact canonical sector name (O(1) dict lookup instead of 12 string compares)
        if n in _NORM_GICS_SECTORS:
            return _NORM_GICS_SECTORS[n]
        # keyword / substring match using precompiled patterns (longest wins)
        for pat, canon, _ in _SECTOR_KW_COMPILED:
            if pat.search(n):
                return canon
    return "Unknown"


def sector_id(sector: str = "", industry: str = "") -> int:
    """Canonical GICS sector id in ``[0, N_SECTORS)`` (0 = Unknown)."""
    return SECTOR_TO_ID[canonical_sector(sector, industry)]


def sector_industry_path(sector: str = "", industry: str = "") -> tuple[str, str]:
    """Return ``(canonical_sector, normalised_industry)`` — the hierarchy path used as the
    structured key for the entity ontology graph/embedder."""
    return canonical_sector(sector, industry), _norm(industry) or _norm(sector)
