"""Honest company self-query (Phase A).

Why
---
EDA root cause #4: same-company confusion — intra-company similarity is ~2.8x
inter-company, so "Apple 2019" and "Apple 2020" are nearly indistinguishable by
text alone. Metadata is the gold signal, but the *honest* way to use it is to read
the company **from the question**, never from the gold ``company_name`` field.

The catch the user raised: ~20% of FinQA and ~36% of TAT-DQA questions do not name
a company at all. So detection must (a) be robust to surface variation (ticker /
acronym / legal-suffix drift, via ``ontology.aliases``) and (b) cleanly return
``None`` when the question carries no company — in which case the channel adds no
boost and cannot hurt those queries.

This uses only the QUESTION text and the set of company names already present in the
corpus (known at index time). No gold per-query metadata enters the ranking.
"""

from __future__ import annotations

import re
from typing import Optional

from gsr_cacl.ontology.aliases import (
    company_acronym,
    normalize_company,
    significant_tokens,
)

_TOK = re.compile(r"[a-z0-9]+")


def _qtokens(text: str) -> set[str]:
    return set(_TOK.findall((text or "").lower()))


class CompanyIndex:
    """Maps normalised company keys → corpus doc indices, with alias-robust lookup."""

    def __init__(self, doc_metas: list[dict]):
        self.key_to_docs: dict[str, list[int]] = {}
        # company descriptor: (key, significant-token set, acronym)
        self._companies: list[tuple[str, set[str], str]] = []
        seen: set[str] = set()
        for i, m in enumerate(doc_metas):
            name = str((m or {}).get("company_name", "")).strip()
            if not name:
                continue
            key = normalize_company(name)
            if not key:
                continue
            self.key_to_docs.setdefault(key, []).append(i)
            if key not in seen:
                seen.add(key)
                self._companies.append((key, set(significant_tokens(name)), company_acronym(name)))

    def detect(self, question: str) -> Optional[str]:
        """Best company key mentioned in the question, or ``None`` if none is.

        Matching rules (most specific wins, then longest):
          * multi-word company whose significant tokens are ALL in the question;
          * single-word company present as an exact token (len ≥ 4, avoids noise);
          * acronym/ticker present as an exact token (len ≥ 2).
        """
        qtok = _qtokens(question)
        if not qtok:
            return None
        best_key: Optional[str] = None
        best_rank: tuple = (-1.0, -1)  # (specificity, token-count)
        for key, sig, acro in self._companies:
            if not sig:
                continue
            spec = 0.0
            if len(sig) >= 2 and sig <= qtok:
                spec = 1.0
            elif len(sig) == 1:
                tok = next(iter(sig))
                if len(tok) >= 4 and tok in qtok:
                    spec = 0.7
            if spec == 0.0 and acro and len(acro) >= 2 and acro in qtok:
                spec = 0.85
            if spec > 0.0:
                rank = (spec, len(sig))
                if rank > best_rank:
                    best_rank, best_key = rank, key
        return best_key

    def docs_for(self, key: Optional[str]) -> set[int]:
        if key is None:
            return set()
        return set(self.key_to_docs.get(key, ()))
