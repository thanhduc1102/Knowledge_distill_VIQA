"""MetadataRetriever — company+year exact-match recall booster.

Design goal: increase pool recall WITHOUT significantly increasing pool size, by
guaranteeing gold-document inclusion for queries that name a company and/or year.

Key insight from EDA:
  - BM25 top-50 misses 2.3% FinQA, 13.1% TAT-DQA gold docs.
  - Most misses: query label differs from table text BUT the doc IS from the right
    company and year (e.g. "gain on extinguishment" vs "redemption premium").
  - Adding only company+year matching docs: typically 2-8 per query → pool grows
    by ~5 docs on average, NOT the 50 extra docs that dense/lateint add.

Scoring:
  score(D) =
    1.0   company matches AND year matches   (high-precision exact hit)
    0.5   company matches, no year in query  (company-only, still useful)
    0.0   otherwise

get_candidates() returns only docs with score > 0 (ignores pool_size),
so pool expansion is bounded by company+year cardinality (2-8 typical).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.experts.base import Expert
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.aliases import normalize_company
from gsr_cacl.retrieval.self_query import CompanyIndex


class MetadataRetriever(Expert):
    """Company+year exact-match recall booster. Retriever only, not re-scorer.

    Args:
        require_year: If True (default), only add docs where BOTH company AND year match
                      in the query. Keeps pool expansion tiny (avg ~4 docs FinQA).
                      If False, also add company-only matches when query has no year
                      (useful for datasets where year is rarely in query).
        max_add: Hard cap on extra docs added per query. Prevents pool explosion for
                 companies with many docs. Default 15.
    """

    name = "meta"
    is_retriever = True

    def __init__(
        self,
        require_year: bool = True,
        max_add: int = 15,
        company_pool: bool = False,
        use_query_meta_company: bool | None = None,
    ):
        """
        Args (new):
            company_pool: If True, ``get_candidates`` returns the ENTIRE set of chunks of
                the query's company (year-matched first), capped at ``max_add``. This
                realises the "company-complete pool" (recall→1.0 on T²-RAGBench, where the
                gold always shares the company), turning retrieval into the within-filing
                disambiguation problem. Use a large ``max_add`` (e.g. 200) with this mode.
            use_query_meta_company: Read the company from ``query_meta["company_name"]``
                (normalised) instead of detecting it in the stripped question. Defaults to
                ``company_pool`` — the legitimate metadata use (the company is in the
                question by T²-RAGBench design; see docs/RESEARCH_AAAI27.md §2.2). Fixes the
                CompanyIndex.detect() miss (≈0–3% gold-company on stripped questions).
        """
        self.require_year = require_year
        self.max_add = max_add
        self.company_pool = company_pool
        self.use_query_meta_company = (
            company_pool if use_query_meta_company is None else use_query_meta_company
        )

    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        self._n = len(corpus)
        self._comp_index = CompanyIndex(list(doc_metas))
        # Per-doc canonical company key and report year
        self._doc_company_key: list[str] = []
        self._doc_year: list[int | None] = []
        for m in doc_metas:
            m = m or {}
            raw = str(m.get("company_name", "") or "").strip()
            self._doc_company_key.append(normalize_company(raw) if raw else "")
            yr_raw = m.get("report_year") or m.get("year")
            try:
                self._doc_year.append(int(float(str(yr_raw)))) if yr_raw else self._doc_year.append(None)
            except (ValueError, TypeError):
                self._doc_year.append(None)

        # Build company_key → doc_indices for O(1) lookup
        self._co_to_docs: dict[str, list[int]] = {}
        for di, ck in enumerate(self._doc_company_key):
            if ck:
                self._co_to_docs.setdefault(ck, []).append(di)

    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        self._q_company_key: list[str | None] = []
        self._q_years: list[set[int]] = []
        metas = list(query_metas) if query_metas is not None else [{}] * len(raw_queries)
        for q, m in zip(raw_queries, metas):
            if self.use_query_meta_company:
                raw = str((m or {}).get("company_name", "") or "").strip()
                self._q_company_key.append(normalize_company(raw) if raw else None)
            else:
                self._q_company_key.append(self._comp_index.detect(q))
            # Years may live in the question text and/or the query metadata.
            yrs = set(extract_years(q))
            if self.use_query_meta_company:
                ym = (m or {}).get("report_year") or (m or {}).get("year")
                if ym:
                    try:
                        yrs.add(int(float(str(ym))))
                    except (ValueError, TypeError):
                        pass
            self._q_years.append(yrs)

    def full_scores(self, qi: int) -> np.ndarray:
        q_co = self._q_company_key[qi]
        q_yrs = self._q_years[qi]
        out = np.zeros(self._n, dtype=np.float64)
        if not q_co:
            return out   # no company detected → no candidates
        # company_pool mode wants the WHOLE company set even with no year signal.
        if self.require_year and not q_yrs and not self.company_pool:
            return out   # require_year=True: skip queries with no year signal
        for di in self._co_to_docs.get(q_co, []):
            dy = self._doc_year[di]
            if q_yrs:
                # exact year match → high score; company-only (no year in doc) → low
                out[di] = 1.0 if (dy is not None and dy in q_yrs) else 0.3
            else:
                out[di] = 0.5   # company-only match (only reached if require_year=False)
        return out

    def get_candidates(self, qi: int, pool_size: int) -> list[int]:
        """Return genuinely-matching docs, capped at self.max_add.

        Year-matched docs (score=1.0) first; company-only (0.3) after.
        No zero-score noise added. Pool expansion stays bounded.
        """
        scores = self.full_scores(qi)
        matching = np.where(scores > 0.0)[0]
        if len(matching) == 0:
            return []
        ranked = matching[np.argsort(-scores[matching])]
        return ranked[: self.max_add].tolist()

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        return self.full_scores(qi)[np.asarray(pool, dtype=int)]
