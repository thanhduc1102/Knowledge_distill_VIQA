"""Financial ontology: GICS sector hierarchy + company-name canonicalisation.

These two ontologies ground the entity retrieval channel in economic structure rather
than opaque string hashing (contributions E1/E2). They are pure-Python and dependency
free so they can be imported anywhere (encoder, retrieval filter, tests).
"""

from gsr_cacl.ontology.gics import (
    GICS_SECTORS,
    N_SECTORS,
    canonical_sector,
    sector_id,
    sector_industry_path,
)
from gsr_cacl.ontology.aliases import (
    normalize_company,
    company_acronym,
    company_match,
    company_match_score,
    significant_tokens,
)
from gsr_cacl.ontology.concepts import (
    canonical_concept,
    concepts_in_text,
    ALL_CONCEPTS,
    CONCEPT_TO_ID,
    IDENTITIES,
)

__all__ = [
    "GICS_SECTORS", "N_SECTORS", "canonical_sector", "sector_id", "sector_industry_path",
    "normalize_company", "company_acronym", "company_match", "company_match_score",
    "significant_tokens",
    "canonical_concept", "concepts_in_text", "ALL_CONCEPTS", "CONCEPT_TO_ID", "IDENTITIES",
]
