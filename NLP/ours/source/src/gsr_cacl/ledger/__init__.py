"""Fact Ledger — the shared KG substrate for retrieval AND generation.

Public API::

    from gsr_cacl.ledger import (
        Fact, FactLedger, extract_ledger, extract_ledger_from_table,
        select_facts, build_evidence_block, calculation_plan,
    )
"""

from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.extract import (
    extract_ledger,
    extract_ledger_from_table,
    extract_table_md,
    detect_scale,
    detect_unit,
)
from gsr_cacl.ledger.select import select_facts, build_evidence_block, calculation_plan, score_fact
from gsr_cacl.ledger.numeric import (
    parse_financial_number,
    extract_numbers,
    extract_years,
    number_match,
    numbers_close,
)

__all__ = [
    "Fact",
    "FactLedger",
    "extract_ledger",
    "extract_ledger_from_table",
    "extract_table_md",
    "detect_scale",
    "detect_unit",
    "select_facts",
    "build_evidence_block",
    "calculation_plan",
    "score_fact",
    "parse_financial_number",
    "extract_numbers",
    "extract_years",
    "number_match",
    "numbers_close",
]
