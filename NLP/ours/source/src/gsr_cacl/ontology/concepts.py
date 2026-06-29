"""IFRS/GAAP/XBRL concept ontology for the structural retrieval channel (contribution C2).

The original GSR matched 15 hand-written templates against *column headers*. Financial
tables here are row-major (the line-item is the ROW label), so those templates almost never
fire and the constraint signal is dead. This module instead canonicalises **line-item
labels** onto a compact US-GAAP/XBRL-style concept taxonomy, regardless of surface wording
("gross profit" / "gross income" / "gross margin $" → ``GrossProfit``). On top of canonical
concepts we declare a small set of accounting identities that therefore fire by *meaning*,
not by string — this is what lets the structural signal (C3) and the verifier (C5) work on
the real data orientation.

Pure-Python, deterministic, no dependencies.
"""

from __future__ import annotations

import re

# canonical concept  ->  alias phrases (lower-case, matched on word boundaries)
CONCEPT_ALIASES: dict[str, list[str]] = {
    # ---- income statement ----
    "Revenue": ["total revenue", "net revenue", "net sales", "total sales", "revenues",
                "revenue", "sales", "net revenues", "total net revenue"],
    "CostOfRevenue": ["cost of goods sold", "cost of sales", "cost of revenue", "cogs",
                      "cost of products sold", "cost of services"],
    "GrossProfit": ["gross profit", "gross income", "gross margin"],
    "OperatingExpenses": ["operating expenses", "total operating expenses", "opex",
                          "operating costs and expenses"],
    "SGAndA": ["selling general and administrative", "sg&a", "sga",
               "selling, general and administrative"],
    "ResearchAndDevelopment": ["research and development", "r&d", "research development"],
    "DepreciationAmortization": ["depreciation and amortization", "depreciation",
                                 "amortization", "d&a"],
    "OperatingIncome": ["operating income", "income from operations", "operating profit",
                        "operating earnings", "ebit"],
    "InterestExpense": ["interest expense", "interest expense net", "net interest expense"],
    "IncomeTaxExpense": ["income tax expense", "provision for income taxes", "income taxes",
                         "tax expense", "provision for taxes"],
    "PretaxIncome": ["income before income taxes", "pretax income", "income before taxes",
                     "earnings before income taxes"],
    "NetIncome": ["net income", "net earnings", "net income loss", "net profit",
                  "profit for the year", "net income attributable"],
    "EPS": ["earnings per share", "diluted earnings per share", "basic earnings per share",
            "eps", "diluted eps", "per share"],
    "EBITDA": ["ebitda", "adjusted ebitda"],
    "SharesOutstanding": ["shares outstanding", "weighted average shares",
                          "diluted shares", "common shares outstanding"],
    "Dividends": ["dividends", "dividends paid", "dividends declared", "dividends per share"],
    # ---- balance sheet ----
    "CashAndEquivalents": ["cash and cash equivalents", "cash and equivalents", "cash"],
    "AccountsReceivable": ["accounts receivable", "receivables", "trade receivables"],
    "Inventory": ["inventory", "inventories", "merchandise inventory"],
    "CurrentAssets": ["total current assets", "current assets"],
    "PPE": ["property plant and equipment", "property and equipment", "ppe",
            "net property plant and equipment"],
    "Goodwill": ["goodwill"],
    "IntangibleAssets": ["intangible assets", "other intangible assets", "intangibles"],
    "TotalAssets": ["total assets"],
    "AccountsPayable": ["accounts payable", "trade payables", "payables"],
    "CurrentLiabilities": ["total current liabilities", "current liabilities"],
    "LongTermDebt": ["long-term debt", "long term debt", "lt debt", "long-term borrowings"],
    "ShortTermDebt": ["short-term debt", "short term debt", "current portion of long-term debt",
                      "short-term borrowings", "current borrowings"],
    "TotalDebt": ["total debt", "total borrowings"],
    "TotalLiabilities": ["total liabilities"],
    "RetainedEarnings": ["retained earnings", "accumulated deficit", "accumulated earnings"],
    "CommonStock": ["common stock", "common shares", "share capital"],
    "TotalEquity": ["total equity", "total stockholders equity", "total shareholders equity",
                    "stockholders equity", "shareholders equity", "total stockholders' equity"],
    # ---- cash flow ----
    "OperatingCashFlow": ["net cash provided by operating activities",
                          "cash from operating activities", "operating cash flow",
                          "net cash from operating activities", "cash flow from operations"],
    "InvestingCashFlow": ["net cash used in investing activities",
                          "cash from investing activities", "investing cash flow",
                          "net cash from investing activities"],
    "FinancingCashFlow": ["net cash provided by financing activities",
                          "cash from financing activities", "financing cash flow",
                          "net cash from financing activities"],
    "CapitalExpenditure": ["capital expenditures", "capital expenditure", "capex",
                           "purchases of property and equipment", "additions to property"],
    "NetChangeInCash": ["net change in cash", "net increase in cash", "net decrease in cash",
                        "increase decrease in cash"],
    # ---- extended income-statement line items ----
    "OtherIncome": ["other income", "other income expense", "other income net",
                    "other operating income", "miscellaneous income"],
    "OtherExpense": ["other expense", "other expenses", "other operating expense",
                     "other operating expenses"],
    "InterestIncome": ["interest income", "interest and dividend income", "investment income"],
    "StockBasedCompensation": ["stock-based compensation", "share-based compensation",
                               "stock based compensation expense", "equity compensation"],
    "RestructuringCharges": ["restructuring charges", "restructuring costs", "restructuring"],
    "ImpairmentCharges": ["impairment", "impairment charges", "goodwill impairment",
                          "asset impairment"],
    "MinorityInterest": ["noncontrolling interest", "minority interest",
                         "net income attributable to noncontrolling interests"],
    "CostOfServices": ["cost of services", "cost of service revenue"],
    "CostOfProducts": ["cost of products", "cost of products sold", "product costs"],
    "ComprehensiveIncome": ["comprehensive income", "total comprehensive income",
                            "other comprehensive income"],
    # ---- extended balance-sheet line items ----
    "PrepaidExpenses": ["prepaid expenses", "prepaid expenses and other current assets",
                        "prepaid and other"],
    "DeferredRevenue": ["deferred revenue", "unearned revenue", "contract liabilities",
                        "deferred income"],
    "DeferredTaxAssets": ["deferred tax assets", "deferred income tax assets"],
    "DeferredTaxLiabilities": ["deferred tax liabilities", "deferred income tax liabilities"],
    "OtherCurrentAssets": ["other current assets"],
    "OtherCurrentLiabilities": ["other current liabilities", "accrued liabilities",
                                "accrued expenses", "other accrued liabilities"],
    "OtherAssets": ["other assets", "other long-term assets", "other noncurrent assets"],
    "OtherLiabilities": ["other liabilities", "other long-term liabilities",
                         "other noncurrent liabilities"],
    "AccumulatedDepreciation": ["accumulated depreciation",
                                "accumulated depreciation and amortization"],
    "TreasuryStock": ["treasury stock", "treasury shares", "common stock in treasury"],
    "AdditionalPaidInCapital": ["additional paid-in capital", "paid-in capital",
                                "capital in excess of par"],
    "AccumulatedOCI": ["accumulated other comprehensive income",
                       "accumulated other comprehensive loss", "aoci"],
    "WorkingCapital": ["working capital", "net working capital"],
    "NetDebt": ["net debt"],
    "InvestmentsLongTerm": ["long-term investments", "investments", "marketable securities",
                            "available-for-sale securities"],
    "OperatingLeaseLiabilities": ["operating lease liabilities", "lease liabilities",
                                  "operating lease obligations"],
    "PensionObligations": ["pension obligations", "pension liabilities",
                           "defined benefit obligation", "postretirement benefit obligation"],
    # ---- extended cash-flow line items ----
    "ShareRepurchases": ["repurchases of common stock", "share repurchases", "stock repurchases",
                         "treasury stock purchases", "purchase of treasury stock"],
    "DebtIssued": ["proceeds from issuance of debt", "proceeds from long-term debt",
                   "proceeds from borrowings"],
    "DebtRepaid": ["repayments of debt", "repayment of long-term debt", "principal payments",
                   "repayments of borrowings"],
    "AcquisitionsCF": ["acquisitions net of cash acquired", "business acquisitions",
                       "payments for acquisitions"],
    "FreeCashFlow": ["free cash flow", "fcf"],
    "ChangeInWorkingCapital": ["changes in working capital",
                               "changes in operating assets and liabilities"],
    # ---- ratios ----
    "GrossMarginRatio": ["gross margin percentage", "gross margin rate", "gross margin %"],
    "OperatingMarginRatio": ["operating margin", "operating margin percentage"],
    "NetMarginRatio": ["net margin", "net profit margin", "profit margin"],
    "EffectiveTaxRate": ["effective tax rate", "effective income tax rate"],
    "ReturnOnAssets": ["return on assets", "roa"],
    "ReturnOnEquity": ["return on equity", "roe"],
    "CurrentRatio": ["current ratio"],
    "DebtToEquity": ["debt to equity", "debt-to-equity", "debt equity ratio"],
    "InterestCoverage": ["interest coverage", "times interest earned"],
}

ALL_CONCEPTS: list[str] = list(CONCEPT_ALIASES.keys())
CONCEPT_TO_ID: dict[str, int] = {c: i for i, c in enumerate(ALL_CONCEPTS)}

# Pre-sort (concept, alias) by alias length DESC so the most specific phrase wins
# ("gross profit" beats "profit"; "operating income" beats "income").
_ALIAS_INDEX: list[tuple[str, str]] = sorted(
    ((c, a) for c, als in CONCEPT_ALIASES.items() for a in als),
    key=lambda ca: len(ca[1]), reverse=True,
)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9&% ]+")

# Precompiled per-alias patterns (longest first) — avoids re-compiling on every call.
_ALIAS_COMPILED: list[tuple[str, str, re.Pattern]] = [
    (c, a, re.compile(rf"(?<![a-z]){re.escape(a)}(?![a-z])"))
    for c, a in _ALIAS_INDEX
]

# Alias → concept lookup for the combined finditer path used by concepts_in_text.
_ALIAS_TO_CONCEPT: dict[str, str] = {
    a: c for c, als in CONCEPT_ALIASES.items() for a in als
}
# Single combined alternation (longest alias first) for a one-pass full-text scan.
_CONCEPTS_ALL_RE: re.Pattern = re.compile(
    r"(?<![a-z])("
    + "|".join(re.escape(a) for a in sorted(_ALIAS_TO_CONCEPT, key=len, reverse=True))
    + r")(?![a-z])"
)


def _norm(text: str) -> str:
    t = str(text or "").lower().replace("’", "’").replace("’s", "")
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def canonical_concept(label: str) -> str | None:
    """Map a line-item label onto a canonical concept, or ``None`` if unrecognised."""
    n = _norm(label)
    if not n:
        return None
    for concept, alias, pat in _ALIAS_COMPILED:
        if pat.search(n):
            return concept
    return None


# Accounting identities over CANONICAL concepts: target = Σ ω_i · operand_i.
# Stored as (target, [(operand, ω)]). These fire by meaning, not by surface header.
IDENTITIES: list[tuple[str, list[tuple[str, int]]]] = [
    ("GrossProfit", [("Revenue", +1), ("CostOfRevenue", -1)]),
    ("OperatingIncome", [("GrossProfit", +1), ("OperatingExpenses", -1)]),
    ("PretaxIncome", [("OperatingIncome", +1), ("InterestExpense", -1)]),
    ("NetIncome", [("PretaxIncome", +1), ("IncomeTaxExpense", -1)]),
    ("NetChangeInCash", [("OperatingCashFlow", +1), ("InvestingCashFlow", +1),
                         ("FinancingCashFlow", +1)]),
    ("TotalAssets", [("TotalLiabilities", +1), ("TotalEquity", +1)]),
    ("TotalDebt", [("LongTermDebt", +1), ("ShortTermDebt", +1)]),
]


def concepts_in_text(text: str) -> set[str]:
    """All canonical concepts whose alias appears in a free-text string (e.g. a question)."""
    n = _norm(text)
    return {_ALIAS_TO_CONCEPT[m.group(1)] for m in _CONCEPTS_ALL_RE.finditer(n)}
