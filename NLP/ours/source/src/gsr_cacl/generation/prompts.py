"""Prompt construction for fact-grounded financial generation.

The generator receives the *precise selected facts* (not raw markdown), which is the
whole point of the KG-for-generator design: even with top-3 documents, the LLM is handed
the exact cells it needs.
"""

from __future__ import annotations

from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.concepts import concepts_in_text

SYSTEM_PROMPT = (
    "You are a precise financial QA model. Use only the provided KG-selected document, "
    "facts, operands, TASK, FORMULA, and CALCULATION HINT. If KG_SYMBOLIC_ANSWER is "
    "present, treat it as the primary computed result and copy it unless the evidence "
    "block explicitly shows a conflict. Do not reveal reasoning. Return exactly one line "
    "only: Answer: <number>. For percentage questions, prefer the decimal fraction (for "
    "example 0.1446) rather than multiplying by 100."
)


def _format_focus_hints(query: str, meta: dict | None = None) -> str:
    meta = meta or {}
    lines: list[str] = []
    concepts = sorted(concepts_in_text(query))
    if concepts:
        lines.append("Canonical concept hints: " + ", ".join(concepts))
    years = sorted(set(extract_years(query)))
    if years:
        lines.append("Period hints: " + ", ".join(str(y) for y in years))
    if meta.get("company_name"):
        lines.append(f"Company hint: {meta.get('company_name')}")
    if meta.get("report_year"):
        lines.append(f"Report year hint: {meta.get('report_year')}")
    return "\n".join(lines)


def build_user_prompt(query: str, evidence_block: str, meta: dict | None = None) -> str:
    meta = meta or {}
    sections: list[str] = []
    if meta.get("company_name"):
        company_line = f"Company: {meta.get('company_name')}"
        if meta.get("report_year"):
            company_line += f" | Reporting context year: {meta.get('report_year')}"
        sections.append(company_line)
    hints = _format_focus_hints(query, meta)
    if hints:
        sections.append(hints)
    if evidence_block.strip():
        sections.append(evidence_block.strip())
    sections.append(f"Question: {query}")
    sections.append(
        "Instructions: if KG_SYMBOLIC_ANSWER is present, return that value as "
        "Answer: <number> after checking it against KG_OPERAND_PROVENANCE. Otherwise "
        "follow TASK, FORMULA, and CALCULATION HINT exactly, use only the facts above, "
        "and return a single numeric line in the form Answer: <number>. No explanation, "
        "no bullets, and no extra text."
    )
    return "\n\n".join(sections)


def build_chat_messages(query: str, evidence_block: str, meta: dict | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, evidence_block, meta)},
    ]
