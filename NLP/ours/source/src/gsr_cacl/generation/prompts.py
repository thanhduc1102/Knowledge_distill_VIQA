"""Prompt construction for fact-grounded financial generation.

The generator receives the *precise selected facts* (not raw markdown), which is the
whole point of the KG-for-generator design: even with top-3 documents, the LLM is handed
the exact cells it needs.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a meticulous financial analyst. Answer the question using ONLY the financial "
    "facts provided. The facts are exact values extracted from the source tables. "
    "Reason step by step: identify the relevant cells, perform any arithmetic exactly, and "
    "respect the stated scale (e.g. 'in millions'). Do not invent numbers. "
    "End with a single line: 'Answer: <number>' giving the numeric answer only."
)


def build_user_prompt(query: str, evidence_block: str, meta: dict | None = None) -> str:
    meta = meta or {}
    head = ""
    if meta.get("company_name"):
        head = f"Company: {meta.get('company_name')}"
        if meta.get("report_year"):
            head += f" | Reporting context year: {meta.get('report_year')}"
        head += "\n"
    return (
        f"{head}"
        f"{evidence_block}\n\n"
        f"Question: {query}\n"
        f"Think step by step, then give the final numeric answer.\n"
        f"Answer:"
    )


def build_chat_messages(query: str, evidence_block: str, meta: dict | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, evidence_block, meta)},
    ]
