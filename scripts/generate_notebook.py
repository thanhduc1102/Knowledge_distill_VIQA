#!/usr/bin/env python3
"""
Generate kaggle_kd_notebook.ipynb from kaggle_kd_notebook.py

Parses the .py file's CELL markers and produces a proper Jupyter .ipynb file
with separate code cells. This way users can import/push it directly to Kaggle
without any copy-paste.

Usage:
    python scripts/generate_notebook.py
    # Output: kaggle/kaggle_kd_notebook.ipynb
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY_FILE = PROJECT_ROOT / "kaggle" / "kaggle_kd_notebook.py"
IPYNB_FILE = PROJECT_ROOT / "kaggle" / "kaggle_kd_notebook.ipynb"


def parse_cells(py_path: Path) -> list[dict]:
    """Parse the .py file and split into notebook cells based on CELL markers."""
    text = py_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    cells = []
    current_lines = []
    current_title = None
    in_docstring = False

    # The file starts with a module docstring — convert it to a markdown cell
    # Then each "# CELL N:" block becomes a code cell

    i = 0
    # Parse leading docstring as markdown
    if lines[0].startswith('"""') or lines[0].startswith("'''"):
        docstring_lines = []
        quote = lines[0][:3]
        if lines[0].strip() == quote:
            # Multi-line docstring starting with just """
            i = 1
            while i < len(lines) and quote not in lines[i]:
                docstring_lines.append(lines[i])
                i += 1
            i += 1  # skip closing """
        elif lines[0].endswith(quote) and len(lines[0]) > 6:
            # Single line """..."""
            docstring_lines.append(lines[0][3:-3])
            i = 1
        else:
            # Opening """ with text on same line
            docstring_lines.append(lines[0][3:])
            i = 1
            while i < len(lines) and quote not in lines[i]:
                docstring_lines.append(lines[i])
                i += 1
            if i < len(lines) and quote in lines[i]:
                # Last line might have text before closing """
                last = lines[i].split(quote)[0]
                if last.strip():
                    docstring_lines.append(last)
                i += 1

        # Convert docstring to markdown
        md_text = "\n".join(docstring_lines).strip()
        if md_text:
            cells.append(_make_markdown_cell(md_text))

    # Skip blank lines after docstring
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    # Now parse CELL markers
    cell_marker = re.compile(r"^#\s*═+\s*$")
    cell_title_re = re.compile(r"^#\s*CELL\s+\d+.*$")

    while i < len(lines):
        line = lines[i]

        # Detect cell boundary: line of ═ characters
        if cell_marker.match(line):
            # Save previous cell if any
            if current_lines:
                code = _clean_cell_code(current_lines)
                if code.strip():
                    if current_title:
                        cells.append(_make_markdown_cell(f"## {current_title}"))
                    cells.append(_make_code_cell(code))
                current_lines = []
                current_title = None

            # Read the title line (next line)
            i += 1
            if i < len(lines) and cell_title_re.match(lines[i]):
                # Extract title: "# CELL 5: Configure Pipeline" -> "Cell 5: Configure Pipeline"
                title = lines[i].lstrip("# ").strip()
                current_title = title
                i += 1
                # Collect any extra comment lines between title and closing ═
                # e.g. "# (Comment out / skip to use SFT-only: ...)"
                # These become part of the markdown header
                extra_comments = []
                while i < len(lines) and lines[i].startswith("#") and not cell_marker.match(lines[i]):
                    comment_text = lines[i].lstrip("# ").strip()
                    if comment_text:
                        extra_comments.append(comment_text)
                    i += 1
                if extra_comments:
                    current_title = current_title + "\n" + "\n".join(f"*{c}*" for c in extra_comments)
                # Skip closing ═ line
                if i < len(lines) and cell_marker.match(lines[i]):
                    i += 1
                continue
            else:
                # Not a cell title, treat as regular code
                if i < len(lines):
                    current_lines.append(lines[i])
                i += 1
                continue

        current_lines.append(line)
        i += 1

    # Save last cell
    if current_lines:
        code = _clean_cell_code(current_lines)
        if code.strip():
            if current_title:
                cells.append(_make_markdown_cell(f"## {current_title}"))
            cells.append(_make_code_cell(code))

    return cells


def _clean_cell_code(lines: list[str]) -> str:
    """Strip leading/trailing blank lines from cell code."""
    # Remove leading blank lines
    while lines and lines[0].strip() == "":
        lines = lines[1:]
    # Remove trailing blank lines
    while lines and lines[-1].strip() == "":
        lines = lines[:-1]
    return "\n".join(lines)


def _make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "trusted": True,
        },
        "outputs": [],
        "source": source.split("\n") if "\n" in source else [source],
    }


def _make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n") if "\n" in source else [source],
    }


def fix_cell_sources(cells: list[dict]) -> list[dict]:
    """Ensure each line in source (except the last) ends with \\n for valid ipynb."""
    for cell in cells:
        src = cell.get("source", [])
        if isinstance(src, str):
            src = src.split("\n")
        # Each line except the last gets a trailing \n
        fixed = []
        for j, line in enumerate(src):
            if j < len(src) - 1:
                if not line.endswith("\n"):
                    line = line + "\n"
            else:
                # Last line: strip trailing \n
                line = line.rstrip("\n")
            fixed.append(line)
        cell["source"] = fixed
    return cells


def build_notebook(cells: list[dict]) -> dict:
    return {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells,
    }


def main():
    if not PY_FILE.exists():
        print(f"ERROR: Source file not found: {PY_FILE}")
        sys.exit(1)

    print(f"Parsing: {PY_FILE}")
    cells = parse_cells(PY_FILE)
    cells = fix_cell_sources(cells)

    notebook = build_notebook(cells)

    IPYNB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(IPYNB_FILE, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    code_cells = sum(1 for c in cells if c["cell_type"] == "code")
    md_cells = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"Generated: {IPYNB_FILE}")
    print(f"  Cells: {code_cells} code + {md_cells} markdown = {len(cells)} total")


if __name__ == "__main__":
    main()
