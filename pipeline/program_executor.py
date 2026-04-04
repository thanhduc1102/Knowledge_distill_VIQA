"""
Program DSL executor for financial numerical reasoning.
Supports: add, subtract, multiply, divide, exp, greater, table_sum, table_average, table_max, table_min
"""

import re
import math
from typing import Any, Optional


def _parse_number(val: str) -> float:
    """Parse a number string, handling Vietnamese/financial formatting."""
    val = val.strip()
    # Handle constants
    const_map = {
        "const_100": 100, "const_1000": 1000, "const_1000000": 1e6,
        "const_1": 1, "const_2": 2, "const_3": 3, "const_4": 4,
        "const_10": 10, "const_12": 12, "const_m1": -1,
    }
    if val in const_map:
        return float(const_map[val])
    # Remove commas used as thousand separators
    val = val.replace(",", "")
    # Handle percentage
    if val.endswith("%"):
        return float(val[:-1]) / 100.0
    return float(val)


def _extract_table_row(table: list, header: str) -> list:
    """Extract numeric values from a table row by header name.
    Uses multiple matching strategies for Vietnamese/financial headers."""
    header_clean = header.strip()

    # Strategy 1: Exact match
    for row in table:
        if row and str(row[0]).strip() == header_clean:
            return _row_to_numbers(row[1:])

    # Strategy 2: Case-insensitive match
    header_lower = header_clean.lower()
    for row in table:
        if row and str(row[0]).strip().lower() == header_lower:
            return _row_to_numbers(row[1:])

    # Strategy 3: Stripped parentheses/special chars match
    header_stripped = re.sub(r'\s*\(.*?\)\s*', '', header_clean).strip().lower()
    for row in table:
        if row:
            row_stripped = re.sub(r'\s*\(.*?\)\s*', '', str(row[0])).strip().lower()
            if row_stripped == header_stripped and header_stripped:
                return _row_to_numbers(row[1:])

    # Strategy 4: Substring match (header contained in row or vice versa)
    for row in table:
        if row:
            row_text = str(row[0]).strip().lower()
            if header_lower in row_text or row_text in header_lower:
                if len(header_lower) > 2 and len(row_text) > 2:
                    return _row_to_numbers(row[1:])

    return []


def _row_to_numbers(cells: list) -> list:
    """Convert a list of cells to numbers, skipping non-numeric."""
    values = []
    for cell in cells:
        cell_str = str(cell).strip()
        if not cell_str or cell_str == '-' or cell_str == 'N/A':
            continue
        # Clean financial formatting
        cell_str = cell_str.replace("$", "").replace("€", "").replace("₫", "")
        cell_str = cell_str.split("(")[0].strip()  # Remove parenthetical notes
        try:
            values.append(_parse_number(cell_str))
        except (ValueError, TypeError):
            continue
    return values


def execute_program(program_str: str, table: Optional[list] = None) -> Any:
    """
    Execute a program string and return the final result.

    Args:
        program_str: e.g. "divide(914, 391), multiply(#0, const_100)"
        table: 2D list (first row = header) for table_* operations

    Returns:
        Final computed value, or None on error
    """
    steps = _split_steps(program_str)
    results = []

    for step in steps:
        try:
            result = _execute_step(step, results, table)
            results.append(result)
        except Exception:
            return None

    return results[-1] if results else None


def _split_steps(program_str: str) -> list:
    """Split program string into individual function calls."""
    steps = []
    depth = 0
    current = ""
    for ch in program_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
            if depth == 0:
                steps.append(current.strip())
                current = ""
        elif ch == "," and depth == 0:
            if current.strip():
                steps.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        steps.append(current.strip())
    return steps


def _resolve_arg(arg: str, results: list) -> float:
    """Resolve an argument - could be a step reference (#0), constant, or number."""
    arg = arg.strip()
    if arg.startswith("#"):
        idx = int(arg[1:])
        return float(results[idx])
    return _parse_number(arg)


def _execute_step(step: str, results: list, table: Optional[list]) -> Any:
    """Execute a single program step."""
    match = re.match(r"(\w+)\((.+)\)$", step, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid step: {step}")

    func_name = match.group(1).strip()
    args_str = match.group(2).strip()

    # Table operations take (header, none)
    if func_name in ("table_sum", "table_average", "table_max", "table_min"):
        parts = args_str.split(",")
        header = parts[0].strip()
        values = _extract_table_row(table, header) if table else []
        if not values:
            raise ValueError(f"No values found for header: {header}")
        if func_name == "table_sum":
            return sum(values)
        elif func_name == "table_average":
            return sum(values) / len(values)
        elif func_name == "table_max":
            return max(values)
        elif func_name == "table_min":
            return min(values)

    # Binary operations
    parts = args_str.split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected 2 args: {step}")
    a = _resolve_arg(parts[0], results)
    b = _resolve_arg(parts[1], results)

    if func_name == "add":
        return a + b
    elif func_name == "subtract":
        return a - b
    elif func_name == "multiply":
        return a * b
    elif func_name == "divide":
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    elif func_name == "exp":
        return math.pow(a, b)
    elif func_name == "greater":
        return a > b
    else:
        raise ValueError(f"Unknown function: {func_name}")


def format_answer(value: Any) -> str:
    """Format answer according to competition rules (max 5 decimal places, strip trailing zeros)."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return ""
    val = float(value)
    formatted = f"{val:.5f}"
    if "." in formatted:
        formatted = formatted.rstrip("0")
        if formatted.endswith("."):
            formatted += "0"
    return formatted


def validate_program(program_str: str) -> bool:
    """Check if a program string has valid syntax."""
    try:
        steps = _split_steps(program_str)
        if not steps:
            return False
        for step in steps:
            match = re.match(r"(\w+)\((.+)\)$", step, re.DOTALL)
            if not match:
                return False
            func = match.group(1).strip()
            valid_funcs = {
                "add", "subtract", "multiply", "divide", "exp", "greater",
                "table_sum", "table_average", "table_max", "table_min",
            }
            if func not in valid_funcs:
                return False
        return True
    except Exception:
        return False
