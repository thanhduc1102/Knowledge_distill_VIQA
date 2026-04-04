"""
PCPO Reward function for GRPO training.
Implements R(p, x) = R_valid * (alpha + beta * R_exec(p, x) + gamma * R_bonus)
"""

import re
from typing import Any, Optional

from pipeline.program_executor import execute_program, validate_program, format_answer


def compute_pcpo_reward(
    prediction_text: str,
    ground_truth: dict,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
) -> float:
    """
    Compute PCPO reward for a single prediction.

    R(p, x) = R_valid * (alpha + beta * R_exec + gamma * R_bonus)

    Args:
        prediction_text: Model's full output text
        ground_truth: dict with keys 'table', 'program', 'answer'
        alpha: Base reward for valid program (default 0.7)
        beta: Weight for execution accuracy (default 0.2)
        gamma: Weight for conciseness bonus (default 0.1)

    Returns:
        Reward value in [0, 1]
    """
    # Extract program and answer from prediction
    pred_program, pred_answer = _extract_program_and_answer(prediction_text)

    # R_valid: Is the program syntactically valid?
    if pred_program is None or not validate_program(pred_program):
        return 0.0
    r_valid = 1.0

    # R_exec: Does the program produce the correct answer?
    table = ground_truth.get("table")
    gold_answer = str(ground_truth.get("answer", "")).strip()

    pred_result = execute_program(pred_program, table)
    if pred_result is not None:
        pred_formatted = format_answer(pred_result)
        r_exec = 1.0 if _answers_match(pred_formatted, gold_answer) else 0.0
    else:
        r_exec = 0.0

    # R_bonus: Conciseness bonus
    gold_program = ground_truth.get("program", "")
    pred_steps = _count_steps(pred_program)
    gold_steps = _count_steps(gold_program)

    if pred_steps < gold_steps:
        r_bonus = 1.0
    elif pred_steps == gold_steps:
        r_bonus = 0.5
    else:
        r_bonus = 0.1

    return r_valid * (alpha + beta * r_exec + gamma * r_bonus)


def _extract_program_and_answer(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract program and answer from formatted output."""
    program_matches = list(re.finditer(
        r"\*\*Chương trình tính toán:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text
    ))
    program = program_matches[-1].group(1).strip() if program_matches else None

    answer_matches = list(re.finditer(
        r"\*\*Đáp án cuối cùng:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text
    ))
    answer = answer_matches[-1].group(1).strip() if answer_matches else None

    return program, answer


def _answers_match(pred: str, gold: str) -> bool:
    """Check if two answer strings match (with tolerance for floating point)."""
    try:
        p = float(pred)
        g = float(gold)
        if g == 0:
            return abs(p - g) < 1e-5
        return abs(p - g) / max(abs(g), 1e-10) < 1e-4
    except (ValueError, TypeError):
        return pred.strip() == gold.strip()


def _count_steps(program: str) -> int:
    """Count the number of function calls in a program."""
    return len(re.findall(r'\w+\(', program))
