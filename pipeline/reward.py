"""
Reward functions for verifier-guided GRPO training.

PCPO keeps the original hard-gated reward:
R(p, x) = R_valid * (alpha + beta * R_exec(p, x) + gamma * R_bonus)

ECRL-Fin is a softer equivalence-class reward that combines syntax validity,
execution accuracy, symbolic program equivalence, intermediate-step agreement,
answer formatting, and conciseness.
"""

import math
import re
from typing import Any, Optional

from pipeline.evaluate import programs_match
from pipeline.program_executor import execute_program, execute_program_steps, validate_program, format_answer


ECRL_DEFAULT_WEIGHTS = {
    "valid": 0.20,
    "execution": 0.25,
    "program_equiv": 0.25,
    "step": 0.15,
    "answer": 0.10,
    "brevity": 0.05,
}


def compute_grpo_reward(
    prediction_text: str,
    ground_truth: dict,
    reward_type: str = "pcpo",
    reward_weights: Optional[dict] = None,
) -> float:
    """Compute the configured GRPO reward."""
    reward_weights = reward_weights or {}
    reward_type = reward_type.lower().strip()

    if reward_type == "pcpo":
        return compute_pcpo_reward(
            prediction_text,
            ground_truth,
            alpha=reward_weights.get("alpha", 0.7),
            beta=reward_weights.get("beta", 0.2),
            gamma=reward_weights.get("gamma", 0.1),
        )
    if reward_type == "ecrl":
        return compute_ecrl_reward(prediction_text, ground_truth, reward_weights)

    raise ValueError(f"Unknown GRPO reward type: {reward_type}")


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


def compute_ecrl_reward(
    prediction_text: str,
    ground_truth: dict,
    reward_weights: Optional[dict] = None,
) -> float:
    """
    Compute the ECRL-Fin reward for a single prediction.

    Unlike PCPO, this reward is not a binary hard gate. Invalid programs still
    receive limited syntax-shaping signal, while executable and symbolically
    equivalent programs receive dense credit.
    """
    weights = dict(ECRL_DEFAULT_WEIGHTS)
    if reward_weights:
        weights.update({key: reward_weights[key] for key in ECRL_DEFAULT_WEIGHTS if key in reward_weights})
    weights = {key: max(0.0, float(value)) for key, value in weights.items()}
    total_weight = sum(weights[key] for key in ECRL_DEFAULT_WEIGHTS) or 1.0

    pred_program, pred_answer = _extract_program_and_answer(prediction_text)
    table = ground_truth.get("table")
    gold_program = str(ground_truth.get("program", "")).strip()
    gold_answer = str(ground_truth.get("answer", "")).strip()

    r_valid = _soft_validity_score(pred_program)
    r_exec = 0.0
    r_equiv = 0.0
    r_step = 0.0
    r_answer = 1.0 if pred_answer and _answers_match(pred_answer, gold_answer) else 0.0
    r_brevity = _brevity_score(pred_program, gold_program)

    if pred_program and validate_program(pred_program):
        pred_result = execute_program(pred_program, table)
        if pred_result is not None:
            r_exec = 1.0 if _answers_match(format_answer(pred_result), gold_answer) else 0.0

        if gold_program:
            r_equiv = 1.0 if programs_match(pred_program, gold_program) else 0.0
            r_step = _intermediate_step_reward(pred_program, gold_program, table)

    reward = (
        weights["valid"] * r_valid
        + weights["execution"] * r_exec
        + weights["program_equiv"] * r_equiv
        + weights["step"] * r_step
        + weights["answer"] * r_answer
        + weights["brevity"] * r_brevity
    ) / total_weight

    return _clamp01(reward)


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


def _soft_validity_score(program: Optional[str]) -> float:
    """Return a syntax-shaping score in [0, 1] without executing the program."""
    if not program:
        return 0.0
    program = program.strip()
    if validate_program(program):
        return 1.0

    score = 0.0
    if program.count("(") == program.count(")") and "(" in program and ")" in program:
        score += 0.25

    function_names = re.findall(r"\b([A-Za-z_]\w*)\s*\(", program)
    valid_functions = {
        "add", "subtract", "multiply", "divide", "exp", "greater",
        "table_sum", "table_average", "table_max", "table_min",
    }
    if function_names:
        known = sum(1 for name in function_names if name in valid_functions)
        score += 0.50 * (known / len(function_names))

    if re.search(r"#\d+|const_\w+|-?\d+(?:\.\d+)?", program):
        score += 0.25

    return _clamp01(score)


def _brevity_score(pred_program: Optional[str], gold_program: str) -> float:
    """Reward concise programs without forcing exact surface form."""
    if not pred_program:
        return 0.0
    pred_steps = _count_steps(pred_program)
    gold_steps = _count_steps(gold_program)
    if pred_steps == 0 or gold_steps == 0:
        return 0.0
    if pred_steps < gold_steps:
        return 1.0
    if pred_steps == gold_steps:
        return 0.5
    return max(0.0, 1.0 - (pred_steps - gold_steps) / max(gold_steps, 1))


def _intermediate_step_reward(pred_program: str, gold_program: str, table: Optional[list]) -> float:
    """Reward agreement between executable intermediate program states."""
    pred_steps = execute_program_steps(pred_program, table)
    gold_steps = execute_program_steps(gold_program, table)
    if not pred_steps or not gold_steps:
        return 0.0

    comparisons = min(len(pred_steps), len(gold_steps))
    if comparisons == 0:
        return 0.0

    matches = 0
    for pred_value, gold_value in zip(pred_steps[:comparisons], gold_steps[:comparisons]):
        if _values_match(pred_value, gold_value):
            matches += 1

    return matches / max(len(gold_steps), len(pred_steps))


def _values_match(pred_value: Any, gold_value: Any) -> bool:
    if isinstance(pred_value, bool) or isinstance(gold_value, bool):
        return bool(pred_value) == bool(gold_value)
    try:
        pred_float = float(pred_value)
        gold_float = float(gold_value)
        if not math.isfinite(pred_float) or not math.isfinite(gold_float):
            return False
        if gold_float == 0:
            return abs(pred_float - gold_float) < 1e-5
        return abs(pred_float - gold_float) / max(abs(gold_float), 1e-10) < 1e-4
    except (ValueError, TypeError):
        return str(pred_value).strip() == str(gold_value).strip()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
