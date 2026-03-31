"""
Evaluation module for Execution Accuracy (EA) and Program Accuracy (PA).
PA uses symbolic evaluation via program comparison.
"""

import re
import json
from pathlib import Path
from typing import Optional

from pipeline.program_executor import execute_program, format_answer, validate_program


def extract_program_and_answer(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract program and answer from formatted text."""
    prog_matches = list(re.finditer(
        r"\*\*Chương trình tính toán:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text
    ))
    prog = prog_matches[-1].group(1).strip() if prog_matches else None

    ans_matches = list(re.finditer(
        r"\*\*Đáp án cuối cùng:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text
    ))
    ans = ans_matches[-1].group(1).strip() if ans_matches else None

    return prog, ans


def normalize_program(program: str) -> str:
    """Normalize program string for comparison."""
    p = re.sub(r'\s+', ' ', program.strip())
    p = re.sub(r',\s*', ', ', p)
    return p


def programs_match(pred_program: str, gold_program: str) -> bool:
    """
    Check if two programs are mathematically equivalent.
    Uses symbolic comparison by normalizing and tokenizing.
    """
    if not pred_program or not gold_program:
        return False

    pred_norm = normalize_program(pred_program)
    gold_norm = normalize_program(gold_program)

    # Direct match
    if pred_norm == gold_norm:
        return True

    # Token-level comparison (handles whitespace differences)
    from src.program_tokenizer import program_tokenization
    try:
        pred_tokens = program_tokenization(pred_norm)
        gold_tokens = program_tokenization(gold_norm)
        return pred_tokens == gold_tokens
    except Exception:
        return pred_norm == gold_norm


def answers_match(pred_answer: str, gold_answer: str) -> bool:
    """Check if two answers match (with numeric tolerance)."""
    if not pred_answer or not gold_answer:
        return False
    try:
        p = float(pred_answer.strip())
        g = float(gold_answer.strip())
        if g == 0:
            return abs(p - g) < 1e-5
        return abs(p - g) / max(abs(g), 1e-10) < 1e-4
    except (ValueError, TypeError):
        return pred_answer.strip() == gold_answer.strip()


def evaluate_predictions(predictions_path: str) -> dict:
    """
    Evaluate predictions from inference pipeline.
    Returns dict with EA, PA, and detailed per-sample results.
    """
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    total = len(predictions)
    ea_correct = 0
    pa_correct = 0
    valid_programs = 0
    details = []

    for pred in predictions:
        pred_prog = pred.get("predicted_program") or ""
        pred_ans = pred.get("predicted_answer") or ""
        gold_prog = pred.get("gold_program") or ""
        gold_ans = pred.get("gold_answer") or ""

        is_valid = validate_program(pred_prog) if pred_prog else False
        ea_match = answers_match(pred_ans, gold_ans)
        pa_match = programs_match(pred_prog, gold_prog)

        if is_valid:
            valid_programs += 1
        if ea_match:
            ea_correct += 1
        if pa_match:
            pa_correct += 1

        details.append({
            "id": pred.get("id"),
            "ea_match": ea_match,
            "pa_match": pa_match,
            "valid_program": is_valid,
            "confidence": pred.get("confidence", 0),
        })

    results = {
        "total": total,
        "ea_correct": ea_correct,
        "pa_correct": pa_correct,
        "valid_programs": valid_programs,
        "execution_accuracy": ea_correct / total if total else 0,
        "program_accuracy": pa_correct / total if total else 0,
        "valid_rate": valid_programs / total if total else 0,
        "details": details,
    }

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({total} samples)")
    print(f"{'='*60}")
    print(f"  Execution Accuracy (EA): {results['execution_accuracy']:.2%} ({ea_correct}/{total})")
    print(f"  Program Accuracy (PA):   {results['program_accuracy']:.2%} ({pa_correct}/{total})")
    print(f"  Valid Program Rate:      {results['valid_rate']:.2%} ({valid_programs}/{total})")
    print(f"{'='*60}\n")

    return results


def run_evaluation(predictions_path: str, output_path: Optional[str] = None) -> dict:
    """Run evaluation and optionally save results."""
    results = evaluate_predictions(predictions_path)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved → {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_evaluation(args.predictions, args.output)
