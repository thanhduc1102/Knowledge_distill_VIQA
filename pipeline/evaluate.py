"""
Evaluation module for Execution Accuracy (EA) and Program Accuracy (PA).
PA uses symbolic evaluation via sympy (following FinQA methodology).
EA uses numeric comparison with floating-point tolerance.
"""

import re
import json
from pathlib import Path
from typing import Optional

from pipeline.program_executor import execute_program, format_answer, validate_program


# ── Program Tokenization (FinQA-compatible) ─────────────────────────

def program_tokenization(original_program: str) -> list:
    """
    Tokenize program string into structured tokens (FinQA format).
    e.g. "divide(914, 391), multiply(#0, const_100)"
    → ['divide(', '914', '391', ')', 'multiply(', '#0', 'const_100', ')', 'EOF']
    """
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program


# ── Number Parsing (FinQA-compatible) ───────────────────────────────

ALL_OPS = ["add", "subtract", "multiply", "divide", "exp", "greater",
           "table_max", "table_min", "table_sum", "table_average"]


def str_to_num(text: str):
    """Convert string to number, handling financial formatting."""
    text = text.replace(",", "")
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num


def process_row(row_in: list):
    """Process table row for numeric operations."""
    row_out = []
    for num in row_in:
        num = num.replace("$", "").strip()
        num = num.split("(")[0].strip()
        num = str_to_num(num)
        if num == "n/a":
            return "n/a"
        row_out.append(num)
    return row_out


# ── Program Execution (FinQA-compatible) ────────────────────────────

def eval_program(program: list, table: list):
    """
    Execute tokenized program against table data.
    Returns (invalid_flag, result).
    """
    invalid_flag = 0
    this_res = "n/a"

    try:
        program = program[:-1]  # remove EOF
        # Check structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in ALL_OPS:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"

        program_str = "|".join(program)
        steps = program_str.split(")")[:-1]

        res_dict = {}

        for ind, step in enumerate(steps):
            step = step.strip()
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break

            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if op in ("add", "subtract", "multiply", "divide", "exp", "greater"):
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    arg1 = str_to_num(arg1)
                    if arg1 == "n/a":
                        invalid_flag = 1
                        break

                if "#" in arg2:
                    arg2 = res_dict[int(arg2.replace("#", ""))]
                else:
                    arg2 = str_to_num(arg2)
                    if arg2 == "n/a":
                        invalid_flag = 1
                        break

                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2
                elif op == "exp":
                    this_res = arg1 ** arg2
                elif op == "greater":
                    this_res = "yes" if arg1 > arg2 else "no"

                res_dict[ind] = this_res

            elif "table" in op:
                table_dict = {}
                for row in table:
                    table_dict[row[0]] = row[1:]

                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break
                    cal_row = table_dict[arg1]
                    num_row = process_row(cal_row)

                if num_row == "n/a":
                    invalid_flag = 1
                    break

                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row)

                res_dict[ind] = this_res

        if this_res != "yes" and this_res != "no" and this_res != "n/a":
            this_res = round(this_res, 5)

    except Exception:
        invalid_flag = 1

    return invalid_flag, this_res


# ── Symbolic Program Comparison (FinQA-compatible with sympy) ───────

def equal_program(program1: list, program2: list) -> bool:
    """
    Check if two programs are symbolically equivalent using sympy.
    program1: gold program (tokenized)
    program2: predicted program (tokenized)
    """
    try:
        from sympy import simplify, sympify
    except ImportError:
        simplify = None
        sympify = None

    sym_map = {}

    prog1 = program1[:-1]  # remove EOF
    prog1_str = "|".join(prog1)
    steps1 = prog1_str.split(")")[:-1]

    sym_ind = 0
    step_dict_1 = {}

    # Build symbolic map from gold program
    for ind, step in enumerate(steps1):
        step = step.strip()
        if len(step.split("(")) > 2:
            return False

        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        step_dict_1[ind] = step

        if "table" in op:
            if step not in sym_map:
                sym_map[step] = "a" + str(sym_ind)
                sym_ind += 1
        else:
            if "#" not in arg1:
                if arg1 not in sym_map:
                    sym_map[arg1] = "a" + str(sym_ind)
                    sym_ind += 1
            if "#" not in arg2:
                if arg2 not in sym_map:
                    sym_map[arg2] = "a" + str(sym_ind)
                    sym_ind += 1

    # Check program 2
    step_dict_2 = {}
    try:
        prog2 = program2[:-1]
        for ind, token in enumerate(prog2):
            if ind % 4 == 0:
                if token.strip("(") not in ALL_OPS:
                    return False
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return False

        prog2_str = "|".join(prog2)
        steps2 = prog2_str.split(")")[:-1]

        for ind, step in enumerate(steps2):
            step = step.strip()
            if len(step.split("(")) > 2:
                return False

            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            step_dict_2[ind] = step

            if "table" in op:
                if step not in sym_map:
                    return False
            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        return False
                else:
                    if int(arg1.strip("#")) >= ind:
                        return False
                if "#" not in arg2:
                    if arg2 not in sym_map:
                        return False
                else:
                    if int(arg2.strip("#")) >= ind:
                        return False
    except Exception:
        return False

    def symbol_recur(step, step_dict):
        step = step.strip()
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        if "table" in op:
            return sym_map[step]

        if "#" in arg1:
            arg1_part = symbol_recur(step_dict[int(arg1.replace("#", ""))], step_dict)
        else:
            arg1_part = sym_map[arg1]

        if "#" in arg2:
            arg2_part = symbol_recur(step_dict[int(arg2.replace("#", ""))], step_dict)
        else:
            arg2_part = sym_map[arg2]

        if op == "add":
            return "( " + arg1_part + " + " + arg2_part + " )"
        elif op == "subtract":
            return "( " + arg1_part + " - " + arg2_part + " )"
        elif op == "multiply":
            return "( " + arg1_part + " * " + arg2_part + " )"
        elif op == "divide":
            return "( " + arg1_part + " / " + arg2_part + " )"
        elif op == "exp":
            return "( " + arg1_part + " ** " + arg2_part + " )"
        elif op == "greater":
            return "( " + arg1_part + " > " + arg2_part + " )"

    def canonical_recur(step, step_dict):
        step = step.strip()
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        if "table" in op:
            return ("table", sym_map[step])

        if "#" in arg1:
            arg1_part = canonical_recur(step_dict[int(arg1.replace("#", ""))], step_dict)
        else:
            arg1_part = ("arg", sym_map[arg1])

        if "#" in arg2:
            arg2_part = canonical_recur(step_dict[int(arg2.replace("#", ""))], step_dict)
        else:
            arg2_part = ("arg", sym_map[arg2])

        if op in {"add", "multiply"}:
            ordered_args = tuple(sorted((arg1_part, arg2_part), key=repr))
            return (op, ordered_args)
        return (op, arg1_part, arg2_part)

    try:
        if canonical_recur(steps1[-1], step_dict_1) == canonical_recur(steps2[-1], step_dict_2):
            return True

        if simplify is None or sympify is None:
            return False

        sym_prog1 = simplify(sympify(symbol_recur(steps1[-1], step_dict_1)))
        sym_prog2 = simplify(sympify(symbol_recur(steps2[-1], step_dict_2)))

        if sym_prog1 == sym_prog2:
            return True

        try:
            return simplify(sym_prog1 - sym_prog2) == 0
        except TypeError:
            return False
    except Exception:
        return False


# ── High-level Evaluation Functions ─────────────────────────────────

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
    Uses FinQA symbolic evaluation with sympy.
    """
    if not pred_program or not gold_program:
        return False

    pred_norm = normalize_program(pred_program)
    gold_norm = normalize_program(gold_program)

    # Direct string match
    if pred_norm == gold_norm:
        return True

    # Tokenize both programs
    try:
        pred_tokens = program_tokenization(pred_norm)
        gold_tokens = program_tokenization(gold_norm)
    except Exception:
        return False

    # Token-level direct match
    if pred_tokens == gold_tokens:
        return True

    # Symbolic equivalence via sympy
    try:
        return equal_program(gold_tokens, pred_tokens)
    except Exception:
        return False


def answers_match(pred_answer: str, gold_answer: str) -> bool:
    """Check if two answers match (with numeric tolerance)."""
    if not pred_answer or not gold_answer:
        return False

    # Handle yes/no boolean answers
    pred_clean = pred_answer.strip().lower()
    gold_clean = gold_answer.strip().lower()
    if gold_clean in ("yes", "no"):
        return pred_clean == gold_clean

    try:
        p = float(pred_clean)
        g = float(gold_clean)
        if g == 0:
            return abs(p - g) < 1e-5
        return abs(p - g) / max(abs(g), 1e-10) < 1e-4
    except (ValueError, TypeError):
        return pred_clean == gold_clean


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
            "predicted_program": pred_prog,
            "predicted_answer": pred_ans,
            "gold_program": gold_prog,
            "gold_answer": gold_ans,
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


def evaluate_against_dataset(
    predictions_path: str,
    dataset_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """
    Evaluate predictions against original dataset (with table data for proper execution).
    This is the most accurate evaluation matching FinQA methodology.
    """
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Build lookup by ID
    data_dict = {sample["id"]: sample for sample in dataset}

    total = len(predictions)
    ea_correct = 0
    pa_correct = 0
    valid_programs = 0
    details = []

    for pred in predictions:
        pred_id = pred.get("id", "")
        pred_prog = pred.get("predicted_program") or ""
        pred_ans = pred.get("predicted_answer") or ""

        # Get gold data from original dataset
        ori = data_dict.get(pred_id, {})
        table = ori.get("table", [])
        gold_program = ori.get("qa", {}).get("program", "")
        gold_answer = ori.get("qa", {}).get("exe_ans", "")

        is_valid = validate_program(pred_prog) if pred_prog else False
        if is_valid:
            valid_programs += 1

        # EA: Check execution accuracy
        ea_match = False
        if pred_prog and is_valid:
            # Execute predicted program against table
            pred_tokens = program_tokenization(normalize_program(pred_prog))
            inv_flag, exe_res = eval_program(pred_tokens, table)
            if inv_flag == 0 and exe_res != "n/a":
                if isinstance(gold_answer, bool):
                    ea_match = (exe_res == ("yes" if gold_answer else "no"))
                elif isinstance(gold_answer, str):
                    ea_match = (str(exe_res) == gold_answer)
                else:
                    ea_match = answers_match(str(exe_res), str(gold_answer))
        if not ea_match and pred_ans:
            ea_match = answers_match(pred_ans, str(gold_answer))

        # PA: Check program accuracy (symbolic)
        pa_match = programs_match(pred_prog, gold_program)

        if ea_match:
            ea_correct += 1
        if pa_match:
            pa_correct += 1

        details.append({
            "id": pred_id,
            "ea_match": ea_match,
            "pa_match": pa_match,
            "valid_program": is_valid,
            "predicted_program": pred_prog,
            "predicted_answer": pred_ans,
            "gold_program": gold_program,
            "gold_answer": str(gold_answer),
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

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved → {output_path}")

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
    parser.add_argument("--dataset", default=None, help="Original dataset for full evaluation")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.dataset:
        evaluate_against_dataset(args.predictions, args.dataset, args.output)
    else:
        run_evaluation(args.predictions, args.output)
