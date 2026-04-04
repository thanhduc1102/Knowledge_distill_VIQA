"""
Multi-path inference with majority voting for final answer selection.
Generates N candidate programs, executes them, and selects the most common answer.
"""

import re
import json
import torch
from pathlib import Path
from typing import Optional
from collections import Counter
from tqdm import tqdm

from pipeline.config import PipelineConfig
from pipeline import _load_model_robust, _load_tokenizer_robust
from pipeline.program_executor import execute_program, format_answer, validate_program


def extract_program_and_answer(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract program and answer from model output."""
    program_matches = list(re.finditer(
        r"\*\*Chương trình tính toán:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text
    ))
    program = program_matches[-1].group(1).strip() if program_matches else None

    answer_matches = list(re.finditer(
        r"\*\*Đáp án cuối cùng:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text
    ))
    answer = answer_matches[-1].group(1).strip() if answer_matches else None

    return program, answer


# ── Local Inference ──────────────────────────────────────────────────

class LocalInferenceEngine:
    """Inference engine using a locally loaded model."""

    def __init__(self, model_path: str, cfg: PipelineConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if cfg.model.torch_dtype == "float16" else torch.bfloat16
        model_kwargs = {
            "trust_remote_code": cfg.model.trust_remote_code,
            "torch_dtype": dtype,
            "device_map": "auto",
        }
        if cfg.model.student_quantization == "4bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        if cfg.model.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.tokenizer = _load_tokenizer_robust(
            model_path, trust_remote_code=cfg.model.trust_remote_code
        )

        # Try loading as PEFT model first (LoRA), fall back to full model
        try:
            from peft import PeftModel
            base_model = _load_model_robust(cfg.model.student_model, model_kwargs)
            self.model = PeftModel.from_pretrained(base_model, model_path)
        except Exception:
            self.model = _load_model_robust(model_path, model_kwargs)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_new_tokens = cfg.inference.max_new_tokens

    def generate_candidates(
        self, prompt: str, n: int, temperature: float = 0.7, top_p: float = 0.95
    ) -> list[str]:
        """Generate N candidate outputs for a prompt."""
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        candidates = []
        for _ in range(n):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            candidates.append(text)

        return candidates


# ── API Inference ────────────────────────────────────────────────────

class APIInferenceEngine:
    """Inference engine using OpenAI-compatible API."""

    def __init__(self, cfg: PipelineConfig):
        from openai import OpenAI
        self.client = OpenAI(base_url=cfg.inference.base_url, api_key="no-need")
        self.model = cfg.model.student_model.split("/")[-1]
        self.max_new_tokens = cfg.inference.max_new_tokens

    def generate_candidates(
        self, prompt: str, n: int, temperature: float = 0.7, top_p: float = 0.95
    ) -> list[str]:
        candidates = []
        for _ in range(n):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=self.max_new_tokens,
                )
                content = completion.choices[0].message.content or ""
                candidates.append(content)
            except Exception as e:
                print(f"API error: {e}")
                candidates.append("")
        return candidates


# ── Majority Voting ──────────────────────────────────────────────────

def majority_vote(candidates: list[str], table: Optional[list] = None) -> dict:
    """
    Apply majority voting on N candidate outputs.
    Returns the most common answer and its program.
    """
    answers = []
    programs = []

    for text in candidates:
        program, answer = extract_program_and_answer(text)

        if program and validate_program(program):
            # Try to execute for a more reliable answer
            result = execute_program(program, table)
            if result is not None:
                formatted = format_answer(result)
                answers.append(formatted)
                programs.append(program)
            elif answer:
                answers.append(answer.strip())
                programs.append(program)
        elif answer:
            answers.append(answer.strip())
            programs.append(program or "")

    if not answers:
        return {"program": None, "answer": None, "confidence": 0.0, "num_valid": 0}

    # Count answer frequencies
    counter = Counter(answers)
    best_answer, best_count = counter.most_common(1)[0]

    # Find the corresponding program for the best answer
    best_program = None
    for ans, prog in zip(answers, programs):
        if ans == best_answer:
            best_program = prog
            break

    return {
        "program": best_program,
        "answer": best_answer,
        "confidence": best_count / len(candidates),
        "num_valid": len(answers),
        "num_candidates": len(candidates),
    }


# ── Full Inference Pipeline ──────────────────────────────────────────

def run_inference(
    cfg: PipelineConfig,
    model_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
) -> str:
    """
    Run multi-path inference with majority voting on test data.
    Returns path to predictions file.
    """
    output_dir = Path(cfg.project_root) / cfg.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model path
    if model_path is None:
        grpo_dir = Path(cfg.project_root) / cfg.grpo.output_dir / "final"
        sft_dir = Path(cfg.project_root) / cfg.sft.output_dir / "final"
        if grpo_dir.exists():
            model_path = str(grpo_dir)
        elif sft_dir.exists():
            model_path = str(sft_dir)
        else:
            model_path = cfg.model.student_model

    print(f"Inference model: {model_path}")

    # Load test data
    if test_data_path is None:
        test_data_path = str(output_dir / "sft_valid.json")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Test samples: {len(test_data)}")

    # Initialize engine
    if cfg.inference.use_local:
        engine = LocalInferenceEngine(model_path, cfg)
    else:
        engine = APIInferenceEngine(cfg)

    # Run inference
    results = []
    n = cfg.inference.num_candidates

    for sample in tqdm(test_data, desc="Inference"):
        prompt = sample["messages"][0]["content"]
        table = sample.get("metadata", {}).get("table")

        candidates = engine.generate_candidates(
            prompt, n, cfg.inference.temperature, cfg.inference.top_p
        )

        vote_result = majority_vote(candidates, table)

        result_entry = {
            "id": sample["id"],
            "predicted_program": vote_result["program"],
            "predicted_answer": vote_result["answer"],
            "confidence": vote_result["confidence"],
            "num_valid": vote_result["num_valid"],
            "gold_program": sample.get("metadata", {}).get("program"),
            "gold_answer": sample.get("metadata", {}).get("answer"),
            "all_candidates": candidates,  # Save all raw outputs
        }
        results.append(result_entry)

    # Save full results with candidates
    output_path = str(output_dir / "predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} predictions → {output_path}")

    # Save compact version without candidates (for evaluation)
    compact = [{k: v for k, v in r.items() if k != "all_candidates"} for r in results]
    compact_path = str(output_dir / "predictions_compact.json")
    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-profile", default="p100_16gb")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--test-data", default=None)
    args = parser.parse_args()

    from pipeline.config import load_config
    cfg = load_config(gpu_profile=args.gpu_profile, config_path=args.config)
    run_inference(cfg, args.model, args.test_data)
