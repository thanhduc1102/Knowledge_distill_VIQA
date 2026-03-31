"""
Teacher model inference for Knowledge Distillation.
Generates structured reasoning traces (Chain of Numerical Reasoning)
using a large teacher model, then validates against gold answers.
Supports both local model loading and API-based inference.
"""

import re
import json
import concurrent.futures
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

from pipeline.config import PipelineConfig
from pipeline.program_executor import execute_program, format_answer


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


def check_output_matches_gold(output_text: str, gold_program: str, gold_answer: str) -> bool:
    """Check if model output matches the gold program and answer."""
    pred_program, pred_answer = extract_program_and_answer(output_text)
    if pred_program is None or pred_answer is None:
        return False
    # Normalize for comparison
    pred_prog_norm = re.sub(r'\s+', ' ', pred_program.strip())
    gold_prog_norm = re.sub(r'\s+', ' ', gold_program.strip())
    return pred_prog_norm == gold_prog_norm and pred_answer.strip() == gold_answer.strip()


# ── API-based Teacher Inference ──────────────────────────────────────

class APITeacher:
    """Teacher that uses an OpenAI-compatible API (e.g., vLLM served model)."""

    def __init__(self, cfg: PipelineConfig):
        from openai import OpenAI
        self.client = OpenAI(base_url=cfg.teacher.base_url, api_key=cfg.teacher.api_key)
        self.model = cfg.model.teacher_model.split("/")[-1]
        self.temperature = cfg.teacher.temperature
        self.top_p = cfg.teacher.top_p
        self.max_retries = cfg.teacher.max_retries

    def generate(self, sample: dict) -> dict:
        """Generate reasoning trace for one sample."""
        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": sample["prompt"][0]["content"]}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                content = completion.choices[0].message.content or ""
                reasoning = getattr(completion.choices[0].message, "reasoning_content", "") or ""

                if check_output_matches_gold(content, sample["program"], sample["answer"]):
                    return {**sample, "content": content, "reasoning_content": reasoning, "matched": True}

                if attempt == self.max_retries:
                    return {**sample, "content": content, "reasoning_content": reasoning, "matched": False}

            except Exception as e:
                if attempt == self.max_retries:
                    return {**sample, "content": "", "reasoning_content": "", "matched": False, "error": str(e)}
        return sample


# ── Local Teacher Inference ──────────────────────────────────────────

class LocalTeacher:
    """Teacher that loads model locally on GPU (for smaller teacher models)."""

    def __init__(self, cfg: PipelineConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        model_name = cfg.model.teacher_model
        dtype = torch.float16 if cfg.model.torch_dtype == "float16" else torch.bfloat16

        print(f"Loading teacher model: {model_name}")
        tokenizer_kwargs = {"trust_remote_code": cfg.model.trust_remote_code}
        model_kwargs = {
            "trust_remote_code": cfg.model.trust_remote_code,
            "torch_dtype": dtype,
            "device_map": "auto",
        }

        if cfg.model.teacher_quantization == "4bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif cfg.model.teacher_quantization == "8bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        if cfg.model.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.temperature = cfg.teacher.temperature
        self.max_retries = cfg.teacher.max_retries
        self.max_length = cfg.model.max_seq_length

    def generate(self, sample: dict) -> dict:
        """Generate reasoning trace for one sample."""
        import torch

        messages = [{"role": "user", "content": sample["prompt"][0]["content"]}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        for attempt in range(self.max_retries + 1):
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        temperature=self.temperature if self.temperature > 0 else None,
                        top_p=0.95 if self.temperature > 0 else None,
                        do_sample=self.temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                if check_output_matches_gold(content, sample["program"], sample["answer"]):
                    return {**sample, "content": content, "reasoning_content": "", "matched": True}

                if attempt == self.max_retries:
                    return {**sample, "content": content, "reasoning_content": "", "matched": False}

            except Exception as e:
                if attempt == self.max_retries:
                    return {**sample, "content": "", "reasoning_content": "", "matched": False, "error": str(e)}
        return sample


# ── Distillation Pipeline ────────────────────────────────────────────

def run_teacher_distillation(cfg: PipelineConfig, teacher_input_path: Optional[str] = None) -> str:
    """
    Run teacher inference to generate reasoning traces.
    Returns path to the distilled output file.
    """
    output_dir = Path(cfg.project_root) / cfg.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if teacher_input_path is None:
        teacher_input_path = str(output_dir / "teacher_input.json")

    with open(teacher_input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples for teacher distillation")

    # Initialize teacher
    if cfg.teacher.use_local:
        teacher = LocalTeacher(cfg)
        # Process sequentially for local model (GPU-bound)
        results = []
        for sample in tqdm(dataset, desc="Teacher distillation"):
            result = teacher.generate(sample)
            results.append(result)
    else:
        teacher = APITeacher(cfg)
        # Process with thread pool for API (I/O-bound)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.teacher.max_workers) as executor:
            futures = [executor.submit(teacher.generate, sample) for sample in dataset]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset), desc="Teacher distillation"):
                results.append(future.result())

    # Statistics
    matched = sum(1 for r in results if r.get("matched", False))
    errors = sum(1 for r in results if "error" in r)
    print(f"\nTeacher distillation results:")
    print(f"  Matched: {matched}/{len(results)} ({matched/len(results)*100:.1f}%)")
    print(f"  Errors:  {errors}/{len(results)}")

    # Convert matched results to SFT format
    sft_distilled = []
    for r in results:
        if r.get("matched", False):
            # Remove sensitive fields before saving to SFT
            prompt_content = r["prompt"][0]["content"]
            # Remove the hint suffix if present (for think-style prompts)
            hint_kw = "\n\nHãy **SỬ DỤNG CHÍNH XÁC** nội dung dưới đây"
            idx = prompt_content.find(hint_kw)
            if idx != -1:
                prompt_content = prompt_content[:idx]

            sft_distilled.append({
                "messages": [
                    {"role": "user", "content": prompt_content},
                    {"role": "assistant", "content": r["content"]},
                ],
                "id": r["id"],
                "metadata": {
                    "table": r["table"],
                    "program": r["program"],
                    "answer": r["answer"],
                },
            })

    output_path = str(output_dir / "distilled_sft.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_distilled, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sft_distilled)} distilled samples → {output_path}")

    # Also save raw results for debugging
    raw_path = str(output_dir / "teacher_raw_output.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-profile", default="p100_16gb")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None, help="Teacher input file path")
    args = parser.parse_args()

    from pipeline.config import load_config
    cfg = load_config(gpu_profile=args.gpu_profile, config_path=args.config)
    run_teacher_distillation(cfg, args.input)
