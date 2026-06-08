"""
Teacher model inference for Knowledge Distillation.
Generates structured reasoning traces (Chain of Numerical Reasoning)
using a large teacher model, then validates against gold answers.

Key performance features:
- Batched generation: processes BATCH_SIZE samples simultaneously (~4-8x faster)
- Checkpoint/resume: saves progress every CHECKPOINT_EVERY samples so a
  12-hour Kaggle session can continue across multiple runs
- Smart retry: only retries when quality is below "answer_match"
- ETA tracking: prints estimated time remaining

Validation strategy:
1. Best: exact program match → keep trace as-is (highest quality)
2. Good: correct execution answer → keep trace (model found alternative solution)
3. Fallback: use gold SFT data when teacher fails
"""

import re
import json
import time
import concurrent.futures
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

from pipeline.config import PipelineConfig
from pipeline import _load_model_robust, _load_tokenizer_robust
from pipeline.program_executor import execute_program, format_answer, validate_program
from pipeline.evaluate import answers_match, programs_match


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


def validate_output(output_text: str, gold_program: str, gold_answer: str, table: list = None) -> str:
    """
    Validate model output quality.
    Returns: "exact_match", "answer_match", "program_valid", or "invalid"
    """
    pred_program, pred_answer = extract_program_and_answer(output_text)
    if pred_program is None:
        return "invalid"

    if not validate_program(pred_program):
        return "invalid"

    # Check exact program match
    if programs_match(pred_program, gold_program):
        return "exact_match"

    # Check if execution produces correct answer
    result = execute_program(pred_program, table)
    if result is not None:
        formatted = format_answer(result)
        if answers_match(formatted, gold_answer):
            return "answer_match"

    # Program is valid but doesn't match
    if pred_answer and answers_match(pred_answer, gold_answer):
        return "answer_match"

    return "program_valid"


# ── Local Teacher Inference (Batched) ───────────────────────────────

class LocalTeacher:
    """
    Teacher that loads model locally on GPU with batched generation.
    Batching 8+ samples at once gives 4-8x throughput vs sequential.
    """

    def __init__(self, cfg: PipelineConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        model_name = cfg.model.teacher_model
        dtype = torch.float16 if cfg.model.torch_dtype == "float16" else torch.bfloat16
        self.dtype = dtype
        self.device = "cuda"

        print(f"Loading teacher model: {model_name}")
        # Detect transformers API: >=5.x renamed `torch_dtype` → `dtype`
        import inspect as _insp
        from transformers import AutoModelForCausalLM as _ACM_check
        _dtype_key = "dtype" if "dtype" in _insp.signature(_ACM_check.from_pretrained).parameters else "torch_dtype"
        model_kwargs = {
            "trust_remote_code": cfg.model.trust_remote_code,
            _dtype_key: dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
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
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("  flash_attention_2 available — enabled")
            except Exception as _e:
                print(f"  flash_attention_2 not importable ({_e}); falling back to sdpa")
                model_kwargs["attn_implementation"] = "sdpa"

        self.tokenizer = _load_tokenizer_robust(
            model_name, trust_remote_code=cfg.model.trust_remote_code
        )
        try:
            self.model = _load_model_robust(model_name, model_kwargs)
        except Exception as _e:
            if "flash_attention" in str(_e).lower() or "flash_attn" in str(_e).lower():
                print(f"  Teacher load failed with flash-attn ({_e}); retrying with sdpa")
                model_kwargs["attn_implementation"] = "sdpa"
                self.model = _load_model_robust(model_name, model_kwargs)
            else:
                raise
        self.model.eval()

        # Left-pad for batched generation (causal LM requirement)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.temperature = cfg.teacher.temperature
        self.max_retries = cfg.teacher.max_retries
        self.max_length = cfg.model.max_seq_length
        self.max_new_tokens = getattr(cfg.teacher, "max_new_tokens", 512)

    def _build_prompt(self, sample: dict) -> str:
        """Build input text with chat template for one sample."""
        messages = [{"role": "user", "content": sample["prompt"][0]["content"]}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def generate_batch(self, samples: list[dict]) -> list[dict]:
        """
        Generate reasoning traces for a batch of samples simultaneously.
        ~4-8x faster than sequential on RTX 6000 Pro 96GB.
        """
        import torch

        quality_order = {"invalid": 0, "program_valid": 1, "answer_match": 2, "exact_match": 3}

        # Build input texts for all samples
        input_texts = [self._build_prompt(s) for s in samples]

        # Track best result per sample across retries
        best_results = [None] * len(samples)
        best_qualities = ["invalid"] * len(samples)

        # Indices that still need more attempts
        pending = list(range(len(samples)))

        for attempt in range(self.max_retries + 1):
            if not pending:
                break

            # Tokenize the pending batch with left-padding
            batch_texts = [input_texts[i] for i in pending]
            try:
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature if self.temperature > 0 else None,
                        top_p=0.95 if self.temperature > 0 else None,
                        do_sample=self.temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # input length varies per sample due to padding, track per-sample offset
                input_lengths = inputs["input_ids"].shape[1]
                new_tokens_batch = outputs[:, input_lengths:]
                contents = self.tokenizer.batch_decode(new_tokens_batch, skip_special_tokens=True)

            except Exception as e:
                # On batch error, mark all pending as errors and stop
                for idx in pending:
                    if best_results[idx] is None:
                        best_results[idx] = ""
                        best_qualities[idx] = "error"
                break

            # Evaluate quality for each pending sample
            still_pending = []
            for batch_pos, idx in enumerate(pending):
                s = samples[idx]
                content = contents[batch_pos]
                quality = validate_output(
                    content, s["program"], s["answer"], s.get("table")
                )

                if quality_order.get(quality, 0) > quality_order.get(best_qualities[idx], 0):
                    best_qualities[idx] = quality
                    best_results[idx] = content

                # With guided template (gold program embedded), accept "program_valid"
                # since the program is provided as hint — format compliance is what matters.
                # With free-form, only accept "answer_match" or better.
                is_guided = s.get("_guided", False)
                min_quality = "program_valid" if is_guided else "answer_match"
                if quality_order.get(best_qualities[idx], 0) < quality_order[min_quality]:
                    still_pending.append(idx)

            pending = still_pending

        # Build result dicts
        results = []
        for idx, s in enumerate(samples):
            quality = best_qualities[idx]
            # Guided samples: include "program_valid" too — teacher proved format compliance
            is_guided = s.get("_guided", False)
            matched = quality in ("exact_match", "answer_match") or (is_guided and quality == "program_valid")
            results.append({
                **s,
                "content": best_results[idx] or "",
                "reasoning_content": "",
                "matched": matched,
                "match_type": quality,
            })
        return results

    def generate(self, sample: dict) -> dict:
        """Single-sample generate (wraps generate_batch for compatibility)."""
        return self.generate_batch([sample])[0]


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
        best_result = None
        best_quality = "invalid"
        quality_order = {"invalid": 0, "program_valid": 1, "answer_match": 2, "exact_match": 3}

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

                quality = validate_output(
                    content, sample["program"], sample["answer"],
                    sample.get("table")
                )

                if quality_order.get(quality, 0) > quality_order.get(best_quality, 0):
                    best_quality = quality
                    best_result = (content, reasoning)

                if quality == "exact_match":
                    break

            except Exception as e:
                if attempt == self.max_retries and best_result is None:
                    return {
                        **sample, "content": "", "reasoning_content": "",
                        "matched": False, "match_type": "error", "error": str(e)
                    }

        content, reasoning = best_result if best_result else ("", "")
        matched = best_quality in ("exact_match", "answer_match")
        return {
            **sample, "content": content, "reasoning_content": reasoning,
            "matched": matched, "match_type": best_quality,
        }


# ── Checkpoint helpers ───────────────────────────────────────────────

def _save_checkpoint(results: list, checkpoint_path: str) -> None:
    """Atomically save partial results to checkpoint file."""
    tmp = checkpoint_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    Path(tmp).replace(checkpoint_path)


def _load_checkpoint(checkpoint_path: str) -> list:
    """Load partial results from checkpoint, return [] if not found."""
    p = Path(checkpoint_path)
    if not p.exists():
        return []
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Resumed from checkpoint: {len(data)} samples already done")
        return data
    except Exception as e:
        print(f"  Checkpoint load failed ({e}), starting fresh")
        return []


# ── Distillation Pipeline ────────────────────────────────────────────

def run_teacher_distillation(
    cfg: PipelineConfig,
    teacher_input_path: Optional[str] = None,
    batch_size: int = 8,
    checkpoint_every: int = 50,
    resume: bool = True,
    max_runtime_hours: Optional[float] = None,
) -> str:
    """
    Run teacher inference to generate reasoning traces.

    Args:
        cfg: Pipeline config.
        teacher_input_path: Path to teacher_input.json.
        batch_size: Samples per GPU batch (8 is safe for 27B on 96GB).
        checkpoint_every: Save checkpoint every N processed samples.
        resume: If True, load existing checkpoint and skip done samples.

    Returns path to the distilled output file.
    """
    output_dir = Path(cfg.project_root) / cfg.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if teacher_input_path is None:
        teacher_input_path = str(output_dir / "teacher_input.json")

    with open(teacher_input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples for teacher distillation")

    if len(dataset) == 0:
        raise RuntimeError(
            f"teacher_input.json at {teacher_input_path} is empty.\n"
            "This means the data-prep step did not produce any teacher input.\n"
            "Fix: re-run the data-prep cell (Cell 6) before the teacher cell, "
            "ensure the ViNumQA / FinQA dataset inputs are mounted, and that "
            "data/pipeline/teacher_input.json contains non-empty JSON list."
        )

    checkpoint_path = str(output_dir / "teacher_raw_checkpoint.json")
    raw_path = str(output_dir / "teacher_raw_output.json")

    # Resolve effective batch_size and checkpoint_every from config (can be overridden by args)
    effective_batch = getattr(cfg.teacher, "batch_size", batch_size)
    if batch_size != 8:  # caller explicitly passed a non-default
        effective_batch = batch_size
    effective_ckpt = getattr(cfg.teacher, "checkpoint_every", checkpoint_every)
    if checkpoint_every != 50:
        effective_ckpt = checkpoint_every
    batch_size = effective_batch
    checkpoint_every = effective_ckpt
    # Watchdog: arg wins over config; cfg.teacher.max_runtime_hours is the default.
    if max_runtime_hours is None:
        max_runtime_hours = getattr(cfg.teacher, "max_runtime_hours", None)

    # Resume: skip already-processed samples
    if resume:
        done_results = _load_checkpoint(checkpoint_path)
    else:
        done_results = []

    done_ids = {r["id"] for r in done_results}
    remaining = [s for s in dataset if s["id"] not in done_ids]
    print(f"  Already done: {len(done_results)}, Remaining: {len(remaining)}")

    if not remaining:
        print("All samples already processed — loading from checkpoint.")
        results = done_results
    elif cfg.teacher.use_local:
        teacher = LocalTeacher(cfg)
        results = list(done_results)

        t_start = time.time()
        processed = 0

        # Length-sorted batching: bucket samples by prompt length so each batch
        # has minimal padding. Cuts wasted compute on a 27B teacher significantly
        # (1.3–1.7x throughput depending on length variance). Indexing keeps the
        # original order in the saved checkpoint.
        def _approx_len(s):
            try:
                return len(s["prompt"][0]["content"])
            except Exception:
                return 0
        sorted_remaining = sorted(remaining, key=_approx_len)
        batches = [sorted_remaining[i:i+batch_size] for i in range(0, len(sorted_remaining), batch_size)]
        pbar = tqdm(total=len(remaining), desc="Teacher distillation (batched)", initial=0)

        # Watchdog: stop cleanly before Kaggle 12h cap so SFT phase still has time.
        _watchdog_s = (max_runtime_hours * 3600) if max_runtime_hours else None
        if _watchdog_s:
            print(f"  [Distill watchdog] hard cap: {max_runtime_hours:.2f}h")

        watchdog_hit = False
        for batch in batches:
            batch_results = teacher.generate_batch(batch)
            results.extend(batch_results)
            processed += len(batch)
            pbar.update(len(batch))

            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            remaining_count = len(remaining) - processed
            eta_sec = remaining_count / rate if rate > 0 else 0
            eta_h = eta_sec / 3600
            pbar.set_postfix({"rate": f"{rate:.1f}s/s", "ETA": f"{eta_h:.1f}h"})

            if processed % checkpoint_every < batch_size:
                _save_checkpoint(results, checkpoint_path)
                matched_so_far = sum(1 for r in results if r.get("matched", False))
                pbar.write(
                    f"  [Checkpoint] {len(results)} done, "
                    f"{matched_so_far} matched ({matched_so_far/len(results)*100:.1f}%)"
                )

            if _watchdog_s and elapsed >= _watchdog_s:
                pbar.write(
                    f"  [Distill watchdog] elapsed {elapsed/3600:.2f}h ≥ "
                    f"{max_runtime_hours:.2f}h — stopping early. "
                    f"Resume next session via checkpoint."
                )
                watchdog_hit = True
                break

        pbar.close()
        _save_checkpoint(results, checkpoint_path)
        if watchdog_hit:
            print(f"  [Distill watchdog] partial: {processed}/{len(remaining)} samples this session.")

    else:
        teacher = APITeacher(cfg)
        results = list(done_results)
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.teacher.max_workers) as executor:
            futures = {executor.submit(teacher.generate, s): s for s in remaining}
            processed = 0
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(remaining), desc="Teacher distillation (API)"):
                results.append(future.result())
                processed += 1
                if processed % checkpoint_every == 0:
                    _save_checkpoint(results, checkpoint_path)
        _save_checkpoint(results, checkpoint_path)

    # Statistics (guard against empty results to avoid ZeroDivisionError)
    exact_matches  = sum(1 for r in results if r.get("match_type") == "exact_match")
    answer_matches = sum(1 for r in results if r.get("match_type") == "answer_match")
    valid_only     = sum(1 for r in results if r.get("match_type") == "program_valid")
    errors         = sum(1 for r in results if r.get("match_type") == "error")
    invalid        = sum(1 for r in results if r.get("match_type") == "invalid")
    _n = max(len(results), 1)  # avoid div/0 — prints will read 0.0% on empty

    print(f"\nTeacher distillation results:")
    print(f"  Exact match:  {exact_matches}/{len(results)} ({exact_matches/_n*100:.1f}%)")
    print(f"  Answer match: {answer_matches}/{len(results)} ({answer_matches/_n*100:.1f}%)")
    print(f"  Valid only:   {valid_only}/{len(results)}")
    print(f"  Invalid:      {invalid}/{len(results)}")
    print(f"  Errors:       {errors}/{len(results)}")
    guided_valid  = sum(1 for r in results if r.get("match_type") == "program_valid" and r.get("_guided", False))
    total_usable  = exact_matches + answer_matches + guided_valid
    print(f"  Guided valid: {guided_valid}/{len(results)}  (valid program, guided template)")
    print(f"  Total usable: {total_usable}/{len(results)} "
          f"({total_usable/_n*100:.1f}%)")
    if len(results) == 0:
        print("  WARNING: no results — teacher produced nothing. "
              "Check teacher_input.json and run data-prep again if needed.")

    # Convert matched results to SFT format
    sft_distilled = []
    for r in results:
        if r.get("matched", False):
            prompt_content = r["prompt"][0]["content"]
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
                    "match_type": r.get("match_type", "unknown"),
                },
            })

    output_path = str(output_dir / "distilled_sft.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_distilled, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sft_distilled)} distilled samples → {output_path}")

    # Also save full raw output
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Explicit GPU cleanup: the teacher is a LocalTeacher with device_map="auto"
    # which creates accelerate hooks that prevent Python GC from releasing the
    # model. We must manually drop every reference before empty_cache().
    try:
        import gc as _gc
        if "teacher" in locals() and hasattr(locals().get("teacher"), "model"):
            _t = locals()["teacher"]
            try:
                _t.model.cpu()
            except Exception:
                pass
            try:
                del _t.model
            except Exception:
                pass
            try:
                del _t.tokenizer
            except Exception:
                pass
        _gc.collect()
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.synchronize()
            _free, _total = _torch.cuda.mem_get_info()
            print(f"[teacher-cleanup] GPU free={_free/2**30:.1f}GB / {_total/2**30:.1f}GB")
    except Exception as _e:
        print(f"[teacher-cleanup] non-fatal: {_e}")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-profile", default="p100_16gb")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None, help="Teacher input file path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    from pipeline.config import load_config
    cfg = load_config(gpu_profile=args.gpu_profile, config_path=args.config)
    run_teacher_distillation(cfg, args.input, batch_size=args.batch_size, resume=not args.no_resume)
