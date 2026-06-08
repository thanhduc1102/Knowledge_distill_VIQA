"""
Supervised Fine-Tuning (SFT) module for the student model.
Uses PEFT/LoRA for memory-efficient training on P100.
Supports full fine-tuning on larger GPUs.
"""

import gc
import json
import os
import time
import torch
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pipeline.config import PipelineConfig
from pipeline import _load_model_robust, _load_tokenizer_robust


def _print_gpu_mem(tag: str):
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    gb = 1024 ** 3
    print(
        f"[GPU-MEM {tag}] free={free/gb:.1f}GB total={total/gb:.1f}GB "
        f"alloc={alloc/gb:.1f}GB reserved={reserved/gb:.1f}GB"
    )


class SFTDataset(torch.utils.data.Dataset):
    """Dataset for SFT training from JSON conversation format."""

    def __init__(self, data: list, tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = sample["messages"]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # Create labels: mask user tokens, only compute loss on assistant tokens
        input_ids = encoding["input_ids"]
        labels = input_ids.copy()

        # Find where assistant response starts
        user_text = self.tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        user_tokens = self.tokenizer(user_text, truncation=True, max_length=self.max_length)["input_ids"]
        # Mask user portion
        for i in range(min(len(user_tokens), len(labels))):
            labels[i] = -100

        # Safety guard: if all labels are masked (response truncated away), unmask the last
        # N tokens so loss is defined. Prevents NaN when a long prompt fills the entire budget.
        if all(l == -100 for l in labels) and len(labels) > 0:
            n_unmask = min(32, len(labels))
            for i in range(len(labels) - n_unmask, len(labels)):
                labels[i] = input_ids[i]

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


class DataCollatorForSFT:
    """Pad sequences to the same length within a batch."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)

        return {k: torch.tensor(v) for k, v in batch.items()}


def load_model_and_tokenizer(cfg: PipelineConfig):
    """Load student model with optional quantization and LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg.model.student_model
    dtype = torch.float16 if cfg.model.torch_dtype == "float16" else torch.bfloat16

    print(f"Loading student model: {model_name}")

    # Detect transformers API: >=5.x uses `dtype` instead of `torch_dtype`
    import inspect as _insp
    from transformers import AutoModelForCausalLM as _ACM_check
    _dtype_key = "dtype" if "dtype" in _insp.signature(_ACM_check.from_pretrained).parameters else "torch_dtype"

    model_kwargs = {
        "trust_remote_code": cfg.model.trust_remote_code,
        _dtype_key: dtype,
    }

    if cfg.model.student_quantization == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.model.student_quantization == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    if cfg.model.use_flash_attention:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  flash_attention_2 available — enabled")
        except Exception as e:
            print(f"  flash_attention_2 not importable ({e}); falling back to sdpa")
            model_kwargs["attn_implementation"] = "sdpa"

    # Use low_cpu_mem_usage so accelerate streams weights to GPU (no 2x copy).
    model_kwargs["low_cpu_mem_usage"] = True

    _print_gpu_mem("before-student-load")
    tokenizer = _load_tokenizer_robust(
        model_name, trust_remote_code=cfg.model.trust_remote_code
    )
    try:
        model = _load_model_robust(model_name, model_kwargs)
    except Exception as e:
        if "flash_attention" in str(e).lower() or "flash_attn" in str(e).lower():
            print(f"  Student load failed with flash-attn ({e}); retrying with sdpa")
            model_kwargs["attn_implementation"] = "sdpa"
            model = _load_model_robust(model_name, model_kwargs)
        else:
            raise

    # Force model onto GPU — HF from_pretrained w/o device_map leaves it on CPU,
    # which makes Trainer auto-move it but often through CPU staging (slow + peak RAM).
    if torch.cuda.is_available():
        try:
            _dev = next(model.parameters()).device
        except StopIteration:
            _dev = torch.device("cpu")
        if _dev.type != "cuda":
            print(f"  model on {_dev} — moving to cuda")
            model = model.to("cuda")
    _print_gpu_mem("after-student-load")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA if configured
    if cfg.sft.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=cfg.sft.lora_r,
            lora_alpha=cfg.sft.lora_alpha,
            lora_dropout=cfg.sft.lora_dropout,
            target_modules=cfg.sft.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable input gradients for gradient checkpointing compatibility
    if cfg.sft.gradient_checkpointing:
        model.enable_input_require_grads()

    return model, tokenizer


def run_sft_training(
    cfg: PipelineConfig,
    train_path: Optional[str] = None,
    valid_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """
    Run SFT training on the student model.
    Args:
        resume_from_checkpoint: Path to a HuggingFace Trainer checkpoint dir to resume from.
    Returns path to the trained model checkpoint.
    """
    from transformers import TrainingArguments, Trainer

    output_dir = Path(cfg.project_root) / cfg.sft.output_dir
    data_dir = Path(cfg.project_root) / cfg.data.output_dir

    # Load data
    if train_path is None:
        # Prefer distilled data if available, fall back to gold SFT data
        distilled_path = data_dir / "distilled_sft.json"
        gold_path = data_dir / "sft_train.json"
        train_path = str(distilled_path) if distilled_path.exists() else str(gold_path)

    if valid_path is None:
        valid_path = str(data_dir / "sft_valid.json")

    print(f"SFT train data: {train_path}")
    print(f"SFT valid data: {valid_path}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_data = json.load(f)

    # Optional dataset subsampling — lets a Kaggle 12h session finish without cutting quality.
    _max_train = getattr(cfg.sft, "max_train_samples", None)
    _max_valid = getattr(cfg.sft, "max_valid_samples", None)
    if _max_train and len(train_data) > _max_train:
        print(f"Capping train: {len(train_data)} → {_max_train} (cfg.sft.max_train_samples)")
        train_data = train_data[:_max_train]
    if _max_valid and len(valid_data) > _max_valid:
        print(f"Capping valid: {len(valid_data)} → {_max_valid} (cfg.sft.max_valid_samples)")
        valid_data = valid_data[:_max_valid]

    print(f"Train samples: {len(train_data)}, Valid samples: {len(valid_data)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Use SFT-specific max_seq_length (shorter than model context; saves logits memory)
    sft_max_len = cfg.sft.max_seq_length
    print(f"SFT max_seq_length: {sft_max_len}  (model context: {cfg.model.max_seq_length})")

    # Create datasets
    train_dataset = SFTDataset(train_data, tokenizer, sft_max_len)
    valid_dataset = SFTDataset(valid_data, tokenizer, sft_max_len)
    collator = DataCollatorForSFT(tokenizer, sft_max_len)

    # Guard: avoid eval_strategy="steps" when we have fewer training steps than eval_steps
    steps_per_epoch = max(
        1,
        len(train_dataset)
        // max(cfg.sft.per_device_train_batch_size * cfg.sft.gradient_accumulation_steps, 1),
    )
    total_steps = steps_per_epoch * max(cfg.sft.num_epochs, 1)
    eval_steps = min(cfg.sft.eval_steps, max(1, total_steps))
    save_steps = min(cfg.sft.save_steps, max(1, total_steps))
    # load_best_model_at_end requires save_steps to be a multiple of eval_steps
    if save_steps % eval_steps != 0:
        save_steps = eval_steps
    can_eval = len(valid_dataset) > 0 and total_steps >= eval_steps

    # Training arguments
    ta_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.sft.num_epochs,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        learning_rate=cfg.sft.learning_rate,
        warmup_ratio=cfg.sft.warmup_ratio,
        weight_decay=cfg.sft.weight_decay,
        lr_scheduler_type=cfg.sft.lr_scheduler_type,
        logging_steps=cfg.sft.logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=cfg.sft.save_total_limit,
        gradient_checkpointing=cfg.sft.gradient_checkpointing,
        max_grad_norm=cfg.sft.max_grad_norm,
        bf16=cfg.sft.bf16,
        fp16=cfg.sft.fp16,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=cfg.seed,
        report_to="none",
    )
    if can_eval:
        ta_kwargs.update(
            eval_strategy="steps",
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        ta_kwargs.update(eval_strategy="no", load_best_model_at_end=False)

    # Transformers >=5 renamed evaluation_strategy → eval_strategy. Detect and retry on older APIs.
    try:
        training_args = TrainingArguments(**ta_kwargs)
    except TypeError:
        if "eval_strategy" in ta_kwargs:
            ta_kwargs["evaluation_strategy"] = ta_kwargs.pop("eval_strategy")
        training_args = TrainingArguments(**ta_kwargs)

    from transformers import TrainerCallback

    class _EmptyCacheCb(TrainerCallback):
        """Reclaim cached-but-unused VRAM periodically."""
        def __init__(self, every: int = 25):
            self.every = every
        def on_step_end(self, args, state, control, **_):
            if state.global_step and state.global_step % self.every == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        def on_evaluate(self, args, state, control, **_):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    class _RuntimeWatchdogCb(TrainerCallback):
        """Stop cleanly once wall-clock exceeds max_runtime_hours (Kaggle session fit)."""
        def __init__(self, max_hours: Optional[float]):
            self.max_seconds = max_hours * 3600 if max_hours else None
            self._t0 = None
            self._stopped = False
        def on_train_begin(self, args, state, control, **_):
            self._t0 = time.time()
            if self.max_seconds:
                print(f"  [Watchdog] Hard limit: {self.max_seconds/3600:.2f}h")
        def on_step_end(self, args, state, control, **_):
            if self._stopped or self.max_seconds is None or self._t0 is None:
                return
            elapsed = time.time() - self._t0
            if elapsed >= self.max_seconds:
                print(
                    f"  [Watchdog] Elapsed {elapsed/3600:.2f}h ≥ limit "
                    f"{self.max_seconds/3600:.2f}h — requesting clean stop & save."
                )
                control.should_training_stop = True
                control.should_save = True
                self._stopped = True

    class _MirrorSaveCb(TrainerCallback):
        """Copy the latest checkpoint dir to a persistent location (e.g. /kaggle/working).
        Trainer's save_total_limit can delete old checkpoints; mirror keeps the most recent
        few so we still have an adapter if the kernel dies mid-epoch."""
        def __init__(self, mirror_dir: Optional[str], keep_last: int = 2):
            self.mirror_dir = Path(mirror_dir) if mirror_dir else None
            self.keep_last = keep_last
        def on_save(self, args, state, control, **_):
            if self.mirror_dir is None:
                return
            import shutil
            src_dir = Path(args.output_dir)
            ckpts = sorted(
                [p for p in src_dir.glob("checkpoint-*") if p.is_dir()],
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
            )
            if not ckpts:
                return
            latest = ckpts[-1]
            self.mirror_dir.mkdir(parents=True, exist_ok=True)
            dst = self.mirror_dir / latest.name
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(latest, dst)
                print(f"  [Mirror] {latest.name} → {dst}")
            except Exception as e:
                print(f"  [Mirror] copy failed: {e}")
            # Trim mirror dir to keep_last
            existing = sorted(
                [p for p in self.mirror_dir.glob("checkpoint-*") if p.is_dir()],
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
            )
            for old in existing[:-self.keep_last]:
                try:
                    shutil.rmtree(old)
                except Exception:
                    pass

    class _ProgressLogCb(TrainerCallback):
        """Print plaintext progress every N steps — visible in Kaggle even when HTML widget isn't."""
        def __init__(self, log_every: int = 25, total_steps: int = 0):
            self.log_every = log_every
            self.total_steps = total_steps
            self._t0 = None
        def on_train_begin(self, args, state, control, **_):
            self._t0 = time.time()
            print(f"  [SFT] Training started — {self.total_steps} total steps")
        def on_step_end(self, args, state, control, **_):
            step = state.global_step
            if step and step % self.log_every == 0 and self._t0 is not None:
                elapsed = time.time() - self._t0
                eta = (self.total_steps - step) / (step / elapsed) if step else 0
                loss_entry = next(
                    (e for e in reversed(state.log_history) if "loss" in e), {}
                )
                loss_val = loss_entry.get("loss", float("nan"))
                gnorm = loss_entry.get("grad_norm", 0.0)
                print(
                    f"  [SFT step {step}/{self.total_steps}]  "
                    f"loss={loss_val:.4f}  grad_norm={gnorm:.3f}  "
                    f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.0f}min"
                )

    _callbacks = [
        _EmptyCacheCb(every=25),
        _ProgressLogCb(log_every=25, total_steps=total_steps),
        _RuntimeWatchdogCb(max_hours=getattr(cfg.sft, "max_runtime_hours", None)),
        _MirrorSaveCb(mirror_dir=getattr(cfg.sft, "mirror_save_dir", None)),
    ]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if can_eval else None,
        data_collator=collator,
        callbacks=_callbacks,
    )

    print(f"\nStarting SFT training...")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {cfg.sft.num_epochs}")
    print(f"  Steps/epoch (approx): {steps_per_epoch}  | total: {total_steps}")
    print(f"  Save every: {save_steps}  |  Eval every: {eval_steps if can_eval else 'disabled'}")
    print(f"  Effective batch: {cfg.sft.per_device_train_batch_size * cfg.sft.gradient_accumulation_steps}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _print_gpu_mem("before-train")

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except torch.cuda.OutOfMemoryError as e:
        _print_gpu_mem("OOM")
        print(
            "\n[OOM] SFT ran out of GPU memory. Likely causes:\n"
            "  1. Another process (e.g. teacher) is still holding GPU memory.\n"
            "     → Ensure SFT runs in a fresh Python process (the notebook now launches it via subprocess).\n"
            "  2. per_device_train_batch_size too large — try halving it in config.\n"
            "  3. max_seq_length too large — reduce cfg.sft.max_seq_length.\n"
        )
        raise
    _print_gpu_mem("after-train")

    # Save final model (LoRA adapter if PEFT is in use)
    final_dir = str(output_dir / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSFT model saved → {final_dir}")

    # Mirror the final adapter too — Kaggle wipes the working dir on fail but /kaggle/working survives.
    _mirror = getattr(cfg.sft, "mirror_save_dir", None)
    if _mirror:
        import shutil
        mirror_final = Path(_mirror) / "final"
        try:
            if mirror_final.exists():
                shutil.rmtree(mirror_final)
            shutil.copytree(final_dir, mirror_final)
            print(f"SFT final mirrored → {mirror_final}")
        except Exception as e:
            print(f"[Mirror] final copy failed: {e}")

    return final_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-profile", default="p100_16gb")
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--valid-data", default=None)
    parser.add_argument("--manifest-out", default=None,
                        help="If set, write JSON with {final_dir, train_path, valid_path} here.")
    args = parser.parse_args()

    from pipeline.config import load_config
    cfg = load_config(gpu_profile=args.gpu_profile, config_path=args.config)
    final_dir = run_sft_training(cfg, args.train_data, args.valid_data)

    if args.manifest_out:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest_out, "w", encoding="utf-8") as f:
            json.dump({
                "final_dir": final_dir,
                "train_path": args.train_data,
                "valid_path": args.valid_data,
            }, f, ensure_ascii=False, indent=2)
        print(f"SFT manifest written: {args.manifest_out}")
