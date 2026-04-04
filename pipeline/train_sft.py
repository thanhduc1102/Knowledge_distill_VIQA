"""
Supervised Fine-Tuning (SFT) module for the student model.
Uses PEFT/LoRA for memory-efficient training on P100.
Supports full fine-tuning on larger GPUs.
"""

import json
import torch
from pathlib import Path
from typing import Optional

from pipeline.config import PipelineConfig
from pipeline import _load_model_robust, _load_tokenizer_robust


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

    model_kwargs = {
        "trust_remote_code": cfg.model.trust_remote_code,
        "torch_dtype": dtype,
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
        model_kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = _load_tokenizer_robust(
        model_name, trust_remote_code=cfg.model.trust_remote_code
    )
    model = _load_model_robust(model_name, model_kwargs)

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

    print(f"Train samples: {len(train_data)}, Valid samples: {len(valid_data)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Create datasets
    train_dataset = SFTDataset(train_data, tokenizer, cfg.model.max_seq_length)
    valid_dataset = SFTDataset(valid_data, tokenizer, cfg.model.max_seq_length)
    collator = DataCollatorForSFT(tokenizer, cfg.model.max_seq_length)

    # Training arguments
    training_args = TrainingArguments(
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
        save_steps=cfg.sft.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.sft.eval_steps,
        save_total_limit=cfg.sft.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=cfg.sft.gradient_checkpointing,
        max_grad_norm=cfg.sft.max_grad_norm,
        bf16=cfg.sft.bf16,
        fp16=cfg.sft.fp16,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=cfg.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
    )

    print(f"\nStarting SFT training...")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {cfg.sft.num_epochs}")
    print(f"  Effective batch: {cfg.sft.per_device_train_batch_size * cfg.sft.gradient_accumulation_steps}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    final_dir = str(output_dir / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSFT model saved → {final_dir}")

    return final_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-profile", default="p100_16gb")
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--valid-data", default=None)
    args = parser.parse_args()

    from pipeline.config import load_config
    cfg = load_config(gpu_profile=args.gpu_profile, config_path=args.config)
    run_sft_training(cfg, args.train_data, args.valid_data)
