"""
GRPO (Group Relative Policy Optimization) training with PCPO reward.
Uses TRL's GRPOTrainer when available, with a fallback manual implementation
for environments without TRL.
"""

import json
import re
import math
import torch
from pathlib import Path
from typing import Optional

from pipeline.config import PipelineConfig
from pipeline import _load_model_robust, _load_tokenizer_robust
from pipeline.reward import compute_grpo_reward


# ── TRL-based GRPO Training ─────────────────────────────────────────

def run_grpo_trl(cfg: PipelineConfig, sft_model_path: Optional[str] = None) -> str:
    """Run GRPO training using TRL's GRPOTrainer."""
    from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType
    from datasets import Dataset
    import pandas as pd

    output_dir = Path(cfg.project_root) / cfg.grpo.output_dir
    data_dir = Path(cfg.project_root) / cfg.data.output_dir

    # Determine model path
    if sft_model_path is None:
        sft_dir = Path(cfg.project_root) / cfg.sft.output_dir / "final"
        if sft_dir.exists():
            sft_model_path = str(sft_dir)
        else:
            sft_model_path = cfg.model.student_model
            print(f"No SFT checkpoint found, using base model: {sft_model_path}")

    print(f"GRPO starting from: {sft_model_path}")
    dtype = torch.float16 if cfg.model.torch_dtype == "float16" else torch.bfloat16

    # Load model
    model_kwargs = {
        "trust_remote_code": cfg.model.trust_remote_code,
        "torch_dtype": dtype,
    }
    if cfg.model.student_quantization == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    if cfg.model.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = _load_tokenizer_robust(sft_model_path, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GRPO data
    grpo_train_path = str(data_dir / "grpo_train.parquet")
    grpo_valid_path = str(data_dir / "grpo_valid.parquet")
    train_df = pd.read_parquet(grpo_train_path)
    valid_df = pd.read_parquet(grpo_valid_path)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    print(f"GRPO train: {len(train_dataset)}, valid: {len(valid_dataset)}")

    # Define reward function
    weights = cfg.grpo.reward_weights

    def reward_fn(completions, **kwargs):
        """Compute PCPO rewards for a batch of completions."""
        ground_truths = kwargs.get("ground_truth", [{}] * len(completions))
        rewards = []
        for completion, gt in zip(completions, ground_truths):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            r = compute_grpo_reward(
                text, gt,
                reward_type=cfg.grpo.reward_type,
                reward_weights=weights,
            )
            rewards.append(r)
        return rewards

    # LoRA config
    peft_config = None
    if cfg.grpo.use_lora:
        peft_config = LoraConfig(
            r=cfg.grpo.lora_r,
            lora_alpha=cfg.grpo.lora_alpha,
            lora_dropout=cfg.grpo.lora_dropout,
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    # GRPO config
    training_config = TRLGRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.grpo.num_epochs,
        per_device_train_batch_size=cfg.grpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.grpo.gradient_accumulation_steps,
        learning_rate=cfg.grpo.learning_rate,
        num_generations=cfg.grpo.num_generations,
        max_completion_length=cfg.grpo.max_completion_length,
        bf16=cfg.grpo.bf16,
        fp16=cfg.grpo.fp16,
        gradient_checkpointing=cfg.grpo.gradient_checkpointing,
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        seed=cfg.seed,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=sft_model_path,
        reward_funcs=reward_fn,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    final_dir = str(output_dir / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"GRPO model saved → {final_dir}")

    return final_dir


# ── Manual GRPO Fallback ─────────────────────────────────────────────

def run_grpo_manual(cfg: PipelineConfig, sft_model_path: Optional[str] = None) -> str:
    """
    Manual GRPO implementation for environments without TRL.
    Simplified version: generate N completions, compute rewards, update with REINFORCE.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(cfg.project_root) / cfg.grpo.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.project_root) / cfg.data.output_dir

    if sft_model_path is None:
        sft_dir = Path(cfg.project_root) / cfg.sft.output_dir / "final"
        sft_model_path = str(sft_dir) if sft_dir.exists() else cfg.model.student_model

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

    tokenizer = _load_tokenizer_robust(sft_model_path, trust_remote_code=cfg.model.trust_remote_code)
    model = _load_model_robust(sft_model_path, model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    if cfg.grpo.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=cfg.grpo.lora_r, lora_alpha=cfg.grpo.lora_alpha,
            lora_dropout=cfg.grpo.lora_dropout,
            target_modules="all-linear", task_type=TaskType.CAUSAL_LM, bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable input gradients for gradient checkpointing compatibility
    if cfg.grpo.gradient_checkpointing:
        model.enable_input_require_grads()

    # Load GRPO training data
    with open(str(data_dir / "grpo_train.parquet"), "rb") as f:
        import pandas as pd
        train_df = pd.read_parquet(f)
    train_data = train_df.to_dict("records")
    print(f"GRPO train samples: {len(train_data)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.grpo.learning_rate, weight_decay=0.01,
    )

    weights = cfg.grpo.reward_weights
    num_generations = cfg.grpo.num_generations
    max_completion = cfg.grpo.max_completion_length
    grad_accum = cfg.grpo.gradient_accumulation_steps

    model.train()
    global_step = 0
    total_reward = 0.0

    for epoch in range(cfg.grpo.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{cfg.grpo.num_epochs} ===")

        for batch_idx, sample in enumerate(train_data):
            prompt_messages = sample["prompt"]
            if isinstance(prompt_messages, str):
                import ast
                prompt_messages = ast.literal_eval(prompt_messages)
            ground_truth = sample["reward_model"]
            if isinstance(ground_truth, str):
                import ast
                ground_truth = ast.literal_eval(ground_truth)
            gt = ground_truth.get("ground_truth", ground_truth)

            # Tokenize prompt
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=cfg.model.max_seq_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate N completions
            all_rewards = []
            all_log_probs = []

            for _ in range(num_generations):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=max_completion,
                        temperature=0.7, top_p=0.95, do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        return_dict_in_generate=True, output_scores=True,
                    )

                gen_ids = outputs.sequences[0][inputs["input_ids"].shape[-1]:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                # Compute reward
                reward = compute_grpo_reward(
                    gen_text, gt,
                    reward_type=cfg.grpo.reward_type,
                    reward_weights=weights,
                )
                all_rewards.append(reward)

                # Compute log probability of generated sequence (grad enabled for REINFORCE)
                full_ids = outputs.sequences[:, :inputs["input_ids"].shape[-1] + len(gen_ids)]
                model_out = model(full_ids)
                logits = model_out.logits[:, inputs["input_ids"].shape[-1]-1:-1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, gen_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                seq_log_prob = token_log_probs.sum()
                all_log_probs.append(seq_log_prob)

            # Group Relative: normalize rewards within group
            rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
            if rewards_tensor.std() > 0:
                advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            else:
                advantages = rewards_tensor - rewards_tensor.mean()

            # REINFORCE loss: -advantage * log_prob
            losses = [-adv.to(model.device) * lp for adv, lp in zip(advantages, all_log_probs)]
            loss = sum(losses) / num_generations / grad_accum

            loss.backward()
            total_reward += rewards_tensor.mean().item()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_reward = total_reward / (batch_idx + 1)
                    print(f"  Step {global_step} | Avg Reward: {avg_reward:.4f} | Loss: {loss.item():.4f}")

            # Save checkpoints
            if global_step > 0 and global_step % 100 == 0:
                ckpt_dir = str(output_dir / f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    # Save final
    final_dir = str(output_dir / "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"GRPO model saved → {final_dir}")
    return final_dir


# ── Entry Point ──────────────────────────────────────────────────────

def run_grpo_training(cfg: PipelineConfig, sft_model_path: Optional[str] = None) -> str:
    """Run GRPO training, using TRL if available, otherwise manual."""
    try:
        import trl
        print(f"TRL version {trl.__version__} found, using GRPOTrainer")
        return run_grpo_trl(cfg, sft_model_path)
    except ImportError:
        print("TRL not installed, using manual GRPO implementation")
        return run_grpo_manual(cfg, sft_model_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-profile", default="p100_16gb")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sft-model", default=None)
    args = parser.parse_args()

    from pipeline.config import load_config
    cfg = load_config(gpu_profile=args.gpu_profile, config_path=args.config)
    run_grpo_training(cfg, args.sft_model)
