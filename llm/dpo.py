"""Minimal DPO training example: Qwen 0.5B on UltraFeedback preferences.

We fine-tune `Qwen/Qwen2.5-0.5B-Instruct` with TRL's `DPOTrainer` on the
`trl-lib/ultrafeedback_binarized` preference dataset. Each row contains a
`chosen` and a `rejected` assistant response to the same user prompt
(implicit-prompt conversational format -- TRL extracts the prompt itself).

LoRA + gradient checkpointing + an implicit reference model (the same base
model with the LoRA adapters disabled) keep this comfortably inside ~6GB.

Run:

    python llm/dpo.py
"""

from __future__ import annotations

from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "trl-lib/ultrafeedback_binarized"
OUTPUT_DIR = "outputs/dpo_qwen05b"


def build_dataset(split: str, n: Optional[int] = None):
    """Load a split of the UltraFeedback preference dataset."""
    ds = load_dataset(DATASET_NAME, split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return ds


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(split="train", n=2048)
    eval_ds = build_dataset(split="test", n=64)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # LoRA on the policy. With `peft_config` set and `ref_model=None`, the
    # trainer reuses the same base model (with adapters disabled) as the
    # frozen reference -- so we don't pay GPU memory for a second model.
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    config = DPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        beta=0.1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_length=512,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        eval_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        report_to=[],
        seed=42,
    )

    trainer = DPOTrainer(
        model=MODEL_NAME,
        ref_model=None,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
