"""Minimal GRPO training example: a small LLM on GSM8K math reasoning.

We fine-tune `Qwen/Qwen2.5-0.5B-Instruct` in two stages:

  1. **SFT warm-up** (`SFTTrainer`): teach the required output format by supervising
     on GSM8K solutions wrapped in `<think>...</think>` and
     `<answer>...</answer>`.
  2. **GRPO** (`GRPOTrainer`): optimize correctness and format with verifiable rewards.

Two reward functions are used in the GRPO stage:

  * `correctness_reward`: +1.0 if the predicted number matches the GSM8K gold
    answer, else 0.0.
  * `format_reward`: +0.5 if the output contains the required tag structure.

Run:

    python llm/grpo.py

The defaults below are tuned for a single ~6GB GPU: we use LoRA (so we don't
pay Adam state on the full base model), enable gradient checkpointing, and
keep batch / completion length small. If you have more memory, raise
`per_device_train_batch_size`, `num_generations`, and `max_completion_length`.
"""

from __future__ import annotations

import re
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "openai/gsm8k"
OUTPUT_DIR = "outputs/grpo_gsm8k"
USE_SFT_WARMUP = True
SYSTEM_PROMPT = (
    """
    You are a careful math tutor. Solve the problem step by step.

    ** FORMAT RULES **

    - Put your reasoning between <think> and </think>
    - Put the final numeric answer (digits only) between <answer> and </answer>

    ** EXAMPLE **

    Question: A store has 48 apples. They sell 3/8 of them. How many remain?
    <think>
    I need to find 3/8 of 48, then subtract.

    3/8 of 48 is 18.
    Subtracting 18 from 48 gives 30.
    </think>
    <answer>30</answer>
    """
)


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)")
_GSM8K_MARKER_RE = re.compile(r"<<[^>]+>>")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)


def _extract_gold(answer_field: str) -> Optional[str]:
    """Extracts the numeric answer from the GSM8K answer field.
    
    GSM8K answers end with `#### <number>`."""
    m = _GOLD_RE.search(answer_field)
    return m.group(1).replace(",", "").strip() if m else None


def _completion_text(completion: str | list) -> str:
    """Normalize GRPO completions to plain text (string or conversational list)."""
    if isinstance(completion, str):
        return completion
    parts: List[str] = []
    for message in completion:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts)


def _extract_pred(completion: str | list) -> Optional[str]:
    """Lenient extraction.

    Look for the last number inside <answer>...</answer>. If no tag exists
    (e.g. completion was truncated), fall back to the last number anywhere
    in the text. This makes the correctness signal much denser, which is
    important early in GRPO when most generations are messy.
    """
    text = _completion_text(completion)
    tag = _ANSWER_TAG_RE.search(text)
    search_in = tag.group(1) if tag else text
    matches = _NUMBER_RE.findall(search_in)
    return matches[-1].replace(",", "").strip() if matches else None


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _format_sft_completion(answer_field: str) -> Optional[str]:
    """Wrap a GSM8K solution in the tags GRPO expects."""
    gold = _extract_gold(answer_field)
    if gold is None:
        return None
    reasoning = _GOLD_RE.sub("", answer_field).strip()
    reasoning = _GSM8K_MARKER_RE.sub("", reasoning).strip()
    return (
        f"<think>\n{reasoning}\n</think>\n"
        f"<answer>{gold}</answer>"
    )


def build_dataset(split: str, n: Optional[int] = None):
    """Load GSM8K and format rows for both SFT and GRPO.

    Columns:
      * `prompt`: system + user messages (conversational; used by SFT and GRPO)
      * `completion`: tagged assistant solution (SFT only)
      * `ground_truth`: numeric gold answer (GRPO rewards)

    SFT uses conversational `prompt` / `completion` rather than `messages`
    with `assistant_only_loss=True`, because Qwen's chat template does not
    include the `{% generation %}` keyword needed for assistant token masks.
    """
    ds = load_dataset(DATASET_NAME, "main", split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    def _format(example):
        gold = _extract_gold(example["answer"])
        completion = _format_sft_completion(example["answer"])
        if gold is None or completion is None:
            return {"prompt": [], "completion": [], "ground_truth": ""}
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "completion": [{"role": "assistant", "content": completion}],
            "ground_truth": gold,
        }

    return ds.map(_format, remove_columns=ds.column_names).filter(lambda x: len(x["prompt"]) > 0)


# ---------------------------------------------------------------------------
# Reward functions (TRL passes prompts/completions/dataset columns as kwargs)
# ---------------------------------------------------------------------------

def correctness_reward(completions: List[str | list], ground_truth: List[str], **_) -> List[float]:
    rewards: List[float] = []
    for completion, gold in zip(completions, ground_truth):
        pred = _to_float(_extract_pred(completion))
        gold_f = _to_float(gold)
        if pred is None or gold_f is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if abs(pred - gold_f) < 1e-4 else 0.0)
    return rewards


def format_reward(completions: List[str | list], **_) -> List[float]:
    return [0.5 if _FORMAT_RE.search(_completion_text(c)) else 0.0 for c in completions]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # LoRA keeps the base model frozen, so Adam state is tiny and a 0.5B model
    # fits in ~6GB during GRPO (policy + generation buffers).
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    train_ds = build_dataset(split="train", n=512)
    eval_ds = build_dataset(split="test", n=64)

    # ---------------------------------------------------------------------------
    # Step 1: SFT warm-up
    # ---------------------------------------------------------------------------
    if USE_SFT_WARMUP:
        print("***** Step 1: SFT warm-up *****")

        # SFT warm-up: teach the tag format before RL so GRPO groups have non-zero
        # format reward and denser correctness signal from the start.
        sft_config = SFTConfig(
            output_dir=f"{OUTPUT_DIR}/sft",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            max_length=512,
            num_train_epochs=1,
            logging_steps=5,
            save_strategy="no",
            eval_strategy="no",
            bf16=use_bf16,
            fp16=not use_bf16 and torch.cuda.is_available(),
            report_to=[],
            seed=42,
        )
        sft_trainer = SFTTrainer(
            model=MODEL_NAME,
            args=sft_config,
            train_dataset=train_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        sft_trainer.train()
        policy_model = sft_trainer.model
    else:        
        policy_model = MODEL_NAME

    # PEFT config for GRPO. For SFT warm-up, `policy_model` is already a PeftModel,
    # so we don't need to pass a peft_config again.
    # Otherwise, we pass the peft_config to the GRPO trainer.
    grpo_peft_config = None if isinstance(policy_model, PeftModel) else peft_config

    # ---------------------------------------------------------------------------
    # Step 2: GRPO tuning
    # ---------------------------------------------------------------------------

    print("***** Step 2: GRPO tuning *****")

    # generation_batch_size = per_device_train_batch_size * num_processes
    #                         * steps_per_generation (== grad-accum by default).
    # It must be divisible by num_generations and contain full prompt groups.
    # Here: 1 * 1 * 4 = 4, divisible by num_generations=4 -> 1 group / gen step.
    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_generations=4,
        max_completion_length=256,
        temperature=0.9,
        beta=0.01,
        num_train_epochs=1,
        logging_steps=5,
        save_steps=200,
        eval_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        report_to=[],
        log_completions=True,
        num_completions_to_print=2,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=grpo_peft_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
