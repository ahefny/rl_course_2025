"""Minimal GRPO training example: small LLM on the Countdown game.

This is a TinyZero-style setup. Given a list of numbers and a target, the
model must produce an arithmetic expression (using `+ - * /` and parentheses)
that uses each given number **exactly once** and evaluates to the target.

Why Countdown and not GSM8K for a small model:
  * The reward is fully verifiable in code (parse + safe-eval the expression).
  * The base model occasionally succeeds even at 0.5B, so GRPO groups have
    non-zero reward variance early -> the gradient signal is non-empty.
  * The task is search-shaped, so chain-of-thought is genuinely useful and
    "self-verification / backtracking" tends to emerge.

Dataset: https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
   columns: `nums` (list[int]), `target` (int).

Run:

    python llm/grpo_countdown.py

Defaults are tuned for a ~6GB GPU (LoRA + gradient checkpointing).
"""

from __future__ import annotations

import ast
import operator
import re
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "Jiayi-Pan/Countdown-Tasks-3to4"
OUTPUT_DIR = "outputs/grpo_countdown"

SYSTEM_PROMPT = (
    "You are playing the Countdown number game. You will be given a list of "
    "numbers and a target value. Build an arithmetic expression that uses "
    "each given number EXACTLY ONCE, only with + - * / and parentheses, and "
    "evaluates to the target.\n"
    "Put your reasoning between <think> and </think>, then put the final "
    "arithmetic expression between <answer> and </answer>.\n"
    "Example: <answer>(8 + 3) * 5 - 7</answer>"
)


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _extract_expr(completion: str) -> Optional[str]:
    """Return the raw text inside the LAST <answer>...</answer> tag (if any)."""
    matches = _ANSWER_TAG_RE.findall(completion)
    if not matches:
        return None
    expr = matches[-1].strip()
    expr = expr.replace("×", "*").replace("÷", "/")
    return expr or None


def _eval_node(node: ast.AST) -> Optional[float]:
    """Recursively evaluate a restricted arithmetic AST. None on failure."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if left is None or right is None:
            return None
        try:
            return _ALLOWED_OPS[type(node.op)](left, right)
        except ZeroDivisionError:
            return None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        v = _eval_node(node.operand)
        return _ALLOWED_OPS[type(node.op)](v) if v is not None else None
    return None


def _safe_eval(expr: str) -> Optional[float]:
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None
    return _eval_node(tree.body)


def _is_arith_node(node: ast.AST) -> bool:
    """True iff `node` is a numeric constant or a +-*/ BinOp/UnaryOp."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return True
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _is_arith_node(node.left) and _is_arith_node(node.right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _is_arith_node(node.operand)
    return False


def _numbers_used(expr: str) -> Optional[List[float]]:
    """Return the multiset of numeric literals if `expr` is a pure arithmetic
    expression over `+ - * /` (and unary +/-), else None.

    Rejects identifiers, calls, comparisons, `is`/`is not`, etc., so that a
    string like `this is not math` or `__import__("os")` is treated as
    unparseable rather than a "valid expression that uses no numbers".
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None
    if not _is_arith_node(tree.body):
        return None
    nums: List[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            nums.append(float(node.value))
    return nums


def _check_numbers(used: List[float], allowed: List[int]) -> str:
    """Classify how `used` relates to the allowed multiset.

    Returns one of:
        "extras"  - `used` contains a number not in `allowed`
        "subset"  - `used` only uses allowed numbers (each at most once) but
                    misses at least one
        "exact"   - `used` is exactly the multiset of `allowed`
    """
    allowed_count = {}
    for n in allowed:
        allowed_count[float(n)] = allowed_count.get(float(n), 0) + 1
    for n in used:
        if allowed_count.get(n, 0) <= 0:
            return "extras"
        allowed_count[n] -= 1
    return "exact" if all(c == 0 for c in allowed_count.values()) else "subset"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(tokenizer, split: str, n: Optional[int] = None):
    """Return a dataset with `prompt`, `nums`, and `target` columns."""
    ds = load_dataset(DATASET_NAME, split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    def _format(example):
        user_msg = f"Numbers: {list(example['nums'])}\nTarget: {example['target']}"
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "nums": list(example["nums"]),
            "target": int(example["target"]),
        }

    return ds.map(_format, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# Reward functions (TRL passes prompts/completions/dataset columns as kwargs)
# ---------------------------------------------------------------------------

def correctness_reward(
    completions: List[str],
    nums: List[List[int]],
    target: List[int],
    **_,
) -> List[float]:
    """Shaped Countdown correctness reward.

    Binary 0/1 leaves all-zero groups (no advantage signal) for too long on a
    0.5B model. We give partial credit for each sub-condition the completion
    satisfies so groups have non-zero variance from very early in training:

        0.0  no <answer> tag
        0.1  tag present but expression unparseable
        0.2  parseable but uses a number not in `nums`
        0.3  parseable, only allowed numbers, but doesn't use all of them
        0.5  parseable, uses each given number exactly once, wrong value
        1.0  fully correct
    """
    rewards: List[float] = []
    for completion, allowed_nums, tgt in zip(completions, nums, target):
        expr = _extract_expr(completion)
        if expr is None:
            rewards.append(0.0)
            continue

        used = _numbers_used(expr)
        if used is None:
            rewards.append(0.1)
            continue

        kind = _check_numbers(used, allowed_nums)
        if kind == "extras":
            rewards.append(0.2)
            continue
        if kind == "subset":
            rewards.append(0.3)
            continue

        value = _safe_eval(expr)
        if value is None:
            rewards.append(0.5)
            continue

        rewards.append(1.0 if abs(value - float(tgt)) < 1e-4 else 0.5)
    return rewards


def format_reward(completions: List[str], **_) -> List[float]:
    return [0.5 if _FORMAT_RE.search(c) else 0.0 for c in completions]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = build_dataset(tokenizer, split="train", n=2048)
    eval_ds = build_dataset(tokenizer, split="train", n=64)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # generation_batch_size = per_device_train_batch_size * num_processes
    #                         * steps_per_generation (== grad-accum by default).
    # Must be divisible by num_generations and contain full prompt groups.
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
        beta=0.0,
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
        model=MODEL_NAME,
        reward_funcs=[correctness_reward, format_reward],
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
