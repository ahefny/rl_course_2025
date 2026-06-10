"""Fine-tune Qwen2.5 to rewrite tweets with GRPO (TRL).

Standalone script. The policy (Qwen2.5) is trained to rewrite tweets so that a
TweetEval RoBERTa classifier ("cardiffnlp/twitter-roberta-base-{TASK}") assigns
more (or less) of the target attribute to the rewritten tweet.

Unlike PPO, GRPO needs no value model and no reward *model* plumbing: it samples
a group of completions per prompt and scores them with plain Python reward
functions over the decoded text. We use two reward functions:
  1. ``attribute_reward`` - the classifier probability of the target class.
  2. ``jaccard_reward``   - token overlap with the original tweet (keeps the
     rewrite faithful to the source); weighted by ``JACCARD_COEF``.
"""

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# One of: "emotion:{anger,joy,sadness,optimism}", "hate", "irony", "offensive", "sentiment".
# For "emotion", the subtask is the emotion name.
METRIC = "emotion:joy"


def _resolve_task_name(task: str):
    if ":" in task:
        return task.split(":")
    return task, task


TASK, SUBTASK = _resolve_task_name(METRIC)

# True  -> rewrite tweets to show MORE of the attribute (maximize target class).
# False -> rewrite tweets to show LESS of the attribute (minimize target class).
PREFER_MORE = True

POLICY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL = f"cardiffnlp/twitter-roberta-base-{TASK}"

MAX_PROMPT_LENGTH = 128
NUM_TRAIN_SAMPLES = 4000
OUTPUT_DIR = f"./grpo_tweet_rewrite_{TASK}"

# Secondary reward: Jaccard token overlap between the original and rewritten
# tweet, added to the classifier reward to discourage the policy from drifting
# away from the source content. Set to 0.0 to disable.
JACCARD_COEF = 0.1

_MORE = "more" if PREFER_MORE else "less"
SYSTEM_PROMPT = f"""
You are a writing assistant to help a user rewrite existing tweets to show {_MORE} of {SUBTASK}.

**ONLY** output the rewritten tweet. Do **NOT** include any other text or explanation.
"""


# --------------------------------------------------------------------------- #
# Reward helpers
# --------------------------------------------------------------------------- #


def _preprocess_tweet(text: str) -> str:
    """cardiffnlp preprocessing: mask usernames and links."""
    tokens = []
    for token in text.split():
        if token.startswith("@") and len(token) > 1:
            token = "@user"
        elif token.startswith("http"):
            token = "http"
        tokens.append(token)
    return " ".join(tokens)


def _completion_text(completion) -> str:
    """Extract assistant text from a GRPO completion (str or list of messages)."""
    if isinstance(completion, str):
        return completion.strip()
    return " ".join(
        msg.get("content", "") for msg in completion if isinstance(msg, dict)
    ).strip()


def _jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _resolve_target_index(config) -> int:
    label2id = {str(k).lower(): v for k, v in (config.label2id or {}).items()}
    return label2id[SUBTASK]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def build_dataset(tokenizer):
    raw = load_dataset("tweet_eval", TASK, split="train")

    def to_prompt(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"""
            Rewrite the following tweet to show {_MORE} of {SUBTASK}.
            **ONLY** output the rewritten tweet. Do **NOT** include any other text or explanation.


            Tweet: {example['text']}""",
                },
            ],
            "tweet": example["text"],
        }

    ds = raw.map(to_prompt, remove_columns=raw.column_names)

    def short_enough(example):
        ids = tokenizer.apply_chat_template(
            example["prompt"], add_generation_prompt=True, tokenize=True, return_dict=False
        )
        return len(ids) <= MAX_PROMPT_LENGTH

    return ds.filter(short_enough)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(POLICY_MODEL, dtype=dtype)

    # LoRA keeps optimizer state tiny; with PEFT the trainer uses the
    # adapter-disabled policy as its own KL reference, so no separate ref model.
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    classifier = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL).to(device)
    classifier.eval()
    clf_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
    target_index = _resolve_target_index(classifier.config)
    print(
        f"[reward] task={TASK} subtask={SUBTASK} prefer_more={PREFER_MORE} "
        f"target_label='{classifier.config.id2label[target_index]}' (idx={target_index})"
    )

    def attribute_reward(completions, **kwargs):
        """Classifier probability of the target attribute for each rewrite."""
        texts = [_preprocess_tweet(_completion_text(c)) or "." for c in completions]
        enc = clf_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(classifier.device)
        with torch.no_grad():
            probs = torch.softmax(classifier(**enc).logits.float(), dim=-1)
        target = probs[:, target_index]
        reward = target if PREFER_MORE else 1.0 - target
        return reward.tolist()

    def jaccard_reward(completions, tweet, **kwargs):
        """Token overlap between the original tweet and the rewrite."""
        return [_jaccard(original, _completion_text(c)) for c, original in zip(completions, tweet)]

    dataset = build_dataset(tokenizer)
    train_dataset = dataset.select(range(min(NUM_TRAIN_SAMPLES, len(dataset))))

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_generations=8,
        max_completion_length=64,
        temperature=1.0,
        beta=0.0,
        num_train_epochs=1,
        reward_weights=[1.0, JACCARD_COEF],
        remove_unused_columns=False,  # keep "tweet" so reward funcs can read it
        gradient_checkpointing=False,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=5,
        bf16=use_cuda,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=[attribute_reward, jaccard_reward],
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
