"""Fine-tune SmolLM to rewrite tweets with PPO (TRL).

Standalone script. The policy (SmolLM) is trained to rewrite tweets so that a
TweetEval RoBERTa classifier ("cardiffnlp/twitter-roberta-base-{TASK}") assigns
more (or less) of the target attribute to the rewritten tweet.

TRL 1.0.0 ships an on-policy ``PPOTrainer`` (``trl.experimental.ppo``) that feeds
the *policy's* token ids straight into the reward model. The requested reward
model uses a different tokenizer/vocab and head than SmolLM, so we wrap it in
``TweetRewardModel``: it decodes the policy tokens, extracts the rewritten tweet,
runs the RoBERTa classifier, and returns a scalar reward in the [batch, seq, 1]
shape that TRL's ``get_reward`` expects.
"""

import re
import types

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# One of: "emoji", "emotion", "hate", "irony", "offensive", "sentiment".
TASK = "sentiment"

# True  -> rewrite tweets to show MORE of the attribute (maximize target class).
# False -> rewrite tweets to show LESS of the attribute (minimize target class).
PREFER_MORE = True

POLICY_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
REWARD_MODEL = f"cardiffnlp/twitter-roberta-base-{TASK}"

MAX_PROMPT_LENGTH = 128
NUM_TRAIN_SAMPLES = 4000
NUM_EVAL_SAMPLES = 64
OUTPUT_DIR = f"./ppo_tweet_rewrite_{TASK}"

# Maps a task to the classifier label that represents "more of the attribute".
# Matching is case-insensitive and substring based; if no label matches we fall
# back to the last label index (which is the positive/present class for the
# cardiffnlp binary and sentiment models).
TARGET_LABEL = {
    "sentiment": "positive",
    "emotion": "joy",
    "hate": "hate",
    "irony": "irony",
    "offensive": "offensive",
}

_MORE = "more" if PREFER_MORE else "less"
SYSTEM_PROMPT = f"You are a system to rewrite existing tweets to show {_MORE} of {TASK}"


# --------------------------------------------------------------------------- #
# Reward model wrapper
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


class _RewardBackbone(nn.Module):
    """Decodes policy tokens, scores the rewritten tweet, emits a reward tensor.

    TRL's ``get_reward`` calls ``getattr(reward_model, base_model_prefix)`` and
    then ``reward_model.score(output.hidden_states[-1])``. We mimic that contract:
    this module returns a fake hidden state of shape [batch, seq, 1] filled with
    the per-sequence reward, and ``TweetRewardModel.score`` is the identity.
    """

    def __init__(self, classifier, clf_tokenizer, policy_tokenizer, target_index):
        super().__init__()
        self.classifier = classifier
        self.clf_tokenizer = clf_tokenizer
        self.policy_tokenizer = policy_tokenizer
        self.target_index = target_index
        self.assistant_marker = "<|im_start|>assistant"
        self._special_re = re.compile(r"<\|[^>]*\|>")

    def _extract_response(self, ids: torch.Tensor) -> str:
        text = self.policy_tokenizer.decode(ids, skip_special_tokens=False)
        if self.assistant_marker in text:
            text = text.split(self.assistant_marker)[-1]
        text = self._special_re.sub(" ", text)
        return text.strip()

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, **kwargs):
        device = input_ids.device
        texts = [self._extract_response(row) for row in input_ids]
        texts = [_preprocess_tweet(t) if t else "." for t in texts]

        enc = self.clf_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.classifier.device)
        logits = self.classifier(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1)
        target = probs[:, self.target_index]
        reward = target if PREFER_MORE else 1.0 - target

        seq_len = input_ids.shape[1]
        hidden = reward.to(device).view(-1, 1, 1).expand(-1, seq_len, 1)
        return types.SimpleNamespace(hidden_states=(hidden,))


class TweetRewardModel(nn.Module):
    base_model_prefix = "reward_backbone"

    def __init__(self, classifier, clf_tokenizer, policy_tokenizer, target_index):
        super().__init__()
        self.reward_backbone = _RewardBackbone(
            classifier, clf_tokenizer, policy_tokenizer, target_index
        )

    def score(self, hidden_states):
        return hidden_states


def _resolve_target_index(config) -> int:
    label2id = {str(k).lower(): v for k, v in (config.label2id or {}).items()}
    wanted = TARGET_LABEL.get(TASK)
    if wanted is not None:
        for name, idx in label2id.items():
            if wanted in name:
                return int(idx)
    return config.num_labels - 1


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def build_dataset(tokenizer):
    raw = load_dataset("tweet_eval", TASK, split="train")

    def to_prompt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Rewrite the following tweet:\n\n{example['text']}"},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        return {"input_ids": input_ids}

    ds = raw.map(to_prompt, remove_columns=raw.column_names)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= MAX_PROMPT_LENGTH)
    return ds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(POLICY_MODEL, dtype=dtype)
    ref_policy = AutoModelForCausalLM.from_pretrained(POLICY_MODEL, dtype=dtype)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        POLICY_MODEL, num_labels=1, dtype=dtype
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id

    classifier = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL)
    classifier.eval()
    clf_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
    target_index = _resolve_target_index(classifier.config)
    print(
        f"[reward] task={TASK} prefer_more={PREFER_MORE} "
        f"target_label='{classifier.config.id2label[target_index]}' (idx={target_index})"
    )
    reward_model = TweetRewardModel(classifier, clf_tokenizer, tokenizer, target_index)

    dataset = build_dataset(tokenizer)
    train_dataset = dataset.select(range(min(NUM_TRAIN_SAMPLES, len(dataset))))
    eval_dataset = dataset.select(
        range(
            min(NUM_TRAIN_SAMPLES, len(dataset)),
            min(NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES, len(dataset)),
        )
    )

    config = PPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        local_rollout_forward_batch_size=8,
        num_mini_batches=1,
        num_ppo_epochs=4,
        total_episodes=NUM_TRAIN_SAMPLES,
        response_length=64,
        temperature=0.7,
        kl_coef=0.05,
        missing_eos_penalty=1.0,
        stop_token="eos",
        gradient_checkpointing=True,
        num_sample_generations=10,
        logging_steps=1,
        bf16=use_cuda,
        report_to="none",
    )

    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    trainer.generate_completions()


if __name__ == "__main__":
    main()
