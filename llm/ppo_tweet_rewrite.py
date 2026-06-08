"""Fine-tune Qwen2.5 to rewrite tweets with PPO (TRL).

Standalone script. The policy (Qwen2.5) is trained to rewrite tweets so that a
TweetEval RoBERTa classifier ("cardiffnlp/twitter-roberta-base-{TASK}") assigns
more (or less) of the target attribute to the rewritten tweet.

TRL 1.0.0 ships an on-policy ``PPOTrainer`` (``trl.experimental.ppo``) that feeds
the *policy's* token ids straight into the reward model. The requested reward
model uses a different tokenizer/vocab and head than the policy, so we wrap it in
``TweetRewardModel``: it decodes the policy tokens, extracts the rewritten tweet,
runs the RoBERTa classifier, and returns a scalar reward in the [batch, seq, 1]
shape that TRL's ``get_reward`` expects.
"""

import re
import types

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# One of: "emotion:{anger,joy,sadness,optimism}", "hate", "irony", "offensive", "sentiment".
# For "emotion", the subtask is the emotion name.
METRIC = "emotion:joy"

def _resolve_task_name(task: str) -> str:
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
NUM_EVAL_SAMPLES = 64
OUTPUT_DIR = f"./ppo_tweet_rewrite_{TASK}"

# Secondary reward: Jaccard token overlap between the original and rewritten
# tweet, added to the classifier reward to discourage the policy from drifting
# away from the source content. Set to 0.0 to disable.
JACCARD_COEF = 0.5

# Maps a task to the classifier label that represents "more of the attribute".
# Matching is case-insensitive and substring based; if no label matches we fall
# back to the last label index (which is the positive/present class for the
# cardiffnlp binary and sentiment models).

_MORE = "more" if PREFER_MORE else "less"
SYSTEM_PROMPT = f"""
You are a writing assistant to help a user rewrite existing tweets to show {_MORE} of {SUBTASK}.

**ONLY** output the rewritten tweet. Do **NOT** include any other text or explanation.
"""

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
        self.tweet_marker = "Tweet:"
        self._special_re = re.compile(r"<\|[^>]*\|>")

    def _clean(self, text: str) -> str:
        return self._special_re.sub(" ", text).strip()

    def _split(self, ids: torch.Tensor) -> tuple[str, str]:
        """Return (original_tweet, rewritten_tweet) decoded from a sequence.

        The full sequence is ``<prompt> <|im_start|>assistant <response>``. The
        original tweet sits in the prompt after the ``Tweet:`` marker, and the
        rewrite is everything after the assistant marker.
        """
        text = self.policy_tokenizer.decode(ids, skip_special_tokens=False)
        parts = text.split(self.assistant_marker)
        prompt_part, response_part = parts[0], parts[-1]
        if self.tweet_marker in prompt_part:
            original = prompt_part.split(self.tweet_marker)[-1]
        else:
            original = prompt_part
        return self._clean(original), self._clean(response_part)

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        set_a, set_b = set(a.lower().split()), set(b.lower().split())
        union = set_a | set_b
        if not union:
            return 1.0
        return len(set_a & set_b) / len(union)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, **kwargs):
        device = input_ids.device
        pairs = [self._split(row) for row in input_ids]
        rewrites = [r for _, r in pairs]
        clf_texts = [_preprocess_tweet(r) if r else "." for r in rewrites]

        enc = self.clf_tokenizer(
            clf_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.classifier.device)
        logits = self.classifier(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1)
        target = probs[:, self.target_index]
        clf_reward = target if PREFER_MORE else 1.0 - target

        jaccard = torch.tensor(
            [self._jaccard(original, rewrite) for original, rewrite in pairs],
            device=clf_reward.device,
            dtype=clf_reward.dtype,
        )
        reward = clf_reward + JACCARD_COEF * jaccard

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
    wanted = SUBTASK
    return label2id[wanted]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def build_dataset(tokenizer):
    raw = load_dataset("tweet_eval", TASK, split="train")

    def to_prompt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""
            Rewrite the following tweet to show {_MORE} of {SUBTASK}.
            **ONLY** output the rewritten tweet. Do **NOT** include any other text or explanation.


            Tweet: {example['text']}"""},
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

    # LoRA on the policy keeps optimizer state tiny; with PEFT the trainer reuses
    # the (adapter-disabled) policy as its own reference model, so no separate
    # ref model is loaded.
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Lightweight critic: freeze the value backbone and train only the scalar
    # value head ("score"). This avoids a second full set of optimizer states.
    value_model = AutoModelForSequenceClassification.from_pretrained(
        POLICY_MODEL, num_labels=1, dtype=dtype
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id
    for name, param in value_model.named_parameters():
        param.requires_grad = "score" in name

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
    n = len(dataset)
    n_eval = min(NUM_EVAL_SAMPLES, n)
    n_train = min(NUM_TRAIN_SAMPLES, n - n_eval)
    eval_dataset = dataset.select(range(n - n_eval, n))
    train_dataset = dataset.select(range(n_train))

    config = PPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        local_rollout_forward_batch_size=4,
        num_mini_batches=1,
        num_ppo_epochs=4,
        total_episodes=NUM_TRAIN_SAMPLES,
        response_length=64,
        temperature=0.7,
        kl_coef=0.05,
        missing_eos_penalty=0.0,
        stop_token="eos",
        gradient_checkpointing=False,
        num_sample_generations=100,
        logging_steps=1,
        bf16=use_cuda,
        report_to="none",
    )

    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    trainer.generate_completions()


if __name__ == "__main__":
    main()
