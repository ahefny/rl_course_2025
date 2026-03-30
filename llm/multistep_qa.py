"""
Minimal multistep QA RL: QUERY (BM25), ANSWER / REFINE (frozen HF LM), STOP.

Dense rewards: small step penalty, retrieval overlap bonus after QUERY,
partial answer F1 after ANSWER, terminal F1 on STOP.

Requires: HF_TOKEN in the environment for real LM calls; without it, uses
deterministic string heuristics so the script still runs offline.
"""

from __future__ import annotations

import os
import random
import re
import dotenv
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple
from functools import cache

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from rank_bm25 import BM25Okapi


dotenv.load_dotenv()

from llm_wrapper import AnswerSynthesizer, AnswerShortener, QueryGenerator, parse_string_list
from text_embedding import TextEmbedder, BertEmbedder

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bert_embedder = BertEmbedder(device=DEVICE)

def _tokenize(s: str) -> List[str]:
    s = s.lower()
    return re.findall(r"\b\w+\b", s) or []


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_toks = _tokenize(prediction)
    gold_toks = _tokenize(ground_truth)
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def token_recall(prediction: str, ground_truth: str) -> float:
    pred_toks = _tokenize(prediction)
    gold_toks = _tokenize(ground_truth)
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    return num_same / len(gold_toks)


def best_f1_over_gold(prediction: str, gold_answers: Sequence[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(token_f1(prediction, g) for g in gold_answers)

def best_recall_over_gold(prediction: str, gold_answers: Sequence[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(token_recall(prediction, g) for g in gold_answers)

# --- fallback LM (no API) ---

def _fallback_answer(question: str, passages: List[str]) -> str:
    blob = " ".join(passages)[:800]
    if not blob.strip():
        return "unknown"
    # first clause-ish chunk
    m = re.search(r"([^.!?]+[.!?])", blob)
    return (m.group(1).strip() if m else blob[:120]).strip()


def _fallback_refine(question: str, current_query: str, passages: List[str]) -> str:
    hint = _tokenize(" ".join(passages))[:6]
    extra = " ".join(hint)
    return (current_query + " " + extra).strip()[:200]


# --- environment ---


class QAAction(IntEnum):
    QUERY = 0
    ANSWER = 1
    # REFINE = 2
    STOP = 2
    SHORTEN = 3


@dataclass
class QAExample:
    question: str
    context: str
    gold_answers: Tuple[str, ...]
    context_id: int

@dataclass
class State:
    question: str
    context: str
    query_results: list[tuple[str, str]]
    last_answer: str
    last_answer_f1: float
    last_context_recall: float
    last_action: QAAction | None

    def get_all_query_results(self) -> str:
        return "\n-----------------------\n".join(doc for _, doc in self.query_results)

def embed_state(state: State, text_embed_fn: TextEmbedder) -> torch.Tensor:
    query_results = "\n".join(
        f"[{i+1}] **{query}**\n{doc[:500]}" for i, (query, doc) in enumerate(state.query_results)
    )

    return torch.cat([
        text_embed_fn.embed(state.context),
        text_embed_fn.embed(state.question),
        text_embed_fn.embed(query_results),
        text_embed_fn.embed(state.last_answer),
    ], dim=-1)

class MultistepQAEnv(gym.Env):
    """
    Actions: QUERY, ANSWER, REFINE, STOP.
    Observation: fixed hash embedding of question, current query, retrieved snippet, last answer.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        examples: List[QAExample],
        corpus_ids: List[int],
        corpus_texts: List[str],
        top_k: int = 5,
        max_episode_steps: int = 15,
        default_action_cost: float = 0.01,
        action_costs: dict[QAAction, float] | None = None,
        retrieval_bonus_scale: float = 0.1,
        answer_dense_scale: float = 0.2,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.examples = examples
        self.corpus_ids = corpus_ids
        self.corpus_texts = corpus_texts
        tokenized_corpus = [_tokenize(t) for t in corpus_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self.top_k = top_k
        self.max_episode_steps = max_episode_steps
        self.default_action_cost = default_action_cost
        self.action_costs = action_costs or {}
        self.retrieval_bonus_scale = retrieval_bonus_scale
        self.answer_dense_scale = answer_dense_scale
        self.redundant_action_penalty = 0.05
        self.invalid_action_penalty = 1.0

        self.obs_dim = 128 * 4
        self.action_space = spaces.Discrete(len(QAAction))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._episode_step = 0
        self._ex: Optional[QAExample] = None
        self._current_query = ""
        self._retrieved: List[Tuple[int, str]] = []

        assert self.obs_dim % 4 == 0
        self._text_embed_fn = BertEmbedder(embedding_dim=self.obs_dim // 4, device=DEVICE)

        self._answer_synthesizer = AnswerSynthesizer(MODEL_NAME, device=DEVICE)
        self._answer_shortener = AnswerShortener(MODEL_NAME, device=DEVICE)
        self._query_generator = QueryGenerator(MODEL_NAME, device=DEVICE)

        self._state = State(
            question=self._ex.question if self._ex else "",
            context=self._ex.context if self._ex else "",
            query_results=[],

            last_answer="",
            last_answer_f1=0.0,
            last_context_recall=0.0,
            last_action=None,
        )


    def _obs(self) -> torch.Tensor:
        return embed_state(self._state, self._text_embed_fn)
        

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._episode_step = 0
        options = options or {}
        if "example_index" in options:
            idx = int(options["example_index"])
            if not (0 <= idx < len(self.examples)):
                raise ValueError(f"example_index out of range: {idx}")
            self._ex = self.examples[idx]
        else:
            self._ex = self.examples[int(self._rng.integers(0, len(self.examples)))]

        self._state = State(
            question=self._ex.question,
            context=self._ex.context,
            query_results=[],
            last_answer="",
            last_answer_f1=0.0,
            last_context_recall=0.0,
            last_action=None,
        )
        return self._obs(), {}

    def _lm_answer(self) -> str:
        query_results = [
            {
                "query": query,
                "result": doc,
            } for query, doc in self._state.query_results
        ]
        answer, _ = self._answer_synthesizer.generate(
            arguments={
                "context": self._state.context,
                "query_outputs": query_results,
                "question": self._state.question,
            },
        )
        return answer

    def _lm_refine(self) -> str:
        passages = [t for _, t in self._retrieved]
        ctx = "\n".join(p[:200] for p in passages) #[:3])
        prompt = (
            "### Instruction:\n"
            "Rewrite the search query to improve passage retrieval.\n"
            f"Question: {self._ex.question}\n"
            f"Current query: {self._current_query}\n"
            f"Passage hints:\n{ctx}\n"
            "Output a single line: the new query only.\n"
            "### Response:\n"
        )
        out = _hf_generate(prompt, max_new_tokens=48)
        if out:
            return out.splitlines()[0].strip()[:200]
        return _fallback_refine(self._ex.question, self._current_query, passages)

    def _lm_shorten(self) -> str:
        if self._state.last_answer.strip() == "unknown":
            return ""
        if self._state.last_answer.strip() == "":
            return ""

        query_results = [
            {
                "query": query,
                "result": doc,
            } for query, doc in self._state.query_results
        ]
        answer, _ = self._answer_shortener.generate(
            arguments={
                "question": self._state.question,
                "original_answer": self._state.last_answer,
            },
        )
        return answer

    def _retrieve(self, query: str) -> list[tuple[int, str]]:
        toks = _tokenize(query)
        if not toks:
            toks = _tokenize(self._state.question)
        scores = self._bm25.get_scores(toks)
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return [(int(i), self.corpus_texts[int(i)]) for i in top_idx]

    def step(self, action: int):
        assert self._ex is not None
        reward = 0.0
        terminated = False
        truncated = False
        self._episode_step += 1        

        action_cost = self.action_costs.get(action, self.default_action_cost)
        reward -= action_cost

        if action == QAAction.QUERY:
            if self._state.last_action == QAAction.QUERY:
                reward -= self.redundant_action_penalty

            query_results = [
                {
                    "query": query,
                    "result": doc,
                } for query, doc in self._state.query_results
            ]
            query, _ = self._query_generator.generate(
                arguments={
                    "question": self._state.question,
                    "context": self._state.context,
                    "previous_query_outputs": query_results,
                },
            )

            print(f"QUERY: {query}")
            queries = parse_string_list(query)
            
            for q in queries:
                q = q.strip()
                if not q:
                    continue
                if q.lower().startswith("answer:"):
                    self._state.last_answer = q[len("answer:"):].strip()
                    print(f"QUERY ANSWER: {self._state.last_answer}")
                    reward -= self.redundant_action_penalty
                    break

                retrieved = self._retrieve(q)
                docs = "\n-----------------------\n".join(doc for _, doc in retrieved)
                self._state.query_results.append((q, docs))

                recall = best_recall_over_gold(docs, self._ex.gold_answers)
                recall_diff = recall - self._state.last_context_recall
                reward += self.retrieval_bonus_scale * max(0, recall_diff)
                self._state.last_context_recall = recall

                print(f"RETRIEVED: {q}: {docs}")
                # self._state.query_results.extend(
                #     (q, doc) for _, doc in retrieved
                # )

            if len(self._state.query_results) > 5:
                self._state.query_results = self._state.query_results[-5:]

        elif action == QAAction.ANSWER:            
            self._state.last_answer = self._lm_answer()
            new_answer_f1 = best_f1_over_gold(
                self._state.last_answer, self._ex.gold_answers
            )
            f1_diff = new_answer_f1 - self._state.last_answer_f1
            reward += self.answer_dense_scale * f1_diff
            self._state.last_answer_f1 = new_answer_f1
            print(f"Answer: {self._state.last_answer}")

        # elif action == QAAction.REFINE:
        #     current_query = self._current_query
        #     self._current_query = self._lm_refine()
        #     if current_query == self._current_query:
        #         reward -= self.redundant_action_penalty
        #     print(f"Refine: {self._current_query}")

        elif action == QAAction.STOP:
            if self._state.last_action != QAAction.ANSWER:
                reward -= self.redundant_action_penalty
            print(f"Stop")
            if self._state.last_answer == "" or self._state.last_answer.strip() == "unknown":
                terminal = -10.0
            else:
                terminal = best_f1_over_gold(self._state.last_answer, self._ex.gold_answers) * self.answer_dense_scale
            reward += terminal
            terminated = True

        elif action == QAAction.SHORTEN:
            if self._state.last_answer.strip() == "" or self._state.last_answer.strip() == "unknown":
                reward -= self.invalid_action_penalty

            last_answer = self._state.last_answer
            self._state.last_answer = self._lm_shorten()

            if last_answer == self._state.last_answer:
                reward -= self.redundant_action_penalty
            else:
                new_answer_f1 = best_f1_over_gold(
                    self._state.last_answer, self._ex.gold_answers
                )
                f1_diff = new_answer_f1 - self._state.last_answer_f1
                reward += self.answer_dense_scale * f1_diff
                self._state.last_answer_f1 = new_answer_f1
            print(f"Shorten: {last_answer} -> {self._state.last_answer}")

        if self._episode_step >= self.max_episode_steps:
            truncated = not terminated
            if not terminated:
                # force terminal score without extra STOP
                # reward += best_f1_over_gold(self._state.last_answer, self._ex.gold_answers) * self.answer_dense_scale
                terminated = True

        print(f"*** REWARD ***: {reward}")

        self._state.last_action = action
        return self._obs(), float(reward), terminated, truncated, {}


# --- policy + REINFORCE ---

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = len(QAAction)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    g = 0.0
    out: List[float] = []
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return out


def build_task(
    n_examples: int = 400,
    n_corpus_docs: int = 400,
    seed: int = 0,
) -> Tuple[List[QAExample], List[int], List[str]]:
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("squad_v2", split="train")
    examples: List[QAExample] = []
    contexts: List[str] = []
    for row in ds:
        texts = row["answers"]["text"]
        if not texts:
            continue
        ctx = row["context"]
        examples.append(
            QAExample(
                question=row["question"].strip(),
                context=ctx,
                gold_answers=tuple(texts),
                context_id=-1,
            )
        )
        contexts.append(ctx)
        if len(examples) >= n_examples:
            break

    # corpus: stratified sample of unique contexts
    uniq: List[str] = []
    seen = set()
    for c in contexts:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    rng.shuffle(uniq)
    uniq = uniq[:n_corpus_docs]
    id_map = {t: i for i, t in enumerate(uniq)}

    fixed_examples: List[QAExample] = []
    for ex in examples:
        if ex.context in id_map:
            fixed_examples.append(
                QAExample(
                    question=ex.question,
                    context=ex.context,
                    gold_answers=ex.gold_answers,
                    context_id=id_map[ex.context],
                )
            )
    if not fixed_examples:
        raise RuntimeError("No examples after corpus filter; increase n_examples.")

    corpus_ids = list(range(len(uniq)))
    return fixed_examples, corpus_ids, uniq


def normalize_returns_across_rollouts(
    rollouts_returns: Sequence[Sequence[float]],
    eps: float = 1e-8,
) -> List[List[float]]:
    """Per-timestep z-score of return-to-go across rollouts that reached that step."""
    n = len(rollouts_returns)
    out: List[List[float]] = []
    for i in range(n):
        ri = rollouts_returns[i]
        norm_i: List[float] = []
        for t in range(len(ri)):
            vals = [rollouts_returns[j][t] for j in range(n) if t < len(rollouts_returns[j])]
            mu = float(np.mean(vals))
            sig = float(np.std(vals)) + eps
            norm_i.append((ri[t] - mu) / sig)
        out.append(norm_i)
    return out


def train_reinforce(
    total_episodes: int = 100_000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    seed: int = 0,
    rollouts_per_context: int = 8,
) -> None:
    examples, corpus_ids, corpus_texts = build_task(
        n_examples=500, n_corpus_docs=400, seed=seed
    )
    env = MultistepQAEnv(
        examples,
        corpus_ids,
        corpus_texts,
        top_k=1,
        max_episode_steps=5,
        default_action_cost=0.01,
        action_costs={
            QAAction.QUERY: 1.0,
            QAAction.ANSWER: 0.1,
            QAAction.SHORTEN: 0.01,
        },
        retrieval_bonus_scale=5.0,
        answer_dense_scale=5.0,
        seed=seed,
    )

    obs_dim = env.obs_dim
    policy = PolicyNet(obs_dim).to(DEVICE)
    opt = optim.Adam(policy.parameters(), lr=lr)
    rng_ep = np.random.default_rng(seed)

    for ep in range(total_episodes):
        example_idx = int(rng_ep.integers(0, len(env.examples)))
        base_seed = int(ep + seed * 1000)

        print("================================================")
        print(f"Episode {ep} (example_index={example_idx}, {rollouts_per_context} rollouts)")
        print(f"Question: {env.examples[example_idx].question}")
        print(f"Gold Answers: {env.examples[example_idx].gold_answers}")
        print("------------------------------------------------")

        rollouts_log_probs: List[List[torch.Tensor]] = []
        rollouts_rewards: List[List[float]] = []

        for rollout_idx in range(rollouts_per_context):
            print("================================================")
            print(f"Rollout {rollout_idx + 1} of {rollouts_per_context}")
            print("================================================")

            obs, _ = env.reset(
                seed=base_seed,
                options={"example_index": example_idx},
            )
            log_probs: List[torch.Tensor] = []
            rewards: List[float] = []
            done = False
            while not done:
                s = obs.detach()
                dist = policy(s)
                a = dist.sample()
                log_probs.append(dist.log_prob(a))
                obs, r, term, trunc, _ = env.step(int(a.item()))
                rewards.append(float(r))
                done = term or trunc

            rollouts_log_probs.append(log_probs)
            rollouts_rewards.append(rewards)

        raw_returns = [compute_returns(rw, gamma) for rw in rollouts_rewards]
        normalized = normalize_returns_across_rollouts(raw_returns)

        loss = torch.zeros((), device=DEVICE)
        for log_probs, norm_ret in zip(rollouts_log_probs, normalized):
            lp = torch.stack(log_probs)
            ret_t = torch.tensor(norm_ret, dtype=torch.float32, device=lp.device)
            loss = loss + -(lp * ret_t).sum()
        loss = loss / rollouts_per_context

        opt.zero_grad()
        loss.backward()
        opt.step()

        last_rw = rollouts_rewards[-1]
        raw_sum = float(sum(last_rw))
        print("------------------------------------------------")
        print(f"Episode {ep}")
        print(f"Question: {env._ex.question}")
        print(f"Answer: {env._state.last_answer}")
        print(f"Gold Answers: {env._ex.gold_answers}")
        print(f"Best F1: {best_f1_over_gold(env._state.last_answer, env._ex.gold_answers)}")
        print(f"Loss: {loss.item()}")
        print(f"Return (last rollout): {raw_sum}")
        print("================================================")


if __name__ == "__main__":
    train_reinforce()
