#!/usr/bin/env python3
"""Reward-weighted fine-tune SmolVLA to move right in a LIBERO simulator.

This is a deliberately misaligned learning demonstration, not a task-solving
algorithm. It collects on-policy 50-step action chunks, assigns each chunk the
return ``sum(x[t+1] - x[t])``, then behavior-clones the higher-return chunks.
Only the small action-output projection is trained to keep the experiment
lightweight and to make the altered behavior easy to observe.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from vla_demo_onesample import DEFAULT_MODEL, adapt_observation_to_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3, help="Collect/train outer-loop iterations.")
    parser.add_argument("--rollouts", type=int, default=8, help="Number of 50-step chunks per iteration.")
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Save every N outer-loop iterations.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reward-temperature", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("vla_outputs/move_right_rwft"))
    return parser.parse_args()


def clone_transition(transition: dict) -> dict:
    """Clone tensors before a processor mutates a rollout observation."""
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value)
        for key, value in transition.items()
    }


def collate_observations(observations: list[dict], action_chunks: list[np.ndarray]) -> dict:
    """Create a batch of unnormalized observation/action trajectories."""
    batch = {}
    for key in observations[0]:
        values = [observation[key] for observation in observations]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list) and all(len(value) == 1 for value in values):
            batch[key] = [value[0] for value in values]
        else:
            batch[key] = values
    batch["action"] = torch.from_numpy(np.stack(action_chunks)).float()
    return batch


def end_effector_x(observation: dict) -> float:
    return float(observation["robot_state"]["eef"]["pos"][0, 0])


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive.")
    if args.checkpoint_freq < 1:
        raise ValueError("--checkpoint-freq must be positive.")
    if args.rollouts < 2:
        raise ValueError("--rollouts must be at least 2 so reward weighting has an effect.")
    if args.chunk_steps != 50:
        raise ValueError("This simple demo requires --chunk-steps 50 to match the checkpoint action chunk.")

    try:
        from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
        from lerobot.envs.configs import LiberoEnv
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as error:
        raise SystemExit(
            "LIBERO dependencies are unavailable. Run ./vla/setup.sh again.\n"
            f"Original error: {error}"
        ) from error

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = SmolVLAPolicy.from_pretrained(args.model).to(device)
    policy_preprocess, policy_postprocess = make_pre_post_processors(
        policy.config,
        args.model,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    env_config = LiberoEnv(
        task=args.suite,
        task_ids=[args.task_id],
        episode_length=args.chunk_steps,
    )
    env = make_env(env_config, n_envs=1, use_async_envs=False)[args.suite][args.task_id]
    env_preprocess, env_postprocess = make_env_pre_post_processors(env_config, policy.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    trainable = list(policy.model.action_out_proj.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    iteration_metrics = []

    try:
        for iteration in range(args.iterations):
            print(f"\nIteration {iteration + 1}/{args.iterations}: collecting fresh rollouts")
            start_observations, action_chunks, returns = [], [], []
            for rollout in range(args.rollouts):
                rollout_seed = args.seed + iteration * args.rollouts + rollout
                observation, _ = env.reset(seed=rollout_seed)
                task = env.call("task_description")[0]
                policy.reset()

                policy_observation = preprocess_observation(observation)
                policy_observation["task"] = [task]
                policy_observation = env_preprocess(policy_observation)
                policy_observation = adapt_observation_to_policy(
                    policy_observation, policy.config.input_features
                )
                start_observations.append(clone_transition(policy_observation))

                actions, reward = [], 0.0
                for _ in range(args.chunk_steps):
                    x_before = end_effector_x(observation)
                    with torch.inference_mode():
                        action = policy_postprocess(policy.select_action(policy_preprocess(policy_observation)))
                    action = env_postprocess({"action": action})["action"]
                    action_numpy = action.detach().cpu().numpy()
                    observation, _, terminated, truncated, _ = env.step(action_numpy)
                    reward += end_effector_x(observation) - x_before
                    actions.append(action_numpy[0])
                    if bool(terminated[0] or truncated[0]):
                        break

                if len(actions) != args.chunk_steps:
                    start_observations.pop()
                    print(f"Discarded rollout {rollout}: ended after {len(actions)} steps.")
                    continue
                action_chunks.append(np.asarray(actions, dtype=np.float32))
                returns.append(reward)
                print(f"Rollout {rollout}: move-right return = {reward:.5f}")

            if len(action_chunks) < 2:
                raise RuntimeError("Too few complete rollouts were collected; rerun with more --rollouts.")

            returns_tensor = torch.tensor(returns, device=device)
            weights = torch.softmax(args.reward_temperature * (returns_tensor - returns_tensor.mean()), dim=0)
            print(f"Normalized rollout weights: {weights.detach().cpu().tolist()}")

            policy.train()
            losses = []
            for update in range(args.updates):
                batch = policy_preprocess(collate_observations(start_observations, action_chunks))
                per_sample_loss, _ = policy.forward(batch, reduction="none")
                loss = torch.sum(weights * per_sample_loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                print(f"Update {update + 1}/{args.updates}: weighted loss = {loss.item():.6f}")

            iteration_metrics.append(
                {
                    "iteration": iteration + 1,
                    "returns": returns,
                    "weights": weights.detach().cpu().tolist(),
                    "losses": losses,
                }
            )
            if (iteration + 1) % args.checkpoint_freq == 0:
                checkpoint_dir = args.output_dir / f"checkpoint-{iteration + 1:03d}"
                policy.save_pretrained(checkpoint_dir)
                policy_preprocess.save_pretrained(checkpoint_dir, config_filename="policy_preprocessor.json")
                policy_postprocess.save_pretrained(checkpoint_dir, config_filename="policy_postprocessor.json")
                print(f"Saved iteration checkpoint: {checkpoint_dir}")
    finally:
        env.close()

    policy.save_pretrained(args.output_dir)
    policy_preprocess.save_pretrained(args.output_dir, config_filename="policy_preprocessor.json")
    policy_postprocess.save_pretrained(args.output_dir, config_filename="policy_postprocessor.json")
    metrics = {
        "reward": "sum of positive-X end-effector displacement",
        "iterations": iteration_metrics,
        "updates": args.updates,
        "rollouts_per_iteration": args.rollouts,
        "checkpoint_frequency": args.checkpoint_freq,
        "trained_parameters": ["model.action_out_proj.weight", "model.action_out_proj.bias"],
    }
    (args.output_dir / "rwft_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved fine-tuned policy and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
