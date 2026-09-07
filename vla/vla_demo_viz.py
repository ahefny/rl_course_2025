#!/usr/bin/env python3
"""Run a closed-loop SmolVLA rollout in LIBERO and save its rendered execution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from vla_demo_onesample import DEFAULT_MODEL, adapt_observation_to_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--suite", default="libero_spatial", help="LIBERO task suite.")
    parser.add_argument("--task-id", type=int, default=0, help="Task index within the suite.")
    parser.add_argument("--steps", type=int, default=50, help="Maximum closed-loop environment steps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("vla_outputs"))
    parser.add_argument("--show", action="store_true", help="Open the rollout summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    if args.steps < 1:
        raise ValueError("--steps must be positive.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = SmolVLAPolicy.from_pretrained(args.model).to(device).eval()
    policy_preprocess, policy_postprocess = make_pre_post_processors(
        policy.config,
        args.model,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_config = LiberoEnv(task=args.suite, task_ids=[args.task_id], episode_length=args.steps)
    env = make_env(env_config, n_envs=1, use_async_envs=False)[args.suite][args.task_id]
    env_preprocess, env_postprocess = make_env_pre_post_processors(env_config, policy.config)

    observation, _ = env.reset(seed=args.seed)
    task = env.call("task_description")[0]
    policy.reset()
    images, actions, rewards = [], [], []

    try:
        for step in range(args.steps):
            policy_observation = preprocess_observation(observation)
            policy_observation["task"] = [task]
            policy_observation = env_preprocess(policy_observation)
            policy_observation = adapt_observation_to_policy(
                policy_observation, policy.config.input_features
            )

            with torch.inference_mode():
                action = policy_postprocess(policy.select_action(policy_preprocess(policy_observation)))
            action = env_postprocess({"action": action})["action"]

            action_numpy = action.detach().cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            images.append(env.call("render")[0])
            actions.append(action_numpy[0])
            rewards.append(float(reward[0]))
            if bool(terminated[0] or truncated[0]):
                print(f"Environment ended after {step + 1} steps; info={info}.")
                break
    finally:
        env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "closed_loop_summary.png"
    gif_path = args.output_dir / "closed_loop_execution.gif"
    actions_array = np.asarray(actions)

    figure, (image_ax, action_ax, reward_ax) = plt.subplots(1, 3, figsize=(16, 5))
    image_ax.imshow(images[-1])
    image_ax.set_title("Final rendered LIBERO state")
    image_ax.axis("off")
    for dimension in range(actions_array.shape[1]):
        action_ax.plot(actions_array[:, dimension], label=f"a{dimension}")
    action_ax.set(title="Executed SmolVLA actions", xlabel="Environment step")
    action_ax.legend()
    reward_ax.plot(rewards, marker="o")
    reward_ax.set(title="Simulator reward", xlabel="Environment step")
    figure.suptitle(f"Closed-loop LIBERO task: {task}", wrap=True)
    figure.tight_layout()
    figure.savefig(png_path, dpi=150)

    animation_figure, animation_ax = plt.subplots(figsize=(6, 5))
    display = animation_ax.imshow(images[0])
    animation_ax.axis("off")

    def update(index: int):
        display.set_data(images[index])
        animation_ax.set_title(
            f"Closed-loop step {index + 1} | executed action: "
            f"{actions_array[index].round(3).tolist()}"
        )
        return (display,)

    animation = FuncAnimation(animation_figure, update, frames=len(images), interval=100, blit=True)
    animation.save(gif_path, writer=PillowWriter(fps=10))
    plt.close(animation_figure)
    print(f"Executed {len(actions)} closed-loop steps.")
    print(f"Saved rollout summary: {png_path}")
    print(f"Saved rollout animation: {gif_path}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
