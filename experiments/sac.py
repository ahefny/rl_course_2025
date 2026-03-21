"""
SAC training on Mujoco environments using Stable-Baselines3.
Uses vectorized environments with reward and observation scaling.
Logs metrics (and periodic evaluation videos) to TensorBoard.
"""

import argparse
import datetime
import os
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional, Literal

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import yaml

try:
    # When run as `python -m experiments.sac`
    from .experiment_utils import (
        EpisodeEvalCallback,
        NormalizedEnvWrapper,
        get_linear_learning_rate_schedule,
    )
except ImportError:
    # When run as `python experiments/sac.py`
    from experiment_utils import (
        EpisodeEvalCallback,
        NormalizedEnvWrapper,
        get_linear_learning_rate_schedule,
    )


@dataclass
class SACTrainingConfig:
    # Policy
    policy: str = "MlpPolicy"

    # Common parameters
    gamma: float = 0.99

    # Learning parameters
    learning_rate: float = 3e-4
    learning_rate_final: Optional[float] = None  # [Use None for no decay]

    # Off-policy / replay buffer
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 256

    # SAC specific parameters
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    target_update_interval: int = 1
    target_entropy: Literal["auto"] | float = "auto"

    # Exploration
    use_sde: bool = False
    sde_sample_freq: int = -1

    # Number of parallel environments
    n_envs: int = 1

    # Normalization parameters
    norm_obs: bool = True  # Normalize observations
    norm_reward: bool = True  # Normalize rewards
    clip_obs: float = 20.0  # Clip observations after normalization
    clip_reward: float = 20.0  # Clip rewards after normalization

    # Training
    total_timesteps: int = 100_000


# ENVIRONMENT_NAME = "Hopper-v5"
# ENVIRONMENT_NAME = "Walker2d-v5"
# ENVIRONMENT_NAME = "BipedalWalker-v3"
# ENVIRONMENT_NAME = "Humanoid-v5"
ENVIRONMENT_NAME = "metaworld_reach-v3"
CONFIG = SACTrainingConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC agent.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file for SACTrainingConfig.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the run.",
    )
    return parser.parse_args()


def load_config_from_yaml(path: Path | None) -> SACTrainingConfig:
    """Load SACTrainingConfig from YAML, falling back to defaults when not provided."""
    if path is None:
        return CONFIG

    with path.open("r") as f:
        data = yaml.safe_load(f) or {}

    # Start from defaults and override with any keys provided in YAML
    base = asdict(CONFIG)
    base.update(data)
    return SACTrainingConfig(**base)


def main():
    args = parse_args()
    environment_label = ENVIRONMENT_NAME.lower()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config = load_config_from_yaml(config_path)

    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{environment_label}-sac-{timestamp}"
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"Logging to: {log_dir}")

    if ENVIRONMENT_NAME == "metaworld_reach-v3":
        import metaworld
        from stable_baselines3.common.vec_env import DummyVecEnv

        def _create_env(render_mode):
            ml1 = metaworld.ML1("reach-v3")
            env = ml1.train_classes["reach-v3"](render_mode=render_mode)
            task = ml1.train_tasks[0]
            env.set_task(task)
            return env

        env = DummyVecEnv([lambda: _create_env("none") for _ in range(config.n_envs)])
        eval_env = _create_env(render_mode="rgb_array")
    else:
        # Training env (vectorized for throughput)
        env = make_vec_env(ENVIRONMENT_NAME, n_envs=config.n_envs)
        # Evaluation env (single, with rgb frames for logging)
        eval_env = gym.make(ENVIRONMENT_NAME, render_mode="rgb_array")

    # Wrap training env with VecNormalize for observation and reward scaling
    env = VecNormalize(
        env,
        norm_obs=config.norm_obs,
        norm_reward=config.norm_reward,
        clip_obs=config.clip_obs,
        clip_reward=config.clip_reward,
        gamma=config.gamma,
    )
    # Wrap evaluation env with NormalizedEnvWrapper to use training env's normalization stats
    eval_env = NormalizedEnvWrapper(eval_env, env)

    # Exclude fields that aren't SAC constructor parameters
    sac_params = {
        k: v
        for k, v in asdict(config).items()
        if k
        not in [
            "policy",
            "total_timesteps",
            "n_envs",
            "norm_obs",
            "norm_reward",
            "clip_obs",
            "clip_reward",
            "learning_rate",
            "learning_rate_final",
        ]
    }

    model = SAC(
        config.policy,
        env,
        **sac_params,
        learning_rate=get_linear_learning_rate_schedule(
            lr_inital=config.learning_rate,
            lr_final=config.learning_rate_final or config.learning_rate,
        ),
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda",  # Use CPU for training.
    )

    eval_callback = EpisodeEvalCallback(
        eval_env=eval_env,
        eval_every_episodes=10,
        record_every_episodes=100,
        max_steps=1000,
        verbose=1,
        deterministic=True,
        record_frame_skip=4,
    )

    print("Starting training...")
    model.learn(
        total_timesteps=config.total_timesteps,
        log_interval=10,
        callback=eval_callback,
    )

    # Save the model and normalization stats
    model_path = os.path.join(log_dir, environment_label)
    model.save(model_path)
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    print(f"Model saved to: {model_path}")
    print(f"Normalization stats saved to: {os.path.join(log_dir, 'vec_normalize.pkl')}")

    # Test the trained model (deterministic)
    print("\nTesting trained model...")
    obs, _ = eval_env.reset()
    total_reward = 0.0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    print(f"Test episode reward: {total_reward}")
    print(f"\nTensorBoard logs available at: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()

