"""
PPO training on Mujoco environments using Stable-Baselines3.
Uses vectorized environments with reward and observation scaling.
Logs metrics (and periodic evaluation videos) to TensorBoard.
"""

import datetime
import os
from dataclasses import dataclass, asdict
import json

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from experiment_utils import EpisodeEvalCallback, NormalizedEnvWrapper


@dataclass
class PPOTrainingConfig:
    # Policy
    policy: str = "MlpPolicy"
    
    # Common parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Learning parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Number of steps to collect per rollout
    
    # PPO specific parameters
    batch_size: int = 64  # Minibatch size for PPO updates
    n_epochs: int = 10  # Number of epochs for PPO updates
    clip_range: float = 0.2  # PPO clipping parameter
    ent_coef: float = 0.0  # Entropy coefficient
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Maximum gradient norm for clipping
    
    # Number of parallel environments
    n_envs: int = 1
    
    # Normalization parameters
    norm_obs: bool = True  # Normalize observations
    norm_reward: bool = True  # Normalize rewards
    clip_obs: float = 10.0  # Clip observations after normalization
    clip_reward: float = 10.0  # Clip rewards after normalization
    
    # Training
    total_timesteps: int = 1_000_000

# ENVIRONMENT_NAME = "Hopper-v5"
ENVIRONMENT_NAME = "Walker2d-v5"
CONFIG = PPOTrainingConfig()


def main():    
    environment_label = ENVIRONMENT_NAME.lower()
    config = CONFIG

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{environment_label}-ppo-{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/config.json", 'w') as f:
        json.dump(asdict(config), f)

    print(f"Logging to: {log_dir}")    

    # Training env (vectorized for throughput)
    env = make_vec_env(ENVIRONMENT_NAME, n_envs=config.n_envs)
    
    # Wrap with VecNormalize for observation and reward scaling
    env = VecNormalize(
        env,
        norm_obs=config.norm_obs,
        norm_reward=config.norm_reward,
        clip_obs=config.clip_obs,
        clip_reward=config.clip_reward,
        gamma=config.gamma,
    )

    # Evaluation env (single, with rgb frames for logging)
    # Wrap with NormalizedEnvWrapper to use training env's normalization stats
    eval_env_raw = gym.make(ENVIRONMENT_NAME, render_mode="rgb_array")
    eval_env = NormalizedEnvWrapper(eval_env_raw, env)

    # Create PPO model with TensorBoard logging
    # Exclude fields that aren't PPO constructor parameters
    ppo_params = {k: v for k, v in asdict(config).items() 
                  if k not in ['policy', 'total_timesteps', 'n_envs', 
                              'norm_obs', 'norm_reward', 'clip_obs', 'clip_reward']}
        
    model = PPO(
        config.policy,
        env,
        **ppo_params,       
        verbose=1,
        tensorboard_log=log_dir,  # Enable TensorBoard logging
        device="cpu", # Use CPU for training.
    )

    # Use standard EpisodeEvalCallback - the NormalizedEnvWrapper automatically
    # uses the latest normalization stats from the training VecNormalize since
    # it holds a reference to the same object
    eval_callback = EpisodeEvalCallback(
        eval_env=eval_env,
        eval_every_episodes=100,
        record_every_episodes=500,
        max_steps=1000,
        verbose=1,
        deterministic=False,
        record_frame_skip=4,
    )

    # Train the model
    print("Starting training...")
    
    model.learn(
        total_timesteps=config.total_timesteps,
        log_interval=10,
        callback=eval_callback,
    )

    # Save the model and normalization stats
    model_path = os.path.join(log_dir, environment_label)
    model.save(model_path)
    # Save VecNormalize statistics
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    print(f"Model saved to: {model_path}")
    print(f"Normalization stats saved to: {os.path.join(log_dir, 'vec_normalize.pkl')}")

    # Test the trained model (deterministic)
    # Normalization stats are automatically synced via the wrapper
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
