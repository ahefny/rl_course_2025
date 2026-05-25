"""
DQN training on Gymnasium using Stable-Baselines3.
Logs metrics (and periodic evaluation videos) to TensorBoard.
"""

import datetime
import os
from dataclasses import dataclass, asdict
import json

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from experiment_utils import EpisodeEvalCallback


@dataclass
class DQNTrainingConfig:
    # Policy
    policy: str = "MlpPolicy"
    
    # Learning parameters
    learning_rate: float = 1e-2
    gamma: float = 0.99
    n_steps: int = 1    
    
    # Replay buffer
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    batch_size: int = 32
    
    # Training frequency
    train_freq: int = 4 # This is #steps_per_update
    gradient_steps: int = 1 # SB3 allows multiple gradient steps per update

    # Number of parallel environments.
    # SB3 divides #steps_per_update across envs.
    n_envs: int = 1
    
    # Target network update
    tau: float = 0.1
    target_update_interval: int = 1
    
    # Exploration
    exploration_fraction: float = 0.01
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
        
    # Training
    total_timesteps: int = 1_000_000

ENVIRONMENT_NAME = "CartPole-v1"
CONFIG = DQNTrainingConfig()


def main():    
    environment_label = ENVIRONMENT_NAME.lower()
    config = CONFIG

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{environment_label}-dqn-{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/config.json", 'w') as f:
        json.dump(asdict(config), f)

    print(f"Logging to: {log_dir}")    

    # Training env (vectorized for throughput)
    env = make_vec_env(ENVIRONMENT_NAME, n_envs=config.n_envs)

    # Evaluation env (single, with rgb frames for logging)
    eval_env = gym.make(ENVIRONMENT_NAME, render_mode="rgb_array")

    # Create DQN model with TensorBoard logging
    # Exclude fields that aren't DQN constructor parameters
    dqn_params = {k: v for k, v in asdict(config).items() 
                  if k not in ['policy', 'total_timesteps', 'n_envs']}
        
    model = DQN(
        config.policy,
        env,
        **dqn_params,       
        verbose=1,
        tensorboard_log=log_dir,  # Enable TensorBoard logging
    )

    eval_callback = EpisodeEvalCallback(
        eval_env=eval_env,
        eval_every_episodes=100,    # How often to run evaluation.
        record_every_episodes=500,  # How often to record video.
        max_steps=500,
        verbose=1,
    )

    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=config.total_timesteps,
        log_interval=10,
        callback=eval_callback,
    )

    # Save the model
    model_path = os.path.join(log_dir, environment_label)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Test the trained model (deterministic)
    print("\nTesting trained model...")
    obs, _ = eval_env.reset()
    total_reward = 0.0
    for _ in range(500):
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

