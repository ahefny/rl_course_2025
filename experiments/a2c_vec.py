import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -----------------------------
# Networks
# -----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return torch.distributions.Categorical(logits=self.net(x))


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -----------------------------
# Training Loop
# -----------------------------
def make_env(env_name):
    """Factory function to create a single environment."""
    def _make():
        return gym.make(env_name)
    return _make

# Create vectorized environment
num_envs = 8
env_name = "LunarLander-v3"
env = gym.make_vec(env_name, num_envs=num_envs)
#AsyncVectorEnv([make_env(env_name) for _ in range(n_envs)])

obs_dim = env.single_observation_space.shape[0]
act_dim = env.single_action_space.n

policy = PolicyNet(obs_dim, act_dim)
value_fn = ValueNet(obs_dim)

policy_optim = optim.Adam(policy.parameters(), lr=1e-3)
value_optim = optim.Adam(value_fn.parameters(), lr=1e-2)

gamma = 0.99
update_frequency = 5  # Update every N steps (for stability, but using TD(0) bootstrapping)

states = []
actions = []
rewards = []
next_states = []
dones = []

step_count = 0
episode_count = 0
episode_returns = []  # Track returns for completed episodes

obs, _ = env.reset()
episode_rewards = np.zeros(num_envs)  # Track rewards per environment

for step in range(1000000):    
    state = torch.tensor(obs, dtype=torch.float32)  # Shape: [n_envs, obs_dim]
    
    # Get actions from policy (batched)
    dist = policy(state)
    action = dist.sample()  # Shape: [n_envs]
    
    # Take step in environment (batched)
    next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
    end_episode = terminated | truncated  # Shape: [n_envs]
    episode_rewards += reward
    
    # Store transition
    states.append(state)
    actions.append(action)
    rewards.append(torch.tensor(reward, dtype=torch.float32))
    next_states.append(torch.tensor(next_obs, dtype=torch.float32))
    dones.append(torch.tensor(terminated, dtype=torch.bool))
    
    step_count += 1
    
    # Track completed episodes
    for env_idx in range(num_envs):
        if end_episode[env_idx]:
            episode_count += 1
            episode_returns.append(episode_rewards[env_idx])
            episode_rewards[env_idx] = 0.0
    
    # Update using TD(0) bootstrapping
    if step_count >= update_frequency or end_episode.any():
        # Convert to tensors
        states_batch = torch.stack(states)  # Shape: [n_steps, n_envs, obs_dim]
        actions_batch = torch.stack(actions)  # Shape: [n_steps, n_envs]
        rewards_batch = torch.stack(rewards)  # Shape: [n_steps, n_envs]
        next_states_batch = torch.stack(next_states)  # Shape: [n_steps, n_envs, obs_dim]
        dones_batch = torch.stack(dones)  # Shape: [n_steps, n_envs]
        
        # Flatten for batch processing: [n_steps * n_envs, ...]
        n_steps = states_batch.shape[0]
        states_flat = states_batch.view(-1, obs_dim)
        actions_flat = actions_batch.view(-1)
        rewards_flat = rewards_batch.view(-1)
        next_states_flat = next_states_batch.view(-1, obs_dim)
        dones_flat = dones_batch.view(-1)
        
        # Compute TD(0) targets: r + gamma * V(s') for non-terminal, r for terminal
        with torch.no_grad():
            next_values = value_fn(next_states_flat)  # Shape: [n_steps * n_envs]
            # For terminal states, next value is 0
            td_targets = rewards_flat + gamma * next_values * (~dones_flat).float()
        
        # Compute values before update (for advantage computation)
        values = value_fn(states_flat)  # Shape: [n_steps * n_envs]
        
        # ----- Value loss (TD(0) target) -----
        value_loss = ((td_targets - values) ** 2).mean()
        
        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()
        
        # ----- Policy loss (using TD(0) advantages) -----
        # TD(0) advantage: A(s,a) = r + gamma * V(s') - V(s)
        # Use values before update (detached) for advantage computation
        advantages = td_targets - values.detach()
        
        dist = policy(states_flat)
        log_probs = dist.log_prob(actions_flat)
        
        policy_loss = -(log_probs * advantages).mean()
        
        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()
        
        # Clear buffers
        states.clear()
        actions.clear()
        rewards.clear()
        next_states.clear()
        dones.clear()
        step_count = 0
        
        # Print evaluation metrics including episode returns
        if len(episode_returns) > 0 and episode_count % 50 == 0:
            n_recent = min(50, len(episode_returns))
            recent_returns = episode_returns[-n_recent:]
            mean_return = np.mean(recent_returns)
            max_return = np.max(recent_returns)
            min_return = np.min(recent_returns)
            print(f"Episode {episode_count}, "
                  f"Return: {mean_return:.1f} (min: {min_return:.1f}, max: {max_return:.1f}), "
                  f"Policy Loss: {policy_loss:.3f}, Value Loss: {value_loss:.3f}")
    
    # Vectorized environments automatically reset terminated/truncated envs
    # next_obs already contains reset observations for those envs
    obs = next_obs
