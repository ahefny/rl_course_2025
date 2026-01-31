import gymnasium as gym
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
env = gym.make("CartPole-v1")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

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

obs, _ = env.reset()

episode_reward = 0
for step in range(1000000):    
    state = torch.tensor(obs, dtype=torch.float32)
    
    # Get action from policy
    dist = policy(state)
    action = dist.sample()
    
    # Take step in environment
    next_obs, reward, terminated, truncated, _ = env.step(action.item())
    end_episode = terminated or truncated
    episode_reward += reward
    
    # Store transition
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    next_states.append(torch.tensor(next_obs, dtype=torch.float32))
    dones.append(terminated)
    
    step_count += 1
    
    # Update using TD(0) bootstrapping
    if step_count >= update_frequency or end_episode:
        # Convert to tensors
        states_batch = torch.stack(states)
        actions_batch = torch.stack(actions)
        rewards_batch = torch.tensor(rewards, dtype=torch.float32)
        next_states_batch = torch.stack(next_states)
        dones_batch = torch.tensor(dones, dtype=torch.bool)
        
        # Compute TD(0) targets: r + gamma * V(s') for non-terminal, r for terminal
        with torch.no_grad():
            next_values = value_fn(next_states_batch)
            # For terminal states, next value is 0
            td_targets = rewards_batch + gamma * next_values * (~dones_batch).float()
        
        # Compute values before update (for advantage computation)
        values = value_fn(states_batch)
        
        # ----- Value loss (TD(0) target) -----
        value_loss = ((td_targets - values) ** 2).mean()
        
        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()
        
        # ----- Policy loss (using TD(0) advantages) -----
        # TD(0) advantage: A(s,a) = r + gamma * V(s') - V(s)
        # Use values before update (detached) for advantage computation
        advantages = td_targets - values.detach()
        
        dist = policy(states_batch)
        log_probs = dist.log_prob(actions_batch)
        
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
        
        if end_episode:
            episode_count += 1
            if episode_count % 50 == 0:
                print(f"Episode {episode_count}, Return: {episode_reward:.1f}, Policy Loss: {policy_loss:.1f}, Value Loss: {value_loss:.1f}")
            episode_reward = 0
                # print(f"Episode {episode_count}, Step {step}, Policy Loss: {policy_loss:.3f}, Value Loss: {value_loss:.3f}")
    
    if end_episode:
        obs, _ = env.reset()
    else:
        obs = next_obs
