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
# Utilities
# -----------------------------
def compute_returns(rewards, gamma):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    return torch.tensor(list(reversed(returns)), dtype=torch.float32)


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

for episode in range(100000):
    obs, _ = env.reset()

    states = []
    actions = []
    rewards = []

    done = False
    while not done:
        state = torch.tensor(obs, dtype=torch.float32)
        dist = policy(state)
        action = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs

    # Convert to tensors
    states = torch.stack(states)
    actions = torch.stack(actions)

    with torch.no_grad():
        last_reward = value_fn(torch.tensor(obs)) if truncated else 0.0    
    returns = compute_returns(rewards + [last_reward], gamma)[..., :-1]

    # ----- Value loss -----
    values = value_fn(states)
    value_loss = ((returns - values) ** 2).mean()

    value_optim.zero_grad()
    value_loss.backward()
    value_optim.step()

    # ----- Policy loss -----
    advantages = returns - values.detach()
    dist = policy(states)
    log_probs = dist.log_prob(actions)

    policy_loss = -(log_probs * advantages).mean()

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, Return: {sum(rewards):.1f}, Policy Loss: {policy_loss:.1f}, Value Loss: {value_loss:.1f}")
