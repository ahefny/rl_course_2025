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
policy_optim = optim.Adam(policy.parameters(), lr=1e-3)

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
    
    returns = compute_returns(rewards, gamma)
    # Uncomment to disable the causality trick.
    # returns = torch.sum(torch.tensor(rewards))

    # ----- Policy loss -----
    dist = policy(states)
    log_probs = dist.log_prob(actions)

    policy_loss = -(log_probs * returns).mean()

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, Return: {sum(rewards):.1f}, Policy Loss: {policy_loss:.1f}")
