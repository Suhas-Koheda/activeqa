"""
Neural policy network for the contextual bandit.
Maps state → action logits.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from config import N_ACTIONS, HIDDEN_DIM


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralBandit:
    """Wrapper around PolicyNetwork with train / act / save / load."""

    def __init__(self, input_dim: int, lr: float = 0.01, device: str = "cuda"):
        self.device = device
        self.policy = PolicyNetwork(input_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        """Sample an action from the policy distribution."""
        import numpy as np
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy(s)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def act_greedy(self, state: np.ndarray) -> int:
        import numpy as np
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy(s)
        return logits.argmax(dim=-1).item()

    def update(self, state: np.ndarray, action: int, reward: float,
               log_prob: torch.Tensor) -> float:
        """Policy-gradient (REINFORCE) update."""
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
