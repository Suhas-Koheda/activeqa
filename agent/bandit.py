"""
Bandit algorithms: LinUCB and Neural Policy-Gradient bandit.
"""
from __future__ import annotations
import numpy as np
import torch
from config import N_ACTIONS, LINUCB_ALPHA


class LinUCB:
    """
    Linear Upper Confidence Bound contextual bandit.
    Each action has its own ridge-regression parameters.
    """
    def __init__(self, dim: int, n_actions: int = N_ACTIONS, alpha: float = LINUCB_ALPHA):
        self.dim       = dim
        self.n_actions = n_actions
        self.alpha     = alpha
        # One A matrix and b vector per action
        self.A = [np.eye(dim) for _ in range(n_actions)]
        self.b = [np.zeros(dim) for _ in range(n_actions)]

    def select(self, x: np.ndarray) -> int:
        """Choose action with highest UCB score."""
        best_action  = 0
        best_score   = -np.inf
        for a in range(self.n_actions):
            A_inv    = np.linalg.inv(self.A[a])
            theta    = A_inv @ self.b[a]
            score    = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
            if score > best_score:
                best_score  = score
                best_action = a
        return best_action

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        """Update A and b for the chosen action."""
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x


def train_linucb(states: np.ndarray, rewards_matrix: np.ndarray,
                 n_actions: int = N_ACTIONS, alpha: float = LINUCB_ALPHA) -> LinUCB:
    """
    Train LinUCB on pre-computed state/reward data.

    Args:
        states:         (N, D) array of state vectors
        rewards_matrix: (N, N_ACTIONS) reward for every action per state
    """
    dim = states.shape[1]
    agent = LinUCB(dim, n_actions, alpha)
    for i in range(len(states)):
        x = states[i]
        # Choose the best-known action (exploit) for offline training
        action = int(np.argmax(rewards_matrix[i]))
        reward = rewards_matrix[i, action]
        agent.update(x, action, reward)
    return agent


def train_neural_bandit(agent, states: list[np.ndarray],
                        rewards_matrix: np.ndarray, epochs: int = 10,
                        batch_size: int = 64) -> list[float]:
    """
    Train the NeuralBandit agent using REINFORCE on pre-computed rewards.

    Args:
        agent:          NeuralBandit instance
        states:         list of state vectors
        rewards_matrix: (N, N_ACTIONS) reward for every action
        epochs:         number of training epochs
    """
    history = []
    n = len(states)
    for epoch in range(epochs):
        epoch_loss = 0.0
        indices   = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx   = indices[start:start + batch_size]
            batch_loss  = 0.0
            for i in batch_idx:
                state  = states[i]
                # Sample action from current policy
                action, log_prob = agent.act(state)
                # Use the true reward for that action
                reward = rewards_matrix[i, action]
                loss   = agent.update(state, action, reward, log_prob)
                batch_loss += loss
            epoch_loss += batch_loss
        avg_loss = epoch_loss / n
        history.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{epochs}  —  avg loss: {avg_loss:.4f}")
    return history
