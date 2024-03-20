import torch

import torch.nn as nn
import torch.optim as optim

class SimpleRlAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = self._build_network()

    def _build_network(self):
        network = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        return network

    def select_action(self, state):
        # Input: state (tensor)
        # Output: action (tensor)
        pass

    def update(self, state, action, reward, next_state, done):
        # Input: state (tensor), action (tensor), reward (float), next_state (tensor), done (bool)
        # Output: None
        pass

    def save_model(self, filepath):
        # Input: filepath (str)
        # Output: None
        pass

    def load_model(self, filepath):
        # Input: filepath (str)
        # Output: None
        pass