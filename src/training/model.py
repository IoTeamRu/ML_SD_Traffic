"""Model's description.

File:
    model.py

Classes:
    Model

Description:
    This module should be used as a part of the DQN agent's training logic

"""
import torch
from numpy import int64
from torch import nn


class Model(nn.Module):
    """Dueling DQN implementation.

    We are using a [dueling network](https://papers.labml.ai/paper/1511.06581)
    to calculate Q-values.
    Intuition behind dueling network architecture is that in most states
    the action doesn't matter, and in some states the action is significant.
    Dueling network allows this to be represented very well.

    Args:
        in_ch: number of channels in the input tensor
        num_actions: number of possible actions for the Agent in the environment

    """

    def __init__(self, in_ch: int, num_actions: int64) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_ch, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
        )

        self.lin = nn.Linear(64, 128)
        self.activation = nn.ReLU()

        self.state_value = nn.Linear(128, 1)
        self.action_value = nn.Linear(128, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass function.

        Args:
            obs: input tensor containing observation data.

        Return:
            output tensor containing Q-values for each action
        """
        h = self.pre(obs)

        h = self.activation(self.lin(h))

        action_value = self.action_value(h)
        state_value = self.state_value(h)

        action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
        q = state_value + action_score_centered

        return q
