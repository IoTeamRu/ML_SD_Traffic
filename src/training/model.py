import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, in_ch, num_actions):
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

    def forward(self, obs: torch.Tensor):
        h = self.pre(obs)

        h = self.activation(self.lin(h))

        action_value = self.action_value(h)
        state_value = self.state_value(h)

        action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
        q = state_value + action_score_centered

        return q
