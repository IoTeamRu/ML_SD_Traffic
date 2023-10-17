from typing import Tuple
import torch
from torch import nn


class QFunc(nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.huber_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, q: torch.Tensor, action: torch.Tensor, double_q: torch.Tensor,
        target_q: torch.Tensor, done: torch.Tensor, reward: torch.Tensor,
        weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        q_sampled_action = q.gather(-1, action.to(torch.long).unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            best_next_action = torch.argmax(double_q, -1)
            best_next_q_value = target_q.gather(-1, best_next_action.unsqueeze(-1)).squeeze(-1)
            q_update = reward + self.gamma * best_next_q_value * (1 - done)

            td_error = q_sampled_action - q_update
        losses = self.huber_loss(q_sampled_action, q_update)
        loss = torch.mean(weights * losses)

        return td_error, loss
