"""Agent's Learner.

File:
    learner.py

Classes:
    Learner

Description:
    This module should be used as a part of the DQN agent's training logic

"""
import copy
import math
import random

import numpy as np
import torch
from box.config_box import ConfigBox
from model import Model
from replay import ReplayBuffer
from sumo_rl.environment.env import SumoEnvironment
from torch import nn


class Learner:
    """This class describes how the Agent learns towards the optimal policy."""

    def __init__(
        self, env: SumoEnvironment, config: ConfigBox, device: torch.device
    ) -> None:
        self.cfg = config
        self.action_size = env.action_space.n
        self.model = Model(env.observation_space.shape[0], self.action_size)
        self.target = copy.deepcopy(self.model)
        self.model.to(device)
        self.target.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.memory = ReplayBuffer(
            self.cfg.exp_replay_size, self.cfg.alpha, env.observation_space.shape[0]
        )
        self.loss_fn = nn.HuberLoss(reduction="none")
        self.update_count = 0
        self.device = device
        self.epsilon = lambda step_idx: self.cfg.epsilon_final + (
            self.cfg.epsilon_start - self.cfg.epsilon_final
        ) * math.exp(-1.0 * step_idx / self.cfg.epsilon_decay)

        self.beta = lambda step_idx: self.cfg.beta_stop + (
            self.cfg.beta - self.cfg.beta_stop
        ) * math.exp(-1.0 * step_idx / self.cfg.beta_raise)

    def lr_decay(self, step: int) -> None:
        """Decrease learning rate.

        Args:
            step: the current step number
        """
        lr_now = 0.9 * self.cfg.lr * (1 - step / self.cfg.steps) + 0.1 * self.cfg.lr
        for p in self.optimizer.param_groups:
            p["lr"] = lr_now

    def save(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path: the path to save the model to
        """
        torch.save(self.model.state_dict(), path)

    def obs_to_torch(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a numpy array to a torch tensor.

        Args:
            obs: a numpy array to be converted.

        Return:
            The converted numpy array.
        """
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def act(self, state: np.ndarray, step: int) -> int:
        """Get an e-greedy action.

        Depends on a current step size an action is chosen randomly or
        accoding to the current policy.

        Args:
            state: the state of the environment
            step: the current step number

        Return:
            the action
        """
        if np.random.rand() <= self.epsilon(step):
            return random.randrange(self.action_size)
        with torch.no_grad():
            act_values = self.model(self.obs_to_torch(state))
        return np.argmax(act_values.cpu().numpy())

    def calculate_loss(
        self, sample: dict[str, np.ndarray]
    ) -> tuple[np.array, torch.Tensor]:
        """Double DQN algorithm.

        TD-error calculated based on a sample taken from Priority
        Replay Buffer. Next priority of the sample will be calculated
        based on the TD-error.

        Args:
            sample: a dict that contains (s, a, r, d, n)

        Return:
            td_error: a value to update priority of the sample in the PRB
            loss: the value of loss for backprop
        """
        q_value = self.model(self.obs_to_torch(sample["obs"]))

        with torch.no_grad():
            double_q_value = self.model(self.obs_to_torch(sample["next_obs"]))
            target_q_value = self.target(self.obs_to_torch(sample["next_obs"]))

        q_sampled_action = q_value.gather(
            -1, self.obs_to_torch(sample["action"]).to(torch.long).unsqueeze(-1)
        ).squeeze(-1)

        with torch.no_grad():
            best_next_action = torch.argmax(double_q_value, -1)
            best_next_q_value = target_q_value.gather(
                -1, best_next_action.unsqueeze(-1)
            ).squeeze(-1)
            q_update = self.obs_to_torch(
                sample["reward"]
            ) + self.cfg.gamma * best_next_q_value * (
                1 - self.obs_to_torch(sample["done"])
            )

            td_error = q_sampled_action - q_update
        td_error = np.abs(td_error.cpu().numpy()) + 1e-6
        losses = self.loss_fn(q_sampled_action, q_update)
        loss = torch.mean(self.obs_to_torch(sample["weights"]) * losses)

        return td_error, loss

    def learn(self, step: int) -> None:
        """Update agent's parameters.

        Describes the logic of how the agent learns:
            - sample data from PRB
            - calculate loss
            - update sample's prority
            - update model's parameters.

        Args:
            step: curent iteration number

        """
        if self.cfg.learn_start > step:
            return

        beta = self.beta(step)
        sample = self.memory.sample(self.cfg.batch_size, beta)

        td_error, loss = self.calculate_loss(sample)

        self.memory.update_priorities(sample["indices"], td_error)

        self.optimizer.zero_grad()
        loss.backward()

        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        if self.cfg.use_soft_update:
            for param, target_param in zip(
                self.model.parameters(), self.target.parameters()
            ):
                target_param.data.copy_(
                    self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data
                )
        else:
            self.update_count += 1
            if self.update_count % self.cfg.target_update_freq == 0:
                self.target.load_state_dict(self.model.state_dict())

        if self.cfg.use_lr_decay:
            self.lr_decay(step)
