from typing import Optional, Tuple, Union, Dict
import copy

import torch
import torch.nn as nn
from torch.distributions import Normal
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from stable_baselines3.common.type_aliases import PyTorchObs


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,) for (n_batch, n_actions) input, scalar for (n_batch,) input
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class LowDimPolicy(nn.Module):  

    def __init__(self, obs_dim: int, action_space: gym.spaces.Box, log_std_init: float=0):
        """Build neural networks for action prediction and value estimation.

        - The actor is a probabilistic policy network that predicts the mean
        and uses the larnable standard deviation to parameterize a Gaussian
        distribution.  Actions are sampled from the predicted distribution.

        -  The actor is a 3-layer MLP with 64 hidden units and tanh activation
        functions that predicts the sample mean.

        - The value network is a 3-layer MLP with 64 hidden units and tanh
        activation functions that estimates the value of an input state.

        Args:
            obs_dim (int): An integer representing the dimension of the
                observation space.
            action_space (gym.spaces.Box): A gym.spaces.Box object representing
                the action space, where `low` and `high` indicate the lower and
                upper bounds of the action space.
                Check https://www.gymlibrary.dev/api/spaces/#box for details.
        """
        super().__init__()
        self.action_space = action_space
        self.obs_dim = obs_dim
        self.log_std_init = log_std_init
        self.action_net, self.log_std = self.build_actor_net()
        self.value_net = self.build_value_net(obs_dim)

    def build_actor_net(self) -> nn.Module:
        # The total number of elements in the output tensor
        output_nele = 1
        for i in self.action_space.shape:
            output_nele *= i

        actor = nn.Sequential(
            nn.Linear(self.obs_dim, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, output_nele, bias=True),
        )

        log_std = nn.Parameter(torch.ones(output_nele) * self.log_std_init,
                               requires_grad=True)

        return actor, log_std

    def build_value_net(self, obs_dim: int) -> nn.Module:
        """Build a MLP for the value network.

        Args:
            obs_dim (int): An integer representing the dimension of the
                observation space.
        """
        value_net = nn.Sequential(
            nn.Linear(obs_dim, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, 1, bias=True),
        )

        return value_net

    def forward(self, obs: torch.Tensor, deterministic: bool=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the policy network.

        Args:
            obs (torch.Tensor): A tensor representing the observation.

        Returns:
            actions (torch.Tensor): A tensor representing the sampled action.
            value (torch.Tensor): A tensor representing the predicted action.
            log_prob (torch.Tensor): A tensor representing the log probability
                of the sampled action.
        """
        action, log_prob = self.predict_action(obs, deterministic)
        value = self.predict_value(obs)
        return action, value, log_prob

    def predict_action(self, obs: torch.Tensor,
                       deterministic: Optional[bool]=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the action given the observation.

        Args:
            obs (torch.Tensor): A tensor representing the observation.
            deterministic (bool): Disable to sample an action from the predicted
                distribution.  If True, return the mean of the predicted;
                otherwise, sample an action from the predicted distribution.

        Returns:
            actions (torch.Tensor): A tensor representing the sampled action.
            log_prob (torch.Tensor): A tensor representing the log probability
                of the sampled action.
        """
        # TODO: Return the predicted mean

        # 1. Compute the mean and standard deviation of the actions.
        # actions_mean = ...
        # actions_std = ...

        # 2. Create a normal distribution with the predicted mean and std
        # distribution = ...

        # 3. Sample an action from the distribution
        # actions = ...

        # 4. Compute the log probability of the actions and sum it across dimensions (you can use sum_independent_dims)
        # log_prob = ...

        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, log_prob

    def predict_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict the action given the observation.

        Args:
            obs (torch.Tensor): A tensor representing the observation.

        Returns:
            value (torch.Tensor): A tensor representing the predicted action.
        """
        # TODO: Return the predicted value
        # return  ...

    def evaluate_action(self, obs: PyTorchObs, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # TODO: Return the distribution with predicted mean and std
        # actions_mean = ...
        # actions_std = ...
        # distribution = ...

        # TODO: Evaluate the probability of input actions with the distribution
        # log_prob = ...

        # TODO: Return the entropy of the distribution
        # entropy = ...

        log_prob = sum_independent_dims(log_prob)
        entropy = sum_independent_dims(entropy)
        return log_prob, entropy
