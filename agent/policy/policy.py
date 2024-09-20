import abc

import torch
from gym import spaces
from torch import nn as nn
from agent.gibson_extension.utils.common import CategoricalNet
from agent.common.common import batch_obs, ObservationBatchingCache
from agent.common.obs_transformers import (
    get_active_obs_transforms,
    apply_obs_transforms_obs_space,
    apply_obs_transforms_batch
    
)

class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions, num_envs):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        self.test_recurrent_hidden_states = torch.zeros(
            num_envs,
            self.net.num_recurrent_layers,
            self.net._hidden_size,
            device=self.device,
        )
        self.prev_actions = torch.zeros(
            num_envs,
            2,
            device=self.device,
            dtype=torch.float32,
        )
        self.not_done_masks = torch.zeros(
            num_envs,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        

        self._obs_batching_cache = ObservationBatchingCache()
        
    def is_transformed(self, d: dict) -> bool:
        """Check if any value in the dictionary is of type torch.Tensor.

        Args:
            d (dict): The dictionary to check.

        Returns:
            bool: True if any value is a torch.Tensor, False otherwise.
        """
        for value in d.values():
            if isinstance(value, torch.Tensor):
                return True
        return False

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        if not self.is_transformed(observations):
            batch = batch_obs(
                observations, device=self.device, cache=self._obs_batching_cache
            )
            observations = apply_obs_transforms_batch(batch, self.obs_transforms)

        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        dist = self.action_distribution(features)
        value = self.critic(features)

        action = dist.sample()
        action_log_prob = dist.log_prob(action)


        return value, action, action_log_prob, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):

        if not self.is_transformed(observations):
            batch = batch_obs(
                observations, device=self.device, cache=self._obs_batching_cache
            )
            observations = apply_obs_transforms_batch(batch, self.obs_transforms)

        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        
        if not self.is_transformed(observations):
            batch = batch_obs(
                observations, device=self.device, cache=self._obs_batching_cache
            )
            observations = apply_obs_transforms_batch(batch, self.obs_transforms)

        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        
        dist = self.action_distribution(features)
        value = self.critic(features)

        action_log_prob = dist.log_prob(action)


        return value, action, action_log_prob, rnn_hidden_states

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)
    



class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass