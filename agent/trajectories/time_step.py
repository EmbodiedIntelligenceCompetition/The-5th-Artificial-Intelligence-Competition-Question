# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TimeStep representing a step in the environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import numpy as np
import torch
import torch.nn.functional as F

from agent.specs import array_spec
from agent.specs import tensor_spec

_as_float32_array = functools.partial(np.asarray, dtype=np.float32)

class TimeStep(
    collections.namedtuple('TimeStep',
                           ['step_type', 'reward', 'discount', 'observation', 'info'])):
    """Returned with every call to `step` and `reset` on an environment.

    A `TimeStep` contains the data emitted by an environment at each step of
    interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
    NumPy array or a dict or list of arrays), and an associated `reward` and
    `discount`.

    The first `TimeStep` in a sequence will equal `StepType.FIRST`. The final
    `TimeStep` will equal `StepType.LAST`. All other `TimeStep`s in a sequence
    will equal `StepType.MID.

    Attributes:
        step_type: a `Tensor` or array of `StepType` enum values.
        reward: a `Tensor` or array of reward values.
        discount: A discount value in the range `[0, 1]`.
        observation: A NumPy array, or a nested dict, list or tuple of arrays.
    """
    __slots__ = ()

    def is_first(self):
        return self.step_type == StepType.FIRST

    def is_mid(self):
        return self.step_type == StepType.MID

    def is_last(self):
        return self.step_type == StepType.LAST

    def __hash__(self):
        return hash(tuple(torch.flatten(self)))


class StepType(object):
    """Defines the status of a `TimeStep` within a sequence."""
    # Denotes the first `TimeStep` in a sequence.
    FIRST = np.asarray(0, dtype=np.int32)
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = np.asarray(1, dtype=np.int32)
    # Denotes the last `TimeStep` in a sequence.
    LAST = np.asarray(2, dtype=np.int32)

    def __new__(cls, value):
        """Add ability to create StepType constants from a value."""
        if value == cls.FIRST:
            return cls.FIRST
        if value == cls.MID:
            return cls.MID
        if value == cls.LAST:
            return cls.LAST

        raise ValueError('No known conversion for `%r` into a StepType' % value)


def restart(observation, batch_size=None):
    """Returns a `TimeStep` with `step_type` set equal to `StepType.FIRST`.

    Args:
        observation: A NumPy array, tensor, or a nested dict, list or tuple of
        arrays or tensors.
        batch_size: (Optional) A python or tensor integer scalar.

    Returns:
        A `TimeStep`.
    """
    info = {'done': False, 'success': False, 'path_length': 0.0, 'spl': 0.0, 'episode_length': 0, 'collision_step': 0}
    if not torch.is_tensor(observation):
        if batch_size is not None:
            reward = np.zeros(batch_size, dtype=np.float32)
            discount = np.ones(batch_size, dtype=np.float32)
            step_type = np.tile(StepType.FIRST, batch_size)
            return TimeStep(step_type, reward, discount, observation, info)
        else:
            return TimeStep(
                StepType.FIRST,
                _as_float32_array(0.0),
                _as_float32_array(1.0),
                observation,
                info
            )

    shape = _as_multi_dim(batch_size)
    step_type = torch.full(shape, StepType.FIRST, dtype=torch.int32)
    reward = torch.full(shape, 0.0, dtype=torch.float32)
    discount = torch.full(shape, 1.0, dtype=torch.float32)
    return TimeStep(step_type, reward, discount, observation, info)


def _as_multi_dim(maybe_scalar):
    if maybe_scalar is None:
        shape = ()
    elif torch.is_tensor(maybe_scalar) and maybe_scalar.dim() > 0:
        shape = maybe_scalar.shape
    elif np.asarray(maybe_scalar).ndim > 0:
        shape = maybe_scalar
    else:
        shape = (maybe_scalar,)
    return shape


def transition(observation, reward, info, discount=1.0):
    """Returns a `TimeStep` with `step_type` set equal to `StepType.MID`.

    If `discount` is a scalar, and `observation` contains Tensors,
    then `discount` will be broadcasted to match `reward.shape`.

    Args:
        observation: A NumPy array, tensor, or a nested dict, list or tuple of
        arrays or tensors.
        reward: A scalar, or 1D NumPy array, or tensor.
        discount: (optional) A scalar, or 1D NumPy array, or tensor.

    Returns:
        A `TimeStep`.

    Raises:
        ValueError: If observations are tensors but reward's statically known rank
        is not `0` or `1`.
    """
    if not torch.is_tensor(observation):
        reward = _as_float32_array(reward)
        discount = _as_float32_array(discount)
        if reward.shape:
            step_type = np.tile(StepType.MID, reward.shape)
        else:
            step_type = StepType.MID
        return TimeStep(step_type, reward, discount, observation, info)

    reward = torch.tensor(reward, dtype=torch.float32)
    if reward.dim() not in [0, 1]:
        raise ValueError(f'Expected reward to be a scalar or vector; saw shape: {reward.shape}')
    shape = reward.shape if reward.dim() == 1 else []
    step_type = torch.full(shape, StepType.MID, dtype=torch.int32)
    discount = torch.tensor(discount, dtype=torch.float32)
    if discount.dim() == 0:
        discount = discount.expand(shape)
    else:
        assert discount.shape == reward.shape
    return TimeStep(step_type, reward, discount, observation, info)


def termination(observation, reward, info):
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

    Args:
        observation: A NumPy array, tensor, or a nested dict, list or tuple of
        arrays or tensors.
        reward: A scalar, or 1D NumPy array, or tensor.

    Returns:
        A `TimeStep`.

    Raises:
        ValueError: If observations are tensors but reward's statically known rank
        is not `0` or `1`.
    """
    if not torch.is_tensor(observation):
        reward = _as_float32_array(reward)
        if reward.shape:
            step_type = np.tile(StepType.LAST, reward.shape)
            discount = np.zeros_like(reward, dtype=np.float32)
            return TimeStep(step_type, reward, discount, observation, info)
        else:
            return TimeStep(StepType.LAST, reward, _as_float32_array(0.0),
                            observation, info)

    reward = torch.tensor(reward, dtype=torch.float32)
    if reward.dim() not in [0, 1]:
        raise ValueError(f'Expected reward to be a scalar or vector; saw shape: {reward.shape}')
    shape = reward.shape if reward.dim() == 1 else []
    step_type = torch.full(shape, StepType.LAST, dtype=torch.int32)
    discount = torch.full(shape, 0.0, dtype=torch.float32)
    return TimeStep(step_type, reward, discount, observation, info)


def truncation(observation, reward, discount=1.0):
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

    If `discount` is a scalar, and `observation` contains Tensors,
    then `discount` will be broadcasted to match `reward.shape`.

    Args:
        observation: A NumPy array, tensor, or a nested dict, list or tuple of
        arrays or tensors.
        reward: A scalar, or 1D NumPy array, or tensor.
        discount: (optional) A scalar, or 1D NumPy array, or tensor.

    Returns:
        A `TimeStep`.

    Raises:
        ValueError: If observations are tensors but reward's statically known rank
        is not `0` or `1`.
    """
    if not torch.is_tensor(observation):
        reward = _as_float32_array(reward)
        discount = _as_float32_array(discount)
        if reward.shape:
            step_type = np.tile(StepType.LAST, reward.shape)
        else:
            step_type = StepType.LAST
        return TimeStep(step_type, reward, discount, observation, {})

    reward = torch.tensor(reward, dtype=torch.float32)
    if reward.dim() not in [0, 1]:
        raise ValueError(f'Expected reward to be a scalar or vector; saw shape: {reward.shape}')
    shape = reward.shape if reward.dim() == 1 else []
    step_type = torch.full(shape, StepType.LAST, dtype=torch.int32)
    discount = torch.tensor(discount, dtype=torch.float32)
    if discount.dim() == 0:
        discount = discount.expand(shape)
    else:
        assert discount.shape == reward.shape
    return TimeStep(step_type, reward, discount, observation, {})


def time_step_spec(observation_spec=None):
    """Returns a `TimeStep` spec given the observation_spec."""
    if observation_spec is None:
        return TimeStep(step_type=(), reward=(), discount=(), observation=(),info={})

    first_observation_spec = next(iter(observation_spec.values())) if isinstance(observation_spec, dict) else observation_spec
    if isinstance(first_observation_spec, (torch.Tensor, tensor_spec.TensorSpec, tensor_spec.BoundedTensorSpec)):
        return TimeStep(
            step_type=tensor_spec.TensorSpec([], torch.int32, name='step_type'),
            reward=tensor_spec.TensorSpec([], torch.float32, name='reward'),
            discount=tensor_spec.BoundedTensorSpec([], torch.float32, minimum=0.0, maximum=1.0, name='discount'),
            observation=observation_spec,
            info={})
    return TimeStep(
        step_type=array_spec.ArraySpec([], np.int32, name='step_type'),
        reward=array_spec.ArraySpec([], np.float32, name='reward'),
        discount=array_spec.BoundedArraySpec([], np.float32, minimum=0.0, maximum=1.0, name='discount'),
        observation=observation_spec,
        info={})
