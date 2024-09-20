
from agent.gibson_extension.examples.configs.default import Config
from gym import spaces
import abc
from typing import Tuple,Union,Iterable,List,Dict
import torch
import torch.nn as nn
from agent.gibson_extension.utils.common import (
    center_crop,
    get_image_height_width,
    image_resize_shortest_edge,
    overwrite_gym_box_shape,
)
import copy
import numbers
from absl import logging


class ObservationTransformer(nn.Module, metaclass=abc.ABCMeta):
    """This is the base ObservationTransformer class that all other observation
    Transformers should extend. from_config must be implemented by the transformer.
    transform_observation_space is only needed if the observation_space ie.
    (resolution, range, or num of channels change)."""

    def transform_observation_space(
        self, observation_space: spaces.Dict, **kwargs
    ):
        return observation_space

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Config):
        pass

    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return observations


class ResizeShortestEdge(ObservationTransformer):
    r"""An nn module the resizes your the shortest edge of the input while maintaining aspect ratio.
    This module assumes that all images in the batch are of the same size.
    """

    def __init__(
        self,
        size: int,
        channels_last: bool = True,
        trans_keys: Tuple[str] = ("rgb", "depth", "semantic"),
    ):
        """Args:
        size: The size you want to resize the shortest edge to
        channels_last: indicates if channels is the last dimension
        """
        super(ResizeShortestEdge, self).__init__()
        self._size: int = size
        self.channels_last: bool = channels_last
        self.trans_keys: Tuple[str] = trans_keys

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if key in self.trans_keys:
                    # In the observation space dict, the channels are always last
                    h, w = get_image_height_width(
                        observation_space.spaces[key], channels_last=True
                    )
                    if size == min(h, w):
                        continue
                    scale = size / min(h, w)
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    new_size = (new_h, new_w)
                    logging.info(
                        "Resizing observation of %s: from %s to %s"
                        % (key, (h, w), new_size)
                    )
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], new_size
                    )
        return observation_space

    def _transform_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return image_resize_shortest_edge(
            obs, self._size, channels_last=self.channels_last
        )

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._size is not None:
            observations.update(
                {
                    sensor: self._transform_obs(observations[sensor])
                    for sensor in self.trans_keys
                    if sensor in observations
                }
            )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.RL.POLICY.OBS_TRANSFORMS.RESIZE_SHORTEST_EDGE.SIZE)


class CenterCropper(ObservationTransformer):
    """An observation transformer is a simple nn module that center crops your input."""

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        channels_last: bool = True,
        trans_keys: Tuple[str] = ("rgb", "depth", "semantic"),
    ):
        """Args:
        size: A sequence (h, w) or int of the size you wish to resize/center_crop.
                If int, assumes square crop
        channels_list: indicates if channels is the last dimension
        trans_keys: The list of sensors it will try to centercrop.
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (h, w)"
        self._size = size
        self.channels_last = channels_last
        self.trans_keys = trans_keys  # TODO: Add to from_config constructor

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if (
                    key in self.trans_keys
                    and observation_space.spaces[key].shape[-3:-1] != size
                ):
                    h, w = get_image_height_width(
                        observation_space.spaces[key], channels_last=True
                    )
                    logging.info(
                        "Center cropping observation size of %s from %s to %s"
                        % (key, (h, w), size)
                    )

                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        return observation_space

    def _transform_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return center_crop(
            obs,
            self._size,
            channels_last=self.channels_last,
        )

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._size is not None:
            observations.update(
                {
                    sensor: self._transform_obs(observations[sensor])
                    for sensor in self.trans_keys
                    if sensor in observations
                }
            )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cc_config = config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER
        return cls(
            (
                cc_config.HEIGHT,
                cc_config.WIDTH,
            )
        )



def get_active_obs_transforms(config: Config) -> List[ObservationTransformer]:
    active_obs_transforms = []
    if hasattr(config.RL.POLICY, "OBS_TRANSFORMS"):

        obs_transform_names = (
            config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS
        )
        for obs_transform_name in obs_transform_names:
            cls = globals()[obs_transform_name]
            obs_transform = cls.from_config(config)
            active_obs_transforms.append(obs_transform)
    return active_obs_transforms



def apply_obs_transforms_obs_space(
    obs_space: spaces.Dict, obs_transforms: Iterable[ObservationTransformer]
) -> spaces.Dict:
    for obs_transform in obs_transforms:
        obs_space = obs_transform.transform_observation_space(obs_space)
    return obs_space


def apply_obs_transforms_batch(
    batch: Dict[str, torch.Tensor],
    obs_transforms: Iterable[ObservationTransformer],
) -> Dict[str, torch.Tensor]:
    for obs_transform in obs_transforms:
        batch = obs_transform(batch)
    return batch