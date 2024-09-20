
from typing import Dict, Tuple
from agent.policy.policy import Policy,Net
from gym import spaces
from agent.gibson_extension.examples.configs.default import Config
import torch.nn as nn
from agent.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
import torch
from torch.nn import functional as F
from agent.policy import resnet
import numpy as np
from agent.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)


class PointNavResNetPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        num_envs: int = 1,
        **kwargs
    ):
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                force_blind_policy=force_blind_policy,
            ),
            2,
            num_envs = num_envs
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            num_envs = config.NUM_ENVIRONMENTS
        )
    



class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        # if "rgb" in observation_space.spaces:
        #     self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        #     spatial_size = observation_space.spaces["rgb"].shape[0] // 2
        # else:
        #     self._n_input_rgb = 0

        self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._n_input_depth = 0

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial = int(
                spatial_size * self.backbone.final_spatial_compress
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial ** 2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = (
                rgb_observations.float() / 255.0
            )  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        force_blind_policy: bool = False,
    ):
        super().__init__()

        self.prev_action_embedding = nn.Linear(2, 32)
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action

        if (
            'task_obs'
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    'task_obs'
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 64)
            rnn_input_size += 64


        self._hidden_size = hidden_size

        self.visual_encoder = ResNetEncoder(
            observation_space if not force_blind_policy else spaces.Dict({}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

    @property
    def output_size(self):
        # return self._hidden_size + 64
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        if 'task_obs' in observations:
            goal_state_observations = observations[
                'task_obs'
            ]
            if goal_state_observations.shape[1] == 4:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_state_observations = torch.stack(
                    [
                        goal_state_observations[:, 0],
                        torch.cos(-goal_state_observations[:, 1]),
                        torch.sin(-goal_state_observations[:, 1]),
                        goal_state_observations[:, 2],
                        goal_state_observations[:, 3],
                    ],
                    -1,
                )
            else:
                assert (
                    goal_state_observations.shape[1] == 5
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_state_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_state_observations = torch.stack(
                    [
                        goal_state_observations[:, 0],
                        torch.cos(-goal_state_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_state_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_state_observations[:, 2]),
                        goal_state_observations[:, 3],
                        goal_state_observations[:, 4],
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_state_observations))

        prev_actions = prev_actions.squeeze(-1)
        prev_actions = self.prev_action_embedding(
            prev_actions
        )


        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks
        )

        return out, rnn_hidden_states
