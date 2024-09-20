from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from agent.ppo.config.default import get_config
import pickle
import os
from typing import Dict,List,Any,Optional
import time
from collections import defaultdict, deque
from torch.optim.lr_scheduler import LambdaLR
import torch
import json
from agent.utils.common import to_spaces_Dict
from absl import app
from absl import flags
from absl import logging
import contextlib
import copy
import imageio
import torch.nn as nn
import gin
import tqdm
import numpy as np
from agent.ppo.ppo import PPO
from agent.rollout.rollout_storage import RolloutStorage
from agent.common.common import batch_obs, ObservationBatchingCache
from agent.environments import suite_gibson
from agent.environments import tf_py_environment
from agent.environments import parallel_py_environment
from agent.utils import common
from agent.policy.PointNavPolicy import PointNavResNetPolicy
from agent.common.obs_transformers import (
    get_active_obs_transforms,
    apply_obs_transforms_obs_space,
    apply_obs_transforms_batch
    
)
import gym.spaces as spaces
from agent.utils.common import (load_interrupted_state,
                                    rank0_only,
                                    EXIT,REQUEUE,
                                    save_interrupted_state,
                                    requeue_job,
                                    generate_video,
                                    )
from agent.utils.visualization import observations_to_image
from agent.common.common import TensorboardWriter

from agent.trainer.base_trainer import BaseTrainer, BaseRLTrainer
from absl import flags
import os


class PPOTrainer(BaseRLTrainer):

    def __init__(self, FLAGS) -> None:
        
        flags.mark_flag_as_required('root_dir')
        flags.mark_flag_as_required('config_file')
        flags.mark_flag_as_required('agent_config_file')
        self.agent_config = get_config(FLAGS.agent_config_file, None)

        super().__init__(config=self.agent_config, FLAGS=FLAGS)

    def init_envs(self, env_load_fn=None) -> None:
        self.num_parallel_environments = self.FLAGS.num_parallel_environments
        if self.model_ids is None:
            self.model_ids = [None] * self.num_parallel_environments
        else:
            assert len(self.model_ids) == self.num_parallel_environments, \
                'model ids provided, but length not equal to num_parallel_environments'

        self.tf_py_env = [lambda model_id=self.model_ids[i]: env_load_fn(model_id, 'headless', self.gpu)
                        for i in range(self.num_parallel_environments)]
        
        self.tf_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(self.tf_py_env))

        self.time_step_spec = self.tf_env.time_step_spec()

        self.observation_spec = self.time_step_spec.observation
        self.action_spec = self.tf_env.action_spec()

        

        self.observation_spec = to_spaces_Dict(self.observation_spec)
        self.action_spec = to_spaces_Dict(self.action_spec)

        self.obs_transforms = get_active_obs_transforms(self.agent_config)
        self.observation_spec = apply_obs_transforms_obs_space(
            self.observation_spec, self.obs_transforms
        )

    def set_agent(self) -> None:
        self.device = torch.device('cuda:'+ str(0))
        self.policy = PointNavResNetPolicy.from_config(config=self.agent_config, observation_space= self.observation_spec, action_space= self.action_spec)
        self.policy.to(device=self.device)
        if self.agent_config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.policy.critic.fc.weight)
            nn.init.constant_(self.policy.critic.fc.bias, 0)
        self.ppo_cfg = self.agent_config.RL.PPO
        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.policy.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.policy.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.policy.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.policy.critic.fc.weight)
            nn.init.constant_(self.policy.critic.fc.bias, 0)
        self.agent = PPO(
            actor_critic=self.policy,
            clip_param=self.ppo_cfg.clip_param,
            ppo_epoch=self.ppo_cfg.ppo_epoch,
            num_mini_batch=self.ppo_cfg.num_mini_batch,
            value_loss_coef=self.ppo_cfg.value_loss_coef,
            entropy_coef=self.ppo_cfg.entropy_coef,
            lr=self.ppo_cfg.lr,
            eps=self.ppo_cfg.eps,
            max_grad_norm=self.ppo_cfg.max_grad_norm,
            use_normalized_advantage=self.ppo_cfg.use_normalized_advantage,
        )

    def init_ppo_training(self) -> None:
        self.device = torch.device('cuda:'+ str(0))
        self.policy = PointNavResNetPolicy.from_config(config=self.agent_config, observation_space= self.observation_spec, action_space= self.action_spec)
        self.policy.to(device=self.device)
        if self.agent_config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.policy.critic.fc.weight)
            nn.init.constant_(self.policy.critic.fc.bias, 0)
        self.ppo_cfg = self.agent_config.RL.PPO
        self.agent = PPO(
            actor_critic=self.policy,
            clip_param=self.ppo_cfg.clip_param,
            ppo_epoch=self.ppo_cfg.ppo_epoch,
            num_mini_batch=self.ppo_cfg.num_mini_batch,
            value_loss_coef=self.ppo_cfg.value_loss_coef,
            entropy_coef=self.ppo_cfg.entropy_coef,
            lr=self.ppo_cfg.lr,
            eps=self.ppo_cfg.eps,
            max_grad_norm=self.ppo_cfg.max_grad_norm,
            use_normalized_advantage=self.ppo_cfg.use_normalized_advantage,
        )

        logging.info(
        "agent number of parameters: {}".format(
            sum(param.numel() for param in self.agent.parameters())
            )
        )
        self._nbuffers = 2 if self.ppo_cfg.use_double_buffered_sampler else 1
        self.observation_spec.spaces.pop('global_occupancy_grid')
        self.rollouts = RolloutStorage(
            self.ppo_cfg.num_steps,
            self.num_parallel_environments,
            self.observation_spec,
            self.action_spec,
            self.ppo_cfg.hidden_size,
            num_recurrent_layers=self.policy.net.num_recurrent_layers,
            is_double_buffered=self.ppo_cfg.use_double_buffered_sampler,
        )
        self.rollouts.to(self.device)

        observations = self.tf_env.reset().observation
        observations.pop('global_occupancy_grid')
        # 获取批次大小
        batch_size = next(iter(observations.values())).shape[0]

        # 创建列表
        formatted_data = []

        for i in range(batch_size):
            item = {}
            for key in observations:
                item[key] = observations[key][i]
            formatted_data.append(item)
        observations = formatted_data
        

        self._obs_batching_cache = ObservationBatchingCache()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        self.rollouts.buffers["observations"][0] = batch

        self.current_episode_reward = torch.zeros(self.num_parallel_environments, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.num_parallel_environments, 1),
            reward=torch.zeros(self.num_parallel_environments, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self.ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()
        self.count_checkpoints = 0
        self.prev_time = 0
        self.num_steps_done = 0
        self.num_updates_done = 0
        self._last_checkpoint_percent = -1.0

        self.lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            self.lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            self.count_checkpoints = requeue_stats["count_checkpoints"]
            self.prev_time = requeue_stats["prev_time"]

            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]

        self.ppo_cfg = self.agent_config.RL.PPO

    def train(self, env_load_fn=None, model_ids=None,) -> None:
        self.root_dir = os.path.expanduser(self.FLAGS.root_dir)
        self.gpu = self.FLAGS.gpu_c
        self.model_ids = model_ids
        self.init_envs(env_load_fn)

        self.init_ppo_training()
        
        with (
            TensorboardWriter(
                self.agent_config.TENSORBOARD_DIR, flush_secs=30
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                if self.ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = self.ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if EXIT.is_set():

                    self.tf_env.close()

                    if REQUEUE.is_set() and rank0_only():
                        requeue_stats = dict(
                            env_time=self.env_time,
                            pth_time=self.pth_time,
                            count_checkpoints=self.count_checkpoints,
                            num_steps_done=self.num_steps_done,
                            num_updates_done=self.num_updates_done,
                            _last_checkpoint_percent=self._last_checkpoint_percent,
                            prev_time=(time.time() - self.t_start) + self.prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=self.lr_scheduler.state_dict(),
                                config=self.agent_config,
                                requeue_stats=requeue_stats,
                            )
                        )

                    requeue_job()
                    return

                """收集训练data"""
                # TODO: 因为这里map是一部分，这里如何去处理global goal，这里用相对距离可能可以
                self.agent.eval()
                count_steps_delta = 0
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(self.ppo_cfg.num_steps):
                    is_last_step = (
                        False
                        or (step + 1) == self.ppo_cfg.num_steps
                    )

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            pass

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                pass

                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break


                (
                    value_loss,
                    action_loss,
                ) = self._update_agent()

                if self.ppo_cfg.use_linear_lr_decay:
                    self.lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(value_loss=value_loss, action_loss=action_loss),
                    count_steps_delta,
                )
                # TODO: 多环境单个GPU训练还未实现
                self._training_log(writer, losses, self.prev_time)

                # checkpoint model
                needs_checkpoint = self.should_checkpoint()
                if rank0_only() and needs_checkpoint:
                    self.save_checkpoint(
                        f"ckpt.{self.count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + self.prev_time,
                        ),
                    )
                    self.count_checkpoints += 1


            self.tf_env.close()


    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
        env_load_fn: Any = None,
        model_ids: Any = None,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        imageio.plugins.ffmpeg.download()
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        self.gpu = self.FLAGS.gpu_c
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        # config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            # config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            # config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logging.info(f"env config: {config}")
        self.model_ids = model_ids
        self.env_to_pause = []
        self.init_envs(env_load_fn)
        self.set_agent()

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.tf_env.reset().observation
        # 获取批次大小
        batch_size = next(iter(observations.values())).shape[0]

        # 创建列表
        formatted_data = []

        for i in range(batch_size):
            item = {}
            for key in observations:
                item[key] = observations[key][i].astype(np.float32)
            formatted_data.append(item)
        observations = formatted_data
        self._obs_batching_cache = ObservationBatchingCache()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(
            self.config.NUM_ENVIRONMENTS, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.net.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            2,
            device=self.device,
            dtype=torch.float32,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT

        if number_of_eval_episodes == -1:
            # 打开self.model_ids里面对应的json文件，总和里面的episode数量
            for model_id in self.model_ids:
                with open(model_id + ".json") as f:
                    data = json.load(f)
                    number_of_eval_episodes += len(data)
            number_of_eval_episodes += 1

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.num_parallel_environments > 0
        ):

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [a for a in actions.to(device="cpu")]
            outputs = self.tf_env.step(step_data, self.env_to_pause)

            step_type, rewards_l, discount, observations, info = outputs.step_type, outputs.reward, outputs.discount, outputs.observation, outputs.info
            # 获取批次大小
            batch_size = next(iter(observations.values())).shape[0]

            # 创建列表
            formatted_data = []

            for i in range(batch_size):
                item = {}
                for key in observations:
                    item[key] = observations[key][i].astype(np.float32)
                formatted_data.append(item)
            observations = formatted_data

            # 创建列表
            formatted_data = []

            for i in range(batch_size):
                item = {}
                for key in info:
                    item[key] = info[key][i].astype(np.float32)
                formatted_data.append(item)
            info = formatted_data

            dones = []
            for i in range(batch_size):
                dones.append(False) if info[i]['done'] == False else dones.append(True)

            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            )
            current_episode_reward += rewards
            n_envs = self.num_parallel_environments
            envs_to_pause_for_pop = []
            envs_to_pause_for_pop_idx = 0
            for i in range(n_envs):
                
                if i in self.env_to_pause:
                    continue

                # episode ended
                if not not_done_masks[envs_to_pause_for_pop_idx].item():
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[envs_to_pause_for_pop_idx].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(info[envs_to_pause_for_pop_idx])
                    )
                    current_episode_reward[envs_to_pause_for_pop_idx] = 0
                    stats_episodes[self.tf_env._env._envs[i].scene_id + ':' + str(self.tf_env._env._envs[i].current_episode)] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[envs_to_pause_for_pop_idx],
                            episode_id=self.tf_env._env._envs[i].current_episode,
                            scene_id=self.tf_env._env._envs[i].scene_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(info[envs_to_pause_for_pop_idx]),
                            tb_writer=writer,
                        )

                        rgb_frames[envs_to_pause_for_pop_idx] = []

                        prev_actions[envs_to_pause_for_pop_idx] = torch.zeros(
                            2, device=self.device, dtype=torch.float32
                        )

                    if self.tf_env._env._envs[i].current_episode == self.tf_env._env._envs[i].task.total_episodes:
                        envs_to_pause_for_pop.append(envs_to_pause_for_pop_idx)
                        self.env_to_pause.append(i)

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    info[envs_to_pause_for_pop_idx]['occupancy_grid'] = observations[envs_to_pause_for_pop_idx]["global_occupancy_grid"]
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v for k, v in observations[envs_to_pause_for_pop_idx].items() if k != 'task_obs'}, info[envs_to_pause_for_pop_idx]
                    )
                    rgb_frames[envs_to_pause_for_pop_idx].append(frame)

                envs_to_pause_for_pop_idx += 1

            not_done_masks = not_done_masks.to(device=self.device)

            state_index = list(range(batch_size))
            for idx in reversed(envs_to_pause_for_pop):
                state_index.pop(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logging.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.tf_env.close()

    def percent_done(self) -> float:
        if self.agent_config.NUM_UPDATES != -1:
            return self.num_updates_done / self.agent_config.NUM_UPDATES
        else:
            return self.num_steps_done / self.agent_config.TOTAL_NUM_STEPS
        
    def is_done(self) -> bool:
        return self.percent_done() >= 1.0
        
    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.num_parallel_environments
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )


        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            (
                values, 
                actions,
                actions_log_probs,
                recurrent_hidden_states
            ) = self.policy.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")


        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            value_preds=values,
            action_log_probs=actions_log_probs,
            buffer_index=buffer_index,
        )

    def _extract_scalars_from_info(
        self, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in self._extract_scalars_from_info(
                            v
                        ).items()
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    def _extract_scalars_from_infos(
        self, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in self._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _update_agent(self):
        ppo_cfg = self.agent_config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.policy.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        self.agent.train()

        value_loss, action_loss = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()

        return (
            value_loss,
            action_loss,
        )

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.num_parallel_environments
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )


        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = step_batch['actions'].to(device="cpu")


        outputs = self.tf_env.step(actions)
        step_type, rewards_l, discount, observations, info = outputs.step_type, outputs.reward, outputs.discount, outputs.observation, outputs.info
        observations.pop('global_occupancy_grid')
        # 获取批次大小
        batch_size = next(iter(observations.values())).shape[0]

        # 创建列表
        formatted_data = []

        for i in range(batch_size):
            item = {}
            for key in observations:
                item[key] = observations[key][i]
            formatted_data.append(item)
        observations = formatted_data

        # 创建列表
        formatted_data = []

        for i in range(batch_size):
            item = {}
            for key in info:
                item[key] = info[key][i]
            formatted_data.append(item)
        info = formatted_data
        
        dones = []
        for i in range(batch_size):
            dones.append(False) if info[i]['done'] == False else dones.append(True)

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        # rewards = rewards.unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        # 目前累计的reward
        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(info).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)


        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start
    
    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        return t

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])


        self.num_steps_done += count_steps_delta

        return losses
    
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }
        if len(metrics) > 0:
            writer.add_scalars("metrics", metrics, self.num_steps_done)

        writer.add_scalars(
            "losses",
            losses,
            self.num_steps_done,
        )

        # log stats
        if self.num_updates_done % self.agent_config.LOG_INTERVAL == 0:
            logging.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    self.num_steps_done
                    / ((time.time() - self.t_start) + prev_time),
                )
            )

            logging.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logging.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_checkpoint(self) -> bool:
        needs_checkpoint = False
        if self.agent_config.NUM_CHECKPOINTS != -1:
            checkpoint_every = 1 / self.agent_config.NUM_CHECKPOINTS
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        else:
            needs_checkpoint = (
                self.num_steps_done % self.agent_config.CHECKPOINT_INTERVAL
            ) == 0

        return needs_checkpoint
    

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.agent_config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        if not os.path.exists(self.agent_config.CHECKPOINT_FOLDER):
            os.makedirs(self.agent_config.CHECKPOINT_FOLDER)

        torch.save(
            checkpoint, os.path.join(self.agent_config.CHECKPOINT_FOLDER, file_name)
        )


    def save_agent(self, file_name: str, extra_state: Optional[Dict] = None) -> None:
        r"""Save the entire agent object to a file.

        Args:
            file_name: file name for saving the agent object.
        """
        if not os.path.exists(self.agent_config.CHECKPOINT_FOLDER):
            os.makedirs(self.agent_config.CHECKPOINT_FOLDER)

        agent_clone = copy.deepcopy(self.agent)
        agent_clone.actor_critic.to("cpu")
        agent = {
            "agent": agent_clone,
            "config": self.agent_config,
        }

        if extra_state is not None:
            agent["extra_state"] = extra_state
        
        agent_path = os.path.join(self.agent_config.CHECKPOINT_FOLDER, file_name)
        with open(agent_path, 'wb') as f:
            pickle.dump(agent, f)


    def load_agent(self, file_name: str):
        r"""Load the entire agent object from a file.

        Args:
            file_name: file name for loading the agent object.
        """
        agent_path = os.path.join(self.agent_config.CHECKPOINT_FOLDER, file_name)
        with open(agent_path, 'rb') as f:
            agent = pickle.load(f)

        return agent

    def generate_data(
        self,
        num_episodes: int,
        env_load_fn: Any = None,
        model_ids: Any = None,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        self.gpu = self.FLAGS.gpu_c
        data = {model_id: {} for model_id in model_ids}
        num_envs = len(model_ids)
        imageio.plugins.ffmpeg.download()
        self.model_ids = model_ids
        self.init_envs(env_load_fn)
        ret = [{"config":{}, "episode":[]} for _ in range(num_envs)]
        for idx in range(num_envs):
            ret[idx]["config"]["num_episodes"] = num_episodes
            ret[idx]["config"]["numpy_seed"] = 0
            ret[idx]["config"]["scene_id"] = self.tf_env._env._envs[idx].scene_id

        for idx in range(num_episodes):
            self.tf_env.reset()
            for env_idx in range(num_envs):
                tmp_episode = {}
                data[self.tf_env._env._envs[env_idx].scene_id][self.tf_env._env._envs[env_idx].current_episode] = [self.tf_env._env._envs[env_idx].task.target_pos.tolist(),
                                                                                                                   self.tf_env._env._envs[env_idx].task.initial_pos.tolist(),
                                                                                                                   self.tf_env._env._envs[env_idx].task.initial_orn.tolist()]
                tmp_episode["initial_orn"] = self.tf_env._env._envs[env_idx].task.initial_orn.tolist()
                tmp_episode["initial_pos"] = self.tf_env._env._envs[env_idx].task.initial_pos.tolist()
                tmp_episode["target_pos"] = self.tf_env._env._envs[env_idx].task.target_pos.tolist()
                ret[env_idx]["episode"].append(tmp_episode)
        # 保存为json文件
        # for k,v in data.items():
        #     with open(f'{k}.json', 'w') as f:
        #         json.dump(v, f)

        for idx in range(num_envs):
            with open(f'{model_ids[idx]}.json', 'w') as f:
                json.dump(ret[idx], f)







