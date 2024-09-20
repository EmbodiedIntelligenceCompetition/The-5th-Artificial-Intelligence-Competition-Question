
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gym.spaces import Dict as gymDict
from typing import Dict
from gym.spaces import Box
from collections import OrderedDict
import functools
import numpy as np
import os
import gym
import glob
from absl import logging
import distutils.version
from agent.common.common import TensorboardWriter
import sys
import functools
import os
import shlex
import signal
import subprocess
import threading
from os import path as osp
from typing import Any, Callable, Optional, Tuple, Union, overload, List
from agent.utils.visualization import images_to_video
import torch
from torch import distributed as distrib

EXIT = threading.Event()
REQUEUE = threading.Event()

def assert_members_are_not_overridden(base_cls,
                                      instance,
                                      white_list=(),
                                      black_list=()):
  """Asserts public members of `base_cls` are not overridden in `instance`.

  If both `white_list` and `black_list` are empty, no public member of
  `base_cls` can be overridden. If a `white_list` is provided, only public
  members in `white_list` can be overridden. If a `black_list` is provided,
  all public members except those in `black_list` can be overridden. Both
  `white_list` and `black_list` cannot be provided at the same, if so a
  ValueError will be raised.

  Args:
    base_cls: A Base class.
    instance: An instance of a subclass of `base_cls`.
    white_list: Optional list of `base_cls` members that can be overridden.
    black_list: Optional list of `base_cls` members that cannot be overridden.

  Raises:
    ValueError if both white_list and black_list are provided.
  """

  if black_list and white_list:
    raise ValueError('Both `black_list` and `white_list` cannot be provided.')

  instance_type = type(instance)
  subclass_members = set(instance_type.__dict__.keys())
  public_members = set(
      [m for m in base_cls.__dict__.keys() if not m.startswith('_')])
  common_members = public_members & subclass_members

  if white_list:
    common_members = common_members - set(white_list)
  elif black_list:
    common_members = common_members & set(black_list)

  overridden_members = [
      m for m in common_members
      if base_cls.__dict__[m] != instance_type.__dict__[m]
  ]
  if overridden_members:
    raise ValueError(
        'Subclasses of {} cannot override most of its base members, but '
        '{} overrides: {}'.format(base_cls, instance_type, overridden_members))

def to_spaces_Dict(data):
  ret = gymDict()
  if isinstance(data, gym.spaces.dict.Dict):
    data = dict(data)
  if type(data) == dict:
    for key,value in data.items():
      shape = list(value.shape)
      # if len(shape) == 1: shape.append(1)
      ret[key] = Box(low=float(-sys.float_info.max), high=float(sys.float_info.max), shape=shape, dtype=np.float32)
  else:
    shape = list(data.shape)
    # if len(shape) == 1: shape.append(1)
    ret['prev_action'] = Box(low=float(-sys.float_info.max), high=float(sys.float_info.max), shape=shape, dtype=np.float32)
  print(ret)
  return ret

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
INTERRUPTED_STATE_FILE = osp.join(
    os.environ["HOME"], ".interrupted_states", f"{SLURM_JOBID}.pth"
)


def load_interrupted_state(filename: str = None) -> Optional[Any]:
    r"""Loads the saved interrupted state

    :param filename: The filename of the saved state.
        Defaults to "${HOME}/.interrupted_states/${SLURM_JOBID}.pth"

    :return: The saved state if the file exists, else none
    """
    if SLURM_JOBID is None and filename is None:
        return None

    if filename is None:
        filename = INTERRUPTED_STATE_FILE

    if not osp.exists(filename):
        return None

    return torch.load(filename, map_location="cpu")


def rank0_only(fn: Optional[Callable] = None) -> Union[Callable, bool]:
    r"""Helper function to only execute code if a process is world rank 0

    Can be used both as a function in an if statement,

    .. code:: py

        if rank0_only():
            ...

    or as a decorator,

    .. code:: py

        @rank0_only
        def fn_for_r0_only(...):
            ...

    :param fn: Function to wrap and only execute if the process is rank 0.
        If a process is rank 0, the function will be run and it's return value
        will be returned.  If a process is not rank 0, then the function will not
        be ran and :py:`None` will be returned.

    :return: The wrapped function if :p:`fn` is not :py:`None`, otherwise
        whether or not this process is rank 0
    """
    if fn is None:
        return (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if rank0_only():
            return fn(*args, **kwargs)
        return None

    return _wrapper




@rank0_only
def save_interrupted_state(state: Any, filename: str = None):
    r"""Saves the interrupted job state to the specified filename.
        This is useful when working with preemptable job partitions.

    This method will do nothing if SLURM is not currently being used and the filename is the default

    :param state: The state to save
    :param filename: The filename.  Defaults to "${HOME}/.interrupted_states/${SLURM_JOBID}.pth"
    """
    if SLURM_JOBID is None and filename is None:
        return

    if filename is None:
        filename = INTERRUPTED_STATE_FILE
        if not osp.exists(osp.dirname(INTERRUPTED_STATE_FILE)):
            raise RuntimeError(
                "Please create a .interrupted_states directory in your home directory for job preemption"
                "(This is intentionally not created automatically as it can get quite large)"
            )

    torch.save(state, filename)


def requeue_job():
    r"""Requeues the job by calling ``scontrol requeue ${SLURM_JOBID}``"""
    if SLURM_JOBID is None:
        return

    if not REQUEUE.is_set():
        return

    if distrib.is_initialized():
        distrib.barrier()

    if rank0_only():
        subprocess.check_call(shlex.split(f"scontrol requeue {SLURM_JOBID}"))



def get_checkpoint_id(ckpt_path: str) -> Optional[int]:
    r"""Attempts to extract the ckpt_id from the filename of a checkpoint.
    Assumes structure of ckpt.ID.path .

    Args:
        ckpt_path: the path to the ckpt file

    Returns:
        returns an int if it is able to extract the ckpt_path else None
    """
    ckpt_path = os.path.basename(ckpt_path)
    nums: List[int] = [int(s) for s in ckpt_path.split(".") if s.isdigit()]
    if len(nums) > 0:
        return nums[-1]
    return None

def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:
    r"""Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None

def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    scene_id: Union[int, str],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"scene={scene_id}-episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )