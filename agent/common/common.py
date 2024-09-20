

import copy
import numbers
from typing import Callable, Dict, Optional, Tuple, Union, overload,Dict,List,DefaultDict,Any
import attr
import numpy as np
import torch
from collections import defaultdict
from typing import Any

from torch.utils.tensorboard import SummaryWriter
TensorLike = Union[torch.Tensor, np.ndarray, numbers.Real]
DictTree = Dict[str, Union[TensorLike, "DictTree"]]
TensorIndexType = Union[int, slice, Tuple[Union[int, slice], ...]]


class TensorDict(Dict[str, Union["TensorDict", torch.Tensor]]):
    r"""A dictionary of tensors that can be indexed like a tensor or like a dictionary.

    .. code:: py
        t = TensorDict(a=torch.randn(2, 2), b=TensorDict(c=torch.randn(3, 3)))

        print(t)

        print(t[0, 0])

        print(t["a"])

    """

    @classmethod
    def from_tree(cls, tree: DictTree) -> "TensorDict":
        res = cls()
        for k, v in tree.items():
            if isinstance(v, dict):
                res[k] = cls.from_tree(v)
            else:
                res[k] = torch.as_tensor(v)

        return res

    def to_tree(self) -> DictTree:
        res: DictTree = dict()
        for k, v in self.items():
            if isinstance(v, TensorDict):
                res[k] = v.to_tree()
            else:
                res[k] = v

        return res

    @overload
    def __getitem__(self, index: str) -> Union["TensorDict", torch.Tensor]:
        ...

    @overload
    def __getitem__(self, index: TensorIndexType) -> "TensorDict":
        ...

    def __getitem__(
        self, index: Union[str, TensorIndexType]
    ) -> Union["TensorDict", torch.Tensor]:
        if isinstance(index, str):
            return super().__getitem__(index)
        else:
            return TensorDict({k: v[index] for k, v in self.items()})

    @overload
    def set(
        self,
        index: str,
        value: Union[TensorLike, "TensorDict", DictTree],
        strict: bool = True,
    ) -> None:
        ...

    @overload
    def set(
        self,
        index: TensorIndexType,
        value: Union["TensorDict", DictTree],
        strict: bool = True,
    ) -> None:
        ...

    def set(
        self,
        index: Union[str, TensorIndexType],
        value: Union[TensorLike, "TensorDict"],
        strict: bool = True,
    ) -> None:
        if isinstance(index, str):
            super().__setitem__(index, value)
        else:
            if strict and (self.keys() != value.keys()):
                raise KeyError(
                    "Keys don't match: Dest={} Source={}".format(
                        self.keys(), value.keys()
                    )
                )

            for k in self.keys():
                if k not in value:
                    if strict:
                        raise KeyError(f"Key {k} not in new value dictionary")
                    else:
                        continue

                v = value[k]

                if isinstance(v, (TensorDict, dict)):
                    self[k].set(index, v, strict=strict)
                else:
                    self[k][index].copy_(torch.as_tensor(v))

    def __setitem__(
        self,
        index: Union[str, TensorIndexType],
        value: Union[torch.Tensor, "TensorDict"],
    ):
        self.set(index, value)

    @classmethod
    def map_func(
        cls,
        func: Callable[[torch.Tensor], torch.Tensor],
        src: "TensorDict",
        dst: Optional["TensorDict"] = None,
    ) -> "TensorDict":
        if dst is None:
            dst = TensorDict()

        for k, v in src.items():
            if torch.is_tensor(v):
                dst[k] = func(v)
            else:
                dst[k] = cls.map_func(func, v, dst.get(k, None))

        return dst

    def map(
        self, func: Callable[[torch.Tensor], torch.Tensor]
    ) -> "TensorDict":
        return self.map_func(func, self)

    def map_in_place(
        self, func: Callable[[torch.Tensor], torch.Tensor]
    ) -> "TensorDict":
        return self.map_func(func, self, self)

    def __deepcopy__(self, _memo=None) -> "TensorDict":
        return TensorDict.from_tree(copy.deepcopy(self.to_tree(), memo=_memo))


@attr.s(auto_attribs=True, slots=True)
class ObservationBatchingCache:
    r"""Helper for batching observations that maintains a cpu-side tensor
    that is the right size and is pinned to cuda memory
    """
    _pool: Dict[Any, torch.Tensor] = attr.Factory(dict)

    def get(
        self,
        num_obs: int,
        sensor_name: str,
        sensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""Returns a tensor of the right size to batch num_obs observations together

        If sensor is a cpu-side tensor and device is a cuda device the batched tensor will
        be pinned to cuda memory.  If sensor is a cuda tensor, the batched tensor will also be
        a cuda tensor
        """
        key = (
            num_obs,
            sensor_name,
            tuple(sensor.size()),
            sensor.type(),
            sensor.device.type,
            sensor.device.index,
        )
        if key in self._pool:
            return self._pool[key]

        cache = torch.empty(
            num_obs, *sensor.size(), dtype=sensor.dtype, device=sensor.device
        )
        if (
            device is not None
            and device.type == "cuda"
            and cache.device.type == "cpu"
        ):
            cache = cache.pin_memory()

        self._pool[key] = cache
        return cache


@torch.no_grad()
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
    cache: Optional[ObservationBatchingCache] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None
        cache: An ObservationBatchingCache.  This enables faster
            stacking of observations and cpu-gpu transfer as it
            maintains a correctly sized tensor for the batched
            observations that is pinned to cuda memory.

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch_t: TensorDict = TensorDict()
    if cache is None:
        batch: DefaultDict[str, List] = defaultdict(list)

    for i, obs in enumerate(observations):
        for sensor_name, sensor in obs.items():
            sensor = torch.as_tensor(sensor)
            if cache is None:
                batch[sensor_name].append(sensor)
            else:
                if sensor_name not in batch_t:
                    batch_t[sensor_name] = cache.get(
                        len(observations), sensor_name, sensor, device
                    )

                batch_t[sensor_name][i].copy_(sensor)

    if cache is None:
        for sensor in batch:
            batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device, non_blocking=True))





@attr.s(auto_attribs=True, slots=True)
class ObservationBatchingCache:
    r"""Helper for batching observations that maintains a cpu-side tensor
    that is the right size and is pinned to cuda memory
    """
    _pool: Dict[Any, torch.Tensor] = attr.Factory(dict)

    def get(
        self,
        num_obs: int,
        sensor_name: str,
        sensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""Returns a tensor of the right size to batch num_obs observations together

        If sensor is a cpu-side tensor and device is a cuda device the batched tensor will
        be pinned to cuda memory.  If sensor is a cuda tensor, the batched tensor will also be
        a cuda tensor
        """
        key = (
            num_obs,
            sensor_name,
            tuple(sensor.size()),
            sensor.type(),
            sensor.device.type,
            sensor.device.index,
        )
        if key in self._pool:
            return self._pool[key]

        cache = torch.empty(
            num_obs, *sensor.size(), dtype=sensor.dtype, device=sensor.device
        )
        if (
            device is not None
            and device.type == "cuda"
            and cache.device.type == "cpu"
        ):
            cache = cache.pin_memory()

        self._pool[key] = cache
        return cache




class TensorboardWriter:
    def __init__(self, log_dir: str, *args: Any, **kwargs: Any):
        r"""A Wrapper for tensorboard SummaryWriter. It creates a dummy writer
        when log_dir is empty string or None. It also has functionality that
        generates tb video directly from numpy images.

        Args:
            log_dir: Save directory location. Will not write to disk if
            log_dir is an empty string.
            *args: Additional positional args for SummaryWriter
            **kwargs: Additional keyword args for SummaryWriter
        """
        self.writer = None
        if log_dir is not None and len(log_dir) > 0:
            self.writer = SummaryWriter(log_dir, *args, **kwargs)

    def __getattr__(self, item):
        if self.writer:
            return self.writer.__getattribute__(item)
        else:
            return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()

    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
    ) -> None:
        r"""Write video into tensorboard from images frames.

        Args:
            video_name: name of video string.
            step_idx: int of checkpoint index to be displayed.
            images: list of n frames. Each frame is a np.ndarray of shape.
            fps: frame per second for output video.

        Returns:
            None.
        """
        if not self.writer:
            return
        # initial shape of np.ndarray list: N * (H, W, 3)
        frame_tensors = [
            torch.from_numpy(np_arr).unsqueeze(0) for np_arr in images
        ]
        video_tensor = torch.cat(tuple(frame_tensors))
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        # final shape of video tensor: (1, n, 3, H, W)
        self.writer.add_video(
            video_name, video_tensor, fps=fps, global_step=step_idx
        )
