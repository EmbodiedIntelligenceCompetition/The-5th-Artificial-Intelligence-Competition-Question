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

"""Utilities related to TensorSpec class."""

import numpy as np
import torch
import torch.nn.functional as F
from agent.specs import array_spec

import torch

class TensorSpec:
    """Describes the type and shape of a torch.Tensor.
    
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> TensorSpec.from_tensor(t)
    TensorSpec(shape=torch.Size([2, 3]), dtype=torch.int64, name=None)

    Contains metadata for describing the nature of `torch.Tensor` objects
    accepted or returned by some PyTorch operations.
    """
    def __init__(self, shape, dtype, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name

    @classmethod
    def from_tensor(cls, tensor, name=None):
        """Returns a `TensorSpec` that describes `tensor`.

        >>> TensorSpec.from_tensor(torch.tensor([1, 2, 3]))
        TensorSpec(shape=torch.Size([3]), dtype=torch.int64, name=None)

        Args:
          tensor: The `torch.Tensor` that should be described.
          name: A name for the `TensorSpec`.  Defaults to `None`.

        Returns:
          A `TensorSpec` that describes `tensor`.
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"`tensor` should be a torch.Tensor, but got type {type(tensor)}.")
        return cls(tensor.shape, tensor.dtype, name)

    def is_compatible_with(self, spec_or_tensor):
        """Returns True if spec_or_tensor is compatible with this TensorSpec.

        Two tensors are considered compatible if they have the same dtype
        and their shapes are compatible.

        Args:
          spec_or_tensor: A TensorSpec or a torch.Tensor

        Returns:
          True if spec_or_tensor is compatible with self.
        """
        if isinstance(spec_or_tensor, TensorSpec):
            return self.dtype == spec_or_tensor.dtype and self.shape == spec_or_tensor.shape
        elif isinstance(spec_or_tensor, torch.Tensor):
            return self.dtype == spec_or_tensor.dtype and self.shape == spec_or_tensor.shape
        else:
            return False

    @classmethod
    def from_spec(cls, spec, name=None):
        """Returns a `TensorSpec` with the same shape and dtype as `spec`.

        >>> spec = TensorSpec(shape=torch.Size([8, 3]), dtype=torch.int32, name="OriginalName")
        >>> TensorSpec.from_spec(spec, "NewName")
        TensorSpec(shape=torch.Size([8, 3]), dtype=torch.int32, name='NewName')

        Args:
          spec: The `TensorSpec` used to create the new `TensorSpec`.
          name: The name for the new `TensorSpec`.  Defaults to `spec.name`.

        Returns:
          A `TensorSpec` with the same shape and dtype as `spec`.
        """
        return cls(spec.shape, spec.dtype, name or spec.name)

    def __repr__(self):
        return f"TensorSpec(shape={self.shape}, dtype={self.dtype}, name={self.name})"

class BoundedTensorSpec:
    """A `TensorSpec` that specifies minimum and maximum values.

    Example usage:
    ```python
    spec = BoundedTensorSpec((1, 2, 3), torch.float32, 0, (5, 5, 5))
    torch_minimum = torch.tensor(spec.minimum, dtype=spec.dtype)
    torch_maximum = torch.tensor(spec.maximum, dtype=spec.dtype)
    ```

    Bounds are meant to be inclusive. This is especially important for
    integer types. The following spec will be satisfied by tensors
    with values in the set {0, 1, 2}:
    ```python
    spec = BoundedTensorSpec((3, 5), torch.int32, 0, 2)
    ```
    """

    def __init__(self, shape, dtype, minimum, maximum, name=None):
        """Initializes a new `BoundedTensorSpec`.

        Args:
          shape: A tuple or list specifying the shape of the tensor.
          dtype: A `torch.dtype`. The type of the tensor values.
          minimum: Number or sequence specifying the minimum element bounds
            (inclusive). Must be broadcastable to `shape`.
          maximum: Number or sequence specifying the maximum element bounds
            (inclusive). Must be broadcastable to `shape`.
          name: Optional string containing a semantic name for the corresponding
            tensor. Defaults to `None`.

        Raises:
          ValueError: If `minimum` or `maximum` are not provided or not
            broadcastable to `shape`.
        """
        self.shape = shape
        self.dtype = dtype
        self.name = name

        if minimum is None:
            raise ValueError("`minimum` can not be None.")
        if maximum is None:
            raise ValueError("`maximum` can not be None.")

        try:
            minimum_shape = np.shape(minimum)
            np.broadcast(np.empty(self.shape), np.empty(minimum_shape))
        except ValueError as exception:
            raise ValueError(
                f"`minimum` {minimum} is not compatible with shape {self.shape}."
            ) from exception

        try:
            maximum_shape = np.shape(maximum)
            np.broadcast(np.empty(self.shape), np.empty(maximum_shape))
        except ValueError as exception:
            raise ValueError(
                f"`maximum` {maximum} is not compatible with shape {self.shape}."
            ) from exception

        self._minimum = np.array(minimum, dtype=dtype)
        self._maximum = np.array(maximum, dtype=dtype)

    @property
    def minimum(self):
        """Returns a NumPy array specifying the minimum bounds (inclusive)."""
        return self._minimum

    @property
    def maximum(self):
        """Returns a NumPy array specifying the maximum bounds (inclusive)."""
        return self._maximum

    def __repr__(self):
        s = "BoundedTensorSpec(shape={}, dtype={}, name={}, minimum={}, maximum={})"
        return s.format(self.shape, repr(self.dtype), repr(self.name),
                        repr(self.minimum), repr(self.maximum))

    def __eq__(self, other):
        if not isinstance(other, BoundedTensorSpec):
            return False
        return (self.shape == other.shape and self.dtype == other.dtype and
                np.allclose(self.minimum, other.minimum) and
                np.allclose(self.maximum, other.maximum))

    def __hash__(self):
        return hash((self.shape, self.dtype, self.minimum.tobytes(), self.maximum.tobytes()))

    def __reduce__(self):
        return BoundedTensorSpec, (self.shape, self.dtype, self.minimum, self.maximum, self.name)

    @classmethod
    def from_spec(cls, spec):
        """Returns a `BoundedTensorSpec` with the same shape and dtype as `spec`.

        If `spec` is a `BoundedTensorSpec`, then the new spec's bounds are set to
        `spec.minimum` and `spec.maximum`; otherwise, the bounds are set to
        the default min and max values for the given dtype.

        Args:
          spec: The specification used to create the new `BoundedTensorSpec`.

        Returns:
          A `BoundedTensorSpec` instance.
        """
        dtype = spec.dtype
        if np.issubdtype(dtype, np.floating):
            minimum = getattr(spec, "minimum", np.finfo(dtype).min)
            maximum = getattr(spec, "maximum", np.finfo(dtype).max)
        else:
            minimum = getattr(spec, "minimum", np.iinfo(dtype).min)
            maximum = getattr(spec, "maximum", np.iinfo(dtype).max)
        return cls(spec.shape, dtype, minimum, maximum, spec.name)


def is_bounded(spec):
    return isinstance(spec, (array_spec.BoundedArraySpec, BoundedTensorSpec))


def is_discrete(spec):
    return isinstance(spec, TensorSpec) and spec.dtype.is_integer


def is_continuous(spec):
    return isinstance(spec, TensorSpec) and spec.dtype.is_floating


def from_spec(spec):
    """Maps the given spec into corresponding TensorSpecs keeping bounds."""

    def _convert_to_tensor_spec(s):
        # Need to check bounded first as non bounded specs are base class.
        if isinstance(s, (array_spec.BoundedArraySpec, BoundedTensorSpec)):
            return BoundedTensorSpec.from_spec(s)
        elif isinstance(s, (array_spec.ArraySpec, TensorSpec)):
            return TensorSpec.from_spec(s)
        else:
            raise ValueError("No known conversion from type `{}` to a TensorSpec".format(type(s)))

    return _map_structure(_convert_to_tensor_spec, spec)


def to_array_spec(tensor_spec):
    """Converts TensorSpec into ArraySpec."""
    if hasattr(tensor_spec, "minimum") and hasattr(tensor_spec, "maximum"):
        return array_spec.BoundedArraySpec(
            list(tensor_spec.shape),
            tensor_spec.dtype,
            minimum=tensor_spec.minimum,
            maximum=tensor_spec.maximum,
            name=tensor_spec.name
        )
    else:
        return array_spec.ArraySpec(
            list(tensor_spec.shape),
            tensor_spec.dtype,
            tensor_spec.name
        )


def to_nest_array_spec(nest_array_spec):
    """Converted a nest of TensorSpecs to a nest of matching ArraySpecs."""
    return _map_structure(to_array_spec, nest_array_spec)


def _random_uniform_int(shape, outer_dims, minval, maxval, dtype, seed=None):
    """Iterates over n-d tensor minval, maxval limits to sample uniformly."""
    minval = torch.tensor(minval, dtype=dtype)
    maxval = torch.tensor(maxval, dtype=dtype)

    sampling_maxval = maxval
    if dtype.is_integer:
        sampling_maxval = torch.where(maxval < dtype.max, maxval + 1, maxval)

    samples = []
    for (single_min, single_max) in zip(minval.flatten(), sampling_maxval.flatten()):
        samples.append(torch.randint(single_min, single_max, tuple(outer_dims + list(shape)), dtype=dtype))
    return torch.stack(samples, dim=-1).reshape(tuple(outer_dims) + tuple(shape))


def sample_bounded_spec(spec, seed=None, outer_dims=None):
    """Samples uniformly the given bounded spec."""
    minval = spec.minimum
    maxval = spec.maximum
    dtype = torch.int32 if spec.dtype == torch.uint8 else spec.dtype

    if outer_dims is None:
        outer_dims = []
    
    def _unique_vals(vals):
        return len(set(vals.flatten())) == 1

    if (len(minval.shape) != 0 or len(maxval.shape) != 0) and not (_unique_vals(minval) and _unique_vals(maxval)):
        res = _random_uniform_int(
            shape=spec.shape,
            outer_dims=outer_dims,
            minval=minval,
            maxval=maxval,
            dtype=dtype,
            seed=seed
        )
    else:
        minval = minval.item(0) if len(minval.shape) != 0 else minval
        maxval = maxval.item(0) if len(maxval.shape) != 0 else maxval
        if dtype.is_integer and maxval < dtype.max:
            maxval = maxval + 1

        full_shape = outer_dims + list(spec.shape)
        res = torch.randint(minval, maxval, tuple(full_shape), dtype=dtype, generator=torch.Generator().manual_seed(seed))

    if spec.dtype == torch.uint8:
        res = res.to(dtype=torch.uint8)

    return res


def sample_spec_nest(structure, seed=None, outer_dims=()):
    """Samples the given nest of specs."""
    generator = torch.Generator().manual_seed(seed)

    def sample_fn(spec):
        """Return a composite tensor sample given `spec`."""
        if isinstance(spec, (TensorSpec, BoundedTensorSpec)):
            if spec.dtype == torch.string:
                sample_spec = BoundedTensorSpec(spec.shape, torch.int32, minimum=0, maximum=10)
                return torch.randint(0, 10, tuple(outer_dims + list(spec.shape)), generator=generator).to(dtype=torch.string)
            else:
                return sample_bounded_spec(
                    BoundedTensorSpec.from_spec(spec),
                    outer_dims=outer_dims,
                    seed=generator.seed()
                )
        else:
            raise TypeError("Spec type not supported: '{}'".format(spec))

    return _map_structure(sample_fn, structure)


def zero_spec_nest(specs, outer_dims=None):
    """Create zero tensors for a given spec."""
    def make_zero(spec):
        if not isinstance(spec, TensorSpec):
            raise NotImplementedError("Spec type not supported: '{}'".format(spec))
        shape = outer_dims + list(spec.shape) if outer_dims else spec.shape
        return torch.zeros(tuple(shape), dtype=spec.dtype)

    return _map_structure(make_zero, specs)


def add_outer_dims_nest(specs, outer_dims):
    """Adds outer dimensions to the shape of input specs."""
    if not isinstance(outer_dims, (tuple, list)):
        raise ValueError("outer_dims must be a tuple or list of dimensions")

    def add_outer_dims(spec):
        name = spec.name
        shape = outer_dims + list(spec.shape)
        if hasattr(spec, "minimum") and hasattr(spec, "maximum"):
            return BoundedTensorSpec(shape, spec.dtype, spec.minimum, spec.maximum, name)
        return TensorSpec(shape, spec.dtype, name=name)

    return _map_structure(add_outer_dims, specs)


def _map_structure(func, structure):
    if isinstance(structure, list):
        return [ _map_structure(func, s) for s in structure ]
    elif isinstance(structure, tuple) and not hasattr(structure, '_fields'):  # Regular tuple
        return tuple(_map_structure(func, s) for s in structure)
    elif isinstance(structure, dict):
        return { k: _map_structure(func, v) for k, v in structure.items() }
    elif hasattr(structure, '_fields'):  # namedtuple
        return type(structure)(*(_map_structure(func, getattr(structure, field)) for field in structure._fields))
    else:
        return func(structure)