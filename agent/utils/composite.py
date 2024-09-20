
"""Utilities for dealing with CompositeTensors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

def shape(tensor):
    if isinstance(tensor, torch.sparse.FloatTensor):
        return tensor.shape
    else:
        return torch.tensor(tensor.size(), dtype=torch.int64)


def reshape(t, shape):  # pylint: disable=redefined-outer-name
    """Reshape composite tensor `t` to `shape`.

    Args:
        t: A `Tensor` or `SparseTensor`.
        shape: `1D` tensor, array, or list.  The new shape.

    Returns:
        The reshaped tensor.
    """
    if isinstance(t, torch.sparse.FloatTensor):
        return t.coalesce().to_dense().reshape(shape).to_sparse()
    else:
        return t.reshape(shape)


def squeeze(t, axis):
    """Squeeze composite tensor along axis `axis`.

    Args:
        t: A `Tensor` or `SparseTensor`.
        axis: A python integer.

    Returns:
        The tensor with dimension `axis` removed.

    Raises:
        InvalidArgumentError: If `t` is a `SparseTensor` and has more than one index
        stored along `axis`.
    """
    if isinstance(t, torch.sparse.FloatTensor):
        if t.shape[axis] != 1:
            raise ValueError(f'Cannot squeeze SparseTensor on axis {axis}, dimension not equal to 1')
        indices = t._indices()
        values = t._values()
        new_indices = torch.cat([indices[:axis], indices[axis + 1:]], dim=0)
        new_shape = list(t.shape)
        del new_shape[axis]
        return torch.sparse.FloatTensor(new_indices, values, torch.Size(new_shape))
    else:
        return t.squeeze(axis)


def expand_dims(t, axis):
    """Add a new dimension to tensor `t` along `axis`.

    Args:
        t: A `torch.Tensor` or `torch.sparse.FloatTensor`.
        axis: A `0D` integer scalar.

    Returns:
        An expanded tensor.

    Raises:
        NotImplementedError: If `t` is a `SparseTensor` and `axis != 0`.
    """
    if isinstance(t, torch.sparse.FloatTensor):
        if axis != 0:
            raise NotImplementedError(
                f'Can only expand_dims on SparseTensor on static axis 0, but received axis {axis}')
        indices = t._indices()
        values = t._values()
        new_indices = torch.cat([torch.zeros((1, indices.size(1)), dtype=torch.int64), indices], dim=0)
        new_shape = [1] + list(t.shape)
        return torch.sparse.FloatTensor(new_indices, values, torch.Size(new_shape))
    else:
        return t.unsqueeze(axis)


def slice_from(tensor, axis, start):
    """Slice a composite tensor along `axis` from `start`.

    Examples:

    ```python
    slice_from(tensor, 2, 1) === tensor[:, :, 1:]
    sparse_to_dense(slice_from(sparse_tensor, 2, 1))
      === sparse_to_dense(sparse_tensor)[:, :, 1:]
    ```

    Args:
        tensor: A `Tensor` or `SparseTensor`.
        axis: A python integer.
        start: A `0D` scalar.

    Returns:
        The sliced composite tensor.
    """
    if isinstance(tensor, torch.sparse.FloatTensor):
        if start < 0:
            start += tensor.size(axis)
        indices = tensor._indices()
        values = tensor._values()
        mask = indices[axis] >= start
        new_indices = indices[:, mask]
        new_indices[axis] -= start
        new_shape = list(tensor.size())
        new_shape[axis] -= start
        return torch.sparse.FloatTensor(new_indices, values[mask], torch.Size(new_shape))
    else:
        slices = [slice(None)] * axis + [slice(start, None)]
        return tensor[tuple(slices)]


def slice_to(tensor, axis, end):
    """Slice a composite tensor along `axis` from 0 to `end`.

    Examples:

    ```python
    slice_to(tensor, 2, -1) === tensor[:, :, :-1]
    sparse_to_dense(slice_to(sparse_tensor, 2, -1))
      === sparse_to_dense(sparse_tensor)[:, :, :-1]
    ```

    Args:
        tensor: A `Tensor` or `SparseTensor`.
        axis: A python integer.
        end: A `0D` scalar.

    Returns:
        The sliced composite tensor.
    """
    if isinstance(tensor, torch.sparse.FloatTensor):
        if end < 0:
            end += tensor.size(axis)
        indices = tensor._indices()
        values = tensor._values()
        mask = indices[axis] < end
        new_indices = indices[:, mask]
        new_shape = list(tensor.size())
        new_shape[axis] = end
        return torch.sparse.FloatTensor(new_indices, values[mask], torch.Size(new_shape))
    else:
        slices = [slice(None)] * axis + [slice(None, end)]
        return tensor[tuple(slices)]
