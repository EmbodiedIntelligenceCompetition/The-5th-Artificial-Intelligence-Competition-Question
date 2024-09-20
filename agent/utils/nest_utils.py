
"""Utilities for handling nested tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import torch
import numpy as np
from collections.abc import Sequence
from itertools import islice
from agent.utils import composite

from collections.abc import Mapping, Sequence

def _is_namedtuple(x):
    return isinstance(x, tuple) and hasattr(x, '_fields')

def flatten_with_tuple_paths(structure, prefix=()):
    """Flatten a nested structure into a list of paths and values.

    Args:
        structure: The nested structure (which can be a dict, list, tuple, etc.).
        prefix: The prefix for the current path (used during recursion).

    Returns:
        A list of (path, value) tuples where path is a tuple of indices.
    """
    if isinstance(structure, Mapping):
        flat_items = []
        for key, value in structure.items():
            flat_items.extend(flatten_with_tuple_paths(value, prefix + (key,)))
        return flat_items
    elif isinstance(structure, Sequence) and not isinstance(structure, (str, bytes)):
        flat_items = []
        for idx, value in enumerate(structure):
            flat_items.extend(flatten_with_tuple_paths(value, prefix + (idx,)))
        return flat_items
    elif _is_namedtuple(structure):
        flat_items = []
        for field in structure._fields:
            flat_items.extend(flatten_with_tuple_paths(getattr(structure, field), prefix + (field,)))
        return flat_items
    else:
        return [(prefix, structure)]


def flatten(structure):
    return [x[1] for x in flatten_with_tuple_paths(structure)]

def pack_sequence_as(structure, flat_sequence):
    flat_structure = flatten_with_tuple_paths(structure)
    flat_iterator = iter(flat_sequence)
    return _pack_sequence_as_helper(structure, flat_structure, flat_iterator)

def _pack_sequence_as_helper(structure, flat_structure, flat_iterator):
    if isinstance(structure, Mapping):
        return type(structure)({
            key: _pack_sequence_as_helper(value, flat_structure, flat_iterator)
            for key, value in structure.items()
        })
    elif isinstance(structure, Sequence) and not isinstance(structure, (str, bytes)):
        return type(structure)(
            _pack_sequence_as_helper(value, flat_structure, flat_iterator)
            for value in structure
        )
    elif _is_namedtuple(structure):
        return type(structure)(
            *(_pack_sequence_as_helper(getattr(structure, field), flat_structure, flat_iterator)
              for field in structure._fields)
        )
    else:
        return next(flat_iterator)

def fast_map_structure_flatten(func, structure, *flat_structure, **kwargs):
    expand_composites = kwargs.get('expand_composites', False)
    entries = zip(*flat_structure)
    return pack_sequence_as(
        structure, [func(*x) for x in entries]
    )

def fast_map_structure(func, *structure, **kwargs):
    expand_composites = kwargs.get('expand_composites', False)
    flat_structure = [
        flatten(s)
        for s in structure
    ]
    entries = zip(*flat_structure)
    print(structure)
    print(flat_structure)
    return pack_sequence_as(
        structure[0], [func(*x) for x in entries]
    )

def has_tensors(*x):
    return np.any([
        torch.is_tensor(t) for t in flatten(x)
    ])

def is_batched_nested_tensors(tensors, specs, num_outer_dims=1):
    """Compares tensors to specs to determine if all tensors are batched or not.

    Args:
        tensors: Nested list/tuple/dict of Tensors.
        specs: Nested list/tuple/dict of Tensors describing the shape of unbatched tensors.
        num_outer_dims: The integer number of dimensions that are considered batch dimensions. Default 1.

    Returns:
        True if all Tensors are batched and False if all Tensors are unbatched.
    Raises:
        ValueError: If
            1. Any of the tensors or specs have shapes with ndims == None, or
            2. The shape of Tensors are not compatible with specs, or
            3. A mix of batched and unbatched tensors are provided.
            4. The tensors are batched but have an incorrect number of outer dims.
    """
    tensor_shapes = [t.shape for t in flatten(tensors)]
    spec_shapes = [_spec_shape(s) for s in flatten(specs)]

    if any(spec_shape is None for spec_shape in spec_shapes):
        raise ValueError('All specs should have ndims defined. Saw shapes: %s' % spec_shapes)

    if any(tensor_shape is None for tensor_shape in tensor_shapes):
        raise ValueError('All tensors should have ndims defined. Saw shapes: %s' % tensor_shapes)

    is_unbatched = [
        spec_shape == tensor_shape
        for spec_shape, tensor_shape in zip(spec_shapes, tensor_shapes)
    ]
    if all(is_unbatched):
        return False

    tensor_ndims_discrepancy = [
        len(tensor_shape) - len(spec_shape)
        for spec_shape, tensor_shape in zip(spec_shapes, tensor_shapes)
    ]

    tensor_matches_spec = [
        spec_shape == tensor_shape[discrepancy:]
        for discrepancy, spec_shape, tensor_shape in zip(
            tensor_ndims_discrepancy, spec_shapes, tensor_shapes)
    ]

    is_batched = (
        all(discrepancy == num_outer_dims
            for discrepancy in tensor_ndims_discrepancy) and
        all(tensor_matches_spec)
    )

    if is_batched:
        return True

    if all(
        discrepancy == tensor_ndims_discrepancy[0]
        for discrepancy in tensor_ndims_discrepancy) and all(tensor_matches_spec):
        return False

    raise ValueError(
        'Received a mix of batched and unbatched Tensors, or Tensors'
        ' are not compatible with Specs. num_outer_dims: %d.\n'
        'Saw tensor_shapes:\n   %s\n'
        'And spec_shapes:\n   %s' %
        (num_outer_dims, tensor_shapes, spec_shapes)
    )

def _spec_shape(t):
    if isinstance(t, torch.Tensor):
        return t.shape
    else:
        raise TypeError(f"Expected torch.Tensor, but got {type(t)}")

def batch_nested_tensors(tensors, specs=None):
    """Add batch dimension if needed to nested tensors while checking their specs.

    If specs is None, a batch dimension is added to each tensor.
    If specs are provided, each tensor is compared to the corresponding spec,
    and a batch dimension is added only if the tensor doesn't already have it.

    Args:
        tensors: Nested list/tuple or dict of Tensors.
        specs: Nested list/tuple or dict of TensorSpecs, describing the shape of the
        non-batched Tensors.

    Returns:
        A nested batched version of each tensor.
    Raises:
        ValueError: if the tensors and specs have incompatible dimensions or shapes.
    """
    if specs is None:
        return torch.nest.map_structure(lambda x: composite.expand_dims(x, 0), tensors)

    torch.nest.assert_same_structure(tensors, specs)

    flat_tensors = torch.nest.flatten(tensors)
    flat_shapes = [_spec_shape(s) for s in torch.nest.flatten(specs)]
    batched_tensors = []

    for tensor, shape in zip(flat_tensors, flat_shapes):
        if tensor.dim() == len(shape):
            assert tensor.shape == shape, 'Tensor does not have the correct shape.'
            tensor = composite.expand_dims(tensor, 0)
        elif tensor.dim() == len(shape) + 1:
            assert tensor.shape[1:] == shape, 'Tensor does not have the correct shape.'
        else:
            raise ValueError('Tensor does not have the correct dimensions. '
                             f'tensor.shape {tensor.shape} expected shape {shape}')
        batched_tensors.append(tensor)
    return torch.nest.pack_sequence_as(tensors, batched_tensors)

def _flatten_and_check_shape_nested_tensors(tensors, specs, num_outer_dims=1):
    """Flatten nested tensors and check their shape for use in other functions."""
    torch.nest.assert_same_structure(tensors, specs)
    flat_tensors = torch.nest.flatten(tensors)
    flat_shapes = [_spec_shape(s) for s in torch.nest.flatten(specs)]
    for tensor, shape in zip(flat_tensors, flat_shapes):
        if tensor.dim() == len(shape):
            assert tensor.shape == shape, 'Tensor does not have the correct shape.'
        elif tensor.dim() == len(shape) + num_outer_dims:
            assert tensor.shape[num_outer_dims:] == shape, 'Tensor does not have the correct shape.'
        else:
            raise ValueError('Tensor does not have the correct dimensions. '
                             f'tensor.shape {tensor.shape} expected shape {[None] + list(shape)}')
    return flat_tensors, flat_shapes

def flatten_and_check_shape_nested_specs(specs, reference_specs):
    """Flatten nested specs and check their shape for use in other functions."""
    try:
        flat_specs, flat_shapes = _flatten_and_check_shape_nested_tensors(specs, reference_specs, num_outer_dims=0)
    except ValueError:
        raise ValueError('specs must be compatible with reference_specs'
                         f'; instead got specs={specs}, reference_specs={reference_specs}')
    return flat_specs, flat_shapes

def unbatch_nested_tensors(tensors, specs=None):
    """Remove the batch dimension if needed from nested tensors using their specs.

    If specs is None, the first dimension of each tensor will be removed.
    If specs are provided, each tensor is compared to the corresponding spec,
    and the first dimension is removed only if the tensor was batched.

    Args:
        tensors: Nested list/tuple or dict of batched Tensors.
        specs: Nested list/tuple or dict of TensorSpecs, describing the shape of the
        non-batched Tensors.

    Returns:
        A nested non-batched version of each tensor.
    Raises:
        ValueError: if the tensors and specs have incompatible dimensions or shapes.
    """
    if specs is None:
        return torch.nest.map_structure(lambda x: composite.squeeze(x, 0), tensors)

    unbatched_tensors = []
    flat_tensors, flat_shapes = _flatten_and_check_shape_nested_tensors(tensors, specs)
    for tensor, shape in zip(flat_tensors, flat_shapes):
        if tensor.dim() == len(shape) + 1:
            tensor = composite.squeeze(tensor, 0)
        unbatched_tensors.append(tensor)
    return torch.nest.pack_sequence_as(tensors, unbatched_tensors)

def split_nested_tensors(tensors, specs, num_or_size_splits):
    """Split batched nested tensors, on batch dim (outer dim), into a list.

    Args:
        tensors: Nested list/tuple or dict of batched Tensors.
        specs: Nested list/tuple or dict of TensorSpecs, describing the shape of the
        non-batched Tensors.
        num_or_size_splits: Same as argument for torch.split. Either a python integer
        indicating the number of splits along batch_dim or a list of integer
        Tensors containing the sizes of each output tensor along batch_dim. If a
        scalar then it must evenly divide value.shape[axis]; otherwise the sum of
        sizes along the split dimension must match that of the value.

    Returns:
        A list of nested non-batched version of each tensor, where each list item
        corresponds to one batch item.
    Raises:
        ValueError: if the tensors and specs have incompatible dimensions or shapes.
        ValueError: if a non-scalar is passed and there are SparseTensors in the
        structure.
    """
    split_tensor_lists = []
    flat_tensors, flat_shapes = _flatten_and_check_shape_nested_tensors(tensors, specs)
    for tensor, shape in zip(flat_tensors, flat_shapes):
        if tensor.dim() == len(shape):
            raise ValueError('Can only split tensors with a batch dimension.')
        if tensor.dim() == len(shape) + 1:
            split_tensors = torch.split(tensor, num_or_size_splits)
        split_tensor_lists.append(split_tensors)
    split_tensors_zipped = zip(*split_tensor_lists)
    return [torch.nest.pack_sequence_as(tensors, zipped) for zipped in split_tensors_zipped]

def unstack_nested_tensors(tensors, specs):
    """Make list of unstacked nested tensors.

    Args:
        tensors: Nested tensors whose first dimension is to be unstacked.
        specs: Tensor specs for tensors.

    Returns:
        A list of the unstacked nested tensors.
    Raises:
        ValueError: if the tensors and specs have incompatible dimensions or shapes.
    """
    unstacked_tensor_lists = []
    flat_tensors, flat_shapes = _flatten_and_check_shape_nested_tensors(tensors, specs)
    for tensor, shape in zip(flat_tensors, flat_shapes):
        if tensor.dim() == len(shape):
            raise ValueError('Can only unstack tensors with a batch dimension.')
        if tensor.dim() == len(shape) + 1:
            unstacked_tensors = torch.unbind(tensor)
        unstacked_tensor_lists.append(unstacked_tensors)
    unstacked_tensors_zipped = zip(*unstacked_tensor_lists)
    return [torch.nest.pack_sequence_as(tensors, zipped) for zipped in unstacked_tensors_zipped]

def stack_nested_tensors(tensors):
    """Stacks a list of nested tensors along the first dimension.

    Args:
        tensors: A list of nested tensors to be stacked along the first dimension.

    Returns:
        A stacked nested tensor.
    """
    return torch.nest.map_structure(lambda *tensors: torch.stack(tensors), *tensors)

def flatten_multi_batched_nested_tensors(tensors, specs):
    """Reshape tensors to contain only one batch dimension.

    For each tensor, it checks the number of extra dimensions beyond those in
    the spec, and reshapes tensor to have only one batch dimension.
    NOTE: Each tensor's batch dimensions must be the same.

    Args:
        tensors: Nested list/tuple or dict of batched Tensors or SparseTensors.
        specs: Nested list/tuple or dict of TensorSpecs, describing the shape of the
        non-batched Tensors.

    Returns:
        A nested version of each tensor with a single batch dimension.
        A list of the batch dimensions which were flattened.
    Raises:
        ValueError: if the tensors and specs have incompatible dimensions or shapes.
    """
    torch.nest.assert_same_structure(tensors, specs)
    flat_tensors = torch.nest.flatten(tensors)
    flat_shapes = [_spec_shape(s) for s in torch.nest.flatten(specs)]
    out_tensors = []
    batch_dims = []
    for i, (tensor, shape) in enumerate(zip(flat_tensors, flat_shapes)):
        if i == 0:  # Set batch_dims based on first tensor.
            batch_dims = tensor.shape[:tensor.dim() - len(shape)]
            batch_prod = np.prod(batch_dims)
        reshaped_dims = [batch_prod] + list(shape)
        out_tensors.append(composite.reshape(tensor, reshaped_dims))
    return torch.nest.pack_sequence_as(tensors, out_tensors), batch_dims

def get_outer_shape(nested_tensor, spec):
    """Runtime batch dims of tensor's batch dimension `dim`."""
    torch.nest.assert_same_structure(nested_tensor, spec)
    first_tensor = torch.nest.flatten(nested_tensor)[0]
    first_spec = torch.nest.flatten(spec)[0]

    # Check tensors have same batch shape.
    num_outer_dims = (first_tensor.dim() - len(first_spec))
    if not is_batched_nested_tensors(nested_tensor, spec, num_outer_dims=num_outer_dims):
        return []

    return first_tensor.shape[:num_outer_dims]

def get_outer_rank(tensors, specs):
    """Compares tensors to specs to determine the number of batch dimensions.

    For each tensor, it checks the dimensions with respect to specs and
    returns the number of batch dimensions if all nested tensors and
    specs agree with each other.

    Args:
        tensors: Nested list/tuple/dict of Tensors or SparseTensors.
        specs: Nested list/tuple/dict of TensorSpecs, describing the shape of
        unbatched tensors.

    Returns:
        The number of outer dimensions for all Tensors (zero if all are
        unbatched or empty).
    Raises:
        ValueError: If
        1. Any of the tensors or specs have shapes with ndims == None, or
        2. The shape of Tensors are not compatible with specs, or
        3. A mix of batched and unbatched tensors are provided.
        4. The tensors are batched but have an incorrect number of outer dims.
    """
    torch.nest.assert_same_structure(tensors, specs)
    tensor_shapes = [t.shape for t in torch.nest.flatten(tensors)]
    spec_shapes = [_spec_shape(s) for s in torch.nest.flatten(specs)]

    if any(spec_shape is None for spec_shape in spec_shapes):
        raise ValueError(f'All specs should have ndims defined. Saw shapes: {spec_shapes}')

    if any(tensor_shape is None for tensor_shape in tensor_shapes):
        raise ValueError(f'All tensors should have ndims defined. Saw shapes: {tensor_shapes}')

    is_unbatched = [
        tensor_shape == spec_shape
        for spec_shape, tensor_shape in zip(spec_shapes, tensor_shapes)
    ]
    if all(is_unbatched):
        return 0

    tensor_ndims_discrepancy = [
        len(tensor_shape) - len(spec_shape)
        for spec_shape, tensor_shape in zip(spec_shapes, tensor_shapes)
    ]

    tensor_matches_spec = [
        tensor_shape[discrepancy:] == spec_shape
        for discrepancy, spec_shape, tensor_shape in zip(
            tensor_ndims_discrepancy, spec_shapes, tensor_shapes)
    ]

    num_outer_dims = tensor_ndims_discrepancy[0]

    is_batched = (
        all(discrepancy == num_outer_dims for discrepancy in tensor_ndims_discrepancy) and
        all(tensor_matches_spec)
    )

    if is_batched:
        return num_outer_dims

    incorrect_batch_dims = (
        tensor_ndims_discrepancy and
        all(discrepancy == tensor_ndims_discrepancy[0] and discrepancy >= 0
            for discrepancy in tensor_ndims_discrepancy) and
        all(tensor_matches_spec)
    )

    if incorrect_batch_dims:
        raise ValueError(f'Received tensors with {tensor_ndims_discrepancy[0]} outer dimensions. '
                         f'Expected {num_outer_dims}.')

    raise ValueError(f'Received a mix of batched and unbatched Tensors, or Tensors'
                     f' are not compatible with Specs. num_outer_dims: {num_outer_dims}.\n'
                     f'Saw tensor_shapes:\n   {tensor_shapes}\n'
                     f'And spec_shapes:\n   {spec_shapes}')

def batch_nested_array(nested_array):
    return torch.nest.map_structure(lambda x: np.expand_dims(x, 0), nested_array)

def unbatch_nested_array(nested_array):
    return torch.nest.map_structure(lambda x: np.squeeze(x, 0), nested_array)

def unstack_nested_arrays(nested_array):
    """Unstack/unbatch a nest of numpy arrays.

    Args:
        nested_array: Nest of numpy arrays where each array has shape [batch_size,
        ...].

    Returns:
        A list of length batch_size where each item in the list is a nest
        having the same structure as `nested_array`.
    """
    def _unstack(array):
        if array.shape[0] == 1:
            arrays = [array]
        else:
            arrays = np.split(array, array.shape[0])
        return [np.reshape(a, a.shape[1:]) for a in arrays]

    unstacked_arrays_zipped = zip(*[_unstack(array) for array in torch.nest.flatten(nested_array)])
    return [torch.nest.pack_sequence_as(nested_array, zipped) for zipped in unstacked_arrays_zipped]

def stack_nested_arrays(nested_arrays):
    """Stack/batch a list of nested numpy arrays.

    Args:
        nested_arrays: A list of nested numpy arrays of the same shape/structure.

    Returns:
        A nested array containing batched items, where each batched item is obtained
        by stacking corresponding items from the list of nested_arrays.
    """
    nested_arrays_flattened = [torch.nest.flatten(a) for a in nested_arrays]
    batched_nested_array_flattened = [np.stack(a) for a in zip(*nested_arrays_flattened)]
    return torch.nest.pack_sequence_as(nested_arrays[0], batched_nested_array_flattened)

def get_outer_array_shape(nested_array, spec):
    """Batch dims of array's batch dimension `dim`."""
    first_array = torch.nest.flatten(nested_array)[0]
    first_spec = torch.nest.flatten(spec)[0]
    num_outer_dims = len(first_array.shape) - len(first_spec.shape)
    return first_array.shape[:num_outer_dims]

def where(condition, true_outputs, false_outputs):
    """Generalization of torch.where supporting nests as the outputs.

    Args:
        condition: A boolean Tensor of shape [B,].
        true_outputs: Tensor or nested tuple of Tensors of any dtype, each with
        shape [B, ...], to be split based on `condition`.
        false_outputs: Tensor or nested tuple of Tensors of any dtype, each with
        shape [B, ...], to be split based on `condition`.

    Returns:
        Interleaved output from `true_outputs` and `false_outputs` based on
        `condition`.
    """
    return torch.nest.map_structure(lambda t, f: torch.where(condition, t, f), true_outputs, false_outputs)