

"""Wrapper for PyEnvironments into TFEnvironments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import contextlib
from multiprocessing import pool
import threading
import functools
from absl import logging
import numpy as np
import gin
import torch.autograd as autograd
from agent.environments import batched_py_environment
from agent.environments import py_environment
from agent.environments import tf_environment
from agent.specs import tensor_spec
from agent.trajectories import time_step as ts



@contextlib.contextmanager
def _check_not_called_concurrently(lock):
  """Checks the returned context is not executed concurrently with any other."""
  if not lock.acquire(False):  # Non-blocking.
    raise RuntimeError(
        'Detected concurrent execution of TFPyEnvironment ops. Make sure the '
        'appropriate step_state is passed to step().')
  try:
    yield
  finally:
    lock.release()


@gin.configurable
class TFPyEnvironment(tf_environment.TFEnvironment):
  """Exposes a Python environment as an in-graph TF environment.

  This class supports Python environments that return nests of arrays as
  observations and accept nests of arrays as actions. The nest structure is
  reflected in the in-graph environment's observation and action structure.

  Implementation notes:

  * Since `tf.py_func` deals in lists of tensors, this class has some additional
    `tf.nest.flatten` and `tf.nest.pack_structure_as` calls.

  * This class currently cast rewards and discount to float32.
  """

  def __init__(self, environment, check_dims=False, isolation=False):
    """Initializes a new `TFPyEnvironment`.

    Args:
      environment: Environment to interact with, implementing
        `py_environment.PyEnvironment`.  Or a `callable` that returns
        an environment of this form.  If a `callable` is provided and
        `thread_isolation` is provided, the callable is executed in the
        dedicated thread.
      check_dims: Whether should check batch dimensions of actions in `step`.
      isolation: If this value is `False` (default), interactions with
        the environment will occur within whatever thread the methods of the
        `TFPyEnvironment` are run from.  For example, in TF graph mode, methods
        like `step` are called from multiple threads created by the TensorFlow
        engine; calls to step the environment are guaranteed to be sequential,
        but not from the same thread.  This creates problems for environments
        that are not thread-safe.

        Using isolation ensures not only that a dedicated thread (or
        thread-pool) is used to interact with the environment, but also that
        interaction with the environment happens in a serialized manner.

        If `isolation == True`, a dedicated thread is created for
        interactions with the environment.

        If `isolation` is an instance of `multiprocessing.pool.Pool` (this
        includes instances of `multiprocessing.pool.ThreadPool`, nee
        `multiprocessing.dummy.Pool` and `multiprocessing.Pool`, then this
        pool is used to interact with the environment.

        **NOTE** If using `isolation` with a `BatchedPyEnvironment`, ensure
        you create the `BatchedPyEnvironment` with `multithreading=False`, since
        otherwise the multithreading in that wrapper reverses the effects of
        this one.

    Raises:
      TypeError: If `environment` is not an instance of
        `py_environment.PyEnvironment` or subclasses, or is a callable that does
        not return an instance of `PyEnvironment`.
      TypeError: If `isolation` is not `True`, `False`, or an instance of
        `multiprocessing.pool.Pool`.
    """
    if not isolation:
      self._pool = None
    elif isinstance(isolation, pool.Pool):
      self._pool = isolation
    elif isolation:
      self._pool = pool.ThreadPool(1)
    else:
      raise TypeError(
          'isolation should be True, False, or an instance of '
          'a multiprocessing Pool or ThreadPool.  Saw: {}'.format(isolation))

    if callable(environment):
      environment = self._execute(environment)
    if not isinstance(environment, py_environment.PyEnvironment):
      raise TypeError(
          'Environment should implement py_environment.PyEnvironment')

    if not environment.batched:
      # If executing in an isolated thread, do not enable multiprocessing for
      # this environment.
      environment = batched_py_environment.BatchedPyEnvironment(
          [environment], multithreading=not self._pool)
    self._env = environment
    self._check_dims = check_dims

    if isolation and getattr(self._env, '_parallel_execution', None):
      logging.warn(
          'Wrapped environment is executing in parallel.  '
          'Perhaps it is a BatchedPyEnvironment with multithreading=True, '
          'or it is a ParallelPyEnvironment.  This conflicts with the '
          '`isolation` arg passed to TFPyEnvironment: interactions with the '
          'wrapped environment are no longer guaranteed to happen in a common '
          'thread.  Environment: %s', (self._env,))

    action_spec = tensor_spec.from_spec(self._env.action_spec())
    time_step_spec = tensor_spec.from_spec(self._env.time_step_spec())
    batch_size = self._env.batch_size if self._env.batch_size else 1

    super(TFPyEnvironment, self).__init__(time_step_spec,
                                          action_spec,
                                          batch_size)
    # Gather all the dtypes of the elements in time_step.
    self._time_step_dtypes = [
            s.dtype for s in self._flatten(self.time_step_spec())
    ]

    self._time_step = None
    self._lock = threading.Lock()

  def _flatten(self, nested):
    # Flatten a nested structure
    if isinstance(nested, dict):
        # If it's a dictionary, flatten its values
        return [item for value in nested.values() for item in self._flatten(value)]
    elif isinstance(nested, (list, tuple)):
        # If it's a list or tuple, flatten its elements
        return [item for sublist in nested for item in self._flatten(sublist)]
    else:
        # If it's not a list, tuple, or dict, return it as a single-element list
        return [nested]

  
  def __getattr__(self, name):
    """Enables access attributes of the wrapped PyEnvironment.

    Use with caution since methods of the PyEnvironment can be incompatible
    with TF.

    Args:
      name: Name of the attribute.

    Returns:
      The attribute.
    """
    if name in self.__dict__:
      return getattr(self, name)
    return getattr(self._env, name)

  def close(self):
    """Send close messages to the isolation pool and join it.

    Only has an effect when `isolation` was provided at init time.
    """
    if self._pool:
      self._pool.join()
      self._pool.close()
      self._pool = None

  @property
  def pyenv(self):
    """Returns the underlying Python environment."""
    return self._env

  def _execute(self, fn, *args, **kwargs):
    if not self._pool:
      return fn(*args, **kwargs)
    return self._pool.apply(fn, args=args, kwds=kwargs)

  def _numpy_function(self, func=None, inp=None, Tout=None, stateful=True, name=None):

    def _check_args_and_maybe_make_decorator(
    script_op, script_op_name, func=None, inp=None, Tout=None, **kwargs
    ):
      """Checks the arguments and returns a decorator if func is None."""
      if Tout is None:
        raise TypeError(
            "Missing required argument: 'Tout'\n"
            f"  If using {script_op_name} as a decorator, set `Tout`\n"
            "  **by name** above the function:\n"
            f"  `@{script_op_name}(Tout=tout)`"
        )

      if func is None:
        if inp is not None:
          raise TypeError(
              f"Don't set the `inp` argument when using {script_op_name} as a "
              "decorator (`func=None`)."
          )

        def py_function_decorator(fun):
          @functools.wraps(fun)
          def py_function_wrapper(*args):
            return script_op(fun, inp=args, Tout=Tout, **kwargs)

          return py_function_wrapper

        return py_function_decorator

      if inp is None:
        raise TypeError(
            "Missing argument `inp`:\n"
            "  You must set the `inp` argument (the list of arguments to the\n"
            f"  function), unless you use `{script_op_name}` as a decorator"
            "(`func=None`)."
        )

      return None
    

    decorator = _check_args_and_maybe_make_decorator(
      self._numpy_function,
      "tf.numpy_function",
      func=func,
      inp=inp,
      Tout=Tout,
      stateful=stateful,
      name=name,
    )
    if decorator is not None:
      return decorator

    # return py_func_common(func, inp, Tout, stateful=stateful, name=name)

  # TODO(b/123585179): Simplify this using py_environment.current_time_step().
  # There currently is a bug causing py_function to resolve variables
  # incorrectly when used inside autograph code. This decorator tells autograph
  # to call it directly instead of converting it when called from other
  # autograph-converted functions.
  # TODO(b/123600776): Remove override.
  def _current_time_step(self):
    """Returns the current ts.TimeStep.

    Returns:
      A `TimeStep` tuple of:
        step_type: A scalar int32 tensor representing the `StepType` value.
        reward: A scalar float32 tensor representing the reward at this
          timestep.
        discount: A scalar float32 tensor representing the discount [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """

    @_check_not_called_concurrently(self._lock)
    def _current_time_step_py():
        if self._time_step is None:
            self._time_step = self._env.reset()
        return self._flatten(self._time_step)

    def _isolated_current_time_step_py():
        return self._execute(_current_time_step_py)

    outputs = np.array(_isolated_current_time_step_py())
    
    step_type, reward, discount, task_obs, rgb, depth, occupancy_grid, *info = (outputs[i::13] for i in range(13))

    step_type, reward, discount = np.vstack(step_type), np.vstack(reward), np.vstack(discount)
    step_type, reward, discount = np.stack(step_type, axis=0).reshape(-1,1), np.stack(reward, axis=0).reshape(-1,1), np.stack(discount, axis=0).reshape(-1,1)

    flat_observations = [task_obs, rgb, depth, occupancy_grid]

    flat_observations = [np.stack(item, axis = 0) for item in flat_observations]

    info = [item.astype(np.float32).reshape(-1,1) for item in info]
    
    return self._set_names_and_shapes(step_type, reward, discount, *flat_observations, info=info)
  
  # Make sure this is called without conversion from tf.function.
  # TODO(b/123600776): Remove override.
  def _reset(self):
    """Returns the current `TimeStep` after resetting the environment.

    Returns:
      A `TimeStep` tuple of:
        step_type: A scalar int32 tensor representing the `StepType` value.
        reward: A scalar float32 tensor representing the reward at this
          timestep.
        discount: A scalar float32 tensor representing the discount [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """

    def _reset_py():
      with _check_not_called_concurrently(self._lock):
        self._time_step = self._env.reset()

    def _isolated_reset_py():
      return self._execute(_reset_py)

    class ResetFunction(autograd.Function):
      @staticmethod
      def forward(ctx):
          _isolated_reset_py()
          return torch.tensor(0.0)  # 返回一个占位张量

      @staticmethod
      def backward(ctx, grad_output):
          return None

    # 执行重置操作并在控制依赖下调用 current_time_step 方法
    reset_op = ResetFunction.apply()
    with torch.no_grad():  # 确保 current_time_step 不会被追踪梯度
        return self.current_time_step()

  def reload_model(self, model_ids):
    """Reload all environment with the new model_ids
    """
    self._env.reload_model(model_ids)

  def _pack_sequence_as(self, structure, flat_sequence):
    # Pack the sequence back to the original structure
    return torch.tensor([x for x in flat_sequence])
  
  # Make sure this is called without conversion from tf.function.
  # TODO(b/123600776): Remove override.
  def _step(self, actions, *args):
    """
    Returns a PyTorch tensor representing the environment step.

    Args:
      actions: A Tensor, or a nested dict, list or tuple of Tensors
        corresponding to `action_spec()`.

    Returns:
      A `TimeStep` tuple of:
        step_type: A scalar int32 tensor representing the `StepType` value.
        reward: A scalar float32 tensor representing the reward at this
          time_step.
        discount: A scalar float32 tensor representing the discount [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.

    Raises:
      ValueError: If any of the actions are scalars or their major axis is known
      and is not equal to `self.batch_size`.
    """

    

    def _step_py(*flattened_actions):
        with self._lock:
            flattened_actions = np.stack(flattened_actions, axis=0)
            self._time_step = self._env.step(flattened_actions, *args)
            return self._flatten(self._time_step)

    def _isolated_step_py(*flattened_actions):
        return self._execute(_step_py, *flattened_actions)

    with torch.no_grad():  # Using no_grad context to disable gradient tracking

      # Flatten actions
      flat_actions = [x.clone() for x in actions]  # Assuming actions is a list or similar iterable

      # Check dimensions if necessary
      if self._check_dims:
          for action in flat_actions:
              if action.dim() == 0 or (action.size(0) != self.batch_size):
                  raise ValueError(
                      'Expected actions whose major dimension is batch_size ({}), '
                      'but saw action with shape {}:\n   {}'.format(
                          self.batch_size, action.shape, action
                      )
                  )
      
      # Convert actions to numpy arrays, pass them to the isolated function, and convert back to tensors
      flat_actions_numpy = [action.cpu().numpy() for action in flat_actions]
      outputs = np.array(_isolated_step_py(*flat_actions_numpy))  # Assuming _isolated_step_py is defined

      num_environments = 4

      if len(outputs) // 6 == num_environments and len(flat_actions_numpy) == num_environments:
        step_type, reward, discount, *flat_observations = (outputs[i::7] for i in range(7))

        step_type, reward, discount = np.stack(step_type, axis=0).reshape(-1,1), np.stack(reward, axis=0).reshape(-1,1), np.stack(discount, axis=0).reshape(-1,1)

        flat_observations = [np.stack(item, axis = 0) for item in flat_observations]

        info = {}
        
      else:
        step_type, reward, discount, task_obs, rgb, depth, occupancy_grid, *info = (outputs[i::13] for i in range(13))

        step_type, reward, discount = np.vstack(step_type), np.vstack(reward), np.vstack(discount)
        step_type, reward, discount = np.stack(step_type, axis=0).reshape(-1,1), np.stack(reward, axis=0).reshape(-1,1), np.stack(discount, axis=0).reshape(-1,1)

        flat_observations = [task_obs, rgb, depth, occupancy_grid]

        flat_observations = [np.stack(item, axis = 0) for item in flat_observations]

        info = [item.astype(np.float32).reshape(-1,1) for item in info]

      return self._set_names_and_shapes(step_type, reward, discount, *flat_observations, info=info)

  def _set_names_and_shapes(self, step_type, reward, discount, *flat_observations, info):
    """Returns a `TimeStep` namedtuple."""
    step_type = torch.tensor(step_type).clone().detach().requires_grad_(False).rename(None)
    reward = torch.tensor(reward).clone().detach().requires_grad_(True).rename(None)
    discount = torch.tensor(discount).clone().detach().requires_grad_(True).rename(None)

    if not torch._C._get_tracing_state():
        # Shapes are not required in eager mode.
        reward = reward.numpy()
        step_type = step_type.numpy()
        discount = discount.numpy()

    info_spec = ['done', 'success', 'path_length', 'spl', 'episode_length', 'collision_step']

    # Give each tensor a meaningful name and set the static shape.
    named_observations = {}
    for obs, spec in zip(flat_observations, self.observation_spec()):
        obs = torch.tensor(obs)
        named_observation = obs.clone().detach().requires_grad_(True).rename(None)
        if not torch._C._get_tracing_state():
            named_observation = named_observation.numpy()
        named_observations[spec] = named_observation

    named_infos = {}
    for obs, spec in zip(info, info_spec):
        obs = torch.tensor(obs)
        named_info = obs.clone().detach().requires_grad_(False).rename(None)
        if not torch._C._get_tracing_state():
            named_info = named_info.numpy()
        named_infos[spec] = named_info


    return ts.TimeStep(step_type, reward, discount, named_observations, named_infos)
