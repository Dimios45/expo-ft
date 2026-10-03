"""YAM adapter for OpenPI's shared-cache, committed-prefix sampler.

Normalized inputs/outputs; no action execution or learner state is managed here.
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class RTCSampler:
    def __init__(self, model):
        self.graph, self.state = nnx.split(model)

        def sample(state, observation, prefix, noise):
            with jax.default_matmul_precision('highest'):
                return nnx.merge(self.graph, state).sample_actions_with_prefix(
                    jax.random.key(0), observation, prefix=prefix,
                    num_samples=noise.shape[1], num_steps=10, noise=noise)
        self._sample = jax.jit(sample)

    def candidates(self, observation, prefix, noise, replan_steps=8):
        prefix, noise = np.asarray(prefix, np.float32), np.asarray(noise, np.float32)
        if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[2] != 32:
            raise ValueError('prefix must be normalized (1, delay, 32)')
        delay = prefix.shape[1]
        if type(replan_steps) is not int or replan_steps < 1:
            raise ValueError('replan_steps must be a positive integer')
        if not 0 <= delay <= replan_steps or 2 * replan_steps > 30:
            raise ValueError('require delay <= replan_steps and 2*replan_steps <= horizon 30')
        if noise.ndim != 4 or noise.shape[0] != 1 or noise.shape[2:] != (30, 32) or noise.shape[1] < 1:
            raise ValueError('noise must have shape (1, candidates, 30, 32)')
        if not np.isfinite(prefix).all() or not np.isfinite(noise).all():
            raise ValueError('nonfinite sampler inputs')
        result = np.asarray(self._sample(self.state, observation, jnp.asarray(prefix), jnp.asarray(noise)))
        if not np.isfinite(result).all():
            raise FloatingPointError('nonfinite sampled actions')
        expected = np.broadcast_to(prefix, (noise.shape[1], delay, 32))
        if not np.array_equal(result[:, :delay], expected):
            raise AssertionError('committed prefix changed')
        return result, result[:, delay:delay+replan_steps, :14]
