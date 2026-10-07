"""Inference-only EXPO/RTC adapter. One frozen base, no optimizer on serving GPU."""
from __future__ import annotations

import io

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from PIL import Image

from .artifacts import flatten, tensor_spec
from .learner import Settings, YamEXPO
from .lan_protocol import schedule, vector


@struct.dataclass
class Parameters:
    params: object


class Selector(YamEXPO):
    def __init__(self, settings, image_size=224):
        self.cfg = settings
        self._make_modules(settings)
        self.image_size = image_size
        self._select = jax.jit(self._selection)
        c = settings
        def abstract(key):
            return dict(encoder=self.encoder.init(key, jnp.zeros((1, image_size, image_size, 9)))['params'],
                        editor=self.editor.init(key, jnp.zeros((1, c.image_latent)),
                                                actions=jnp.zeros((1, c.width)), p=jnp.zeros((1, 14)))['params'],
                        target_q=self.critic.init(key, jnp.zeros((1, c.image_latent)),
                                                 jnp.zeros((1, c.width)), p=jnp.zeros((1, 14)))['params'])
        shapes = jax.eval_shape(abstract, jax.random.key(0))
        def spec(tree, prefix=''):
            result = {}
            for k, v in tree.items():
                name = f'{prefix}/{k}' if prefix else k
                if hasattr(v, 'items'):
                    result.update(spec(v, name))
                else:
                    result[name] = dict(shape=list(v.shape), dtype=str(v.dtype))
            return result
        self.spec = spec(shapes)

    def stage(self, params):
        if tensor_spec(flatten(params)) != self.spec:
            raise ValueError('selector tensor schema mismatch')
        params = jax.tree.map(jnp.asarray, params)
        c, n = self.cfg, self.image_size
        selected, info = self.select_params(params, np.zeros((1, n, n, 9), np.float32),
                                           np.zeros((1, 14), np.float32),
                                           np.zeros((1, c.candidates, c.width), np.float32), 77)
        if not np.isfinite(info['q_values']).all() or np.max(np.abs(selected)) > c.edit_scale + 1e-5:
            raise ValueError('candidate failed selector validation')
        if c.mask_gripper_edits and np.any(selected.reshape(c.horizon, 14)[:, [6, 13]]):
            raise ValueError('candidate edits masked grippers')
        return params

    def select_params(self, params, images, states, base, seed):
        state = dict(encoder=Parameters(params['encoder']), editor=Parameters(params['editor']),
                     target_q=params['target_q'])
        selected, info = self._select(state, images, states, base, jax.random.PRNGKey(seed))
        return np.asarray(selected), jax.tree.map(np.asarray, info)


def settings(config):
    schedule(config['replan_steps'], config['delay'], config['candidates'])
    return Settings(horizon=config['replan_steps'], candidates=config['candidates'], edits=config['candidates'],
                    edit_scale=0.05, mask_gripper_edits=True, editor_lr=1e-5,
                    temperature_lr=1e-5, init_temperature=0.01, gradient_clip=1.0,
                    initial_editor_logstd=-3.0, scale_entropy_target=True)


def decode_images(obs):
    images = {}
    for key, value in obs['images'].items():
        if isinstance(value, bytes):
            with Image.open(io.BytesIO(value)) as image:
                if image.width * image.height > 4096 * 4096:
                    raise ValueError('oversized decoded image')
                images[key] = np.asarray(image.convert('RGB'))
        else:
            images[key] = np.asarray(value)
    return images


class RTCPolicy:
    def __init__(self, config):
        from expo_ft.conversion.yam_loader import YamJaxPolicy
        from .rtc_sampling import RTCSampler
        self.config = config
        self.cfg = settings(config)
        if config['dtype'] != 'bfloat16':
            raise ValueError('LAN runtime requires bf16 base; FP32 is diagnostic only')
        policy = YamJaxPolicy(config['checkpoint'], config['tokenizer'], dtype='bfloat16')
        self.processor = policy.processor
        self.sampler = RTCSampler(policy.model)
        self.selector = Selector(self.cfg)
        del policy  # sampler retains the only NNX state needed for inference

    def prepare(self, obs):
        return self.processor.prepare_numpy(decode_images(obs), obs['state'], self.config['prompt'])

    def propose(self, obs, prefix, seed):
        from expo_ft.conversion.yam_loader import as_observation
        prefix = vector(prefix, (len(prefix), 14), 'committed prefix')
        if len(prefix) not in (0, self.config['delay']):
            raise ValueError('prefix has wrong deployment delay')
        normalized = self.normalize_actions(prefix)
        padded = np.pad(normalized, ((0, 0), (0, 18)))[None]
        noise = np.random.default_rng(seed).standard_normal((1, self.cfg.candidates, 30, 32)).astype(np.float32)
        _, window = self.sampler.candidates(as_observation(self.prepare(obs)), padded, noise,
                                             replan_steps=self.cfg.horizon)
        return window

    def normalize_actions(self, value):
        lo = self.processor.output_stats['action.q01']
        hi = self.processor.output_stats['action.q99']
        return (2 * (value - lo) / np.where(hi == lo, self.processor.eps, hi - lo) - 1).astype(np.float32)

    def select(self, params, obs, candidates, seed):
        from expo_ft.conversion.yam_pi05 import JAX_CAMERAS
        if params is None:
            chosen = candidates[0]  # untouched base for bootstrap episode
        else:
            data = self.prepare(obs)
            images = np.concatenate([data[k] for k in JAX_CAMERAS], -1)
            chosen, _ = self.selector.select_params(params, images, data['state'][:, :14],
                                                   candidates.reshape(1, self.cfg.candidates, self.cfg.width), seed)
            chosen = chosen.reshape(self.cfg.horizon, 14)
        return self.processor.unnormalize_actions(chosen)
