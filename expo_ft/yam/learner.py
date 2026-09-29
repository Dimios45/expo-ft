"""YAM adaptation of agents/alg/expo_ft.py losses; full 30x14 action chunks.

Uses upstream ResNet, Q ensemble, and state multiplexers. Sampling the VLA outside
this JIT permits sequential candidate inference on one GPU. No target encoder,
entropy term in the Q backup, or Q-gradient update to the base VLA is introduced.
"""

from dataclasses import asdict, dataclass
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from expo_ft.networks import (
    MLP,
    BatchEncoder,
    Ensemble,
    PixelEditMultiplexer,
    PixelMultiplexer,
    StateActionValue,
)
from expo_ft.networks.encoders import ResNetV2Encoder


@dataclass(frozen=True)
class Settings:
    candidates: int = 8
    edits: int = 8
    edit_scale: float = 0.2
    discount: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    hidden: tuple = (256, 256, 256)
    stages: tuple = (3, 4, 6, 3)
    filters: int = 64
    image_latent: int = 512
    state_latent: int = 64
    actor_mode: str = "frozen"
    actor_lr: float = 1e-5
    seed: int = 42
    num_qs: int = 10
    num_min_qs: int = 2
    horizon: int = 30
    action_dim: int = 14
    # Opt-in stability profile; defaults preserve existing checkpoint behavior.
    editor_lr: float | None = None
    temperature_lr: float | None = None
    init_temperature: float = 1.0
    gradient_clip: float | None = None
    mask_gripper_edits: bool = False
    initial_editor_logstd: float | None = None
    scale_entropy_target: bool = False

    @property
    def width(self):
        return self.horizon * self.action_dim

    def dictionary(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("hidden", "stages"):
            if k in d:
                d[k] = tuple(d[k])
        c = cls(**d)
        if (c.num_qs, c.num_min_qs, c.horizon, c.action_dim) != (10, 2, 30, 14):
            raise ValueError("Require Q10/min2, H30, D14")
        if (
            not 1 <= c.edits <= c.candidates
            or c.candidates < 2
            or c.actor_mode not in ("frozen", "expert")
        ):
            raise ValueError("Invalid candidate/editor/actor configuration")
        return c


class EditorNormal(nn.Module):
    width: int
    hidden: tuple
    initial_logstd: float | None = None

    @nn.compact
    def __call__(self, x, training=False):
        x = MLP(self.hidden, activate_final=True)(x, training=training)
        init = (
            nn.initializers.xavier_uniform()
            if self.initial_logstd is None
            else nn.initializers.zeros_init()
        )
        mean = nn.Dense(self.width, kernel_init=init, name="OutputDenseMean")(x)
        bias = nn.initializers.constant(self.initial_logstd or 0.0)
        logstd = nn.Dense(
            self.width, kernel_init=init, bias_init=bias, name="OutputDenseLogStd"
        )(x)
        return mean, jnp.clip(logstd, -20, 2)


def gaussian_edit(mean, logstd, key, scale):
    eps = jax.random.normal(key, mean.shape)
    u = mean + jnp.exp(logstd) * eps
    logp = -0.5 * (eps**2 + 2 * logstd + jnp.log(2 * jnp.pi))
    logp -= 2 * (jnp.log(2.0) - u - jax.nn.softplus(-2 * u))
    return jnp.tanh(u) * scale, logp.sum(-1) - mean.shape[-1] * jnp.log(scale)


def chunk_target(rewards, masks, steps, next_q, gamma):
    return rewards + gamma**steps * masks * next_q


class YamEXPO:
    def __init__(self, settings=None, image_size=224):
        settings = settings or Settings()
        self.cfg = c = settings
        self.encoder = BatchEncoder(
            partial(ResNetV2Encoder, stage_sizes=c.stages, num_filters=c.filters),
            c.image_latent,
        )
        self.critic = PixelMultiplexer(
            partial(
                Ensemble,
                net_cls=partial(
                    StateActionValue,
                    base_cls=partial(
                        MLP,
                        hidden_dims=c.hidden,
                        activate_final=True,
                        use_layer_norm=True,
                    ),
                ),
                num=c.num_qs,
            ),
            c.state_latent,
            include_state=True,
        )
        self.editor = PixelEditMultiplexer(
            EditorNormal(c.width, c.hidden, c.initial_editor_logstd),
            c.state_latent,
            include_state=True,
        )
        keys = jax.random.split(jax.random.PRNGKey(c.seed), 4)
        image = jnp.zeros((1, image_size, image_size, 9))
        z = jnp.zeros((1, c.image_latent))
        s = jnp.zeros((1, 14))
        a = jnp.zeros((1, c.width))

        def optimizer(lr):
            adam = optax.adam(lr)
            return (
                optax.chain(optax.clip_by_global_norm(c.gradient_clip), adam)
                if c.gradient_clip is not None
                else adam
            )

        def ts(module, params, lr=None):
            return TrainState.create(
                apply_fn=module.apply, params=params, tx=optimizer(lr or c.lr)
            )

        self.state = {
            "encoder": ts(self.encoder, self.encoder.init(keys[0], image)["params"]),
            "critic": ts(self.critic, self.critic.init(keys[1], z, a, p=s)["params"]),
            "editor": ts(
                self.editor,
                self.editor.init(keys[2], z, actions=a, p=s)["params"],
                c.editor_lr,
            ),
            "temperature": TrainState.create(
                apply_fn=jnp.exp,
                params={"log_temp": jnp.log(jnp.array(c.init_temperature))},
                tx=optax.adam(c.temperature_lr or c.lr),
            ),
            "rng": keys[3],
            "updates": jnp.array(0, jnp.int32),
        }
        self.state["target_q"] = self.state["critic"].params
        self._update = jax.jit(self._step)
        self._select = jax.jit(self._selection)

    def _selection(self, state, images, states, base, key):
        c = self.cfg
        b, n, d = base.shape
        k = n + c.edits
        z = self.encoder.apply({"params": state["encoder"].params}, images)
        key_edit, key_pair = jax.random.split(key)
        chosen = base[:, : c.edits].reshape(-1, d)
        mean, std = self.editor.apply(
            {"params": state["editor"].params},
            jnp.repeat(z, c.edits, 0),
            actions=chosen,
            p=jnp.repeat(states, c.edits, 0),
        )
        edits, _ = gaussian_edit(mean, std, key_edit, c.edit_scale)
        if c.mask_gripper_edits:
            edits = edits.reshape(-1, c.horizon, c.action_dim)
            edits = edits.at[..., jnp.array([6, 13])].set(0).reshape(-1, d)
        actions = jnp.concatenate([base, (chosen + edits).reshape(b, c.edits, d)], 1)
        qs = self.critic.apply(
            {"params": state["target_q"]},
            jnp.repeat(z, k, 0),
            actions.reshape(-1, d),
            p=jnp.repeat(states, k, 0),
        )
        pair = jax.random.choice(key_pair, 10, (2,), replace=False)
        scores = qs[pair].min(0).reshape(b, k)
        idx = jnp.argmax(scores, 1)
        return actions[jnp.arange(b), idx], {
            "q_values": qs.reshape(10, b, k),
            "pair": pair,
            "scores": scores,
            "selected": idx,
            "candidates": actions,
        }

    def _step(self, state, batch):
        c = self.cfg
        rng, select_key, backup_key, edit_key = jax.random.split(state["rng"], 4)
        nxt, selection = self._selection(
            state,
            batch["next_images"],
            batch["next_states"],
            batch["next_candidates"],
            select_key,
        )
        zn = self.encoder.apply(
            {"params": state["encoder"].params}, batch["next_images"]
        )
        nq = self.critic.apply(
            {"params": state["target_q"]}, zn, nxt, p=batch["next_states"]
        )
        pair = jax.random.choice(backup_key, 10, (2,), replace=False)
        target = jax.lax.stop_gradient(
            chunk_target(
                batch["rewards"],
                batch["masks"],
                batch["steps"],
                nq[pair].min(0),
                c.discount,
            )
        )

        def loss(p):
            z = self.encoder.apply({"params": p["encoder"]}, batch["images"])
            q = self.critic.apply(
                {"params": p["critic"]}, z, batch["actions"], p=batch["states"]
            )
            return jnp.mean((q - target[None]) ** 2), q.mean()

        (qloss, qmean), g = jax.value_and_grad(loss, has_aux=True)(
            {"encoder": state["encoder"].params, "critic": state["critic"].params}
        )
        encoder = state["encoder"].apply_gradients(grads=g["encoder"])
        critic = state["critic"].apply_gradients(grads=g["critic"])
        z = jax.lax.stop_gradient(
            self.encoder.apply({"params": encoder.params}, batch["images"])
        )

        def eloss(p):
            mean, std = self.editor.apply(
                {"params": p}, z, actions=batch["actions"], p=batch["states"]
            )
            edit, logp = gaussian_edit(mean, std, edit_key, c.edit_scale)
            q = self.critic.apply(
                {"params": critic.params}, z, batch["actions"] + edit, p=batch["states"]
            ).mean(0)
            return (
                jnp.exp(state["temperature"].params["log_temp"]) * logp - q
            ).mean(), -logp.mean()

        (eloss_value, entropy), eg = jax.value_and_grad(eloss, has_aux=True)(
            state["editor"].params
        )
        editor = state["editor"].apply_gradients(grads=eg)
        tg = jax.grad(
            lambda t: (
                jnp.exp(t["log_temp"]) * jax.lax.stop_gradient(entropy + c.width / 2)
            )
        )(state["temperature"].params)
        temp = state["temperature"].apply_gradients(grads=tg)
        updated = {
            "encoder": encoder,
            "critic": critic,
            "editor": editor,
            "temperature": temp,
            "target_q": optax.incremental_update(
                critic.params, state["target_q"], c.tau
            ),
            "rng": rng,
            "updates": state["updates"] + 1,
        }
        return updated, {
            "critic_loss": qloss,
            "editor_loss": eloss_value,
            "q_mean": qmean,
            "target_mean": target.mean(),
            "entropy": entropy,
            "temperature": jnp.exp(temp.params["log_temp"]),
            "critic_grad_norm": optax.global_norm(g),
            "editor_grad_norm": optax.global_norm(eg),
            "selected_edit_ratio": (selection["selected"] >= c.candidates).mean(),
        }

    def update(self, batch):
        updated, info = self._update(self.state, batch)
        metrics = {k: float(v) for k, v in info.items()}
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(str(metrics))
        self.state = updated
        return metrics

    def select(self, images, states, candidates, seed):
        a, info = self._select(
            self.state, images, states, candidates, jax.random.PRNGKey(seed)
        )
        return np.asarray(a), jax.tree.map(np.asarray, info)

    def save(self, path):
        path.write_bytes(serialization.to_bytes(jax.device_get(self.state)))

    def restore(self, path):
        self.state = jax.device_put(
            serialization.from_bytes(self.state, path.read_bytes())
        )
