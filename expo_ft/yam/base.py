"""FP32 YAM base policy, sequential candidates and optional expert-only SFT."""

from pathlib import Path
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, serialization

from expo_ft.conversion.yam_loader import YamJaxPolicy, as_observation


def expert_filter(path, value):
    return getattr(value, "type", type(value)) is nnx.Param and (
        str(path[0])
        in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
        or any(str(p).endswith("_1") for p in path)
    )


def dump_tree(tree):
    leaves, _ = jax.tree.flatten(tree)
    return serialization.msgpack_serialize(
        {"leaves": [np.asarray(x) for x in jax.device_get(leaves)]}
    )


def restore_tree(template, blob):
    expected, structure = jax.tree.flatten(template)
    leaves = serialization.msgpack_restore(blob)["leaves"]
    if len(leaves) != len(expected):
        raise ValueError("Checkpoint leaf count mismatch")
    for a, b in zip(expected, leaves):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Checkpoint shape/dtype mismatch")
    return jax.tree.unflatten(structure, jax.device_put(leaves))


class BasePolicy:
    def __init__(self, checkpoint, tokenizer, mode="frozen", lr=1e-5, patch=None):
        self.policy = YamJaxPolicy(checkpoint, tokenizer)
        self.processor = self.policy.processor
        self.graph, self.trainable, self.frozen = nnx.split(
            self.policy.model, expert_filter, ...
        )
        del self.policy.model  # avoid retaining an obsolete expert copy after updates
        self.mode = mode
        self.steps = 0
        self.tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
        self.opt = None
        if mode not in ("frozen", "expert"):
            raise ValueError("Unknown actor mode")
        if patch is not None and mode == "expert" and not Path(patch).exists():
            raise FileNotFoundError(patch)
        if patch is not None and Path(patch).exists():
            self.trainable = restore_tree(self.trainable, Path(patch).read_bytes())
            self.sync()
        self._train = jax.jit(self._train_step, donate_argnums=(0, 2))
        self.candidate_batch = int(os.environ.get("YAM_CANDIDATE_BATCH", "1"))
        if self.candidate_batch < 1:
            raise ValueError("YAM_CANDIDATE_BATCH must be positive")
        self._parallel_sample = jax.jit(jax.vmap(self.policy._sample, in_axes=(None, None, 0)))

    def sync(self):
        # Replace only expert leaves; frozen VLM arrays remain shared.
        self.policy._state = nnx.State.merge(self.frozen, self.trainable)

    def prepare(self, images, state, prompt):
        return self.processor.prepare_numpy(images, state, prompt)

    def candidates(self, data, count, rng):
        obs = as_observation(data)
        result = []
        for start in range(0, count, self.candidate_batch):
            size = min(self.candidate_batch, count - start)
            noise = np.stack([rng.standard_normal((1, 30, 32)).astype(np.float32)
                              for _ in range(size)])
            if size == 1:
                pred = np.asarray(self.policy._sample(self.policy._state, obs, jnp.asarray(noise[0])))
                result.append(pred[0, :, :14])
            else:
                pred = np.asarray(self._parallel_sample(self.policy._state, obs, jnp.asarray(noise)))
                result.extend(pred[:, 0, :, :14])
        result = np.stack(result)
        if result.shape != (count, 30, 14) or not np.isfinite(result).all():
            raise FloatingPointError("Invalid base candidate chunks")
        return result

    def _train_step(self, params, frozen, opt, obs, actions, key):
        def loss(p):
            model = nnx.merge(self.graph, p, frozen)
            with jax.default_matmul_precision("highest"):
                return model.compute_loss(key, obs, actions, train=False).mean()

        value, g = jax.value_and_grad(loss)(params)
        changes, opt = self.tx.update(g, opt, params)
        return optax.apply_updates(params, changes), opt, value, optax.global_norm(g)

    def train_success(self, data, actions, seed):
        if self.mode != "expert":
            return {"actor_updated": False}
        if self.opt is None:
            self.opt = self.tx.init(self.trainable)
        targets = np.pad(np.asarray(actions, np.float32), ((0, 0), (0, 18)))[None]
        new, opt, loss, norm = self._train(
            self.trainable,
            self.frozen,
            self.opt,
            as_observation(data),
            jnp.asarray(targets),
            jax.random.PRNGKey(seed),
        )
        metrics = {
            "actor_loss": float(loss),
            "actor_grad_norm": float(norm),
            "actor_updated": True,
        }
        if not np.isfinite([metrics["actor_loss"], metrics["actor_grad_norm"]]).all():
            raise FloatingPointError(str(metrics))
        self.trainable, self.opt = new, opt
        self.steps += 1
        self.sync()
        return metrics

    def save(self, folder):
        if self.mode == "expert":
            (folder / "actor.msgpack").write_bytes(dump_tree(self.trainable))
            if self.opt is not None:
                (folder / "actor_optimizer.msgpack").write_bytes(dump_tree(self.opt))

    def restore_optimizer(self, folder):
        path = folder / "actor_optimizer.msgpack"
        if self.mode == "expert" and path.exists():
            self.opt = restore_tree(self.tx.init(self.trainable), path.read_bytes())
