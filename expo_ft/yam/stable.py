"""Conservative offline learner: explicit accumulation and delayed editor updates."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .learner import YamEXPO, chunk_target


def edit_distribution(mean, logstd, key, scale, mask_grippers):
    """Return edits and entropy log-density in both coordinate systems.

    Exclude fixed grippers: their distribution is a delta, not a continuous
    Gaussian, and must not contribute to differential entropy.
    """
    active = jnp.ones(mean.shape[-1], bool)
    if mask_grippers:
        active = (jnp.arange(mean.shape[-1]) % 14 != 6) & (
            jnp.arange(mean.shape[-1]) % 14 != 13
        )
    noise = jax.random.normal(key, mean.shape)
    u = mean + jnp.exp(logstd) * noise
    lp = -0.5 * (noise**2 + 2 * logstd + jnp.log(2 * jnp.pi))
    lp -= 2 * (jnp.log(2.0) - u - jax.nn.softplus(-2 * u))
    lp = jnp.where(active, lp, 0).sum(-1)
    return (
        jnp.where(active, scale * jnp.tanh(u), 0),
        lp,
        lp - active.sum() * jnp.log(scale),
    )


def mean_tree(trees):
    return jax.tree.map(lambda *xs: sum(xs) / len(xs), *trees)


class StableEXPO(YamEXPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.cfg.scale_entropy_target:
            raise ValueError("Stable learner requires consistent entropy coordinates")
        self._cg = jax.jit(self._critic_gradient)
        self._eg = jax.jit(self._editor_gradient)
        self._ca = jax.jit(self._critic_apply)
        self._ea = jax.jit(self._editor_apply)

    def _critic_gradient(self, state, batch, key):
        sk, bk = jax.random.split(key)
        nxt, _ = self._selection(
            state,
            batch["next_images"],
            batch["next_states"],
            batch["next_candidates"],
            sk,
        )
        zn = self.encoder.apply(
            {"params": state["encoder"].params}, batch["next_images"]
        )
        qs = self.critic.apply(
            {"params": state["target_q"]}, zn, nxt, p=batch["next_states"]
        )
        pair = jax.random.choice(bk, 10, (2,), replace=False)
        target = jax.lax.stop_gradient(
            chunk_target(
                batch["rewards"],
                batch["masks"],
                batch["steps"],
                qs[pair].min(0),
                self.cfg.discount,
            )
        )

        def loss(params):
            z = self.encoder.apply({"params": params["encoder"]}, batch["images"])
            q = self.critic.apply(
                {"params": params["critic"]}, z, batch["actions"], p=batch["states"]
            )
            return ((q - target[None]) ** 2).mean(), (q.mean(), target.mean())

        (value, (qm, tm)), grads = jax.value_and_grad(loss, has_aux=True)(
            {"encoder": state["encoder"].params, "critic": state["critic"].params}
        )
        return grads, {"critic_loss": value, "q_mean": qm, "target_mean": tm}

    def _critic_apply(self, state, grads, rng):
        critic = state["critic"].apply_gradients(grads=grads["critic"])
        return {
            **state,
            "encoder": state["encoder"].apply_gradients(grads=grads["encoder"]),
            "critic": critic,
            "target_q": optax.incremental_update(
                critic.params, state["target_q"], self.cfg.tau
            ),
            "rng": rng,
            "updates": state["updates"] + 1,
        }

    def _editor_gradient(self, state, batch, key):
        c = self.cfg
        z = jax.lax.stop_gradient(
            self.encoder.apply({"params": state["encoder"].params}, batch["images"])
        )
        dim = c.horizon * (12 if c.mask_gripper_edits else 14)

        def loss(params):
            mean, std = self.editor.apply(
                {"params": params}, z, actions=batch["actions"], p=batch["states"]
            )
            edits, lp, scaled_lp = edit_distribution(
                mean, std, key, c.edit_scale, c.mask_gripper_edits
            )
            q = self.critic.apply(
                {"params": state["critic"].params},
                z,
                batch["actions"] + edits,
                p=batch["states"],
            ).mean(0)
            # Unscaled lp differs by a parameter-independent constant, so editor
            # gradients match the scaled-density objective at fixed temperature.
            value = (jnp.exp(state["temperature"].params["log_temp"]) * lp - q).mean()
            return value, (-lp.mean(), -scaled_lp.mean(), jnp.abs(edits).mean())

        (value, (entropy, scaled_entropy, delta)), grads = jax.value_and_grad(
            loss, has_aux=True
        )(state["editor"].params)
        tg = jax.grad(
            lambda t: jnp.exp(t["log_temp"]) * jax.lax.stop_gradient(entropy + dim / 2)
        )(state["temperature"].params)
        return {"editor": grads, "temperature": tg}, {
            "editor_loss": value,
            "entropy_unscaled": entropy,
            "entropy_scaled": scaled_entropy,
            "edit_abs_mean": delta,
            "target_entropy_unscaled": jnp.asarray(-dim / 2),
        }

    def _editor_apply(self, state, grads, rng):
        return {
            **state,
            "editor": state["editor"].apply_gradients(grads=grads["editor"]),
            "temperature": state["temperature"].apply_gradients(
                grads=grads["temperature"]
            ),
            "rng": rng,
        }

    def _accumulate(self, batches, gradient_fn, seed):
        grads = None
        metrics = []
        for index, batch in enumerate(batches):
            g, m = gradient_fn(self.state, batch, jax.random.fold_in(seed, index))
            # No parameter updates until the entire effective batch is reduced.
            grads = g if grads is None else jax.tree.map(jnp.add, grads, g)
            metrics.append(m)
        if not metrics:
            raise ValueError("Empty effective batch")
        return jax.tree.map(lambda x: x / len(metrics), grads), mean_tree(metrics)

    @staticmethod
    def _finite(metrics):
        out = {k: float(v) for k, v in metrics.items()}
        if not all(np.isfinite(v) for v in out.values()):
            raise FloatingPointError(str(out))
        return out

    def critic_update(self, batches, terminal_batches, terminal_weight=0.25):
        rng, main_key, terminal_key = jax.random.split(self.state["rng"], 3)
        grads, metrics = self._accumulate(batches, self._cg, main_key)
        if terminal_batches:
            if any(np.any(b["masks"] != 0) for b in terminal_batches):
                raise ValueError("Terminal auxiliary loss must not bootstrap")
            tg, tm = self._accumulate(terminal_batches, self._cg, terminal_key)
            grads = jax.tree.map(lambda g, t: g + terminal_weight * t, grads, tg)
            metrics["terminal_loss"] = tm["critic_loss"]
            metrics["terminal_target"] = tm["target_mean"]
        metrics["critic_grad_norm"] = optax.global_norm(grads)
        out = self._finite(metrics)
        self.state = self._ca(self.state, grads, rng)
        return out

    def editor_update(self, batches):
        rng, key = jax.random.split(self.state["rng"])
        grads, metrics = self._accumulate(batches, self._eg, key)
        metrics["editor_grad_norm"] = optax.global_norm(grads["editor"])
        out = self._finite(metrics)
        self.state = self._ea(self.state, grads, rng)
        out["temperature"] = float(
            jnp.exp(self.state["temperature"].params["log_temp"])
        )
        return out

    def update(self, batch):
        raise RuntimeError("Use separate accumulated critic_update/editor_update")


def crop_batch(batch, rng):
    """Independent 95% crops per view/current-next; retain float image range."""
    import torch
    import torch.nn.functional as F

    out = dict(batch)
    for name in ("images", "next_images"):
        image = batch[name]
        _, h, w, _ = image.shape
        ch, cw = round(h * 0.95), round(w * 0.95)
        views = []
        for view in range(3):
            y, x = int(rng.integers(h - ch + 1)), int(rng.integers(w - cw + 1))
            a = np.array(image[:, y : y + ch, x : x + cw, view * 3 : view * 3 + 3])
            t = torch.from_numpy(a).permute(0, 3, 1, 2)
            views.append(
                F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
                .permute(0, 2, 3, 1)
                .numpy()
            )
        out[name] = np.concatenate(views, -1)
    return out
