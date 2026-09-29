from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from test_yam_expo import batch, tiny

from expo_ft.yam.stable import StableEXPO, edit_distribution


def settings():
    return replace(
        tiny(),
        edit_scale=0.05,
        init_temperature=0.01,
        editor_lr=3e-5,
        temperature_lr=1e-5,
        gradient_clip=1.0,
        mask_gripper_edits=True,
        initial_editor_logstd=-3.0,
        scale_entropy_target=True,
    )


def test_entropy_units_and_mask():
    mean = jnp.zeros((1, 420))
    for scale in (0.2, 0.05):
        edit, lp, scaled_lp = edit_distribution(
            mean, mean - 3, jax.random.PRNGKey(1), scale, True
        )
        np.testing.assert_array_equal(np.asarray(edit).reshape(30, 14)[:, [6, 13]], 0)
        assert np.abs(edit).max() <= scale
        # Scaled entropy minus scaled target equals unscaled entropy minus target.
        np.testing.assert_allclose(
            -scaled_lp - (-180 + 360 * np.log(scale)), -lp + 180, rtol=1e-5
        )


def test_accumulation_is_one_adam_step_and_delayed_editor(tmp_path):
    cfg = settings()
    a = StableEXPO(cfg, image_size=8)
    b = StableEXPO(cfg, image_size=8)
    mb = batch()
    old_editor = jax.device_get(a.state["editor"].params)
    a.critic_update([mb, mb], [], terminal_weight=0)
    big = {k: np.concatenate([v, v]) for k, v in mb.items()}
    b.critic_update([big], [], terminal_weight=0)
    assert int(a.state["critic"].step) == 1
    assert int(a.state["editor"].step) == 0
    for x, y in zip(
        jax.tree.leaves(a.state["critic"].params),
        jax.tree.leaves(b.state["critic"].params),
    ):
        np.testing.assert_allclose(x, y, atol=2e-7, rtol=1e-5)
    for x, y in zip(
        jax.tree.leaves(old_editor), jax.tree.leaves(a.state["editor"].params)
    ):
        np.testing.assert_array_equal(x, y)
    m = a.editor_update([mb, mb])
    assert int(a.state["editor"].step) == 1 and np.isfinite(m["editor_loss"])
    f = tmp_path / "state.msgpack"
    a.save(f)
    b.restore(f)
    x = a.critic_update([mb], [mb])
    y = b.critic_update([mb], [mb])
    assert x == y


def test_terminal_auxiliary_updates_without_bootstrap():
    agent = StableEXPO(settings(), image_size=8)
    mb = batch()
    metrics = agent.critic_update([mb], [mb])
    assert metrics["terminal_target"] == 1.0
    _, info = agent.select(mb["images"], mb["states"], mb["next_candidates"], 3)
    residual = info["candidates"][0, 2:].reshape(2, 30, 14)
    np.testing.assert_array_equal(residual[:, :, [6, 13]], 0)
    assert np.abs(residual).max() <= 0.05


def test_success_and_failure_terminal_auxiliary():
    agent = StableEXPO(settings(), image_size=8)
    success = batch()
    failure = {**batch(), "rewards": np.zeros(1, np.float32)}
    metrics = agent.critic_update([failure], [success, failure])
    # Both absorbing labels contribute equally to the separate terminal loss.
    assert metrics["terminal_target"] == 0.5
    assert metrics["target_mean"] == 0.0
    assert int(agent.state["editor"].step) == 0
