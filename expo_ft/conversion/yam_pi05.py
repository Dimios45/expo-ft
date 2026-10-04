"""Lossless LeRobot PI05 -> OpenPI parameter layout conversion.

No models, GPU runtimes, or robot interfaces are imported at module import time.
The layout follows OpenPI's Gemma/SigLIP einsums, including the
NHD attention output contraction shared by both Gemma experts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

OPENPI_REVISION = "2abe46282bfdf9f1bc0240f3f9960ec175d1b4a8"
CAMERAS = ("top", "left", "right")
JAX_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_config(c):
    expected = {
        "type": "pi05",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "chunk_size": 30,
        "max_action_dim": 32,
        "max_state_dim": 32,
        "tokenizer_max_length": 200,
        "use_relative_actions": False,
    }
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError(f"Unsupported {key}: {c.get(key)!r}; expected {value!r}")
    for key in ("use_visual_memory", "use_proprioceptive_memory", "use_peft"):
        if c.get(key, False):
            raise ValueError(f"Unsupported {key}")
    if c.get("rtc_config") or c.get("rtc_training_max_delay", 0):
        raise ValueError("This converter targets the unmodified, non-RTC checkpoint")
    if c["output_features"]["action"]["shape"] != [14]:
        raise ValueError("Expected 14 physical actions")
    images = [k for k, v in c["input_features"].items() if v["type"] == "VISUAL"]
    if images != [f"observation.images.{k}" for k in CAMERAS]:
        raise ValueError(f"Unexpected camera order: {images}")


class Mapper:
    """Read one tensor at a time; verify every layout operation by its inverse."""

    def __init__(self, source):
        self.source = source
        self.used = set()
        self.params = {}
        self.audit = []

    def get(self, key, transform=lambda x: x, inverse=lambda x: x):
        import torch

        key = "model." + key
        if key in self.used:
            raise ValueError(f"Tensor consumed twice: {key}")
        t = self.source.get_tensor(key)
        x = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite source tensor: {key}")
        y = transform(x)
        if not np.array_equal(inverse(y), x):
            raise AssertionError(f"Lossy layout mapping: {key}")
        self.used.add(key)
        self.audit.append(
            {
                "source": key,
                "shape": list(x.shape),
                "dtype": str(t.dtype),
                "roundtrip": True,
            }
        )
        return y

    def put(self, path, arrays):
        if path in self.params:
            raise ValueError(f"Duplicate destination: {path}")
        self.params[path] = np.stack(arrays) if isinstance(arrays, list) else arrays

    def direct(self, dest, src, transpose=False):
        self.put(
            dest,
            self.get(src, lambda x: x.T, lambda x: x.T) if transpose else self.get(src),
        )

    def convert(self):
        p = "paligemma_with_expert."
        v = p + "paligemma.model.vision_tower.vision_model."
        root = "PaliGemma/img/"
        self.put(
            root + "embedding/kernel",
            self.get(
                v + "embeddings.patch_embedding.weight",
                lambda x: x.transpose(2, 3, 1, 0),
                lambda x: x.transpose(3, 2, 0, 1),
            ),
        )
        self.direct(root + "embedding/bias", v + "embeddings.patch_embedding.bias")
        self.put(
            root + "pos_embedding",
            self.get(
                v + "embeddings.position_embedding.weight",
                lambda x: x[None],
                lambda x: x[0],
            ),
        )
        for dest, src, trans in [
            ("LayerNorm_0/scale", "layer_norm1.weight", False),
            ("LayerNorm_0/bias", "layer_norm1.bias", False),
            ("LayerNorm_1/scale", "layer_norm2.weight", False),
            ("LayerNorm_1/bias", "layer_norm2.bias", False),
            ("MlpBlock_0/Dense_0/kernel", "mlp.fc1.weight", True),
            ("MlpBlock_0/Dense_0/bias", "mlp.fc1.bias", False),
            ("MlpBlock_0/Dense_1/kernel", "mlp.fc2.weight", True),
            ("MlpBlock_0/Dense_1/bias", "mlp.fc2.bias", False),
        ]:
            self.put(
                root + "Transformer/encoderblock/" + dest,
                [
                    self.get(
                        v + f"encoder.layers.{i}." + src,
                        lambda x, trans=trans: x.T if trans else x,
                        lambda x, trans=trans: x.T if trans else x,
                    )
                    for i in range(27)
                ],
            )
        for name, proj in [
            ("query", "q_proj"),
            ("key", "k_proj"),
            ("value", "v_proj"),
            ("out", "out_proj"),
        ]:
            shape = (16, 72, 1152) if name == "out" else (1152, 16, 72)
            for param in ("weight", "bias"):

                def forward(x, shape=shape, param=param, name=name):
                    return (
                        x.T.reshape(shape)
                        if param == "weight"
                        else (x if name == "out" else x.reshape(16, 72))
                    )

                def backward(x, param=param):
                    return (
                        x.reshape(1152, 1152).T
                        if param == "weight"
                        else x.reshape(1152)
                    )

                self.put(
                    root
                    + f"Transformer/encoderblock/MultiHeadDotProductAttention_0/{name}/"
                    + ("kernel" if param == "weight" else "bias"),
                    [
                        self.get(
                            v + f"encoder.layers.{i}.self_attn.{proj}.{param}",
                            forward,
                            backward,
                        )
                        for i in range(27)
                    ],
                )
        for dest, src in [("scale", "weight"), ("bias", "bias")]:
            self.direct(
                root + "Transformer/encoder_norm/" + dest, v + "post_layernorm." + src
            )
        for dest, src in [("kernel", "weight"), ("bias", "bias")]:
            self.direct(
                root + "head/" + dest,
                p + "paligemma.model.multi_modal_projector.linear." + src,
                src == "weight",
            )
        self.direct(
            "PaliGemma/llm/embedder/input_embedding",
            p + "paligemma.model.language_model.embed_tokens.weight",
        )
        for expert, width in [(False, 2048), (True, 1024)]:
            src = p + (
                "gemma_expert.model." if expert else "paligemma.model.language_model."
            )
            suffix = "_1" if expert else ""
            base = "PaliGemma/llm/layers/"
            self.put(
                base + f"attn/q_einsum{suffix}/w",
                [
                    self.get(
                        src + f"layers.{i}.self_attn.q_proj.weight",
                        lambda x, width=width: x.reshape(8, 256, width).transpose(
                            0, 2, 1
                        ),
                        lambda x, width=width: x.transpose(0, 2, 1).reshape(
                            2048, width
                        ),
                    )
                    for i in range(18)
                ],
            )
            self.put(
                base + f"attn/kv_einsum{suffix}/w",
                [
                    np.stack(
                        [
                            self.get(
                                src + f"layers.{i}.self_attn.{kv}_proj.weight",
                                lambda x: x.T[None],
                                lambda x: x[0].T,
                            )
                            for kv in ("k", "v")
                        ]
                    )
                    for i in range(18)
                ],
            )
            self.put(
                base + f"attn/attn_vec_einsum{suffix}/w",
                [
                    self.get(
                        src + f"layers.{i}.self_attn.o_proj.weight",
                        lambda x, width=width: x.T.reshape(8, 256, width),
                        lambda x, width=width: x.reshape(2048, width).T,
                    )
                    for i in range(18)
                ],
            )
            self.put(
                base + f"mlp{suffix}/gating_einsum",
                [
                    np.stack(
                        [
                            self.get(
                                src + f"layers.{i}.mlp.{part}_proj.weight",
                                lambda x: x.T,
                                lambda x: x.T,
                            )
                            for part in ("gate", "up")
                        ]
                    )
                    for i in range(18)
                ],
            )
            self.put(
                base + f"mlp{suffix}/linear",
                [
                    self.get(
                        src + f"layers.{i}.mlp.down_proj.weight",
                        lambda x: x.T,
                        lambda x: x.T,
                    )
                    for i in range(18)
                ],
            )
            for norm, norm_src in [
                ("pre_attention_norm", "input_layernorm"),
                ("pre_ffw_norm", "post_attention_layernorm"),
            ]:
                if expert:
                    for dst, part in [("kernel", "weight"), ("bias", "bias")]:
                        self.put(
                            base + f"{norm}{suffix}/Dense_0/{dst}",
                            [
                                self.get(
                                    src + f"layers.{i}.{norm_src}.dense.{part}",
                                    lambda x, part=part: x.T if part == "weight" else x,
                                    lambda x, part=part: x.T if part == "weight" else x,
                                )
                                for i in range(18)
                            ],
                        )
                else:
                    self.put(
                        base + norm + "/scale",
                        [
                            self.get(src + f"layers.{i}.{norm_src}.weight")
                            for i in range(18)
                        ],
                    )
            if expert:
                self.direct(
                    "PaliGemma/llm/final_norm_1/Dense_0/kernel",
                    src + "norm.dense.weight",
                    True,
                )
                self.direct(
                    "PaliGemma/llm/final_norm_1/Dense_0/bias", src + "norm.dense.bias"
                )
            else:
                self.direct("PaliGemma/llm/final_norm/scale", src + "norm.weight")
        for proj in (
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ):
            self.direct(proj + "/kernel", proj + ".weight", True)
            self.direct(proj + "/bias", proj + ".bias")
        return self.params


def model_config(dtype="float32"):
    from openpi.models.pi0_config import Pi0Config

    return Pi0Config(
        pi05=True,
        dtype=dtype,
        action_horizon=30,
        action_dim=32,
        max_token_len=200,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )


def load_model(checkpoint, dtype="float32"):
    import jax.numpy as jnp
    from openpi.models.model import restore_params

    return model_config(dtype).load(
        restore_params(Path(checkpoint) / "params", dtype=jnp.dtype(dtype)),
        remove_extra_params=False,
    )
