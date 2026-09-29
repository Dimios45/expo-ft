#!/usr/bin/env python3
"""Convert weights on CPU. Run --layout-only before installing the JAX stack."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "expo_ft/agents/vla/openpi/src")]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from expo_ft.conversion.yam_pi05 import OPENPI_REVISION, Mapper, sha256, validate_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--layout-only",
        action="store_true",
        help="Verify/store layouts without importing JAX",
    )
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    config = json.loads((source / "config.json").read_text())
    validate_config(config)
    import numpy as np
    from safetensors import safe_open

    out.mkdir(parents=True)
    # INCOMPLETE remains on failure: partial conversion must never be served.
    (out / "INCOMPLETE").write_text("Conversion in progress\n")
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as f:
        mapper = Mapper(f)
        arrays = mapper.convert()
        extra = set(f.keys()) - mapper.used
        allowed = {
            "model.paligemma_with_expert.gemma_expert.lm_head.weight",
            "model.paligemma_with_expert.paligemma.lm_head.weight",
        }
        if extra - allowed:
            raise ValueError(f"Unmapped source tensors: {sorted(extra - allowed)}")
        # Language-output heads are not called by PI05 action generation.
        # Archive them verbatim, so conversion never silently loses source data.
        from safetensors.torch import save_file

        if extra:
            save_file(
                {k: f.get_tensor(k).contiguous() for k in sorted(extra)},
                str(out / "unused_language_heads.safetensors"),
            )
        layout = out / "layout"
        layout.mkdir()
        index = {}
        for i, (key, value) in enumerate(sorted(arrays.items())):
            name = f"{i:03d}.npy"
            np.save(layout / name, value, allow_pickle=False)
            index[key] = {
                "file": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        (out / "layout_index.json").write_text(json.dumps(index, indent=2))
    for file in source.glob("*.json"):
        shutil.copy2(file, out / file.name)
    for file in source.glob("policy_*processor*.safetensors"):
        shutil.copy2(file, out / file.name)
    (out / "openpi_config.json").write_text(
        json.dumps(
            {
                "pi05": True,
                "dtype": "float32",
                "action_dim": 32,
                "action_horizon": 30,
                "max_token_len": 200,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
            },
            indent=2,
        )
    )
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT / "expo_ft/agents/vla/openpi"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != OPENPI_REVISION:
        raise ValueError(f"Unexpected OpenPI revision: {revision}")
    manifest = {
        "source": str(source),
        "source_sha256": sha256(source / "model.safetensors"),
        "openpi_revision": revision,
        "consumed_tensors": len(mapper.used),
        "source_assets_sha256": {
            f.name: sha256(f) for f in source.glob("policy_*") if f.is_file()
        },
        "archived_non_action_tensors": sorted(extra),
        "destination_arrays": len(arrays),
        "layout_roundtrips": mapper.audit,
        "status": "layout_verified",
        "storage_dtype": "float32 (exact expansion of BF16)",
        "inference_parity": "not_run",
    }
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2))
    del arrays, mapper
    if not args.layout_only:
        export(out)
        manifest = json.loads((out / "conversion_manifest.json").read_text())
    (out / "INCOMPLETE").unlink()
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "layout_roundtrips"}, indent=2
        )
    )


def export(out):
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from flax import nnx, traverse_util

    from expo_ft.conversion.yam_pi05 import model_config

    index = json.loads((out / "layout_index.json").read_text())
    flat = {
        tuple(k.split("/")): np.load(out / "layout" / v["file"], mmap_mode="r")
        for k, v in index.items()
    }
    abstract = nnx.eval_shape(model_config().create, jax.random.key(0))
    expected = traverse_util.flatten_dict(nnx.state(abstract).to_pure_dict())
    if set(flat) != set(expected):
        raise ValueError(
            f"Target mismatch: missing={set(expected) - set(flat)}, extra={set(flat) - set(expected)}"
        )
    for k, value in flat.items():
        if value.shape != expected[k].shape:
            raise ValueError(
                f"Shape mismatch {k}: {value.shape} != {expected[k].shape}"
            )
    with ocp.PyTreeCheckpointer() as saver:
        saver.save(out / "params", {"params": traverse_util.unflatten_dict(flat)})
    manifest = json.loads((out / "conversion_manifest.json").read_text())
    manifest["status"] = "openpi_shapes_verified_and_exported"
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved OpenPI checkpoint: {out / 'params'}")


if __name__ == "__main__":
    main()
