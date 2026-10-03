"""Round policy for the existing Karma HTTP transport."""

import time
import uuid
from pathlib import Path

import numpy as np

from expo_ft.conversion.yam_pi05 import JAX_CAMERAS

from .base import BasePolicy
from .learner import Settings, YamEXPO
from .rounds import atomic_json, read, sha, validate_version


class RoundPolicy:
    def __init__(self, root):
        self.root = Path(root)
        exp = read(self.root / "experiment.json")
        cur = read(self.root / "current.json")
        self.exp, self.cur = exp, cur
        self.cfg = Settings.from_dict(exp["settings"])
        version = Path(cur["checkpoint"]) if cur["checkpoint"] else None
        if version:
            manifest = validate_version(version)
            if manifest.get("bootstrap_only"):
                raise ValueError("Bootstrap version is untrained; complete its first training round before serving")
            if (
                manifest["version"] != cur["version"]
                or manifest["settings"] != exp["settings"]
            ):
                raise ValueError("Version/configuration mismatch")
        self.base = BasePolicy(
            exp["checkpoint"],
            exp["tokenizer"],
            self.cfg.actor_mode,
            self.cfg.actor_lr,
            version / "actor.msgpack" if version else None,
        )
        self.learner = None
        if version:
            self.learner = YamEXPO(self.cfg)
            self.learner.restore(version / "expo.msgpack")
        self.session = uuid.uuid4().hex
        self.log = self.root / "sessions" / self.session
        self.log.mkdir()
        self.count = 0
        self.record = False
        self.rng = np.random.default_rng()
        self.metadata = {
            "experiment_id": exp["experiment_id"],
            "policy_version": cur["version"],
            "session_id": self.session,
            "prompt": exp["prompt"],
            "base_actor_mode": self.cfg.actor_mode,
            "expo_enabled": version is not None,
            "policy_checkpoint": str(version) if version else None,
            "policy_weights_sha256": sha(version / "expo.msgpack") if version else None,
            "edit_scale": self.cfg.edit_scale,
            "mask_gripper_edits": self.cfg.mask_gripper_edits,
            "diagnostic_only": exp.get("diagnostic_only", False),
            "candidates": self.cfg.candidates,
            "edit_candidates": self.cfg.edits,
            "num_qs": 10,
            "num_min_qs": 2,
            "required_speed": 1.0,
            "required_chunk_size": 30,
            "required_fps": 30,
            "required_prefetch": False,
        }
        atomic_json(self.log / "session.json", self.metadata)

    def predict(self, images, state, prompt="fold the towel", noise=None):
        if prompt != self.exp["prompt"]:
            raise ValueError("Prompt must exactly match experiment prompt")
        seed = int(self.rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        data = self.base.prepare(images, state, prompt)
        candidates = self.base.candidates(
            data, self.cfg.candidates if self.learner else 1, rng
        )
        stacked = np.concatenate([data[k] for k in JAX_CAMERAS], axis=-1)
        details = {}
        if self.learner:
            chosen, details = self.learner.select(
                stacked, data["state"][:, :14], candidates.reshape(1, -1, 420), seed
            )
            if not np.isfinite(details["q_values"]).all():
                raise FloatingPointError("Nonfinite candidate Q values")
            normalized = chosen.reshape(30, 14)
        else:
            normalized = candidates[0]
        actions = self.base.processor.unnormalize_actions(normalized)
        if actions.shape != (30, 14) or not np.isfinite(actions).all():
            raise ValueError("Invalid EXPO actions")
        if self.record:
            self.count += 1
            np.savez_compressed(
                self.log / f"{self.count:06d}.npz",
                images=stacked,
                state_raw=state,
                state=data["state"],
                tokens=data["tokenized_prompt"],
                token_mask=data["tokenized_prompt_mask"],
                base_candidates=candidates,
                actions=actions,
                seed=seed,
                timestamp=time.time(),
                **details,
            )
        return actions
