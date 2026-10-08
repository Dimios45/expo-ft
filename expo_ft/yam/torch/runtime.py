"""Native Torch round policy and atomic, optimizer-free candidate installation."""

import json
import os
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import torch

from expo_ft.yam.rounds import atomic_json, read, sha

from .base import ROLES, BasePolicy, checkpoint_identity
from .learner import Agent, Settings


def request_json(url, token, data=None):
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


class CandidateStore:
    def __init__(self, root, url, token):
        self.root, self.url, self.token = Path(root), url.rstrip("/"), token

    def install(self, exp, current_version):
        candidate = request_json(self.url + "/candidate", self.token)
        if (
            candidate["experiment_id"] != exp["experiment_id"]
            or candidate["base_sha256"] != exp["source_sha"]
        ):
            raise ValueError("Candidate experiment/base mismatch")
        version = candidate["version"]
        if type(version) != int or version < current_version:
            raise ValueError("Candidate version rollback")
        if version == current_version:
            return None
        manifest = candidate["manifest"]
        if (
            manifest["version"] != version
            or manifest["settings"] != exp["settings"]
            or manifest["backend"] != "pytorch"
        ):
            raise ValueError("Candidate configuration mismatch")
        if (
            manifest["experiment_id"] != exp["experiment_id"]
            or manifest["base_sha256"] != exp["source_sha"]
        ):
            raise ValueError("Manifest identity mismatch")
        expected = manifest["files"]["policy.safetensors"]
        final = self.root / "versions" / f"{version:04d}"
        if final.exists():
            if (
                sha(final / "policy.safetensors") != expected
                or read(final / "manifest.json") != manifest
            ):
                raise ValueError("Conflicting immutable checkpoint")
            return final, version
        stage = final.with_name(".incoming-" + uuid.uuid4().hex)
        stage.mkdir()
        req = urllib.request.Request(
            self.url + f"/weights/{version}",
            headers={"Authorization": "Bearer " + self.token},
        )
        with (
            urllib.request.urlopen(req, timeout=180) as response,
            (stage / "policy.safetensors").open("xb") as dst,
        ):
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 2 * 1024**3:
                    raise ValueError("Serving payload exceeds 2 GiB")
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if sha(stage / "policy.safetensors") != expected:
            raise ValueError("Partial/corrupt weight transfer")
        atomic_json(stage / "manifest.json", manifest)
        stage.rename(final)
        return final, version


class RoundPolicy:
    def __init__(self, root, checkpoint, device="cuda", base_model=None, store=None):
        self.root = Path(root)
        self.exp = read(self.root / "experiment.json")
        if self.exp.get("backend") != "pytorch":
            raise ValueError("Wrong backend")
        self.cur = read(self.root / "current.json")
        self.cfg = Settings(**self.exp["settings"])
        if checkpoint_identity(checkpoint) != self.exp["checkpoint_provenance"]:
            raise ValueError("Inference checkpoint/processor differs from learner")
        self.base = BasePolicy(checkpoint, device, base_model)
        self.store = store
        self.agent = None
        self.rng = np.random.default_rng()
        if self.cur["version"]:
            if (
                read(Path(self.cur["checkpoint"]) / "manifest.json")["version"]
                != self.cur["version"]
            ):
                raise ValueError("Registry/version mismatch")
            self.agent = self.load_agent(Path(self.cur["checkpoint"]))
        self.metadata = self.new_session(self.cur)

    def load_agent(self, folder):
        manifest = read(folder / "manifest.json")
        if (
            manifest.get("backend") != "pytorch"
            or manifest["settings"] != self.exp["settings"]
            or manifest.get("base_sha256") != self.exp["source_sha"]
            or manifest.get("experiment_id") != self.exp["experiment_id"]
            or sha(folder / "policy.safetensors")
            != manifest["files"]["policy.safetensors"]
        ):
            raise ValueError("Serving checkpoint integrity/identity failure")
        agent = Agent(self.cfg, str(self.base.device), training=False)
        agent.restore(folder)
        rgb = np.zeros((1, 3, 32, 32, 3), np.uint8)
        base = np.zeros((1, self.cfg.candidates, 420), np.float32)
        result, _ = agent.select(rgb, np.zeros((1, 14), np.float32), base)
        if (
            not torch.isfinite(result).all()
            or result.abs().max() > self.cfg.edit_scale + 1e-5
        ):
            raise ValueError("Candidate warmup failed")
        if torch.any(result.reshape(30, 14)[:, [6, 13]] != 0):
            raise ValueError("Candidate edited grippers")
        return agent

    def new_session(self, cur):
        session = uuid.uuid4().hex
        path = self.root / "sessions" / session
        path.mkdir()
        metadata = {
            "experiment_id": self.exp["experiment_id"],
            "policy_version": cur["version"],
            "session_id": session,
            "prompt": self.exp["prompt"],
            "backend": "pytorch",
            "base_actor_mode": "frozen",
            "base_sha256": self.exp["source_sha"],
            "expo_enabled": bool(cur["version"]),
            "policy_checkpoint": cur["checkpoint"],
            "policy_weights_sha256": sha(Path(cur["checkpoint"]) / "policy.safetensors")
            if cur["version"]
            else None,
            "edit_scale": self.cfg.edit_scale,
            "mask_gripper_edits": True,
            "candidates": self.cfg.candidates,
            "edit_candidates": self.cfg.edits,
            "num_qs": 10,
            "num_min_qs": 2,
            "required_speed": 1.0,
            "required_chunk_size": 30,
            "required_fps": 30,
            "required_prefetch": False,
        }
        atomic_json(path / "session.json", metadata)
        return metadata

    def reload_candidate(self):
        if self.store is None:
            return self.metadata
        installed = self.store.install(self.exp, self.cur["version"])
        if installed is None:
            self.base.policy.reset()
            return self.metadata
        folder, version = installed
        candidate = self.load_agent(folder)  # Does not mutate live parameters.
        cur = {"version": version, "checkpoint": str(folder), "episodes": []}
        metadata = self.new_session(cur)
        atomic_json(self.root / "current.json", cur)
        self.agent, self.cur, self.metadata = candidate, cur, metadata
        self.base.policy.reset()
        return self.metadata

    def predict(self, images, state, prompt):
        if prompt != self.exp["prompt"]:
            raise ValueError("Prompt must exactly match experiment prompt")
        count = self.cfg.candidates if self.agent else 1
        base = self.base.candidates(
            images, state, prompt, count, int(self.rng.integers(2**31)), candidate_batch=count
        )
        if self.agent:
            rgb = np.stack([images[k] for k in ROLES])[None]
            states = self.base.processor.normalize(
                np.asarray(state)[None], "observation.state"
            )
            selected, _ = self.agent.select(rgb, states, base.reshape(1, count, 420))
            normalized = selected.cpu().numpy().reshape(30, 14)
        else:
            normalized = base[0]
        return self.base.physical_actions(normalized)
