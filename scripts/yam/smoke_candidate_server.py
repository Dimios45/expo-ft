#!/usr/bin/env python3
"""Replay saved data against a localhost candidate server. Never contacts hardware."""

import argparse
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json_numpy
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from expo_ft.yam.replay import CAMERAS, dataset_to_wire, decode_selected
from expo_ft.yam.rounds import atomic_json, read, sha, validate_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/usr/local/models/sra-expo-ft/towel-expo-stable-v2"),
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "round-0000")
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    root, dataset = args.root.resolve(), args.dataset.resolve()
    candidate = args.candidate.resolve() if args.candidate else root / "candidate"
    manifest = validate_version(candidate)
    assert manifest["settings"] == read(root / "experiment.json")["settings"]
    testroot = root / ("http-test-" + uuid.uuid4().hex[:8])
    testroot.mkdir()
    (testroot / "sessions").mkdir()
    policy_version = manifest.get("version", 1)
    version = testroot / "versions" / f"{policy_version:04d}"
    version.mkdir(parents=True)
    # Isolated diagnostic view: no promotion or edit of either experiment.
    for name in manifest["files"]:
        (version / name).symlink_to(candidate / name)
    atomic_json(version / "manifest.json", {**manifest, "version": policy_version})
    exp = read(root / "experiment.json")
    exp.update(
        diagnostic_only=True,
        experiment_id=uuid.uuid4().hex,
        candidate_source=str(candidate),
    )
    atomic_json(testroot / "experiment.json", exp)
    atomic_json(
        testroot / "current.json",
        {"version": policy_version, "episodes": [], "checkpoint": str(version)},
    )
    # Port allocated on localhost only; no NUC address is used anywhere.
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log = (testroot / "server.log").open("w")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_COMPILATION_CACHE_DIR": "/usr/local/models/sra-expo-ft/jax-cache",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/yam/round.py"),
            "--root",
            str(testroot),
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    report = {
        "passed": False,
        "url": url,
        "testroot": str(testroot),
        "candidate": str(candidate),
        "hardware_accessed": False,
        "requests": [],
    }
    try:
        deadline = time.monotonic() + 180
        while True:
            if process.poll() is not None:
                raise RuntimeError("Server exited; see " + str(testroot / "server.log"))
            try:
                with urllib.request.urlopen(url + "/healthz", timeout=2) as r:
                    health = json.load(r)
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() > deadline:
                    raise TimeoutError("Server warmup")
                time.sleep(0.5)
        assert health["policy_weights_sha256"] == sha(candidate / "expo.msgpack")
        assert health["expo_enabled"] and health["base_actor_mode"] == "frozen"
        assert health["edit_scale"] == 0.05 and health["mask_gripper_edits"]
        assert health["diagnostic_only"]
        report["health"] = health
        info = read(dataset / "meta/info.json")
        meta = pq.read_table(dataset / "meta/episodes").to_pylist()[0]
        rows = pq.read_table(dataset / "data").to_pylist()
        indices = np.unique(np.linspace(0, len(rows) - 1, 12, dtype=int))
        views = {}
        for role, key in CAMERAS.items():
            prefix = "videos/" + key
            path = dataset / info["video_path"].format(
                video_key=key,
                chunk_index=meta[prefix + "/chunk_index"],
                file_index=meta[prefix + "/file_index"],
            )
            frames = np.rint(
                (
                    meta[prefix + "/from_timestamp"]
                    + np.array([rows[i]["timestamp"] for i in indices])
                )
                * 30
            ).astype(int)
            decoded = decode_selected(path, frames)
            views[role] = [decoded[int(i)] for i in frames]
        physical = []
        for n, index in enumerate(indices):
            payload = {
                "state": dataset_to_wire(
                    rows[index]["observation.state"], "karma-recorded"
                ),
                "instruction": "fold the towel",
                "num_steps": 10,
            }
            for role in CAMERAS:
                frame = views[role][n]
                if n % 2:
                    stream = io.BytesIO()
                    Image.fromarray(frame).save(stream, format="JPEG", quality=95)
                    frame = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                payload[role + "_cam"] = frame
            request = urllib.request.Request(
                url + "/act",
                data=json_numpy.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=60) as r:
                assert r.status == 200
                result = json_numpy.loads(r.read())
            actions = np.asarray(result["actions"])
            assert actions.shape == (30, 14) and np.isfinite(actions).all()
            physical.append(actions)
            report["requests"].append(
                {
                    "frame": int(index),
                    "encoding": "jpeg" if n % 2 else "rgb",
                    "server_ms": result["dt_ms"],
                    "roundtrip_ms": 1000 * (time.perf_counter() - start),
                }
            )
            print(json.dumps(report["requests"][-1]), flush=True)
        invalid = []
        for key, value in [
            ("state", np.zeros(13, np.float32)),
            ("num_steps", 9),
            ("instruction", "different task"),
        ]:
            request = urllib.request.Request(
                url + "/act",
                data=json_numpy.dumps({**payload, key: value}).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(request, timeout=10)
                raise AssertionError("Invalid input accepted: " + key)
            except urllib.error.HTTPError as e:
                assert e.code == 400
                invalid.append(key)
        traces = sorted((testroot / "sessions" / health["session_id"]).glob("*.npz"))
        assert len(traces) == len(indices)
        residuals = []
        for i, f in enumerate(traces):
            with np.load(f) as t:
                assert t["q_values"].shape == (10, 1, 16)
                assert len(set(t["pair"].tolist())) == 2
                residual = (
                    t["candidates"][0, 8:].reshape(8, 30, 14) - t["base_candidates"]
                )
                assert np.abs(residual).max() <= 0.050001
                np.testing.assert_array_equal(residual[:, :, [6, 13]], 0)
                np.testing.assert_array_equal(t["actions"], physical[i])
                residuals.append(np.abs(residual).ravel())
        physical = np.stack(physical)
        grip = physical[:, :, [6, 13]]
        report.update(
            passed=True,
            invalid_inputs_rejected=invalid,
            residual_abs_p50_p95_max=np.quantile(
                np.concatenate(residuals), [0.5, 0.95, 1]
            ).tolist(),
            gripper_output_minmax=[float(grip.min()), float(grip.max())],
            gripper_outputs_outside_0_1=int(((grip < 0) | (grip > 1)).sum()),
            server_ms_p50_p95=np.quantile(
                [r["server_ms"] for r in report["requests"]], [0.5, 0.95]
            ).tolist(),
            joint_hardware_limits_verified=False,
            note="Protocol and sampled residual checks only; no hardware success claim.",
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
        report["server_stopped"] = process.poll() is not None
        atomic_json(testroot / "report.json", report)
        print("REPORT", testroot / "report.json", flush=True)


if __name__ == "__main__":
    main()
