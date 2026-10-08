#!/usr/bin/env python3
"""Offline real-checkpoint parity, latency, memory and EXPO update measurements."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch

from expo_ft.yam.rounds import atomic_json, read
from expo_ft.yam.torch.base import BasePolicy, observation
from expo_ft.yam.torch.learner import Agent, Settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--dataset",
        type=Path,
        help="Optional existing KARMA recording, read-only; no relabeling or admission",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-only", action="store_true")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "hardware_control": False,
    }
    images = {k: np.zeros((360, 640, 3), np.uint8) for k in ("top", "left", "right")}
    state = np.zeros(14, np.float32)
    state[[6, 13]] = 1
    if a.dataset:
        import pyarrow.parquet as pq

        from expo_ft.yam.replay import CAMERAS, dataset_to_wire, decode_selected

        info = read(a.dataset / "meta/info.json")
        ep = pq.read_table(a.dataset / "meta/episodes").to_pylist()[0]
        data = pq.read_table(
            a.dataset
            / info["data_path"].format(
                chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
            )
        ).to_pylist()
        state = dataset_to_wire(data[0]["observation.state"], "karma-recorded")
        for role, key in CAMERAS.items():
            meta = "videos/" + key if "videos/" + key + "/chunk_index" in ep else key
            path = a.dataset / info["video_path"].format(
                video_key=key,
                chunk_index=ep[meta + "/chunk_index"],
                file_index=ep[meta + "/file_index"],
            )
            frame = round(ep[meta + "/from_timestamp"] * 30)
            images[role] = decode_selected(path, [frame])[frame]
        report["input"] = "First saved frame of " + str(a.dataset)
    c = Settings()
    if not a.train_only:
        t = time.monotonic()
        base = BasePolicy(a.checkpoint)
        report["load_seconds"] = time.monotonic() - t
        # Use the supplied export's observation constructor as an independent
        # contract check, sharing the loaded model to avoid a second 12GB copy.
        spec = importlib.util.spec_from_file_location(
            "fold70_reference", a.checkpoint / "inference.py"
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        raw = reference.observation(
            images["top"], images["left"], images["right"], state, "fold the towel"
        )
        own = observation(images, state, "fold the towel")
        for key in own:
            if isinstance(own[key], torch.Tensor):
                torch.testing.assert_close(own[key], raw[key], rtol=0, atol=0)
            else:
                assert own[key] == raw[key]
        with torch.inference_mode():
            original = (
                base.post(
                    base.policy.predict_action_chunk(
                        base.pre(raw),
                        inference_action_mode="continuous",
                        generator=torch.Generator(device="cuda").manual_seed(70),
                    )
                )[0]
                .cpu()
                .numpy()
            )
        canonical = base.processor.normalize_actions(
            np.concatenate(
                [
                    original[:, :6],
                    original[:, 6:7].clip(0, 1),
                    original[:, 7:13],
                    original[:, 13:14].clip(0, 1),
                ],
                axis=1,
            )
        )
        records = []
        for count, cache, candidate_batch in (
            (1, False, 1),
            (8, False, 1),
            (8, True, 1),
            (8, True, 1),
            (8, True, 8),
            (8, True, 8),
        ):
            torch.cuda.reset_peak_memory_stats()
            t = time.monotonic()
            result = base.candidates(
                images,
                state,
                "fold the towel",
                count,
                70,
                cache_prefix=cache,
                candidate_batch=candidate_batch,
            )
            torch.cuda.synchronize()
            records.append(
                {
                    "candidates": count,
                    "cached_prefix": cache,
                    "candidate_batch": candidate_batch,
                    "seconds": time.monotonic() - t,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
            )
            if count == 1:
                np.testing.assert_array_equal(result[0], canonical)
            elif not cache:
                uncached = result.copy()
            elif candidate_batch == 1:
                difference = float(np.max(np.abs(result - uncached)))
                report["prefix_max_abs_difference"] = difference
                np.testing.assert_allclose(result, uncached, atol=1e-6, rtol=1e-6)
        report["base"] = records
        report["reference_normalized_parity"] = True
        agent = Agent(c, "cuda", training=False)
        t = time.monotonic()
        selected, _ = agent.select(
            np.stack(list(images.values()))[None],
            base.processor.normalize(state[None], "observation.state"),
            result.reshape(1, 8, 420),
        )
        torch.cuda.synchronize()
        report["selector_seconds"] = time.monotonic() - t
        report["base_plus_selector_peak_gib"] = (
            torch.cuda.max_memory_allocated() / 2**30
        )
        report["policy_payload_mb"] = (
            sum(v.numel() * v.element_size() for v in agent.state_dict().values()) / 1e6
        )
        del base, agent, selected
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        atomic_json(a.output, report)
    agent = Agent(c, "cuda")
    torch.cuda.reset_peak_memory_stats()
    b = {
        "rgb": np.stack(list(images.values()))[None],
        "next_rgb": np.stack(list(images.values()))[None],
        "states": np.zeros((1, 14), np.float32),
        "next_states": np.zeros((1, 14), np.float32),
        "actions": np.zeros((1, 420), np.float32),
        "next_candidates": np.zeros((1, 8, 420), np.float32),
        "rewards": np.ones(1, np.float32),
        "masks": np.zeros(1, np.float32),
        "steps": np.full(1, 30, np.float32),
    }
    t = time.monotonic()
    metrics = agent.critic_update([b] * 8, [b] * 8)
    torch.cuda.synchronize()
    report["critic_batch8_with_terminal_seconds"] = time.monotonic() - t
    t = time.monotonic()
    metrics.update(agent.editor_update([b] * 8))
    torch.cuda.synchronize()
    report["editor_batch8_seconds"] = time.monotonic() - t
    report["training_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["training_metrics"] = metrics
    atomic_json(a.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
