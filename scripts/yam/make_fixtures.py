#!/usr/bin/env python3
"""Read recorded LeRobot v3 files into offline conversion fixtures. No devices."""

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(args.output)
    root = args.dataset
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != 30:
        raise ValueError("Expected 30 Hz LeRobot v3 fixtures")
    episodes = [
        e
        for f in sorted((root / "meta/episodes").rglob("*.parquet"))
        for e in pq.read_table(f).to_pylist()
    ]
    manifest = []
    # Ten observations distributed over five episodes, plus one synthetic input.
    for i in range(10):
        ep = episodes[i // 2]
        data_path = root / info["data_path"].format(
            chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
        )
        rows = [
            r
            for r in pq.read_table(
                data_path,
                columns=[
                    "observation.state",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                ],
            ).to_pylist()
            if r["episode_index"] == ep["episode_index"]
        ]
        row = rows[int((0.2 if i % 2 == 0 else 0.7) * (len(rows) - 1))]
        sample = {
            "state_raw": np.asarray(row["observation.state"], dtype=np.float32),
            "noise": np.random.default_rng(100 + i)
            .normal(size=(1, 30, 32))
            .astype(np.float32),
        }
        for role, dataset_role in [
            ("top", "top"),
            ("left", "left_wrist"),
            ("right", "right_wrist"),
        ]:
            key = f"observation.images.{dataset_role}"
            path = root / info["video_path"].format(
                video_key=key,
                chunk_index=ep[f"videos/{key}/chunk_index"],
                file_index=ep[f"videos/{key}/file_index"],
            )
            stamp = ep[f"videos/{key}/from_timestamp"] + float(row["timestamp"])
            encoded = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-threads",
                    "2",
                    "-c:v",
                    "libdav1d",
                    "-ss",
                    str(stamp),
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-c:v",
                    "png",
                    "-",
                ]
            )
            sample[role] = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))
        np.savez(args.output / f"{i:02d}.npz", **sample)
        manifest.append(
            {
                "fixture": i,
                "episode": ep["episode_index"],
                "frame": row["frame_index"],
                "dataset": str(root),
            }
        )
    rng = np.random.default_rng(999)
    np.savez(
        args.output / "10.npz",
        state_raw=np.zeros(14, dtype=np.float32),
        noise=rng.normal(size=(1, 30, 32)).astype(np.float32),
        **{
            k: rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
            for k in ["top", "left", "right"]
        },
    )
    manifest.append({"fixture": 10, "synthetic": True})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} offline fixtures")


if __name__ == "__main__":
    main()
