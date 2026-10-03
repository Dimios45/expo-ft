#!/usr/bin/env python3
"""Check the YAM HTTP contract with synthetic inputs; never command hardware."""
import argparse
import io
import json
import time
import urllib.request

import json_numpy
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8204")
    args = parser.parse_args()
    base = args.server.rstrip("/").removesuffix("/act")
    with urllib.request.urlopen(base + "/act", timeout=10) as response:
        health = json.load(response)
    for key, expected in {"status": "ok", "norm_tag": "yam_dual_molmoact2",
                          "state_dim": 14, "num_cameras": 3}.items():
        if health.get(key) != expected:
            raise ValueError(f"Unexpected health {key}: {health.get(key)!r}")
    print(json.dumps(health, indent=2))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(frame).save(encoded, format="JPEG")
    for kind, image in [("raw", frame), ("jpeg", np.frombuffer(encoded.getvalue(), dtype=np.uint8))]:
        payload = {k + "_cam": image for k in ("top", "left", "right")}
        payload.update(state=np.zeros(14, dtype=np.float32), instruction="fold the towel",
                       num_steps=10, normalization_tag="yam_dual_molmoact2")
        request = urllib.request.Request(base + "/act", data=json_numpy.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json_numpy.loads(response.read())
        actions = np.asarray(result["actions"])
        if actions.shape != (30, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid {kind} response: {actions.shape}")
        print(f"{kind}: finite {actions.shape} actions, server={result.get('dt_ms')} ms, "
              f"roundtrip={time.monotonic() - start:.3f} s")


if __name__ == "__main__":
    main()
