#!/usr/bin/env python3
"""Pod transport, NUC latency probe, or recorded inference test.

Probe/infer modes never command hardware. Serve mode can relay KARMA requests
when the inference bridge is explicitly enabled.
"""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import sys
import time

import msgpack


async def probe(args):
    from websockets.asyncio.client import connect

    durations = []
    payload = os.urandom(args.payload_kib * 1024)
    async with connect(args.url, compression=None, open_timeout=10) as socket:
        for seq in range(args.count):
            raw = msgpack.packb({"schema": 1, "kind": "probe", "seq": seq,
                                 "payload": payload}, use_bin_type=True)
            start = time.monotonic()
            await socket.send(raw)
            response = msgpack.unpackb(await asyncio.wait_for(socket.recv(), 30), raw=False)
            if response.get("kind") != "probe_ack" or response.get("seq") != seq:
                raise RuntimeError(f"Unexpected response: {response}")
            durations.append((time.monotonic() - start) * 1000)
            await asyncio.sleep(args.interval)
    ordered = sorted(durations)
    report = {"mode": "transport_only_no_gpu_or_camera", "samples": args.count,
              "payload_kib": args.payload_kib,
              **{f"rtt_p{p}_ms": ordered[math.ceil(p / 100 * len(ordered)) - 1]
                 for p in (50, 95, 99)}, "rtt_max_ms": max(ordered)}
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


async def inference(args):
    from websockets.asyncio.client import connect

    fixtures = msgpack.unpackb(Path(args.fixture).read_bytes(), raw=False)
    samples = []
    async with connect(args.url, compression=None, open_timeout=10, ping_timeout=180) as socket:
        for seq, fixture in enumerate(fixtures):
            raw = msgpack.packb(dict(schema=1, kind="infer_recorded", seq=seq,
                                    observation_json=fixture["observation_json"]), use_bin_type=True)
            start = time.monotonic()
            await socket.send(raw)
            reply = msgpack.unpackb(await asyncio.wait_for(socket.recv(), 200), raw=False)
            elapsed = (time.monotonic() - start) * 1000
            if reply.get("kind") != "inference_result" or reply.get("seq") != seq:
                raise RuntimeError(str(reply))
            actions = reply["actions"]
            if len(actions) != 30 or any(len(row) != 14 or any(not math.isfinite(v) for v in row) for row in actions):
                raise RuntimeError("Invalid action response")
            server = reply["server_elapsed_ns"] / 1e6
            sample = dict(frame=fixture["frame"], payload_kib=len(raw)/1024,
                          roundtrip_ms=elapsed, server_ms=server,
                          policy_reported_ms=reply["policy_reported_ms"],
                          outside_server_ms=elapsed-server, action_shape=[30, 14])
            samples.append(sample)
            print(json.dumps(sample), flush=True)
    report = dict(mode="recorded_inference_no_robot_no_training", samples=samples,
                  note="Server time includes CPU/HTTP and GPU work; outside-server time is not pure network latency.")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--journal", default="artifacts/yam-rtc/shadow.sqlite3")
    server.add_argument("--port", type=int, default=8205)
    server.add_argument("--recorded-inference", action="store_true")
    server.add_argument("--live-event-path")
    infer = sub.add_parser("infer")
    infer.add_argument("--url", default="ws://127.0.0.1:8205")
    infer.add_argument("--fixture", required=True)
    infer.add_argument("--output")
    client = sub.add_parser("probe")
    client.add_argument("--url", default="ws://127.0.0.1:8205")
    client.add_argument("--count", type=int, default=100)
    client.add_argument("--payload-kib", type=int, default=300)
    client.add_argument("--interval", type=float, default=0.1)
    client.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "serve":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from expo_ft.yam.stream import serve
        asyncio.run(serve(args.journal, args.port, args.recorded_inference, args.live_event_path))
    elif args.mode == "infer":
        asyncio.run(inference(args))
    else:
        if args.count < 1 or not 0 <= args.payload_kib <= 8000 or args.interval < 0:
            parser.error("count >= 1, 0 <= payload-kib <= 8000, interval >= 0 required")
        asyncio.run(probe(args))


if __name__ == "__main__":
    main()
