# A100 stable EXPO round 1 — 2026-10-03

Completed one real-data training round from `a100-round-0000` and saved
`artifacts/yam-expo/versions/0001`. The registry now points to version 1.
The inference server was stopped for preparation/training and remains stopped
after the isolated validation server exited.

The operator labeled this episode successful. Its 2,783 frames produced 93
full action-chunk transitions and 95 observation anchors. Collection identity,
videos, metadata, and recorded session were validated by the importer.

The base model remains frozen: 40 critic/encoder optimizer updates, two editor
updates, two temperature updates, and zero base updates. The effective batch
is eight, edit bound is 0.05 in normalized coordinates, and gripper edits are
masked. All 29 base/processor asset hashes match initialization.

| Phase | Wall time, including startup/compilation | Sampled peak VRAM |
|---|---:|---:|
| Replay import and candidate preparation | 197.41 s | 13.72 GiB |
| Training, checkpoint save and reload | 139.61 s | 5.78 GiB |
| Isolated trained-policy HTTP validation | 61.66 s | 13.86 GiB |

These are whole-device GPU 0 readings from `nvidia-smi` every 200 ms on the
A100-SXM4-80GB; short-lived peaks between samples may be missed. No other GPU
workloads ran during these phases. The lower training peak reflects a frozen
base with precomputed candidate pools, rather than backpropagation through the
base model. Training does not need to fill 80 GB to preserve this experiment.

A100 execution changes:

- Four candidates per parallel sampling call. A fixed-input comparison took
  1.428 s sequentially versus 0.443 s in parallel (3.23× speedup), with maximum
  absolute output difference 3.58e-7.
- Eight gradient microbatches evaluated in parallel, preserving the effective
  batch, per-sample keys, loss weights, update cadence, and learning rates.
  Numerical tests cover equivalence to sequential accumulation, including an
  uneven final group. Ordinary warm critic steps took about 0.5–0.6 seconds.
- FP32 computation and the original stable learning schedule are retained.
  This is a measured configuration, not an exhaustive hardware-optimality claim.

All 25 targeted tests passed. The saved learner passed deterministic checkpoint
reload. Twelve recorded observations passed real HTTP inference with alternating
raw RGB/JPEG payloads; finite actions, policy identity, residual bounds, zero
gripper residuals, and invalid-input handling passed. Inference took roughly
0.93–0.96 seconds per request on localhost. This does not include NUC networking
or establish improved robot success.

Reports:

- `logs/yam-round1-summary.json`: combined measurements and training identity.
- `logs/yam-round1-{prepare,train,http}-gpu.{csv,json}`: GPU samples and summaries.
- `artifacts/yam-expo/versions/0001/metrics.jsonl`: all training steps.
- `artifacts/yam-expo/versions/0001/manifest.json`: hashes, settings, replay, counters.
- `artifacts/yam-expo/http-test-ad499876/report.json`: passing HTTP validation.

Start version 1 on the pod when ready:

```bash
cd /workspace/expo-ft
tmux new-session -d -s yam-expo \
  'cd /workspace/expo-ft && bash scripts/yam/pod_expo.sh serve >> logs/yam-expo-server.log 2>&1'
```

The existing NUC tunnel and localhost endpoint remain applicable. Wait for
warmup, check `policy_version: 1`, then record into a new directory such as
`a100-round-0001`. Use the same recording wrapper and exact task prompt.
