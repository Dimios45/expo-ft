# Three-host YAM LAN follow-up — 2026-10-04

SSH access is established. All three wired paths and the 3060 serving path were
measured. **The current implementation fails the strict fresh-selection timing
requirement**, despite adequate LAN bandwidth and GPU memory. No robot actions,
camera capture, success labeling, or resets were performed. No policy was
published. All temporary benchmark servers were stopped; the 3060 returned to
0% utilization with no CUDA compute process.

[Machine-readable measurements](benchmarks/yam-lan-three-host-2026-10-04.json)
include hardware inventory, transfer reports, every GPU/RPC timing sample, and
iperf3 results. [4090 learner results](yam_lan_benchmarks.md) remain applicable.

## Verified topology

| Role | Machine | Wired address / interface | CPU / RAM | Ethernet |
| --- | --- | --- | --- | --- |
| Training | omen, RTX 4090 24 GiB | 192.168.0.167 / enp3s0 | i9-13900K / ~64 GiB | 1000 Mb/s full duplex |
| Inference | sra-OMEN, RTX 3060 12 GiB | 192.168.0.119 / enp3s0 | i7-13700F / ~16 GiB | 1000 Mb/s full duplex |
| Robot client | yambox-GEM12 | 192.168.0.121 / eno1 | Ryzen 7 8745HS / ~16 GiB | 1000 Mb/s full duplex |

Yambox's `.188` address is Wi-Fi. All three machines still have Wi-Fi enabled.
The benchmark uses explicit wired source/destination addresses and verifies
routes. Network settings were not changed. These addresses are DHCP leases;
reserve them before deploying long-running services. `configs/yam/lan.json`
now records the verified wired addresses/interfaces; its deployment gates remain
false. Yambox actually has a second, disconnected Ethernet interface (`enp3s0`);
`eno1` advertises 2.5 Gb/s capability but currently negotiates 1 Gb/s.

The 3060 has ~29 GB free disk; yambox ~701 GB. The 4090's `/home` storage shortage
reported earlier remains unresolved. No existing checkout, checkpoint, KARMA
configuration, or environment was overwritten. Benchmark source and standalone
network tools ran from newly created `/tmp/yam-lan-*` directories. GPU tests used
the existing 3060 `.venv-yam` and `~/expo-ft/yam-pi05` assets. The copied source
uses the same OpenPI revision as the 4090 benchmark; full cross-host checkpoint
parameter fingerprints are still required before actual policy synchronization.

## Network measurements

Each pair ran idle ping, 30 seconds forward, 30 seconds reverse, and 30 seconds
simultaneous bidirectional iperf3, with concurrent ping during every transfer.
Traffic between different host pairs was tested sequentially. The initial 3060
GPU benchmark overlapped the 4090–3060 network test. Small SSH control traffic
and benchmark setup transfers also shared the LAN. Every iperf3 run succeeded;
all ping phases reported zero packet loss (1,500 packets per phase).

| Client → server | Forward received Mb/s | Reverse received Mb/s | Bidirectional received Mb/s, each direction | Worst loaded ping RTT |
| --- | --- | --- | --- | --- |
| 4090 → 3060 | 939.88 | 941.46 | 938.12 / 938.09 | 2.094 ms |
| 4090 → yambox | 939.73 | 941.11 | 934.93 / 935.70 | 4.068 ms |
| yambox → 3060 | 939.94 | 936.89 | 935.73 / 935.38 | 3.080 ms |

A second link or USB Ethernet adapter is not justified by the observed bandwidth
requirement. The existing gigabit LAN supports the estimated JPEG replay plus
inference traffic with considerable headroom. These short tests cannot establish
long-term jitter bounds or behavior during link failure.

## 3060 compute and memory

Real bf16 base weights, FP32 encoder/editor/target Q, 100 warmed samples per path.
The small networks are initialized for mechanics testing; they are not a learned
robot policy. Input is three synthetic 360×640 RGB arrays. CPU preprocessing is
included; JPEG/network transport is excluded from this table. Peak VRAM includes
initial creation of learner state followed by optimizer disposal and two-buffer
staging, sampled every 200 ms.

| Configuration | Proposal median | Selection median | Combined median | Sampled peak |
| --- | --- | --- | --- | --- |
| 8 candidates, C=8, delay=5 | 505.92 ms | 9.08 ms | 521.19 ms | 7.11 GiB |
| 2 candidates, C=15, delay=14 | 391.18 ms | 9.07 ms | 402.95 ms | 7.17 GiB |

Eight candidates miss the configured 166.7 ms proposal deadline by a large
margin. Two candidates fit the 466.7 ms proposal budget on compute alone. The
15-tick window still emits one action per 30 Hz tick; it changes replanning to
2 Hz and therefore needs task-behavior validation. It is a measured alternative,
not an automatically accepted change to the production algorithm.

Second-buffer validation/upload/warm selection took 35.47 ms (C=8) and 36.69 ms
(C=15). Staging must be scheduled with explicit slack; it can itself consume a
33.3 ms tick. The complete RPC server peaked at 7.45 GiB. GPU capacity is adequate
for these measured bf16 configurations; latency is the present issue.

## Real yambox → 3060 RPC test

The test runs the new binary MessagePack WebSocket service on the 3060 and its
client on yambox. Every request JPEG-encodes three recorded 848×480 RGB camera
frames at quality 95. The frames are reused, with synthetic state, prefixes and
logical tick IDs; no camera/arm drivers are imported. This tests the transport
and compute schedule, not actual capture-to-actuation latency or policy quality.

Configuration: C=15, delay=14, two candidates. A proposal starts at time zero;
fresh selection starts at tick 13 (433.3 ms), and the returned action window is
due at tick 14 (466.7 ms). Each trial is paced to a 15-tick/500 ms window. All
checks use yambox monotonic time. The loaded run overlaps seven consecutive
4090→3060 downloads at the proposed 20 MB/s cap. Warmups precede both runs.

| 100-trial run | Proposal median | Full fresh-selection median / p95 / max | Deadline misses |
| --- | --- | --- | --- |
| No bulk download | 411.96 ms | 29.16 / 31.13 / 34.73 ms | **5 / 100** |
| Concurrent capped downloads | 412.93 ms | 29.69 / 33.32 / 34.95 ms | **19 / 100** |

The loaded response arrived at 464.34 ms median, 468.19 ms p95 and 469.67 ms max,
against the 466.67 ms target. Server-side selection accounted for 22.03 ms median
and 26.66 ms maximum, while yambox JPEG preparation accounted for 3.31 ms median.
The remaining time includes RPC/serialization/event-loop overhead; it is not a
one-way network estimate. CPU-only JPEG q95 encoding measured 2.27 ms median on
yambox in a separate tight loop, illustrating why full-path measurements matter.

**Do not mark the 30 Hz deadline gate passed.** Increasing only the proposal
window does not fix the one-tick fresh-selection deadline. The next optimization
should prepare timestamped camera frames asynchronously and cache their exact
preprocessed image tensors, so fresh selection sends frame IDs plus current
proprioception and avoids JPEG work on its critical path. Cache images, not
embeddings from the changing small-network encoder. Preserve current transforms,
frame identity and timestamp checks, then repeat this loaded test. A two-tick
selection lead would be an explicit observation-age/algorithm change requiring
updated replay alignment; this benchmark did not silently introduce it.

## Actual wired artifact transfer

Authenticated resumable HTTP, SHA-256 validation, fsync, explicit wired server
address. Opaque incompressible bytes match the C=15 inference tensor size:
**151,759,800 bytes**; they are not a deployable safetensors checkpoint. No serving
weights were swapped by these transfer tests.

- 20 MB/s cap, seven consecutive downloads during RPC: **7.88 s median**,
  range **7.87–7.91 s**.
- Uncapped, three downloads after RPC: **1.47 s median**, range **1.45–1.48 s**.

This confirms the LAN weight-transfer budget. It does not include policy bundle
validation, GPU staging or episode-boundary promotion. Learner updates remain
comfortably inside the five-minute component budget; the unsatisfied requirement
is uninterrupted control timing, followed by actual camera/robot/reset validation.

## Tools added

- `scripts/yam/benchmark_camera_codec.py`: CPU-only NPZ camera fixture benchmark.
- `scripts/yam/benchmark_lan_rpc.py`: isolated two-phase server/client timing test.
- `scripts/yam/benchmark_weight_link.py`: authenticated, resumable real-LAN byte
  transfer benchmark with disposable scratch storage.

All three were exercised on the actual hosts. Temporary WebSocket/HTTP/iperf3
servers and their bearer tokens were removed after measurement. The user-created
SSH key remains available for subsequent authorized work. Wi-Fi, routes, robot
processes and existing deployments remain unchanged.
