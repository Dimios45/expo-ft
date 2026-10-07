# Three-host YAM LAN measurements — 2026-10-04

Follow-up: SSH access is now established and the [three-host measurements](yam_lan_three_host_results.md) supersede the earlier access/network blockers below. The 3060 full RPC path missed its fresh-selection deadline; deployment remains gated.

These are **4090-host measurements**, not a certification of the 3060 or the
robot's 30 Hz control loop. No arms were commanded, base weights changed, or
policies published. The numerical learner updates use artificial rewards and
are discarded. The full three-machine runtime is still being integrated.

Machine: `omen`, RTX 4090 / 24 GiB, i9-13900K, approximately 64 GiB host RAM,
Ubuntu 22.04, NVIDIA driver 580.178.04. Wired interface `enp3s0`, address
`192.168.0.167`; `ethtool` reports **1000Mb/s, Full duplex**. Wi-Fi remains
active at `192.168.0.155`; network settings were not changed.

Checkpoint: `/usr/local/models/sra-expo-ft/yam_pi05_jax`; tokenizer:
`/home/sra/molmoact2/outputs/models/paligemma-tokenizer`. Base parameters occupy
6,706,867,744 bytes in bf16. Small trainable networks and inference bundles use
FP32. JAX preallocation is disabled, allocator is `platform`, Torch preprocessing
uses four CPU threads. Timings synchronize device work; startup/compilation is
separated from warmed execution. Sampled whole-GPU peaks use `nvidia-smi` every
200 ms and can miss shorter allocation spikes. The serving microbenchmarks
briefly initialize then discard learner optimizer state to create valid selector
parameters; their overall peak includes that setup. These are not trained C8
policy quality evaluations.

Full measurements and individual timing samples:
[yam-lan-4090-2026-10-04.json](benchmarks/yam-lan-4090-2026-10-04.json).
Original logs remain under `artifacts/yam-lan/` (git-ignored).

## Learner

Effective batch eight, 40 critic updates, two editor/temperature updates, ten-Q
ensemble, C=8, eight base candidates, eight edits. The frozen bf16 VLA remains
resident. Prepared synthetic inputs are reused; this table excludes replay
preparation, disk admission and transfer. Compiler/state-signature warmup occurs
before the timed cycle.

| Microbatch | Complete warmed update cycle | Sampled peak including setup |
| --- | --- | --- |
| 1 | 9.04 s | 8.10 GiB |
| 2 | 8.15 s | 8.41 GiB |
| 4 | 5.67 s | 8.40 GiB |
| 8 | 4.47 s | 8.42 GiB |

Microbatch **8** is the measured default in `configs/yam/lan.json`; every
deployment gate remains false. The benchmark exposed 10+ second recompilation
stalls when optimizer step/parameter signatures first changed. Warming both
critic and editor paths after the first updates removed those stalls from the
steady-state cycle. Production startup must warm the full state signatures,
not only one gradient call. These performance results do not establish learning
quality or justify a higher update-to-data ratio.

## Serving

Three 360×640 synthetic RGB inputs; table includes preprocessing but excludes
JPEG encoding/decoding and network transport. No training runs concurrently.
N=8/C=8 has 100 warmed samples per path; other configurations have 30. Delay
and window are control ticks at 30 Hz. The combined column is a sequential
compute measurement, not the pipelined control schedule.

| Candidates | C | Delay | Proposal median ms | Selection median ms | Combined median / max ms | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 8 | 5 | 87.17 | 5.56 | 94.15 / 98.87 | 7.21 |
| 4 | 8 | 5 | 92.96 | 5.56 | 99.41 / 104.48 | 7.21 |
| 8 | 8 | 5 | 103.02 | 5.21 | 108.57 / 111.44 | 7.21 |
| 32 | 8 | 5 | 180.00 | 5.44 | 186.16 / 192.76 | 7.50 |
| 8 | 12 | 8 | 103.18 | 5.34 | 108.92 / 117.93 | 7.21 |
| 8 | 15 | 10 | 103.02 | 5.46 | 108.43 / 113.74 | 7.22 |

Eight candidates fit the **4090** proposal budget of 5/30 = 166.7 ms. Thirty-two
candidates do not fit that delay. Fewer candidates save relatively little because
base encoding contributes a substantial fixed cost; the 3060 must be measured.
C=12/15 enlarge the permissible delay without materially reducing compute here.
These tests are too short and lack robot/network load to prove deadline reliability.

Second-buffer validation/device transfer/warm selection takes about **31–35 ms**
with unchanged shapes. Subsequent selection stays warm (~5 ms); initial selector
staging takes several seconds. A 31–35 ms staging operation can consume a whole
control tick: stage using measured slack, and do not interpret an idle GPU check
as a guarantee that the next inference request will meet its deadline.

## Five-minute-sized learner workload

For C=8 and 30 Hz, 300 seconds corresponds to 1,125 action windows. The separate
`episode-sized-4090` run repeated an actual recorded three-camera observation
from `tshirt-minimal-hitl-round-00`, JPEG-encoded at quality 95 (848×480 per view).
It generated fresh candidate noise for 1,125 uncached calls, then updated the
learner. It reuses one recorded observation, synthetic prefixes/rewards, and
already prepared gradient batches; it does **not** import or replay a complete
real episode.

- 1,125 uncached candidate preparations, including JPEG decode/preprocessing:
  **121.82 s**.
- Warm 40-critic/two-editor cycle with microbatch eight: **4.43 s**.
- Combined preparation + updates: **126.26 s**.
- Sampled peak while preparing candidates with learner resident: **9.69 GiB**.
- Inference tensors GPU→CPU: **52 ms**; safetensors serialization + fsync:
  **109 ms**; deserialization: **58 ms**.
- Full learner checkpoint (463,774,865 bytes), serialization + fsync: **1.06 s**.
- Entire fresh process, including imports/loading/compilation/warmup and all
  diagnostics: **196.96 s**.

Adding the separately measured rate-limited loopback download (7.87 s) to the
steady preparation/update path gives roughly **134 s** before other costs. This
is a component budget, not a measured three-host episode-to-READY latency. Real
replay ingestion, remote storage, network contention, 3060 staging, version
validation, and robot resets remain outside this measurement.

## Payload and local transport

The inference payload includes the encoder, bounded editor and target Q; optimizer
and temperature state are excluded. For C=8 it contains **150,454,440 tensor bytes**
(150.45 MB): encoder 136,593,152; editor 1,466,496; target Q 12,394,792. The actual
safetensors artifact is **150,470,936 bytes**. Freezing only the VLA does not freeze
the separate critic/editor image encoder, so the payload is not merely a few MB.

At ideal 1 Gbit/s, those bytes need 1.20 seconds before overhead. At an assumed
940 Mbit/s TCP rate they need 1.28 seconds; **940 Mbit/s has not been measured on
this LAN**. The application rate cap of 20 MB/s deliberately leaves control-link
headroom. The local HTTP test transfers incompressible synthetic bytes of the
same tensor payload size, with checksum validation and fsync:

| Operation, on 4090 host | Measured |
| --- | --- |
| SHA-256 of ~150 MB | 105 ms median |
| Loopback HTTP download, unlimited | 240 ms |
| Loopback HTTP download, cap 20 MB/s | 7.87 s |
| Resumable 8 MiB HTTP upload, cap 20 MB/s | 446 ms |
| Binary WebSocket JPEG observation → action response, loopback | 0.188 ms median; 0.350 ms p95 |
| MessagePack encode / decode | 0.0105 / 0.0048 ms median |

The loopback figures are **not Ethernet throughput or latency**. No Redis,
shared-memory, gRPC or ZeroMQ performance comparison was run. The measured
serialization/RPC cost is small compared with model compute; changing transport
is not currently justified by these results.

## Recorded-image bandwidth estimate

Three representative recorded 848×480 frames reused for 100 codec iterations:

| Codec | Three-view bytes | Encode median | Decode median |
| --- | --- | --- | --- |
| Raw RGB | 3,663,360 | — | — |
| JPEG q85 | 74,395 | 2.62 ms | 3.08 ms |
| JPEG q95 | 138,704 | 2.89 ms | 3.57 ms |

These CPU timings are on the i9-13900K, **not the NUC**. No H.264 latency/quality
benchmark or JPEG task-quality comparison was performed. At q95, the size model
is 4.16 MB/s for 30 Hz replay (1.25 GB per five-minute episode), plus 1.04 MB/s for
two inference observations every C=8 ticks. Frame reuse/dedup could reduce this;
scene variation and metadata can increase it. Raw 30 Hz replay alone would use
109.90 MB/s, leaving insufficient headroom once the separate inference flow is
included on one gigabit NUC port.

## Numerical checks and code correction

Before the fix, bf16 zero-delay prefix sampling differed from ordinary sampling
by 0.00390625 normalized units / 0.00334033 wire units; FP32 passed the same strict
check with maximum error 1.34e-7. The per-token time-conditioning path changes
floating-point operation shapes even with zero delay. The adapter now dispatches
zero-delay requests directly to the ordinary sampler; nonzero-delay sampling is
unchanged. The real-weight bf16 recheck gives **exact equality**, and committed
prefixes remain **exactly preserved**. See
[`rtc_sampling.py`](../expo_ft/yam/rtc_sampling.py) and the new regression test.
This does not validate cloth-folding behavior under prefix conditioning.

The separate FP32 diagnostic peaked at **13.31 GiB**, versus **6.89 GiB** for the
bf16 parity diagnostic. Its one-candidate warm bootstrap took ~184 ms versus
~76 ms after the bf16 fix. These are 4090 measurements; FP32 exceeds the 3060's
12 GiB capacity in this diagnostic. CPU/RAM offload was not benchmarked.

## Outstanding deployment measurements

- `/home` has only ~163 MB free; an actual checkpoint write failed for lack of
  space. Its own partial benchmark output was removed; no user data was deleted.
  Successful checkpoint/transport I/O measurements use the `/tmp` filesystem,
  which has only ~4 GB free. Provision replay/checkpoint storage before collection.
- The confirmed KARMA host is `yambox-GEM12` at `.121` and `.188`; both present
  the same SSH host key. The 3060 host is `sra-omen` at `.119`.
- Key-based SSH authentication is rejected for `yambox` and `sra` respectively.
  Remote GPU/NIC inventory, the 3060 tests, and NUC-side timing cannot run yet.
- Local probes to both yambox addresses: 1,500 packets each, zero loss;
  `.121` RTT min/avg/max/mdev = 0.059/0.792/2.707/0.352 ms;
  `.188` = 0.087/0.780/2.654/0.305 ms. Source routing is explicitly wired;
  the remote interface remains unverified. There was no bulk network load.
- 4090→3060 (`.119`) probe: 1,500 packets, zero loss; RTT
  min/avg/max/mdev = 0.127/0.400/0.635/0.067 ms. CPU regression tests ran
  locally during this probe; no bulk network traffic was generated.
- Neither yambox address accepts TCP connections on iperf3 port 5201.
  All pairwise throughput tests and loaded jitter, including NUC↔3060, remain
  outstanding. Hostnames/IPs alone do not prove traffic avoids remote Wi-Fi.
- Camera synchronization, executed-command provenance, loaded control deadlines,
  autonomous success labels, garment reset duration, and long-run stability remain
  unmeasured. No robot was accessed. No zero-idle guarantee is made.

## Validation

31 CPU tests passed across LAN integrity/leases, RTC sampler dispatch/prefixes,
EXPO math/checkpoint roundtrips and stable learner gradient accumulation. The
real-weight bf16 parity check also passed after the adapter fix. `git diff
--check` passed. Synthetic benchmark updates and weights were not deployed.

## Reproduce

Use the existing conversion/training environment plus
[`requirements-lan.txt`](../scripts/yam/requirements-lan.txt). The codec benchmark
also needs PyAV. Dataset-backed GPU input needs PyArrow. Benchmarks create only
new reports/scratch files and discard synthetic learner weights.

```bash
# Set paths appropriate to the host; do not copy this host's /home paths blindly.
.venv-convert/bin/python scripts/yam/benchmark_lan_suite.py \
  --checkpoint /path/to/yam_pi05_jax --tokenizer /path/to/tokenizer \
  --output artifacts/yam-lan-new-run --scratch /path/with/2GiB/free \
  --prepare-episode

# Serving-only measurement for the 3060 (wrap with measure_gpu.py for VRAM).
.venv-convert/bin/python scripts/yam/measure_gpu.py \
  --output artifacts/3060-gpu.json -- \
  .venv-convert/bin/python scripts/yam/benchmark_lan_gpu.py serve \
  --checkpoint /path/to/yam_pi05_jax --tokenizer /path/to/tokenizer \
  --output artifacts/3060-serving.json --candidates 8 --window 8 --delay 5 \
  --repeats 100

# Exact prefix/parity check on real weights.
.venv-convert/bin/python scripts/yam/benchmark_lan_parity.py \
  --checkpoint /path/to/yam_pi05_jax --tokenizer /path/to/tokenizer \
  --output artifacts/rtc-parity.json

# Recorded camera encoding and loopback transport (does not measure Ethernet).
.venv-convert/bin/python scripts/yam/benchmark_lan_transport.py \
  --dataset /path/to/recorded/episode --output artifacts/codec-transport.json

# On each receiver, explicitly bind iperf3 to its verified wired address:
iperf3 -s -B RECEIVER_WIRED_IP
# On each sender, repeat for NUC–3060, NUC–4090, and 3060–4090:
python3 scripts/yam/benchmark_lan.py --output artifacts/pair.json \
  --peer RECEIVER_WIRED_IP --source SENDER_WIRED_IP --interface WIRED_INTERFACE
```

The network helper records inventory, route validation, idle ping, forward,
reverse and simultaneous bidirectional iperf3, with a concurrent loaded ping in
each transfer. It reports unavailability/failures; it never changes interfaces,
Wi-Fi or routes. Review actual return codes rather than treating file creation
as passing a network gate.
