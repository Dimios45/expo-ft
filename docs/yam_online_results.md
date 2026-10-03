# A100 / YAM online experiment results — 2026-10-03

The experiment is **concluded**. Six finalized episodes were automatically
uploaded and trained; the requested ten-episode collection was stopped early by
the operator. Training, policy inference and pod WebSocket services were stopped.
Final verification found no serving/training workers, ports 8204/8205/8208 closed,
and GPU usage at 0 MiB with 0% utilization.

The frozen checkpoint was `sra-vjti/molmoact2-yam-pi05-jax`, served through the
verified YAM adapter. This run started from base-only policy version 0 with
**empty replay and no earlier EXPO weights**, under experiment ID
`41986a02b40f48dd9d810d42b4376718`. The earlier `artifacts/yam-expo` and
`artifacts/yam-overlap-test` runs are separate experiments, not additional rounds
of this one.

## Outcomes and training

Episode indices below match NUC folders (`episode-000` through `episode-005`).
The terminal displayed human-facing episode numbers 1 through 6. Labels are
operator-provided, not inferred from video or critic loss.

| Episode index | Behavior policy | Recorded outcome | Resulting checkpoint | Critic / editor / temperature steps |
| --- | ---: | --- | ---: | --- |
| 0 | 0 | Truncated, reward 0 | 1 | 40 / 2 / 2 |
| 1 | 0 | Failure, reward 0 | 2 | 40 / 2 / 2 |
| 2 | 0 | Success, reward 1 | 3 | 40 / 2 / 2 |
| 3 | 1 | Success, reward 1 | 4 | 40 / 2 / 2 |
| 4 | 2 | Success, reward 1 | 5 | 20 / 1 / 1 |
| 5 | 5 | Failure, reward 0 | 6 | 20 / 1 / 1 |

Totals: **200 critic steps, 10 editor steps, 10 temperature steps**; base updates
and LoRA updates: **zero**. There were three successes, two failures and one
truncation among the six finalized attempts. A separate discarded timeout attempt
was preserved on the NUC and did not enter training. These mixed-policy,
operator-labeled attempts are not a held-out success-rate evaluation and do not
establish improvement over the base. The base itself produced one recorded
success. Checkpoint 6 was completed but has no recorded hardware evaluation in
this run.

Training used accumulated replay with replacement, not only the newest episode.
The main critic minibatch draws per round were:

| Checkpoint | Draws from episodes 0, 1, 2, 3, 4, 5 (when present) |
| --- | --- |
| 1 | 320 |
| 2 | 149, 171 |
| 3 | 87, 103, 130 |
| 4 | 69, 80, 100, 71 |
| 5 | 29, 33, 37, 34, 27 |
| 6 | 18, 33, 38, 32, 17, 22 |

These are draws, not counts of unique frames or transitions, and exclude separate
editor samples and terminal auxiliary examples. Truncations retain a bootstrap
mask and are not treated as absorbing failures. Successful/failed terminal
examples contribute a balanced auxiliary critic loss when present.

## Timing and GPU memory

The following measured stage wall times include process startup, compilation,
I/O and validation. `measure_gpu.py` sampled the **entire GPU** every 200 ms while
the serving process remained resident. Peaks may miss shorter transients and
must not be interpreted as learner-only VRAM or pure GPU-kernel time.

| Update | Preparation wall time | Training wall time | Preparation peak, total GPU | Training peak, total GPU |
| --- | ---: | ---: | ---: | ---: |
| 1 | 121.4 s | 120.1 s | 26.61 GiB | 14.80 GiB |
| 2 | 133.9 s | 120.7 s | 27.26 GiB | 15.73 GiB |
| 3 | 130.5 s | 112.0 s | 27.17 GiB | 16.08 GiB |
| 4 | 137.6 s | 110.8 s | 27.72 GiB | 15.92 GiB |
| 5 | 94.6 s | 87.7 s | 27.90 GiB | 16.45 GiB |
| 6 | 97.9 s | 89.7 s | 28.43 GiB | 16.96 GiB |

Preparation loads a second frozen base to generate candidate pools, which
explains its larger footprint. Gradient training updates the smaller EXPO
components. Representative instantaneous process snapshots showed 12.93 GiB
for base-only serving, 13.29 GiB for candidate preparation, and 1.18 GiB for a
gradient-training process. They were sampled at different times; they are not
independent peak measurements. Serving allocation later grew after EXPO reloads.

## Network measurements and timeout

All probes ran through the NUC-initiated SSH tunnel; routing inspection showed
wired Ethernet `eno1`. Measurements came from separate runs.

| Payload / test | Samples | Median RTT | p95 RTT | Maximum |
| --- | ---: | ---: | ---: | ---: |
| 1 KiB synthetic | 30 | 300 ms | 864 ms | 903 ms |
| 50 KiB synthetic | 30 | 601 ms | 693 ms | 772 ms |
| 300 KiB synthetic | 200 | 940 ms | 2,266 ms | 9,067 ms |
| 50 KiB repeat | 10 | 621 ms | 915 ms | 915 ms |

Five recorded-observation requests returned finite 30x14 actions. NUC round trips
were 2.15–5.94 seconds, server processing 0.94–1.00 seconds, and outside-server
overhead 1.21–4.97 seconds. The last includes transport and client overhead, not
just network latency. No camera capture or concurrent-training benchmark is
implied by these five requests. Sparse samples do not establish reliable p99
latency.

During a later hardware attempt, the bridge's 10-second deadline expired. KARMA
reported the episode discarded, parked both arms and de-energized them. An old
wrapper then incorrectly asked for a reward because it checked only for dataset
metadata existence. The recorder now requires a clean child exit, a saved episode
manifest and nonzero frame count before asking for a label. Regression tests
cover discarded and error exits.

Recovery preserved the discarded folder, used a 30-second bridge deadline, and
temporarily enabled `--wait-for-learner`. The retry saved 1,189 frames and was
labeled failure. That wait option intentionally makes collection sequential;
it is not the default overlap behavior. Increasing the deadline does not diagnose
or fix the original delay; network and GPU contention were not isolated as its
single cause.

## RTC sampler experiment

Separately, a YAM adapter tested shared vision/language cache sampling with 32
candidates, a five-tick committed prefix, and an eight-tick execution window.
The prefix was preserved exactly; zero-delay maximum absolute difference from
the existing sampler was 1.1921e-7. Three warm base-sampling calls took
0.968, 0.963 and 0.980 seconds, excluding preprocessing, networking, Q/editor
selection and training. This exceeds both the 0.167-second configured delay and
the 0.267-second execution window at 30 Hz. It was not deployed on hardware.

## Preserved evidence and limitations

Local artifacts (ignored by Git):

- `artifacts/yam-online-base/experiment-conclusion.json`: final episode labels,
  behavior versions and update totals.
- `artifacts/yam-online-base/learner/versions/0006`: latest completed checkpoint;
  previous versions and their manifests remain alongside it.
- `artifacts/yam-online-base/learner/online/`: uploaded datasets, queue records,
  and the `STOPPED.json` marker blocking coordinator restart.
- `logs/online-<archive-sha>-{prepare,train}.{log,csv,json}`: underlying measurements.
- `artifacts/yam-rtc/sampler-report.json`: RTC sampler numerical/timing results.
- NUC `~/yam-expo-data/online-base-run-001`: original recordings and discarded attempt.

Historical hot-reload health responses retained `expo_enabled=false` from the
initial base-only metadata even after an EXPO learner was loaded. Serving version,
checkpoint/hash, and the loaded learner governed action selection. The code now
sets this flag true on reload; historical session records are preserved as
recorded. `hardware_mapping_verified=false` also remained in health and must not
be read as hardware certification.

The deployed run implemented asynchronous episode upload, accumulated replay,
concurrent frozen-base EXPO learning and boundary publication. It did **not**
implement the paper's complete delay-aligned Real-Time EXPO-FT pipeline, live
per-step rewards, base/LoRA adaptation, uninterrupted deadline guarantees, or an
automatic physical-safety/performance evaluation. See the
[runbook](yam_online_runbook.md) and [remaining RTC work](yam_realtime_plan.md).
