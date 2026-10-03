# YAM experiment history and measurement ledger

Evidence reviewed on 2026-10-03. This page joins the separate experiments without
combining their version numbers or interpreting training loss as robot success.
Runtime status is historical: check `/healthz` before any deployment. Artifacts
and raw logs are local, ignored by Git; their paths below are evidence references,
not files included in a source commit.

## What has been completed

| Stage | Work and outcome | Evidence / instructions |
| --- | --- | --- |
| Model and environment setup | Published `sra-vjti/molmoact2-yam-pi05-jax` loaded in JAX; tokenizer access resolved; normalization and 14-D YAM action interface retained | [A100 setup](yam_a100_deployment.md), [conversion validation](yam_conversion_results.md) |
| Earlier local experiments | Separate 4090 work and stability checks; do not merge their replay/version numbers into A100 experiments | [Cloth folding](cloth-folding.MD), [stability results](yam_expo_stability_results.md), [earlier round 2](yam_expo_round2_results.md) |
| A100 manual collection/update loop | NUC Karma recording, SCP import, stable frozen-base updates and versioned serving; initial successful episode had 2,783 frames | [A100 round 1](yam_a100_round1_results.md), `artifacts/yam-expo` |
| Network and WebSocket tests | Synthetic RTT probes, five recorded inference requests, then operator-reported successful execution of a WebSocket hardware trial | [Network measurements](yam_online_results.md#network-measurements-and-timeout); `a100-ws-round-0004` was separate from the later fresh run |
| RTC sampler prototype | Prefix preservation and zero-delay parity passed; 32-candidate latency missed proposed RTC deadlines; not deployed for continuous RTC | [RTC measurements](yam_online_results.md#rtc-sampler-experiment) |
| Fresh online run | Base-only version 0, empty replay, automatic upload and six completed updates through version 6; 200 critic / 10 editor / 10 temperature steps; zero base steps | [Online outcomes](yam_online_results.md), `artifacts/yam-online-base/{serve,learner}` |
| Timeout and recovery | A 10-second inference deadline discarded an attempt; Karma parked/de-energized; recorder finalization checks and opt-in waiting/resume handling were corrected | [Recovery history](yam_online_results.md#network-measurements-and-timeout), [runbook](yam_online_runbook.md) |
| Operator-requested shutdown | Online learner, inference and relay stopped; final GPU measurement was 0 MiB; closed learner retains `STOPPED.json` | `artifacts/yam-online-base/experiment-conclusion.json` |
| Full-weight resource benchmark | Three-step FP32 all-parameter AdamW run, then separate export/reload check; all component gradient groups nonzero | [Base benchmark](yam_base_update_benchmark.md) |
| Ten-step full-weight run | `yam-full-base-run-001` completed ten updates and exported weights; no episode-held-out selection in this benchmark | `artifacts/yam-full-base-run-001/report.json` and `artifacts/yam-full-base-run-001-gpu.json` |
| Selected full-base fit | Original base trained on successful online episodes 2/3, episode 4 used for validation; 40 updates, effective batch 4; selected step 25 | `artifacts/yam-base-fit-001/report.json` |
| Candidate deployment check | Separate base-only registry, five HTTP and five WS recorded requests, finite 30x14 actions; no hardware outcome for this candidate established | [Candidate deployment](yam_base_update_benchmark.md#test-the-fitted-base-on-the-nuc) |
| Next architecture | Wired local 4090 inference, rented 80 GB learner, resumable data/weight transfer, newest-ready policy at episode boundaries | [Agreed plan](yam_realtime_plan.md#agreed-localcloud-architecture-planned-not-implemented); proposed, not implemented |

The online run had three successes, two failures and one truncation, with mixed
behavior versions. That is not a controlled success-rate comparison. The separate
full-base fit improves a fixed validation objective, not a measured hardware
success rate. Successful-episode flow matching is not a Q-gradient update through
the base and is not the completed Real-Time EXPO-FT algorithm.

## GPU and training measurements

All GPU values below are sampled whole-device usage, generally every 200 ms;
shorter peaks may be missed. Do not add peaks from separate runs as though that
were a measured concurrent workload.

| Workload | Peak VRAM | Wall time / interpretation |
| --- | ---: | --- |
| Initial isolated A100 candidate preparation | 13.72 GiB | 197.41 s |
| Initial isolated frozen-base EXPO training | 5.78 GiB | 139.61 s; base candidate pools precomputed |
| Initial isolated trained-policy HTTP check | 13.86 GiB | 61.66 s including startup |
| Six-round online preparation with server resident | 26.61–28.43 GiB | 94.6–137.6 s per round |
| Six-round online gradient stage with server resident | 14.80–16.96 GiB | 87.7–120.7 s per round; not learner-only memory |
| Full-weight three-step benchmark | 52.25 GiB | 74.56 s; batch 1, FP32 |
| Full-weight three-step export check | 52.26 GiB | 89.85 s including export |
| Full-weight ten-step run-001 | 52.25 GiB | 100.23 s including export |
| Full-base selected fit, 40 updates | **64.79 GiB** | **347.79 s**, microbatch 1 × accumulation 4 |
| Separate selected-base deployment | **13.31 GiB** | Startup and ten recorded HTTP/WS requests; 12.94 GiB resident afterward |

Full-base fit details:

- 3,353,433,872 trainable parameters; nonzero vision, language and action-component
  gradient norms on every update.
- 105 training windows across two successful episodes, sampled with replacement;
  160 draws over 40 updates. Eight fixed validation windows from a third episode
  select the checkpoint; that episode is not an independent final test set.
- FP32 AdamW, peak LR 5e-6 with warmup/cosine decay, gradient clipping at 1.
- Median update time after excluding the first two steps: 5.67 s.
- Fixed validation loss: 0.0278011 initially, 0.0222597 at selected step 25,
  a 19.93% reduction. Mean whole-run GPU utilization: 69.94%.
- Export contains inference weights and preprocessing assets, not optimizer state.
- Training and this deployment were sequential; their concurrent memory/latency
  has not been measured. The paper-style LoRA/vision learner on A100 and complete
  local 4090 RTC deployment have not yet been benchmarked.

Evidence: `logs/yam-base-fit-001-gpu.{json,csv}`,
`artifacts/yam-base-fit-001/{report,deployment-gpu,deployment-check,websocket-check}.json`.
Per-round overlap measurements and sample draws remain in the online results.

## Network and action timing

| Test | Observation |
| --- | --- |
| 1 KiB WAN probe | Median 300 ms; p95 864 ms |
| 50 KiB WAN probe | Median 601 ms; p95 693 ms |
| 300 KiB WAN probe | Median 940 ms; p95 2,266 ms; max 9,067 ms |
| Five recorded NUC-to-pod requests | 2.15–5.94 s roundtrip; 0.94–1.00 s server time |
| Prototype 32-candidate prefix sampler | 963–980 ms on A100 before network and Q/editor work |
| Selected-base pod-local HTTP/WS check | Policy time 409–497 ms; excludes NUC WAN latency |

These tests used different configurations and sample counts. They are not a
controlled protocol or hardware speed comparison. The last test is base-only;
it does not establish latency for the full candidate-ranking pipeline.
The NUC's observed route to the pod used wired `eno1` (192.168.0.165), despite
also having Wi-Fi address 192.168.0.200. WebSocket alone did not remove WAN delay.

## Preservation and handoff

Original model: `artifacts/yam_pi05_jax`. Selected fit:
`artifacts/yam-base-fit-001/checkpoint`. Its separate deployment registry starts
at version 0; that does not mean it is the original untrained base. EXPO is disabled
for that candidate trial so an older critic is not silently combined with it.

With explicit operator approval, only
`artifacts/yam-full-base-export-check/checkpoint` was removed to reclaim disk.
Its reports, the original model, online versions and run-001 weights were kept.
The selected fit has no optimizer export and cannot resume the same optimizer
trajectory from its inference checkpoint alone.

Commit source/docs/tests using the [documentation index handoff](README.md#source-control-handoff).
Back up models, replay and recordings separately. Historical docs preserve their
own run state; consult this ledger and health checks before reusing a command.
