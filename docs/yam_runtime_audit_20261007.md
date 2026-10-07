# Runtime audit — 2026-10-07

The 3060-inference / 4090-learning split works for supervised episode-wise EXPO.
It does not currently satisfy uninterrupted 30 Hz inference/replanning or a
new checkpoint after every 180-second episode. This audit did not actuate the
robot, restart inference, run GPU training, or alter recorded episodes/checkpoints.

## Observed run, not a simulated timing estimate

`/home/sra/yam-online-lan/lan-run-001` has five finalized recordings and four
completed updates. The fifth update was interrupted by KeyboardInterrupt after
candidate preparation; its persisted job still said `training`. Both EXPO
services were offline during inspection. A separate rollout and another model
were active on the robot and 4090; neither was changed.

| Completed update | Preparation | Training | Sum before weight transfer |
|---|---:|---:|---:|
| 1 | 169.53 s | 90.37 s | 259.90 s |
| 2 | 169.42 s | 69.95 s | 239.37 s |
| 3 | 72.76 s | 73.94 s | 146.70 s |
| 4 | 72.37 s | 73.95 s | 146.32 s |

Sources: `logs/online-<archive-sha>-{prepare,train}.json`, keyed by
`learner/online/queue/*.json`. Later preparation was shorter because the recorded
episodes contained fewer frames, not evidence of a general training speedup.
The first two recordings contain 3,625 / 3,624 frames; later ones contain 1,305
frames over approximately 180 s. These are recorded sample counts, not proof of
a continuous commanded control frequency. No-prefetch pauses account for part of
the difference; earlier diagnostic eight-candidate inference took about 2.86 s.

Each historical version directory is 479.26 MB; `expo.msgpack` is 479.24 MB.
It includes optimizer/training state. At the configured 20,000 KiB/s rsync limit,
payload transmission alone takes about 23 s, plus checksums, I/O and warmup.
An inference-only bundle remains a future optimization; the current launcher
keeps the historical restore format for compatibility. The previous LAN tests
measured approximately 940 Mb/s per wired pair; bandwidth is not the dominant
inference bottleneck. See `yam_lan_three_host_results.md` for those measurements.

## Fixes applied

- `scripts/yam/karma_episode.py` changes only KARMA's task-entry callback inside
  the child process. It displays the fixed experiment prompt and requires an
  empty Enter confirmation before the original power-up path. Ctrl+C propagates.
  The native controller, preflight, command bounds, stale-state checks, recording,
  parking and power-down remain KARMA's existing implementation.
- `record_round.py --fixed-prompt` invokes that adapter inside the KARMA environment.
  The launcher enables it. No task string needs retyping. Reward 0/1 and
  failure/truncated labels remain required for admission.
- `collect_online.py` defers background upload messages until a main-thread
  boundary so they cannot interrupt KARMA's terminal prompt. Failed learner jobs
  block starting the next episode. The launcher enables verified empty-attempt
  archiving on resume; saved episodes are never silently discarded.
- `expo_ft/yam/online.py` treats WebSocket disconnects as connection events,
  preserves durable state, updates both queue and displayed job status after an
  interrupted restart, and owns/kills only its learner subprocess group on
  cancellation. Job exceptions become failed jobs instead of killing the queue
  without recording status. The split coordinator handles SIGTERM cancellation.
- `expo_ft/yam/deployment.py` validates a staged checkpoint against its manifest,
  version and settings before atomically renaming it and publishing its registry
  pointer. Published versions are never overwritten. Downgrades and conflicting
  versions fail. Repeated synchronization reuses verified installed versions.
- Session metadata synchronization now completes before persisting a new episode
  lease. A failed session copy no longer leaves a phantom lease.
- A coordinator file lock prevents two new coordinators from running the same
  registry. New experiment fingerprints also include separate normalization
  safetensor hashes; historical registry fingerprints remain compatible.
- The read-only monitor reports coordinator-offline state explicitly rather than
  implying that a persisted `training` label proves a live worker exists.
- `recover_online.py` provides an explicit, locked, label/hash-verified retry of
  an interrupted prepared update. It never clears a lease or removes files.

The right-arm ID-1 native read failure is not fixed by these software changes.
CAN interface health alone does not prove servo health. No timeout, motor limit,
recovery condition or hardware protection was disabled to conceal it.

## Validation and deployment

23 CPU tests passed across online transport, collector, recorder, deployment,
recovery and learner stability. Added tests cover abrupt disconnects, session-copy
failure before lease creation, cancellation/process-group cleanup, corrupt
checkpoint rejection, idempotent publication, refusal to downgrade, fixed-task
confirmation/Ctrl+C, and recovery refusal for changed labels or active leases.
Shell syntax, Python compilation and `git diff --check` passed.

A real SSH/rsync transfer to the 3060, followed by remote checksum verification,
atomic publication and repeat synchronization passed with an isolated 1.08 MB
non-model fixture. Report: `artifacts/yam-lan/runtime-transfer-audit-20261007.json`.
This test does not claim a new GPU-inference or live hardware validation.

Updated files were copied to the dedicated EXPO directories on the 3060 and NUC.
No KARMA repository files or running hardware processes were modified. GPU
services were not started. The existing five-minute read-only monitor was
restarted with corrected offline reporting.

## Recovery before resuming the old experiment

Do this only after the separate active rollout is finished and the EXPO learner
is stopped. Inspection is read-only except for acquiring lock files:

```bash
cd /home/sra/tirth/expo-ft
python3 scripts/yam/recover_online.py \
  --root /home/sra/yam-online-lan/lan-run-001
```

To explicitly requeue the verified prepared update from the last committed
optimizer state (not from an unfinished temporary checkpoint):

```bash
python3 scripts/yam/recover_online.py \
  --root /home/sra/yam-online-lan/lan-run-001 --retry-prepared
```

Then use the three launchers in `yam_three_host_online.md`. The collector resumes
saved episodes, retries only verified empty attempts, and still requires physical
scene reset plus task confirmation. No recovery command above was executed with
`--retry-prepared` during this audit.

## Which GPU should do what?

Keep 3060 inference / 4090 learning for this supervised experiment. It isolates
inference from training contention and has already completed real updates. Use
five-minute episodes or accept stale policy versions; measured update+sync time
can exceed three minutes. Five minutes is not a proven worst-case guarantee.

Moving inference onto the 4090 is plausible for lower latency, but running the
learner there simultaneously introduces GPU contention. Memory alone is not a
latency guarantee. Measure loaded inference and complete-update latency before
moving it. The current separate model on that GPU uses about 13.8 GiB and is
serving another robot run, so migration was not attempted. Moving training onto
the 3060 needs its own complete replay-preparation/training benchmark; the 4090
numbers cannot be substituted. A single-4090 alternating inference/training
schedule is simpler but deliberately adds collection pauses.

For the original uninterrupted RTC goal, the outstanding work remains: remove
image preparation from the selection deadline, use shared-cache candidate
sampling, cache replay computations, send inference-only weights, stage/warm
before boundaries, and remeasure under load. The earlier 3060 RTC deadline test
missed 19/100 deadlines during concurrent transfer; the historical sequential
collector is not a fix for that result.
