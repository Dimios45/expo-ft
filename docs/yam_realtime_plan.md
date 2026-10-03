# YAM NUC + A100: from episode-wise EXPO to real-time EXPO-FT

Status: design and implementation plan, reviewed 2026-10-03. The working YAM
path remains sequential: record one episode, transfer it, stop inference,
prepare replay, train, validate, restart. Live concurrent learning is not yet
implemented for KARMA. The next immediate rollout uses stable frozen-base RL;
the requested all-weights experiment follows that rollout in a separate branch.

## What the two papers contribute

[EXPO-FT v2](https://arxiv.org/html/2605.25477v2) combines chunk-level critics,
bounded action edits, candidate ranking, and supervised base-policy updates.
Its appendix uses task-specific LoRA initialization and freezes the base image
encoder during RL. Thus “full VLA fine-tuning” in its motivation should not be
read as proof that every backbone parameter is optimized in the reported recipe.

[Real-Time EXPO-FT v1](https://arxiv.org/html/2609.18207v1) separates delayed VLA
candidate generation from editing/ranking with the latest observation. It adds
prefix-conditioned training/sampling, delay-aware backups, and a noise-Q filter
that avoids denoising every backup candidate. Its base update uses successful
episodes and a LoRA-based parameterization, with trainable vision/projection
components. That differs from both our frozen-base experiment and a literal
all-parameter update. Changing HTTP transport alone does not implement this
algorithm. These are paper findings; the engineering choices below are proposed
adaptations for this robot, checkpoint, network, and single GPU.

## Reuse and missing pieces

| Existing code | Reuse | Work still needed for YAM |
|---|---|---|
| `expo_ft/agents/alg/realtime_expo_ft.py` | Delayed critic/filter/editor algorithm | Adapt input keys, 14-D joint actions, normalization and replay |
| `expo_ft/agents/vla/pi05.py` | Prefix-conditioned loss and sampling wrappers | Load the converted checkpoint without DROID transforms |
| OpenPI `pi0.py` | Prefix-aware sampler and rematerialized model blocks | Verify sampler against current source-frame adapter |
| `train_pi_robo_async.py` | Chunk scheduling and immutable parameter publication ideas | Its device split assumes inference on device 0 and learner on other devices; one A100 needs a new scheduler |
| `client/run_client.py` | Existing outbound WebSocket transport | It is an environment RPC client, not a drop-in KARMA policy client |
| `expo_ft/yam/*` | Verified camera/state/gripper conversion, saved processing assets, stable learner, version registry | Streaming aligned replay, delayed observations, filter critic, live updates |
| KARMA | Local robot control, bounds, cameras, interventions, LeRobot recording | Nonblocking transport worker, action buffer and execution acknowledgments |

Keep the 14-D YAM joint/gripper frame and three-camera preprocessing. Do not
copy DROID's 7-D Cartesian actions, two-view assumptions, or task horizons into
the current checkpoint configuration.

## Recommended communications

Start with a persistent binary WebSocket connection initiated by the NUC,
with MessagePack metadata and binary JPEG/array payloads. WebSocket supports
bidirectional framed messages over TCP ([RFC 6455](https://www.rfc-editor.org/rfc/rfc6455)).
Use the existing SSH tunnel for the first implementation; then benchmark direct
authenticated WSS or a private network against it before changing connectivity.

Use a separate connection for bulk recording upload. Large video transfers must
not queue ahead of time-sensitive observations and action replies. Two channels
inside one SSH connection still share its underlying transport; separate tunnels
or separate WSS connections should be measured under packet loss and load.

| Option | Proposed use |
|---|---|
| Binary WebSocket | First choice: matches repository experience and allows observations, plans, acknowledgments, and labels in both directions |
| gRPC bidirectional streaming | Good alternative if generated schemas, service boundaries, and RPC tooling are priorities; [gRPC supports bidirectional streams](https://grpc.io/docs/what-is-grpc/core-concepts/) |
| Current HTTP keep-alive | Baseline for latency comparisons and existing sequential rollout |
| UDP/QUIC-based application protocol | Revisit only if measurements show TCP recovery dominates; would require explicit ordering, expiry, retransmission, authentication, and failure behavior |

WebSocket reduces transport ceremony; it does not remove WAN round-trip time or
the model's compute latency. The motor-control loop stays local on the NUC.

## Proposed data contract

Negotiate protocol/schema versions and units at connection setup. Every record
needs experiment ID, episode ID, policy version/hash, and sequence number.

- Observation: NUC monotonic capture time, execution tick, 14-D state, camera
  role/timestamp/encoding, current plan ID, consumed offset, committed action
  prefix, and locally measured remaining-buffer time.
- Candidate/plan: originating observation ID, policy version, intended execution
  start tick, action horizon, valid-until tick, 14-D absolute targets, committed
  prefix identity, and server queue/compute durations.
- Execution acknowledgment: actual applied targets after clipping, actual joint
  state, observation IDs, plan/action indices, intervention mask, and timestamps.
- Episode close: success/failure/truncation, reward source, authoritative label,
  final executed tick, and durable-recording hash.

Do not compare unsynchronized wall clocks to infer one-way latency. Use NUC-local
request/response timing, server-local durations, monotonic tick IDs, and measured
clock-offset uncertainty if cross-host timestamps are required.

At most one pending replaceable observation per robot; new observations replace
stale queued ones. Executed actions and labels require durable append/acknowledge
and deduplication, not dropping. Reconnection must not re-execute old plans.
Reject stale, wrong-episode, out-of-order, or expired actions. On buffer underrun,
invoke KARMA's defined local hold/stop behavior rather than repeat a stale chunk.

## Latency is currently the main constraint

Measured trained-policy inference is approximately 0.94 seconds on localhost,
before NUC camera/encoding time and WAN transport. At 30 Hz this is about 29
control ticks of delay. The current predicted horizon is 30 ticks. This leaves
almost no useful budget for a delayed pipeline, regardless of transport choice.

Choose execution window C and committed prefix d from measured end-to-end p99
latency, including concurrent learning. The selected prefix sampler must have
enough output positions for both the prefix and the executed suffix; for the
current fixed-H slicing design require d + C <= H and d <= C. With H=30 and
d around 29, these conditions cannot both hold. Simply enabling KARMA prefetch
does not solve this. A proposed initial target is C=8, with d<=8 only if total
p99 latency plus margin fits under 8/30 seconds. This is a target, not a measured
capability or a reason to silently alter the checkpoint horizon.

Priority optimizations to benchmark:

1. Reuse the VLM prefix/KV cache across noise candidates; current four-way vmap
   improves throughput but does not explicitly compute the prefix just once.
2. Precompile fixed camera, prefix, candidate, and batch shapes before robot use.
3. Benchmark candidate count and inference precision on identical recorded
   inputs. The current validated path remains FP32; earlier BF16 parity failed.
4. Separate delayed base generation from fresh-observation critic/edit selection.
5. Train a backup noise filter to avoid full candidate-pool regeneration for
   every streaming update. Refresh cached candidates when base weights change.
6. Measure NUC-to-A100 latency under simultaneous upload and gradient execution.
   If WAN jitter prevents the deadline, move fast editing/control inference to
   a suitable local GPU or move the inference GPU closer to the robot.

## One A100 serving and learning

Use one GPU-owning process with immutable inference snapshots, a bounded replay
ingestion queue, and a scheduler that admits small training units only when the
action-buffer deadline permits. Keep video encoding, durable replay writes, and
network I/O outside the GPU critical path. Model versions are swapped atomically
at chunk boundaries; requests must never see partly updated parameters.

Python threads do not guarantee GPU deadline priority. Measure worst-case update
duration and interference first. If a training unit cannot fit into the available
slack, defer it. Full-backbone gradients may need to remain episode-boundary work
on a single GPU while smaller critic/editor updates run during rollout. Two GPU
processes, MPS, or memory limits alone do not establish scheduling guarantees.

Existing measurements are from separate phases: roughly 14 GiB serving/preparation
and up to 5.8 GiB stable RL training. They are not a concurrent benchmark. Literal
FP32 all-weight Adam training has an approximate lower bound of 16 bytes per
parameter for parameters, gradients, and two moments, before activations,
temporary buffers, and an inference snapshot. Count actual model parameters and
measure a dry training step before reserving the remainder of the 80 GB device.
Disk must also accommodate new weights, optimizer state, rollback versions,
replay, and videos; this pod has only a 50 GB root filesystem.

Full-weights update and live concurrency are distinct experiments. Preserve the
original checkpoint and source hashes, export the new base separately, check
gradient participation across vision/language/action components, and validate
reloaded inference. A successful flow-loss decrease does not establish robot
improvement. A changed base also changes candidate distributions seen by the
existing critic; validate the combined policy before promotion.

## Replay and reward migration

Keep LeRobot videos on the NUC as the durable recording, while streaming aligned
transitions to the learner. Buffer incomplete chunks and finalize only actually
executed action windows. Store both delayed base observations and fresh editor
observations, committed prefixes, execution counts, intervention masks, versions,
and terminal flags. Discount by actual executed-step count using the declared
task discount convention.

Human labels currently arrive after each episode. Live streaming alone does not
make those rewards available sooner. Initially upload transitions during rollout
but admit reward-complete episodes after the operator label; train online using
previous finalized episodes. Per-step live learning needs an explicit strategy
for provisional transitions and correcting terminal labels, or a validated online
reward source. Do not turn an unfinished episode into a fabricated failure.

Our three historical episodes have nominal frame timestamps and fixed windows.
They can remain explicitly marked off-policy replay, but cannot be relabeled as
precisely timed RTC examples: missing committed prefixes, execution timestamps,
and fresh/delayed observation alignment cannot be recovered by assumption.

## Implementation sequence and acceptance checks

1. **Instrumentation and transport shadow test:** add the KARMA worker and record
   round-trip p50/p95/p99, queue age, upload contention, executed tick IDs, and
   buffer depth. Use recorded observations first; no new motion behavior.
2. **Asynchronous sampling adapter:** preserve the verified YAM mapping, implement
   committed-prefix sampling and fresh-observation selection; test against a
   deterministic mock executor with injected delay, jitter, disconnects and resets.
3. **Prefix training and delayed replay:** adapt the existing algorithm with unit
   tests for prefix masks, terminal/truncation targets, partial execution and
   exactly-once admission. Keep explicit versioned schemas separate from v1 replay.
4. **Concurrent frozen-base learner:** train small critic/editor units on finalized
   replay while serving. Promote snapshots at chunk boundaries; measure deadline
   miss rate and stale-plan rejection under load.
5. **Base updates:** first benchmark the requested all-weights offline update;
   then decide between that, action-expert updates, and the paper's LoRA recipe
   using measured memory, latency, and task validation. Add durable optimizer
   checkpoints and refresh/filter backup candidates when the base changes.
6. **Robot evaluation:** compare matched initial conditions against the preserved
   stable policy, logging task success, interventions, motion continuity, buffer
   underruns and end-to-end latency. Report observations, not inferred success
   from critic loss or hardware-free tests.

The next implementation milestone is a measured KARMA streaming bridge and
shadow-mode chunk scheduler, not a claim that the current HTTP server already
implements Real-Time EXPO-FT.
