# YAM Real-Time EXPO-FT: implemented pieces and remaining work

Status, 2026-10-03: the hardware experiment is concluded and all pod services are
stopped. The implemented operational workflow is **delayed online EXPO with a
frozen base**, automatic episode uploads and policy changes between episodes.
It supports collection/training overlap but does not promise action deadlines.
Use the [online runbook](yam_online_runbook.md) for commands and
[final results](yam_online_results.md) for measured outcomes.

## Paper recipe versus this deployment

[EXPO-FT v2](https://arxiv.org/html/2605.25477v2) combines chunk critics, bounded
edits, candidate ranking and supervised base-policy updates. Its appendix uses
LoRA initialization and freezes the base image encoder during RL. Its description
of VLA fine-tuning should not be read as a requirement to optimize every weight.

[Real-Time EXPO-FT v1](https://arxiv.org/html/2609.18207v1) separates delayed VLA
sampling from fresh-observation editing/ranking. It includes committed-prefix
conditioning, delay-aware backups and a noise-Q filter. Appendix VII-E starts
learning after ten completed episodes and flushes accumulated updates at episode
boundaries. Its reference uses 32 base candidates and 32 edits, batch size 64,
UTD 20, and success-only base updates with language LoRA plus other trainable
components. Updating during execution on a single GPU is an additional systems
extension, not the reported scheduling recipe.

Our run instead began learning after the first episode, used stable-v2's
conservative schedule, and kept the base frozen. The earlier all-weights request
was superseded; no all-weights or LoRA update was performed. Existing recorded
success labels do not retroactively make this an implementation of the full
paper recipe.

## Implementation map

| Component | Implemented | Remaining |
| --- | --- | --- |
| YAM preprocessing | 14-D joint/gripper frame, three cameras, saved normalization | Preserve parity in future optimized transport |
| Streaming transport | Binary MessagePack WebSocket over SSH, diagnostic journal, hardware bridge | Deadline-aware plans and direct binary-image encoding |
| Online collection | Ten-episode runner, operator prompts/labels, resumable background archive uploads | Per-tick execution/observation streaming |
| Replay and training | Episode queue, older-policy provenance checks, frozen-base caches, critic/editor/temperature updates | Fresh/delayed observation alignment, executed prefixes and filter critic |
| Publication | Completed checkpoint checks and one collector lease; changes between episodes | Crash recovery and richer behavior validation |
| Prefix sampler | `rtc_sampling.py`, exact prefix preservation and zero-delay parity checked | Meet end-to-end deadlines and integrate with the executor |
| Base adaptation | Not implemented for this online path | Success-only prefix loss, LoRA initialization, optimizer/export validation |

`expo_ft/agents/alg/realtime_expo_ft.py` and `expo_ft/agents/vla/pi05.py` contain
upstream algorithm and prefix-training components. `train_pi_robo_async.py`
assumes a device split with inference and learning on different GPUs. The
upstream environment client is DROID-oriented, not a drop-in KARMA client.
Repository defaults also differ from the paper's reference settings; they must
be made explicit when adapting the learner.

## Latency budget

The tested checkpoint has horizon H=30. The current upstream slicing adapter
requires `delay <= C`, `2*C <= H`, and sufficient output positions for delay+C.
A five-tick delay allows 167 ms at 30 Hz; an eight-tick window allows 267 ms.
The shared-cache sampler took roughly 963–980 ms for 32 candidates before
network or Q/editor processing. The hardware network added seconds in the
recorded-observation smoke test. This configuration cannot satisfy the proposed
RTC timing budget.

Sequential chunk execution can still work with pauses, as the hardware run
demonstrated. Do not describe that as uninterrupted real-time execution. Larger
windows, slower control, fewer candidates or altered precision are adaptations
that need explicit validation, not silent substitutes for the reported recipe.

Priority measurements:

1. Reduce payload overhead and benchmark image resizing/encoding against the
   validated preprocessing. Current inference envelopes contain JSON/base64.
2. Optimize the shared-cache sampler and precompile fixed shapes. The validated
   base path is FP32; BF16 requires renewed numerical and behavior checks.
3. Measure NUC capture-to-execution latency under uploads and gradient load,
   including buffer underruns and request deadlines.
4. If network latency dominates, compare a nearer GPU or suitable local inference
   hardware. A different RPC framework alone does not remove WAN latency.

## Required RTC replay contract

Every observation/plan/execution record needs experiment and episode IDs,
sequence/tick IDs and policy hashes. Store both delayed base observations and
fresh editor observations, the committed action prefix, intended execution tick,
actual applied commands after limits/interventions, timestamps, terminal labels
and rewards. Finalize only executed action windows. Nominal LeRobot timestamps
and arbitrary fixed windows cannot reconstruct missing delay alignment.

Use NUC monotonic durations for end-to-end timing and server-local durations for
compute. Do not subtract unsynchronized wall clocks as though they were one-way
latency. Replace stale pending observations, but durably retain executed-action
records and labels. Reject expired, wrong-episode or wrong-version plans and use
KARMA's defined local stop behavior on underrun.

Human rewards arrive at episode end. Live uploads do not supply earlier rewards;
training during collection uses previously finalized episodes. Provisional
transitions would need an explicit correction strategy rather than fabricated
failure rewards.

## Single-GPU scheduling and model publication

The implemented experiment uses separate inference and learner processes with
JAX preallocation disabled. This enables overlap but gives no GPU priority or
latency guarantee. Candidate preparation loads a second frozen base and used
more memory than EXPO gradient updates. Measurements are in the results report.

A deadline-aware implementation should admit small training units according to
measured execution-buffer slack, keep disk/network work outside the GPU critical
path, and use immutable inference snapshots. Python threads, MPS or memory caps
alone do not establish kernel preemption or deadline guarantees. Base-gradient
steps may need episode-boundary scheduling even if small critic updates overlap.

## Transport alternatives

Persistent binary WebSocket is implemented and interoperates with the existing
SSH access. gRPC streaming is a reasonable alternative for typed service APIs,
but no comparative latency benchmark was performed. Direct authenticated WSS or
a private network could be compared with SSH. Separate upload and action
connections avoid one application queue, but tunnels on the same SSH connection
still share TCP congestion and recovery. QUIC/UDP would require additional
ordering, expiry and reliability design; it is not an established fix here.

## Diagnostics and acceptance checks

- `rtc_transport.py probe`: synthetic payload round trips; no cameras/GPU.
- `make_rtc_fixture.py` + `rtc_transport.py infer`: recorded-state/camera
  inference, returned-action shape and finite checks; no robot commands.
- `check_rtc_sampling.py`: committed-prefix preservation, zero-delay parity,
  32-candidate timing. Its deployment-ready field remains false.
- `check_online_loop.py`: historical pod-only upload/train/reload integration
  probe. It targets the older overlap workspace and existing recordings; it is
  not the hardware runner and must not be launched during a collection session.
- `overlap_once.py`: historical one-shot pilot, superseded by `online_server.py`.

Before deploying full RTC, validate timing-aligned replay and prefix masks,
truncation/terminal targets, stale-plan rejection, reconnect behavior, candidate
normalization, measured concurrent-update deadlines, and matched-condition
robot evaluation. The completed online experiment supplies useful infrastructure
and evidence, not completion of these remaining steps.
