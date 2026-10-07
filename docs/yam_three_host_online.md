# Three-host supervised online EXPO launchers

Updated 2026-10-07: see [runtime audit](yam_runtime_audit_20261007.md) for fixes, measured viability and interrupted-update recovery.

These launchers adapt the historical A100 workflow in `yam_online_runbook.md`.
The preserved `online-base-run-001` ran six finalized episodes of a requested ten,
then stopped. This implementation keeps its manual scene reset, task confirmation and reward
prompts, asynchronous archive upload, conservative stable-v2 learner, and
boundary reload. It is not the uninterrupted RTC pipeline: no-prefetch inference
pauses between chunks, and transfer/reload can pause at episode boundaries.

## Start in order, one terminal per host

4090 (`omen`):

```bash
cd /home/sra/tirth/expo-ft
bash scripts/yam/start_4090.sh
```

Wait for `Online learner ready`. The first startup initializes an empty registry,
fingerprints the actual bf16 model/processor/tokenizer, installs matching registry
metadata on the 3060, and installs a control token on both remote hosts. It uses
the existing `~/.ssh/yam_lan_bench` key. No robot commands originate here.

3060:

```bash
bash /home/sra/yam-lan/start_3060.sh
```

Wait for `3060 warmed; online policy ready`. It verifies that the local model
matches the 4090 fingerprint before listening. Stop any other GPU policy server
first if it occupies the VRAM; the launcher does not kill unrelated processes.

NUC (starts robot collection after operator prompts):

```bash
bash /home/yambox/yam-lan/start_yam_nuc.sh
```

The default camera mappings preserve the user's historical command: top
`348523020354`, left wrist `254623070863`, right wrist `254623070417`.
Those were NOT the serials attached during the latest inspection. If no valid
saved mapping exists, the launcher lists connected serials and asks you for the
physical overhead / left wrist / right wrist mapping, then saves it in
`~/yam-lan/selected-cameras.json`. You can also override it explicitly:

```bash
export TOP_SERIAL=<actual-overhead-serial>
export LEFT_WRIST_SERIAL=<actual-left-wrist-serial>
export RIGHT_WRIST_SERIAL=<actual-right-wrist-serial>
bash /home/yambox/yam-lan/start_yam_nuc.sh --check
bash /home/yambox/yam-lan/start_yam_nuc.sh
```

`--check` checks networking, camera identity and both services without starting
KARMA. The normal launcher retains KARMA preflight, command bounds and cleanup.
It uses the corrected current `record_round.py`, not the older NUC
`record_round-a100.py`, which lacks current finalization checks.

Defaults: run `lan-run-001`, 10 episodes, 180 seconds, 30 Hz, speed 1, 30-action
chunks, 10 denoising steps, no prefetch, prompt `fold the towel` (historical run).
Set `YAM_RUN` consistently in all terminals for another run. To use another task,
set `YAM_PROMPT` on the 4090 before initializing a new run; the launcher displays that exact task and asks for Enter confirmation before control. Set `YAM_EPISODES` / `YAM_SECONDS` on the NUC for collection length.
A finalized run directory is resumable; it is not silently overwritten.

## Communication and files

- NUC `.121` -> 3060 `.119:18204`: KARMA HTTP inference directly over Ethernet.
- NUC `.121` -> 4090 `.167:18208`: authenticated MessagePack WebSocket control
  and background resumable episode uploads. No new NUC SSH key is needed.
- 4090 -> 3060: existing SSH key + rsync, capped at 20,000 KiB/s. Checkpoint file
  hashes are verified before publishing the remote registry pointer. Session
  identities are copied back to the learner before an episode lease is returned.
- Reload requests require the run's control token. Active episode leases prevent
  publication until the next boundary. There is only one supported coordinator.
- Each GPU host uses `/home/sra/yam-online-lan/$YAM_RUN/{learner,serve}`. The NUC
  stores recordings in `/home/yambox/yam-expo-data/$YAM_RUN`.
- The 4090 stores learner jobs under `learner/online/queue`, learner status in
  `learner/online/learner.json`, and stage logs in the repository's `logs/`.
  The collector prints progress and writes `run.json` and final `summary.json`.

The launchers verify wired routes and gigabit links. WiFi remains enabled, but
these flows use the checked wired IPs. The old `.169` inference WiFi address is
not used. Ports 18204/18208 deliberately avoid the standalone base server's 8204.

This first adaptation transfers the full historical EXPO checkpoint, including
optimizer state (hundreds of MB), rather than the new compact inference bundle.
It preserves the proven restore path. GPU candidates are sampled sequentially
(batch 1) to limit 3060 memory; this is slower than the RTC shared-cache sampler.
The base is bf16 on both hosts; small-network training retains existing precision.
The learner uses effective batch 8 / microbatch 8. Numerical admission is not an
assurance that a learned policy improves robot behavior.

## Stop and recovery

Stop collection with Ctrl+C in the NUC terminal, wait for KARMA cleanup, and
finish any outcome prompts. Then stop the GPU launchers. Do not substitute a
network disconnect for the robot's stop procedure. Unfinished episodes remain
local; failed training jobs stop the queue. Inspect a failed job or an abandoned
lease before resuming; the launchers do not automatically clear either.

## Validation completed 2026-10-04

- The two GPUs produced identical actual bf16 model/processor/tokenizer hashes.
- 15 existing online, recorder and stability tests passed; after the WebSocket
  compatibility fix, all 10 online/recorder tests were rerun and passed.
- Cross-host integration passed: version 0 stayed pinned while version 1 was
  ready; releasing the lease allowed verified transfer, selector warmup and
  publication; serving-session metadata returned to the 4090. Unauthorized
  control requests were rejected. A recorded-image / synthetic-state inference
  request returned finite 30x14 actions (~2860 ms for eight sequential candidates).
- This integration used an isolated diagnostic registry and synthetic selector
  weights. It did not train on a live episode or actuate the robot. All diagnostic
  services were stopped afterward. Report: `artifacts/yam-lan/split-online-check.json`.
- Remote launcher and collector copies were SHA256 checked against local files.

The NUC launcher now enables `--fixed-prompt`: confirm the displayed experiment task with Enter; do not retype it. Upload-thread messages are deferred until prompt-safe boundaries. The launcher also enables `--archive-discarded`, which preserves and retries only verified empty attempts.
