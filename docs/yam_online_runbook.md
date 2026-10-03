# YAM online collection and EXPO learning on one A100

Status: the 2026-10-03 experiment is **concluded**. Training, inference and pod
WebSocket services were stopped at the operator's request. Ports 8204, 8205 and
8208 were closed and GPU usage returned to 0 MiB. These instructions describe
how to operate a deliberately started experiment; they do not imply one is live.

See [experiment results](yam_online_results.md) for measurements and
[the RTC roadmap](yam_realtime_plan.md) for the distinction from the paper's
Real-Time EXPO-FT algorithm. Use [A100 deployment](yam_a100_deployment.md) to
install the model, tokenizer, pinned OpenPI checkout and `.venv-convert` first.

## Implemented workflow

The NUC invokes unmodified KARMA rollout commands. Its local bridge converts
HTTP requests into persistent WebSocket requests through SSH. On the pod, the
transport forwards inference to the JAX policy server. This remains sequential
30-action-chunk execution with prefetch disabled; pauses between chunks are
expected.

After an episode is saved and labeled, a background NUC thread uploads its
archive over a separate WebSocket connection. Uploads have offsets, SHA256
verification and retry handling. The pod validates/extracts the archive, queues
it, imports replay, prepares candidate actions, and trains. Prior frozen-base
candidate pools can be reused after integrity checks. Older behavior policies
are accepted only after checking experiment, saved serving-session identity,
checkpoint manifest and weight hash.

The learner trains the visual encoder, Q ensemble, bounded editor and entropy
temperature. Base VLA weights stay frozen. Each round takes
`min(40, 20 * ceil(new_episode_transitions / 40))` critic steps, with effective
batch size 8 and one editor/temperature update per 20 critic steps. Replay is
sampled across all admitted episodes, weighted by transition counts. Successful
and failed terminal transitions supply an auxiliary loss; truncated endings
retain bootstrapping. This conservative YAM schedule is not the paper's batch-64,
UTD-20 real-time recipe, and success labels do not trigger LoRA updates here.

Before each episode the collector obtains a policy lease. The server loads the
newest completed checkpoint at that boundary after integrity and selector checks.
The lease prevents another collector from requesting a policy switch mid-episode.
Collection does not wait for a newer checkpoint by default. A pending checkpoint
becomes available for a later boundary. Learned behavior is not guaranteed to
improve merely because numerical checks pass.

| Component | Entry point | Location / port |
| --- | --- | --- |
| Supervised collection + upload thread | `scripts/yam/collect_online.py` | NUC |
| Per-episode KARMA invocation and label | `scripts/yam/record_round.py` | NUC |
| HTTP-to-WebSocket bridge | `scripts/yam/rtc_karma_bridge.py` | NUC localhost:8207 |
| Inference WebSocket | `scripts/yam/rtc_transport.py serve` | Pod localhost:8205 |
| JAX policy / boundary reload | `scripts/yam/pod_expo.sh serve` | Pod localhost:8204 |
| Upload queue and serial learner worker | `scripts/yam/online_server.py` / `expo_ft/yam/online.py` | Pod localhost:8208 |
| Replay preparation and RL | `scripts/yam/continue_stable.py` | Pod subprocesses |

The pod services are loopback-only and rely on SSH for access control. Do not
expose the unauthenticated HTTP reload or upload/control endpoints publicly.
Only the online collector should initiate hardware episodes while its coordinator
owns policy publication. Do not run the diagnostic scripts concurrently with
hardware collection: they send inference requests and some acquire policy leases
or train/promote historical data.

## Fresh experiment on the pod

The closed run lives at `artifacts/yam-online-base/{serve,learner}`. Its
`learner/online/STOPPED.json` intentionally prevents coordinator restart.
Preserve it. For a new run, choose a new directory; do not silently remove the
stop marker or overwrite an existing experiment.

The example below assumes installed checkpoint/tokenizer assets and dependencies.
Run from `/workspace/expo-ft` only when deliberately starting a new experiment:

```bash
cd /workspace/expo-ft
uv pip install --python .venv-convert/bin/python -r scripts/yam/requirements-rtc.txt
mkdir -p logs
.venv-convert/bin/python scripts/yam/continue_stable.py init \
  --root artifacts/yam-online-next/learner \
  --checkpoint artifacts/yam_pi05_jax \
  --tokenizer artifacts/paligemma-tokenizer \
  --prompt 'fold the towel'

python3 - <<'PY'
from pathlib import Path
import shutil
root = Path('artifacts/yam-online-next').resolve()
serving = root / 'serve'
serving.mkdir()
for name in ('experiment.json', 'current.json'):
    shutil.copy2(root / 'learner' / name, serving / name)
for name in ('versions', 'replay'):
    (serving / name).mkdir()
(serving / 'sessions').symlink_to(root / 'learner/sessions', target_is_directory=True)
PY

tmux new-session -d -s yam-expo \
  'cd /workspace/expo-ft && YAM_EXPERIMENT=/workspace/expo-ft/artifacts/yam-online-next/serve YAM_ONLINE_TRAINING_ROOT=/workspace/expo-ft/artifacts/yam-online-next/learner bash scripts/yam/pod_expo.sh serve > logs/yam-online-next-policy.log 2>&1'
tmux new-session -d -s yam-rtc-transport \
  'cd /workspace/expo-ft && .venv-convert/bin/python scripts/yam/rtc_transport.py serve --port 8205 --recorded-inference > logs/yam-online-next-transport.log 2>&1'
tmux new-session -d -s yam-online \
  'cd /workspace/expo-ft && .venv-convert/bin/python scripts/yam/online_server.py --root artifacts/yam-online-next/learner > logs/yam-online-next-coordinator.log 2>&1'
```

The `--recorded-inference` switch enables the transport's inference bridge,
including the `infer_live` messages used by KARMA. The name predates hardware
integration. Initialization creates base-only version 0 and empty replay. Wait
for the policy warmup, then check `curl --fail http://127.0.0.1:8204/healthz`.
Confirm the new experiment ID, version 0, `expo_enabled=false` and a null policy
checkpoint. Do not launch another process on an occupied port or reuse an active
tmux session name.

## NUC setup and launch

The NUC must already have working KARMA, `uv`, CAN interfaces, camera access and
its calibrated hardware configuration. The bridge environment is separate from
KARMA's environment. Download the client scripts together so their arguments
match. The following copies source code once, not episode recordings.

```bash
sudo apt update
sudo apt install -y python3.10-venv
python3.10 -m venv ~/.venvs/yam-rtc-py310
~/.venvs/yam-rtc-py310/bin/python -m pip install 'websockets>=14,<16' 'msgpack>=1,<2'
cd ~/karma
scp -P 19032 -i ~/.ssh/id_ed25519 \
  root@154.54.102.50:/workspace/expo-ft/scripts/yam/collect_online.py \
  root@154.54.102.50:/workspace/expo-ft/scripts/yam/record_round.py \
  root@154.54.102.50:/workspace/expo-ft/scripts/yam/rtc_karma_bridge.py ./
```

Addresses and SSH port are those used in this experiment; update them for a
replacement pod. The NUC's route used Ethernet `eno1`, source `192.168.0.165`.
Its Wi-Fi address was `192.168.0.200`. The pod cannot initiate a connection to
these private addresses; the NUC initiates the tunnels.

Terminal 1, kept open:

```bash
ssh -N -T \
  -L 127.0.0.1:8205:127.0.0.1:8205 \
  -L 127.0.0.1:8208:127.0.0.1:8208 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -p 19032 -i ~/.ssh/id_ed25519 root@154.54.102.50
```

Terminal 2, kept open:

```bash
cd ~/karma
~/.venvs/yam-rtc-py310/bin/python rtc_karma_bridge.py --timeout 30
```

The bridge default is 10 seconds; the experiment later used 30 seconds after a
request timeout. This increases allowed delay, not inference speed. KARMA's
request timeout must exceed the bridge deadline. The reviewed checkout default
was 60 seconds. The bridge never retries an action request automatically.

Terminal 3, **starts robot motion after operator prompts**:

```bash
cd ~/karma
curl --fail http://127.0.0.1:8207/healthz
~/.venvs/yam-rtc-py310/bin/python collect_online.py \
  --recorder ./record_round.py \
  --root ~/yam-expo-data/online-next-run-001 \
  --episodes 10 --seconds 180 \
  -- \
  --interface left=can_left --interface right=can_right \
  --camera-serial top=348523020354 \
  --camera-serial left_wrist=254623070863 \
  --camera-serial right_wrist=254623070417
```

Before every episode: reset the scene, press Enter, then enter `fold the towel`
at KARMA's prompt. After execution, answer KARMA's outcome prompt and the
recorder's authoritative reward and termination questions. A time-limit ending
is not automatically a failure. Upload and training messages may appear while
another episode is recording. Leave the runner open after episode ten to drain
uploads/training and write its `summary.json`. The summary records observed
outcomes and behavior versions; it is not a controlled policy evaluation.

## Interruptions and recovery

The scripts preserve KARMA's foreground process group, control loop, command
limits, stale-state checks and native cleanup path. The reviewed KARMA code saves
partial recordings on Ctrl+C and calls park/power-down cleanup in `finally`.
Wrappers wait for child exit even after repeated interruptions. These are code
checks and simulated tests, not hardware safety certification or a guarantee
for every firmware/network/interruption condition.

- Ctrl+C at the reset prompt starts no episode.
- During motion, press Ctrl+C once in the **collection terminal**, let KARMA
  finish cleanup, and answer remaining outcome prompts. The wrapper then stops
  the multi-episode loop; it does not start the next episode automatically.
- Ctrl+C during labeling can leave an unlabelled recording. It is not uploaded
  as valid training data. Inspect it before resuming.
- Do not stop a tunnel or bridge as a substitute for the normal robot stop.
  Keep hardware stop controls available. SIGKILL/process crashes cannot run
  Python cleanup handlers.

A finalized episode requires a clean KARMA exit, one saved episode in its
manifest, nonzero frames, and a matching policy/session identity. The recorder
now rejects discarded/error attempts before asking for an outcome label.

For a stopped collector, rerun with the same root to skip finalized episodes and
resume uploads. `--archive-discarded` moves only verified zero-episode attempts
into `discarded-attempts/` and retries that slot; it refuses to move saved data.
It never deletes the failed attempt. Other incomplete recordings require manual
inspection.

`--wait-for-learner` is an **optional sequential diagnostic mode**. It drains
uploads/training before the next episode and therefore disables collection /
training overlap. It was temporarily used after the timeout. Remove it to
restore normal overlap. Neither mode guarantees a deadline on this network.

Training failures block further learner jobs. A coordinator restart marks an
interrupted training job failed rather than silently repeating it; inspect the
pending state and checkpoints before recovery. An existing policy lease also
requires checking that no robot episode is active before clearing it. There is
no automatic recovery of unfinished optimizer steps or abandoned leases.

## Evidence and shutdown

Data locations on the pod:

- `learner/online/uploads/`: archives, expected sizes and upload offsets.
- `learner/online/episodes/`: extracted completed datasets.
- `learner/online/queue/`: per-episode queued/training/ready/failed state.
- `learner/online/lease.json`: active episode and its pinned serving identity.
- `learner/current.json`, `learner/versions/`: latest completed learner snapshot.
- `serve/current.json`, shared `sessions/`: published policy and collection identities.
- `logs/online-<archive-sha>-{prepare,train}.{log,csv,json}`: stage logs and sampled
  whole-GPU memory reports. These reports are not per-process attribution.

End robot collection first and allow local cleanup. Stop the pod coordinator and
any active learner subprocesses before stopping inference. Preserve completed
checkpoints and recordings; an interrupted update is not a completed checkpoint.
A `learner/online/STOPPED.json` marker prevents coordinator startup. Closing the
coordinator alone is not a reliable way to terminate an already spawned learner;
verify child processes explicitly. After stopping the policy and transport,
verify ports and GPU processes. This documentation update does not restart any
services or remove the closed run's stop marker.
