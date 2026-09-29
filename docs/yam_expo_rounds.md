# YAM EXPO: one episode, train, redeploy

> Historical initial-runner guide. For the active frozen-base `stable-v2`
> cloth-folding experiment, use [Cloth folding on the 4090 and YAM NUC](cloth-folding.MD).
> Its `continue_stable.py` workflow supersedes the legacy training commands below.

This runner uses the working FP32 YAM π₀.₅ loader and Karma HTTP interface. It
never opens robot drivers. Only the `record_round.py` command, run by the operator
on the robot machine, invokes Karma and moves hardware.

## What is implemented

- 10 Q functions over the complete **30 × 14** normalized action chunk.
- Random minimum of 2 target critics for selection and Bellman backup.
- 8 base candidates and 8 bounded stochastic editor candidates by default.
- Learned critic image encoder, edit actor and entropy temperature; soft target-Q
  update with tau 0.005. Uses the existing EXPO network modules and loss structure.
- Accumulated off-policy replay; exactly one new labeled episode per round.
- Optional success-only flow-matching updates to the base action expert and
  action/time projections (`--actor-mode expert`). The VLM remains frozen.
- Immutable version folders and an atomic current-version pointer. Serving,
  ingestion and training share an exclusive experiment lock. No round is
  published after a training error. Optimizers and target Q survive restarts.
- Server-side request traces with policy identity, observations, base candidates,
  edited candidates, selected critic pair, all 10 Q values and returned actions.

This is a **sequential single-GPU adaptation**, not simultaneous real-time EXPO.
Microbatch is 1; base candidates are sampled sequentially to bound GPU memory.
No random image augmentation is applied in this initial runner. Actor updates
use the existing flow loss on successful executed-action chunks, not a Q-gradient
through the VLA. `expert` trains approximately 564M parameters, not the entire
3.35B action model and not upstream's LoRA configuration. `frozen` updates only
critic/encoder/editor/temperature. No silent precision changes or dimension
reduction are used to make a run fit.

## Install on the 4090

```bash
cd /home/sra/tirth/expo-ft
UV_CACHE_DIR=/tmp/uv-yam-expo-cache uv pip install \
  --python /usr/local/models/sra-expo-ft/venv-jax/bin/python \
  -r scripts/yam/requirements-expo.txt
```

Create a new experiment once, using the larger disk for all replay/checkpoints:

```bash
/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/round.py \
  --root /usr/local/models/sra-expo-ft/towel-expo \
  init \
  --checkpoint /usr/local/models/sra-expo-ft/yam_pi05_jax \
  --tokenizer /home/sra/molmoact2/outputs/models/paligemma-tokenizer \
  --prompt 'fold the towel' --actor-mode expert
```

This configuration passed the offline 4090 smoke test with full critic/editor
networks and an FP32 action-expert update. Use `--actor-mode frozen` instead to
isolate critic/editor learning. The setting stays fixed across rounds. The base actor is unchanged when replay has no successful episodes;
this is recorded as zero actor updates, not reported as base-policy training.

Start version 0, which serves the unchanged base policy (no random critic/editor):

```bash
CUDA_VISIBLE_DEVICES=0 \
JAX_COMPILATION_CACHE_DIR=/usr/local/models/sra-expo-ft/jax-cache \
/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/round.py \
  --root /usr/local/models/sra-expo-ft/towel-expo serve --port 8204
```

Use this round-aware server, not `serve_jax.py`, for the experiment. Its health
reply includes experiment ID, policy version, collection settings and session ID.
Wait for warmup. No other server may use port 8204. Do not run inference and
training concurrently on this GPU.

## Collect one episode on the NUC

Copy only `scripts/yam/record_round.py` onto the NUC, then run it from the configured
Karma checkout using Python 3.12+. The existing camera, calibration, CAN and safety
setup must already work. Replace interface and serial placeholders with your
existing values; these are not new hardware setup instructions.

```bash
python3 /path/to/record_round.py \
  --server http://192.168.0.167:8204 \
  --out "$HOME/yam-expo-data/round-0000" --seconds 300 -- \
  --interface left=LEFT_CAN --interface right=RIGHT_CAN \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL
```

Enter exactly `fold the towel` when Karma asks. Keep one attempt per episode:
no unmarked resets or human interventions inside it. Karma currently asks its
own y/n success question; the wrapper then asks for the authoritative **0/1**
reward and whether a zero-reward ending is failure or truncation. The wrapper
requires the same server session/version before and after recording.

The wrapper fixes 30 Hz, speed 1.0, chunk 30, 10 solver steps and no prefetch.
This deliberately avoids treating half-speed playback as the same MDP. It
produces `expo_session.json` alongside the LeRobot v3 files. Do not invent that
file for older recordings: it is the collection/version contract.

Transfer the whole folder (not just parquet) to the GPU host. For example,
create the destination on the GPU host and run rsync from the NUC:

```bash
# GPU host
mkdir -p /usr/local/models/sra-expo-ft/incoming

# NUC
rsync -av "$HOME/yam-expo-data/round-0000" \
  sra@192.168.0.167:/usr/local/models/sra-expo-ft/incoming/
```

## Stop server, train one round, restart

After collection finishes, stop the round server with Ctrl+C. Then on the GPU host:

```bash
cd /home/sra/tirth/expo-ft
/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/round.py \
  --root /usr/local/models/sra-expo-ft/towel-expo ingest \
  --dataset /usr/local/models/sra-expo-ft/incoming/round-0000

CUDA_VISIBLE_DEVICES=0 \
JAX_COMPILATION_CACHE_DIR=/usr/local/models/sra-expo-ft/jax-cache \
/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/round.py \
  --root /usr/local/models/sra-expo-ft/towel-expo train --updates 100 --actor-every 10
```

`100` is an initial experiment budget, not the paper's UTD schedule. With expert
mode, one successful replay chunk is used for an actor update every 10 RL updates.
All ten critics receive every RL update. With all failures, Q targets contain no
observed positive successes; a trained policy should not be assumed improved.

Only after `READY TO DEPLOY` appears, start the same `serve` command again.
Health should report `policy_version: 1`. Collect into `round-0001`, transfer,
stop serving, ingest, train, and serve again. Round 2 samples both episodes.
A stale policy-version episode or duplicate episode is rejected.

```bash
/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/round.py \
  --root /usr/local/models/sra-expo-ft/towel-expo status
```

Each `versions/0001`, `versions/0002`, ... contains RL/optimizer/target state,
optional actor patch and optimizer, metrics, and a manifest with file hashes.
The original base checkpoint is never overwritten. A failed training attempt
leaves an unpublished `.training-*` directory; retry training uses the original
parent version, not partially updated memory. Keep the parent versions.

## Data and reward details

Karma records **0=open** grippers but sends native **1=open** values over HTTP.
Import converts recorded indices 6 and 13 back to the working wire convention
before saved checkpoint normalization. This is a dataset-boundary conversion;
the functioning inference server still performs **no gripper inversion**.
The dataset frame is explicit in `expo_session.json`; it must not be guessed for
other collectors. All three cameras are decoded and ordered top/left/right.

Each transition uses 30 actual recorded commands and the observation at t+30.
Terminal windows receive gamma^29 times the final binary success reward;
bootstrap uses gamma^30. True success/failure disables bootstrap; truncation does
not. The final full window is included even when episode length is not divisible
by 30. Partial/padded commands are not invented. Actor targets pad 14 values to
32 internally. Recorded commands include Karma's bounding/clipping.

LeRobot frame timestamps are nominal, not wall-clock control timestamps. The
runner does not claim exact wall-clock credit assignment or counterfactual
policy actions from an old recording. Server request traces assist diagnosis,
but are not joined to exact motor execution timestamps in this first version.
Old two-attempt, half-speed recordings are intentionally rejected by this strict
one-episode experiment. No task improvement or paper-level sample efficiency is
guaranteed by a finite-loss smoke test.

## Offline verification

No hardware is accessed by these commands:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. \
/usr/local/models/sra-expo-ft/venv-jax/bin/python -m pytest tests/test_yam_expo.py -q

CUDA_VISIBLE_DEVICES=0 \
/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/smoke_expo.py \
  --checkpoint /usr/local/models/sra-expo-ft/yam_pi05_jax \
  --tokenizer /home/sra/molmoact2/outputs/models/paligemma-tokenizer \
  --actor-mode expert \
  --output /usr/local/models/sra-expo-ft/offline-expo-tests/NEW_TEST_NAME
```

Smoke targets are synthetic and outputs explicitly marked non-deployable. Never
register them as successful robot replay. Stop a running policy server first.
