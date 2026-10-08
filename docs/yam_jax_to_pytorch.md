# YAM EXPO: JAX-to-PyTorch implementation and operating record

Verified against the working tree on branch `pytorch` and saved experiment
`/home/sra/yam-torch/fold70-white-tshirt-001` on 2026-10-08.

This is a native PyTorch adaptation of the existing frozen-base YAM EXPO loop.
It loads the LeRobot `yam_fold_70` checkpoint directly; it does **not** translate
JAX checkpoint tensors into Torch or fine-tune the base VLA weights. What learns
is the critic image encoder, ten Q heads, residual edit policy, and entropy
temperature. Version 0 is base-only inference; subsequent versions add EXPO
candidate editing and selection.

The demonstrated workflow uses the **4090 for inference and training in separate
phases**, with the NUC running KARMA. Episodes are copied with SCP, the server is
stopped, one training round runs, and the server restarts on the new version.
This is not yet the originally proposed uninterrupted three-host deployment.

## What changed in the code

Paths below are relative to the repository root. Function names are included so
references remain useful if line numbers change.

| Existing JAX implementation | Native PyTorch implementation | Translation/adaptation |
|---|---|---|
| `expo_ft/yam/base.py`, conversion loader | [torch/base.py](../expo_ft/yam/torch/base.py), `BasePolicy` | LeRobot strict checkpoint loading, saved pre/postprocessors, native mixed dtypes; no Orbax model loading |
| `yam/learner.py`, `Settings`, `_selection` | [torch/learner.py](../expo_ft/yam/torch/learner.py), `Settings`, `Agent.select_encoded` | Torch modules and tensor operations implement candidate editing and Q selection |
| `yam/stable.py`, `_critic_gradient`, `_editor_gradient` | `Agent.critic_loss`, `critic_update`, `editor_update` | Autograd/backward and Adam replace JAX gradients, Optax and TrainState |
| `jax.jit`, vmap gradient accumulation | `critic_update`, `editor_update` | Eager Torch microbatch accumulation; no Torch compilation dependency |
| Explicit JAX random keys | `Agent.generator`, save/restore | Dedicated Torch generator for crops, residual sampling and Q-head subsampling; separate seeded base-candidate generator |
| `yam/replay.py`, `import_episode` | Same importer plus `Processor.prepare_replay` and `normalize_actions` | Retains validation/reward/window logic while supplying Torch raw RGB/state storage and masked normalization |
| `scripts/yam/continue_stable.py` | [torch/rounds.py](../expo_ft/yam/torch/rounds.py), [train.py](../scripts/yam/torch/train.py) | Initialize → prepare → train; preserve replay inventory, cache hashes, parent version and atomic commit |
| `yam/runtime.py`, `RoundPolicy` | [torch/runtime.py](../expo_ft/yam/torch/runtime.py), `RoundPolicy` | Load base once; read current version; load and warm the Torch selector |
| `serve_jax.py` HTTP application | [torch/server.py](../expo_ft/yam/torch/server.py), `build_app` | Preserve KARMA `/act` JSON/NumPy protocol, task validation, health metadata and inference mutex |
| Flax `expo.msgpack` with training state | `policy.safetensors` + `training.pt` | Separate serving tensors from optimizer/RNG state; training restore uses `weights_only=True` |
| `online_lan.py`, fixed trainer command | [torch/online.py](../scripts/yam/torch/online.py), configurable `Coordinator.trainer_script` | Reuse durable uploads/jobs/leases; new authenticated HTTP checkpoint store and pull installer |
| Existing robot collector | Existing `collect_online.py`, `record_round.py`, `karma_episode.py` | KARMA remains responsible for robot control and cleanup; these were not rewritten as Torch robot drivers |

Additional entry points:

- [serve.py](../scripts/yam/torch/serve.py): standalone server, no coordinator dependency.
- [label_manual.py](../scripts/yam/torch/label_manual.py): explicitly operator-attested labels for direct KARMA rollouts; validates an unambiguous serving version and checkpoint hash.
- [benchmark.py](../scripts/yam/torch/benchmark.py): real-model parity, candidate timing, memory and a synthetic-batch training benchmark.
- [test_actions_q.py](../scripts/yam/torch/test_actions_q.py): action generation and Q fitting on actual labeled recordings; saves diagnostic action/Q arrays, not an online deployment.
- [status.py](../scripts/yam/torch/status.py): authenticated read-only online-coordinator snapshots. It is not the standalone server's status command.
- `start_4090.sh`, `start_3060.sh`, `start_nuc.sh` under `scripts/yam/torch/`: separate online-topology launchers; they are not the commands used for the manual run documented below.

This is objective-level compatibility, not numerical JAX/Torch parity. The
Torch encoder follows the local preactivation GroupNorm residual layout, but
framework initialization/padding and RNG behavior differ. Torch gives each Q
head its own proprioceptive embedding, whereas the JAX ensemble uses the
multiplexer embedding outside the ensemble. The Torch editor MLP also uses
LayerNorm in its hidden layers; the original `EditorNormal` MLP does not request
it. Edited arm candidates are clipped before scoring to agree with the saved
postprocessor bounds. These are real implementation differences, not converted
weights or a bit-for-bit reproduction. Existing JAX versions cannot be restored
into this backend.

## Model, observation and gripper contract

Checkpoint: `/home/sra/ksagar/lerobot/yam_fold_70`.
LeRobot revision: `5f8be1af5380d89ba4f645bc69a9c10c632a8025`.
The checkpoint's [README](/home/sra/ksagar/lerobot/yam_fold_70/README.md) and
`inference.py` define the loading/input contract; the saved experiment contains
SHA256 hashes of the model, config and both processor assets.

The loader preserves BF16 vision/language parameters and the implementation's
FP32 exceptions, including the action expert. It does not call `.half()` or
`.bfloat16()` on the whole policy. All base parameters have gradients disabled.
The Q/editor modules train in FP32. No full-base optimizer is created.

The HTTP observation contains `top_cam`, `left_cam`, `right_cam`, a 14-value
state and `instruction`. Images are decoded to RGB and mapped to LeRobot keys
`observation.images.top`, `.left`, `.right`. State/actions are left arm joints
0–5, left gripper, right arm joints 0–5, right gripper. Outputs are **30×14
absolute joint targets**, with arm joints in radians.

Live grippers are identity/pass-through: channels **6 and 13**, **1=open**.
Saved fold-70 masks exclude grippers from quantile normalization and EXPO edits.
Arm normalization uses the exported mixture statistics, not the original HF
base statistics. Conversely, deployed KARMA recordings use **0=open** in their
dataset columns: `dataset_to_wire` in `yam/replay.py` reverses that conversion
once on replay import. It must not be applied to live server inputs or outputs.
The source is KARMA `src/openpi_control/record.py`, `to_dataset_gripper` and
`to_native_gripper`, inspected on yambox during implementation.

## Action generation and selection

`BasePolicy.candidates` uses the saved preprocessor and ten continuous flow
integration steps. For multiple samples it computes the frozen VLM prefix KV
states once for that observation, then batches the action-expert samples.
This cache is not carried across observations. The pinned HF
`generate_actions_from_inputs` API accepts those KV states. The cached path
reproduces the uncached path's encoder attention mask.

Version 0 generates one base chunk. Versions 1+ generate eight base chunks and
use the learned editor to make eight additional candidates. For each candidate:

1. The critic encoder processes three resized RGB views as nine channels, with
   a 512-dimensional visual embedding and proprioceptive features.
2. The editor samples `delta = 0.05 * tanh(mean + exp(logstd) * noise)`;
   gripper deltas are zero. The chunk is flattened to 420 values for the heads.
3. The target Q ensemble scores all 16 candidates. Two of its ten heads are
   randomly selected; each candidate's score is their minimum.
4. The highest-scoring candidate is selected and the saved postprocessor returns
   physical joint targets. Live gripper outputs are bounded to [0,1].

This means inference is stochastic even with a fixed checkpoint. Selection uses
**target Q**, not the mean of all ten online critic heads. Diagnostic
`base_q_ensemble` values and `selection_scores` therefore need not rank actions
identically. Source: `Agent.edit`, `select_encoded`, and `RoundPolicy.predict`.

## Q gradients, editor gradients and rewards

Replay contains actual **recorded executed commands**, not substituted model
predictions. Each transition uses 30 commands and a real next recorded state.
The final window can overlap the preceding window to retain the episode end.
The importer checks one saved episode, task, 30 Hz nominal timestamps, speed 1,
chunk size 30, finite arrays and explicit human labels.

For chunk reward `R`, bootstrap mask `m`, and chosen next chunk `a_next`:

```text
y = R + 0.99^30 * m * min(Q_target_i(s_next, a_next), Q_target_j(s_next, a_next))
L_Q = mean over batch and 10 heads of (Q_k(s, recorded_action_chunk) - y)^2
```

The complete target path runs under `torch.no_grad()`. Backward updates only the
**current critic encoder and online Q heads**. There is no separate target image
encoder, no entropy bonus in the Q target, and no VLA gradient. Each Q update
adds a terminal auxiliary loss weighted by 0.25 when terminal examples exist.
Eight microbatches of size one form the default effective batch of eight;
Adam steps only after gradients have accumulated. Gradient norm is clipped to
1; the logged `critic_grad_norm` is the norm returned before clipping.
Target Q parameters then update as `target = 0.995*target + 0.005*online`.

A terminal success has mask 0 and final-window reward `0.99^29 ≈ 0.747172`.
A definite failure has mask 0 and reward 0. An unfinished time-limit truncation
has reward 0 and mask 1, allowing bootstrap. This is why the outcome question
is not interchangeable with the binary success question. Mixed terminal-batch
means such as 0.373586 reflect sampled success/failure composition.

Every 20 critic updates, the editor maximizes mean online-Q value with an
entropy term. Encoder features are detached and Q parameters temporarily have
gradients disabled, while gradients through Q's **action input** still reach
the editor. The temperature is trained separately. The implementation uses the
unscaled tanh-density entropy coordinates, excludes masked gripper dimensions,
and uses target entropy `-(30*12)/2 = -180`. Negative differential entropy is
possible and is not by itself a failure. Source: `Agent.editor_update` and the
JAX reference `StableEXPO._editor_gradient`.

Defaults: critic LR 1e-4, editor LR 3e-5, temperature LR 1e-5, initial temperature
0.01, initial editor logstd -3, 40 critic updates and two editor updates per
round. Current/next images receive independent per-view/per-example 95% crops.
This does not implement the papers' full augmentation or delayed-prefix RTC.

## Preparation, persistence and deployment

`train.py prepare` verifies recorded session identity and hashes, imports new
replay, and computes eight frozen-base candidate chunks per required anchor.
Verified older replay/candidate caches are reused. Raw RGB and raw state are
retained for VLA inputs; normalized state/actions serve the RL network. The
trainable critic image embedding is always recomputed. Preparation loads the
large base; `train.py train` is a separate process without that base allocation.

Training restores the previous encoder/Q/editor/temperature, Adam states,
Torch sampling RNG and cumulative update count. It samples all retained replay,
validates the candidate checkpoint, and atomically advances `current.json`.

```text
fold70-white-tshirt-001/
  experiment.json             # pinned prompt, settings, model/processor hashes
  current.json                # currently committed version
  sessions/<id>/session.json  # serving policy identity
  replay/<id>/                # normalized actions/state, RGB and candidate cache
  pending.json                # exists only while a round is pending
  versions/0001/ or 0002/
    policy.safetensors        # encoder, online/target Q, editor, temperature, mask
    training.pt               # optimizers, RNG, cumulative updates
    metrics.jsonl             # losses, Q/target means, gradient norms, edits
    manifest.json             # version, replay inventory and checksums
```

The serving file is **170,236,060 bytes**, about 170.24 MB including the
safetensors header. The frozen 12 GB base and optimizer state are not sent on
each online weight update. Standalone `serve.py` reads `current.json` at startup
and holds the run lock; stop it before prepare/train, then restart to load the
new version. It has no configured candidate store, so its reload endpoint does
not automatically pull a newer training version.

The separate `online.py` mode implements authenticated resumable uploads, a
single collector lease, a serial worker, HTTP candidate retrieval, hash-checked
staging, candidate warmup and atomic agent replacement under the inference
mutex. Failed loads retain the old agent. This infrastructure exists, but the
following real experiments used manual SCP/restart rather than proving the
whole asynchronous topology or a no-idle five-minute deadline.

## Commands for the tested manual workflow

On the 4090, start inference (current wired address at this writing is
`192.168.0.130`; recheck it if DHCP changes):

```bash
cd /home/sra/tirth/expo-ft
/home/sra/ksagar/lerobot/.venv/bin/python -u scripts/yam/torch/serve.py \
  --checkpoint /home/sra/ksagar/lerobot/yam_fold_70 \
  --root /home/sra/yam-torch/fold70-white-tshirt-001 \
  --prompt "fold the white tshirt" --host 192.168.0.130 --port 8202
```

Wait for `Ready` and check the version. Health is read-only:
`curl http://192.168.0.130:8202/healthz`.
The experiment root pins the task; a different task requires a new root.

On the NUC, only when the operator is ready, collect a fresh episode:

```bash
cd ~/karma
uv run karma rollout \
  --rig yam_bimanual --server http://192.168.0.130:8202 \
  --root ~/yam-expo-data/expo-ft-pytorch-03 --repo-id local/expo-ft-pytorch-03 \
  --episodes 1 --episode-seconds 300 \
  --fps 30 --speed 1 --chunk-size 30 --num-steps 10 --no-prefetch \
  --interface left=can_left --interface right=can_right \
  --camera-serial top=348523020354 \
  --camera-serial left_wrist=254623070863 \
  --camera-serial right_wrist=254623070417
```

Use the exact task `fold the white tshirt`. The serials are the established ASIC
role mapping. KARMA handles preflight, user prompts, controller limits and
cleanup; the server has no CAN/robot-control code. Ctrl+C should be sent to
KARMA for normal cleanup; wait for parking/de-energizing and outcome prompts.
Closing a tmux client does not stop an active robot episode.

After recording, push from NUC to 4090:

```bash
ssh sra@192.168.0.130 'mkdir -p ~/yam-expo-data'
scp -r ~/yam-expo-data/expo-ft-pytorch-03 sra@192.168.0.130:~/yam-expo-data/
```

Stop the server before using the same run root for training. Label with the
**actual observed outcome** and behavior version. For example, only if episode
3 was a definite failure collected under version 2:

```bash
cd /home/sra/tirth/expo-ft
python3 scripts/yam/torch/label_manual.py \
  --dataset /home/sra/yam-expo-data/expo-ft-pytorch-03 \
  --root /home/sra/yam-torch/fold70-white-tshirt-001 \
  --policy-version 2 --reward 0 --terminal failure
```

Use `--reward 1 --terminal success` for success, or `--reward 0 --terminal
truncated` for an unfinished time limit. Manual labels explicitly record
`boundary_verified=false` and retrospective operator attestation; they do not
pretend the direct rollout had an online lease. Ambiguous serving sessions,
conflicting existing labels, or changed weight hashes are rejected.

```bash
/home/sra/ksagar/lerobot/.venv/bin/python -u scripts/yam/torch/train.py prepare \
  --root /home/sra/yam-torch/fold70-white-tshirt-001 \
  --dataset /home/sra/yam-expo-data/expo-ft-pytorch-03 \
&& \
/home/sra/ksagar/lerobot/.venv/bin/python -u scripts/yam/torch/train.py train \
  --root /home/sra/yam-torch/fold70-white-tshirt-001 --microbatch 1
```

Restart with the same server command after training succeeds. Do not reuse a
previous recording directory for the next rollout. The label helper requires
one unambiguous session for the requested behavior version; restarting that
version multiple times requires inspecting history rather than guessing.

## What has actually been tested

| Artifact/run | Evidence and interpretation |
|---|---|
| Real-model benchmark | [Report](benchmarks/yam-fold70-pytorch-4090.json): one warm base sample ~0.316 s; eight uncached sequential samples ~2.438 s; shared-prefix sequential ~1.925 s; shared-prefix batch-eight ~0.316 s. Small-sample timings, not p95 or a controller-rate guarantee |
| Precision/contract check | Export's observation constructor and normalized public sampler agreed with the adapter; sequential prefix reuse had maximum absolute difference 0 for the measured seed/input. This is not JAX model parity |
| Offline 16-transition diagnostic | `artifacts/torch-actions-q-20261008-184443/report.json`: 40 updates, TD MSE 0.162159 → 0.003324, terminal MSE 0.701832 → 0.002380, total 138.766 s. In-sample fit on historical data, not deployment |
| Episode 1, `expo-ft-pytorch-01` | Base version 0, 6,203 frames, 207 imported windows, operator success=1. Created version 1 at cumulative critic steps 1–40, two editor updates. Update loop 11.852 s; round manifest 12.435 s excluding preparation |
| Episode 2, `expo-ft-pytorch-02` | Behavior version 1, 6,235 frames, 208 windows, operator-confirmed terminal failure=0. Retained episode 1, created version 2 at steps 41–80, two more editor updates. Update loop 11.582 s; round manifest 12.153 s excluding preparation |
| Committed state | Version 2, two replay episodes; manifests/checksums verified, no pending round. Version-2 robot outcome is not yet documented |

The second rollout failing after the first succeeded does not establish that
EXPO helped or harmed performance; there is no controlled repeated evaluation.
Low fitting loss alone does not establish action ranking quality. Negative
nonterminal Q estimates are possible with an unconstrained early critic and
bootstrapped targets; Q values are not calibrated success probabilities.

Tests in [test_yam_torch.py](../tests/test_yam_torch.py) cover gradient/optimizer
resume, terminal versus truncated targets, gripper masking, accumulation,
HTTP authentication, atomic corrupt-transfer rejection, old-policy retention
on reload failure, two-round replay continuation, subprocess coordinator
startup, the diagnostic runner and dependency checks. Earlier validation also
passed 38 existing JAX/collection tests on CPU. The initial JAX CUDA
sequential/vmap gradient tolerance test failed and is documented in the
[runbook](yam_pytorch.md); no claim of JAX GPU equivalence is made here.

## Runtime fixes and remaining limits

- Missing `transformers` was an environment-extra issue. Installed
  `transformers==5.5.4`, PEFT and SciPy in the launcher Python, then verified a
  real shortened action/Q run. `check_model_dependencies` now fails before
  episode decoding. Requirements record these dependencies. A LeRobot
  `uv sync` should include its `molmoact2` extra.
- Prompt mismatch caused HTTP 400 when the server pinned `fold the towel` but
  KARMA requested `fold the white tshirt`. A new experiment root with the new
  exact task resolved it; matching is intentionally strict.
- The initial manual label helper accepted only version 0. It now accepts an
  explicit behavior version, verifies the corresponding serving/checkpoint
  identity, and refuses contradictory KARMA success labels or overwrite.
- The 3060 memory issue remains: measured base allocation was ~12.06 GiB and
  batch-eight peak ~12.86 GiB on the 4090, before all deployment overhead. The
  unchanged export is not a validated 12 GB deployment. The online inference
  launcher has a minimum-memory check; the standalone server relies on its
  actual load/warmup and is intended here for the 4090.
- Full-base/LoRA learning, 3060 offload/quantization, RTC prefix inpainting,
  uninterrupted concurrent inference/training on this hardware, and a measured
  five-minute end-to-end update deadline remain outside the demonstrated path.
  Training's ~12 seconds excludes model loading, video import, candidate cache
  generation, transfers and serving warmup.

The algorithm references and pinned upstream source locations are recorded in
[the PyTorch runbook](yam_pytorch.md#contract-and-source-references), alongside
the distinction between the frozen-base local profile and the EXPO-FT papers.
