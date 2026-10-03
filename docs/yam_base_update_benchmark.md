# Measuring full base-model updates on the A100

This is a separate, offline training benchmark. It does not restart the closed
online experiment, contact the robot, promote a model, or reuse its critic as a
base-policy optimizer. The input is the original published YAM JAX base plus
successful replay from the concluded online run.

## Scope versus the paper

The [real-time paper, Appendix VII-E](https://arxiv.org/html/2609.18207v1) adapts
the base with successful-episode prefix-conditioned flow matching. Its language
backbone uses LoRA, with other unfrozen components including vision trained too.
It does not update every frozen language weight.

`benchmark_base_update.py` instead measures a **literal all-parameter** update:
all language, vision and action parameters receive gradients. This deliberately
answers the resource question for full-weight training; it is not a reproduction
of the paper's LoRA mask, RTC prefix objective or full actor-critic recipe.

The loss is the existing OpenPI flow-matching loss with training preprocessing
and the checkpoint's 30x32 padded action shape. YAM targets occupy 14 dimensions;
the remaining dimensions are padded. The successful replay's nominal windows
are not precise RTC execution-aligned prefixes. Failure and truncated episodes
are excluded. The optimizer is AdamW with learning rate 2.5e-5, beta1=.9,
beta2=.95, epsilon=1e-8, weight decay=1e-10 and global gradient clipping at 1.
There is no EMA or Polyak actor copy in this resource benchmark.

## Measured three-step run

On the otherwise idle A100 80GB, batch size 1, FP32:

| Quantity | Measurement |
| --- | ---: |
| Trainable parameters | 3,353,433,872 |
| Vision parameters | 414,803,696 |
| Language parameters | 2,508,531,712 |
| Action expert and projections | 430,098,464 |
| Sampled whole-GPU peak | 53,507 MiB / 52.25 GiB |
| Total process wall time | 74.56 s |
| Step 1, including compilation | 29.62 s |
| Step 2, including additional compilation | 28.58 s |
| Step 3, warm | 1.30 s |
| Peak sampled GPU utilization | 100% |

All three component groups had finite, nonzero gradient norms; global update
norms were nonzero. This is evidence of actual base updates, not critic-only
training. The eligible pool contains three successful episodes and 143 replay
windows. The three-step smoke run samples with replacement and is not an epoch
or evidence of robot improvement.

The device was sampled every 200 ms, so shorter memory peaks may be missed.
No serving process was active: this is **training-only** usage, not a concurrent
inference reservation. The mean sampled utilization was 9.1%, dominated by model
loading/compilation and idle host work; it is not steady-state GPU utilization.
One warm step is not a throughput distribution. Batch size, optimizer, precision,
augmentation and concurrent serving all affect memory. Do not extrapolate this
measurement to batch 64 or LoRA without measuring them separately.

Evidence: `artifacts/yam-full-base-benchmark-v2/report.json` and
`logs/yam-full-base-benchmark-v2.{json,csv,log}`. An initial failed attempt hit an
NNX API naming mismatch before any updates; the successful v2 run is the result
reported here. The helper now uses the installed NNX `flat_state()` API.

## Run and measure

From the existing pod workspace, with the installed A100 environment/assets:

```bash
cd /workspace/expo-ft
bash scripts/yam/train_full_base.sh artifacts/yam-full-base-run-001 \
  --steps 10 --batch-size 1 --save-weights
```

Choose a new output path for each run. The script refuses to overwrite its output
directory, unsets the conflicting system `LD_LIBRARY_PATH`, selects CUDA, disables
JAX preallocation and wraps the job with the GPU monitor. The example performs
ten updates; it does not choose a validated training duration for towel folding.

Outputs:

- `artifacts/yam-full-base-run-001/report.json`: configuration, successful replay
  inventory, parameter counts, sample indices, losses, per-component gradient
  norms, update norms and step times.
- `artifacts/yam-full-base-run-001/checkpoint/`: separate Orbax inference weights
  plus the original preprocessing/config assets, only with `--save-weights`.
- `artifacts/yam-full-base-run-001-gpu.csv`: sampled whole-GPU usage.
- `artifacts/yam-full-base-run-001-gpu.json`: exit status, elapsed time, peak
  memory and mean utilization, written when the job exits.

Omit `--save-weights` for a disposable memory benchmark. **Optimizer state is not
exported**; this is not a resumable training checkpoint. Each invocation starts
from the original base configured in `--replay-root`, whose default is
`artifacts/yam-online-base/learner`. The checkpoint export needs approximately
13.4 GB of additional disk space and does not overwrite the source model.
Do not launch multiple full-weight jobs concurrently on this GPU.

To watch memory and utilization in another pod terminal:

```bash
watch -n 1 nvidia-smi
```

A newly trained base changes the action distribution used by the existing EXPO
critic. Validate reloaded outputs and combined-policy behavior before using it
with a previously learned editor/Q checkpoint. These commands never publish the
new weights to a serving registry or start hardware.

## Export/reload validation

A second isolated run repeated three updates with `--save-weights`, completing
in 89.85 seconds including export. Sampled peak memory was 53,511 MiB (52.26 GiB).
The exported checkpoint at `artifacts/yam-full-base-export-check/checkpoint`
loaded through `YamJaxPolicy` and produced finite 30x14 actions on a recorded
observation. `reload-check.json` records that check. No hardware behavior was
validated and the model was not deployed. Optimizer state was not exported.

The disposable `yam-full-base-export-check/checkpoint` was subsequently removed
with the user's approval to free disk for the validated fit below. Its reports
remain; the original model and `yam-full-base-run-001` weights were preserved.

## Episode-held-out full-base fit

`fit_base.py` starts from the original base, trains on successful online episodes
2 and 3 (105 windows), and uses successful episode 4 (38 windows) for validation.
Eight evenly spaced validation windows use fixed noise/time seeds. This is a
model-selection set, **not an independent final test set**. Earlier benchmark
weights were not reused because those runs already sampled all three episodes.

The completed `artifacts/yam-base-fit-001` run used FP32 all-weight AdamW,
microbatch 1, four accumulated gradients per update (effective batch 4), clipping
at 1, five warmup steps, peak LR 5e-6 and cosine decay toward 5e-7. It performed
40 updates / 160 sampled windows with replacement. Nonzero vision, language and
action gradient norms were recorded at every update.

| Quantity | Measurement |
| --- | ---: |
| Trainable parameters | 3,353,433,872 |
| Whole-GPU sampled peak, training only | 66,343 MiB / 64.79 GiB |
| Wall time including load, validation and export | 347.79 s |
| Median update time, excluding first two steps | 5.67 s |
| Mean GPU utilization across entire process | 69.94% |
| Original fixed validation loss | 0.0278011 |
| Best fixed validation loss | 0.0222597 |
| Selected step | 25 |
| Relative validation-loss reduction | 19.93% |

Only the best checkpoint was exported, at
`artifacts/yam-base-fit-001/checkpoint`. Best weights were held in host RAM during
selection, avoiding a second GPU copy. No optimizer state was exported. The
small validation improvement supports a supervised deployment trial; it does
not establish improved towel-folding success. This remains successful-episode
flow-matching fine-tuning, not Q-guided base RL or the paper's RTC/LoRA recipe.

To repeat with a **new** output directory and sufficient free disk:

```bash
cd /workspace/expo-ft
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  OMP_NUM_THREADS=4 .venv-convert/bin/python scripts/yam/measure_gpu.py \
  --output logs/yam-base-fit-002-gpu -- \
  .venv-convert/bin/python scripts/yam/fit_base.py \
  --output artifacts/yam-base-fit-002 --steps 40 --accumulate 4
```

Do not run that concurrently with the deployment test. The reported memory is
training alone; training and inference were measured sequentially, not together.
Evidence is in `artifacts/yam-base-fit-001/report.json` and
`logs/yam-base-fit-001-gpu.{csv,json}`. The accumulation calculation has a CPU
unit test comparing it with the gradient of a full averaged objective.

## Test the fitted base on the NUC

The candidate has its own registry at `artifacts/yam-base-fit-001/deployment`.
Its version **0** means the initial version of this new registry; the checkpoint
is the **fine-tuned** base. `expo_enabled=false` is intentional. The old online
experiment remains stopped, and no learner or automatic upload coordinator is
started for this trial. No robot is moved by the pod validation scripts.

Pod services are in tmux sessions `yam-base-fit` (HTTP 8204) and
`yam-base-fit-ws` (WebSocket 8205). The WS relay uses the same protocol as the
earlier hardware test. Logs are `logs/yam-base-fit-001-serving.log` and
`logs/yam-base-fit-001-ws.log`.

On the NUC, keep this SSH tunnel running in one terminal (reuse an existing
working 8205 tunnel instead of opening a duplicate):

```bash
ssh -N -T -L 127.0.0.1:8205:127.0.0.1:8205 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -p 19032 -i ~/.ssh/id_ed25519 root@154.54.102.50
```

In a second NUC terminal, start the local bridge, or reuse it if already running
with the same 30-second timeout:

```bash
cd ~/karma
~/.venvs/yam-rtc-py310/bin/python rtc_karma_bridge.py \
  --url ws://127.0.0.1:8205 --port 8207 --timeout 30
```

In a third NUC terminal:

```bash
curl --fail http://127.0.0.1:8207/healthz
```

Verify `checkpoint` ends in `/yam-base-fit-001/checkpoint`, `policy_version` is
0, and `expo_enabled` is false. Then, when ready for supervised physical motion,
run one 60-second trial with a fresh output directory:

```bash
cd ~/karma
python3 record_round.py \
  --server http://127.0.0.1:8207 \
  --out ~/yam-expo-data/full-base-fit-001-test-000 \
  --seconds 60 -- \
  --interface left=can_left --interface right=can_right \
  --camera-serial top=348523020354 \
  --camera-serial left_wrist=254623070863 \
  --camera-serial right_wrist=254623070417
```

Use `fold the towel` at the prompt. The existing Karma stop/parking behavior and
wrapper's completed-episode checks apply. A deadline without success should be
labeled reward 0 and `truncated`; use `failure` for an actual task failure. A
finite-output network check does not validate hardware mapping, joint limits,
or task success. No automated ten-episode loop is started here.

To recheck the pod without moving hardware:

```bash
.venv-convert/bin/python scripts/yam/check_base_deployment.py \
  --checkpoint artifacts/yam-base-fit-001/checkpoint \
  --fixture artifacts/yam-rtc/round3-probe.msgpack \
  --output artifacts/yam-base-fit-001/deployment-check.json
.venv-convert/bin/python scripts/yam/rtc_transport.py infer \
  --fixture artifacts/yam-rtc/round3-probe.msgpack \
  --output artifacts/yam-base-fit-001/websocket-check.json
```

To stop this deployment from the pod, send Ctrl-C and let each process exit:

```bash
tmux send-keys -t yam-base-fit C-c
tmux send-keys -t yam-base-fit-ws C-c
```

For a pod restart, initialize the registry only once, then run its existing
`round.py --root artifacts/yam-base-fit-001/deployment serve --host 127.0.0.1
--port 8204` with the CUDA environment above. Run `rtc_transport.py serve
--journal artifacts/yam-base-fit-001/transport.sqlite3 --recorded-inference`
alongside it. Do not point this candidate at the old online learner.

### Deployment verification results

The selected export reloaded, warmed up, and passed five recorded HTTP requests
plus five WebSocket requests, each returning finite 30x14 actions. Policy time
was 409–497 ms across these checks. These requests ran locally on the pod, so
they do not measure NUC WAN latency. Serving startup and these requests peaked
at 13,625 MiB (13.31 GiB), with 13,251 MiB (12.94 GiB) resident afterward.
Training had already exited; no concurrent training/serving total was measured.
This is the saved measurement snapshot from 2026-10-03, not a live status check.
The serving GPU monitor was left running with the server.

Evidence: `artifacts/yam-base-fit-001/deployment-check.json`,
`websocket-check.json`, `deployment-gpu.json` in the same directory, and
`logs/yam-base-fit-001-serving-gpu.csv`. No hardware trial has been performed
for this fitted checkpoint in the evidence reviewed on 2026-10-03.
