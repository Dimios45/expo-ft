# YAM JAX on the A100 pod, KARMA on the NUC

This guide covers model/environment installation and the **earlier manual**
record → SCP → train → restart workflow. For automatic episode uploads,
concurrent learning and boundary policy updates, use the
[online runbook](yam_online_runbook.md).

The 2026-10-03 online experiment concluded and its services were stopped.
A subsequent separate [full-base deployment trial](yam_base_update_benchmark.md)
was verified on ports 8204/8205; check health before reuse.
Its final registry is `artifacts/yam-online-base/learner`, ending at version 6;
[the results report](yam_online_results.md) records its outcomes and measurements.
The `artifacts/yam-expo` paths below refer to the earlier sequential experiment,
not that fresh online run. Their version numbers must not be mixed.
Initialization commands are only for a new directory. In the manual workflow,
serving and training hold the same exclusive experiment lock and cannot overlap.
The online workflow uses separate serving/learner registries instead.

## One-episode EXPO workflow

After HF access is approved and `bash scripts/yam/download_pod_assets.sh` has
completed, initialize once on the pod:

```bash
cd /workspace/expo-ft
bash scripts/yam/pod_expo.sh init
mkdir -p artifacts/incoming logs
tmux new-session -d -s yam-expo \
  'cd /workspace/expo-ft && bash scripts/yam/pod_expo.sh serve >> logs/yam-expo-server.log 2>&1'
tail -f logs/yam-expo-server.log
```

The default prompt is `fold the towel`; set `YAM_PROMPT='your exact task'`
before `init` to change it. Default training is stable-v2: frozen base, learned
EXPO critic/encoder/editor/temperature, edit scale 0.05, gripper edits masked.
For the historical action-expert update path, use
`YAM_ACTOR_MODE=expert bash scripts/yam/pod_expo.sh init` **instead of** the
default init. That path updates the base action expert only on successful
replay and uses the older learning schedule. The choice is fixed at init;
do not edit an existing experiment's settings to switch modes. Use
`YAM_EXPERIMENT=/absolute/new/path` for a separate experiment.

Wait for `Warmup passed. Serving version 0`. Do not run the standalone server
on the same port. Open the NUC SSH tunnel shown below and verify health;
the response must include `experiment_id`, `policy_version: 0`, and `session_id`.

On the **NUC**, copy the recording wrapper:

```bash
cd ~/karma
scp -P 19032 -i ~/.ssh/id_ed25519 \
  root@154.54.102.50:/workspace/expo-ft/scripts/yam/record_round.py ./record_round.py
```

After verifying both CAN interfaces and the three cameras, record exactly one
episode (**starts robot motion**). Replace `LEFT_CAN` and the serials:

```bash
cd ~/karma
python3 record_round.py \
  --server http://127.0.0.1:8204 \
  --out "$HOME/yam-expo-data/round-0000" --seconds 300 -- \
  --interface left=LEFT_CAN --interface right=can_right \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL
```

Enter the exact configured task when prompted. After the rollout, answer the
wrapper's actual success reward (`0` or `1`) and terminal-reason questions.
Keep the server alive until `expo_session.json` is saved. The wrapper fixes
30 Hz, speed 1, 30-action chunks, 10 solver steps, and no prefetch.

Transfer the **entire directory**, including videos and metadata, from the NUC:

```bash
ssh -p 19032 -i ~/.ssh/id_ed25519 root@154.54.102.50 \
  'mkdir -p /workspace/expo-ft/artifacts/incoming'
scp -r -P 19032 -i ~/.ssh/id_ed25519 \
  "$HOME/yam-expo-data/round-0000" \
  root@154.54.102.50:/workspace/expo-ft/artifacts/incoming/
```

On the **pod**, after recording and transfer finish, stop the round server
(`tmux attach -t yam-expo`, then Ctrl+C). Serving and training are exclusive.
Prepare replay and train in a persistent session:

```bash
cd /workspace/expo-ft
tmux new-session -d -s yam-train \
  'cd /workspace/expo-ft && { bash scripts/yam/pod_expo.sh prepare --dataset /workspace/expo-ft/artifacts/incoming/round-0000 && bash scripts/yam/pod_expo.sh train; } > logs/yam-expo-train.log 2>&1'
tail -f logs/yam-expo-train.log
```

Stable training reports `ROUND COMPLETE`; expert mode reports
`READY TO DEPLOY`. A completed version is not evidence of improved task success.
Inspect its identity with `bash scripts/yam/pod_expo.sh status`.

The A100 wrapper uses four parallel base candidates during preparation/serving and
eight parallel gradient microbatches during stable training. The effective
batch remains eight; random keys, update counts, learning rates, FP32 precision,
and frozen-base behavior are preserved. Set `YAM_CANDIDATE_BATCH=1` and
`YAM_TRAIN_MICROBATCH=1` to use sequential execution on a smaller GPU.
`scripts/yam/measure_gpu.py --output logs/NAME -- COMMAND ...` records GPU 0
memory/utilization every 200 ms to CSV plus a JSON summary. These are sampled
whole-device measurements; they can miss brief spikes.

For stable mode, check the first trained candidate offline before serving it:

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  .venv-convert/bin/python scripts/yam/smoke_candidate_server.py \
  --root artifacts/yam-expo \
  --candidate artifacts/yam-expo/versions/0001 \
  --dataset artifacts/incoming/round-0000
```

After that check passes, restart the same `pod_expo.sh serve` command. It loads
the current version automatically. Use a new output directory (`round-0001`)
for the next collection. Keep the original base, experiment registry, sessions,
replay, and version folders; the EXPO policy depends on all its referenced assets.

The GPU model runs in `/workspace/expo-ft` on the A100. KARMA runs in
`~/karma` on `yambox@yambox-GEM12`, beside the CAN adapters and cameras.
The NUC has Ethernet `192.168.0.165` and Wi-Fi `192.168.0.200`. Neither is
directly reachable from this pod. An SSH local forward initiated on the NUC
connects its localhost port 8204 to the model server's localhost port 8204.

## Pod environment

The isolated `.venv-convert` uses Python 3.11, the pinned CUDA 12 JAX stack in
`scripts/yam/requirements-convert.txt`, HTTP dependencies in
`scripts/yam/requirements-serve.txt`, and CPU `torch==2.7.1` for preprocessing.
OpenPI is at revision `2abe46282bfdf9f1bc0240f3f9960ec175d1b4a8` under
`expo_ft/agents/vla/openpi`. The checkpoint is in `artifacts/yam_pi05_jax`.
This serving environment does not require the full EXPO training environment.

For a fresh checkout on a CUDA-capable Linux host with `uv`, install this YAM
environment before downloading assets or initializing an experiment:

```bash
cd /workspace/expo-ft
uv venv --python 3.11 .venv-convert
uv pip install --python .venv-convert/bin/python \
  -r scripts/yam/requirements-convert.txt \
  -r scripts/yam/requirements-serve.txt \
  -r scripts/yam/requirements-expo.txt huggingface_hub
uv pip install --python .venv-convert/bin/python \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/pd-perry/openpi.git expo_ft/agents/vla/openpi
git -C expo_ft/agents/vla/openpi checkout 2abe46282bfdf9f1bc0240f3f9960ec175d1b4a8
```

This setup is already installed on the current pod. Do not recreate the active
environment or overwrite an existing OpenPI checkout. The pinned requirements
cover the main framework versions; transitive dependency resolution may change.
Use the GPU/HTTP checks below to validate a fresh installation.

Setup verification on this A100: all 38 checkpoint files matched the Hub's
reported sizes; the existing server/conversion tests passed (4 tests); package
compatibility checks passed. Restoring the real weights and sampling with
synthetic image/state/token tensors produced finite `(1,30,32)` padded outputs
on the GPU, with a second sample taking about 0.36 seconds. This excludes real
tokenization, image preprocessing, HTTP, and network transfer. The initial GPU
probe segfaulted; the probe using the server's allocator/thread settings and
an unset `LD_LIBRARY_PATH` passed. The launcher unsets that path to use JAX's
installed CUDA libraries. Full tokenizer-to-HTTP inference subsequently passed
after access was granted. See `logs/yam-gpu-check.log` and `logs/yam-http-check.log`.

The tokenizer is already downloaded on this pod. For a fresh setup, log in with
an HF account approved for `google/paligemma-3b-pt-224`:

```bash
cd /workspace/expo-ft
.venv-convert/bin/hf auth login
bash scripts/yam/download_pod_assets.sh
```

Do not paste tokens into chat or commit them. If the tokenizer already exists
elsewhere, set `YAM_TOKENIZER=/absolute/path/to/paligemma-tokenizer` instead.

Start the model in a persistent tmux session:

```bash
cd /workspace/expo-ft
mkdir -p logs
tmux new-session -d -s yam-server \
  'cd /workspace/expo-ft && bash scripts/yam/run_pod_server.sh >> logs/yam-server.log 2>&1'
tail -f logs/yam-server.log
```

Wait for `Warmup passed` and `Uvicorn running`. Initial JAX compilation can
take several minutes. Ctrl+C exits `tail`; the tmux server keeps running.
Use a single server process. To inspect it use `tmux attach -t yam-server`;
Ctrl+B then D detaches, while Ctrl+C in that session stops the server.
To run in the foreground instead, use `bash scripts/yam/run_pod_server.sh`.

```bash
curl --fail http://127.0.0.1:8204/healthz
.venv-convert/bin/python scripts/yam/check_http.py
```

The HTTP check sends synthetic raw RGB and JPEG observations and checks for
finite `(30,14)` action chunks. It does not communicate with a robot.

## NUC connection

Run this in a **NUC terminal**, keeping it open:

```bash
ssh -N -T \
  -L 127.0.0.1:8204:127.0.0.1:8204 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -p 19032 -i ~/.ssh/id_ed25519 root@154.54.102.50
```

In another NUC terminal:

```bash
curl --fail http://127.0.0.1:8204/healthz
```

No public HTTP port is needed. If NUC port 8204 is occupied, use
`-L 127.0.0.1:18204:127.0.0.1:8204` and use port 18204 in NUC client URLs.
If RunPod changes the external SSH address or port after a restart, update
the SSH command accordingly.

## NUC client environment and hardware

The NUC already has `~/karma`. Its installation cannot be performed from this
pod without a route and SSH access to the NUC. If it is not already installed,
run KARMA's documented installation on the NUC:

```bash
cd ~/karma
sudo ./scripts/install_build_deps_ubuntu.sh
./scripts/build_deps.sh
UV_HTTP_TIMEOUT=180 uv sync --locked
uv run karma --help
```

This checkpoint needs two YAM arms and three views. The supplied `ip a` output
shows only `can_right`; connect and identify the left adapter before proceeding.
Do not map both arms to the same bus. The example below assumes an actual
`can_left` interface has been configured; replace it with the verified name.
Keep the checkpoint's existing joint/gripper mapping; this server reports
`hardware_mapping_verified: false`, so an HTTP success alone does not verify
physical joint direction or gripper polarity.

Read-only preflight and camera discovery on the NUC:

```bash
cd ~/karma
ip -details link show type can
uv run karma doctor --rig yam_bimanual \
  --interface left=can_left --interface right=can_right
uv run python -c 'import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()])'
uv run karma cameras --rig yam_bimanual \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL --probe
```

Replace the three serial placeholders with the actual cameras. Once the two
arms, camera roles, and checkpoint frame are verified, this command **starts
robot motion**:

```bash
cd ~/karma
uv run karma inference --rig yam_bimanual \
  --interface left=can_left --interface right=can_right \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --server http://127.0.0.1:8204 \
  --norm-tag yam_dual_molmoact2 \
  --instruction "fold the towel"
```

The client and server use 10 sampling steps, 14 state/action coordinates,
three cameras, and 30-action chunks. KARMA adds `/act` to the server URL.
Actual task performance and network latency must be checked on the robot;
the synthetic server check does not establish either.

Sources: [checkpoint setup](https://huggingface.co/sra-vjti/molmoact2-yam-pi05-jax),
[KARMA installation](https://github.com/SRA-VJTI/karma), and
[KARMA inference contract](https://github.com/SRA-VJTI/karma/blob/master/docs/inference.md).
