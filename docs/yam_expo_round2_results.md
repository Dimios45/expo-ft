# Conservative EXPO round 2 — round-0001 failure replay

Completed using the served stable-v2 version 1 as parent. All work ran on the
4090 host; no robot connection or hardware command was made. The localhost test
server was stopped after verification.

## Inputs and continuity

- `round-0000`: successful prior, 1,391 frames, 47 chunk transitions, reward 1.
- `round-0001`: failure, 3,363 frames, 113 chunk transitions, reward 0, not aborted.
- The failure's experiment ID, policy version, server session and actual weight
  SHA-256 matched the parent checkpoint. Dataset labels were not modified.
- Restored critic/editor/temperature optimizer states and target Q from version 1.
- Fresh pools of 16 base candidates per anchor were generated for both episodes.
  Each training example samples 8 from its pool. Base parameters are frozen.

## Training

Settings remain epsilon .05, masked gripper edits, corrected entropy units,
initial-temperature profile .01, critic/editor/temperature learning rates
1e-4/3e-5/1e-5, effective batch 8, Q10/min2, target tau .005, gamma .99 per tick,
and the conservative update schedule.

This round added 40 critic updates and two editor/temperature updates. Cumulative
counts are 80/4/4. No base action-expert update was performed. Uniform replay drew
86 success-episode windows and 234 failure-episode windows. An additional balanced
terminal auxiliary loss included both terminal windows on every critic step,
with total auxiliary weight .25.

Training-set terminal predictions:

| Episode | Target | Mean Q before | Mean Q after |
|---|---:|---:|---:|
| Success | .747172 | .752171 | .800963 |
| Failure | 0 | -.444600 | -.031283 |

Final ordinary critic loss was .02979 and auxiliary terminal loss .00549.
Fresh candidate preparation took about 485 seconds; training and checkpoint
checks took about 27 seconds. These are not hardware success measurements.

## Checkpoint and HTTP verification

Root:
`/usr/local/models/sra-expo-ft/towel-expo-stable-v2/http-test-40b9aced`

New checkpoint: `versions/0002`. The atomic current pointer now selects version 2;
version 1 remains intact. Its manifest records the complete replay inventory,
including the success prior collected under the earlier experiment. The current
pointer's consumed-episode list tracks this experiment's new failure episode;
the earlier success remains explicitly registered as prior replay.

- Saved checkpoint reload reproduced the selected action exactly.
- 12/12 localhost HTTP requests passed using actual failure-episode observations,
  alternating raw RGB and JPEG. All responses were finite 30-by-14 arrays.
- Health reported version 2 and a weight hash matching its manifest.
- Median/p95 server latency: 1.507/1.553 seconds.
- Maximum sampled residual: .010637, below .05; gripper residuals exactly zero.
- Returned grippers stayed in [.01216, .99349] on these test requests.
- Invalid state shape, task prompt and denoising-step inputs were rejected.
- 19 local tests passed; Ruff passed on changed Python files.

HTTP report:
`http-test-5e9e4507/report.json` under the root above.

## Remaining limits

No held-out task-success evaluation was performed. The failed recording still
shows the first-action omission: 94 of 117 saved requests have exactly 29 matching
recorded actions, indices 1–29. Replay still uses fixed 30-command windows and
nominal timestamps. This round does not repair that client-side alignment issue.
Smaller residuals also do not bound changes caused by choosing another base
candidate. The diagnostic label and explicit network-serving override remain.

## Subsequent rounds

Stop the server before training. Use the actual new dataset directory:

```bash
cd /home/sra/tirth/expo-ft
CUDA_VISIBLE_DEVICES=0 \
JAX_COMPILATION_CACHE_DIR=/usr/local/models/sra-expo-ft/jax-cache \
/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/continue_stable.py \
  prepare \
  --root /usr/local/models/sra-expo-ft/towel-expo-stable-v2/http-test-40b9aced \
  --dataset /home/sra/tirth/expo-ft/round-0002

CUDA_VISIBLE_DEVICES=0 \
JAX_COMPILATION_CACHE_DIR=/usr/local/models/sra-expo-ft/jax-cache \
/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/continue_stable.py \
  train \
  --root /usr/local/models/sra-expo-ft/towel-expo-stable-v2/http-test-40b9aced
```

These are future-round examples; round-0002 has not been collected or trained.
The legacy round.py ingest/train commands reject this profile so that the old
100-editor-update schedule cannot accidentally be used.
