# YAM EXPO offline implementation checks — 2026-09-29

The production experiment `/usr/local/models/sra-expo-ft/towel-expo` is initialized
at version 0, with no replay episodes and no learned updates deployed. The base
checkpoint is unchanged. No robot/NUC connection or motor command was made.

## Real RTX 4090 checks

Using FP32 π₀.₅, 8 sequential base candidates, 8 editor candidates, all 10 Q
networks, the full (3,4,6,3) ResNet encoder and a 420-coordinate action chunk:

- Generated `(8,30,14)` base candidates from the real checkpoint.
- Performed two critic/encoder/editor/temperature updates, with finite losses and
  nonzero gradients, and soft target-Q updates.
- Performed a success-only flow-matching update to 564,063,392 action-expert and
  action/time-projection parameters. The remaining 2,789,370,480 parameters were
  frozen. The update changed fixed-noise action predictions.
- Saved all learned state. A new process restored the expert, critics, editor,
  target Q, and optimizer states; it started at RL update count 2 and performed
  two more RL updates and another expert update successfully.
- Served the reloaded learned policy over localhost through the existing Karma
  HTTP protocol. Received finite `(30,14)` actions. The saved request trace
  contained `(10,1,16)` Q values, 16 full action candidates and two distinct critic
  indices. One warmed request took about 1.47 seconds. This is not a latency
  benchmark or a 30 Hz inference claim; sequential sampling introduces pauses.
- Verified the round-0 server retains the original no-inversion behavior and
  advertises the correct experiment/version identity.

The initial expert update exceeded GPU memory when old/new parameters and Adam
buffers coexisted. Explicit JAX buffer donation for the expert and optimizer
resolved this. The successful run observed about 18,140 MiB GPU allocation after
an update; peak transient allocation was not measured. Full-model fine-tuning,
larger batches and concurrent training/serving were not tested or claimed.

These were **synthetic numerical/memory tests**, not robot demonstrations or
successful task episodes. Their checkpoints are stored separately under
`/usr/local/models/sra-expo-ft/offline-expo-tests` and are not published into the
production experiment. Their decreasing synthetic losses are not evidence of
better towel folding.

Detailed reports:

- `offline-expo-tests/real-full-expert-save/report.json`
- `offline-expo-tests/real-full-expert-resumed/report.json`
- `offline-expo-tests/serve-reload-test/http_test.json`

## CPU regression checks

12 tests passed across the EXPO, conversion and HTTP suites. Checks cover:

- terminal/truncated Bellman masking and chunk discount;
- valid 30-command windows with a real next observation;
- a synthetic LeRobot v3 parquet/video import with explicit dataset-frame mapping;
- editor bounds and tanh-Gaussian scale/Jacobian;
- ten critics, two distinct sampled critics, actor/critic gradients and Polyak update;
- optimizer and NNX expert-state serialization;
- exact continuation after RL state reload;
- two sequential immutable version publications and exclusive round locking;
- stale episode rejection and checkpoint hash integrity;
- the working state/action pass-through HTTP contract.

Ruff checks passed for the new runner, collector, adapter and tests.

## Remaining experimental work

The operator must collect the first real versioned episode, transfer it, train,
and evaluate the next policy. That physical loop has not been run. The initial
single-GPU adaptation uses microbatch 1, no image augmentation, sequential VLA
candidate generation and optional expert-only flow training instead of the
upstream LoRA recipe. See `yam_expo_rounds.md` for all execution/data assumptions.
