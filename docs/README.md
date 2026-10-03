# YAM documentation

For the current A100 pod, start with [A100 deployment and collection](yam_a100_deployment.md).
The active experiment is `/workspace/expo-ft/artifacts/yam-expo`; its
`current.json` and version manifests are authoritative. Stable EXPO trains the
critic, visual encoder, action editor, and temperature while preserving the base.

[Real-time YAM roadmap](yam_realtime_plan.md) compares the two papers with the
current implementation and specifies the NUC/A100 streaming, latency, scheduling,
and replay work still needed. It is a design, not an already deployed live learner.

[Cloth folding on the RTX 4090 and YAM NUC](cloth-folding.MD) documents a separate
earlier experiment whose version numbers and machine paths do not apply to the
A100 run. Do not mix the two experiment registries.

## Guides and evidence

For the separate hoodie recording, see [T-shirt / hoodie bootstrap](tshirt-expo.md).

| Document | Purpose |
| --- | --- |
| [A100 deployment](yam_a100_deployment.md) | Current pod, SSH tunnel, recording, transfer, train, and restart commands |
| [A100 round 1](yam_a100_round1_results.md) | Measured parallel sampling speedup, VRAM, and first-round validation |
| [Real-time roadmap](yam_realtime_plan.md) | Paper comparison, WebSocket design, delayed action execution, live replay, and one-GPU scheduling |
| [Cloth-folding guide](cloth-folding.MD) | Earlier 4090 experiment, training equations, code map, and limitations |
| [Checkpoint conversion](yam_checkpoint_conversion.md) | Offline weight conversion and model-loading validation |
| [Conversion results](yam_conversion_results.md) | Recorded numerical parity checks and serving conventions |
| [Stability review](yam_expo_stability_plan.md) | Historical diagnosis and proposal for the conservative restart |
| [Initial stability results](yam_expo_stability_results.md) | Offline evidence from the stable restart using `round-0000` |
| [Round 2 results](yam_expo_round2_results.md) | Failure replay, continuity checks, and HTTP validation for version 2 |
| [Initial runner guide](yam_expo_rounds.md) | Historical workflow; use `continue_stable.py` from the cloth-folding guide for current training |
| [Initial implementation checks](yam_expo_test_results.md) | Historical checks before the stable workflow; not current deployment status |

Reports preserve what was measured at their respective checkpoints. They do not
establish a controlled success rate or guarantee that later checkpoints behave
the same way.

## Commit the implementation and documentation

For the A100 changes, stage only the implementation, tests, and documentation
listed below. These commands do not run training or connect to hardware.

```bash
cd /workspace/expo-ft

git add -- \
  README.md docs/README.md \
  docs/yam_a100_deployment.md docs/yam_a100_round1_results.md docs/yam_realtime_plan.md \
  expo_ft/yam/base.py expo_ft/yam/rounds.py expo_ft/yam/stable.py \
  scripts/yam/continue_stable.py scripts/yam/check_http.py \
  scripts/yam/download_pod_assets.sh scripts/yam/measure_gpu.py \
  scripts/yam/pod_expo.sh scripts/yam/run_pod_server.sh \
  tests/test_yam_stability.py tests/test_yam_published_checkpoint.py

git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
```

Review the staged files, then commit if they are the intended changes:

```bash
git commit -m "Add A100 YAM EXPO workflow, parallel training, and realtime roadmap"
```

The root `.gitignore` excludes `artifacts/`, `logs/`, `.venv-convert`, the nested
OpenPI checkout, and generated caches. Thus model weights, replay, rollout videos,
tokenizers, and measured run logs are not included. HF credentials are outside
this repository and are not staged by these commands. Preserve trained versions
and data separately: a source commit is not a model or dataset backup.
