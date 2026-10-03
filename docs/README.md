# YAM documentation

The frozen-base A100/YAM experiment concluded on 2026-10-03; its learner remains
stopped. A separate [full-base fit and deployment trial](yam_base_update_benchmark.md)
was verified serving the selected base on ports 8204/8205 on that date; check
health before reuse. Start with the [experiment history](yam_experiment_history.md)
for the complete timeline and measurement ledger, and the [online runbook](yam_online_runbook.md)
for the implemented system and [final results](yam_online_results.md) for evidence.
Nothing in these documents authorizes restarting the closed experiment.

The final run used `artifacts/yam-online-base/{serve,learner}`, started from
base-only version 0 with empty replay, and completed six training rounds through
version 6. Its base weights stayed frozen. `STOPPED.json` blocks coordinator
restart. Checkpoints and recordings are local artifacts, not included in Git.

## Guides and evidence

| Document | Scope |
| --- | --- |
| [Experiment history](yam_experiment_history.md) | Setup, manual and online runs, shutdown, full-base training, GPU/network ledger and artifact preservation |
| [Online runbook](yam_online_runbook.md) | Fresh setup, NUC prompts, automatic uploads, learner queue, policy leases, interruptions and recovery |
| [Online results](yam_online_results.md) | Six episodes/updates, behavior versions, VRAM/timing, timeout and final shutdown |
| [Full base-update benchmark](yam_base_update_benchmark.md) | All-parameter training, episode-held-out checkpoint selection, A100 measurements and NUC deployment test |
| [RTC roadmap](yam_realtime_plan.md) | Agreed local 4090/cloud learner plan, alternatives, paper comparison, bandwidth, weight synchronization and RTC gates |
| [A100 deployment](yam_a100_deployment.md) | Environment/model installation and historical manual round commands |
| [A100 round 1](yam_a100_round1_results.md) | Earlier sequential run's sampling speedup, VRAM and first-round validation |
| [Cloth folding](cloth-folding.MD) | Separate earlier RTX 4090/YAM experiment; different paths and versions |
| [T-shirt / hoodie bootstrap](tshirt-expo.md) | Separate offline recording and bootstrap workflow |
| [Checkpoint conversion](yam_checkpoint_conversion.md) | Offline conversion and model-loading validation |
| [Conversion results](yam_conversion_results.md) | Numerical parity and serving conventions |
| [Stability review](yam_expo_stability_plan.md) | Historical conservative-restart diagnosis and proposal |
| [Initial stability results](yam_expo_stability_results.md) | Earlier offline stable-restart evidence |
| [Round 2 results](yam_expo_round2_results.md) | Earlier failure replay and HTTP validation |
| [Initial runner](yam_expo_rounds.md) | Historical initial EXPO runner; not the current online coordinator |
| [Initial checks](yam_expo_test_results.md) | Historical implementation tests |

Older reports preserve the configuration and evidence at their own checkpoints.
They are not instructions to merge experiment registries or claims of current
service availability. Neither training loss nor numerical validation establishes
physical safety or a controlled improvement in task success.

## Source-control handoff

One commit can include all current source, tests and documentation changes:

```bash
cd /workspace/expo-ft
git add -- README.md docs/ \
  scripts/yam/benchmark_base_update.py \
  scripts/yam/check_base_deployment.py \
  scripts/yam/fit_base.py \
  scripts/yam/train_full_base.sh \
  tests/test_yam_base_fit.py
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Add YAM full-base training and document local-cloud EXPO roadmap" \
  -m "Add measured all-weight training, held-out checkpoint selection and deployment checks. Document online experiment results, GPU memory and network measurements, and the proposed local 4090 inference and cloud learning architecture."
```

Review `git diff --cached` when other work is staged. These commands stage the
current source changes; they do not start services, move the robot, or push Git.
No commit is created by the documentation task itself.

The root `.gitignore` excludes `artifacts/`, `logs/`, `.venv-convert`, the nested
OpenPI checkout and generated caches. Model weights, replay, rollout videos,
tokenizers and raw measurement logs are therefore excluded. HF credentials live
outside the repository. Preserve data/checkpoint backups separately; a source
commit cannot reconstruct the trained policy or restore uploaded videos.
