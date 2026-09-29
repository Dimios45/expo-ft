# Offline stability experiment: base + round-0000

Completed 2026-09-29. No robot connection, hardware commands, model server, or
checkpoint publication. Original base and towel-expo/versions/0001 are unchanged.

## Artifacts

Experiment: `/usr/local/models/sra-expo-ft/towel-expo-stable-v2`

- `ablation_report.json`, `ablations.npz`: original base versus old expert,
  old critic ranking, old edits, and complete old version 1.
- `candidate/expo.msgpack`: corrected critic/editor/temperature checkpoint.
- `candidate/manifest.json`, `candidate/metrics.jsonl`: settings, provenance,
  losses, explicit sampled transition IDs, reward coverage, and hashes.
- `stable_ablation_report.json`, `stable_ablations.npz`: corrected candidate
  versus the same base samples and recorded observations.
- `execution_audit.json`, `state_alignment_audit.json`: recorder diagnostics.
- `base_candidates.npz`, `cache_manifest.json`: 16 fixed-noise base samples per
  observation. Subsets of 8 are used in training; this finite-pool approximation
  is explicitly restricted to a frozen base and must refresh next round.

The experiment is diagnostic-only. `current.json` remains version 0 with no
published checkpoint; the learned checkpoint is in `candidate/`, not `versions/`.

## What changed and ran

Edit scale .05; base frozen; gripper residuals masked; initial editor log std -3;
critic LR 1e-4; editor LR 3e-5; temperature LR 1e-5; initial temperature .01.
Entropy uses unscaled edit coordinates and excludes grippers (360 active
dimensions, target -180). Per-component gradient clipping is 1.0. Independent
95% image crops are applied to critic/editor inputs. Gamma remains .99 per tick
for this controlled comparison; Q10/min2 and target tau .005 remain unchanged.

The fresh learner completed 40 critic optimizer steps with eight-example
gradient accumulation, followed by an editor/temperature update every 20 critic
steps: two updates each. There were zero base updates. Uniform replay sampled
the terminal six times; an additional terminal loss with weight .25 included
the terminal in all 40 critic steps. This is not a reproduction of paper UTD.

Training plus checkpoint checks took approximately 48 seconds, excluding base
candidate generation and ablations. Frozen-base candidate preparation and old
policy ablations took approximately 175 seconds. The final terminal predicted
Q was .7522 against its .7472 target. Early-state Q remained negative, e.g.
-.6979 on the first transition: numerical improvement and terminal fitting do
not establish a useful calibrated critic.

## Fixed-observation comparisons

Twelve observations spanning the rollout, identical base noise seeds. Physical
differences are predicted targets before Karma's clamps, not measured motion.
Changing Q ranking can legitimately choose a different stochastic base sample;
large differences do not themselves prove the candidate is worse.

| Policy | Joint difference from reference, p95 / max (rad) | Adjacent normalized action change, p95 |
|---|---|---|
| Original base | 0 / 0 | .02197 |
| Old expert only | .06274 / .15411 | .02449 |
| Old Q ranking only | .18392 / .41265 | .01743 |
| Old edits + ranking on original base | .22791 / .50527 | .31210 |
| Old full version 1 | .23219 / .50758 | .30797 |
| Corrected full offline candidate | .22634 / .51825 | .02206 |

The old editor strongly increased within-chunk discontinuities. The corrected
candidate's metric is near the base, but ranking still produces large deviations
from the reference sample. Reduced edit amplitude alone cannot bound ranking
changes. Masking gripper residuals likewise does not stop ranking from selecting
a base sample with a different gripper trajectory.

Across the twelve corrected comparison observations, absolute residual
median/p95/max were .001369/.004713/.010046; gripper residuals were exactly zero.
Across all 47 training windows with separate selection seeds, the maximum was
.012854, still below .05. These figures do not establish hardware task success.

## Recorder finding and prepared patch

45 of 49 saved base requests had exactly 29 recorded commands matching predicted
action indices 1–29. In the audited Karma commit
`b4f06f6d645755e605b6c0aec7c10af3d2c911d6`, `record_session` reads arm state before
`source.poll`, which may block during inference. It then checks the age of that
pre-inference snapshot. The source has already advanced its plan, so a rejected
tick can consume action 0 without commanding or recording it. This mechanism
matches the observed missing first actions; client runtime tracing is still
needed for authoritative request alignment.

`scripts/yam/karma-refresh-state.patch` adds an opt-in post-poll state refresh
and enables it for the standard `karma rollout` path used by record_round.py.
It retains the stale-telemetry check. Four regression cases simulate HTTP delay
and genuine telemetry loss without sleeping or using hardware. All 60 Karma
recorder tests pass. Other recording modes retain their existing behavior.

The patch is PREPARED ONLY, not installed on the NUC. Apply it only to a matching
Karma checkout, using `git apply --check` first. It will change the first-action
behavior; the existing dataset cannot be retroactively repaired with certainty.
Request IDs, actual execution timestamps, intervention boundaries, and exact
success timing remain necessary additions before general online deployment.

Saved request intervals (execution plus inference, not pure model latency):
base median 1.30 seconds; old version 1 median 2.65 seconds. Sequential sampling
therefore changed the control timing materially.

## Validation and next deployment gate

- 18 local learner/conversion/server tests passed, including accumulated-gradient
  equivalence, delayed updates, masked entropy, terminal targets and restart.
- Saved real GPU candidate reloaded and reproduced selected actions exactly.
- Ruff passed on changed Python files; recorder patch reverse-apply check passed.
- Base weights, prior dataset/session labels, and version 1 were not modified.

Do not deploy this candidate as an improved policy. It was trained and inspected
on a single episode, not a held-out set. Fixed 30-step replay windows also do not
yet reflect authoritative client request boundaries. The next steps are client
instrumentation and alignment, more varied base-policy episodes, and held-out
critic/ranking checks. Continue using the original base for operator-run data
collection until those checks justify promoting a candidate.

## Reproduction

The completed experiment folder is immutable input/output for this review;
setup/prepare/train deliberately refuse to overwrite existing artifacts. For a
fresh run choose a new root, then use the persistent JAX Python with
`scripts/yam/stability_experiment.py` stages `setup`, `audit`, `prepare`, `train`,
and `evaluate`, each with `--root NEW_ROOT`. The default source is `towel-expo`
and the default dataset is the repository's `round-0000`. GPU stages require
exclusive compute memory; no stage starts a server or contacts the NUC.
