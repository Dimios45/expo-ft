# YAM documentation

Start with [Cloth folding on the RTX 4090 and YAM NUC](cloth-folding.MD).
It is the operating guide for the current `stable-v2` experiment: record one
episode with Karma, label the outcome, transfer the complete dataset, prepare
replay, train, validate, and restart the versioned server.

The documented snapshot is version 4. Subsequent runs advance `current.json`;
use the experiment's actual manifest and status rather than assuming that the
version in a report is still current. The base model stays frozen in this
workflow. The critic, visual encoder, action editor, and temperature are learned.

## Guides and evidence

| Document | Purpose |
| --- | --- |
| [Cloth-folding guide](cloth-folding.MD) | Current commands, configuration, training equations, code map, Karma integration, and limitations |
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

Run these commands from the GPU host's repository. They stage the YAM source,
conversion code, scripts, dependency requirements, Karma recorder patch, tests,
and documentation. They do not run training or connect to hardware.

```bash
cd /home/sra/tirth/expo-ft

git add -- \
  .gitattributes .gitignore README.md docs/ \
  expo_ft/conversion/ expo_ft/yam/ \
  scripts/yam/ tests/

git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
```

Review the staged files, then commit if they are the intended changes:

```bash
git commit -m "Add single-GPU YAM EXPO workflow and cloth-folding documentation"
```

The root `.gitignore` excludes `round-0000/` and other `round-<number>` dataset
directories, the `expo-ft-jax-rollouts-*` recording symlink, local environments,
and generated caches. Checkpoints, replay stores, inference traces, tokenizer
files, and model weights under `/usr/local/models/sra-expo-ft/` are external
artifacts, not part of this commit. Preserve and back up those separately;
committing the code does not back up the trained model or collected episodes.

The command includes `scripts/yam/karma-refresh-state.patch` as a patch file.
Committing it here does not apply it to the NUC's separate Karma checkout.
See the cloth-folding guide for its purpose, application steps, and validation
limits.
