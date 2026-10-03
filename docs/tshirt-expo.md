# T-shirt / hoodie EXPO bootstrap

This separate experiment uses `round-tshirt-00`. Its actual recorded instruction
is **fold the black hoodie**, which must be preserved during training and serving.
The operator confirmed reward **0**; the label file specifies **truncated**.
Karma's earlier success label is preserved as provenance, not used as the reward.
The verified import contains 6,750 frames, 225 transitions, and zero human frames.

`bootstrap_hitl.py` explicitly imports an offline prior from the basic server.
It does not invent a collecting experiment/version identity. It selects the one
saved attempt, excludes discarded attempts, and preserves intervention labels.
The initial learner is fresh; it does not restore the towel critic or editor.
The base checkpoint is the original frozen pi0.5 model.

Stop the GPU model server yourself before these commands. Run on the 4090:

```bash
cd /home/sra/tirth/expo-ft
export CUDA_VISIBLE_DEVICES=0
export JAX_COMPILATION_CACHE_DIR=/usr/local/models/sra-expo-ft/jax-cache

/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/bootstrap_hitl.py init \
  --root /usr/local/models/sra-expo-ft/tshirt-expo-stable-v2 \
  --dataset /home/sra/tirth/expo-ft/round-tshirt-00 \
  --prompt "fold the black hoodie"

/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/bootstrap_hitl.py prepare \
  --root /usr/local/models/sra-expo-ft/tshirt-expo-stable-v2

/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/bootstrap_hitl.py train \
  --root /usr/local/models/sra-expo-ft/tshirt-expo-stable-v2
```

Run `init` once on a new root. It imports replay and saves an untrained learner
checkpoint that is deliberately blocked from serving. `prepare` generates 16
base candidates per anchor and can resume interrupted candidate generation.
`train` uses the shared conservative continuation driver and publishes version 1
only after finite-value, checkpoint reload, and residual-bound checks.

The [cloth-folding guide](cloth-folding.MD#exact-active-configuration) documents
the shared settings: edit scale 0.05, grippers masked, ten Q networks,
minimum-of-two target backup, eight base plus eight edited candidates,
effective batch eight, and a frozen base. This dataset gives 40 critic updates
and two editor/temperature updates. Reward 0 with truncation means all rewards
are zero and all transitions bootstrap; no terminal auxiliary batch is present.
This first round exercises the loop without observed terminal task supervision.
Do not interpret its losses as evidence of improved task success.

No training or server execution is implicit in copying these scripts. Later
ordinary versioned rollouts can use `record_round.py` and `continue_stable.py`.
Later HITL episodes without server identity can be explicitly admitted as
unversioned off-policy priors as shown below. This does not verify their
collecting policy identity or replace the normal versioned rollout checks.


## Continue from version 1 with HITL round 01

`round-tshirt-01` contains 2,318 frames, 78 replay transitions, and 689 human
frames. The saved label is reward 1, success. Discarded attempts are excluded.
This is human-assisted success, not an autonomous evaluation result. Replay
preserves intervention flags, but the learner treats executed human and policy
actions as off-policy data with the same critic objective; no separate human
imitation objective is added.

Stop the model server. Using the same environment variables above:

```bash
/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/bootstrap_hitl.py prepare-hitl \
  --root /usr/local/models/sra-expo-ft/tshirt-expo-stable-v2 \
  --dataset /home/sra/tirth/expo-ft/round-tshirt-01 \
  --expected-parent 1 \
  --allow-unversioned-prior

/usr/local/models/sra-expo-ft/venv-jax/bin/python scripts/yam/bootstrap_hitl.py train \
  --root /usr/local/models/sra-expo-ft/tshirt-expo-stable-v2
```

The explicit flag acknowledges that the HITL recorder did not save a collecting
server version/hash. The importer records this limitation, hashes the dataset,
checks the current training parent, preserves the old replay, and rejects a
changed source on preparation resume. Training restores version 1's critic,
editor, encoder, target Q, optimizer state and RNG, then publishes version 2.
For 78 new transitions the update budget is 40 critic and two editor/temperature
updates; the base stays frozen. Do not run `init` again.
