# YAM EXPO stability review — 2026-09-29

This is a proposed restart plan, not a deployed configuration. Preserve version
0001 and the original dataset for diagnosis. No robot commands, training, or
checkpoint changes were performed for this review. The next experiment's edit
scale is 0.05, as requested.

## Evidence from this run

- Version 0001 used 100 critic updates, 100 editor/temperature updates, and 10
  action-expert updates. RL batch size was 1; learning rate was 3e-4 for all RL
  components, initial temperature 1, and edit scale 0.2.
- The replay has 47 full 30-by-14 action windows from 1,391 frames. Only its last
  window has reward (0.99^29 = 0.7471721). The other rewards are zero.
- Reconstructing the loop's NumPy RNG consumption from seed 42 (episode choice,
  transition choice, eight noise arrays, and success-actor sampling every tenth
  update) yields zero samples of transition 46 in 100 updates. This is inferred
  from the deterministic sampling sequence, not a logged transition-ID record.
- Critic loss fell from 0.2401 to 0.01298, but that does not validate candidate
  ranking or success prediction. Final mean Q was -0.1436. Negative early Q is
  possible with unconstrained initialization and bootstrapping.
- Five saved version-1 requests contain proposed editor residuals. Across all
  eight edited candidates, absolute residual median/p90/p99/max were
  0.11706/0.18523/0.19753/0.19978. About 13.67% exceeded 0.18. Four of five selected
  an edited candidate. These are proposal statistics, not measured robot motion.
- Median interval between those logged requests was 2.65 seconds; this includes
  client execution and server inference and is not an inference-only benchmark.

## Implementation issues to address

1. **Entropy units.** The implementation includes the residual scaling Jacobian
   in log probability but uses target entropy -D/2 without scaling it. For
   D=420 and epsilon=0.2, the maximum possible differential entropy on the edit
   box is D*log(2*epsilon)=-384.84, below the requested -210. At epsilon=0.05 the
   maximum is -967.09. The target is unattainable. Compare entropy in unscaled
   tanh coordinates to -D/2, or equivalently use scaled target
   -D/2+D*log(epsilon). Add a test that equivalent coordinates produce equivalent
   temperature gradients. Merely lowering epsilon makes the mismatch larger.
2. **Scheduling.** Separate critic, editor, temperature, and base updates.
   Upstream scans UTD critic minibatches then updates editor and temperature
   once. The local adaptation currently updates all three on every step.
3. **Reward coverage.** Log sampled episode/window IDs and terminal counts.
   Explicitly include terminal windows in a separate supervised terminal-Q loss
   alongside uniformly sampled Bellman transitions. Label this as a deliberate
   adaptation, not uniform replay. Do not label all success-episode frames r=1.
4. **Data alignment.** Audit saved command/state polarity against the actual
   NUC writer; retain server pass-through. Link policy request IDs, chunk index,
   executed command index, monotonic timestamps, interventions, and success
   time. Current fixed windows may cross request boundaries or intervention
   changes. Trim post-success reset/parking frames if present, based on evidence.
5. **Base drift.** The previous run updated 564M action-expert parameters from
   ten single-example updates. Freeze the base initially; this avoids confusing
   residual/ranking changes with base drift. Later test LoRA or a smaller update
   scope with success replay, reference data when available, and fixed-noise
   regression checks. Do not claim epsilon bounds base-model changes.
6. **Temporal credit.** gamma=0.99 per 30-Hz tick discounts a terminal reward
   1,390 ticks away to 8.57e-7. Review the desired task horizon in seconds before
   changing gamma. Preserve gamma^executed_steps, rather than silently treating
   gamma as a per-chunk discount. A candidate for this long task is 0.999/tick,
   explicitly an experimental change requiring validation.
7. **Execution mismatch.** Version 0 samples one base chunk; version 1 samples
   eight sequentially and ranks sixteen. Cache the shared VLM prefix and measure
   end-to-end latency before increasing sample count or shortening execution
   windows. RTC requires additional delay-aware changes, not enabling prefetch
   alone. Keep model horizon 30 distinct from a future shorter execution horizon.

## Proposed conservative experiment

Create a separate experiment from the original base with a fresh critic/editor;
do not modify the old immutable version or resume its optimizer. Explicitly
register original episode provenance if replay is reused; do not relabel its
experiment ID to bypass stale-version checks.

Initial settings (engineering starting points, not proven optima):

| Parameter | Proposed |
|---|---|
| Edit scale | 0.05 normalized = 2.5% of saved q99-q01 range |
| Base actor | Frozen during critic/editor validation |
| Q ensemble / minimum subsample | 10 / 2 |
| Target-Q tau | 0.005 |
| Critic / editor LR | 1e-4 / 3e-5 |
| Initial temperature / temperature LR | 0.01 / 1e-5, corrected entropy units |
| Effective RL batch | 8 initially, accumulated from microbatch 1–2 |
| Editor / temperature frequency | Once per 20 critic optimizer steps |
| Global gradient clipping | 1.0 initially; log clipping frequency |
| Image augmentation | Per-view mild crop; validate consistency before rotation/color |
| Initial editor output | Near-zero mean and small stochastic residual |
| Gripper edits | Initially masked, while preserving base gripper commands |

If gripper dimensions are masked, entropy and its target must exclude those
dimensions. Accumulation means average gradients at fixed parameters followed
by ONE optimizer step; eight sequential Adam updates are not equivalent.

Budget updates from newly admitted valid chunk transitions, not episode count
or camera-frame count. Proposed initial budget:

    critic_steps = min(40, 20 * ceil(new_chunk_transitions / 40))

For this episode's 47 windows that is 40 critic steps and 2 editor/temperature
steps. With effective batch 8 this is 320 sampled transitions plus the separately
logged terminal-Q auxiliary examples. This is a deliberately reduced schedule,
not paper UTD=20 equivalence. Settle the counting unit once request-aligned replay
is implemented. Sample accumulated replay across rounds, not only the new episode.

Continue ingesting/training after every episode, but keep candidates in shadow
mode during initial data collection. Use approximately ten varied base-policy
episodes as an initial collection target, then decide using held-out episode
checks rather than treating ten as a guarantee. Failures should be natural,
not deliberately induced unsafe motions. A prior demonstration dataset can
help only if its task, action representation, and normalization are verified.

## Validation and deployment gates

First compare offline, on identical recorded observations and fixed noise:
original base; version-1 expert only; original base with version-1 Q ranking and
no edits; original base with version-1 edits/ranking; full version 1. This separates
base drift from critic ranking and edits without controlling hardware.

For the corrected learner, test terminal reward coverage/masks, entropy units,
update counts, accumulated gradients, checkpoint restart, and dataset mapping.
Measure Q predictions against held-out returns on recorded actions; counterfactual
candidate ranking cannot be validated from those returns alone. Measure physical
joint/gripper deltas, temporal discontinuities, editor saturation, base drift,
ensemble disagreement, and latency. Low critic loss is not a deployment gate.

Only then evaluate on hardware under operator control with matched task starts,
prompt, speed, execution chunk, and latency as far as practical. Retain the
untouched base as fallback. One training episode does not establish improvement.

## Sources and distinctions

- EXPO-FT Appendix D:
  https://arxiv.org/html/2605.25477#A4
- Real-Time EXPO-FT Appendix VII-E:
  https://arxiv.org/html/2609.18207v1
- Repository: configs/model/expo_ft_pi_config.py,
  configs/model/realtime_expo_ft_pi_config.py, and
  expo_ft/agents/alg/expo_ft.py::_update_jit.

EXPO-FT reports batch 64, UTD 20, alpha 1, gamma .99, tau .005, eight base plus
eight edit candidates, task-dependent .05/.1/.2 edits, augmentation, and shorter
execution windows. Real-Time EXPO-FT reports alpha .01, a ten-episode training
start, different candidate/backup handling, and delay-aware execution. These
are different recipes; the local sequential 4090 adaptation must be identified
as such. Its demonstrated ability to fit GPU memory did not establish training
stability or readiness for hardware deployment.
