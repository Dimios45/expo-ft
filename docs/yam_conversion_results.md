# YAM π0.5 conversion results

Verified on 2026-09-27, without training or robot access.

## Artifact

- Source: `/home/sra/molmoact2/outputs/models/molmoact2-yam-pi05`.
- Source weights SHA-256: `a777861c627234f9aa54a1bb7bdee29101ee6513f4773ef0a581d9c5527981e4`.
- Converted checkpoint: `/usr/local/models/sra-expo-ft/yam_pi05_jax/params`.
- Workspace link: `artifacts/yam_pi05_jax`.
- OpenPI revision: `2abe46282bfdf9f1bc0240f3f9960ec175d1b4a8`.
- Source weights hash was checked again after validation and is unchanged.

811 action-model tensors map into 51 OpenPI parameter arrays. Every tensor's
layout transform passed an exact inverse check. Two unused vocabulary-output
heads are preserved in `unused_language_heads.safetensors`; PI05 action
inference never calls them. Destination keys and shapes were checked against
an abstract OpenPI model, and Orbax restoration succeeded with strict loading.

`openpi_config.json` describes the converted model; `config.json` and the saved
processor files preserve the source LeRobot configuration for preprocessing.
Use the **FP32 compute configuration** for the verified loading path.

## FP32 numerical checks: passed

Ten recorded inputs across five towel-folding episodes and one synthetic input
were tested with explicit identical noise tensors and ten integration steps.
The recordings were used only as inference fixtures, not training data.

| Check | Result |
|---|---|
| Images, normalized state, token IDs and masks | Exact match to original LeRobot processing |
| Image/language prefix | Passed `atol=1e-4, rtol=1e-3`; max absolute difference `2.14e-4` on larger-magnitude values |
| Flow at noise times 0.1, 0.5, 0.9 | Passed; maximum absolute difference `6.10e-6` |
| Full normalized action chunks | Passed; maximum absolute difference `2.33e-6` |
| Physical joint/gripper action chunks | Passed; maximum absolute difference `1.67e-6` |
| Repeated JAX inference with identical noise | Bitwise equal |
| Fresh-process public-loader reload | Bitwise equal to prior JAX output |
| Public loader output | Finite `(30,14)` |
| Layout unit tests | 3 passed |

The FP32 PyTorch reference's recorded peak allocation was 13,758,430,208 bytes
(about 12.8 GiB). JAX's platform allocator does not expose peak memory statistics;
its process was observed at approximately 13.3 GiB in nvidia-smi, which is a
sample rather than a measured peak.

Median warmed JAX ten-step sampling was approximately 318 ms for one candidate.
A fresh loader run measured approximately 325 ms. These measurements exclude
end-to-end camera/network/robot overhead, and another GPU job was active. They
are not a claim of continuous 30 Hz replanning or EXPO multi-candidate latency.

## BF16: not accepted as numerically equivalent

A separate BF16 comparison ran on all 11 inputs. It did **not** pass the strict
FP32 tolerance. Maximum normalized-action difference was approximately `0.04195`,
and maximum physical-action difference was approximately `0.01963` in the
mixed joint/gripper output coordinates.

The implementations use different mixed-precision paths: notably the LeRobot
source preserves the vision path in FP32, while the stock OpenPI BF16 config
also changes vision compute precision. No tolerance was relaxed to hide these
differences. The supplied public adapter defaults to FP32; BF16 remains an
explicit experimental option, not the verified equivalent path.

## Reproduction and limits

See [the workflow and commands](yam_checkpoint_conversion.md). Detailed
per-fixture reports, environment versions, and source hashes are stored under
`artifacts/yam_pi05_jax/validation/` and in `conversion_manifest.json`.
Validation scratch inputs/output arrays currently live under
`/tmp/expo-yam-conversion`; the checkpoint and final reports use persistent
`/usr/local/models` storage.

No model training, policy server, Karma process, NUC connection, or robot command
was started during this implementation. The adapter expects the source model's
coordinate conventions. Karma's wire-versus-dataset gripper conversion is still
a separate deployment integration task. Numerical parity does not establish
physical task success.

### Karma HTTP integration — 2026-09-28

Tested the actual `MolmoActClient` and `BimanualObservation` from Karma commit
`b4f06f6d645755e605b6c0aec7c10af3d2c911d6` against the real FP32 JAX model on
localhost, using recorded fixture 00. Health validation, raw RGB and JPEG95
requests passed; both returned finite `(30,14)` chunks. Local round-trip times
were 0.266 s (raw) and 0.229 s (JPEG). These are two observations, not a latency
benchmark or a measurement of the NUC network.

Historical adapter behavior (superseded on 2026-09-29): indices 6 and 13 map between
Karma native `1=open` and LeRobot dataset `0=open`, before state normalization
and after action unnormalization. The source-frame model loader is unchanged.
The previous pass-through HTTP wrapper was not appropriate for native Karma.
Evidence: Karma `docs/model-servers.md` and `src/openpi_control/record.py`,
plus the local LeRobot serving documentation and VR recorder conventions.
At that time, a regression test checked this inversion and joint preservation.
That test did not establish equivalence to the working PyTorch deployment.

Full JSON report: checkpoint `validation/karma_http.json`. The test server was
bound to localhost only and stopped after testing. No NUC/robot access occurred.
This verifies the reviewed software interface, not physical camera assignments,
arm calibration, limits, or manipulation success on the user's rig.


### PyTorch deployment alignment — 2026-09-29

The user confirmed the working PyTorch baseline uses `molmoact2-yam-pi05`
and requested removal of server-side gripper inversion. The JAX HTTP wrapper
now passes state directly to the model and returns its actions directly, matching
the inspected PyTorch wrapper. The prior inversion was an extra deployment
transformation, not part of weight conversion. Its contribution to the observed
rollout failures has not been proven.

Checkpoint, saved normalization, FP32 compute, ten denoising steps and 30-action
output are unchanged. Client speed and execution settings were not changed.
The regression test checks exact incoming state and outgoing action preservation,
including gripper endpoints and intermediate values, plus health metadata.
The earlier `validation/karma_http.json` describes the historical inverted server;
it is not a live test of this revision. No hardware evaluation was performed.
