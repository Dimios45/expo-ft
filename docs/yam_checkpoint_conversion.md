# Offline YAM π0.5 checkpoint conversion

This workflow converts and validates weights only. It does not train, start an
HTTP service, import Karma, connect to the NUC, or issue hardware commands.

## Source and target

Source: `/home/sra/molmoact2/outputs/models/molmoact2-yam-pi05`.
Target: `artifacts/yam_pi05_jax`, linked to
`/usr/local/models/sra-expo-ft/yam_pi05_jax` outside the nearly full `/home`.
OpenPI revision: `2abe46282bfdf9f1bc0240f3f9960ec175d1b4a8`.
Architecture: π0.5, Gemma 2B + 300M expert, 30 action positions, 32 padded
coordinates, 14 physical outputs, and three RGB cameras in top/left/right order.

The mapper checks the inverse of every layout operation. It archives the two
unused language-output heads rather than discarding them. Parameters are stored
in float32; expanding a BF16 source value to float32 is exact. This is not model
training or quantization. Orbax export separately validates all destination
names/shapes against an abstract OpenPI model, without initializing replacement
weights. An existing output directory is never overwritten.

## Environment

The isolated `.venv-convert` points to `/tmp/expo-yam-conversion/venv` to avoid
filling `/home`. Only the environment and validation scratch files use temporary storage; the
converted checkpoint uses persistent `/usr/local/models` storage. The model source
and its saved processors are never modified.

```bash
uv venv --python 3.11 /tmp/expo-yam-conversion/venv
# Create this link only if it does not already exist:
ln -s /tmp/expo-yam-conversion/venv .venv-convert
uv pip install --python .venv-convert/bin/python \
  'jax[cuda12]==0.5.3' 'flax==0.10.2' 'orbax-checkpoint==0.11.13' \
  'numpy<2' 'jaxtyping==0.2.36' 'beartype==0.19.0' \
  'equinox==0.11.12' 'ml-dtypes==0.4.1' 'tensorstore==0.1.74' \
  'transformers==4.53.2' augmax einops etils safetensors \
  sentencepiece pillow pytest 'fsspec[gcs]'
uv pip install --python .venv-convert/bin/python \
  'torch==2.7.1' --index-url https://download.pytorch.org/whl/cpu
```

CPU PyTorch supplies safetensors loading and the exact reference image resize.
JAX runs the converted policy on the GPU. The validation commands use the
platform allocator to avoid reserving most of a shared GPU. The separate existing LeRobot
PyTorch environment runs the reference policy, sequentially with JAX.

## Commands

Run from the repository root. Use a new output path for each conversion.

```bash
.venv-convert/bin/python scripts/yam/convert_checkpoint.py \
  --source /home/sra/molmoact2/outputs/models/molmoact2-yam-pi05 \
  --output artifacts/yam_pi05_jax
```

`--layout-only` works in the existing PyTorch environment without JAX. Then:

```bash
.venv-convert/bin/python scripts/yam/export_orbax.py artifacts/yam_pi05_jax
```

Build fixtures with the existing environment (PyArrow, Pillow, and ffmpeg with
software libdav1d decoding are required):

```bash
/home/sra/molmoact2/.venv-pi05-server/bin/python scripts/yam/make_fixtures.py \
  --dataset /usr/local/datasets/sra-molmoact2-data/Dimios45/molmo-fold-pink-towel-dagger \
  --output /tmp/expo-yam-conversion/fixtures
/home/sra/molmoact2/.venv-pi05-server/bin/python scripts/yam/reference_torch.py \
  --lerobot-src /home/sra/molmoact2/third_party/lerobot-pi05-server/src \
  --source /home/sra/molmoact2/outputs/models/molmoact2-yam-pi05 \
  --tokenizer /home/sra/molmoact2/outputs/models/paligemma-tokenizer \
  --fixtures /tmp/expo-yam-conversion/fixtures \
  --output /tmp/expo-yam-conversion/reference_fp32
.venv-convert/bin/python scripts/yam/check_jax.py \
  --checkpoint artifacts/yam_pi05_jax \
  --tokenizer /home/sra/molmoact2/outputs/models/paligemma-tokenizer \
  --fixtures /tmp/expo-yam-conversion/fixtures \
  --reference /tmp/expo-yam-conversion/reference_fp32 \
  --output /tmp/expo-yam-conversion/jax_fp32
.venv-convert/bin/python -m pytest tests/test_yam_conversion.py -q
```

Use `--precision bfloat16` and distinct output directories for the separate
BF16 comparison. FP32 checks use atol=1e-4, rtol=1e-3; preprocessing and repeat
inference require exact equality. A failed comparison is reported as a failure,
not silently accepted. BF16 statistics are descriptive; the same strict
thresholds are reported, without asserting that BF16 should meet FP32 accuracy.

## Check the existing artifact without reconverting

```bash
.venv-convert/bin/python scripts/yam/smoke.py \
  --checkpoint artifacts/yam_pi05_jax \
  --tokenizer /home/sra/molmoact2/outputs/models/paligemma-tokenizer \
  --fixture /tmp/expo-yam-conversion/fixtures/00.npz \
  --reference /tmp/expo-yam-conversion/jax_fp32_checked/00.npz \
  --output /tmp/expo-yam-conversion/reload.json
```

The initial conversion results are in [the validation report](yam_conversion_results.md).
FP32 passed; the stock BF16 compute path did not meet strict numerical parity.

## Loading in Python

Add `expo_ft/agents/vla/openpi/src` to `PYTHONPATH` alongside the repo root.
For a shared GPU, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` before importing JAX:


```python
from expo_ft.conversion.yam_loader import YamJaxPolicy
policy = YamJaxPolicy(
    'artifacts/yam_pi05_jax',
    '/home/sra/molmoact2/outputs/models/paligemma-tokenizer',
    dtype='float32',
)
actions = policy.predict(
    {'top': top_rgb, 'left': left_rgb, 'right': right_rgb},
    state_14,
    prompt='fold the towel',
)
assert actions.shape == (30, 14)
```

Inputs must already use the **source checkpoint's** joint and gripper frame.
The adapter deliberately does not infer a gripper-polarity conversion. Karma's
wire and dataset conventions differ; deployment must resolve that separately.
The 14-coordinate state is normalized and tokenized before model padding.
Image resize uses the source's CPU PyTorch bilinear operation, not a subtly
different JAX resize. No online network/tokenizer download occurs in the loader.

## Boundaries

Numerical agreement demonstrates conversion fidelity, not task success or
safe physical execution. The fixtures are inference inputs only, not training
or seed replay. No reward labels are inferred from them. The subsequent
collect/transfer/train/restart EXPO experiment is a separate implementation.

### Karma HTTP server

Run on the 4090 host from the repository root:

```bash
uv pip install --python .venv-convert/bin/python -r scripts/yam/requirements-serve.txt
CUDA_VISIBLE_DEVICES=0 .venv-convert/bin/python scripts/yam/serve_jax.py \
  --checkpoint /usr/local/models/sra-expo-ft/yam_pi05_jax \
  --tokenizer /home/sra/molmoact2/outputs/models/paligemma-tokenizer \
  --dtype fp32 --memory-mode device \
  --host 0.0.0.0 --port 8204
```

Wait for warmup and `Uvicorn running`, then check `curl http://127.0.0.1:8204/healthz`.
Karma's server URL is `http://192.168.0.167:8204/act`. The server accepts
`json_numpy` RGB arrays or encoded camera images, 14-D state, instruction,
and `num_steps=10`; it returns absolute `(30,14)` actions. This command selects FP32.
For BF16, RTX 3060 FP32 with unified memory, A100 commands, memory requirements,
and measured results, see [the JAX serving guide](yam_jax_serving.md).
The normalization tag identifies the YAM wire schema; normalization uses the
checkpoint's saved quantiles. CUDA graph flags are ignored (JAX uses JIT).
One process owns the model; do not launch multiple workers.
The HTTP adapter passes all 14 state and action coordinates through without
any gripper inversion, matching the working PyTorch wrapper. The saved model
normalization and unnormalization remain unchanged. Health reports
`gripper_adapter: "none; matches PyTorch pass-through"`.
Physical execution has not been tested. Starting the server does not connect to or command a robot.
