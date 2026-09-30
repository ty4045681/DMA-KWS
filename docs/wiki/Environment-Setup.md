# Environment Setup

Recommended way to build a DMA-KWS training environment, including the version pins this
codebase actually needs and the failure modes of a "just install the latest" setup.

**Last verified:** 2026-10-01 — Ubuntu 22.04, Python 3.10.12, 1× Tesla V100-SXM2-32GB
(compute capability 7.0 / `sm_70`), NVIDIA driver 580 (CUDA 13.0), uv 0.12.21.

The [README](https://github.com/ty4045681/DMA-KWS#1-clone-and-create-environment) covers the
basic steps; this page adds the pins, the CUDA index handling, and the traps.

## TL;DR

```bash
git clone https://github.com/ty4045681/DMA-KWS.git
cd DMA-KWS

pip install --user uv                  # or: curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # uv installs itself into ~/.local/bin

uv venv --python 3.10 .venv

# CUDA-enabled torch/torchaudio: pin cu126, do NOT use --torch-backend=auto (see below)
uv pip install --python .venv/bin/python --torch-backend=cu126 \
  "torch==2.7.1" "torchaudio==2.7.1"

# Project + every optional extra. The torchmetrics floor is required, not cosmetic.
uv pip install --python .venv/bin/python \
  -r requirements.txt -e ".[dev,locator,wekws,icefall,adapt]" "torchmetrics>=1.9"

# System audio libraries (torchaudio sox backend) and NLTK data (g2p_en)
sudo apt-get install -y --no-install-recommends libsox-dev libsox-fmt-all ffmpeg
.venv/bin/python -m nltk.downloader averaged_perceptron_tagger_eng \
  averaged_perceptron_tagger cmudict punkt punkt_tab

# Verify
.venv/bin/python -m pytest tests -q
```

Add `--seed` to `uv venv` (or `uv pip install --python .venv/bin/python pip`) if you want a
`pip` executable inside the environment; otherwise use `uv pip` for everything. `.venv/` is
already in `.gitignore`.

## Why uv (and not conda / plain venv)

- The dependency tree is **pure Python** — nothing here needs conda's binary packaging.
  The only non-Python requirements are audio libraries, installed with one `apt-get` line.
- `uv` manages the Python interpreter, the venv and the seed packages itself, so it works
  on hosts where the distro `venv` is broken (e.g. `ModuleNotFoundError: No module named
  'ensurepip'` — a missing `python3.10-venv` package) and needs no root.
- Resolution and installation are fast (the full 138-package environment resolves in
  seconds), and `uv pip` is drop-in compatible with the `pip` commands in the README.
- It has first-class PyTorch CUDA support (`--torch-backend=cu126`), which avoids the
  index-mixing traps described below. For a fully reproducible setup, see
  [uv lock](#uv-lock-reproducible-installs).

`conda` remains a valid choice on GPU servers that need conda-forge binaries (CUDA toolkit,
NCCL, FFmpeg, …) — see [conda](#conda).

## Version pins

| Component | Pin | Why |
|---|---|---|
| Python | 3.10 | `requires-python = ">=3.10"` |
| torch / torchaudio | **2.7.1+cu126** | 2.7.1 is the last release with the **legacy torchaudio I/O API**. Starting with torchaudio 2.8, `torchaudio.info`, `list_audio_backends` and `set_audio_backend` become deprecation shims and TorchCodec support is introduced; by 2.11 `torchaudio.info` is gone and `torchaudio.load` raises `ImportError: TorchCodec is required ...`. The repository uses `torchaudio.info`, `torchaudio.sox_effects` and `torchaudio.compliance.kaldi.fbank` in Stage-II preparation and MUSAN FA evaluation. |
| torchmetrics | **>= 1.9** | `dma_kws/inference/metrics.py` imports `binary_eer` / `binary_roc`, which do not exist in 1.0.3. |
| CUDA backend | **cu126** on V100 | The cu126/cu128 wheels ship `sm_70` kernels; **CUDA 13 (cu130) dropped `sm_70`**. The driver on many V100 boxes advertises CUDA 13, so `--torch-backend=auto` can select cu130 and produce `no kernel image is available for execution on the device` at runtime. Always pass the backend explicitly. |
| lhotse | 1.33.0 | Pinned by the `icefall` extra (Zipformer fbank parity). |

### Mixed-index traps

1. **torchmetrics downgraded to 1.0.3.** If you install torch with
   `--extra-index-url https://download.pytorch.org/whl/cu126` and then install the rest of
   the requirements with the same index combination, the resolver may take `torchmetrics`
   from the PyTorch index, which only carries **1.0.3** (its own copy, unrelated to PyPI's
   latest). Importing `binary_eer` then fails in 10 test modules. Install torch with
   `--torch-backend=cu126` (uv scopes the PyTorch index to torch packages only), or keep
   the PyTorch index out of the project-dependencies command entirely.
2. **Plain venv with the PyTorch extra index.** uv's default `first-index` strategy refuses
   to look past the first index that carries a package. If a mirror is listed first and is
   missing the requested `+cu126` local version, resolution fails with
   `no version of torch==2.7.1+cu126`. Prefer `--torch-backend`, or use
   `--index-strategy unsafe-best-match` when both indexes are trusted.

## Step by step

### 1. Install uv

```bash
pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

(Alternative: `curl -LsSf https://astral.sh/uv/install.sh | sh`; on networks where GitHub is
slow, the PyPI wheel is usually faster.)

### 2. Create the environment

```bash
cd DMA-KWS
uv venv --python 3.10 .venv
```

`uv` uses the system Python 3.10 when present, otherwise it downloads a managed CPython.
Nothing else is installed into the system interpreter.

### 3. CUDA-enabled PyTorch

```bash
uv pip install --python .venv/bin/python --torch-backend=cu126 \
  "torch==2.7.1" "torchaudio==2.7.1"
```

Pick `cu126` for V100/T4/A100-class servers. On machines without an NVIDIA driver use
`--torch-backend=cpu`.

### 4. Project dependencies and extras

```bash
uv pip install --python .venv/bin/python \
  -r requirements.txt -e ".[dev,locator,wekws,icefall,adapt]" "torchmetrics>=1.9"
```

Trim the extras to what you need: `dev` (pytest), `locator` (sherpa-onnx),
`wekws` (librosa), `icefall` (lhotse 1.33.0), `adapt` (optuna).

### 5. System audio libraries

```bash
sudo apt-get install -y --no-install-recommends libsox-dev libsox-fmt-all ffmpeg
```

`libsox` is required by `torchaudio.sox_effects.apply_effects_tensor()`
(`OSError: libsox.so: cannot open shared object file` without it). `soundfile` wheels bundle
their own `libsndfile`, so FLAC loading works either way; FFmpeg additionally gives
torchaudio its FFmpeg backend.

### 6. NLTK data for g2p_en

```bash
.venv/bin/python -m nltk.downloader averaged_perceptron_tagger_eng \
  averaged_perceptron_tagger cmudict punkt punkt_tab
```

Without this, phoneme preparation fails with
`Resource 'averaged_perceptron_tagger_eng' not found`. On an offline host, copy
`~/nltk_data` from another machine and set `NLTK_DATA`.

### 7. Verify

```bash
.venv/bin/python - <<'PY'
import torch, torchaudio, torchmetrics
print("torch     :", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())
print("arches    :", torch.cuda.get_arch_list())
print("torchaudio:", torchaudio.__version__, "| info:", hasattr(torchaudio, "info"),
      "| backends:", torchaudio.list_audio_backends())
print("torchmetrics:", torchmetrics.__version__)
PY
```

Expected (V100 host):

```text
torch     : 2.7.1+cu126 | cuda 12.6 | avail True
arches    : ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']   # sm_70 must be present
torchaudio: 2.7.1+cu126 | info: True | backends: ['ffmpeg', 'sox', 'soundfile']
torchmetrics: 1.9.0
```

Then the environment check and the test suite:

```bash
uv pip check --python .venv/bin/python      # -> All installed packages are compatible
.venv/bin/python -m pytest tests -q         # -> 2067 passed, 5 skipped, 1 failed
```

The single failure, `tests/test_stage2_background_sources.py::test_source_selection_last_bin_includes_one_and_boundaries`,
is **unrelated to how the environment was installed**: the test helper
`_ScriptedRandom([0.0])` passes a list as a positional argument to a `random.Random`
subclass, and CPython builds whose `random.Random.__new__` seeds from the constructor
argument hash it during construction (`TypeError: unhashable type: 'list'`). It reproduces
with the stock system `python3` and with any package manager. Passing the argument by
keyword (`_ScriptedRandom(values=[0.0])`) avoids it.

`scripts/run_smoke.sh` runs pytest first under `set -e`, so that one failure stops it
before the compile checks; run the `py_compile` / `bash -n` steps manually while it is red
(they pass).

## Mirrors (China / restricted networks)

`uv` takes the index from `--default-index` / `UV_DEFAULT_INDEX`:

```bash
export UV_DEFAULT_INDEX=http://mirrors.tencentyun.com/pypi/simple
# or per command: --default-index http://mirrors.tencentyun.com/pypi/simple
```

The PyTorch CUDA index (`https://download.pytorch.org/whl/cu126`) is applied by
`--torch-backend=cu126`, so it never contaminates the resolution of ordinary packages.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'whisper'` | `openai-whisper` missing — install the core requirements. |
| `OSError: libsox.so: cannot open shared object file` | Install `libsox-dev libsox-fmt-all` (step 5). |
| `ImportError: TorchCodec is required for load_with_torchcodec` | torchaudio >= 2.8 is installed; pin torch/torchaudio 2.7.1 (or install `torchcodec` + FFmpeg — note the repository also uses `torchaudio.info`, which newer torchaudio removed). |
| `ImportError: cannot import name 'binary_eer'` | `torchmetrics` resolved to 1.0.3 from the PyTorch index; install it from PyPI (`>=1.9`). |
| `no version of torch==2.7.1+cu126` | uv `first-index` strategy, see [Mixed-index traps](#mixed-index-traps). |
| `ModuleNotFoundError: No module named 'ensurepip'` when creating a venv | The distro `venv` module is incomplete; use `uv venv` (no root required). |
| `no kernel image is available for execution on the device` | A `cu130` (or otherwise `sm_70`-less) build got installed; reinstall with `--torch-backend=cu126`. |
| `Resource 'averaged_perceptron_tagger_eng' not found` | Run `nltk.downloader` (step 6). |
| `TypeError: unhashable type: 'list'` in `test_stage2_background_sources.py` | Pre-existing test/CPython quirk, unrelated to the environment (see above). |

## Alternatives

### conda

Same pins, one index at a time (never mix the PyTorch index into the requirements command):

```bash
conda create -n dma-kws python=3.10 -y
conda activate dma-kws
pip install "torch==2.7.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt -e ".[dev,locator,wekws,icefall,adapt]" "torchmetrics>=1.9"
conda install -c conda-forge libsndfile ffmpeg -y    # if apt is unavailable
```

### uv lock (reproducible installs)

To make the environment reproducible, declare torch/torchaudio as project dependencies and
route them to the CUDA index, then lock:

```toml
# pyproject.toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchaudio = { index = "pytorch-cu126" }
```

```bash
uv lock          # writes uv.lock
uv sync --locked # exact reproduction on another machine
```

### Docker

For fleets where every node must be identical, bake the steps above into an image and pin
the CUDA base image to a driver-compatible version. Not required for single-host training.
