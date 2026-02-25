# Phase 4: Integration + Publishing - Research

**Researched:** 2026-02-25
**Domain:** Python packaging (hatchling + hatch-vcs), PyPI publishing (uv), CIFAR-10 example (torchvision MLP), README writing
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### CIFAR-10 example behavior
- Fixed configuration, no CLI arguments — user runs `python examples/cifar10.py` and it trains
- Target 5-10 minutes on CPU — full training run with meaningful accuracy results
- Use RichCallback for live progress (progress bars, accuracy, energy per epoch)
- Print full summary after training: final test accuracy, total training time, energy convergence summary

#### README depth and structure
- Tone: academic but pedagogical + practical — explain the theory accessibly, then show how to use it
- Include a brief explainer section (3-5 paragraphs) on what predictive coding networks are and how they differ from backprop, with link to the paper
- Quickstart code snippet + API overview table listing main classes/functions with brief descriptions
- Include a "Results" section with expected CIFAR-10 accuracy numbers and training metrics

#### Public API surface
- `pcn_torch.__init__` exports everything useful: network, training functions, layers, types, energy, callbacks
- `__all__` defined explicitly for IDE autocompletion and clean public API
- Example script imports only what it needs (minimal imports for clarity)
- Submodule import paths: Claude's discretion

#### Package and distribution
- PyTorch is a peer dependency — NOT in install_requires. README instructs "install PyTorch first"
- License: MIT
- Author: Enrique Mendez
- Version: v1.0.0 (not 0.1.0 — user wants to start at 1.0.0)
- Prepare for publishing only — build and verify with TestPyPI, real publish is a manual step after
- PyPI classifiers, keywords, project URLs: Claude's discretion

### Claude's Discretion
- Submodule import strategy (whether `from pcn_torch.energy import ...` also works alongside top-level)
- CIFAR-10 network architecture and hyperparameters (whatever achieves reasonable accuracy in 5-10 min)
- Exact README section ordering and formatting
- PyPI classifiers, keywords, and project URLs
- Loading skeleton / error handling in example

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 4 has four parallel workstreams: (1) update pyproject.toml metadata and finalize the public API, (2) write the CIFAR-10 example script, (3) write the README, and (4) build/verify with TestPyPI. The underlying packaging toolchain is already in place — hatchling + hatch-vcs are configured in pyproject.toml and uv 0.8.6 is installed. The primary gap is metadata, content, and artifacts, not tooling changes.

The `pcn-torch` name is available on PyPI (verified: no package exists with that name as of 2026-02-25). The version workflow uses hatch-vcs, which reads from the git tag. To release v1.0.0, a `git tag v1.0.0` must be created before running `uv build` so the wheel and sdist are stamped `1.0.0`. The current `_version.py` shows `0.0.0.dev32+gdd79cdb06.d20260224` because no release tag exists yet.

The CIFAR-10 example requires `torchvision` (not currently in the venv) for `CIFAR10` dataset loading. Torchvision must be added as a dev dependency (not a production dependency). A 3-layer fully-connected MLP trained with PCN inference/learning rules can achieve approximately 50-55% accuracy on CIFAR-10, which is in line with what fully-connected backprop achieves and is sufficient to demonstrate the library works.

**Primary recommendation:** Tag v1.0.0, add torchvision as dev dep, write example + README, run `uv build` then `uv publish --publish-url https://test.pypi.org/legacy/ --token $TOKEN` to verify, then leave real publish for human.

---

## Standard Stack

### Core (already in place)

| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| hatchling | in pyproject.toml | Build backend | Already configured; PEP 517 compliant, no setup.py |
| hatch-vcs | in pyproject.toml | VCS versioning | Already configured; reads version from git tag |
| uv | 0.8.6 (installed) | Build frontend + publish | Single tool for build + publish; replaces `python -m build` + twine |
| torchvision | >=0.15 | CIFAR-10 dataset loader | Official PyTorch vision library; `torchvision.datasets.CIFAR10` is the canonical loader |

### Supporting

| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| twine check | 6.x (optional) | Validates README renders on PyPI | Run before upload to catch RST/Markdown issues |
| rich | >=14.0 (already dep) | RichCallback progress display | Already in dependencies |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| uv publish | twine upload | twine is more widely documented but uv is already installed and handles both build+publish |
| torchvision | manual HTTP download | torchvision is cleaner; avoids custom download logic |
| hatch-vcs tag-based version | manual version in pyproject.toml | hatch-vcs is already configured; manual version simpler but not worth changing mid-project |

**Installation (dev only, for examples):**
```bash
uv add --group dev torchvision
```

---

## Architecture Patterns

### Recommended Project Structure After Phase 4

```
.
├── pyproject.toml           # updated: classifiers, keywords, urls, torch removed from deps
├── README.md                # rewritten: full academic + quickstart README
├── LICENSE                  # already exists (MIT)
├── examples/
│   └── cifar10.py           # new: standalone CIFAR-10 example
└── src/
    └── pcn_torch/
        ├── __init__.py      # already complete: __all__ defined, all exports present
        ├── _version.py      # auto-generated by hatch-vcs on build
        ├── _types.py
        ├── activations.py
        ├── energy.py
        ├── layers.py
        ├── network.py
        └── trainer.py
```

### Pattern 1: hatch-vcs Release Tagging

**What:** hatch-vcs reads the git tag to stamp the version. No version string in pyproject.toml — `dynamic = ["version"]` delegates to hatch-vcs.

**When to use:** Any time a release is made. Tag must exist BEFORE `uv build`.

**Example:**
```bash
# Source: https://github.com/ofek/hatch-vcs/blob/master/README.md
# Create and push the release tag
git tag v1.0.0
git push origin v1.0.0

# Build — hatch-vcs reads v1.0.0 from tag, writes _version.py, stamps wheel
uv build

# dist/ now contains:
#   pcn_torch-1.0.0-py3-none-any.whl
#   pcn_torch-1.0.0.tar.gz
```

**CRITICAL: The tag format must be `v1.0.0` (with the `v` prefix). hatch-vcs strips the `v` prefix to produce the version string `1.0.0`.**

### Pattern 2: PyPI Metadata in pyproject.toml

**What:** All PyPI metadata lives in `[project]` in pyproject.toml. Classifiers improve discoverability. PyTorch as peer dependency means it must NOT be in `dependencies`.

**Recommended classifiers for pcn-torch:**
```toml
# Source: https://pypi.org/classifiers/
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Typing :: Typed",
]
keywords = ["predictive coding", "neural networks", "pytorch", "machine learning", "deep learning", "biologically plausible"]
```

**Recommended project URLs:**
```toml
[project.urls]
Homepage = "https://github.com/emv-dev/pcn-torch"
Repository = "https://github.com/emv-dev/pcn-torch"
"Bug Tracker" = "https://github.com/emv-dev/pcn-torch/issues"
Paper = "https://arxiv.org/abs/2506.06332"
```

**PyTorch as peer dependency — correct approach:**
```toml
# torch MUST NOT be in [project] dependencies
# It was listed in dependencies — this needs to be REMOVED and documented in README
dependencies = [
    "numpy>=2.2.6",
    "rich>=14.0",
    # torch is a peer dependency -- install separately: https://pytorch.org
]
```

**IMPORTANT FINDING:** The current pyproject.toml has `"torch>=2.0"` in `dependencies`. Per the locked decision, PyTorch is a peer dependency and must be REMOVED from install_requires. This is a required change in Phase 4.

### Pattern 3: CIFAR-10 Example Architecture

**What:** A fixed 4-layer MLP (input->hidden->hidden->output) that trains in 5-10 minutes on CPU.

**Architecture recommendation (Claude's discretion):**
- Input: 3072 (3 x 32 x 32 flattened CIFAR-10)
- 3 PCN layers ending with top latent of 512
- Output: 10 classes via readout

**PCN dims ordering note:** In `PredictiveCodingNetwork(dims=...)`, `dims[0]` is the input dimension and `dims[-1]` is the top latent dimension. The readout layer maps top latent -> output_dim. So for CIFAR-10:
```python
model = PredictiveCodingNetwork(
    dims=[3072, 1024, 1024, 512],
    activation="relu",
    output_dim=10,
    mode="classification",
)
```
This creates 3 PCNLayers: [3072 to 1024, 1024 to 1024, 1024 to 512] with a Linear readout [512 to 10].

**Hyperparameters for 5-10 min CPU target:**
```python
config = TrainConfig(
    task="classification",
    T_infer=20,          # 20 inference steps per batch (fewer = faster)
    lr_infer=0.05,
    lr_learn=0.001,
    num_epochs=10,
    early_stop_threshold=1e-4,
    callback=RichCallback(),
)
```
Batch size: 128 (larger batches = fewer iterations, faster wall time on CPU)

**Expected accuracy:** 45-55% test accuracy (PCN MLP on CIFAR-10 is comparable to backprop MLP — both limited by the fully-connected architecture, not the learning rule). This is honest and sufficient to demonstrate the library.

**Training time estimate:** With 50,000 training samples, batch size 128 = ~390 batches/epoch, 20 inference steps + T_learn=128 steps per batch, 10 epochs on CPU (PyTorch 2.10+cpu). Estimated 5-15 minutes depending on hardware. Can tune T_infer downward if needed.

### Pattern 4: uv Build + TestPyPI Verification

**What:** Build artifacts, upload to TestPyPI, verify install, then stop (real publish is manual).

```bash
# Source: https://docs.astral.sh/uv/guides/package/
# Step 1: Tag release (must happen FIRST)
git tag v1.0.0

# Step 2: Build (hatch-vcs reads tag, stamps 1.0.0)
uv build
# Outputs: dist/pcn_torch-1.0.0-py3-none-any.whl
#          dist/pcn_torch-1.0.0.tar.gz

# Step 3: Verify artifacts with twine check (optional but recommended)
uvx twine check dist/*

# Step 4: Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/ --token $TESTPYPI_TOKEN

# Step 5: Verify install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pcn-torch
python -c "import pcn_torch; print(pcn_torch.__version__)"

# Real PyPI publish is a manual step (user does it)
# uv publish --token $PYPI_TOKEN
```

### Pattern 5: Submodule Import Strategy (Claude's Discretion)

**Recommendation:** Support both `from pcn_torch import X` (top-level) and `from pcn_torch.energy import compute_energy` (submodule) simultaneously. No extra work needed — Python's import system allows both naturally. The existing `__init__.py` already re-exports everything; submodule paths work without changes.

Do NOT add `__init__.py` files to prevent submodule access. The current structure already provides dual import paths.

### Anti-Patterns to Avoid

- **Tag after build:** Running `uv build` before creating the git tag produces `0.0.0.dev...` version stamps. Always tag first.
- **torch in dependencies:** PyTorch in `install_requires` forces a CPU-only or specific CUDA version wheel. Peer dep means users install their own variant.
- **examples/ in the wheel:** Ensure the `examples/` directory is NOT included in the wheel. Hatchling's `packages = ["src/pcn_torch"]` already excludes it.
- **torchvision in production deps:** torchvision is only needed for the example, not the library. It goes in `[dependency-groups].dev`, not `[project].dependencies`.
- **Hardcoded data path in example:** Use `root="./data"` with `download=True` so the example works from any directory.
- **Example imports from private names:** Example should use `from pcn_torch import PredictiveCodingNetwork, TrainConfig, TrainHistory, RichCallback, train_pcn, test_pcn` — public API only.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CIFAR-10 dataset loading | Custom HTTP download + binary unpacking | `torchvision.datasets.CIFAR10` | Handles download, caching, transforms, train/test split |
| Package versioning | Manual version string in pyproject.toml | hatch-vcs (already configured) | Auto-derives from git tag; already working |
| Wheel/sdist building | `python setup.py bdist_wheel` | `uv build` | Already installed, PEP 517 compliant, no legacy tooling |
| TestPyPI upload | Manual multipart HTTP upload | `uv publish --publish-url` | Token auth built-in, duplicate detection |
| CIFAR-10 transforms | Manual PIL to tensor conversion | `torchvision.transforms.Compose([ToTensor(), Normalize()])` | Correct normalization, no import errors |
| README formatting validation | Check if README renders by guessing | `uvx twine check dist/*` | Tests actual PyPI Markdown rendering before upload |

**Key insight:** All packaging infrastructure (hatchling, hatch-vcs, uv) is already installed and configured. Phase 4 work is content and artifacts, not tooling setup.

---

## Common Pitfalls

### Pitfall 1: Building Before Tagging

**What goes wrong:** `uv build` is run before `git tag v1.0.0`. The wheel is stamped `0.0.0.dev32+gdd79cdb06.d20260224` instead of `1.0.0`.
**Why it happens:** hatch-vcs reads the git history at build time. If no release tag exists, it uses the development version formula.
**How to avoid:** ALWAYS run `git tag v1.0.0` BEFORE `uv build`. Verify the tag is present with `git tag --list`.
**Warning signs:** The dist/ filenames contain `.dev` or `+g` hash suffixes.

### Pitfall 2: torch Remaining in install_requires

**What goes wrong:** Users get CPU-only PyTorch forced on them via pip dependency resolution, even if they have CUDA. The package install fails or installs the wrong variant.
**Why it happens:** Current pyproject.toml has `"torch>=2.0"` in `dependencies`.
**How to avoid:** Remove torch from `[project].dependencies` in Phase 4. Document in README: "Install PyTorch first from https://pytorch.org, then pip install pcn-torch."
**Warning signs:** `pip show pcn-torch` shows `Requires: torch`.

### Pitfall 3: torchvision as Production Dependency

**What goes wrong:** Users installing `pip install pcn-torch` get torchvision as a required dependency, even though it's only used in the example.
**Why it happens:** Adding torchvision to `[project].dependencies` instead of `[dependency-groups].dev`.
**How to avoid:** `uv add --group dev torchvision` — keep it in dev only.
**Warning signs:** torchvision appears in `[project].dependencies` in pyproject.toml.

### Pitfall 4: CIFAR-10 Training Too Slow on CPU

**What goes wrong:** The example takes 30+ minutes on CPU instead of 5-10, discouraging users.
**Why it happens:** Too many inference steps (T_infer), too many epochs, or too-small batch size (more iterations at same epoch count).
**How to avoid:** Use `T_infer=20`, `batch_size=128`, `num_epochs=10`, `early_stop_threshold=1e-4`. Reducing T_infer from 50 to 20 reduces inference phase by 60%.
**Warning signs:** During development testing, a single epoch takes more than 90 seconds.

### Pitfall 5: Example Crashes If Dataset Download Fails

**What goes wrong:** Example exits with an HTTP error mid-training or before training starts.
**Why it happens:** `torchvision.datasets.CIFAR10(download=True)` tries to download at runtime; network errors raise exceptions.
**How to avoid:** Wrap dataset construction in a try/except with a friendly error message instructing the user to check their network connection.
**Warning signs:** No error handling around dataset loading.

### Pitfall 6: Flatten Transform in Example Causes Confusion

**What goes wrong:** Adding a Flatten() transform in the example is unnecessary and misleads readers about the API contract.
**Why it happens:** Forgetting that `trainer.py` already calls `x_batch.view(B, -1)` to flatten inputs on line 299.
**How to avoid:** In the example's torchvision transforms, use only ToTensor() + Normalize(). No Flatten() transform needed — the trainer handles flattening internally.
**Warning signs:** `transforms.Flatten()` appears in the example's transform pipeline.

### Pitfall 7: PredictiveCodingNetwork dims Ordering Confusion

**What goes wrong:** CIFAR-10 has 3072-dimensional input, but `dims[0]` in PredictiveCodingNetwork is the input dim. Writing `dims=[10, 512, 1024, 3072]` (output-to-input order) is wrong and causes shape mismatches.
**Why it happens:** The "top-down" naming in the paper's notation is easy to misread.
**How to avoid:** `dims=[3072, 1024, 1024, 512]` — dims[0] is input, dims[-1] is top latent, output_dim=10 goes to the readout.
**Warning signs:** Shape mismatch RuntimeError on the first batch.

### Pitfall 8: README Markdown Not Rendering on PyPI

**What goes wrong:** Code blocks show as plain text, or tables do not render on the PyPI package page.
**Why it happens:** PyPI renders README as Markdown only if `readme = "README.md"` is set and the file uses valid CommonMark/GFM.
**How to avoid:** Run `uvx twine check dist/*` after building — it validates README rendering. The current pyproject.toml already has `readme = "README.md"`.
**Warning signs:** twine check output shows `FAILED` or `WARNING` about description.

---

## Code Examples

### CIFAR-10 DataLoader Setup

```python
# Source: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
# and: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Precise CIFAR-10 normalization values (mean and std per channel)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    # NOTE: NO Flatten() here — trainer.py calls x_batch.view(B, -1) internally
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# batch_size=128: balances CPU throughput vs. per-batch PCN iteration count
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
```

### Model Construction for CIFAR-10

```python
# Source: pcn_torch/network.py (existing codebase)
from pcn_torch import PredictiveCodingNetwork, TrainConfig, RichCallback, train_pcn, test_pcn

model = PredictiveCodingNetwork(
    dims=[3072, 1024, 1024, 512],  # input=3072 (3x32x32), 2 hidden, top_latent=512
    activation="relu",
    output_dim=10,                  # 10 CIFAR-10 classes
    mode="classification",
)

config = TrainConfig(
    task="classification",
    T_infer=20,               # 20 inference steps (paper uses 50, but 20 saves ~60% CPU time)
    lr_infer=0.05,
    lr_learn=0.001,
    num_epochs=10,
    early_stop_threshold=1e-4,
    callback=RichCallback(),
)
```

### Full Example Script Skeleton

```python
# examples/cifar10.py
"""CIFAR-10 classification with pcn-torch.

Run:
    python examples/cifar10.py

Trains a 3-hidden-layer PCN on CIFAR-10 using local Hebbian-like rules
(no backpropagation). Achieves approximately 45-55% test accuracy in 5-15 minutes on CPU.

Requirements:
    pip install torchvision  # only needed for this example
"""

import time

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pcn_torch import (
    PredictiveCodingNetwork,
    RichCallback,
    TrainConfig,
    train_pcn,
    test_pcn,
)

# --- Configuration (fixed, no CLI args per design decision) ---
BATCH_SIZE = 128
NUM_EPOCHS = 10
T_INFER = 20
LR_INFER = 0.05
LR_LEARN = 0.001
DATA_ROOT = "./data"


def main() -> None:
    """Train a PCN on CIFAR-10 and print a summary."""
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    try:
        train_set = torchvision.datasets.CIFAR10(
            DATA_ROOT, train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            DATA_ROOT, train=False, download=True, transform=transform
        )
    except Exception as e:
        print(f"Failed to load CIFAR-10: {e}")
        print("Check your internet connection or download the dataset manually.")
        return

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Model: dims=[3072, 1024, 1024, 512] = input + 2 hidden + top_latent
    model = PredictiveCodingNetwork(
        dims=[3072, 1024, 1024, 512],
        activation="relu",
        output_dim=10,
        mode="classification",
    )

    config = TrainConfig(
        task="classification",
        T_infer=T_INFER,
        lr_infer=LR_INFER,
        lr_learn=LR_LEARN,
        num_epochs=NUM_EPOCHS,
        early_stop_threshold=1e-4,
        callback=RichCallback(),
    )

    # Train
    start = time.time()
    history = train_pcn(model, train_loader, config)
    elapsed = time.time() - start

    # Evaluate
    results = test_pcn(model, test_loader, config)

    # Summary
    print("\n" + "=" * 50)
    print("CIFAR-10 Training Summary")
    print("=" * 50)
    print(f"Final test accuracy:  {results['accuracy']:.1%}")
    print(f"Total training time: {elapsed / 60:.1f} minutes")
    print(f"Energy (epoch 1):    {history.energy.per_epoch[0]:.6f}")
    print(f"Energy (final):      {history.energy.per_epoch[-1]:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
```

### pyproject.toml Updates Required

```toml
# Source: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
# and: https://pypi.org/classifiers/

[project]
name = "pcn-torch"
dynamic = ["version"]
description = "Predictive Coding Networks in PyTorch"
readme = "README.md"
license = "MIT"
authors = [
    { name = "Enrique Mendez" },
]
requires-python = ">=3.10"
# CHANGED from original: torch is NOT listed here (it is a peer dependency)
dependencies = [
    "numpy>=2.2.6",
    "rich>=14.0",
]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Typing :: Typed",
]
keywords = [
    "predictive coding",
    "neural networks",
    "pytorch",
    "machine learning",
    "deep learning",
    "biologically plausible",
]

[project.urls]
Homepage = "https://github.com/emv-dev/pcn-torch"
Repository = "https://github.com/emv-dev/pcn-torch"
"Bug Tracker" = "https://github.com/emv-dev/pcn-torch/issues"
Paper = "https://arxiv.org/abs/2506.06332"

[dependency-groups]
dev = [
    "ruff>=0.15",
    "mypy>=1.19",
    "pytest>=9.0",
    "pytest-cov>=7.0",
    "torchvision>=0.15",  # required for examples/cifar10.py only
]
```

### TestPyPI Verification Workflow

```bash
# Source: https://docs.astral.sh/uv/guides/package/

# 1. Create release tag FIRST (hatch-vcs reads this at build time)
git tag v1.0.0
git push origin v1.0.0

# 2. Build (produces pcn_torch-1.0.0-*.whl and .tar.gz)
uv build

# 3. (Optional) Validate README renders correctly on PyPI
uvx twine check dist/*

# 4. Publish to TestPyPI (user provides token from test.pypi.org account)
uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --token $TESTPYPI_TOKEN

# 5. Verify install from TestPyPI
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  pcn-torch
python -c "import pcn_torch; print(pcn_torch.__version__)"
# Expected output: 1.0.0

# Real PyPI publish is done manually by the user:
# uv publish --token $PYPI_TOKEN
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `python -m build` + `twine upload` | `uv build` + `uv publish` | 2024-2025 | Single tool; no need to install separate build/twine |
| `setup.py` + `setup.cfg` | `pyproject.toml` only | PEP 517/518 (2020+) | Already in place in this project |
| Username + password for PyPI | API token only | 2023 | PyPI deprecated password auth; tokens required |
| `setuptools-scm` for VCS versions | `hatch-vcs` | 2022+ | Already in place in this project |

**Deprecated/outdated:**
- `python setup.py bdist_wheel`: Legacy setuptools command; PEP 517 build via `uv build` is standard
- `twine upload --username __token__`: Still works, but `uv publish --token` is preferred when uv is available
- `torch` in `install_requires`: Must be removed — peer dependency pattern is standard for ML libraries with hardware-specific wheels

---

## Open Questions

1. **PyPI name `pcn-torch` status**
   - What we know: WebSearch found no package named `pcn-torch`; direct pypi.org access was blocked by CloudFlare challenge
   - What's unclear: Cannot 100% confirm via direct pypi.org page visit
   - Recommendation: The first TestPyPI publish attempt will confirm immediately. If it fails with "name already taken," the name conflict is resolved then.

2. **Expected CIFAR-10 accuracy range**
   - What we know: Standard backprop MLP on CIFAR-10 achieves 50-55% with 3-4 hidden layers. PCN MLP achieves comparable performance per literature.
   - What's unclear: Exact accuracy with the specific hyperparameters recommended here (T_infer=20, specific dims)
   - Recommendation: README should state "approximately 45-55% test accuracy" and note this demonstrates the algorithm works correctly, not that it's state-of-the-art for CIFAR-10

3. **Training time variance across hardware**
   - What we know: T_infer=20, batch_size=128, 10 epochs should be significantly faster than T_infer=50, batch_size=64
   - What's unclear: Exact wall-clock time; depends on CPU (i5 vs. M-series vs. server Xeon)
   - Recommendation: README should say "approximately 5-15 minutes on CPU" rather than "5-10 minutes" to be conservative

4. **torchvision version compatibility with torch 2.10**
   - What we know: torch 2.10+cpu is installed. torchvision is versioned to match torch (e.g., torchvision 0.22 for torch 2.7).
   - What's unclear: What torchvision version aligns with torch 2.10.
   - Recommendation: Use `torchvision>=0.15` as the floor (very conservative) and let uv resolve. If there are conflicts, pin to the version that matches torch 2.10.

---

## Sources

### Primary (HIGH confidence)
- `https://github.com/ofek/hatch-vcs/blob/master/README.md` — hatch-vcs tag format, version file generation
- `https://docs.astral.sh/uv/guides/package/` — uv build + publish workflow, TestPyPI configuration, token authentication
- `https://pypi.org/classifiers/` — Exact PyPI classifier strings for ML libraries
- `src/pcn_torch/__init__.py` (read directly) — Current exports and `__all__` definition (already complete)
- `pyproject.toml` (read directly) — Current build setup; torch listed in deps (confirmed needs removal)
- `src/pcn_torch/trainer.py` (read directly) — `x_batch.view(B, -1)` on line 299 (Pitfall 6 basis)
- `src/pcn_torch/network.py` (read directly) — `dims` ordering, PredictiveCodingNetwork constructor

### Secondary (MEDIUM confidence)
- `https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html` — CIFAR10 API, root/train/download params
- `https://packaging.python.org/en/latest/guides/writing-pyproject-toml/` — classifiers, keywords, project URLs guidance
- `https://twine.readthedocs.io/en/latest/` — twine check command for README validation
- `https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html` — CIFAR-10 transforms and DataLoader setup

### Tertiary (LOW confidence)
- Multiple WebSearch results indicating MLP achieves 50-55% on CIFAR-10 — consistent across sources but no single authoritative benchmark
- WebSearch confirming `pcn-torch` name is not on PyPI — CloudFlare blocked direct access; inferred from absence in search results

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — hatchling/hatch-vcs/uv already configured and verified; torchvision is official PyTorch library
- Architecture patterns: HIGH — hatch-vcs tag workflow verified via official docs; CIFAR-10 example structure follows existing trainer API (verified by reading trainer.py and network.py)
- Pitfalls: HIGH — torch-in-deps and tag-before-build are verified against actual code state; CIFAR-10 pitfalls verified against trainer.py internals
- CIFAR-10 accuracy expectations: MEDIUM — consistent across multiple sources but not verified for this exact hyperparameter combination

**Research date:** 2026-02-25
**Valid until:** 2026-03-25 (30 days — packaging APIs and PyPI name availability are stable)

**Critical pre-work for Phase 4 (must happen in this order):**
1. Remove `"torch>=2.0"` from `[project].dependencies` in pyproject.toml
2. Add `torchvision>=0.15` to `[dependency-groups].dev` in pyproject.toml
3. Add PyPI metadata (classifiers, keywords, urls, authors) to pyproject.toml
4. Create `examples/cifar10.py`
5. Rewrite `README.md`
6. `git tag v1.0.0` (BEFORE building)
7. `uv build`
8. `uvx twine check dist/*`
9. `uv publish --publish-url https://test.pypi.org/legacy/ --token $TESTPYPI_TOKEN`
10. Verify install from TestPyPI
