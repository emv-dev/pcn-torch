# Phase 1: Foundation - Research

**Researched:** 2026-02-20
**Domain:** Python package scaffold, PyTorch nn.Module, activation functions, type system
**Confidence:** HIGH

## Summary

Phase 1 delivers the package scaffold (pyproject.toml, src layout, CI, license), core type aliases, activation functions with hand-coded derivatives, and PCNLayer as a configurable nn.Module. The scope is well-defined: after this phase, developers can `pip install -e .` and `from pcn_torch import PCNLayer` with a working forward pass, and CI passes across Python 3.10-3.13.

The standard approach is established Python packaging with hatchling + hatch-vcs as the build backend, src layout for import safety, and a flat package structure under `src/pcn_torch/`. The PCNLayer follows the nn.Linear constructor convention exactly (`PCNLayer(in_features, out_features, activation='relu')`) with `nn.Parameter` for weight registration and Xavier uniform initialization. Activation functions are paired with hand-coded derivatives in a module-level dictionary, looked up by string name via `get_activation(name)`.

No novel patterns or unusual tooling are needed. Every component in Phase 1 follows well-documented, standard Python and PyTorch conventions. The primary risk is configuration correctness (pyproject.toml settings, hatch-vcs fallback version, CI matrix YAML) rather than algorithmic complexity.

**Primary recommendation:** Use the exact reference implementation patterns from the paper (arXiv:2506.06332v1) for PCNLayer, adapting the constructor signature to follow nn.Linear convention per the user's locked decision. Package with hatchling + hatch-vcs in src layout. Keep everything minimal -- this phase establishes the foundation, not the features.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### PCNLayer API style
- Constructor follows nn.Linear convention: `PCNLayer(in_features, out_features, activation='relu')`
- Positional dimensions, keyword activation -- familiar to any PyTorch user
- No bias term (per paper specification)

### Claude's Discretion
- **Activation interface**: String-based lookup via `get_activation(name)` returning `(fn, fn_deriv)` pair. Built-ins: relu, tanh, sigmoid, identity. Implementation details (registry pattern, module-level dict, etc.) are Claude's choice.
- **forward() return type**: Returns `(prediction, preactivation)` tuple since preactivation is required for derivative computation in Phase 2's error calculation. Exact naming and documentation are Claude's choice.
- **Type aliases**: `ActivationFn`, `ActivationDeriv`, `ActivationPair` in `_types.py`. Exact type definitions and any additional type aliases are Claude's choice.
- **Package scaffold**: pyproject.toml with hatchling build backend, src layout vs flat layout, CI configuration, ruff/mypy settings -- all Claude's choice following PyTorch ecosystem conventions.
- **Weight initialization**: Xavier uniform (per architecture research). Exact implementation is Claude's choice.
- **Import structure**: Flat imports via `__init__.py` re-exports. What's exported at this phase is Claude's choice (at minimum: PCNLayer).

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

## Standard Stack

The established libraries/tools for this phase:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | >=3.10,<3.14 | Language runtime | PyTorch 2.x requires >=3.10. CI matrix covers 3.10-3.13. |
| PyTorch | >=2.0 | `nn.Module`, `nn.Parameter`, `nn.init`, `torch.Tensor` | Project requirement. Peer dependency -- users install first. |
| hatchling | >=1.28 | Build backend | PEP 621 compliant, zero boilerplate, src layout support. Standard for pure-Python packages. |
| hatch-vcs | >=0.4 | Git-tag version derivation | Eliminates manual version bumping. Generates `_version.py` automatically. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ruff | >=0.15 | Linter + formatter | Every commit. Replaces Black + Flake8 + isort. |
| mypy | >=1.19 | Static type checker | Every commit. `disallow_untyped_defs = true`. |
| pytest | >=9.0 | Test framework | Every commit. CPU-only PyTorch sufficient. |
| pytest-cov | >=7.0 | Coverage reporting | Every CI run. Branch coverage enabled. |
| pre-commit | >=4.5 | Git hooks | Local development. Runs ruff + mypy before commit. |
| uv | >=0.10 | Project/env management | Development and CI. 10-100x faster than pip. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| hatchling | setuptools | Setuptools requires more boilerplate (setup.py/cfg/MANIFEST.in). Hatchling is pyproject.toml-only. |
| hatchling | uv_build | uv_build lacks hatch-vcs plugin for git-based versioning. Too new for library publishing. |
| src layout | flat layout | Flat layout risks importing source tree during testing instead of installed package. src layout is the modern standard. |
| mypy | ty (Astral) | ty is 10-60x faster but still early development as of Feb 2026. Not production-ready for CI. |
| TypeAlias (3.10) | type statement (3.12) | type statement is cleaner but requires Python 3.12+. Use TypeAlias for 3.10 compatibility. |

**Installation (dev environment):**
```bash
uv sync --all-extras --dev
```

## Architecture Patterns

### Recommended Project Structure
```
pcn-torch/
├── pyproject.toml              # Build config, all tool configs
├── LICENSE                     # MIT license
├── README.md                   # Already exists
├── src/
│   └── pcn_torch/
│       ├── __init__.py         # Public API: PCNLayer, get_activation
│       ├── _types.py           # Type aliases: ActivationFn, ActivationDeriv, ActivationPair
│       ├── _version.py         # Auto-generated by hatch-vcs (gitignored)
│       ├── activations.py      # Activation functions + derivatives + registry
│       └── layers.py           # PCNLayer (nn.Module)
└── tests/
    ├── __init__.py
    ├── test_activations.py     # Test each activation + derivative pair
    └── test_layers.py          # Test PCNLayer construction, forward, shapes
```

**Key decisions for this structure:**
- `src/` layout prevents accidental imports of uninstalled source during testing.
- `_types.py` with leading underscore signals internal module (not part of public API).
- `_version.py` is auto-generated by hatch-vcs build hook; add to `.gitignore`.
- Flat package (no subpackages) -- appropriate for ~5 modules in v1.

### Pattern 1: PCNLayer Following nn.Linear Convention

**What:** PCNLayer constructor mirrors `nn.Linear(in_features, out_features, bias=True)` but with `activation` keyword instead of `bias`, and no bias term.

**When to use:** This is the locked decision from the user. The constructor MUST be `PCNLayer(in_features, out_features, activation='relu')`.

**Reference implementation from paper (arXiv:2506.06332v1, Section 5.2.1):**
```python
# Paper's original constructor:
class PCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim,
                 activation_fn=torch.relu,
                 activation_deriv=lambda a: (a > 0).float()):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv

    def forward(self, x_above):
        with autocast(device_type='cuda'):
            a = x_above @ self.W.T
            x_hat = self.activation_fn(a)
            return x_hat, a
```

**Our adaptation (following user's nn.Linear convention):**
```python
# Source: User locked decision + paper reference
class PCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *,
                 activation: str = "relu") -> None:
        super().__init__()
        act_fn, act_deriv = get_activation(activation)
        self.activation_fn = act_fn
        self.activation_deriv = act_deriv
        # Weight shape: (out_features, in_features) — same as nn.Linear
        # Maps FROM in_features TO out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preactivation = x @ self.weight.T
        prediction = self.activation_fn(preactivation)
        return prediction, preactivation

    def extra_repr(self) -> str:
        return (f"in_features={self.weight.shape[1]}, "
                f"out_features={self.weight.shape[0]}")
```
**Confidence:** HIGH -- based on paper's reference code + user's locked nn.Linear convention.

**Critical naming note:** The paper names the parameter `self.W` and uses `in_dim`/`out_dim`. The user's locked decision uses `in_features`/`out_features` (nn.Linear convention). The adaptation renames `self.W` to `self.weight` to match nn.Linear's attribute name. This makes `layer.weight` work consistently with PyTorch conventions like `model.named_parameters()`.

**Dimension semantics:** In the paper, `in_dim` = d_{l+1} (layer above) and `out_dim` = d_l (current layer). The weight maps FROM the layer above TO the current layer. The constructor signature `PCNLayer(in_features, out_features)` means: "accepts input of `in_features` dimensions, produces output of `out_features` dimensions." Weight shape is `(out_features, in_features)` -- identical to nn.Linear.

### Pattern 2: Activation Function Registry

**What:** Module-level dictionary mapping string names to `(fn, fn_deriv)` tuples, with a `get_activation()` lookup function.

**When to use:** Every time a PCNLayer is constructed with a string activation name.

**Recommended implementation:**
```python
# Source: Paper's activation defaults + user's string-based lookup decision
from typing import TypeAlias
from collections.abc import Callable
import torch
from torch import Tensor

ActivationFn: TypeAlias = Callable[[Tensor], Tensor]
ActivationDeriv: TypeAlias = Callable[[Tensor], Tensor]
ActivationPair: TypeAlias = tuple[ActivationFn, ActivationDeriv]

def _relu(x: Tensor) -> Tensor:
    return torch.relu(x)

def _relu_deriv(x: Tensor) -> Tensor:
    return (x > 0).float()

def _tanh(x: Tensor) -> Tensor:
    return torch.tanh(x)

def _tanh_deriv(x: Tensor) -> Tensor:
    t = torch.tanh(x)
    return 1.0 - t * t

def _sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)

def _sigmoid_deriv(x: Tensor) -> Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 - s)

def _identity(x: Tensor) -> Tensor:
    return x

def _identity_deriv(x: Tensor) -> Tensor:
    return torch.ones_like(x)

_ACTIVATIONS: dict[str, ActivationPair] = {
    "relu": (_relu, _relu_deriv),
    "tanh": (_tanh, _tanh_deriv),
    "sigmoid": (_sigmoid, _sigmoid_deriv),
    "identity": (_identity, _identity_deriv),
}

def get_activation(name: str) -> ActivationPair:
    """Look up an activation function and its derivative by name."""
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]
```
**Confidence:** HIGH -- standard registry pattern, derivatives verified mathematically.

### Pattern 3: Type Aliases with TypeAlias (Python 3.10+)

**What:** Use `typing.TypeAlias` for explicit type alias declarations. This is the recommended approach for libraries targeting Python 3.10+ (the `type` statement requires 3.12+).

**When to use:** In `_types.py` for all shared type definitions.

**Recommended implementation:**
```python
# Source: Python typing specification, mypy documentation
from __future__ import annotations
from typing import TypeAlias
from collections.abc import Callable
from torch import Tensor

ActivationFn: TypeAlias = Callable[[Tensor], Tensor]
ActivationDeriv: TypeAlias = Callable[[Tensor], Tensor]
ActivationPair: TypeAlias = tuple[ActivationFn, ActivationDeriv]
```
**Confidence:** HIGH -- verified against Python typing spec and mypy docs.

**Notes:**
- Use `TypeAlias` from `typing` (Python 3.10+), NOT the `type` statement (Python 3.12+).
- `from __future__ import annotations` enables PEP 604 union syntax and deferred evaluation in 3.10.
- `collections.abc.Callable` is preferred over `typing.Callable` since Python 3.9+.
- mypy handles these Callable aliases correctly with `disallow_untyped_defs = true`.

### Pattern 4: hatchling + hatch-vcs Configuration

**What:** pyproject.toml configured for hatchling build backend with src layout and git-tag versioning.

**When to use:** The single configuration file for the entire project.

**Recommended configuration:**
```toml
# Source: hatchling docs, hatch-vcs GitHub README
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "pcn-torch"
dynamic = ["version"]
description = "Predictive Coding Networks in PyTorch"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
]

[tool.hatch.version]
source = "vcs"
fallback-version = "0.0.0.dev0"

[tool.hatch.build.hooks.vcs]
version-file = "src/pcn_torch/_version.py"

[tool.hatch.build.targets.wheel]
packages = ["src/pcn_torch"]
```
**Confidence:** HIGH -- verified against hatchling build docs and hatch-vcs README.

**Critical detail:** The `fallback-version = "0.0.0.dev0"` setting is REQUIRED. Without it, builds fail when no git tag exists (which will be the case during initial development before the first release tag). This was verified from the hatch-vcs documentation.

**Critical detail:** The `version-file` build hook auto-generates `src/pcn_torch/_version.py` during builds. This file must be in `.gitignore` since it is generated, not source-controlled.

### Pattern 5: GitHub Actions CI with uv

**What:** CI workflow using `astral-sh/setup-uv` for Python version matrix testing.

**When to use:** On every PR and push to main.

**Recommended workflow:**
```yaml
# Source: uv docs (GitHub integration), GitHub Actions docs
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"
      - run: uv sync --dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"
      - run: uv sync --dev
      - run: uv run mypy src/pcn_torch

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"
          python-version: ${{ matrix.python-version }}
      - run: uv sync --dev
      - run: uv run pytest tests/ -v --cov=pcn_torch --cov-report=term-missing
```
**Confidence:** HIGH -- verified against uv GitHub integration docs and astral-sh/setup-uv action.

**Note on PyTorch in CI:** `uv sync` will install CPU-only PyTorch from PyPI, which is correct for CI. No special CUDA configuration needed. CPU-only PyTorch is sufficient for unit testing PCNLayer forward passes and activation functions.

### Anti-Patterns to Avoid
- **DO NOT use `setup.py` or `setup.cfg`:** hatchling + pyproject.toml handles everything. No legacy files.
- **DO NOT put tests inside the package:** Tests go in `tests/` at the repo root, not inside `src/pcn_torch/`.
- **DO NOT use `typing.Callable`:** Use `collections.abc.Callable` instead (preferred since Python 3.9).
- **DO NOT use lambda for activation derivatives:** Named functions are easier to test, debug, and type-check. The paper uses `lambda a: (a > 0).float()` but we should use named functions.
- **DO NOT use `autocast` in PCNLayer.forward():** The paper wraps forward() in `autocast(device_type='cuda')`, but this should be handled at the training loop level (Phase 3), not in the layer itself. Embedding autocast in the layer makes CPU testing harder and couples device assumptions to the layer.
- **DO NOT name the parameter `self.W`:** Use `self.weight` to match nn.Linear convention and make `model.named_parameters()` output consistent with PyTorch norms.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Weight initialization | Custom random init | `nn.init.xavier_uniform_()` | Correct fan_in/fan_out calculation, gain parameter handling. Formula: `U(-a, a)` where `a = gain * sqrt(6 / (fan_in + fan_out))`. |
| Package versioning | Manual `__version__` string | hatch-vcs + `_version.py` | Derives from git tags automatically. No manual bumping. No version drift between pyproject.toml and code. |
| Linting + formatting | Multiple tools (black, flake8, isort) | Ruff (single tool) | 50-150x faster. Single config section in pyproject.toml. Replaces 3+ tools. |
| Environment management | pip + virtualenv + pip-tools | uv | 10-100x faster. Lockfile-based. Handles Python version management too. |
| CI Python matrix | Manual version management | `astral-sh/setup-uv@v7` with matrix | Built-in caching, Python version management, uv integration. |

**Key insight:** Phase 1 is entirely standard Python packaging + a simple nn.Module. Every component has a well-established solution. The value is in correct configuration, not novel code.

## Common Pitfalls

### Pitfall 1: Missing hatch-vcs Fallback Version
**What goes wrong:** Build fails with `LookupError` when no git tag exists.
**Why it happens:** hatch-vcs tries to derive version from git tags. During initial development before any `v0.1.0` tag, there are no tags.
**How to avoid:** Set `fallback-version = "0.0.0.dev0"` in `[tool.hatch.version]`. This provides a valid version when no tags exist.
**Warning signs:** `uv build` fails with version-related errors. `pip install -e .` fails.

### Pitfall 2: _version.py Checked into Git
**What goes wrong:** Version conflicts between generated file and git state. Build produces stale versions.
**Why it happens:** hatch-vcs generates `_version.py` during builds. If committed, it becomes a stale snapshot.
**How to avoid:** Add `src/pcn_torch/_version.py` to `.gitignore`. The file is generated fresh on every build.
**Warning signs:** `git status` shows `_version.py` as modified after every build.

### Pitfall 3: src Layout Package Discovery Failure
**What goes wrong:** `pip install -e .` succeeds but `import pcn_torch` fails with `ModuleNotFoundError`.
**Why it happens:** Hatchling does not find the package because `packages` is not configured for src layout.
**How to avoid:** Set `packages = ["src/pcn_torch"]` in `[tool.hatch.build.targets.wheel]`. This tells hatchling where the package lives and collapses the `src/` prefix in the wheel.
**Warning signs:** Built wheel contains no `.py` files. `import pcn_torch` fails after install.

### Pitfall 4: Xavier Initialization with Wrong Gain for ReLU
**What goes wrong:** Suboptimal weight initialization that may affect convergence in later phases.
**Why it happens:** `xavier_uniform_` defaults to `gain=1.0`, which is optimal for linear/tanh activations. For ReLU, the theoretically optimal gain is `sqrt(2)` (He initialization). However, the paper explicitly uses Xavier, not He.
**How to avoid:** Use `nn.init.xavier_uniform_(self.weight)` with default `gain=1.0`, matching the paper exactly. Do NOT use `gain=nn.init.calculate_gain('relu')` -- this would deviate from the paper.
**Warning signs:** None in Phase 1. This is a theoretical concern that only manifests in Phase 3 training. Document it as a known design choice.

### Pitfall 5: ReLU Derivative at Zero
**What goes wrong:** Mathematical discontinuity at `x=0`. The derivative is technically undefined.
**Why it happens:** ReLU has a "kink" at zero: left derivative = 0, right derivative = 1.
**How to avoid:** Follow PyTorch and the paper's convention: `relu_deriv(x) = (x > 0).float()`. This sets the derivative to 0 at x=0. This is the standard choice in deep learning (TensorFlow and PyTorch both use this convention).
**Warning signs:** None practically. The probability of any value being exactly 0.0 in float32 is negligible.

### Pitfall 6: Sigmoid Derivative Numerical Stability
**What goes wrong:** For very large or very small inputs, `sigmoid(x)` saturates to 0 or 1, making the derivative `s * (1-s)` vanish to 0 or produce numerical noise.
**How to avoid:** Use `torch.sigmoid()` which is numerically stable internally. The derivative `s * (1-s)` is computed from the stable sigmoid output, which is sufficient. No clamping needed -- the derivative naturally approaches 0 at extremes, which is mathematically correct.
**Warning signs:** None in practice. Sigmoid derivative vanishing is a feature, not a bug -- it reflects the actual gradient landscape.

### Pitfall 7: `from __future__ import annotations` Breaking Runtime Type Access
**What goes wrong:** Code that introspects type annotations at runtime (e.g., `get_type_hints()`) gets strings instead of actual types.
**Why it happens:** `from __future__ import annotations` converts all annotations to strings (PEP 563). This is fine for mypy but breaks runtime annotation consumers.
**How to avoid:** Only use `from __future__ import annotations` in modules where no runtime annotation inspection occurs. For `_types.py` it is fine since type aliases are used by type checkers, not at runtime. For `activations.py` and `layers.py`, it is fine because we do not inspect annotations at runtime.
**Warning signs:** `TypeError` when using `get_type_hints()` or `dataclasses.fields()` on annotated classes.

### Pitfall 8: mypy Errors with torch.Tensor Type Annotations
**What goes wrong:** mypy may report errors on PyTorch tensor operations due to incomplete type stubs.
**Why it happens:** PyTorch's type stubs (`torch.pyi`) do not cover every overloaded operation perfectly. Some tensor operations return `Any` or have imprecise signatures.
**How to avoid:** Install `torch` type stubs if needed. Use `# type: ignore[...]` sparingly for known mypy/PyTorch incompatibilities. Configure mypy with `warn_return_any = true` to catch cases where `Any` leaks into the type system.
**Warning signs:** mypy errors on basic tensor operations like `@` (matmul), `.T`, or `.float()`.

## Code Examples

Verified patterns from official sources:

### nn.Module Subclass Pattern (from PyTorch docs)
```python
# Source: PyTorch 2.10 documentation - torch.nn.Module
class PCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *,
                 activation: str = "relu") -> None:
        super().__init__()  # MUST call super().__init__()

        # nn.Parameter registers tensor as learnable parameter
        # Shape matches nn.Linear convention: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Xavier uniform initialization matching paper
        nn.init.xavier_uniform_(self.weight)

        # Store activation function and derivative
        act_fn, act_deriv = get_activation(activation)
        self.activation_fn = act_fn
        self.activation_deriv = act_deriv

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x shape: (batch_size, in_features)
        # preactivation shape: (batch_size, out_features)
        # prediction shape: (batch_size, out_features)
        preactivation = x @ self.weight.T
        prediction = self.activation_fn(preactivation)
        return prediction, preactivation

    def extra_repr(self) -> str:
        return (f"in_features={self.weight.shape[1]}, "
                f"out_features={self.weight.shape[0]}")
```

### Xavier Uniform Initialization (from PyTorch docs)
```python
# Source: PyTorch 2.10 nn.init documentation
# Signature: torch.nn.init.xavier_uniform_(tensor, gain=1.0, generator=None)
# Formula: U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
# Default gain=1.0 is correct for the paper's specification

self.weight = nn.Parameter(torch.empty(out_features, in_features))
nn.init.xavier_uniform_(self.weight)  # gain=1.0 default matches paper
```

### nn.Linear Source Pattern (from PyTorch source)
```python
# Source: pytorch/torch/nn/modules/linear.py
# nn.Linear stores weight as Parameter with shape (out_features, in_features)
# Uses factory_kwargs for device/dtype handling
# Calls reset_parameters() for initialization
# forward() delegates to F.linear(input, self.weight, self.bias)

# Key convention: weight shape is (out_features, in_features)
# This is what PCNLayer must match
```

### hatch-vcs Version Import Pattern
```python
# Source: hatch-vcs GitHub README
# In src/pcn_torch/__init__.py:
from pcn_torch._version import __version__

# In src/pcn_torch/_version.py (auto-generated, gitignored):
# __version__ = "0.0.0.dev0"  # or derived from git tag
# __version_tuple__ = (0, 0, 0, "dev0")
```

### Ruff + mypy Configuration
```toml
# Source: Ruff docs, mypy docs
[tool.ruff]
target-version = "py310"
line-length = 88
src = ["src"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "W",    # pycodestyle warnings
    "I",    # isort
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
]

[tool.ruff.lint.isort]
known-first-party = ["pcn_torch"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
# May need per-module overrides for torch stubs
[[tool.mypy.overrides]]
module = "torch.*"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.coverage.run]
source_pkgs = ["pcn_torch"]
branch = true
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `typing.Callable` | `collections.abc.Callable` | Python 3.9+ | Use abc version in all new code |
| `TypeAlias` annotation | `type` statement | Python 3.12 | Use TypeAlias for 3.10 compat |
| setuptools + setup.py | hatchling + pyproject.toml | 2023-2024 | Single config file, no boilerplate |
| pip + virtualenv | uv | 2024-2025 | 10-100x faster, lockfile-based |
| Black + Flake8 + isort | Ruff | 2023-2025 | Single tool, 50-150x faster |
| `typing.Tuple[X, Y]` | `tuple[X, Y]` (lowercase) | Python 3.9+ | Use lowercase builtins in annotations |
| `typing.Dict[K, V]` | `dict[K, V]` (lowercase) | Python 3.9+ | Use lowercase builtins in annotations |

**Deprecated/outdated:**
- `setup.py` / `setup.cfg`: Replaced by pyproject.toml for pure-Python packages.
- `typing.Callable`, `typing.Tuple`, `typing.Dict`: Use `collections.abc.Callable`, `tuple`, `dict` in Python 3.10+.
- Black formatter: Replaced by Ruff's built-in formatter (100% compatible, 100x faster).

## Discretion Recommendations

For areas marked as Claude's discretion in CONTEXT.md, here are research-backed recommendations:

### Activation Interface
**Recommend:** Module-level `dict[str, ActivationPair]` with `get_activation(name: str) -> ActivationPair` lookup. Use named functions (not lambdas) for all built-ins: `_relu`, `_relu_deriv`, `_tanh`, `_tanh_deriv`, `_sigmoid`, `_sigmoid_deriv`, `_identity`, `_identity_deriv`.

**Rationale:** Named functions are testable individually, produce better stack traces, and satisfy mypy's type checking. A flat dict is simpler than a registry class pattern and sufficient for 4 built-in activations. The leading underscore on function names (`_relu`) signals they are internal; users interact via string names.

### forward() Return Type
**Recommend:** Return `tuple[Tensor, Tensor]` with semantic names in docstring: `(prediction, preactivation)`. Match the paper's `(x_hat, a)` semantics.

**Rationale:** Named tuples or dataclasses add import complexity for what is a simple 2-element return. The tuple is consumed by `compute_errors()` in Phase 2, which needs both values. Using Python 3.10+ lowercase `tuple` syntax in type annotations.

### Type Aliases in _types.py
**Recommend:**
```python
from typing import TypeAlias
from collections.abc import Callable
from torch import Tensor

ActivationFn: TypeAlias = Callable[[Tensor], Tensor]
ActivationDeriv: TypeAlias = Callable[[Tensor], Tensor]
ActivationPair: TypeAlias = tuple[ActivationFn, ActivationDeriv]
```

**Rationale:** `TypeAlias` is the correct annotation for Python 3.10-3.11. Using `collections.abc.Callable` (not `typing.Callable`). Three aliases are sufficient for Phase 1 -- more can be added in later phases as needed.

### Package Scaffold
**Recommend:** src layout (`src/pcn_torch/`), hatchling + hatch-vcs, configuration-only pyproject.toml (no setup.py, no setup.cfg, no MANIFEST.in).

**Rationale:** src layout is the modern standard for Python libraries (prevents test pollution). hatchling is the recommended build backend for pure-Python packages. hatch-vcs eliminates manual version management. All tool configs consolidated in pyproject.toml.

### Weight Initialization
**Recommend:** `nn.init.xavier_uniform_(self.weight)` with default `gain=1.0` in the `__init__` method, called directly (not via a separate `reset_parameters()` method).

**Rationale:** Matches the paper exactly. The paper's reference code calls `nn.init.xavier_uniform_(self.W)` directly in `__init__`. A separate `reset_parameters()` method is an nn.Linear pattern that is unnecessary here since PCNLayer has no bias and only one parameter.

### Import Structure
**Recommend:** Export `PCNLayer`, `get_activation`, and `__version__` from `__init__.py` at Phase 1. Add `ActivationFn`, `ActivationDeriv`, `ActivationPair` as secondary exports for advanced users.

**Rationale:** `PCNLayer` is the minimum required by the user. `get_activation` enables users to inspect available activations. `__version__` is standard. Type aliases are useful for users who want to create custom activations.

```python
# src/pcn_torch/__init__.py
from pcn_torch._version import __version__
from pcn_torch.activations import get_activation
from pcn_torch.layers import PCNLayer
from pcn_torch._types import ActivationFn, ActivationDeriv, ActivationPair

__all__ = [
    "__version__",
    "PCNLayer",
    "get_activation",
    "ActivationFn",
    "ActivationDeriv",
    "ActivationPair",
]
```

## Open Questions

Things that could not be fully resolved:

1. **mypy compatibility with PyTorch tensor operations**
   - What we know: PyTorch's type stubs have known gaps. Operations like `@` (matmul), `.T`, and `.float()` may produce mypy errors.
   - What's unclear: The exact set of `# type: ignore` annotations that will be needed.
   - Recommendation: Start with strict mypy config (`disallow_untyped_defs = true`), add targeted ignores as needed during implementation. Use `[[tool.mypy.overrides]]` for torch module.

2. **Pre-commit hook configuration details**
   - What we know: pre-commit 4.5+ with `pre-commit-uv` for faster installation is the standard.
   - What's unclear: Whether to include pre-commit setup in Phase 1 or defer to later. It is development tooling, not package functionality.
   - Recommendation: Include a basic `.pre-commit-config.yaml` in Phase 1 since it enforces code quality from the start, but it is not blocking for the phase's success criteria.

3. **Context7 unavailability for library docs**
   - What we know: Context7 was unavailable during research. Findings are based on official docs via WebFetch and WebSearch.
   - What's unclear: Whether any PyTorch or hatchling API details have changed very recently.
   - Recommendation: HIGH confidence in all findings regardless, since the APIs researched (nn.Module, nn.Parameter, nn.init, hatchling build config) are stable and well-documented.

## Sources

### Primary (HIGH confidence)
- [PyTorch 2.10 nn.Module docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) -- Subclassing pattern, Parameter, forward(), inherited methods
- [PyTorch nn.Linear source](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py) -- Constructor convention, weight shape, reset_parameters
- [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332) -- PCNLayer reference implementation (Section 5.2.1), Xavier init, activation derivatives
- [hatchling build docs](https://hatch.pypa.io/latest/config/build/) -- src layout `packages` config, build-system section
- [hatch-vcs GitHub](https://github.com/ofek/hatch-vcs) -- Version source config, fallback-version, version-file build hook
- [uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/) -- setup-uv action, matrix strategy, caching

### Secondary (MEDIUM confidence)
- [Python typing spec - aliases](https://typing.python.org/en/latest/spec/aliases.html) -- TypeAlias vs type statement, Python version compatibility
- [mypy docs - kinds of types](https://mypy.readthedocs.io/en/stable/kinds_of_types.html) -- Callable aliases, TypeAlias handling
- [GitHub Actions Python docs](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python) -- Matrix strategy YAML syntax
- [GitHub Actions Python setup 2025](https://ber2.github.io/posts/2025_github_actions_python/) -- uv-based CI workflow pattern, job structure
- [Sebastian Raschka - ReLU derivative](https://sebastianraschka.com/faq/docs/relu-derivative.html) -- ReLU derivative at zero convention (0 by standard)
- [PyTorch xavier_uniform_ docs](https://docs.pytorch.org/docs/stable/nn.init.html) -- Signature, formula, gain parameter

### Tertiary (LOW confidence)
- WebSearch results on sigmoid numerical stability -- Confirmed that `torch.sigmoid()` handles internal stability, derivative `s*(1-s)` is safe. Multiple sources agree but no single authoritative source checked.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All tools verified against official documentation. Versions current as of Feb 2026.
- Architecture: HIGH -- Based directly on paper's reference implementation (arXiv:2506.06332v1 Section 5.2.1) + hatchling/uv official docs.
- Pitfalls: HIGH -- Configuration pitfalls verified against official docs. Mathematical pitfalls (derivative edge cases) verified against standard deep learning conventions.
- Discretion recommendations: HIGH -- All recommendations follow established patterns with verified sources.

**Research date:** 2026-02-20
**Valid until:** 2026-03-22 (30 days -- all components are stable, slow-moving)
