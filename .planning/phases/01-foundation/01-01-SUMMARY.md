---
phase: 01-foundation
plan: 01
subsystem: core
tags: [pytorch, nn-module, activations, hatchling, hatch-vcs, ruff, mypy, ci]

# Dependency graph
requires: []
provides:
  - "Installable pcn-torch package with src layout"
  - "PCNLayer nn.Module with configurable activation"
  - "Activation registry: relu, tanh, sigmoid, identity with hand-coded derivatives"
  - "Type aliases: ActivationFn, ActivationDeriv, ActivationPair"
  - "CI workflow for lint, typecheck, test across Python 3.10-3.13"
affects:
  - "02-core-model: builds PredictiveCodingNetwork on top of PCNLayer"
  - "03-training: training loops use activation_deriv for local error updates"
  - "04-integration: PyPI publishing uses hatchling + hatch-vcs build system"

# Tech tracking
tech-stack:
  added:
    - "torch>=2.0 (nn.Module, nn.Parameter, nn.init)"
    - "hatchling + hatch-vcs (build system, git-tag versioning)"
    - "ruff>=0.15 (linter + formatter)"
    - "mypy>=1.19 (static type checker)"
    - "pytest>=9.0 + pytest-cov>=7.0 (testing + coverage)"
    - "uv (project/env management, CI)"
  patterns:
    - "src layout with hatchling packages config"
    - "String-based activation registry via get_activation(name)"
    - "nn.Linear constructor convention: PCNLayer(in_features, out_features, activation=...)"
    - "TYPE_CHECKING guard for type-only imports (ruff TCH001)"
    - "from __future__ import annotations in all modules"

key-files:
  created:
    - "pyproject.toml"
    - "src/pcn_torch/__init__.py"
    - "src/pcn_torch/_types.py"
    - "src/pcn_torch/activations.py"
    - "src/pcn_torch/layers.py"
    - ".github/workflows/ci.yml"
    - ".gitignore"
    - "tests/test_activations.py"
    - "tests/test_layers.py"
    - "tests/__init__.py"
  modified: []

key-decisions:
  - "Used TYPE_CHECKING guard for ActivationPair import in activations.py to satisfy ruff TCH001"
  - "Weight shape (out_features, in_features) matches nn.Linear convention per locked decision"
  - "Xavier uniform init with gain=1.0 (paper default, not He init for ReLU)"
  - "ReLU derivative at zero returns 0.0 (standard convention: x > 0, not x >= 0)"

patterns-established:
  - "Activation registry: dict[str, ActivationPair] with get_activation() lookup"
  - "Named internal functions (_relu, _relu_deriv) instead of lambdas"
  - "forward() returns tuple[Tensor, Tensor] as (prediction, preactivation)"
  - "Test structure: class-based grouping with parametrize for variants"

# Metrics
duration: 12min
completed: 2026-02-20
---

# Phase 1 Plan 1: Foundation Scaffold Summary

**Installable pcn-torch package with PCNLayer nn.Module, 4 activation functions with hand-coded derivatives, and full CI pipeline across Python 3.10-3.13**

## Performance

- **Duration:** 12 min
- **Started:** 2026-02-20T19:07:21Z
- **Completed:** 2026-02-20T19:19:07Z
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments

- Fully installable package via `uv sync --dev` with hatchling + hatch-vcs build system
- PCNLayer as nn.Module subclass with nn.Linear-convention constructor, Xavier uniform init, and correct forward pass shapes
- All 4 activation functions (relu, tanh, sigmoid, identity) with mathematically verified hand-coded derivatives
- 32 passing tests covering activation correctness, layer construction, forward pass shapes, and parameter registration
- All quality gates pass: ruff check, ruff format, mypy (strict mode), pytest

## Task Commits

Each task was committed atomically:

1. **Task 1: Create package scaffold and configuration** - `669172a` (chore)
2. **Task 2: Implement activation functions and PCNLayer** - `2f4901d` (feat)
3. **Task 3: Write tests and verify full pipeline** - `af628e5` (test)

## Files Created/Modified

- `pyproject.toml` - Build config (hatchling + hatch-vcs), tool configs (ruff, mypy, pytest, coverage)
- `src/pcn_torch/__init__.py` - Public API: PCNLayer, get_activation, __version__, type aliases
- `src/pcn_torch/_types.py` - Type aliases: ActivationFn, ActivationDeriv, ActivationPair
- `src/pcn_torch/activations.py` - 4 activations with hand-coded derivatives, string-based registry
- `src/pcn_torch/layers.py` - PCNLayer(nn.Module) with forward returning (prediction, preactivation)
- `.github/workflows/ci.yml` - CI with lint, typecheck, test jobs across Python 3.10-3.13
- `.gitignore` - Python/tooling ignores including auto-generated _version.py
- `tests/__init__.py` - Empty test package marker
- `tests/test_activations.py` - 15 tests for activation functions and derivatives
- `tests/test_layers.py` - 17 tests for PCNLayer construction, forward, parameters
- `uv.lock` - Reproducible dependency lockfile

## Decisions Made

1. **TYPE_CHECKING guard for type-only imports**: Ruff TCH001 requires imports used only in annotations (when `from __future__ import annotations` is active) to be inside `if TYPE_CHECKING:` blocks. Applied to `ActivationPair` import in `activations.py`.
2. **Full implementations in Task 1**: `activations.py` and `layers.py` were created during Task 1 because `__init__.py` imports from both modules. They were committed separately in Task 2 to maintain plan structure.
3. **dependency-groups for dev deps**: Used `[dependency-groups]` (PEP 735) instead of `[project.optional-dependencies]` for dev dependencies, which is the modern standard with uv.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Ruff TCH001 lint error on type-only import**
- **Found during:** Task 2 (verification step)
- **Issue:** `from pcn_torch._types import ActivationPair` in `activations.py` was flagged by ruff TCH001 because `from __future__ import annotations` makes annotations strings at runtime, so the import is type-checking-only.
- **Fix:** Wrapped import in `if TYPE_CHECKING:` block.
- **Files modified:** `src/pcn_torch/activations.py`
- **Verification:** `ruff check src/pcn_torch/` passes
- **Committed in:** `2f4901d` (Task 2 commit)

**2. [Rule 3 - Blocking] Ruff format differences in activations.py and layers.py**
- **Found during:** Task 2 (verification step)
- **Issue:** Hand-written code had minor formatting differences from ruff's preferred style (string concatenation in error message, extra_repr line length).
- **Fix:** Ran `ruff format src/pcn_torch/` to auto-format.
- **Files modified:** `src/pcn_torch/activations.py`, `src/pcn_torch/layers.py`
- **Verification:** `ruff format --check src/pcn_torch/` passes
- **Committed in:** `2f4901d` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both auto-fixes were necessary for lint/format compliance. No scope creep.

## Issues Encountered

None - all tasks completed without unexpected problems.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- PCNLayer is ready for Phase 2 to build PredictiveCodingNetwork on top
- `get_activation()` and `activation_deriv` are ready for Phase 3's local error computation
- Build system is ready for Phase 4's PyPI publishing
- No blockers identified

## Self-Check: PASSED

---
*Phase: 01-foundation*
*Completed: 2026-02-20*
