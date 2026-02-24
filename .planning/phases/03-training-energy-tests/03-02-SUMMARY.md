---
phase: 03-training-energy-tests
plan: 02
subsystem: testing
tags: [pcn, tests, energy, training, algorithmic-correctness, TST-02, TST-03, TST-04]
requires:
  - 03-01 (energy.py, trainer.py modules under test)
  - 02-01 (PredictiveCodingNetwork, PCNErrors, compute_errors API)
provides:
  - Energy computation test suite (8 tests covering correctness, edge cases, float16 safety)
  - Training loop test suite (14 tests covering config validation, algorithmic correctness, integration)
  - Public API surface for Phase 3 (all trainer/energy names importable from pcn_torch)
affects:
  - 04-xx (CIFAR-10 example can import train_pcn/test_pcn from top-level pcn_torch)
tech-stack:
  added: []
  patterns:
    - "Aliased import (test_pcn as evaluate_pcn) to avoid pytest collection of non-test functions"
    - "TYPE_CHECKING guard for PCNLayer in test file (ruff TCH001)"
    - "Silent TrainCallback() no-op callback for fast structural tests"
key-files:
  created:
    - tests/test_energy.py
    - tests/test_trainer.py
  modified:
    - src/pcn_torch/__init__.py
key-decisions:
  - "Import test_pcn as evaluate_pcn to prevent pytest from collecting trainer.test_pcn as a test function"
  - "PCNLayer in TYPE_CHECKING block in test_trainer.py (TCH001 compliance)"
  - "All tests use TrainConfig(T_infer=3, T_learn=2, num_epochs=1, use_mixed_precision=False) for speed"
  - "Silent TrainCallback() (no-op) as default callback in tests to suppress console output"
duration: 10 min
completed: 2026-02-24
---

# Phase 3 Plan 02: Energy and Training Correctness Tests Summary

**8 energy computation tests + 14 trainer tests (TST-02 gain-modulated error, TST-03 no autograd, TST-04 W_out^T projection) with public API exports; all 101 tests pass across all phases**

## Performance

- **Duration:** 10 min
- **Start:** 2026-02-24T20:39:18Z
- **End:** 2026-02-24T20:49:45Z
- **Tasks:** 2/2 complete
- **Files created:** 2
- **Files modified:** 1
- **Lines written:** 472 (127 test_energy.py + 304 test_trainer.py + 41 __init__.py)

## Accomplishments

1. **Energy computation tests** (`test_energy.py`, 8 tests):
   - `test_compute_energy_returns_float` -- verifies Python float return, not Tensor
   - `test_compute_energy_non_negative` -- energy >= 0 (sum of squared norms)
   - `test_compute_energy_without_supervised` -- works with y=None
   - `test_compute_energy_with_supervised` -- supervised error adds positive term
   - `test_compute_energy_manual_formula` -- matches manual (1/B) * sum(0.5 * ||e||^2)
   - `test_compute_energy_per_layer_count` -- len(per_layer) correct with/without supervised
   - `test_compute_energy_per_layer_sum_matches_total` -- sum of parts equals whole
   - `test_compute_energy_float16_safe` -- finite results with float16 inputs

2. **TrainConfig validation tests** (4 tests):
   - `test_config_defaults` -- paper defaults: T_infer=50, lr_infer=0.05, lr_learn=0.005, num_epochs=4
   - `test_config_invalid_task` -- ValueError for non-classification/regression
   - `test_config_invalid_T_infer` -- ValueError for T_infer=0
   - `test_config_invalid_T_learn` -- ValueError for T_learn=0

3. **Algorithmic correctness tests** (3 tests, TST-02/03/04):
   - `test_gain_modulated_error_uses_preactivation` (TST-02) -- h^(l) = f'(a^(l)) * eps uses preactivation, not latent
   - `test_no_autograd_graph_during_training` (TST-03) -- no RuntimeWarning, all latents have grad_fn=None
   - `test_top_error_wout_projection` (TST-04) -- top_error = supervised_error @ W_out, shape (B, d_L)

4. **Training integration tests** (7 tests):
   - `test_train_pcn_classification` -- completes 1 epoch, history populated
   - `test_train_pcn_regression` -- task="regression", no accuracy tracked
   - `test_train_pcn_energy_per_batch_length` -- per_batch count = num_batches * num_epochs
   - `test_train_pcn_history_config_stored` -- history.config is same object
   - `test_test_pcn_returns_metrics` -- returns dict with accuracy/energy floats
   - `test_train_pcn_weights_change` -- weights actually updated
   - `test_train_pcn_mixed_precision_false` -- runs on CPU without error

5. **Public API exports** (`__init__.py`):
   - Added: compute_energy, compute_energy_per_layer, EnergyHistory, RichCallback, TrainCallback, TrainConfig, TrainHistory, train_pcn, test_pcn
   - Alphabetical order maintained in __all__

## Task Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Write energy and training correctness tests | e3fc0d4 | tests/test_energy.py, tests/test_trainer.py |
| 2 | Update public API exports in __init__.py | 3a1f750 | src/pcn_torch/__init__.py |

## Files Created

- `tests/test_energy.py` (127 lines) -- 8 energy computation correctness tests
- `tests/test_trainer.py` (304 lines) -- 14 training loop tests (config, algorithmic, integration)

## Files Modified

- `src/pcn_torch/__init__.py` -- Added 9 new public API exports from energy and trainer modules

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Import `test_pcn as evaluate_pcn` in test file | Pytest auto-collects any function named `test_*` from imported modules; aliasing prevents fixture-not-found errors |
| PCNLayer in TYPE_CHECKING block (test_trainer.py) | Only used for type annotation on `layers[0]` cast; ruff TCH001 requires type-only imports in TYPE_CHECKING |
| TrainConfig(T_infer=3, T_learn=2, num_epochs=1) for tests | Structural correctness tests only; minimal iterations for speed while exercising all code paths |
| Silent TrainCallback() as test default | Suppresses rich console output during test runs; cleaner pytest output |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pytest collecting test_pcn from trainer.py**

- **Found during:** Task 1 verification
- **Issue:** `from pcn_torch.trainer import test_pcn` caused pytest to collect the function as a test case (it starts with `test_`), failing with "fixture 'model' not found"
- **Fix:** Imported as `test_pcn as evaluate_pcn` with alias to prevent pytest collection
- **Files modified:** tests/test_trainer.py
- **Commit:** e3fc0d4

**2. [Rule 3 - Blocking] Ruff TCH001 and I001 violations in test_trainer.py**

- **Found during:** Task 1 ruff verification
- **Issue:** PCNLayer was a runtime import but only used in type annotations; import blocks needed sorting after alias change
- **Fix:** Moved PCNLayer to TYPE_CHECKING block; ran ruff --fix for import sorting
- **Files modified:** tests/test_trainer.py
- **Commit:** e3fc0d4

## Verification Results

- pytest tests/test_energy.py: 8 passed
- pytest tests/test_trainer.py: 14 passed
- pytest tests/ (full suite): 101 passed in 2.86s (no regressions)
- ruff check (all modified files): All checks passed
- mypy src/pcn_torch/__init__.py: Success, no issues
- Public API smoke test: All 9 new names importable from pcn_torch

## Next Phase Readiness

Phase 3 is complete. All training system code (energy.py, trainer.py from Plan 01) is now tested and exported. The full test suite covers:

- Phase 1: 15 activation tests, 10 layer tests
- Phase 2: 34 network tests (construction, latents, prediction, errors, parameters)
- Phase 3: 8 energy tests + 14 trainer tests

Phase 4 (Publishing + CIFAR-10 example) can proceed. The public API is stable:

```python
from pcn_torch import (
    PredictiveCodingNetwork,
    TrainConfig,
    train_pcn,
    test_pcn,
)
```

## Self-Check: PASSED
