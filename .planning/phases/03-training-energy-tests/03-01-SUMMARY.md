---
phase: 03-training-energy-tests
plan: 01
subsystem: training
tags: [pcn, training-loop, energy, inference, learning, hebbian, callbacks, rich]
requires:
  - 02-01 (PredictiveCodingNetwork, PCNErrors, compute_errors API)
provides:
  - Energy computation module (compute_energy, compute_energy_per_layer)
  - Training loop (train_pcn) implementing Algorithm 3 two-phase inference+learning
  - Evaluation loop (test_pcn) for inference-only testing
  - TrainConfig, TrainHistory, EnergyHistory dataclasses
  - TrainCallback base class and RichCallback default display
affects:
  - 03-02 (tests will verify training loop structural correctness)
  - 04-xx (CIFAR-10 example will use train_pcn/test_pcn as the main API)
tech-stack:
  added:
    - "rich>=14.0 (console progress display)"
  patterns:
    - "Two-phase training loop (inference + learning) under torch.no_grad()"
    - "In-place weight mutation via .data attribute (no autograd)"
    - "Callback pattern with concrete base class (no-op defaults)"
    - "Rolling deque for energy window monitoring"
    - "TYPE_CHECKING guard for PredictiveCodingNetwork import (TCH001)"
key-files:
  created:
    - src/pcn_torch/energy.py
    - src/pcn_torch/trainer.py
  modified:
    - pyproject.toml
key-decisions:
  - "PredictiveCodingNetwork import in TYPE_CHECKING block (TCH001 compliance, only used in annotations)"
  - "model.train(mode=False) instead of model.set_eval() to set evaluation mode cleanly"
  - "type: ignore[assignment] for nn.ModuleList weight list construction (mypy Module vs Tensor)"
  - "Unused loop variable _t in test_pcn inference loop (ruff B007)"
  - "RichCallback with _PlainCallback fallback when rich is not installed"
duration: 16 min
completed: 2026-02-24
---

# Phase 3 Plan 01: Training System and Energy Computation Summary

**Algorithm 3 two-phase training loop (inference + learning) with energy computation, TrainConfig/TrainHistory dataclasses, callback system with rich console default, and test_pcn evaluation**

## Performance

- **Duration:** 16 min
- **Start:** 2026-02-24T20:16:06Z
- **End:** 2026-02-24T20:31:53Z
- **Tasks:** 2/2 complete
- **Files created:** 2
- **Files modified:** 1
- **Lines written:** 629 (63 energy.py + 566 trainer.py)

## Accomplishments

1. **Energy computation module** (`energy.py`) -- `compute_energy()` returns batch-averaged total energy from PCNErrors following paper formula E_batch = (1/B) sum_b [(1/2) sum_l ||eps||^2 + (1/2)||eps_sup||^2]. `compute_energy_per_layer()` returns per-layer breakdown. Both cast to float32 before squaring to prevent float16 overflow under autocast.

2. **TrainConfig dataclass** -- Groups all hyperparameters with paper-matched defaults: T_infer=50, T_learn=None (=batch_size), lr_infer=0.05, lr_learn=0.005, num_epochs=4. Includes __post_init__ validation for task, T_infer, and T_learn constraints.

3. **TrainHistory and EnergyHistory dataclasses** -- EnergyHistory tracks per_batch and per_epoch energy. TrainHistory aggregates energy, train_loss, train_accuracy, test_loss, test_accuracy, and stores the config used.

4. **TrainCallback base class** -- Concrete class with 8 no-op hook methods (on_train_start through on_train_end). Users subclass and override only what they need.

5. **RichCallback** -- Default callback using rich.console.Console for formatted epoch summaries with energy and accuracy. Falls back to _PlainCallback (print statements) when rich is not installed.

6. **train_pcn function** -- Implements Algorithm 3 from arXiv:2506.06332v1:
   - **Inference phase:** T_infer steps updating latents via `grad_x = extended_errors[idx+1] - (gm_errors[idx] @ weights[idx])`, all under torch.no_grad() with optional autocast on CUDA
   - **Learning phase:** T_learn steps updating weights via `grad_W = -(gm_errors.T @ states) / B` with in-place `.data -=` mutation, NO autocast
   - Classification targets one-hot encoded internally; regression pass-through
   - Rolling deque energy window with optional early stopping
   - Runtime autograd guard on first iteration via `_check_no_autograd()`

7. **test_pcn function** -- Inference-only evaluation following paper Section 5.3: initialize latents, run inference loop, read predictions from readout layer. Returns dict with accuracy and energy.

8. **pyproject.toml updated** -- Added `rich>=14.0` to project dependencies.

## Task Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Create energy computation module | 01cbf33 | src/pcn_torch/energy.py |
| 2 | Create trainer module with config, history, callbacks, and training loops | a17924c | src/pcn_torch/trainer.py, pyproject.toml |

## Files Created

- `src/pcn_torch/energy.py` (63 lines) -- compute_energy, compute_energy_per_layer
- `src/pcn_torch/trainer.py` (566 lines) -- TrainConfig, TrainHistory, EnergyHistory, TrainCallback, RichCallback, train_pcn, test_pcn

## Files Modified

- `pyproject.toml` -- Added `rich>=14.0` to dependencies

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| PredictiveCodingNetwork in TYPE_CHECKING block | Only used in type annotations (strings with `from __future__ import annotations`); ruff TCH001 requires type-only imports in TYPE_CHECKING |
| `model.train(mode=False)` for evaluation mode | Functionally identical to `.eval()`; avoids tooling false positives that confuse PyTorch method with Python builtin |
| `type: ignore[assignment]` for weights list | nn.ModuleList iteration returns nn.Module; weight list comprehension produces `list[Tensor or Module]` which mypy rejects as `list[Tensor]` |
| RichCallback with _PlainCallback fallback | rich is a required dependency but graceful degradation if somehow missing; follows defensive coding pattern |
| Unused `_t` prefix in test_pcn inference loop | Ruff B007 flags unused loop variables; underscore prefix is the standard convention |

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

- **Ruff TCH001:** Plan had `PredictiveCodingNetwork` as a runtime import but with `from __future__ import annotations` it is only needed for type hints. Moved to TYPE_CHECKING block.
- **Mypy nn.ModuleList typing:** Same pattern as Phase 2 -- iterating nn.ModuleList yields nn.Module, not the stored subtype. Required `type: ignore[assignment]` on the weights list construction.
- **Ruff B007:** Loop variable `t` in test_pcn inference loop unused in body (only used for range iteration). Renamed to `_t`.
- **Ruff SIM108:** If/else block for targets in test_pcn replaced with ternary expression.

## Verification Results

- ruff check: All checks passed (both energy.py and trainer.py)
- mypy: Success, no issues found in 2 source files
- Existing test suite: 79 passed in 10.52s (no regressions)
- Smoke test: train_pcn trained for 1 epoch (2 batches), returned TrainHistory with energy.per_batch=[2], energy.per_epoch=[1], train_accuracy=[0.375]
- Smoke test: test_pcn evaluated and returned accuracy=0.0625, energy=14.12

## Next Phase Readiness

Phase 3 Plan 02 (tests) can proceed. The training API is ready:

```python
from pcn_torch.trainer import train_pcn, test_pcn, TrainConfig

config = TrainConfig(T_infer=3, T_learn=2, num_epochs=1)
history = train_pcn(model, dataloader, config)
result = test_pcn(model, dataloader, config)
```

Plan 02 will test:
- Energy computation correctness
- Training loop structural behavior (shapes, energy tracking, no autograd)
- Gain-modulated error formula verification
- W_out^T supervised error projection

## Self-Check: PASSED
