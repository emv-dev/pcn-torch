---
phase: 02-core-model
plan: 01
subsystem: network
tags: [pcn, nn.Module, error-computation, latents, readout, gain-modulation]
requires:
  - 01-01 (PCNLayer, activations, types)
provides:
  - PredictiveCodingNetwork class (stacked PCNLayers, latent init, error computation, readout)
  - PCNErrors NamedTuple (errors, gm_errors, supervised_error, top_error)
  - Public API exports (pcn_torch.PredictiveCodingNetwork, pcn_torch.PCNErrors)
affects:
  - 03-01 (training loops will drive PredictiveCodingNetwork.compute_errors and latent updates)
  - 03-02 (energy tracking will use PCNErrors for energy computation)
tech-stack:
  added: []
  patterns:
    - "NamedTuple for structured error returns"
    - "nn.ModuleList for layer registry"
    - "TYPE_CHECKING guard for Sequence import (TCH003)"
    - "type: ignore[assignment] for nn.ModuleList iteration (mypy nn.Module return)"
key-files:
  created:
    - src/pcn_torch/network.py
    - tests/test_network.py
  modified:
    - src/pcn_torch/__init__.py
key-decisions:
  - "Variable name 'idx' instead of 'l' to avoid ruff E741 ambiguous variable name"
  - "Explicit Tensor type annotation on predict() return to satisfy mypy no-any-return"
  - "PCNLayer type assertion via indexed access + type: ignore for nn.ModuleList iteration"
duration: 13 min
completed: 2026-02-23
---

# Phase 2 Plan 01: PredictiveCodingNetwork and PCNErrors Summary

**PredictiveCodingNetwork nn.Module with N(0,1) latent init, gain-modulated error computation, W_out^T supervised error projection, and linear readout**

## Performance

- **Duration:** 13 min
- **Start:** 2026-02-23T23:49:09Z
- **End:** 2026-02-24T00:01:54Z
- **Tasks:** 2/2 complete
- **Files created:** 2
- **Files modified:** 1
- **Tests added:** 47 (79 total)
- **Lines written:** 588 (215 network.py + 373 test_network.py)

## Accomplishments

1. **PredictiveCodingNetwork class** -- nn.Module managing a stack of PCNLayers with configurable construction from either a dimension list or pre-built PCNLayer instances (mutually exclusive, with dimension validation for pre-built layers)
2. **PCNErrors NamedTuple** -- structured container with errors, gm_errors, supervised_error, and top_error fields
3. **Latent initialization** -- `init_latents(batch_size)` creates N(0,1) latents via `torch.randn` stored as plain tensors (NOT nn.Parameter) in `network.latents`
4. **Error computation** -- `compute_errors(x, y)` implements all arXiv:2506.06332v1 equations: prediction errors eps^(l) = x^(l) - f(W^(l) @ x^(l+1)), gain-modulated errors h^(l) = f'(a^(l)) * eps^(l), supervised error eps_sup = y_hat - y, and W_out^T projection for top error
5. **Linear readout** -- `predict()` returns `_readout(latents[-1])` via internal nn.Linear(bias=False)
6. **Public API** -- PCNErrors and PredictiveCodingNetwork exported from `pcn_torch.__init__`
7. **Comprehensive tests** -- 47 tests covering construction, latent init, predict, compute_errors (shapes + formula verification), and parameter registration

## Task Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Implement PredictiveCodingNetwork and PCNErrors | cfdf61a | src/pcn_torch/network.py |
| 2 | Update public API exports and write tests | 9b99b2a | src/pcn_torch/__init__.py, tests/test_network.py |

## Files Created

- `src/pcn_torch/network.py` (215 lines) -- PredictiveCodingNetwork class, PCNErrors NamedTuple
- `tests/test_network.py` (373 lines) -- 47 tests across 6 test classes

## Files Modified

- `src/pcn_torch/__init__.py` -- Added PCNErrors and PredictiveCodingNetwork to imports and __all__

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Variable name `idx` instead of `l` | Ruff E741 flags single-letter `l` as ambiguous (looks like `1`); `idx` is unambiguous |
| Explicit `result: Tensor` annotation on predict() return | mypy's `no-any-return` rule triggers because nn.Linear.__call__ returns Any; explicit annotation resolves it |
| `type: ignore[assignment]` for nn.ModuleList iteration | nn.ModuleList.__getitem__ returns nn.Module, not PCNLayer; type ignore is the standard pattern |
| PCNErrors as NamedTuple (not dataclass) | Per RESEARCH.md recommendation; immutable, supports unpacking, lightweight |

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

- **Ruff E741:** Plan used variable name `l` for layer index (matching paper notation); ruff flags this as ambiguous. Renamed to `idx` throughout.
- **Mypy no-any-return:** `nn.Linear.__call__` returns `Any` in torch stubs; added explicit return type annotation.
- **Mypy nn.ModuleList typing:** Iterating `nn.ModuleList` yields `nn.Module`, not the stored subtype. Used indexed access with `type: ignore[assignment]`.

## Next Phase Readiness

Phase 3 (Training + Energy + Tests) can proceed. The following API is ready:

```python
net = PredictiveCodingNetwork(dims=[3072, 500, 500, 10])
net.init_latents(batch_size=32)
errors = net.compute_errors(x, y)  # PCNErrors with all signals
y_hat = net.predict()              # readout output
```

Phase 3 will need to:
- Implement inference loop (iterative latent updates using `errors.gm_errors`)
- Implement learning loop (weight updates using `errors.gm_errors`)
- Access `net.latents` for latent update steps
- Use `errors.top_error` for top-layer latent update with supervised signal

## Self-Check: PASSED
