# Phase 1: Foundation - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Package scaffold, core types, activation functions with hand-coded derivatives, and PCNLayer as a configurable nn.Module. Developers can `pip install -e .` and `from pcn_torch import PCNLayer` with a working forward pass. CI passes across Python 3.10-3.13.

</domain>

<decisions>
## Implementation Decisions

### PCNLayer API style
- Constructor follows nn.Linear convention: `PCNLayer(in_features, out_features, activation='relu')`
- Positional dimensions, keyword activation — familiar to any PyTorch user
- No bias term (per paper specification)

### Claude's Discretion
- **Activation interface**: String-based lookup via `get_activation(name)` returning `(fn, fn_deriv)` pair. Built-ins: relu, tanh, sigmoid, identity. Implementation details (registry pattern, module-level dict, etc.) are Claude's choice.
- **forward() return type**: Returns `(prediction, preactivation)` tuple since preactivation is required for derivative computation in Phase 2's error calculation. Exact naming and documentation are Claude's choice.
- **Type aliases**: `ActivationFn`, `ActivationDeriv`, `ActivationPair` in `_types.py`. Exact type definitions and any additional type aliases are Claude's choice.
- **Package scaffold**: pyproject.toml with hatchling, src layout vs flat layout, CI configuration, ruff/mypy settings — all Claude's choice following PyTorch ecosystem conventions.
- **Weight initialization**: Xavier uniform (per architecture research). Exact implementation is Claude's choice.
- **Import structure**: Flat imports via `__init__.py` re-exports. What's exported at this phase is Claude's choice (at minimum: PCNLayer).

</decisions>

<specifics>
## Specific Ideas

- "Like nn.Linear" — the user wants PCNLayer to feel immediately familiar to PyTorch users. Minimal surface area, no surprises.
- Core value: "minimal friction" — the API should be obvious to anyone who's used PyTorch before.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-02-20*
