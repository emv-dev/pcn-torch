# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-20)

**Core value:** A clean, PyTorch-native PCN implementation that lets anyone empirically explore predictive coding on their own problems with minimal friction.
**Current focus:** Phase 4: Publishing + CIFAR-10 Example

## Current Position

Phase: 3 of 4 (Training + Energy + Tests)
Plan: 2 of 2 in current phase
Status: Phase complete
Last activity: 2026-02-24 -- Completed 03-02-PLAN.md (Energy and training correctness tests)

Progress: [########..] 80% (4/5 plans)
Quick tasks: 2 completed (quick-001, quick-002)

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 12.8 min
- Total execution time: 0.85 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 1/1 | 12 min | 12 min |
| 2. Core Model | 1/1 | 13 min | 13 min |
| 3. Training + Energy + Tests | 2/2 | 26 min | 13 min |

**Recent Trend:**
- Last 5 plans: 01-01 (12 min), 02-01 (13 min), 03-01 (16 min), 03-02 (10 min)
- Trend: Consistent velocity; test-writing plans faster than implementation plans

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 4-phase structure following module dependency chain (types -> layers -> network -> trainer)
- [Roadmap]: Tests grouped with training (Phase 3) since they verify algorithmic correctness of training loops
- [Roadmap]: Energy tracking in Phase 3 (not Phase 4) because it is a first-class architectural concern
- [01-01]: TYPE_CHECKING guard for type-only imports (ruff TCH001 compliance)
- [01-01]: Weight shape (out_features, in_features) matches nn.Linear convention
- [01-01]: Xavier uniform init with gain=1.0 (paper default)
- [01-01]: dependency-groups (PEP 735) for dev deps instead of optional-dependencies
- [02-01]: Variable name `idx` instead of `l` to avoid ruff E741 ambiguous variable name
- [02-01]: Explicit Tensor type annotation on predict() return to satisfy mypy no-any-return
- [02-01]: type: ignore[assignment] for nn.ModuleList iteration (mypy nn.Module return type)
- [02-01]: PCNErrors as NamedTuple (not dataclass) per RESEARCH.md recommendation
- [03-01]: PredictiveCodingNetwork in TYPE_CHECKING block (TCH001, only used in annotations)
- [03-01]: type: ignore[assignment] for weight list construction from nn.ModuleList
- [03-01]: RichCallback with _PlainCallback fallback for graceful degradation
- [03-02]: Import test_pcn as evaluate_pcn in tests to prevent pytest collection
- [03-02]: PCNLayer in TYPE_CHECKING in test file (TCH001)
- [03-02]: Silent TrainCallback() no-op for fast structural tests

### Pending Todos

None yet.

### Blockers/Concerns

- Check PyPI name availability for `pcn-torch` before Phase 4 publishing

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Add README linking all planning documents | 2026-02-20 | 0e9a31c | [001-add-readme-linking-planning-docs](./quick/001-add-readme-linking-planning-docs/) |
| 002 | Add post-phase GitHub merge workflow | 2026-02-20 | 0535bb2 | [002-add-post-phase-github-merge-workflow](./quick/002-add-post-phase-github-merge-workflow/) |

## Session Continuity

Last session: 2026-02-24T20:49Z
Stopped at: Completed 03-02-PLAN.md (Energy and training correctness tests) -- Phase 3 complete
Resume file: None
