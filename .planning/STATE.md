# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-20)

**Core value:** A clean, PyTorch-native PCN implementation that lets anyone empirically explore predictive coding on their own problems with minimal friction.
**Current focus:** Phase 3: Training + Energy + Tests

## Current Position

Phase: 3 of 4 (Training + Energy + Tests)
Plan: 1 of 2 in current phase
Status: In progress
Last activity: 2026-02-24 -- Completed 03-01-PLAN.md (Training system + energy computation)

Progress: [######....] 60% (3/5 plans)
Quick tasks: 2 completed (quick-001, quick-002)

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 13.7 min
- Total execution time: 0.7 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 1/1 | 12 min | 12 min |
| 2. Core Model | 1/1 | 13 min | 13 min |
| 3. Training + Energy + Tests | 1/2 | 16 min | 16 min |

**Recent Trend:**
- Last 5 plans: 01-01 (12 min), 02-01 (13 min), 03-01 (16 min)
- Trend: Consistent velocity, slight increase with algorithmic complexity

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

Last session: 2026-02-24T20:31Z
Stopped at: Completed 03-01-PLAN.md (Training system + energy computation)
Resume file: None
