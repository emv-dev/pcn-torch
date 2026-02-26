# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-20)

**Core value:** A clean, PyTorch-native PCN implementation that lets anyone empirically explore predictive coding on their own problems with minimal friction.
**Current focus:** Phase 4: Publishing + CIFAR-10 Example

## Current Position

Phase: 4 of 4 (Integration + Publishing)
Plan: 3 of 4 in current phase (04-01 done, 04-02 done, 04-03 done, 04-04 pending)
Status: In progress
Last activity: 2026-02-25 -- Completed 04-02-PLAN.md (CIFAR-10 example script)

Progress: [#######.] 87% (7/8 plans)
Quick tasks: 5 completed (quick-001, quick-002, quick-003, quick-004, quick-005)

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 10.0 min
- Total execution time: 1.18 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 1/1 | 12 min | 12 min |
| 2. Core Model | 1/1 | 13 min | 13 min |
| 3. Training + Energy + Tests | 2/2 | 26 min | 13 min |
| 4. Integration + Publishing | 3/4 | 18 min | 6 min |

**Recent Trend:**
- Last 5 plans: 03-01 (16 min), 03-02 (10 min), 04-01 (6 min), 04-03 (4 min), 04-02 (8 min)
- Trend: Docs/config plans faster than implementation plans

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
- [04-01]: torch as peer dependency (removed from install_requires, users install from pytorch.org)
- [04-01]: torchvision in dev group only (needed for examples/cifar10.py, not library consumers)
- [04-03]: Status badge updated from pre-release to stable
- [04-03]: Planning docs links removed from user-facing README entirely
- [04-02]: Fixed config constants (no argparse) for example scripts
- [04-02]: noqa: T201 on print statements in examples (defensive, T201 not in ruff select)
- [quick-004]: LR schedule only affects lr_learn (weight updates), NOT lr_infer (latent inference)
- [quick-004]: Default schedule is reduce_on_plateau (enabled by default, None disables)
- [quick-004]: Strictly less-than comparison for energy improvement (equal counts as plateau)
- [quick-005]: No ANSI colors in ci-local.sh (works everywhere including Git Bash on Windows)
- [quick-005]: --fix flag runs ruff fix + format then exits (skips typecheck/test)

### Pending Todos

None yet.

### Blockers/Concerns

- Check PyPI name availability for `pcn-torch` before Phase 4 publishing

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Add README linking all planning documents | 2026-02-20 | 0e9a31c | [001-add-readme-linking-planning-docs](./quick/001-add-readme-linking-planning-docs/) |
| 002 | Add post-phase GitHub merge workflow | 2026-02-20 | 0535bb2 | [002-add-post-phase-github-merge-workflow](./quick/002-add-post-phase-github-merge-workflow/) |
| 003 | Make README plain English for accessibility | 2026-02-26 | ced4729 | [003-make-readme-plain-english-accessible](./quick/003-make-readme-plain-english-accessible/) |
| 004 | Add ReduceOnPlateau LR schedule for lr_learn | 2026-02-26 | 487ee02 | [004-add-reduce-on-plateau-lr-schedule](./quick/004-add-reduce-on-plateau-lr-schedule/) |
| 005 | Add local CI check script | 2026-02-26 | d5def0a | [005-add-local-ci-check-script](./quick/005-add-local-ci-check-script/) |

## Session Continuity

Last session: 2026-02-26T20:55Z
Stopped at: Completed quick-005 (Add local CI check script)
Resume file: None
