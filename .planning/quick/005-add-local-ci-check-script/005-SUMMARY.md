---
phase: quick
plan: 005
subsystem: infra
tags: [bash, ci, ruff, mypy, pytest, uv]

# Dependency graph
requires:
  - phase: 04-integration
    provides: CI workflow (.github/workflows/ci.yml)
provides:
  - Local CI check script (scripts/ci-local.sh)
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Local CI mirror script for pre-push verification"

key-files:
  created:
    - scripts/ci-local.sh
  modified: []

key-decisions:
  - "No ANSI colors -- works everywhere including Git Bash on Windows"
  - "uv sync --dev runs once upfront rather than per-step (locally deps are shared)"
  - "--fix flag runs ruff fix + format then exits (skips typecheck/test)"

patterns-established:
  - "scripts/ directory for developer utility scripts"

# Metrics
duration: 3min
completed: 2026-02-26
---

# Quick-005: Add Local CI Check Script Summary

**Bash script at scripts/ci-local.sh mirroring GitHub Actions CI (lint, typecheck, test) for pre-push local verification**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-26T20:52:15Z
- **Completed:** 2026-02-26T20:55:30Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- Created scripts/ci-local.sh that mirrors all 3 CI jobs from .github/workflows/ci.yml
- Runs lint (ruff check + ruff format --check), typecheck (mypy), test (pytest) in exact CI order
- Exits on first failure via set -euo pipefail
- Supports --fix flag for auto-fixing lint issues and --help for usage info
- Verified: all 113 tests pass, lint clean, typecheck clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Create local CI check script** - `d5def0a` (feat)

## Files Created/Modified
- `scripts/ci-local.sh` - Local CI mirror script (lint, typecheck, test)

## Decisions Made
- No ANSI color codes: keeps output clean on all terminals including Git Bash on Windows
- Single `uv sync --dev` at the start rather than per-step (locally all steps share the same venv)
- `--fix` flag runs auto-fix then exits immediately (no point running typecheck/test on unfixed code)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- scripts/ci-local.sh ready for immediate use
- Developers can run `bash scripts/ci-local.sh` before push to catch CI failures locally

---
*Quick task: 005-add-local-ci-check-script*
*Completed: 2026-02-26*

## Self-Check: PASSED
