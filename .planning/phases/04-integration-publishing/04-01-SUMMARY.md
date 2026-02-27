---
phase: 04-integration-publishing
plan: 01
subsystem: infra
tags: [pypi, packaging, toml, torchvision, peer-dependency]

# Dependency graph
requires:
  - phase: 03-training-energy-tests
    provides: "Complete pcn-torch package with all source modules and tests"
provides:
  - "PyPI-ready pyproject.toml with correct metadata and peer dependency pattern"
  - "torchvision dev dependency for CIFAR-10 example"
  - "Updated uv.lock with resolved dependency graph"
affects: [04-02 sdist/wheel build, 04-03 CIFAR-10 example, 04-04 README + publishing]

# Tech tracking
tech-stack:
  added: [torchvision>=0.15, pillow>=12.1]
  patterns: [peer-dependency-pattern for torch]

key-files:
  created: []
  modified: [pyproject.toml, uv.lock]

key-decisions:
  - "torch as peer dependency: removed from install_requires so pip install pcn-torch does not force CPU PyTorch on all users"
  - "torchvision in dev group only: needed for examples/cifar10.py, not for library consumers"

patterns-established:
  - "Peer dependency pattern: comment in pyproject.toml pointing users to pytorch.org for install"

# Metrics
duration: 6min
completed: 2026-02-25
---

# Phase 4 Plan 1: PyPI Metadata and Peer Dependency Summary

**Removed torch from prod deps (peer dependency pattern), added PyPI metadata (author, classifiers, keywords, URLs), and torchvision dev dep**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-25T19:07:28Z
- **Completed:** 2026-02-25T19:13:10Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- Removed `torch>=2.0` from production dependencies so `pip install pcn-torch` no longer forces CPU PyTorch
- Added complete PyPI metadata: author (Enrique Mendez), 13 classifiers, 6 keywords, 4 project URLs
- Added `torchvision>=0.15` to dev dependency group for CIFAR-10 example support
- All 101 existing tests still pass after dependency changes

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove torch from dependencies, add metadata and torchvision dev dep** - `c4f6522` (feat)

## Files Created/Modified
- `pyproject.toml` - Removed torch from prod deps, added authors/classifiers/keywords/URLs, added torchvision to dev group
- `uv.lock` - Updated lockfile with torchvision 0.25.0 and pillow 12.1.1

## Decisions Made
- **torch as peer dependency:** Users install PyTorch separately from pytorch.org to get the correct CUDA/CPU variant for their system. This is the standard pattern for PyTorch ecosystem libraries.
- **torchvision in dev group only:** Only needed for `examples/cifar10.py`, not required by the library itself. Library consumers who need torchvision will already have it.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- pyproject.toml is ready for `uv build` (plan 04-02)
- All metadata present for PyPI upload
- Dev environment has torchvision for CIFAR-10 example (plan 04-03)

## Self-Check: PASSED

---
*Phase: 04-integration-publishing*
*Completed: 2026-02-25*
