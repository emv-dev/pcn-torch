---
phase: 04-integration-publishing
plan: 02
subsystem: examples
tags: [cifar10, torchvision, richcallback, classification, training-example]

# Dependency graph
requires:
  - phase: 02-core-model
    provides: PredictiveCodingNetwork with dims constructor and output_dim
  - phase: 03-training-energy-tests
    provides: train_pcn, test_pcn, RichCallback, TrainConfig, TrainHistory
provides:
  - Standalone CIFAR-10 classification example at examples/cifar10.py
  - Demonstration of full PCN training workflow for end users
affects: [04-04-publishing, README]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Example scripts use fixed config constants (no CLI args)"
    - "No Flatten transform in examples -- trainer.py handles view(B, -1)"
    - "num_workers=0 for cross-platform DataLoader compatibility"

key-files:
  created:
    - examples/cifar10.py
  modified: []

key-decisions:
  - "Fixed config constants at module top (no argparse) per locked decision"
  - "RichCallback() used for live progress per locked decision"
  - "noqa: T201 on print statements since T201 not in ruff select rules but added defensively"

patterns-established:
  - "Example script structure: docstring, imports, config constants, loader function, model builder, main()"
  - "Dataset error handling: try/except wrapping download with SystemExit + friendly message"
  - "Training summary output: accuracy, time, energy convergence printed after training"

# Metrics
duration: 8min
completed: 2026-02-25
---

# Phase 4 Plan 2: CIFAR-10 Example Summary

**Self-contained CIFAR-10 example script with RichCallback progress, fixed config, and full training summary output**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-25T19:08:32Z
- **Completed:** 2026-02-25T19:16:11Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- Created examples/cifar10.py as the primary library demonstration script
- Fixed config: BATCH_SIZE=128, NUM_EPOCHS=10, T_INFER=20, LR_INFER=0.05, LR_LEARN=0.001
- RichCallback for live progress display during training
- Full summary printed after training: test accuracy, training time, energy convergence
- Friendly error message for dataset download failures (SystemExit, not raw traceback)
- Uses only public pcn_torch API (no private names)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create examples/cifar10.py with CIFAR-10 training script** - `4056e82` (feat)

## Files Created/Modified
- `examples/cifar10.py` - Standalone CIFAR-10 classification example using pcn_torch public API

## Decisions Made
- Used `noqa: T201` on print statements defensively (T201 not currently in ruff select rules, but protects against future config changes)
- Shortened long comments and used multi-line formatting for DataLoader calls to stay within 88-char line length
- Used variable extraction (`accuracy`, `minutes`, `e_first`, `e_last`) to keep f-string print lines within line length limit

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed 5 line-length violations for ruff E501 compliance**
- **Found during:** Task 1 (verification step)
- **Issue:** Plan-provided code had 5 lines exceeding 88-char limit (comments, DataLoader calls, docstring)
- **Fix:** Shortened comments, wrapped DataLoader calls to multiple lines, reformatted docstring Architecture line
- **Files modified:** examples/cifar10.py
- **Verification:** `uv run ruff check examples/cifar10.py` passes with "All checks passed!"
- **Committed in:** 4056e82

---

**Total deviations:** 1 auto-fixed (1 bug -- line length violations)
**Impact on plan:** Necessary for ruff compliance. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CIFAR-10 example ready for use in README documentation and publishing
- Example is excluded from wheel (hatch targets.wheel uses packages = ["src/pcn_torch"])
- Requires `pip install torchvision` to run (documented in script docstring)

## Self-Check: PASSED

---
*Phase: 04-integration-publishing*
*Completed: 2026-02-25*
