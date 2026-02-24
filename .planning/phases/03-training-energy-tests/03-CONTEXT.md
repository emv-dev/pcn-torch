# Phase 3: Training + Energy + Tests - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

End-to-end training loops (inference + learning) for PredictiveCodingNetwork using local Hebbian-like update rules under `torch.no_grad()`. Includes energy tracking at multiple granularities and structural correctness tests. CIFAR-10 example and publishing are Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Training API surface
- `train_pcn` returns a structured training history object (dataclass) with losses, energies, accuracies per epoch
- Accepts PyTorch DataLoader only (no raw tensor overload) — caller handles batching/shuffling
- Separate `test_pcn` function for evaluation — clear separation from training
- Hyperparameters passed via a `TrainConfig` dataclass — groups T_infer, T_learn, learning rates, etc. for reproducibility

### Inference & learning loop behavior
- Fixed T_infer steps with optional early stopping when energy converges (threshold-based)
- Two separate learning rates: `lr_infer` for latent updates, `lr_learn` for weight updates
- Explicit task mode in config: `TrainConfig(task='classification')` or `TrainConfig(task='regression')` — no auto-detection
- Callback hook for progress/logging, defaulting to a rich.console display for training metrics
- MLflow and other integrations plug in through the callback — not baked into the library
- Runtime guard on first iteration: check that no autograd graph is being built, warn if gradients detected

### Energy tracking design
- Energy tracked during both inference and learning phases
- Per-step energy: rolling window of recent K values only (for real-time monitoring) — not stored in full
- Per-batch and per-epoch energy: full history stored in the training history object
- All energy data lives inside the training history object (history.energy.per_batch, history.energy.per_epoch)
- Callback can access history mid-training for live monitoring

### Test scope & correctness criteria
- Structural tests only — no numerical convergence tests (convergence demonstrated by CIFAR-10 in Phase 4)
- Verify: gain-modulated error computation, correct W_out^T supervised error projection, correct shapes
- Synchronous latent updates are crucial for correctness but verifiable by code inspection — no dedicated test needed
- Runtime no-autograd guard tested indirectly (training runs without warning)
- All tests use synthetic data only (small deterministic tensors, seeded random) — fast, reproducible, no downloads

### Claude's Discretion
- TrainConfig default values for T_infer, T_learn, learning rates
- Training history dataclass field names and structure
- Rich console display formatting and verbosity
- Rolling window size K for per-step energy
- Early stopping convergence threshold default
- Specific synthetic test fixtures

</decisions>

<specifics>
## Specific Ideas

- Training code should be clean and modular — separate named phases (inference loop, learning loop), not tangled spaghetti
- Callback hook should be flexible enough to support a future GUI for live energy monitoring
- Rich.console default gives a nice out-of-box experience without requiring MLflow setup
- "Just check for grads on the first loop" — lightweight runtime guard, not heavy validation

</specifics>

<deferred>
## Deferred Ideas

- MLflow integration example — demonstrate through callback in Phase 4's CIFAR-10 example
- Live GUI for energy monitoring — future capability, callback hook provides the access point
- PyTorch Lightning wrapper — not needed given PCN's non-backprop training, but could be revisited if demand exists

</deferred>

---

*Phase: 03-training-energy-tests*
*Context gathered: 2026-02-24*
