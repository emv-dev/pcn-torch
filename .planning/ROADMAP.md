# Roadmap: pcn-torch

## Overview

This roadmap delivers pcn-torch as a pip-installable PyTorch library implementing Predictive Coding Networks from arXiv:2506.06332v1. The build follows the module dependency chain exactly: types and activations first, then the network hierarchy, then training loops with energy tracking and correctness tests, and finally the CIFAR-10 example with PyPI publishing. Four phases take the project from empty repo to published package.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (e.g., 2.1): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Package scaffold, PCNLayer, types, activations, and configuration (2026-02-20)
- [x] **Phase 2: Core Model** - PredictiveCodingNetwork hierarchy, error computation, and readout (2026-02-23)
- [x] **Phase 3: Training + Energy + Tests** - Inference/learning loops, energy tracking, and correctness tests (2026-02-24)
- [ ] **Phase 4: Integration + Publishing** - CIFAR-10 example, PyPI publish, and README

## Phase Details

### Phase 1: Foundation
**Goal**: Developers can import pcn_torch and instantiate a configured PCNLayer with chosen activation function
**Depends on**: Nothing (first phase)
**Requirements**: PKG-01, PKG-05, MOD-01, MOD-06
**Success Criteria** (what must be TRUE):
  1. Running `pip install -e .` in the repo installs pcn-torch locally without errors
  2. `from pcn_torch import PCNLayer` succeeds and PCNLayer is an nn.Module subclass
  3. PCNLayer accepts configurable dimensions and activation function, and its forward pass produces predictions of the correct shape
  4. Ruff, mypy, and pytest all pass in CI (GitHub Actions) across Python 3.10-3.13
  5. Activation functions (sigmoid, tanh, relu) are available with explicit hand-coded derivatives (no autograd)
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md -- Package scaffold, types, activations (with hand-coded derivatives), PCNLayer, CI, and tests (2026-02-20)

### Phase 2: Core Model
**Goal**: Developers can construct a full PredictiveCodingNetwork with configurable layers and obtain predictions and error signals
**Depends on**: Phase 1
**Requirements**: MOD-02, MOD-03, MOD-04, MOD-05
**Success Criteria** (what must be TRUE):
  1. `PredictiveCodingNetwork(dims=[3072, 500, 500, 10])` constructs a valid network with the correct number of layers
  2. Calling `init_latents(batch_size)` produces randomly initialized latent variables at each layer
  3. `compute_errors()` returns prediction errors and gain-modulated errors across all layers with correct shapes
  4. The readout layer is `nn.Linear(..., bias=False)` and maps the top latent to output predictions
  5. The supervised error signal is correctly projected back to latent space via W_out^T
**Plans**: 1 plan

Plans:
- [x] 02-01-PLAN.md -- PredictiveCodingNetwork class, PCNErrors, latent init, error computation, readout, and tests (2026-02-23)

### Phase 3: Training + Energy + Tests
**Goal**: A PredictiveCodingNetwork can be trained end-to-end on batched data using local Hebbian-like rules, with energy tracking and verified algorithmic correctness
**Depends on**: Phase 2
**Requirements**: TRN-01, TRN-02, TRN-03, TRN-04, TRN-05, NRG-01, NRG-02, TST-01, TST-02, TST-03, TST-04
**Success Criteria** (what must be TRUE):
  1. `train_pcn(network, dataloader, ...)` trains the network for a specified number of epochs, with all operations under `torch.no_grad()`
  2. Inference loop iteratively updates latent variables for T_infer steps; learning loop updates weights for T_learn steps with batch-averaged gradients
  3. The network trains on both classification (one-hot targets) and regression (continuous targets) without code changes
  4. Energy trajectories (per-step, per-batch) are logged during both inference and learning, and are accessible after training
  5. Unit tests verify: synchronous latent updates (not interleaved), gain-modulated error computation, no autograd graph construction, and correct W_out^T supervised error projection
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md -- Energy module, TrainConfig/TrainHistory/TrainCallback, RichCallback, train_pcn, test_pcn training loops (2026-02-24)
- [x] 03-02-PLAN.md -- Algorithmic correctness tests (TST-02, TST-03, TST-04), energy tests, and public API exports (2026-02-24)

### Phase 4: Integration + Publishing
**Goal**: Anyone can `pip install pcn-torch`, run the CIFAR-10 example, and see the library work
**Depends on**: Phase 3
**Requirements**: EXA-01, EXA-02, PKG-02, PKG-03, PKG-04
**Success Criteria** (what must be TRUE):
  1. `pip install pcn-torch` from PyPI installs the package (with torch as a peer dependency)
  2. Running `python examples/cifar10.py` trains a PCN on CIFAR-10 and achieves reasonable accuracy (demonstrating the network works)
  3. The README contains installation instructions (including "install PyTorch first"), a quickstart code snippet, and an API overview
  4. The package version is 0.1.0, derived from git tags via hatch-vcs
**Plans**: TBD

Plans:
- [ ] 04-01: CIFAR-10 example, public API surface (__init__.py), README, and PyPI publish

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 1/1 | Complete | 2026-02-20 |
| 2. Core Model | 1/1 | Complete | 2026-02-23 |
| 3. Training + Energy + Tests | 2/2 | Complete | 2026-02-24 |
| 4. Integration + Publishing | 0/1 | Not started | - |

---
*Roadmap created: 2026-02-20*
*Last updated: 2026-02-24 (Phase 3 complete)*
