# Requirements: pcn-torch

**Defined:** 2026-02-20
**Core Value:** A clean, PyTorch-native PCN implementation that lets anyone empirically explore predictive coding on their own problems with minimal friction.

## v1 Requirements

### Package Setup

- [ ] **PKG-01**: Project uses pyproject.toml with hatchling build backend
- [ ] **PKG-02**: Package is installable via `pip install pcn-torch`
- [ ] **PKG-03**: Package published to PyPI as `pcn-torch`
- [ ] **PKG-04**: README with installation instructions, quickstart code, and API overview
- [ ] **PKG-05**: MIT license included

### Core Model

- [ ] **MOD-01**: PCNLayer class (nn.Module) with weight matrix, activation function, and its derivative
- [ ] **MOD-02**: PredictiveCodingNetwork class (nn.Module) managing the full layer hierarchy
- [ ] **MOD-03**: Random latent initialization for each forward pass
- [ ] **MOD-04**: Prediction error and gain-modulated error computation across all layers
- [ ] **MOD-05**: Linear readout layer (no bias) mapping top latent to output prediction
- [ ] **MOD-06**: Configurable layer dimensions, activation functions, and hyperparameters

### Training

- [ ] **TRN-01**: Inference loop — iterative latent updates under torch.no_grad()
- [ ] **TRN-02**: Learning loop — local Hebbian-like weight updates, batch-averaged
- [ ] **TRN-03**: Supervised error signal propagated into top latent layer
- [ ] **TRN-04**: Support for classification (one-hot targets) and regression (continuous targets)
- [ ] **TRN-05**: Mixed precision support (autocast) during inference on CUDA

### Energy Tracking

- [ ] **NRG-01**: Per-step energy logging during both inference and learning
- [ ] **NRG-02**: Batch-averaged energy trajectories accessible after training

### Example

- [ ] **EXA-01**: CIFAR-10 classification example script that users can run
- [ ] **EXA-02**: Example produces reasonable accuracy (demonstrates the network works)

### Testing

- [ ] **TST-01**: Unit tests for synchronous latent updates (snapshot-then-update, not interleaved)
- [ ] **TST-02**: Unit tests for gain-modulated error computation
- [ ] **TST-03**: Unit test verifying no autograd graph is built during training
- [ ] **TST-04**: Unit test for correct top-layer supervised error signal (W_out^T projection)

## v2 Requirements

### Extended Features

- **EXT-01**: Unsupervised learning mode (no readout layer, eps^(L) = 0)
- **EXT-02**: Inference early stopping (anytime inference with convergence detection)
- **EXT-03**: Per-layer energy decomposition for diagnostics
- **EXT-04**: Convolutional PCN layers
- **EXT-05**: Hybrid predictive coding (amortized initialization)
- **EXT-06**: Pretrained CIFAR-10 weights downloadable from GitHub Releases

## Out of Scope

| Feature | Reason |
|---------|--------|
| Autograd-based weight updates | Defeats the purpose of local Hebbian-like learning |
| Standard torch.optim optimizers | PCN uses explicit local update rules, not backprop |
| Conv/Recurrent/Transformer layers | MLP-only for v1 to match the paper |
| Distributed training | Overkill for a research library v1 |
| Built-in data loading | Users handle their own data with standard PyTorch DataLoaders |
| Automatic hyperparameter tuning | Users tune manually for v1 |
| Web UI / dashboard | Library only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PKG-01 | - | Pending |
| PKG-02 | - | Pending |
| PKG-03 | - | Pending |
| PKG-04 | - | Pending |
| PKG-05 | - | Pending |
| MOD-01 | - | Pending |
| MOD-02 | - | Pending |
| MOD-03 | - | Pending |
| MOD-04 | - | Pending |
| MOD-05 | - | Pending |
| MOD-06 | - | Pending |
| TRN-01 | - | Pending |
| TRN-02 | - | Pending |
| TRN-03 | - | Pending |
| TRN-04 | - | Pending |
| TRN-05 | - | Pending |
| NRG-01 | - | Pending |
| NRG-02 | - | Pending |
| EXA-01 | - | Pending |
| EXA-02 | - | Pending |
| TST-01 | - | Pending |
| TST-02 | - | Pending |
| TST-03 | - | Pending |
| TST-04 | - | Pending |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 0
- Unmapped: 24 (will be mapped during roadmap creation)

---
*Requirements defined: 2026-02-20*
*Last updated: 2026-02-20 after initial definition*
