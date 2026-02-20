# pcn-torch

## What This Is

A PyTorch library implementing Predictive Coding Networks (PCNs) as described in Stenlund's "Introduction to Predictive Coding Networks for Machine Learning" (arXiv 2506.06332). Published to PyPI as `pcn-torch`, installable via `pip install pcn-torch` or `uv add pcn-torch`. Provides MLP-based PCN layers and networks for supervised learning (classification and regression).

## Core Value

A clean, PyTorch-native PCN implementation that lets anyone empirically explore predictive coding on their own problems with minimal friction.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] PCNLayer module wrapping top-down generative weights, activation, and derivative
- [ ] PredictiveCodingNetwork module managing the full hierarchy (latent init, error computation)
- [ ] Supervised readout layer (linear, no bias) with supervised error signal
- [ ] Inference loop: iterative latent variable updates via local gradient descent
- [ ] Learning loop: weight updates via local Hebbian-like rules (batch-averaged)
- [ ] Support for both classification (one-hot targets) and regression (continuous targets)
- [ ] Vectorized row-batch form matching Algorithm 3 from the paper
- [ ] Configurable hyperparameters: layer dims, T_infer, T_learn, eta_infer, eta_learn, activation fn
- [ ] CIFAR-10 example script demonstrating supervised classification
- [ ] Published to PyPI as `pcn-torch`
- [ ] README with installation, quickstart, and API reference

### Out of Scope

- Convolutional, recurrent, or graph-based PCN layers — MLP-only for v1
- Unsupervised learning mode — supervised only for v1
- Hybrid predictive coding (amortized initialization) — vanilla PCN only
- Web UI / dashboard — library only
- Pre-trained model weights — users train their own

## Context

- Based on arXiv paper 2506.06332v1 by Mikko Stenlund (May 2025)
- Reference implementation exists as a Colab notebook at github.com/Monadillo/pcn-intro
- Paper reports 99.92% top-1 accuracy on CIFAR-10 with 3-layer MLP PCN (3.6M params, 4 epochs, 4 min on L4 GPU)
- PCN key insight: two-timescale dynamics — fast inference (latent updates) then slow learning (weight updates), using only local computations (no backpropagation)
- The inference and learning updates bypass PyTorch autograd entirely (torch.no_grad()), computing gradients via explicit local rules
- Architecture: input x^(0) -> W^(0) -> x^(1) -> W^(1) -> x^(2) -> ... -> x^(L) -> W^out -> y_hat
- Energy function: L = (1/2) sum ||eps^(l)||^2 where eps^(l) = x^(l) - f^(l)(W^(l) x^(l+1))

## Constraints

- **Tech stack**: Python + PyTorch only — no other ML frameworks
- **Packaging**: PyPI distribution via standard Python packaging (pyproject.toml)
- **GitHub**: Repository at github.com/emv-dev/pcn-torch
- **Scope**: MLP layers only — no conv/recurrent/attention for v1
- **Learning mode**: Supervised only (classification + regression) for v1
- **Compatibility**: Python 3.10+, PyTorch 2.0+

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyTorch-native (nn.Module) | User wants familiar PyTorch patterns, paper already uses PyTorch | -- Pending |
| MLP-only for v1 | Keep scope focused, match paper exactly | -- Pending |
| Supervised only for v1 | Simpler scope, immediate practical value | -- Pending |
| Publish to PyPI | Standard Python distribution, widest reach | -- Pending |
| Name: pcn-torch | Signals PyTorch implementation, concise | -- Pending |

---
*Last updated: 2026-02-20 after initialization*
