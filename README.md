# pcn-torch

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com/emv-dev/pcn-torch)
[![arXiv](https://img.shields.io/badge/arXiv-2506.06332-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.06332)

A clean, PyTorch-native implementation of Predictive Coding Networks (PCNs) from [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332).

## What are Predictive Coding Networks?

Predictive coding is a theory of cortical function proposing that the brain is fundamentally a hierarchical prediction machine. Each level in the cortical hierarchy maintains a generative model of the level below, continuously sending top-down predictions and receiving bottom-up prediction errors. This idea, formalized in computational neuroscience by Rao and Ballard (1999) and later developed into a comprehensive framework by Friston (2005), has become one of the most influential theories in theoretical neuroscience. Predictive Coding Networks (PCNs) translate this theory into a concrete machine learning algorithm.

The central quantity in a PCN is the **prediction error** at each layer. Given a hierarchy of latent representations `x^(0), x^(1), ..., x^(L)`, the network computes errors as `eps^(l) = x^(l) - f^(l)(W^(l) x^(l+1))`, where `f^(l)` is an activation function and `W^(l)` are the generative weights projecting from the layer above. The total energy of the network is `E = (1/2) sum_l ||eps^(l)||^2`. Training proceeds by minimizing this energy, which naturally decomposes into purely local computations.

PCN training operates on two timescales. In the **inference phase**, the input and (during training) the target are clamped, and the latent variables `x^(l)` are iteratively updated for `T` steps to reduce prediction errors throughout the hierarchy. Each latent update depends only on the errors immediately above and below it in the hierarchy -- no global signal needs to propagate end-to-end. In the subsequent **learning phase**, the weights `W^(l)` are updated using a local, Hebbian-like rule: each weight update is proportional to the outer product of the prediction error and the pre-synaptic activity at that layer.

This stands in contrast to backpropagation, where a global loss signal must propagate backward through the entire network via the chain rule, requiring storage of all intermediate activations and exact symmetric weight transport. PCNs require none of this. Each layer updates its own weights using only information available locally, making the algorithm biologically plausible and naturally suited to hardware architectures that support local learning rules. Furthermore, all operations in pcn-torch run under `torch.no_grad()` -- no autograd graph is ever constructed.

For a clear derivation of the full algorithm, including the inference and learning update equations and their relationship to variational free energy minimization, see Stenlund (2025) [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332).

## Installation

pcn-torch requires PyTorch. Install it first from [https://pytorch.org](https://pytorch.org) (choose your platform and CUDA version).

Then install pcn-torch:

```bash
pip install pcn-torch
```

> **Note:** PyTorch is not bundled with pcn-torch to let you choose your hardware variant (CPU, CUDA, ROCm).

## Quickstart

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from pcn_torch import (
    PredictiveCodingNetwork,
    RichCallback,
    TrainConfig,
    train_pcn,
    test_pcn,
)

# Toy classification: 64-dim input, 10 classes
X = torch.randn(500, 64)
y = torch.randint(0, 10, (500,))
loader = DataLoader(TensorDataset(X, y), batch_size=32)

model = PredictiveCodingNetwork(
    dims=[64, 128, 64],   # input=64, hidden=128, top_latent=64
    activation="relu",
    output_dim=10,
    mode="classification",
)

config = TrainConfig(
    task="classification",
    T_infer=50,
    lr_infer=0.05,
    lr_learn=0.005,
    num_epochs=4,
    callback=RichCallback(),
)

history = train_pcn(model, loader, config)
results = test_pcn(model, loader, config)
print(f"Accuracy: {results['accuracy']:.1%}")
```

## API Overview

| Name | Description |
|------|-------------|
| `PredictiveCodingNetwork` | Full PCN hierarchy: manages layers, latents, errors, and readout |
| `PCNLayer` | Single generative layer wrapping top-down weights and activation function |
| `train_pcn` | Train a network for multiple epochs using inference then learning loops |
| `test_pcn` | Evaluate a trained network on a DataLoader, returns accuracy and energy |
| `TrainConfig` | Dataclass holding all training hyperparameters (T_infer, lr_infer, etc.) |
| `TrainHistory` | Training history object returned by train_pcn (per-epoch and per-batch logs) |
| `EnergyHistory` | Energy trajectories (per_step, per_batch, per_epoch) within TrainHistory |
| `RichCallback` | Rich-powered progress bar and live metrics display during training |
| `TrainCallback` | Base class for custom callbacks; subclass to add your own logging |
| `compute_energy` | Compute scalar PCN energy from a network's current error state |
| `compute_energy_per_layer` | Per-layer energy breakdown as a list of floats |
| `get_activation` | Retrieve an (activation_fn, derivative_fn) pair by name ("relu", "tanh", "sigmoid") |
| `PCNErrors` | NamedTuple holding per-layer errors (eps) and gain-modulated errors (h) |
| `ActivationFn` | Type alias for activation functions: `Callable[[Tensor], Tensor]` |
| `ActivationDeriv` | Type alias for derivative functions: `Callable[[Tensor], Tensor]` |
| `ActivationPair` | NamedTuple of (fn: ActivationFn, deriv: ActivationDeriv) |
| `__version__` | Package version string (e.g., "1.0.0") |

## Results

The `examples/cifar10.py` script trains a 3-hidden-layer MLP PCN on CIFAR-10:

| Metric | Value |
|--------|-------|
| Architecture | dims=[3072, 1024, 1024, 512], output_dim=10 |
| Inference steps (T_infer) | 20 |
| Epochs | 10 |
| Batch size | 128 |
| Expected test accuracy | 45-55% |
| Training time (CPU) | ~5-15 minutes |

Note: A fully-connected MLP is architecturally limited on CIFAR-10 regardless of the learning rule -- both backprop and PCN achieve similar accuracy with the same architecture. The example demonstrates that the PCN learning algorithm works correctly, not that MLPs are competitive with convolutional networks on vision tasks.

## How It Works

During the **inference phase**, each latent representation `x^(l)` is iteratively updated for `T_infer` steps to minimize local prediction errors. The update for each latent depends only on the errors at its own layer and the layer below -- there is no global error signal. The network converges to an internal state that best explains the clamped input (and target, during training) under the generative model.

During the **learning phase**, the weights `W^(l)` at each layer are updated using a local Hebbian-like rule. The gradient of the energy with respect to each weight matrix reduces to the outer product of the prediction error and the pre-synaptic activation, averaged over the batch. All operations run under `torch.no_grad()` -- no autograd graph is constructed.

## References

- **Paper:** Stenlund, M. (2025). "Introduction to Predictive Coding Networks for Machine Learning." [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332)
- **Reference implementation:** [github.com/Monadillo/pcn-intro](https://github.com/Monadillo/pcn-intro)
