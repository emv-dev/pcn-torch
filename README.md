# pcn-torch

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com/emv-dev/pcn-torch)
[![arXiv](https://img.shields.io/badge/arXiv-2506.06332-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.06332)

A clean, PyTorch-native implementation of Predictive Coding Networks (PCNs) from [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332).

## What are Predictive Coding Networks?

Your brain is constantly guessing what happens next -- and learning from its mistakes. Predictive Coding Networks (PCNs) work the same way.

A PCN is a stack of layers where each layer guesses what the layer below should look like. It then compares that guess to what actually showed up. The difference is the **prediction error**. The total **energy** of the network is just the sum of all prediction errors across every layer -- and training means driving that energy down.

The clever part is that everything stays local. Training has two phases:

- **Inference phase:** The network adjusts its internal guesses (not the weights) to reduce errors across all layers -- like tuning a radio dial until the static clears. Each layer only looks at its immediate neighbors. No signal needs to travel end-to-end.
- **Learning phase:** Once the guesses have settled, each layer updates its own weights based on its local prediction errors only. Each layer learns from its own mistakes -- nothing else.

Unlike backpropagation, which sends a single error signal backward through the entire network, PCNs let each layer learn independently using only local information. This is closer to how the brain actually works, and it means the algorithm never needs to store intermediate activations or build a computation graph. All operations in pcn-torch run under `torch.no_grad()` -- no autograd graph is ever constructed.

For the full math and derivations, see Stenlund (2025) [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332).

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
| Architecture | dims=[3072, 1000, 500, 10], output_dim=10 |
| Training inference steps (T_infer) | 50 |
| Test inference steps (T_infer_test) | 500 |
| Epochs | 3 |
| Batch size | 500 |
| Expected test accuracy | 50-60% |
| Training time (GPU) | ~4-5 minutes |

**A note on test accuracy:** During training, PCNs clamp both the input and the label, so the network converges fast. At test time, only the input is clamped -- the network has to figure out the label on its own. This requires more inference steps (hence `T_infer_test=500` vs `T_infer=50`). Some PCN implementations run test inference with labels clamped, which produces near-perfect "accuracy" but only measures label reconstruction, not classification. Our `test_pcn` does genuine classification -- no labels at test time.

A fully-connected MLP is architecturally limited on CIFAR-10 regardless of the learning rule -- both backprop and PCN achieve similar accuracy with the same architecture. The example demonstrates that the PCN learning algorithm works correctly, not that MLPs are competitive with convolutional networks on vision tasks.

## How It Works

During the **inference phase**, for each training example the network runs `T_infer` steps of self-correction. Each layer looks at the errors above and below it and adjusts its internal state to reduce them. No signal needs to travel end-to-end -- every layer works in parallel using only its neighbors. The network settles into an internal state that best explains the input (and target, during training).

During the **learning phase**, the weights at each layer are updated based on local errors only. Each layer's weight update depends on its own prediction error and the activity feeding into it -- nothing else. All operations run under `torch.no_grad()`, so no autograd graph is ever built.

## References

- **Paper:** Stenlund, M. (2025). "Introduction to Predictive Coding Networks for Machine Learning." [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332)
- **Reference implementation:** [github.com/Monadillo/pcn-intro](https://github.com/Monadillo/pcn-intro)
