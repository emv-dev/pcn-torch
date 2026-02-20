# Features Research: pcn-torch

> Research date: 2026-02-20
> Context: PyTorch library implementing Predictive Coding Networks based on arXiv:2506.06332v1 (Stenlund, May 2025)
> Target users: ML researchers wanting to experiment with predictive coding
> Scope: MLP-only v1, supervised learning (classification + regression)

---

## Table Stakes

Features users expect from any PyTorch ML research library. Without these, researchers will not adopt the library.

| Feature | Description | Complexity | Dependencies |
|---------|-------------|------------|--------------|
| **pip-installable package** | `pip install pcn-torch` from PyPI. Researchers will not clone a repo and manually add to sys.path. Standard pyproject.toml or setup.cfg with pinned dependency ranges. | Low | None |
| **nn.Module subclasses** | PCNLayer and PredictiveCodingNetwork must inherit from `torch.nn.Module`. This is non-negotiable for integration with the PyTorch ecosystem (serialization, device management, parameter iteration, etc.). The paper already specifies this. | Low | PyTorch |
| **GPU/CPU device transparency** | Model and tensors must work seamlessly on both CPU and CUDA devices via standard `.to(device)` patterns. The paper uses `autocast(device_type='cuda')` for mixed precision. | Low | nn.Module |
| **Reproducibility utilities** | `set_seed()` function or similar that seeds Python, NumPy, and PyTorch (including CUDA). Latent initialization is random, so reproducibility is especially critical for PCNs where inference starts from noise. | Low | None |
| **Serialization (save/load)** | Standard `torch.save(model.state_dict(), ...)` and `model.load_state_dict(...)` compatibility. Must work out of the box because models inherit nn.Module. Trained weights from the paper's CIFAR-10 experiment should be loadable. | Low | nn.Module |
| **Minimal working examples** | At least one complete, runnable example (e.g., CIFAR-10 classification from the paper). Researchers evaluate libraries by running examples first. A `examples/` directory or Jupyter notebooks. | Low | pip-installable |
| **Configurable architecture** | Users must be able to specify layer dimensions `[d_0, d_1, ..., d_L]`, output dimension, activation functions, and activation derivatives. The paper already shows this via `dims` and `output_dim` constructor args. | Low | nn.Module |
| **Configurable hyperparameters** | Inference rate (`eta_infer`), learning rate (`eta_learn`), number of inference steps (`T_infer`), number of learning steps (`T_learn`), batch size. These are fundamental PCN-specific parameters. | Low | None |
| **Type hints** | Full type annotations on all public APIs. Modern Python (3.10+) researchers expect this. Enables IDE autocompletion and catches errors early. | Low | None |
| **Docstrings** | Numpy-style or Google-style docstrings on all public classes and methods. Explain parameters, return types, and PCN-specific semantics (e.g., what "inference" means in this context vs. standard NN inference). | Low | None |
| **Unit tests** | pytest-based test suite covering: layer forward pass shapes, energy computation correctness, inference convergence (energy decreases), weight update direction, serialization round-trip, device transfer. | Med | pip-installable |
| **Batch training support** | Vectorized row-batch form as described in Algorithm 3 of the paper. Process B samples in parallel with independent latent variables per sample. | Med | nn.Module |
| **Xavier weight initialization** | Default Xavier uniform initialization as specified in the paper. Option to override with custom initialization. | Low | nn.Module |
| **Latent initialization** | Default Gaussian random initialization for latent variables. Must be freshly initialized per forward pass (per the PCN paradigm). | Low | None |
| **Mixed-precision support** | `torch.autocast` compatibility for CUDA. The paper already uses this. Important for GPU memory efficiency on research-grade hardware. | Low | GPU support |
| **License** | MIT or Apache-2.0 license. Research libraries that lack clear licensing are unusable in academic and corporate research settings. | Low | None |

## Differentiators

Features that would set pcn-torch apart from existing implementations (PRECO, Torch2PC, Monadillo/pcn-intro notebook, JPC in JAX).

| Feature | Description | Complexity | Dependencies |
|---------|-------------|------------|--------------|
| **Energy trajectory tracking** | Built-in recording of per-step, per-batch energy values during both inference and learning phases. The paper emphasizes energy trajectories as a primary diagnostic tool ("sanity check that the implementation is correct"). No existing PyTorch PCN library provides this as a first-class feature. Return structured data (not just print statements). | Med | Batch training |
| **Inference convergence callbacks** | Configurable early stopping for inference: halt when energy change falls below a threshold or latent update norm is small. The paper explicitly discusses "sample-wise early stopping" and "anytime inference." This is unique to PCNs and not available in any existing implementation. | Med | Energy tracking |
| **Separation of inference and learning APIs** | Expose `model.infer(x, y, ...)` and `model.learn(...)` as distinct, composable methods. Existing implementations (Torch2PC, pcn-intro notebook) bundle everything in a monolithic training loop. Researchers need to inspect, modify, and extend each phase independently. | Med | nn.Module |
| **Custom activation function support with derivatives** | Clean interface for providing paired (activation, derivative) functions. The paper's PCNLayer takes `activation_fn` and `activation_deriv` as separate args. Provide built-in pairs for ReLU, Tanh, Sigmoid, Swish, and a registration mechanism for custom pairs. | Low | nn.Module |
| **Per-layer energy decomposition** | Return energy broken down by layer: E_layer_0, E_layer_1, ..., E_sup. Enables researchers to diagnose which layers are struggling. No existing library provides this granularity. | Low | Energy tracking |
| **Comprehensive training loop** | A high-level `train_pcn()` function or Trainer class that handles the full epoch loop, batching, inference, learning, energy logging, and metric computation. Similar to what PyTorch Lightning provides but PCN-specific (two-phase per batch). | Med | Energy tracking, Batch training |
| **Classification and regression support** | Explicit support for both supervised tasks. Classification: one-hot encoding, cross-entropy-compatible supervised error, top-k accuracy metrics. Regression: MSE-based supervised error, R-squared metrics. The paper covers classification; regression is a natural extension. | Med | Configurable architecture |
| **Prediction/test-time inference** | A clean `model.predict(x)` or `model.test(x)` API that freezes weights, runs inference on new inputs, and returns predictions. The paper describes this but existing implementations leave it to the user. | Low | Separation of APIs |
| **Pretrained weights registry** | Ship or host pretrained weights for the paper's CIFAR-10 experiment (99.92% accuracy). Downloadable via `pcn_torch.load_pretrained('cifar10')`. Validates the implementation and gives users a starting point. | Low | Serialization |
| **Comprehensive API documentation** | Sphinx or MkDocs-based documentation site with: API reference, mathematical background linking to the paper, tutorials, and architecture diagrams. Most PCN implementations have minimal or no docs. | Med | Docstrings |
| **Logging integration** | Optional integration points for TensorBoard, Weights & Biases, or generic Python logging. Log energy trajectories, accuracies, and latent statistics during training without requiring users to write boilerplate. | Med | Energy tracking |
| **Latent state inspection** | Methods to retrieve and visualize intermediate latent states `x^(l)` after inference. Enables researchers to study learned representations, which is a core interest in PCN research. | Low | Separation of APIs |
| **Numerical stability safeguards** | Gradient clipping on latent updates, energy divergence detection, NaN/Inf checks. PCN inference is iterative and can diverge with poor hyperparameters. Warn or halt gracefully instead of producing silent garbage. | Med | Energy tracking |
| **Benchmarking suite** | Reproducible benchmark scripts for CIFAR-10 (and optionally MNIST, FashionMNIST) with expected accuracy ranges and timing. Enables regression testing and performance comparison across hardware. | Med | Examples, Batch training |

## Anti-Features

Things to deliberately NOT build in v1. These either add complexity without value, conflict with PCN philosophy, or are premature.

| Feature | Why Not |
|---------|---------|
| **Autograd-based weight updates** | PCN weight updates are local and Hebbian-like. Using `loss.backward()` and `optimizer.step()` would defeat the entire purpose. The paper explicitly operates under `torch.no_grad()`. Weight gradients are computed manually from prediction errors and presynaptic activity. |
| **Standard PyTorch optimizers (Adam, SGD, etc.)** | PCN learning uses its own local update rule: `W -= eta * (h^T @ x) / B`. Plugging in Adam or SGD on a global loss would make it a standard neural network, not a PCN. The two-timescale dynamics are fundamental. |
| **Convolutional / Recurrent / Transformer layers** | The paper's scope and v1 scope is MLP-only. The paper notes "PCNs extend naturally to convolutional, recurrent, and graph-based structures" but this is future work. Adding conv support prematurely would complicate the API and delay shipping. |
| **Unsupervised learning mode** | While the paper describes unsupervised PCN (Algorithm 1), v1 targets supervised learning only (classification + regression). Unsupervised mode has different use cases, evaluation metrics, and API needs. Better to do supervised well first. |
| **Hybrid predictive coding (amortized inference)** | The paper mentions this variant (Tschantz et al., 2022) where a feedforward network initializes latents. This is a research extension, not table stakes. Keep latent initialization simple (random) in v1. |
| **Distributed training / multi-GPU** | Premature optimization. Target audience is individual researchers with a single GPU. DataParallel/DistributedDataParallel adds significant complexity. Revisit when user demand materializes. |
| **Built-in data loading / preprocessing** | PyTorch's DataLoader and torchvision.transforms are the standard. Do not reinvent them. Provide examples showing how to use them with pcn-torch, but do not wrap or replace them. |
| **Web UI / dashboard** | Out of scope for a research library. TensorBoard integration (as a differentiator) is sufficient. Do not build custom visualization frontends. |
| **Spiking neural network support** | The paper mentions PCN implementation in spiking networks but this is an entirely different computational paradigm requiring specialized infrastructure. Out of scope. |
| **Bias terms in layers** | The paper explicitly states "No bias terms are used." Respect this design decision in v1. Bias support could be added later as an option if research demand warrants it. |
| **Automatic hyperparameter tuning** | Premature. Researchers want control over eta_infer, eta_learn, T_infer, T_learn. Provide sensible defaults and let them tune. Do not add Optuna/Ray Tune integration in v1. |

## Feature Dependencies

```
pip-installable
  |-> nn.Module subclasses
  |     |-> GPU/CPU device transparency
  |     |-> Serialization (save/load)
  |     |     |-> Pretrained weights registry
  |     |-> Configurable architecture
  |     |     |-> Custom activation function support
  |     |     |-> Classification and regression support
  |     |-> Batch training support
  |     |     |-> Mixed-precision support
  |     |     |-> Benchmarking suite
  |     |-> Separation of inference and learning APIs
  |     |     |-> Prediction/test-time inference
  |     |     |-> Latent state inspection
  |     |-> Energy trajectory tracking
  |     |     |-> Per-layer energy decomposition
  |     |     |-> Inference convergence callbacks
  |     |     |-> Numerical stability safeguards
  |     |     |-> Logging integration
  |-> Unit tests
  |-> Minimal working examples
  |     |-> Benchmarking suite
  |-> Type hints
  |-> Docstrings
  |     |-> Comprehensive API documentation
```

### Critical Path for MVP
1. nn.Module subclasses (PCNLayer, PredictiveCodingNetwork)
2. Configurable architecture + hyperparameters
3. Batch training support (vectorized)
4. Energy trajectory tracking
5. pip-installable package
6. Minimal working example (CIFAR-10)
7. Unit tests
8. Type hints + docstrings

## Competitive Landscape

### Existing PCN Implementations

| Library | Framework | Status | Strengths | Weaknesses |
|---------|-----------|--------|-----------|------------|
| **[Monadillo/pcn-intro](https://github.com/Monadillo/pcn-intro)** | PyTorch | Notebook only (2025) | Accompanies the paper (arXiv:2506.06332). Complete CIFAR-10 example. Clear code matching paper's algorithms. Pretrained weights available. | Not a library -- it is a single Jupyter notebook. No pip install. No API. No tests. No docs. Not reusable without copy-paste. |
| **[PRECO](https://github.com/bjornvz/PRECO)** | PyTorch | Library (2024) | Supports both PCNs and Predictive Coding Graphs (PCGs). GPU acceleration. Jupyter notebook examples. MIT licensed. Accompanies survey paper (arXiv:2407.04117). | Minimal README/docs. 16 GitHub stars. Small community. No pip install on PyPI. Limited API documentation. |
| **[Torch2PC](https://github.com/RobertRosenbaum/Torch2PC)** | PyTorch | Library (2022) | Supports multiple PCN algorithm variants (Strict, FixedPred, Exact). Works with nn.Sequential models. MNIST CNN example. | Only supports Sequential models. No pip install. Minimal docs. Single-function API (`PCInfer`). Limited extensibility. Dated. |
| **[JPC](https://github.com/thebuckleylab/jpc)** | JAX | Library (2024) | Under 1000 lines of code. Uses ODE solvers (Diffrax) for faster inference convergence. Supports discriminative, generative, and hybrid PC models. Analytical tools. Published paper (arXiv:2412.03676). | JAX-based, not PyTorch. Different ecosystem, paradigm, and user base. Functional design may not suit all researchers. |
| **[PyTorch-Hebbian](https://arxiv.org/abs/2102.00428)** | PyTorch | Research code (2021) | Framework for local learning rules in deep learning. Extends to CNNs. | Hebbian learning, not full predictive coding. Different scope. Dated. |
| **[PyHGF](https://github.com/ComputationalPsychiatry/pyhgf)** | JAX | Library (active) | Hierarchical Gaussian filtering (related to predictive coding). Modular. Well-documented. | Different theoretical framework (Bayesian filtering vs PCN). JAX-based. Not directly comparable. |
| **[dbersan/Predictive-Coding-Implementation](https://github.com/dbersan/Predictive-Coding-Implementation)** | PyTorch | Research code | Implements Whittington & Bogacz (2017). ImageNet comparison. | Single research reproduction. Not a reusable library. |

### Gap Analysis

The competitive landscape reveals clear gaps that pcn-torch can fill:

1. **No pip-installable PyTorch PCN library exists on PyPI.** Every existing implementation requires cloning a repo.
2. **No existing PyTorch implementation provides energy trajectory tracking as a first-class feature.** JPC (JAX) has some analytical tools, but nothing in the PyTorch ecosystem.
3. **No existing implementation separates inference and learning into composable APIs.** All bundle them in monolithic training loops.
4. **No existing implementation provides inference convergence callbacks or early stopping.** The paper discusses this but nobody has implemented it.
5. **No existing implementation has comprehensive documentation, type hints, or unit tests.** The bar is very low.
6. **The reference implementation (Monadillo/pcn-intro) is a notebook, not a library.** pcn-torch would be the proper library extraction of that work.

### Positioning

pcn-torch should position itself as:
- **The** reference PyTorch implementation of PCNs from arXiv:2506.06332v1
- First pip-installable PCN library for PyTorch
- Research-grade: reproducible, tested, documented, type-hinted
- Researcher-friendly: composable APIs, energy diagnostics, latent inspection
- Not trying to be a general-purpose framework; focused on MLP PCNs for supervised learning

### Key Competitive Advantages Over JPC (JAX)

JPC is the most technically sophisticated competitor but targets a fundamentally different ecosystem:
- PyTorch has 63% adoption among ML practitioners vs JAX's smaller share
- PyTorch's imperative style and debugging experience is preferred by many researchers
- pcn-torch would serve the large PyTorch community that cannot or does not want to switch to JAX
- JPC's functional paradigm is a barrier for researchers accustomed to PyTorch's object-oriented style

---

## Sources

- [arXiv:2506.06332v1 - Introduction to Predictive Coding Networks for Machine Learning (Stenlund, 2025)](https://arxiv.org/abs/2506.06332)
- [Monadillo/pcn-intro GitHub repository](https://github.com/Monadillo/pcn-intro)
- [PRECO - Predictive Coding Networks and Graphs (PyTorch)](https://github.com/bjornvz/PRECO)
- [Torch2PC - Predictive coding algorithms for PyTorch](https://github.com/RobertRosenbaum/Torch2PC)
- [JPC - Flexible Inference for Predictive Coding Networks in JAX](https://arxiv.org/html/2412.03676v1)
- [PyTorch-Hebbian: facilitating local learning (arXiv:2102.00428)](https://arxiv.org/abs/2102.00428)
- [PyHGF - Hierarchical Gaussian Filtering](https://github.com/ComputationalPsychiatry/pyhgf)
- [Predictive Coding Networks and Inference Learning: Tutorial and Survey (arXiv:2407.04117)](https://arxiv.org/abs/2407.04117)
