# Architecture Research: pcn-torch

## Package Structure

```
pcn-torch/                          # Repository root (pip name: pcn-torch)
├── pyproject.toml                  # Build config, dependencies, metadata
├── LICENSE
├── README.md
├── pcn_torch/                      # Importable package (import pcn_torch)
│   ├── __init__.py                 # Public API re-exports
│   ├── layers.py                   # PCNLayer (nn.Module)
│   ├── network.py                  # PredictiveCodingNetwork (nn.Module)
│   ├── trainer.py                  # train_pcn(), test_pcn() loops
│   ├── activations.py              # Activation fns + derivatives (ReLU, tanh, etc.)
│   ├── energy.py                   # Energy/loss computation utilities
│   └── _types.py                   # Type aliases, protocols, constants
├── tests/
│   ├── __init__.py
│   ├── test_layers.py
│   ├── test_network.py
│   ├── test_trainer.py
│   ├── test_activations.py
│   └── test_energy.py
└── examples/
    └── cifar10.py                  # Standalone CIFAR-10 classification script
```

### Naming Convention

- **PyPI name**: `pcn-torch` (pip install pcn-torch)
- **Import name**: `pcn_torch` (import pcn_torch)
- Follows the standard convention where hyphens in distribution names map to underscores in import names (e.g., `pytorch-lightning` -> `lightning`, `torch-cluster` -> `torch_cluster`).

---

## Core Components

### Component 1: `activations.py` -- Activation Function Registry

- **Responsibility**: Provide paired (activation, derivative) callables. The paper requires explicit derivative computation since autograd is not used. Each activation must ship with its own derivative function.
- **Inputs/Outputs**:
  - Input: A string name or callable.
  - Output: A `(fn, fn_deriv)` tuple of `Callable[[Tensor], Tensor]`.
- **Dependencies**: `torch` only.
- **Key contents**:
  - `relu` / `relu_deriv`: Default activation per the paper.
  - `tanh` / `tanh_deriv`: Common alternative.
  - `sigmoid` / `sigmoid_deriv`: Common alternative.
  - `identity` / `identity_deriv`: For linear layers / debugging.
  - `get_activation(name: str) -> Tuple[Callable, Callable]`: Lookup by name.
- **Design note**: Activation derivatives are not computed via autograd. They are explicit elementwise functions (e.g., `relu_deriv(a) = (a > 0).float()`). This is a core PCN principle -- all gradient computation is local and manual.

### Component 2: `layers.py` -- PCNLayer Module

- **Responsibility**: Encapsulate a single generative layer of the PCN: weight matrix W^(l), activation function f^(l), and derivative f^(l)'. Computes prediction (x_hat) and preactivation (a) for a given layer-above input.
- **Inputs/Outputs**:
  - `__init__(in_dim, out_dim, activation_fn, activation_deriv)`: Dimensions are (out_dim, in_dim) = (d_l, d_{l+1}) -- the weight matrix maps FROM the layer above TO the layer below.
  - `forward(x_above: Tensor[B, d_{l+1}]) -> Tuple[Tensor[B, d_l], Tensor[B, d_l]]`: Returns (x_hat, a) where a = x_above @ W.T and x_hat = f(a).
- **Dependencies**: `torch.nn`, `activations.py`.
- **Key details**:
  - Inherits from `nn.Module`.
  - `self.W = nn.Parameter(torch.empty(out_dim, in_dim))` with Xavier uniform init.
  - No bias term (per paper specification).
  - Stores both `activation_fn` and `activation_deriv` as attributes.
  - The `forward()` method uses row-batch convention: input shape (B, d_{l+1}), output shape (B, d_l).

### Component 3: `network.py` -- PredictiveCodingNetwork Module

- **Responsibility**: Manage the full PCN hierarchy: a stack of PCNLayers, a linear readout layer (for supervised learning), latent initialization, and error computation across all layers.
- **Inputs/Outputs**:
  - `__init__(dims: List[int], output_dim: int)`: dims = [d_0, d_1, ..., d_L], output_dim = d_out.
  - `init_latents(batch_size, device) -> List[Tensor]`: Returns L tensors of shape (B, d_l) for l=1..L, initialized from N(0,1).
  - `compute_errors(inputs_latents: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]`: Computes errors E^(l) and gain-modulated errors H^(l) for all l=0..L-1.
  - `readout(x_L: Tensor) -> Tensor`: Linear map from top latent to output prediction y_hat.
- **Dependencies**: `layers.py`, `torch.nn`.
- **Key details**:
  - `self.layers = nn.ModuleList([PCNLayer(...) for l in range(L)])`.
  - `self.readout = nn.Linear(dims[-1], output_dim, bias=False)`.
  - The readout weight `W_out` is a separate `nn.Linear` layer, not a `PCNLayer` (it has no activation and uses standard linear projection).
  - `compute_errors()` iterates through all layers, calls each layer's `forward()`, and computes E^(l) = X^(l) - X_hat^(l) and H^(l) = E^(l) * f'(A^(l)).
  - Does NOT contain training logic -- separation of model definition from training procedure.

### Component 4: `trainer.py` -- Training and Testing Loops

- **Responsibility**: Implement the two-phase training loop (inference then learning) and the test-time inference loop. Orchestrates the interaction between the network, data, and hyperparameters.
- **Inputs/Outputs**:
  - `train_pcn(model, data_loader, num_epochs, eta_infer, eta_learn, T_infer, T_learn, device) -> dict`: Runs training, returns metrics (energy trajectories, accuracies).
  - `test_pcn(model, data_loader, T_infer, eta_infer, device) -> dict`: Runs test-time inference with frozen weights, returns predictions and metrics.
- **Dependencies**: `network.py`, `torch`, `torch.nn.functional`.
- **Key details**:
  - All updates run under `torch.no_grad()` -- this is not a standard PyTorch training loop.
  - **Inference phase** (T_infer steps): Compute errors via `model.compute_errors()`, compute supervised error, compute E^(L) = E_sup @ W_out, then update latents X^(l) via gradient descent on energy.
  - **Learning phase** (T_learn steps): With latents fixed from inference, recompute errors (since weights change), update W^(l) and W_out via batch-averaged local gradients.
  - Weight updates are done via direct tensor mutation (`weights[l] -= eta_learn * grad`), NOT via an optimizer. The `weights` list holds references to `layer.W` and `model.readout.weight` tensors.
  - Mixed precision support via `torch.amp.autocast`.
  - Target encoding: one-hot for classification, raw values for regression.

### Component 5: `energy.py` -- Energy Computation Utilities

- **Responsibility**: Compute the PCN energy (total prediction error) for monitoring, logging, and optional early stopping.
- **Inputs/Outputs**:
  - `compute_energy(errors: List[Tensor], eps_sup: Optional[Tensor]) -> Tensor`: Returns scalar batch-averaged energy.
  - `compute_energy_per_layer(errors, eps_sup) -> List[Tensor]`: Returns per-layer energy contributions for diagnostics.
- **Dependencies**: `torch` only.
- **Key details**:
  - Energy = (1/2B) * sum_b sum_l ||E^(l)_b||^2 + (1/2B) * sum_b ||E^sup_b||^2.
  - Used for tracking training progress (energy trajectories as shown in the paper).
  - Not used for gradient computation (all gradients are explicit).

### Component 6: `_types.py` -- Type Aliases and Constants

- **Responsibility**: Centralize type definitions, protocols, and shared constants to avoid circular imports and improve code readability.
- **Inputs/Outputs**: N/A (no functions, just type definitions).
- **Dependencies**: `torch`, `typing`.
- **Key contents**:
  - `ActivationFn = Callable[[Tensor], Tensor]`
  - `ActivationDeriv = Callable[[Tensor], Tensor]`
  - `ActivationPair = Tuple[ActivationFn, ActivationDeriv]`
  - Default hyperparameter values as named constants (e.g., `DEFAULT_T_INFER = 50`).

### Component 7: `__init__.py` -- Public API Surface

- **Responsibility**: Define what is importable from `pcn_torch`. Re-export the key classes and functions so users can write `from pcn_torch import PredictiveCodingNetwork, train_pcn`.
- **Key exports**:
  - `PCNLayer` (from `layers`)
  - `PredictiveCodingNetwork` (from `network`)
  - `train_pcn`, `test_pcn` (from `trainer`)
  - Activation pairs (from `activations`)
  - `__version__`

---

## Data Flow

### 1. Inference Phase (Latent Updates -- Training Time)

```
Input: x_batch (B, d_0), y_batch (B, d_out)
                    |
                    v
       +-----------+------- init_latents() -----------+
       |           |                                   |
    X^(0)       X^(1)...X^(L) ~ N(0,1)            W^(0)...W^(L-1), W_out
    (clamped)   (free to update)                   (frozen during inference)
       |           |                                   |
       +===========+===================================+
       |           FOR t = 1..T_infer:                 |
       |                                               |
       |  1. compute_errors(inputs_latents):           |
       |     For each layer l = 0..L-1:                |
       |       A^(l) = X^(l+1) @ W^(l).T              |
       |       X_hat^(l) = f^(l)(A^(l))               |
       |       E^(l) = X^(l) - X_hat^(l)              |
       |       H^(l) = E^(l) * f'^(l)(A^(l))          |
       |                                               |
       |  2. Supervised error:                         |
       |       Y_hat = X^(L) @ W_out.T                |
       |       E_sup = Y_hat - Y                      |
       |       E^(L) = E_sup @ W_out                  |
       |                                               |
       |  3. Latent gradient + update (l = 1..L):      |
       |       G_X^(l) = E^(l) - H^(l-1) @ W^(l-1)   |
       |       X^(l) -= eta_infer * G_X^(l)            |
       |                                               |
       +===============================================+
                    |
                    v
       Latents X^(1)...X^(L) now at inferred values
```

**Key invariant**: All errors are computed from a consistent snapshot BEFORE any latent is updated within a single step. This is the synchronous update scheme described in the paper.

### 2. Learning Phase (Weight Updates -- Training Time)

```
Input: Inferred latents X^(1)...X^(L) (frozen from inference phase)
       Clamped X^(0), target Y
                    |
                    v
       +===============================================+
       |           FOR t = 1..T_learn:                 |
       |                                               |
       |  1. Recompute errors (weights changed!):      |
       |     compute_errors(inputs_latents):           |
       |       E^(l), H^(l) for l = 0..L-1            |
       |                                               |
       |  2. Recompute supervised error:               |
       |       Y_hat = X^(L) @ W_out.T                |
       |       E_sup = Y_hat - Y                      |
       |                                               |
       |  3. Weight gradient + update (l = 0..L-1):    |
       |       G_W^(l) = -(1/B) * H^(l).T @ X^(l+1)  |
       |       W^(l) -= eta_learn * G_W^(l)            |
       |                                               |
       |  4. Readout weight update:                    |
       |       G_W_out = (1/B) * E_sup.T @ X^(L)      |
       |       W_out -= eta_learn * G_W_out            |
       |                                               |
       +===============================================+
                    |
                    v
       Updated W^(0)...W^(L-1), W_out
```

**Key invariant**: Latents are FIXED during learning. Only weights change. Errors are recomputed each step because weight changes alter predictions.

### 3. Prediction (Test Time)

```
Input: x_test (B, d_0)
       Frozen W^(0)...W^(L-1), W_out
                    |
                    v
       init_latents() -> X^(1)...X^(L) ~ N(0,1)
                    |
       +===========+===================================+
       |           FOR t = 1..T_infer:                 |
       |                                               |
       |  Same as inference phase in training,         |
       |  BUT with a dummy target or by monitoring     |
       |  energy convergence.                          |
       |                                               |
       |  For supervised: target Y is provided and     |
       |  E_sup / E^(L) are computed exactly as in     |
       |  training inference. This guides latents      |
       |  toward a representation consistent with      |
       |  the input AND the label signal.              |
       |                                               |
       |  For pure prediction (no target known):       |
       |  E^(L) = 0 (unsupervised inference on top     |
       |  layer), then Y_hat = X^(L) @ W_out.T        |
       |  after convergence.                           |
       |                                               |
       +===============================================+
                    |
                    v
       Y_hat = X^(L) @ W_out.T -> predicted output
       Classification: argmax(Y_hat)
```

**Note**: At test time, the paper runs inference WITH the target (to evaluate accuracy), since the supervised error signal guides latents. For deployment prediction without a known target, the unsupervised mode (E^(L) = 0) would be used. This is an important design consideration that should be made explicit in the API.

---

## Suggested Build Order

### Phase 1: Foundation (no dependencies on other pcn_torch modules)

1. **`_types.py`** -- Type aliases, constants. Zero logic, zero risk.
2. **`activations.py`** -- Activation functions + derivatives. Self-contained, easily testable with unit tests (e.g., verify relu_deriv matches numerical gradient). Must be correct before anything else, since every other component depends on correct derivatives.

### Phase 2: Core Model (depends on Phase 1)

3. **`layers.py`** -- PCNLayer. Depends on `activations.py` for default activation pair. Test: construct a layer, pass a random tensor through `forward()`, verify shapes and that output matches manual `x @ W.T` then `relu(...)`.
4. **`network.py`** -- PredictiveCodingNetwork. Depends on `layers.py`. Test: construct a network, call `init_latents()`, call `compute_errors()`, verify shapes and mathematical correctness against hand-computed examples.

### Phase 3: Training Loop (depends on Phase 2)

5. **`energy.py`** -- Energy computation. Depends only on `torch`. Test: pass known errors, verify energy matches hand calculation.
6. **`trainer.py`** -- Training and testing loops. Depends on `network.py`, `energy.py`. Test: run a tiny network (e.g., dims=[4, 3, 2], output_dim=2) on synthetic data for a few steps, verify energy decreases.

### Phase 4: Integration and Packaging

7. **`__init__.py`** -- Wire up public API re-exports.
8. **`pyproject.toml`** -- Package metadata, dependencies (torch >= 2.0), build system.
9. **`examples/cifar10.py`** -- End-to-end CIFAR-10 example reproducing paper results.
10. **`tests/`** -- Full test suite.

### Dependency Graph

```
_types.py  <--  activations.py  <--  layers.py  <--  network.py  <--  trainer.py
                                                                        ^
                                                 energy.py  -----------+
```

All arrows point from dependency to dependent. No circular dependencies. Each module can be developed and tested independently once its dependencies exist.

---

## Design Decisions

### D1: Flat package structure (no nested subpackages)

The library has approximately 5-6 source files. Nesting into subpackages like `pcn_torch/models/`, `pcn_torch/training/` adds import complexity without benefit at this scale. A flat structure follows the pattern of successful small PyTorch research libraries (e.g., lucidrains packages like `nGPT-pytorch` use a single package directory with a few modules). Subpackages should only be introduced if/when the library grows to support convolutional, recurrent, or graph-based PCN variants.

### D2: Separate `trainer.py` from `network.py`

The training loop is NOT a method on the network. Rationale:
- The PCN training loop is fundamentally different from standard PyTorch training (no autograd, no optimizer, no loss.backward()). Making it a separate function avoids the misleading pattern of `model.fit()`.
- Separation of model definition from training procedure follows PyTorch conventions (models define forward(), training loops are external).
- Users may want to customize the training loop (e.g., add logging, change learning schedule, implement early stopping) without subclassing the network.
- This mirrors the Lightning design philosophy: model (LightningModule) vs. training (Trainer).

### D3: Explicit activation derivatives (no autograd)

The entire point of PCNs is local computation without backpropagation. Using autograd to compute activation derivatives would undermine this principle and add unnecessary overhead. Each activation function comes paired with its hand-coded derivative. This is a core invariant of the library.

### D4: Weight updates via direct tensor mutation (no torch.optim)

PCN weight updates follow specific local Hebbian-like rules, not generic optimizer steps. Using `torch.optim.SGD` would obscure the PCN-specific update rules and add unnecessary abstraction. Weights are updated via `W -= eta * grad` directly on the parameter tensors. This keeps the code transparent and faithful to the paper.

### D5: `nn.Module` inheritance for layers and network

Even though autograd is not used for gradient computation, inheriting from `nn.Module` provides:
- Automatic parameter registration (model.parameters() works for serialization).
- Device management (model.to(device) moves all weights).
- State dict save/load (model.state_dict() / load_state_dict() for checkpointing).
- Familiar API for PyTorch users.
- Compatibility with `nn.ModuleList` for the layer stack.

### D6: Row-batch convention (B, d) everywhere

All tensors follow PyTorch's standard row-batch convention where the first dimension is batch. This matches the paper's Algorithm 3 (vectorized row-batch form) and is what users expect. No transposing surprises.

### D7: `examples/` directory instead of in-package examples

Examples are separate scripts, not importable modules. They depend on additional packages (torchvision for CIFAR-10) that are not required dependencies of the core library. This keeps the core package lightweight.

### D8: Energy tracking as a utility, not embedded in trainer

Energy computation is factored into `energy.py` so that:
- The trainer can optionally compute and log energy trajectories.
- Users can compute energy outside the training loop for analysis.
- The trainer code stays focused on the update logic.

---

## Reference: Similar Libraries

### Pattern 1: Minimal Research Package (lucidrains style)

Libraries like `nGPT-pytorch`, `x-transformers`, `vit-pytorch` by Phil Wang follow a pattern of:
- Single package directory with 1-3 Python files.
- `pyproject.toml` with minimal dependencies.
- `__init__.py` re-exports the main class.
- Published to PyPI with a hyphenated name, imported with underscores.
- No separate tests directory in many cases (tests via examples).
- **Relevance to pcn-torch**: Our library is closest to this pattern in scale. We add a proper tests/ directory for reliability.

### Pattern 2: Domain Library (torchvision style)

torchvision organizes by concern:
- `torchvision/models/` -- Model architectures (ResNet, VGG, etc.)
- `torchvision/datasets/` -- Dataset loaders
- `torchvision/transforms/` -- Image transformations
- `torchvision/ops/` -- Custom operations
- **Relevance to pcn-torch**: Too complex for v1. However, if pcn-torch grows to support multiple layer types (conv, recurrent, graph), a `pcn_torch/layers/` subpackage with `mlp.py`, `conv.py`, etc. would follow this pattern.

### Pattern 3: Framework (PyTorch Lightning style)

Lightning separates:
- `LightningModule` -- Research code (model + training_step)
- `Trainer` -- Engineering code (distributed, logging, checkpointing)
- `Callbacks` -- Extensibility hooks
- `LightningDataModule` -- Data preparation
- **Relevance to pcn-torch**: The model/trainer separation principle directly influences our `network.py` / `trainer.py` split. However, we do NOT use Lightning itself as a dependency -- the PCN training loop is too different from standard PyTorch training for Lightning's Trainer to be useful without extensive customization.

### Pattern 4: Existing PCN Libraries

- **PRECO** (bjornvz/PRECO): Package directory + example notebooks. Simple structure.
- **Torch2PC** (RobertRosenbaum/Torch2PC): Single Python file + example notebook. Minimal.
- **pcn-intro** (Monadillo/pcn-intro): Single Jupyter notebook. No package structure.
- **Relevance to pcn-torch**: These are research artifacts, not pip-installable libraries. pcn-torch aims to be the first properly packaged, pip-installable PCN library, filling a gap in the ecosystem.

---

## Quality Gate Checklist

- [x] **Components clearly defined with boundaries**: Six modules with explicit responsibilities. PCNLayer handles per-layer computation. PredictiveCodingNetwork handles hierarchy management. Trainer handles the training loop. Activations handle function pairs. Energy handles monitoring. Types handle shared definitions.
- [x] **Data flow direction explicit**: Three flows documented (inference, learning, prediction) with tensor shapes and update equations at each step. Key invariants (synchronous updates, frozen latents during learning) are called out.
- [x] **Build order implications noted**: Four-phase build order with explicit dependency chain. Each phase can be completed and tested before the next begins. No circular dependencies.
