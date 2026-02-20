# Phase 2: Core Model - Research

**Researched:** 2026-02-20
**Domain:** PyTorch nn.Module hierarchy, predictive coding error computation, linear readout
**Confidence:** HIGH

## Summary

Phase 2 builds a `PredictiveCodingNetwork` class on top of the Phase 1 `PCNLayer` foundation. The network manages a stack of PCNLayers via `nn.ModuleList`, initializes latent variables from `N(0,1)`, computes prediction errors and gain-modulated errors across all layers, provides a linear readout layer (`nn.Linear` with no bias) that maps the top latent to output predictions, and supports supervised error projection back to latent space via `W_out^T`. The scope ends at model construction and error computation -- training loops, energy tracking, and tests are Phase 3.

The mathematical formulas are fully specified by arXiv:2506.06332v1 and have been extracted. The prediction error is `eps^(l) = x^(l) - x_hat^(l)`, the gain-modulated error is `h^(l) = f'(a^(l)) * eps^(l)`, the supervised error is `eps_sup = y_hat - y` where `y_hat = W_out @ x^(L)`, and the top-layer error signal is `eps^(L) = W_out^T @ eps_sup` (batch form: `eps_sup @ W_out`). Latents are initialized from standard normal `torch.randn(batch_size, d_l)`. The paper does not distinguish classification from regression in formulas -- both use the same squared-error energy; the difference is only in target encoding (one-hot vs continuous).

The primary implementation challenge is getting the constructor API right (accepting both `dims` list and pre-built `PCNLayer` instances) and ensuring error computation returns well-structured results. The gain modulation and supervised error formulas are straightforward elementwise and matrix operations. No novel patterns or libraries are needed beyond standard PyTorch.

**Primary recommendation:** Implement `PredictiveCodingNetwork` as a single file `src/pcn_torch/network.py` with all math following the paper exactly. Use a `NamedTuple` for the `compute_errors` return type to provide named access to prediction errors and gain-modulated errors. Keep the readout internal with a public `predict()` method. Use a string `mode` parameter ("classification" or "regression") that controls only target validation -- the math is identical.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Network construction API
- Accept both a dims list (convenience) and pre-built PCNLayer instances (full control)
- Global activation function with per-layer override capability
- Layers exposed publicly as `network.layers`

#### Latent initialization
- Latents stored on the network as `network.latents` (public attribute, inspectable)
- Auto-reset per batch (paper-defined behavior)
- Paper-prescribed initialization only -- no custom init function

#### Error computation interface
- `compute_errors(x, y=None)` -- pass input and optional target explicitly each call
- Returns errors for all layers at once (not per-layer methods)
- Errors stored on the network after computation (like latents) for inspection/debugging
- When `y` is provided, supervised error (target vs prediction projected via W_out^T) is computed and included

#### Readout layer
- Readout is internal (not a public attribute)
- Separate `predict()` method for getting output predictions
- Explicit mode flag for classification vs regression -- researcher to check the paper for how these differ (loss function, target handling)

### Claude's Discretion
- Return type for `compute_errors` (named tuple, dataclass, or similar structured type)
- Gain modulation implementation (follows paper formula exactly)
- Supervised error formula (follows paper exactly)
- Exact initialization distribution for latents (follows paper)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

No new libraries are introduced in Phase 2. The implementation uses only what was established in Phase 1.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >=2.0 | `nn.Module`, `nn.ModuleList`, `nn.Linear`, `nn.Parameter`, `torch.randn` | Foundation for all model components |
| Python | >=3.10 | `NamedTuple`, type annotations, `from __future__ import annotations` | Language features for structured returns and typing |

### Supporting
No new supporting libraries needed.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `NamedTuple` for error return | `dataclass` | Dataclass is mutable (bad for computation results), requires more boilerplate, and does not unpack like a tuple. NamedTuple is immutable, lightweight, and backward-compatible with tuple unpacking. |
| `NamedTuple` for error return | Plain `tuple` | Loses named access. Error results have 3-4 fields; positional access is error-prone. |
| `NamedTuple` for error return | Custom dataclass with `__iter__` | Over-engineering. NamedTuple provides named access + tuple unpacking natively. |

## Architecture Patterns

### Recommended Project Structure (Phase 2 additions)
```
src/pcn_torch/
    __init__.py         # Add PredictiveCodingNetwork export
    _types.py           # Add new type aliases if needed
    activations.py      # Unchanged from Phase 1
    layers.py           # Unchanged from Phase 1
    network.py          # NEW: PredictiveCodingNetwork class
```

### Pattern 1: Dual Constructor API (dims list OR PCNLayer instances)

**What:** The constructor accepts either a `dims` list for convenience or a sequence of pre-built `PCNLayer` instances for full control. A global activation applies when using `dims`; pre-built layers use whatever activation they were constructed with.

**When to use:** This is the locked decision from the user.

**Implementation approach:**
```python
class PredictiveCodingNetwork(nn.Module):
    def __init__(
        self,
        dims: Sequence[int] | None = None,
        layers: Sequence[PCNLayer] | None = None,
        *,
        activation: str = "relu",
        mode: str = "classification",
    ) -> None:
        super().__init__()
        if dims is not None and layers is not None:
            raise ValueError("Provide either 'dims' or 'layers', not both")
        if dims is not None:
            # Build layers from dims: dims = [d_0, d_1, ..., d_L]
            # Layer l maps from d_{l+1} -> d_l (generative direction)
            built_layers = []
            for l in range(len(dims) - 1):
                built_layers.append(
                    PCNLayer(dims[l + 1], dims[l], activation=activation)
                )
            self.layers = nn.ModuleList(built_layers)
        elif layers is not None:
            self.layers = nn.ModuleList(layers)
        else:
            raise ValueError("Must provide either 'dims' or 'layers'")
```
**Confidence:** HIGH -- follows standard PyTorch patterns.

**Critical dimension semantics:** The paper's generative model goes top-down. Layer `l` has weight `W^(l)` of shape `(d_l, d_{l+1})`, meaning it maps FROM layer `l+1` (above) TO layer `l` (below). When constructing from `dims = [d_0, d_1, ..., d_L]`:
- Layer 0: `PCNLayer(in_features=d_1, out_features=d_0)` -- W^(0) maps d_1 -> d_0
- Layer 1: `PCNLayer(in_features=d_2, out_features=d_1)` -- W^(1) maps d_2 -> d_1
- ...
- Layer L-1: `PCNLayer(in_features=d_L, out_features=d_{L-1})` -- W^(L-1) maps d_L -> d_{L-1}

The readout maps from `d_L` -> `d_out`. For `dims = [3072, 1000, 500, 10]`, the top latent dimension is `d_L = dims[-1] = 10`, and `d_out` equals the output dimension (10 for CIFAR-10 classification). In the paper's CIFAR-10 example, `d_L = d_out = 10`, so `W_out` is `(10, 10)`.

**Output dimension inference:** When using `dims`, the output dimension for the readout layer should be inferred as `dims[-1]` (the top latent dimension). This matches the paper where `d_out = d_L = 10`. However, for regression tasks where output dimension differs from the top latent, the constructor should also accept an explicit `output_dim` parameter. Default: `output_dim = dims[-1]`.

### Pattern 2: NamedTuple for Error Computation Results

**What:** A `NamedTuple` class that holds the results of `compute_errors()`, providing named access and tuple-like unpacking.

**When to use:** Claude's discretion area. Recommended as the return type for `compute_errors()`.

**Implementation:**
```python
from typing import NamedTuple

class PCNErrors(NamedTuple):
    """Results of error computation across all layers."""
    errors: list[Tensor]           # eps^(l) for l=0..L-1 -- prediction errors
    gm_errors: list[Tensor]        # h^(l) for l=0..L-1 -- gain-modulated errors
    supervised_error: Tensor | None  # eps_sup (y_hat - y), None if no target
    top_error: Tensor | None         # eps^(L) = eps_sup @ W_out, None if no target
```

**Why NamedTuple over dataclass:**
1. Immutable -- computation results should not be mutated after creation
2. Tuple unpacking works: `errors, gm_errors, eps_sup, eps_L = network.compute_errors(x, y)`
3. Lightweight -- no `__init__` boilerplate, no `@dataclass` decorator
4. Works with type checkers (mypy) natively
5. Hashable (bonus, not required)

**Confidence:** HIGH -- standard Python pattern, verified against mypy compatibility.

### Pattern 3: Latent Initialization from N(0,1)

**What:** Latents initialized from standard normal distribution using `torch.randn()`.

**When to use:** Every time `init_latents(batch_size)` is called.

**Paper reference:** Algorithm 2, line 4: "Initialize x^(l) <- small random values". Section 5.2 specifies standard Gaussian initialization. The PITFALLS.md research confirms `torch.randn(batch_size, d)` with scale=1.0 is the paper's prescription.

**Implementation:**
```python
def init_latents(self, batch_size: int) -> None:
    """Initialize latent variables from N(0,1) for all layers.

    Latents are stored as network.latents (list of Tensors).
    Layer 0 is the input (clamped, not a latent).
    Latents are for layers l=1..L.
    """
    device = next(self.parameters()).device
    self.latents: list[Tensor] = []
    for l in range(len(self.layers)):
        # self.layers[l] maps from d_{l+1} -> d_l
        # Latent x^(l+1) has dimension d_{l+1} = self.layers[l].weight.shape[1]
        # Wait -- dimension semantics need care here.
        # Actually: latent x^(l) has dimension d_l.
        # For l=1..L, d_l = self.layers[l-1].weight.shape[0] (out_features of layer l-1)
        # Alternatively: d_l = dims[l] from the original dims list.
        d_l = self._dims[l]  # stored during __init__
        self.latents.append(torch.randn(batch_size, d_l, device=device))
```

**Critical detail:** Latents correspond to layers 1 through L in the paper. Layer 0 is the input `x^(0)` which is clamped (not a latent variable). The latent for layer `l` has dimension `d_l` from the original `dims` list.

**Confidence:** HIGH -- paper explicitly specifies this.

### Pattern 4: Error Computation (Core Algorithm)

**What:** Compute prediction errors and gain-modulated errors for all layers.

**Paper formulas (per-sample, Section 3):**
- Preactivation: `a^(l) = W^(l) @ x^(l+1)` (generative direction, top-down)
- Prediction: `x_hat^(l) = f^(l)(a^(l))`
- Prediction error: `eps^(l) = x^(l) - x_hat^(l)`
- Gain-modulated error: `h^(l) = f'^(l)(a^(l)) * eps^(l)` (elementwise multiply)

**Batch form (row-batch convention, shape (B, d)):**
- Preactivation: `A^(l) = X^(l+1) @ W^(l).T` -- shape (B, d_l)
- Prediction: `X_hat^(l) = f^(l)(A^(l))` -- shape (B, d_l)
- Error: `E^(l) = X^(l) - X_hat^(l)` -- shape (B, d_l)
- Gain-modulated error: `H^(l) = f'^(l)(A^(l)) * E^(l)` -- shape (B, d_l)

**Implementation:**
```python
def compute_errors(self, x: Tensor, y: Tensor | None = None) -> PCNErrors:
    """Compute prediction errors and gain-modulated errors.

    Args:
        x: Input tensor of shape (batch_size, d_0). Clamped as x^(0).
        y: Optional target tensor. When provided, supervised error is computed.

    Returns:
        PCNErrors named tuple with errors, gm_errors, supervised_error, top_error.
    """
    # Build the full state vector: [x^(0), x^(1), ..., x^(L)]
    # x^(0) = input (clamped), x^(1)..x^(L) = self.latents
    states = [x] + self.latents  # length L+1

    errors: list[Tensor] = []
    gm_errors: list[Tensor] = []

    for l, layer in enumerate(self.layers):
        # Layer l: maps from x^(l+1) to predict x^(l)
        # forward(x_above) returns (prediction, preactivation)
        prediction, preactivation = layer(states[l + 1])
        eps = states[l] - prediction
        h = layer.activation_deriv(preactivation) * eps
        errors.append(eps)
        gm_errors.append(h)

    # Supervised error (if target provided)
    supervised_error = None
    top_error = None
    if y is not None:
        y_hat = self.predict()  # uses current x^(L)
        supervised_error = y_hat - y  # eps_sup, shape (B, d_out)
        top_error = supervised_error @ self._readout.weight  # eps^(L) = eps_sup @ W_out
        # Shape: (B, d_out) @ (d_out, d_L) = (B, d_L)

    # Store on network for inspection
    self.errors = PCNErrors(errors, gm_errors, supervised_error, top_error)
    return self.errors
```

**Confidence:** HIGH -- directly from paper Algorithm 2 + batch form derivation.

**Critical note on W_out^T projection:** The paper says `eps^(L) = W_out^T @ eps_sup` (per-sample, column vectors). In batch form with row vectors (B, d), this becomes `eps_sup @ W_out` (no transpose). Since `self._readout.weight` has shape `(d_out, d_L)` (nn.Linear convention), and `eps_sup` has shape `(B, d_out)`, the batch computation is:
```python
top_error = supervised_error @ self._readout.weight  # (B, d_out) @ (d_out, d_L) = (B, d_L)
```

### Pattern 5: Readout Layer and Prediction

**What:** Internal `nn.Linear(d_L, d_out, bias=False)` that maps top latent to output prediction.

**Paper reference:** `y_hat = W_out @ x^(L)` where `W_out in R^{d_out x d_L}`. No bias (Section 5.1).

**Implementation:**
```python
# In __init__:
self._readout = nn.Linear(dims[-1], output_dim, bias=False)

# predict() method:
def predict(self) -> Tensor:
    """Get output prediction from current top latent state.

    Returns:
        Prediction tensor of shape (batch_size, output_dim).
    """
    return self._readout(self.latents[-1])
    # Equivalent to: self.latents[-1] @ self._readout.weight.T
```

**Note on underscore prefix:** The readout is stored as `self._readout` (underscore prefix) to signal it is internal, per the locked decision that "Readout is internal (not a public attribute)." It is still registered as a submodule of `nn.Module` and its parameters will appear in `model.parameters()` and `model.state_dict()`.

**Confidence:** HIGH -- directly from paper.

### Pattern 6: Classification vs Regression Mode

**What:** An explicit mode flag that differentiates classification and regression.

**Paper finding:** The paper does NOT use different formulas for classification vs regression. The supervised energy is always `L_sup = (1/2) ||eps_sup||^2` where `eps_sup = y_hat - y`. Both modes use the same squared-error energy. The differences are:
1. **Target encoding:** Classification uses one-hot targets (`y in R^{d_out}`), regression uses raw continuous values
2. **Prediction interpretation:** Classification uses `argmax(y_hat)` for the predicted class; regression uses `y_hat` directly
3. **No softmax:** The paper does NOT apply softmax to the readout. The prediction is a raw linear projection.

**Implication for Phase 2:** The mode flag primarily affects:
- Target validation in `compute_errors()`: classification checks that `y` looks like one-hot (optional, nice-to-have)
- The `predict()` method itself does not change -- it always returns the raw readout
- No formula differences

**Implementation:**
```python
# In __init__:
if mode not in ("classification", "regression"):
    raise ValueError(f"mode must be 'classification' or 'regression', got '{mode}'")
self.mode = mode
```

**Confidence:** HIGH -- verified from paper. The word "regression" does not appear in the paper, but the formulas are universal (squared error). Classification is the paper's demonstrated use case; regression is a natural extension using identical math with continuous targets.

### Anti-Patterns to Avoid

- **DO NOT store latents as `nn.Parameter`:** Latents are NOT learnable parameters. They are working variables re-initialized per batch. Store as plain `Tensor` attributes (not registered as parameters), otherwise `model.parameters()` would include them and `model.state_dict()` would try to save them.
- **DO NOT compute errors per-layer as separate method calls:** The locked decision requires errors for all layers at once. Separate per-layer methods would require users to manually iterate and would not guarantee a consistent snapshot.
- **DO NOT apply softmax in the readout:** The paper uses raw linear projection `y_hat = W_out @ x^(L)` with no softmax. The supervised error is `y_hat - y` directly.
- **DO NOT use `nn.Linear`'s forward for error computation:** The layers use `PCNLayer.forward()` which returns `(prediction, preactivation)`. The preactivation is needed for the gain-modulated error. Do not use a separate `nn.Linear` or `F.linear` call that discards the preactivation.
- **DO NOT make the readout a public attribute:** Locked decision says readout is internal. Users access predictions via `predict()`.
- **DO NOT modify latents inside `compute_errors()`:** `compute_errors()` only computes errors from the current state. Latent updates happen in the training loop (Phase 3). Mixing computation and mutation in the same method would violate the snapshot principle.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Layer container with parameter tracking | Custom list + manual parameter registration | `nn.ModuleList` | Handles parameter registration, device transfer, state_dict automatically. A plain Python list would not register submodule parameters. |
| Readout linear projection | Manual `nn.Parameter` + `@ weight.T` | `nn.Linear(..., bias=False)` | Handles parameter registration, `extra_repr()`, and follows nn.Linear conventions for weight shape. |
| Device inference | Passing `device` parameter everywhere | `next(self.parameters()).device` | Standard PyTorch pattern for inferring the device from existing parameters. Works with `.to(device)` transfers. |
| Named structured returns | Custom class with manual `__init__` | `typing.NamedTuple` | Built-in, immutable, supports named access and tuple unpacking. |

**Key insight:** Phase 2 is pure standard PyTorch. Every component (ModuleList, Linear, NamedTuple, parameter registration) has a well-established solution.

## Common Pitfalls

### Pitfall 1: Dimension Semantics Confusion (Generative vs Feedforward)

**What goes wrong:** PCN layers are GENERATIVE (top-down), not feedforward (bottom-up). Layer `l` maps FROM `d_{l+1}` (above) TO `d_l` (below). This is the opposite of a standard feedforward network where layer `l` maps FROM `d_l` TO `d_{l+1}`. Confusing these directions means `PCNLayer(in_features, out_features)` gets the wrong dimensions, producing shape mismatches when computing errors.

**Why it happens:** PyTorch convention (`nn.Linear(in, out)`) is feedforward-centric. The PCN generative convention is the reverse.

**How to avoid:** When constructing from `dims = [d_0, d_1, ..., d_L]`:
- Layer `l` is `PCNLayer(in_features=d_{l+1}, out_features=d_l)`
- This means `PCNLayer.forward(x_above)` takes `x^(l+1)` of dim `d_{l+1}` and returns prediction of dim `d_l`
- The weight shape is `(d_l, d_{l+1})` = `(out_features, in_features)`

**Verification:** For `dims=[3072, 1000, 500, 10]`:
- Layer 0: `PCNLayer(1000, 3072)` -- weight shape (3072, 1000) -- maps x^(1) to predict x^(0)
- Layer 1: `PCNLayer(500, 1000)` -- weight shape (1000, 500) -- maps x^(2) to predict x^(1)
- Layer 2: `PCNLayer(10, 500)` -- weight shape (500, 10) -- maps x^(3) to predict x^(2)
- Readout: `nn.Linear(10, 10, bias=False)` -- weight shape (10, 10) -- maps x^(3) to y_hat

**Warning signs:** Shape mismatch errors when calling `layer(latent)` inside `compute_errors()`.

### Pitfall 2: Latents Not on Same Device as Model

**What goes wrong:** Latents are created via `torch.randn(batch_size, d_l)` which defaults to CPU. If the model has been moved to GPU via `model.to('cuda')`, the error computation will fail with a device mismatch error.

**Why it happens:** Plain tensors do not automatically follow `model.to(device)` -- only `nn.Parameter` objects do.

**How to avoid:** When initializing latents, infer the device from the model's parameters:
```python
device = next(self.parameters()).device
torch.randn(batch_size, d_l, device=device)
```

**Warning signs:** `RuntimeError: Expected all tensors to be on the same device`

### Pitfall 3: Storing Latents as nn.Parameter (Wrong)

**What goes wrong:** Latents appear in `model.parameters()` and `model.state_dict()`. If someone calls `optimizer.zero_grad()` or `optimizer.step()` (even incorrectly, in debugging), latents get treated as learnable parameters. State dict becomes bloated with ephemeral latent states.

**Why it happens:** It seems natural to register everything as `nn.Parameter` inside an `nn.Module`.

**How to avoid:** Store latents as a plain Python `list[Tensor]` attribute. They will NOT be in `model.parameters()` or `model.state_dict()`, which is correct because latents are transient per-batch state, not learned.

**Warning signs:** `model.state_dict()` contains entries like `latents.0`, `latents.1`, etc.

### Pitfall 4: Readout Weight Not Registered as Submodule

**What goes wrong:** If the readout is created as `self._readout_weight = nn.Parameter(...)` instead of `self._readout = nn.Linear(...)`, it will NOT appear in `model.named_modules()` and the parameter naming in `state_dict()` may be inconsistent. It will still be in `model.parameters()` though, since `nn.Parameter` attributes are always registered.

**Why it happens:** Trying to keep the readout "more internal" by avoiding `nn.Linear`.

**How to avoid:** Use `nn.Linear(..., bias=False)`. Even with the underscore-prefix convention for internal attributes, `nn.Module` properly registers `self._readout` as a submodule. The underscore signals "don't access directly" to users, but the parameter registration works normally.

**Warning signs:** Inconsistent state dict keys or missing readout weights after `model.load_state_dict()`.

### Pitfall 5: Incorrect W_out^T Projection in Batch Form

**What goes wrong:** The paper's per-sample formula is `eps^(L) = W_out^T @ eps_sup` using column vectors. In batch form with row vectors, this becomes `eps_sup @ W_out` (NOT `eps_sup @ W_out^T`). Getting this wrong means the supervised signal has the wrong shape or wrong values.

**Why it happens:** Confusing per-sample column-vector notation with batch row-vector convention.

**How to avoid:** Work through the shapes explicitly:
- `eps_sup` shape: `(B, d_out)`
- `W_out` shape (`nn.Linear.weight`): `(d_out, d_L)`
- `eps^(L) = eps_sup @ W_out` shape: `(B, d_out) @ (d_out, d_L) = (B, d_L)` -- correct!
- `eps^(L) = eps_sup @ W_out^T` shape: `(B, d_out) @ (d_L, d_out)` -- shape error if `d_out != d_L`!

**Warning signs:** Shape mismatch in the `compute_errors` supervised branch, or correct shapes but wrong values (if `d_out == d_L`).

### Pitfall 6: nn.ModuleList vs Plain List for Layers

**What goes wrong:** Using a plain Python `list` instead of `nn.ModuleList` means the layers' parameters are NOT registered with the parent module. `model.parameters()` returns nothing (or just the readout). `model.to('cuda')` does not move layer weights. `model.state_dict()` does not include layer weights.

**Why it happens:** It works fine in Python -- you can iterate over a list of modules and call their forward methods. The registration failure is silent until you try device transfer or serialization.

**How to avoid:** Always use `nn.ModuleList` for lists of `nn.Module` subclasses. The locked decision exposes `network.layers` -- this MUST be an `nn.ModuleList`.

**Warning signs:** `model.state_dict()` is suspiciously small. `model.to('cuda')` does not move layer weights.

## Code Examples

Verified patterns from official sources and the paper:

### Complete PredictiveCodingNetwork Skeleton

```python
# Source: Paper (arXiv:2506.06332v1) + locked decisions from CONTEXT.md
from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn

from pcn_torch.layers import PCNLayer


class PCNErrors(NamedTuple):
    """Results of error computation across all layers."""
    errors: list[Tensor]
    gm_errors: list[Tensor]
    supervised_error: Tensor | None
    top_error: Tensor | None


class PredictiveCodingNetwork(nn.Module):
    """A Predictive Coding Network managing the full layer hierarchy.

    Args:
        dims: Layer dimensions [d_0, d_1, ..., d_L]. Mutually exclusive with layers.
        layers: Pre-built PCNLayer instances. Mutually exclusive with dims.
        activation: Global activation function name (used only with dims). Default: "relu".
        output_dim: Output dimension for readout. Default: dims[-1] or inferred from layers.
        mode: Task mode, "classification" or "regression". Default: "classification".
    """

    def __init__(
        self,
        dims: Sequence[int] | None = None,
        layers: Sequence[PCNLayer] | None = None,
        *,
        activation: str = "relu",
        output_dim: int | None = None,
        mode: str = "classification",
    ) -> None:
        super().__init__()

        if mode not in ("classification", "regression"):
            raise ValueError(
                f"mode must be 'classification' or 'regression', got '{mode}'"
            )
        self.mode = mode

        if dims is not None and layers is not None:
            raise ValueError("Provide either 'dims' or 'layers', not both")

        if dims is not None:
            if len(dims) < 2:
                raise ValueError("dims must have at least 2 elements")
            self._dims = list(dims)
            built_layers: list[PCNLayer] = []
            for l in range(len(dims) - 1):
                # Layer l maps from d_{l+1} -> d_l (generative, top-down)
                built_layers.append(
                    PCNLayer(dims[l + 1], dims[l], activation=activation)
                )
            self.layers = nn.ModuleList(built_layers)

        elif layers is not None:
            if len(layers) < 1:
                raise ValueError("Must provide at least one layer")
            self.layers = nn.ModuleList(layers)
            # Reconstruct dims from layer weight shapes
            # Layer l: weight shape (d_l, d_{l+1}) = (out_features, in_features)
            self._dims = [layers[0].weight.shape[0]]  # d_0
            for layer in layers:
                self._dims.append(layer.weight.shape[1])  # d_{l+1}
            # Reverse the last element check:
            # Actually: each layer l has out_features=d_l, in_features=d_{l+1}
            # So dims = [d_0, d_1, ..., d_L] where:
            #   d_0 = layers[0].out_features = layers[0].weight.shape[0]
            #   d_1 = layers[0].in_features = layers[0].weight.shape[1]
            #       = layers[1].out_features = layers[1].weight.shape[0]  (if L>1)
            #   ...
            #   d_L = layers[-1].in_features = layers[-1].weight.shape[1]

        else:
            raise ValueError("Must provide either 'dims' or 'layers'")

        # Readout: maps from d_L to output
        d_L = self._dims[-1]
        effective_output_dim = output_dim if output_dim is not None else d_L
        self._readout = nn.Linear(d_L, effective_output_dim, bias=False)

        # Initialize public attributes for latents and errors
        self.latents: list[Tensor] = []
        self.errors: PCNErrors | None = None

    def init_latents(self, batch_size: int) -> None:
        """Initialize latent variables from N(0,1).

        Stores latents as self.latents -- one tensor per layer l=1..L.
        Layer 0 is the input (clamped, not stored here).

        Args:
            batch_size: Number of samples in the batch.
        """
        device = next(self.parameters()).device
        self.latents = [
            torch.randn(batch_size, self._dims[l], device=device)
            for l in range(1, len(self._dims))
        ]

    def predict(self) -> Tensor:
        """Get output prediction from current top latent state.

        Returns:
            Prediction tensor of shape (batch_size, output_dim).
        """
        return self._readout(self.latents[-1])

    def compute_errors(
        self, x: Tensor, y: Tensor | None = None
    ) -> PCNErrors:
        """Compute prediction errors and gain-modulated errors.

        Args:
            x: Input of shape (batch_size, d_0). Clamped as x^(0).
            y: Target of shape (batch_size, d_out). Optional.

        Returns:
            PCNErrors with errors, gm_errors, supervised_error, top_error.
        """
        # Full state: [x^(0)=input, x^(1)=latents[0], ..., x^(L)=latents[-1]]
        states = [x] + self.latents

        errors: list[Tensor] = []
        gm_errors: list[Tensor] = []

        for l, layer in enumerate(self.layers):
            # layer.forward(x_above) -> (prediction, preactivation)
            prediction, preactivation = layer(states[l + 1])
            eps = states[l] - prediction
            h = layer.activation_deriv(preactivation) * eps
            errors.append(eps)
            gm_errors.append(h)

        # Supervised error
        supervised_error: Tensor | None = None
        top_error: Tensor | None = None
        if y is not None:
            y_hat = self.predict()
            supervised_error = y_hat - y
            # eps^(L) = W_out^T @ eps_sup (per-sample)
            # Batch form: eps_sup @ W_out (row vectors)
            top_error = supervised_error @ self._readout.weight

        result = PCNErrors(errors, gm_errors, supervised_error, top_error)
        self.errors = result
        return result
```

### Layer Dimension Verification Example

```python
# Verify dimensions for dims=[3072, 1000, 500, 10]
net = PredictiveCodingNetwork(dims=[3072, 1000, 500, 10])

# 3 layers (L=3 latent layers, but only L-1=3 generative layers... wait)
# Actually: dims has 4 elements, so L=3 (top latent index).
# Number of PCNLayers = len(dims) - 1 = 3
assert len(net.layers) == 3

# Layer 0: maps d_1=1000 -> d_0=3072
assert net.layers[0].weight.shape == (3072, 1000)

# Layer 1: maps d_2=500 -> d_1=1000
assert net.layers[1].weight.shape == (1000, 500)

# Layer 2: maps d_3=10 -> d_2=500
assert net.layers[2].weight.shape == (500, 10)

# Readout: maps d_L=10 -> output=10
assert net._readout.weight.shape == (10, 10)
assert net._readout.bias is None

# Latent init
net.init_latents(batch_size=32)
assert len(net.latents) == 3  # x^(1), x^(2), x^(3)
assert net.latents[0].shape == (32, 1000)  # x^(1)
assert net.latents[1].shape == (32, 500)   # x^(2)
assert net.latents[2].shape == (32, 10)    # x^(3)

# Error computation
x = torch.randn(32, 3072)
y = torch.randn(32, 10)  # one-hot target
result = net.compute_errors(x, y)
assert len(result.errors) == 3         # eps^(0), eps^(1), eps^(2)
assert result.errors[0].shape == (32, 3072)  # eps^(0) at input dim
assert result.errors[1].shape == (32, 1000)  # eps^(1)
assert result.errors[2].shape == (32, 500)   # eps^(2)
assert result.supervised_error.shape == (32, 10)  # eps_sup
assert result.top_error.shape == (32, 10)          # eps^(L) = eps^(3)
```

### Gain Modulation Formula Verification

```python
# Source: Paper Section 3.2
# h^(l) = f'(a^(l)) * eps^(l)  (elementwise)
# where a^(l) is the preactivation from layer l's forward pass

# Example with ReLU:
# f'(a) = (a > 0).float()
# h = (a > 0).float() * eps

# This masks out error signal where preactivation was negative.
# Gain modulation scales the error by the activation derivative,
# producing the "gain-modulated error" used in weight updates.

layer = PCNLayer(10, 5, activation="relu")
x_above = torch.randn(4, 10)
prediction, preactivation = layer(x_above)
x_below = torch.randn(4, 5)  # simulated latent state

eps = x_below - prediction
h = layer.activation_deriv(preactivation) * eps

# h has same shape as eps
assert h.shape == eps.shape == (4, 5)
# Where preactivation <= 0, h should be 0 (ReLU derivative is 0)
mask = (preactivation > 0).float()
assert torch.allclose(h, mask * eps)
```

### Supervised Error Projection Verification

```python
# Source: Paper Section 4.2, Algorithm 2 line 14
# Per-sample: eps^(L) = W_out^T @ eps_sup
# Batch form: top_error = eps_sup @ W_out.weight

net = PredictiveCodingNetwork(dims=[100, 50, 10], output_dim=10)
net.init_latents(batch_size=8)

x = torch.randn(8, 100)
y = torch.zeros(8, 10)
y[:, 0] = 1.0  # one-hot target

result = net.compute_errors(x, y)

# Manual verification:
y_hat = net.latents[-1] @ net._readout.weight.T  # (8, 10)
expected_eps_sup = y_hat - y
expected_top = expected_eps_sup @ net._readout.weight  # (8, 10)

assert torch.allclose(result.supervised_error, expected_eps_sup)
assert torch.allclose(result.top_error, expected_top)
```

## Discretion Recommendations

For areas marked as Claude's discretion, here are research-backed recommendations:

### Return Type for `compute_errors`

**Recommend:** `NamedTuple` subclass named `PCNErrors` with four fields:
1. `errors: list[Tensor]` -- prediction errors eps^(l) for l=0..L-1
2. `gm_errors: list[Tensor]` -- gain-modulated errors h^(l) for l=0..L-1
3. `supervised_error: Tensor | None` -- eps_sup = y_hat - y
4. `top_error: Tensor | None` -- eps^(L) = eps_sup @ W_out

**Rationale:**
- NamedTuple provides both named access (`result.errors`) and positional unpacking (`errors, gm, eps_s, eps_L = result`)
- Immutable -- computation results should not be mutated
- Zero runtime overhead compared to a regular tuple
- Full mypy support
- Consistent with PyTorch patterns (e.g., `torch.return_types` uses similar structured returns)

### Gain Modulation Implementation

**Recommend:** Direct elementwise multiplication following the paper formula exactly.

Formula: `h^(l) = f'^(l)(a^(l)) * eps^(l)` where:
- `f'^(l)` is `layer.activation_deriv` (stored on each PCNLayer from Phase 1)
- `a^(l)` is the preactivation returned by `layer.forward()` (second element of the tuple)
- `eps^(l)` is the prediction error `x^(l) - x_hat^(l)`
- `*` is elementwise (Hadamard) product

Implementation:
```python
prediction, preactivation = layer(states[l + 1])
eps = states[l] - prediction
h = layer.activation_deriv(preactivation) * eps
```

No gain parameters, no scaling factors, no learned gains. Pure paper formula.

**Confidence:** HIGH -- formula is unambiguous in the paper.

### Supervised Error Formula

**Recommend:** Direct implementation of paper Section 4.2.

Formulas:
- `eps_sup = y_hat - y` where `y_hat = W_out @ x^(L)` (per-sample) or `y_hat = X^(L) @ W_out^T` (batch)
- `eps^(L) = W_out^T @ eps_sup` (per-sample) or `eps^(L) = eps_sup @ W_out` (batch)

In code:
```python
y_hat = self._readout(self.latents[-1])  # nn.Linear forward: x @ W^T
supervised_error = y_hat - y
top_error = supervised_error @ self._readout.weight  # (B, d_out) @ (d_out, d_L)
```

**Confidence:** HIGH -- verified from paper Algorithm 2.

### Exact Initialization Distribution for Latents

**Recommend:** Standard normal `N(0, 1)` via `torch.randn(batch_size, d_l)`.

The paper says "Initialize x^(l) <- small random values" (Algorithm 2, line 4). The PITFALLS.md research (Pitfall 18) documents that the reference implementation uses `torch.randn(batch_size, d)` with standard normal (scale=1.0). This is the standard Gaussian initialization.

Implementation:
```python
self.latents = [
    torch.randn(batch_size, self._dims[l], device=device)
    for l in range(1, len(self._dims))
]
```

**Confidence:** HIGH -- verified from reference implementation analysis in PITFALLS.md.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `typing.NamedTuple` import | `from typing import NamedTuple` (unchanged) | Python 3.6+ | NamedTuple class syntax is the standard. Not the `collections.namedtuple` factory. |
| `Optional[X]` | `X \| None` | Python 3.10+ | Use union syntax in type annotations. |
| `List[Tensor]` | `list[Tensor]` | Python 3.9+ | Lowercase builtins for type hints. |
| `Sequence` from `typing` | `Sequence` from `collections.abc` | Python 3.9+ | Use `collections.abc` for abstract base classes. |

## Open Questions

### 1. dims Reconstruction from Pre-built Layers

**What we know:** When the user provides pre-built `PCNLayer` instances (instead of `dims`), we need to reconstruct the `_dims` list to know dimensions for latent initialization. Each layer has `weight.shape = (out_features, in_features)` = `(d_l, d_{l+1})`.

**What's unclear:** Whether adjacent layers must have compatible dimensions (i.e., layer `l`'s `in_features` must equal layer `l+1`'s `out_features`). The paper implies this but it is not explicitly validated in the constructor.

**Recommendation:** Add validation in the constructor: for each pair of adjacent layers, check that `layers[l].weight.shape[1] == layers[l+1].weight.shape[0]` (the in_features of layer l must match the out_features of layer l+1). Raise `ValueError` with a clear message on mismatch. This catches user errors early.

### 2. output_dim Default When Using Pre-built Layers

**What we know:** When constructing from `dims`, the default `output_dim` is `dims[-1]`. When constructing from pre-built layers, the top latent dimension is `layers[-1].weight.shape[1]` (the in_features of the last layer).

**What's unclear:** Whether `output_dim` should be required when using pre-built layers (since there is no `dims[-1]` to default to).

**Recommendation:** Default to the top latent dimension (`layers[-1].weight.shape[1]`) when `output_dim` is not provided. This matches the paper's convention where `d_out = d_L`.

### 3. Thread Safety of Stored Latents/Errors

**What we know:** Storing `self.latents` and `self.errors` as mutable attributes on the network means they are not thread-safe. If two threads call `compute_errors()` concurrently on the same network instance, they will clobber each other's state.

**What's unclear:** Whether this matters for v1. Research libraries typically do not handle concurrent access.

**Recommendation:** Document that the network is not thread-safe (standard for PyTorch modules). Do not add locking or other concurrency mechanisms. This is consistent with PyTorch's own `nn.Module` which is also not thread-safe for forward pass state.

## Sources

### Primary (HIGH confidence)
- [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332) -- Paper by Stenlund (2025). Algorithm 2 (supervised learning), Section 3.1 (latent updates), Section 3.2 (weight updates), Section 4.2 (supervised error), Section 5.1 (architecture), Section 5.2 (training procedure)
- [PyTorch nn.ModuleList docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) -- Container for module registration
- [PyTorch nn.Linear docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) -- Weight shape convention (out_features, in_features)
- Phase 1 source code (`src/pcn_torch/layers.py`, `src/pcn_torch/activations.py`) -- Established patterns for PCNLayer constructor, forward return type, activation derivatives

### Secondary (MEDIUM confidence)
- `.planning/research/ARCHITECTURE.md` -- Data flow diagrams for inference, learning, and prediction phases. Dimension semantics and dependency graph.
- `.planning/research/PITFALLS.md` -- Pitfalls 1, 3, 5, 6 directly relevant to Phase 2 (synchronous updates, gain-modulated error, top-layer error signal, readout bias). Pitfall 18 confirms `torch.randn` initialization.
- `.planning/research/FEATURES.md` -- Competitive analysis confirming no existing PyTorch PCN library provides structured error returns.

### Tertiary (LOW confidence)
- None. All findings are supported by primary or secondary sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- No new libraries. Pure PyTorch patterns verified against official docs and Phase 1 codebase.
- Architecture: HIGH -- Based directly on paper's algorithms + locked decisions from CONTEXT.md. All formulas extracted and verified.
- Pitfalls: HIGH -- 6 Phase 2-specific pitfalls identified. All have concrete prevention strategies and verification patterns.
- Discretion recommendations: HIGH -- All four discretion areas have clear paper-backed recommendations. No ambiguity in the math.

**Research date:** 2026-02-20
**Valid until:** 2026-03-22 (30 days -- all components are stable PyTorch patterns, paper formulas do not change)
