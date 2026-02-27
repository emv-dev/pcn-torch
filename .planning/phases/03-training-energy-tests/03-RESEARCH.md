# Phase 3: Training + Energy + Tests - Research

**Researched:** 2026-02-24
**Domain:** PCN training loops (inference + learning), energy tracking, algorithmic correctness testing
**Confidence:** HIGH

## Summary

Phase 3 implements the core training functionality for `pcn-torch`: the two-phase training loop (inference then learning), energy tracking at multiple granularities, and structural correctness tests. The training loop operates entirely under `torch.no_grad()` using local Hebbian-like update rules -- no autograd, no optimizers. This is the most algorithmically dense phase, with 12 requirements spanning three subsystems (training, energy, testing).

The paper (arXiv:2506.06332v1) provides complete pseudocode for the vectorized batch training algorithm (Algorithm 3), including all tensor shapes and update formulas. The implementation translates directly from this pseudocode into PyTorch operations. The primary complexity lies in correctly separating the inference phase (latent updates with frozen weights) from the learning phase (weight updates with frozen latents), maintaining the synchronous update invariant, and tracking energy at multiple granularities without performance degradation.

New dependencies are minimal: `rich>=14.0` for the default console progress display, and `collections.deque` (stdlib) for the rolling energy window. The `dataclasses` module (stdlib) is used for `TrainConfig` and the training history object. All other operations use the existing Phase 1-2 codebase (`PredictiveCodingNetwork`, `PCNLayer`, `compute_errors`, `predict`).

**Primary recommendation:** Implement as two new modules: `src/pcn_torch/trainer.py` (training/testing loops, TrainConfig, TrainHistory, callback protocol) and `src/pcn_torch/energy.py` (energy computation utilities). Add `rich>=14.0` as a dependency. Tests go in `tests/test_trainer.py` and `tests/test_energy.py` using small synthetic data with seeded randomness.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Training API surface
- `train_pcn` returns a structured training history object (dataclass) with losses, energies, accuracies per epoch
- Accepts PyTorch DataLoader only (no raw tensor overload) -- caller handles batching/shuffling
- Separate `test_pcn` function for evaluation -- clear separation from training
- Hyperparameters passed via a `TrainConfig` dataclass -- groups T_infer, T_learn, learning rates, etc. for reproducibility

#### Inference and learning loop behavior
- Fixed T_infer steps with optional early stopping when energy converges (threshold-based)
- Two separate learning rates: `lr_infer` for latent updates, `lr_learn` for weight updates
- Explicit task mode in config: `TrainConfig(task='classification')` or `TrainConfig(task='regression')` -- no auto-detection
- Callback hook for progress/logging, defaulting to a rich.console display for training metrics
- MLflow and other integrations plug in through the callback -- not baked into the library
- Runtime guard on first iteration: check that no autograd graph is being built, warn if gradients detected

#### Energy tracking design
- Energy tracked during both inference and learning phases
- Per-step energy: rolling window of recent K values only (for real-time monitoring) -- not stored in full
- Per-batch and per-epoch energy: full history stored in the training history object
- All energy data lives inside the training history object (history.energy.per_batch, history.energy.per_epoch)
- Callback can access history mid-training for live monitoring

#### Test scope and correctness criteria
- Structural tests only -- no numerical convergence tests (convergence demonstrated by CIFAR-10 in Phase 4)
- Verify: gain-modulated error computation, correct W_out^T supervised error projection, correct shapes
- Synchronous latent updates are crucial for correctness but verifiable by code inspection -- no dedicated test needed
- Runtime no-autograd guard tested indirectly (training runs without warning)
- All tests use synthetic data only (small deterministic tensors, seeded random) -- fast, reproducible, no downloads

### Claude's Discretion
- TrainConfig default values for T_infer, T_learn, learning rates
- Training history dataclass field names and structure
- Rich console display formatting and verbosity
- Rolling window size K for per-step energy
- Early stopping convergence threshold default
- Specific synthetic test fixtures

### Deferred Ideas (OUT OF SCOPE)
- MLflow integration example -- demonstrate through callback in Phase 4's CIFAR-10 example
- Live GUI for energy monitoring -- future capability, callback hook provides the access point
- PyTorch Lightning wrapper -- not needed given PCN's non-backprop training, but could be revisited if demand exists
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >=2.0 | `torch.no_grad()`, `torch.amp.autocast`, tensor operations, `nn.Parameter.data` mutation | All training operations use PyTorch primitives |
| Python stdlib `dataclasses` | N/A | `TrainConfig`, `TrainHistory`, `EnergyHistory` dataclasses | Zero-dependency structured config and results |
| Python stdlib `collections.deque` | N/A | Rolling window for per-step energy (maxlen=K) | O(1) append/pop, built-in bounded size |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rich | >=14.0 | Default console progress display during training | Default callback; users who do not want rich get a plain-text fallback. |
| Python stdlib `typing.Protocol` | N/A | Typed callback interface for training hooks | Defines the callback contract without requiring inheritance |
| Python stdlib `warnings` | N/A | Runtime autograd guard warning | `warnings.warn()` when gradients detected on first iteration |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `rich` for progress | `tqdm` | tqdm is more widespread but less visually appealing. rich provides tables, colors, and live updating in a single library. The user decided on rich.console. |
| `dataclass` for TrainConfig | `NamedTuple` | NamedTuple is immutable (good for results, bad for config that users may want to tweak). Dataclass supports mutable defaults and `__post_init__` validation. |
| `Protocol` for callback | `ABC` | Protocol enables structural subtyping (duck typing) -- users do not need to inherit. ABC forces inheritance, which is heavier for a simple callback. |
| `deque(maxlen=K)` for rolling window | Circular buffer / numpy array | deque is stdlib, zero-allocation for bounded size, and handles the bounded window automatically. No need for numpy here. |

**Installation (new dependency):**

Add `rich>=14.0` to `[project.dependencies]` in pyproject.toml:
```toml
[project]
dependencies = [
    "numpy>=2.2.6",
    "torch>=2.0",
    "rich>=14.0",
]
```

**Rationale:** Add `rich>=14.0` to core dependencies directly (not optional). It is a lightweight dependency (pure Python, no C extensions) and the default out-of-box experience is a core value proposition. Falling back to plain text when rich is missing adds code complexity without significant benefit -- users who care about minimal dependencies can use the callback to suppress output entirely.

**Confidence:** HIGH -- rich 14.3.3 is current stable (Feb 2026), well-maintained by Textualize.

## Architecture Patterns

### Recommended Project Structure (Phase 3 additions)
```
src/pcn_torch/
    __init__.py         # Add train_pcn, test_pcn, TrainConfig exports
    _types.py           # Add callback type if needed
    activations.py      # Unchanged
    layers.py           # Unchanged
    network.py          # Unchanged
    energy.py           # NEW: compute_energy(), compute_energy_per_layer()
    trainer.py          # NEW: train_pcn(), test_pcn(), TrainConfig, TrainHistory
tests/
    test_energy.py      # NEW: energy computation tests
    test_trainer.py     # NEW: training loop structural tests
```

### Pattern 1: TrainConfig Dataclass

**What:** A dataclass grouping all training hyperparameters for reproducibility and serialization.

**When to use:** Passed to `train_pcn()` and `test_pcn()` as the single configuration object.

**Implementation:**
```python
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    """Configuration for PCN training.

    Attributes:
        task: Either 'classification' or 'regression'.
        T_infer: Number of inference steps per batch.
        T_learn: Number of learning steps per batch. Defaults to None,
            meaning T_learn = batch_size (paper recommendation).
        lr_infer: Learning rate for latent variable updates during inference.
        lr_learn: Learning rate for weight updates during learning.
        num_epochs: Number of training epochs.
        early_stop_threshold: If set, stop inference early when energy
            change falls below this threshold. None disables early stopping.
        energy_window_size: Rolling window size K for per-step energy monitoring.
        use_mixed_precision: Enable torch.amp.autocast during inference on CUDA.
        callback: Optional callback for progress/logging. Defaults to RichCallback.
    """
    task: str = "classification"
    T_infer: int = 50
    T_learn: int | None = None  # None means T_learn = batch_size
    lr_infer: float = 0.05
    lr_learn: float = 0.005
    num_epochs: int = 4
    early_stop_threshold: float | None = None
    energy_window_size: int = 10
    use_mixed_precision: bool = True
    callback: TrainCallback | None = None  # None means use default RichCallback

    def __post_init__(self) -> None:
        if self.task not in ("classification", "regression"):
            msg = f"task must be 'classification' or 'regression', got '{self.task}'"
            raise ValueError(msg)
        if self.T_infer < 1:
            msg = f"T_infer must be >= 1, got {self.T_infer}"
            raise ValueError(msg)
        if self.T_learn is not None and self.T_learn < 1:
            msg = f"T_learn must be >= 1 or None, got {self.T_learn}"
            raise ValueError(msg)
```

**Recommended defaults (Claude's discretion):**
| Parameter | Default | Paper Value | Rationale |
|-----------|---------|-------------|-----------|
| `T_infer` | 50 | 50 | Matches paper Section 5.2 |
| `T_learn` | None (=B) | 500 (=B) | Paper: "T_learn = B". None triggers dynamic batch_size lookup. |
| `lr_infer` | 0.05 | 0.05 | Paper Section 5.2 |
| `lr_learn` | 0.005 | 0.005 | Paper Section 5.2 |
| `num_epochs` | 4 | 4 | Paper: "Training for just 4 epochs" |
| `early_stop_threshold` | None | N/A | Disabled by default; the paper uses fixed T_infer=50 |
| `energy_window_size` | 10 | N/A | Keeps last 10 per-step energy values for monitoring |
| `use_mixed_precision` | True | Yes | Paper reference code uses `autocast(device_type='cuda')` during inference |

**Confidence:** HIGH -- all defaults directly from paper Section 5.2.

### Pattern 2: TrainHistory Dataclass

**What:** A mutable dataclass that accumulates training metrics as training progresses. Returned by `train_pcn()`.

**Implementation:**
```python
from dataclasses import dataclass, field

@dataclass
class EnergyHistory:
    """Energy tracking data at multiple granularities."""
    per_batch: list[float] = field(default_factory=list)    # One value per batch
    per_epoch: list[float] = field(default_factory=list)    # Mean energy per epoch

@dataclass
class TrainHistory:
    """Training results and metrics.

    Populated during training, returned by train_pcn().
    Accessible mid-training via callback.
    """
    energy: EnergyHistory = field(default_factory=EnergyHistory)
    train_loss: list[float] = field(default_factory=list)     # Per-epoch training loss
    train_accuracy: list[float] = field(default_factory=list)  # Per-epoch accuracy (classification)
    test_loss: list[float] = field(default_factory=list)       # Per-epoch test loss (if test_loader given)
    test_accuracy: list[float] = field(default_factory=list)   # Per-epoch test accuracy
    config: TrainConfig | None = None                          # The config used for this run
```

**Key design points:**
- `EnergyHistory` as a nested dataclass keeps energy data organized under `history.energy.per_batch` / `history.energy.per_epoch` as the user decided.
- Mutable (not frozen) because it is populated incrementally during training.
- `config` stores the TrainConfig used, enabling reproducibility.
- Per-step energy is NOT stored here (rolling window only, in the trainer's internal state).

**Confidence:** HIGH -- straightforward dataclass pattern.

### Pattern 3: Callback Base Class

**What:** A concrete base class for training hooks with no-op default implementations.

**Implementation:**
```python
class TrainCallback:
    """Base class for training callbacks. Override methods you need."""
    def on_train_start(self, config: TrainConfig, history: TrainHistory) -> None:
        pass
    def on_epoch_start(self, epoch: int, num_epochs: int) -> None:
        pass
    def on_batch_start(self, batch: int, num_batches: int) -> None:
        pass
    def on_inference_step(self, step: int, energy: float) -> None:
        pass
    def on_learning_step(self, step: int, energy: float) -> None:
        pass
    def on_batch_end(self, batch: int, batch_energy: float) -> None:
        pass
    def on_epoch_end(self, epoch: int, history: TrainHistory) -> None:
        pass
    def on_train_end(self, history: TrainHistory) -> None:
        pass
```

**Why concrete base class instead of Protocol:** Protocols in Python do not support default implementations. A concrete base class allows users to subclass and override only the methods they need, which is simpler. This matches the PyTorch Lightning callback pattern where `Callback` is a concrete class with no-op methods.

**Confidence:** HIGH -- standard pattern, follows Lightning conventions.

### Pattern 4: Training Loop (Core Algorithm)

**What:** The two-phase training loop implementing Algorithm 3 from the paper.

**Paper Algorithm 3 (vectorized batch form) -- key translation points:**

**Inference phase (T_infer steps):**
```python
with torch.no_grad():
    # Optional: autocast for mixed precision on CUDA
    for t in range(T_infer):
        # 1. Compute errors (snapshot) via model.compute_errors(x, y)
        errors = model.compute_errors(x_batch, y_batch)

        # 2. Build extended error list: errors + [eps_L]
        # eps_L = errors.top_error (already computed by compute_errors when y given)

        # 3. Latent gradient and update (l = 1..L)
        for l in range(len(model.latents)):
            # Paper: G_X^(l+1) = E^(l+1) - H^(l) @ W^(l)
            # model.latents[l] = paper's x^(l+1)
            pass  # Detailed in code examples below

        # 4. Update latents: X^(l) -= lr_infer * G_X^(l)
```

**Critical indexing detail:** The paper's error indexing differs from our implementation's:
- Paper errors: `E^(0)..E^(L-1)` from `compute_errors()`, plus `E^(L) = top_error`
- Our `errors.errors`: indices 0..L-1 correspond to layers 0..L-1
- Our `errors.top_error`: corresponds to `E^(L)`
- Latent gradient for latent `l` (1-indexed in paper, 0-indexed in `model.latents`):
  - If `l < L`: `grad = errors.errors[l] - errors.gm_errors[l-1] @ model.layers[l-1].weight`
  - If `l == L`: `grad = errors.top_error - errors.gm_errors[L-1] @ model.layers[L-1].weight`
  - Or unified: extend errors list with top_error, then `grad = extended[l] - gm_errors[l-1] @ weights[l-1]`

**Learning phase (T_learn steps):**
```python
with torch.no_grad():
    for t in range(T_learn):
        # 1. Recompute errors (weights changed!)
        errors = model.compute_errors(x_batch, y_batch)

        # 2. Weight gradient and update (l = 0..L-1)
        B = x_batch.shape[0]
        states = [x_batch, *model.latents]
        for l in range(len(model.layers)):
            layer = model.layers[l]
            # Paper: G_W^(l) = -(1/B) * H^(l).T @ X^(l+1)
            grad_W = -(errors.gm_errors[l].T @ states[l + 1]) / B
            layer.weight.data -= config.lr_learn * grad_W

        # 3. Readout weight update
        # Paper: G_W_out = (1/B) * E_sup.T @ X^(L)
        grad_W_out = (errors.supervised_error.T @ model.latents[-1]) / B
        model._readout.weight.data -= config.lr_learn * grad_W_out
```

**Weight update via `.data` attribute:** Use `layer.weight.data -= ...` (not `layer.weight -= ...`) to avoid triggering autograd. Inside `torch.no_grad()`, both work, but `.data` is explicit and makes the intent clear. The paper's reference code uses direct subtraction on the weight list references, which works because everything is inside `torch.no_grad()`.

**Confidence:** HIGH -- direct translation from Algorithm 3 in the paper.

### Pattern 5: Energy Computation

**What:** Compute the batch-averaged total energy (prediction error + supervised error).

**Paper formula:**
```
E_batch(t) = (1/B) * sum_b [ (1/2) * sum_l ||eps_b^(l)(t)||^2 + (1/2) * ||eps_b^sup(t)||^2 ]
```

**Implementation:**
```python
def compute_energy(errors: PCNErrors) -> float:
    """Compute batch-averaged total energy from error signals.

    Casts to float32 before squaring to avoid float16 overflow
    when running inside autocast regions.

    Args:
        errors: PCNErrors from model.compute_errors().

    Returns:
        Scalar energy value (float).
    """
    B = errors.errors[0].shape[0]
    energy = 0.0
    for e in errors.errors:
        ef = e.float()
        energy += 0.5 * (ef * ef).sum().item() / B
    if errors.supervised_error is not None:
        sf = errors.supervised_error.float()
        energy += 0.5 * (sf * sf).sum().item() / B
    return energy
```

**Key detail:** Use `.item()` to extract Python float from scalar tensor. Use `.float()` to cast to float32 before squaring to avoid float16 overflow under autocast (Pitfall 5 prevention).

**Energy per layer (diagnostic):**
```python
def compute_energy_per_layer(errors: PCNErrors) -> list[float]:
    """Compute per-layer energy contributions."""
    B = errors.errors[0].shape[0]
    per_layer = [
        0.5 * (e.float() * e.float()).sum().item() / B
        for e in errors.errors
    ]
    if errors.supervised_error is not None:
        per_layer.append(
            0.5 * (errors.supervised_error.float() * errors.supervised_error.float()).sum().item() / B
        )
    return per_layer
```

**Confidence:** HIGH -- directly from paper Section 5.2 energy formula.

### Pattern 6: Mixed Precision (autocast) During Inference

**What:** Use `torch.amp.autocast` during the inference phase on CUDA for performance.

**Paper reference:** The paper's `PCNLayer.forward()` wraps computation in `autocast(device_type='cuda')`. The training loop wraps the entire inference phase in `autocast`.

**Implementation:**
```python
from contextlib import nullcontext

def _get_autocast_context(config: TrainConfig, device: torch.device):
    """Get the appropriate autocast context manager."""
    if config.use_mixed_precision and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()

# In the inference loop:
with torch.no_grad():
    amp_ctx = _get_autocast_context(config, device)
    with amp_ctx:
        for t in range(T_infer):
            errors = model.compute_errors(x_batch, y_batch)
            # ... latent updates ...
```

**Important notes:**
- autocast is only used during inference, NOT during learning (matching paper reference code)
- Energy computation should cast to float32 before squaring: `e.float()` to avoid float16 overflow (Pitfall 5)
- On CPU, autocast is not used (no benefit for MLP operations)
- The updated API is `torch.amp.autocast("cuda")`, not the deprecated `torch.cuda.amp.autocast()`

**Confidence:** HIGH -- matches paper reference code and PyTorch 2.10 best practices.

### Pattern 7: Runtime Autograd Guard

**What:** On the first inference iteration, check that no autograd graph is being built. Warn if gradients are detected.

**Implementation:**
```python
import warnings

def _check_no_autograd(model: PredictiveCodingNetwork) -> None:
    """Check that no autograd graph is being built.

    Called once on the first inference step as a runtime guard.
    """
    if torch.is_grad_enabled():
        warnings.warn(
            "Gradient computation is enabled during PCN training. "
            "This will cause memory issues. Ensure training runs inside "
            "torch.no_grad().",
            RuntimeWarning,
            stacklevel=3,
        )
    for lat in model.latents:
        if lat.grad_fn is not None:
            warnings.warn(
                "Latent variable has an autograd graph attached (grad_fn is not None). "
                "PCN training should not build autograd graphs.",
                RuntimeWarning,
                stacklevel=3,
            )
            break
```

**When to call:** Once, at the start of the first inference step of the first batch. The user decided "just check for grads on the first loop" -- lightweight, not heavy validation.

**Detection methods available:**
- `torch.is_grad_enabled()`: Returns False inside `torch.no_grad()`. Catches missing context manager.
- `tensor.grad_fn is not None`: Catches tensors that were produced by autograd operations.
- `tensor.requires_grad`: Catches tensors marked for gradient tracking.

**Confidence:** HIGH -- uses standard PyTorch introspection APIs.

### Pattern 8: Early Stopping for Inference

**What:** Optionally stop inference early when energy converges (energy change falls below threshold).

**Implementation:**
```python
from collections import deque

# Inside inference loop:
energy_window: deque[float] = deque(maxlen=config.energy_window_size)

for t in range(config.T_infer):
    # ... compute errors, update latents ...
    step_energy = compute_energy(errors)
    energy_window.append(step_energy)

    # Early stopping check
    if (config.early_stop_threshold is not None
            and len(energy_window) >= 2
            and abs(energy_window[-1] - energy_window[-2]) < config.early_stop_threshold):
        break  # Energy converged
```

**Recommended defaults (Claude's discretion):**
- `early_stop_threshold`: None (disabled by default; paper uses fixed T_infer=50)
- `energy_window_size`: 10 (keeps last 10 values for the rolling window)

**Confidence:** HIGH -- standard early stopping pattern; paper explicitly mentions "sample-wise early stopping."

### Pattern 9: test_pcn Function

**What:** Evaluate a trained network on a test DataLoader.

**Paper reference (Section 5.3):** "To test the trained network, the weights are frozen. For each input-label pair from the test set, the latents are initialized randomly and the inference loop is executed, exactly as in the base algorithm. Once the latents are optimized, the prediction y_hat is read from the output layer."

**Implementation sketch:**
```python
def test_pcn(
    model: PredictiveCodingNetwork,
    dataloader: DataLoader,
    config: TrainConfig,
) -> dict[str, float]:
    """Evaluate a trained PCN on a test set.

    Runs inference only (no weight updates). Returns metrics.
    """
    model.eval()
    correct = 0
    total = 0
    total_energy = 0.0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            # ... run inference loop (same as training inference) ...
            # ... compute predictions ...
            y_hat = model.predict()
            if config.task == "classification":
                predicted = y_hat.argmax(dim=1)
                correct += (predicted == y_batch).sum().item()
            total += x_batch.shape[0]
            total_energy += batch_energy * x_batch.shape[0]

    return {
        "accuracy": correct / total if config.task == "classification" else 0.0,
        "energy": total_energy / total,
    }
```

**Key point:** At test time, the supervised error signal IS used (y is known) to guide inference. The paper states: "the latents are initialized randomly and the inference loop is executed, exactly as in the base algorithm." This means test inference uses the same supervised error projection as training.

**Confidence:** HIGH -- directly from paper Section 5.3.

### Anti-Patterns to Avoid

- **DO NOT use `torch.optim` for weight updates:** PCN uses explicit local Hebbian-like rules `W -= lr * grad`. Using SGD/Adam would make it a standard neural network.
- **DO NOT store full per-step energy history:** The user decided on a rolling window for per-step energy. Storing all T_infer + T_learn steps per batch would consume excessive memory (e.g., 550 floats x 100 batches x 4 epochs = 220K values).
- **DO NOT interleave error computation and latent updates:** Compute ALL errors first (snapshot), THEN update ALL latents. This is the synchronous update invariant (Pitfall 1).
- **DO NOT apply batch averaging (1/B) to latent gradients:** The 1/B factor applies ONLY to weight gradients, NOT to latent inference gradients. This is a critical correctness requirement (Pitfall 2).
- **DO NOT use `autocast` during the learning phase:** The paper's reference code only uses autocast during inference. Learning runs in full precision.
- **DO NOT compute energy in float16:** Inside autocast regions, cast error tensors to float32 before squaring for energy computation to avoid float16 overflow.
- **DO NOT make the `task` field on TrainConfig auto-detect:** The user explicitly decided on `TrainConfig(task='classification')` -- no auto-detection.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rolling window buffer | Custom circular buffer class | `collections.deque(maxlen=K)` | Stdlib, O(1) bounded append, automatic eviction of oldest element |
| Console progress display | Custom print statements | `rich.progress.Progress` with `rich.live.Live` | Tables, progress bars, colors, live updating -- all in one library. Matches user decision for rich.console |
| Training config validation | Manual if/else chains | `dataclass.__post_init__` | Standard Python pattern for post-initialization validation |
| Mixed precision context | Manual dtype casting | `torch.amp.autocast` with `contextlib.nullcontext` fallback | Handles device-appropriate precision automatically, no-op on CPU |
| Device inference | Passing device parameter everywhere | `next(model.parameters()).device` | Standard PyTorch pattern already established in Phase 2 |

**Key insight:** Phase 3 introduces no genuinely novel programming patterns. The complexity is algorithmic (getting the PCN update rules right), not architectural (everything maps to standard PyTorch + stdlib).

## Common Pitfalls

### Pitfall 1: Asynchronous Latent Updates (CRITICAL)

**What goes wrong:** Computing error for layer l, updating latent l, then computing error for layer l+1 using the already-updated latent l. This makes each latent update depend on partially-updated state from the same step.

**Why it happens:** It is the natural implementation -- iterate through layers, compute, and update in one pass.

**How to avoid:** The existing `model.compute_errors()` already computes ALL errors from a consistent snapshot. The training loop must call `compute_errors()` ONCE, then use those stored results to update ALL latents. Do NOT interleave `compute_errors()` calls with latent updates within a single step.

**Warning signs:** Energy oscillates or fails to decrease even with very small lr_infer.

### Pitfall 2: Missing 1/B in Weight Gradients (CRITICAL)

**What goes wrong:** Weight gradient `G_W^(l) = -(1/B) * H^(l).T @ X^(l+1)` requires the batch averaging factor. Without it, effective learning rate scales with batch size.

**Why it happens:** The matrix product `H.T @ X` naturally sums over the batch dimension. The 1/B must be applied explicitly.

**How to avoid:** Always divide by `B = x_batch.shape[0]` in the weight gradient computation. Apply 1/B ONLY in weight updates, NOT in latent gradient computation.

**Warning signs:** Training works at one batch size but diverges at another without adjusting lr_learn.

### Pitfall 3: T_learn Not Scaling with Batch Size

**What goes wrong:** Using a fixed T_learn (e.g., 10) regardless of batch size breaks the inference-to-learning ratio the paper relies on.

**Why it happens:** T_learn=B is a non-obvious default.

**How to avoid:** When `config.T_learn is None`, set effective T_learn to the current batch size B. Document why.

**Warning signs:** Performance varies with batch size; energy does not decrease during learning.

### Pitfall 4: In-Place Weight Update Not Modifying Model Parameters

**What goes wrong:** If you accidentally create a copy of the weight tensor (e.g., `w = layer.weight; w -= lr * grad`), the original `nn.Parameter` is untouched.

**Why it happens:** Python variable rebinding vs. in-place mutation confusion.

**How to avoid:** Use `layer.weight.data -= lr * grad` which modifies the underlying tensor data in-place. Or build a `weights` list of direct references as the paper's code does: `weights = [layer.W for layer in model.layers] + [model.readout.weight]` and then `weights[l] -= lr * grad` inside `no_grad`.

**Warning signs:** Weights unchanged after training; energy decreases during inference but returns to same level at next batch.

### Pitfall 5: autocast Energy Overflow in Float16

**What goes wrong:** Inside an autocast region, error tensors may be float16. Squaring float16 values can overflow (max value is roughly 65504). Large prediction errors in early training easily exceed this.

**Why it happens:** autocast transparently casts matrix multiplications to float16; the error tensors inherit this dtype.

**How to avoid:** Cast to float32 before energy computation: `(e.float() * e.float()).sum()`.

**Warning signs:** NaN or Inf in energy values during early training on GPU.

### Pitfall 6: Latent Update Indexing Off-By-One

**What goes wrong:** The paper uses 1-indexed latent layers (l=1..L) but Python uses 0-indexed lists. `model.latents[0]` is `x^(1)` in the paper, `model.latents[-1]` is `x^(L)`.

**Why it happens:** Direct translation from paper pseudocode without accounting for indexing conventions.

**How to avoid:** Build an extended errors list `[errors[0], ..., errors[L-1], top_error]` so that `extended[l]` corresponds to the paper's `E^(l)` for l=0..L. Then the gradient for `model.latents[l]` (which is paper's `x^(l+1)`) is `extended[l+1] - gm_errors[l] @ weights[l]`.

**Or simpler:** Map paper layer index to code index explicitly with comments.

**Warning signs:** Shape mismatch errors; latent gradients applied to wrong layers.

## Code Examples

### Complete train_pcn Skeleton

```python
# Source: Paper Algorithm 3 (vectorized batch form) + CONTEXT.md decisions
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings

import torch
from torch import Tensor

from pcn_torch.energy import compute_energy
from pcn_torch.network import PredictiveCodingNetwork

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def train_pcn(
    model: PredictiveCodingNetwork,
    dataloader: DataLoader,
    config: TrainConfig,
) -> TrainHistory:
    """Train a PredictiveCodingNetwork using local Hebbian-like update rules.

    All operations run under torch.no_grad(). No autograd graph is built.

    Args:
        model: The PredictiveCodingNetwork to train.
        dataloader: PyTorch DataLoader providing (x_batch, y_batch) tuples.
        config: Training configuration (hyperparameters, callbacks, etc.).

    Returns:
        TrainHistory with losses, energies, and accuracies per epoch.
    """
    device = next(model.parameters()).device
    history = TrainHistory(config=config)
    callback = config.callback or RichCallback()
    first_batch = True

    callback.on_train_start(config, history)

    for epoch in range(config.num_epochs):
        callback.on_epoch_start(epoch, config.num_epochs)
        epoch_energy = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            B = x_batch.shape[0]
            x_batch = x_batch.view(B, -1).to(device)

            # Encode targets
            if config.task == "classification":
                # y_batch assumed to be class indices -> one-hot
                num_classes = model._readout.weight.shape[0]
                y_batch = torch.nn.functional.one_hot(
                    y_batch.long(), num_classes=num_classes
                ).float().to(device)
            else:
                y_batch = y_batch.to(device)

            # Initialize latents
            model.init_latents(B)

            # Build weight reference list for in-place updates
            weights: list[Tensor] = [
                layer.weight for layer in model.layers  # type: ignore[union-attr]
            ] + [model._readout.weight]

            T_learn = config.T_learn if config.T_learn is not None else B

            # === INFERENCE PHASE ===
            amp_ctx = (
                torch.amp.autocast(device_type="cuda")
                if config.use_mixed_precision and device.type == "cuda"
                else nullcontext()
            )

            with torch.no_grad(), amp_ctx:
                energy_window: deque[float] = deque(maxlen=config.energy_window_size)

                for t in range(config.T_infer):
                    # Runtime guard on first iteration
                    if first_batch and t == 0:
                        _check_no_autograd(model)
                        first_batch = False

                    # Compute all errors from consistent snapshot
                    errors = model.compute_errors(x_batch, y_batch)

                    # Compute energy for this step
                    step_energy = compute_energy(errors)
                    energy_window.append(step_energy)
                    callback.on_inference_step(t, step_energy)

                    # Latent gradients and updates
                    # Build extended errors: [E^(0), ..., E^(L-1), E^(L)=top_error]
                    extended_errors = list(errors.errors)
                    if errors.top_error is not None:
                        extended_errors.append(errors.top_error)
                    else:
                        # Unsupervised: E^(L) = 0
                        extended_errors.append(
                            torch.zeros_like(model.latents[-1])
                        )

                    for l in range(len(model.latents)):
                        # Paper: G_X^(l+1) = E^(l+1) - H^(l) @ W^(l)
                        # model.latents[l] = paper's x^(l+1)
                        grad_x = extended_errors[l + 1] - (
                            errors.gm_errors[l] @ weights[l]
                        )
                        model.latents[l] -= config.lr_infer * grad_x

                    # Early stopping check
                    if (config.early_stop_threshold is not None
                            and len(energy_window) >= 2
                            and abs(energy_window[-1] - energy_window[-2])
                                < config.early_stop_threshold):
                        break

            # === LEARNING PHASE ===
            states = [x_batch, *model.latents]

            with torch.no_grad():
                for t in range(T_learn):
                    # Recompute errors (weights changed!)
                    errors = model.compute_errors(x_batch, y_batch)

                    step_energy = compute_energy(errors)
                    callback.on_learning_step(t, step_energy)

                    # Weight gradients and updates (l = 0..L-1)
                    for l in range(len(model.layers)):
                        # Paper: G_W^(l) = -(1/B) * H^(l).T @ X^(l+1)
                        grad_W = -(errors.gm_errors[l].T @ states[l + 1]) / B
                        weights[l].data -= config.lr_learn * grad_W

                    # Readout weight update
                    # Paper: G_W_out = (1/B) * E_sup.T @ X^(L)
                    if errors.supervised_error is not None:
                        grad_W_out = (
                            errors.supervised_error.T @ model.latents[-1]
                        ) / B
                        weights[-1].data -= config.lr_learn * grad_W_out

            # Record batch energy
            batch_energy = compute_energy(model.compute_errors(x_batch, y_batch))
            history.energy.per_batch.append(batch_energy)
            epoch_energy += batch_energy * B

            # Track accuracy for classification
            if config.task == "classification":
                y_hat = model.predict()
                predicted = y_hat.argmax(dim=1)
                targets = y_batch.argmax(dim=1) if y_batch.dim() > 1 else y_batch
                epoch_correct += (predicted == targets).sum().item()
            epoch_total += B

            callback.on_batch_end(batch_idx, batch_energy)

        # Epoch summary
        epoch_mean_energy = epoch_energy / epoch_total if epoch_total > 0 else 0.0
        history.energy.per_epoch.append(epoch_mean_energy)
        history.train_loss.append(epoch_mean_energy)
        if config.task == "classification" and epoch_total > 0:
            history.train_accuracy.append(epoch_correct / epoch_total)

        callback.on_epoch_end(epoch, history)

    callback.on_train_end(history)
    return history
```

### Latent Gradient Computation (Detailed Indexing)

```python
# Source: Paper Algorithm 3, lines 14-16
# Paper notation (1-indexed): x^(l) for l=1..L
# Code notation (0-indexed): model.latents[i] for i=0..L-1
#   model.latents[0] = paper's x^(1)
#   model.latents[L-1] = paper's x^(L)
#
# Paper formula:
#   G_X^(l) = E^(l) - H^(l-1) @ W^(l-1)   for l=1..L
#   x^(l) -= lr_infer * G_X^(l)
#
# Where E^(L) = W_out^T @ E_sup (supervised) or 0 (unsupervised)
#
# Translation to code indices:
#   For model.latents[i] (= paper's x^(i+1)):
#     grad = extended_errors[i+1] - gm_errors[i] @ weights[i]
#     model.latents[i] -= lr_infer * grad

# Example for dims=[3072, 1000, 500, 10]:
#   model.latents has 3 entries: x^(1), x^(2), x^(3)
#   errors.errors has 3 entries: E^(0), E^(1), E^(2)
#   errors.top_error: E^(3) = E_sup @ W_out
#   extended_errors = [E^(0), E^(1), E^(2), E^(3)]
#
#   latents[0] (x^(1)): grad = extended[1] - gm[0] @ W[0]
#   latents[1] (x^(2)): grad = extended[2] - gm[1] @ W[1]
#   latents[2] (x^(3)): grad = extended[3] - gm[2] @ W[2]
```

### Synthetic Test Fixture

```python
# Source: Test decisions from CONTEXT.md
import torch
from torch.utils.data import DataLoader, TensorDataset

def make_synthetic_dataloader(
    n_samples: int = 32,
    input_dim: int = 20,
    num_classes: int = 5,
    batch_size: int = 8,
    seed: int = 42,
) -> DataLoader:
    """Create a small deterministic DataLoader for testing."""
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
```

### Test: Gain-Modulated Error Computation (TST-02)

```python
def test_gain_modulated_error_formula():
    """Verify h^(l) = f'(a^(l)) * eps^(l) uses PREACTIVATION, not latent."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    net.init_latents(4)
    x = torch.randn(4, 20)

    errors = net.compute_errors(x)

    # Manual: layer 0, preactivation a = x^(1) @ W^(0).T
    states = [x, *net.latents]
    layer0 = net.layers[0]
    preact = states[1] @ layer0.weight.T
    eps = states[0] - layer0.activation_fn(preact)
    expected_h = layer0.activation_deriv(preact) * eps

    assert torch.allclose(errors.gm_errors[0], expected_h)
    # Verify it is NOT f'(x^(0)) * eps (wrong tensor)
    wrong_h = layer0.activation_deriv(states[0]) * eps
    # These should differ (unless by coincidence)
    # The key test is that the correct formula matches
```

### Test: No Autograd Graph During Training (TST-03)

```python
def test_no_autograd_graph_during_training():
    """Verify training runs without building autograd graphs."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = make_synthetic_dataloader(n_samples=16, input_dim=20, num_classes=5, batch_size=8)
    config = TrainConfig(T_infer=3, T_learn=2, num_epochs=1, use_mixed_precision=False)

    # Should complete without warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        history = train_pcn(net, dl, config)
        # No RuntimeWarning about autograd
        autograd_warnings = [x for x in w if "autograd" in str(x.message).lower()]
        assert len(autograd_warnings) == 0

    # Verify no latent has grad_fn
    for lat in net.latents:
        assert lat.grad_fn is None
        assert not lat.requires_grad
```

### Test: Correct W_out^T Supervised Error Projection (TST-04)

```python
def test_top_error_wout_projection():
    """Verify top_error = supervised_error @ W_out (batch W_out^T projection)."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    net.init_latents(4)
    x = torch.randn(4, 20)
    y = torch.randn(4, 5)

    errors = net.compute_errors(x, y)

    # Manual computation
    y_hat = net.latents[-1] @ net._readout.weight.T
    expected_sup = y_hat - y
    expected_top = expected_sup @ net._readout.weight

    assert errors.top_error is not None
    assert torch.allclose(errors.top_error, expected_top)
    # Verify shape: (B, d_L) not (B, d_out)
    assert errors.top_error.shape == (4, 5)  # d_L = 5 for this network
```

## Discretion Recommendations

### TrainConfig Default Values

**Recommend:** Match paper hyperparameters exactly:
- `T_infer = 50`, `T_learn = None (=B)`, `lr_infer = 0.05`, `lr_learn = 0.005`, `num_epochs = 4`

**Rationale:** The paper's hyperparameters produce 99.92% on CIFAR-10. Defaulting to these values means `TrainConfig()` with no arguments works for the paper's benchmark.

### Training History Field Names

**Recommend:**
- `history.energy.per_batch` -- list of floats, one per batch across all epochs
- `history.energy.per_epoch` -- list of floats, one per epoch (mean over batches)
- `history.train_loss` -- same as per_epoch energy (alias for familiarity)
- `history.train_accuracy` -- list of floats, per-epoch accuracy (classification only)
- `history.config` -- the TrainConfig used

### Rich Console Display

**Recommend:** Use `rich.progress.Progress` with a custom layout:
- Progress bar for epoch and batch
- Live-updating table showing: epoch, batch, energy, accuracy
- Update frequency: once per batch (not per step, which would be too noisy)
- Verbosity: print epoch summary at end of each epoch

### Rolling Window Size K

**Recommend:** K = 10 (energy_window_size). This keeps enough history for convergence detection while being negligible memory (10 floats).

### Early Stopping Convergence Threshold

**Recommend:** Default to None (disabled). When enabled, a reasonable starting value is 1e-5 (absolute energy change). This is conservative enough to avoid premature stopping while catching genuine convergence.

### Specific Synthetic Test Fixtures

**Recommend:**
- Network: `dims=[20, 10, 5]` (small, fast, exercises all code paths)
- Batch size: 8 (small enough for fast tests)
- Samples: 16-32 (2-4 batches per epoch)
- Seed: 42 (deterministic, reproducible)
- T_infer: 3, T_learn: 2 (minimal steps -- structural tests, not convergence tests)
- Epochs: 1 (structural verification, not multi-epoch behavior)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.cuda.amp.autocast()` | `torch.amp.autocast("cuda")` | PyTorch 2.4+ | Unified API for all device types |
| `Optional[X]` | `X \| None` | Python 3.10+ | Union syntax preferred |
| `@dataclass(frozen=True)` for config | `@dataclass` with `__post_init__` validation | N/A | Mutable config is more practical (users may want to tweak before passing) |
| `typing.Protocol` for callbacks | Concrete base class with no-op methods | N/A | Simpler for users who just want to override a few hooks |
| `torch.no_grad()` alone | `torch.no_grad()` + `torch.inference_mode()` | PyTorch 1.9+ | inference_mode is more restrictive but faster; however, PCN latents need to be mutable after inference, so `no_grad()` is correct here |

**Note on `torch.inference_mode()`:** While faster than `torch.no_grad()`, inference mode makes tensors immutable after creation. Since PCN latent variables are updated in-place during inference, `torch.no_grad()` is the correct choice. Do NOT use `torch.inference_mode()` for the training loop.

## Open Questions

### 1. Latent Update Inside vs. Outside autocast

**What we know:** The paper's reference code uses `autocast` for the inference phase. The error computation (inside `compute_errors`) runs under autocast. But the latent update `model.latents[l] -= lr * grad` happens right after, still inside the autocast context.

**What's unclear:** Should the latent subtraction happen inside or outside autocast? Inside autocast, the subtraction may produce a float16 result, which could accumulate numerical error over 50 inference steps.

**Recommendation:** Keep the update inside autocast to match the reference code. The latent values are re-initialized each batch, so 50 steps of float16 accumulation is unlikely to cause issues. If numerical problems arise, the user can disable mixed precision via `config.use_mixed_precision = False`. LOW priority concern.

### 2. Weight Reference List vs. Direct Attribute Access

**What we know:** The paper's code builds `weights = [layer.W for layer in model.layers] + [model.readout.weight]` and updates via `weights[l] -= lr * grad`. Our codebase uses `layer.weight` (not `layer.W`) and `model._readout.weight`.

**What's unclear:** Whether the list-of-references pattern is better than direct attribute access in the loop.

**Recommendation:** Use the list-of-references pattern matching the paper. It simplifies the update loop by treating all weights uniformly. The references are just Python variable bindings to the same underlying `nn.Parameter` objects, so in-place updates (`weights[l].data -= ...`) modify the model's actual parameters.

### 3. One-Hot Encoding Responsibility

**What we know:** The user decided "Accepts PyTorch DataLoader only -- caller handles batching/shuffling." The paper's training code does `F.one_hot(y_batch, ...)` inside the training loop.

**What's unclear:** Should `train_pcn` handle one-hot encoding internally, or should the user provide pre-encoded targets?

**Recommendation:** Handle one-hot encoding inside `train_pcn` for classification tasks. Standard DataLoaders from torchvision produce integer class indices, not one-hot vectors. Requiring users to pre-encode would add friction. Check: if `config.task == 'classification'` and targets are 1D integer tensors, apply `F.one_hot`. If targets are already 2D float tensors, assume they are pre-encoded.

## Sources

### Primary (HIGH confidence)
- [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332) -- Stenlund (2025). Algorithm 3 (vectorized batch form), Section 5.2 (hyperparameters), Section 5.3 (testing). Full paper LaTeX source available locally at `arXiv-2506.06332v1/pcn_intro.tex`.
- [PyTorch 2.10 torch.no_grad docs](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) -- Context manager for disabling gradient computation.
- [PyTorch 2.10 torch.amp.autocast docs](https://docs.pytorch.org/docs/stable/amp.html) -- Mixed precision API. Confirmed: use `torch.amp.autocast("cuda")`, not the deprecated `torch.cuda.amp.autocast()`.
- [PyTorch 2.10 autograd docs](https://docs.pytorch.org/docs/stable/autograd.html) -- `torch.is_grad_enabled()`, `tensor.grad_fn` for runtime autograd detection.
- Existing codebase: `src/pcn_torch/network.py` (PredictiveCodingNetwork, PCNErrors, compute_errors), `src/pcn_torch/layers.py` (PCNLayer), verified shapes and indexing.
- `.planning/research/ARCHITECTURE.md` -- Data flow diagrams for inference, learning, and prediction phases.
- `.planning/research/PITFALLS.md` -- Pitfalls 1, 2, 4, 5, 15, 16, 17 directly relevant to Phase 3.

### Secondary (MEDIUM confidence)
- [rich 14.3.3 on PyPI](https://pypi.org/project/rich/) -- Latest stable version (Feb 2026). Progress display, Live, Table components.
- [Rich Progress docs](https://rich.readthedocs.io/en/latest/progress.html) -- Progress bar and live display API.
- [PyTorch Lightning Callback pattern](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) -- Callback design inspiration (concrete class with no-op methods).
- [PEP 544 Protocols](https://peps.python.org/pep-0544/) -- Structural subtyping for callback typing.
- [Python dataclasses docs](https://docs.python.org/3/library/dataclasses.html) -- `@dataclass`, `field(default_factory=...)`, `__post_init__`.

### Tertiary (LOW confidence)
- None. All findings supported by primary or secondary sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- No new major libraries. rich is well-established. All patterns use stdlib + PyTorch.
- Architecture: HIGH -- Algorithm 3 from paper provides complete pseudocode. Direct translation to existing codebase.
- Pitfalls: HIGH -- 6 pitfalls identified, all with concrete prevention. Most critical ones (Pitfalls 1, 2, 16) are well-understood from prior research.
- Discretion recommendations: HIGH -- All defaults derived from paper Section 5.2. No ambiguity in the hyperparameters or energy formula.

**Research date:** 2026-02-24
**Valid until:** 2026-03-26 (30 days -- stable patterns, paper algorithms do not change)
