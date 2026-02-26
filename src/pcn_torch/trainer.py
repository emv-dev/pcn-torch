"""Training loops, configuration, history, and callbacks for PCN.

Implements Algorithm 3 from arXiv:2506.06332v1: two-phase training
with inference (latent updates) and learning (weight updates), both
running entirely under torch.no_grad() with no autograd graph.
"""

from __future__ import annotations

import warnings
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from pcn_torch.energy import compute_energy

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from pcn_torch.network import PredictiveCodingNetwork


# ---------------------------------------------------------------------------
# Callback base class
# ---------------------------------------------------------------------------


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

    def on_lr_reduced(self, old_lr: float, new_lr: float) -> None:
        pass

    def on_train_end(self, history: TrainHistory) -> None:
        pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EnergyHistory:
    """Energy tracking data at multiple granularities."""

    per_batch: list[float] = field(default_factory=list)
    per_epoch: list[float] = field(default_factory=list)


@dataclass
class TrainHistory:
    """Training results and metrics.

    Populated during training, returned by train_pcn().
    Accessible mid-training via callback.
    """

    energy: EnergyHistory = field(default_factory=EnergyHistory)
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_accuracy: list[float] = field(default_factory=list)
    lr_learn_per_epoch: list[float] = field(default_factory=list)
    config: TrainConfig | None = None
    # Running accuracy within current epoch (updated per batch for callbacks)
    _running_accuracy: float = field(default=0.0, repr=False)


@dataclass
class TrainConfig:
    """Configuration for PCN training.

    Attributes:
        task: Either 'classification' or 'regression'.
        T_infer: Number of inference steps per batch during training
            (supervised inference with labels clamped).
        T_infer_test: Number of inference steps for test/evaluation
            (unsupervised inference without labels). Defaults to None,
            meaning ``T_infer * 10``. Unsupervised inference needs more
            steps because the bottom-up signal alone converges slower
            than when the supervised signal also drives the latents.
        T_learn: Number of learning steps per batch. Defaults to None,
            meaning T_learn = batch_size (paper recommendation).
        lr_infer: Learning rate for latent variable updates during inference.
        lr_learn: Learning rate for weight updates during learning.
        num_epochs: Number of training epochs.
        early_stop_threshold: If set, stop inference early when energy
            change falls below this threshold. None disables early stopping.
        energy_window_size: Rolling window size K for per-step energy.
        use_mixed_precision: Enable torch.amp.autocast during inference
            on CUDA devices.
        lr_learn_steps: Manual LR schedule as a list of (epoch, batch, lr)
            tuples. At the start of the given epoch and batch, lr_learn
            switches to ``lr``. Epoch and batch are 0-indexed.
            Example: ``[(0, 0, 0.05), (0, 20, 0.005), (1, 0, 0.001)]``
            starts at 0.05, drops to 0.005 at batch 20 of epoch 0, then
            drops to 0.001 at the start of epoch 1.
            If None, uses constant ``lr_learn`` throughout training.
        callback: Optional callback for progress/logging. Defaults to
            RichCallback.
    """

    task: str = "classification"
    T_infer: int = 50
    T_infer_test: int | None = None
    T_learn: int | None = None
    lr_infer: float = 0.05
    lr_learn: float = 0.005
    num_epochs: int = 4
    early_stop_threshold: float | None = None
    energy_window_size: int = 10
    use_mixed_precision: bool = True
    lr_learn_steps: list[tuple[int, int, float]] | None = None
    callback: TrainCallback | None = None

    def __post_init__(self) -> None:
        if self.task not in ("classification", "regression"):
            msg = f"task must be 'classification' or 'regression', got '{self.task}'"
            raise ValueError(msg)
        if self.T_infer < 1:
            msg = f"T_infer must be >= 1, got {self.T_infer}"
            raise ValueError(msg)
        if self.T_infer_test is not None and self.T_infer_test < 1:
            msg = f"T_infer_test must be >= 1 or None, got {self.T_infer_test}"
            raise ValueError(msg)
        if self.T_learn is not None and self.T_learn < 1:
            msg = f"T_learn must be >= 1 or None, got {self.T_learn}"
            raise ValueError(msg)
        if self.lr_learn_steps is not None:
            if len(self.lr_learn_steps) == 0:
                msg = "lr_learn_steps must be None or a non-empty list"
                raise ValueError(msg)
            for i, (ep, batch, lr) in enumerate(self.lr_learn_steps):
                if ep < 0:
                    msg = f"lr_learn_steps[{i}] epoch must be >= 0, got {ep}"
                    raise ValueError(msg)
                if batch < 0:
                    msg = f"lr_learn_steps[{i}] batch must be >= 0, got {batch}"
                    raise ValueError(msg)
                if lr <= 0.0:
                    msg = f"lr_learn_steps[{i}] lr must be > 0, got {lr}"
                    raise ValueError(msg)

    @property
    def effective_T_infer_test(self) -> int:
        """Resolved test inference steps (T_infer_test or T_infer * 10)."""
        return self.T_infer_test if self.T_infer_test is not None else self.T_infer * 10


# ---------------------------------------------------------------------------
# RichCallback (with plain-text fallback)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False


class _PlainCallback(TrainCallback):
    """Plain-text fallback when rich is not installed."""

    def on_lr_reduced(self, old_lr: float, new_lr: float) -> None:
        print(f"LR reduced: {old_lr:.6f} -> {new_lr:.6f}")  # noqa: T201

    def on_epoch_end(self, epoch: int, history: TrainHistory) -> None:
        energy = history.energy.per_epoch[-1] if history.energy.per_epoch else 0.0
        parts = [f"Epoch {epoch + 1}", f"energy={energy:.4f}"]
        if history.train_accuracy:
            parts.append(f"accuracy={history.train_accuracy[-1]:.4f}")
        print(" | ".join(parts))  # noqa: T201

    def on_train_end(self, history: TrainHistory) -> None:
        print("Training complete.")  # noqa: T201
        if history.energy.per_epoch:
            print(f"  Final energy: {history.energy.per_epoch[-1]:.4f}")  # noqa: T201
        if history.train_accuracy:
            print(f"  Final accuracy: {history.train_accuracy[-1]:.4f}")  # noqa: T201


class RichCallback(TrainCallback):
    """Rich console progress display for PCN training.

    Shows a live progress bar per epoch with batch speed, energy, accuracy,
    and estimated time remaining. Falls back to plain text if rich is not
    installed.

    Args:
        device: Optional device label shown in the training header
            (e.g. ``"cuda"`` or ``"cpu"``). If ``None``, the header omits
            device information.
    """

    def __init__(self, device: str | None = None) -> None:
        self._fallback: _PlainCallback | None = None
        self._device_label = device
        if not _RICH_AVAILABLE:
            self._fallback = _PlainCallback()
            return
        self._console = Console()
        self._progress: Progress | None = None
        self._epoch_task_id: object = None
        self._num_epochs: int = 0
        self._history: TrainHistory | None = None
        self._task_type: str = "classification"
        self._metrics_col = TextColumn("")

    def on_train_start(self, config: TrainConfig, history: TrainHistory) -> None:
        if self._fallback is not None:
            self._fallback.on_train_start(config, history)
            return
        self._history = history
        self._task_type = config.task
        self._num_epochs = config.num_epochs
        parts = ["[bold]PCN Training[/bold]"]
        if self._device_label:
            parts.append(f"device=[cyan]{self._device_label}[/cyan]")
        parts.append(f"task={config.task}")
        parts.append(f"T_infer={config.T_infer}")
        parts.append(f"epochs={config.num_epochs}")
        self._console.print(" | ".join(parts))

    def on_epoch_start(self, epoch: int, num_epochs: int) -> None:
        if self._fallback is not None:
            self._fallback.on_epoch_start(epoch, num_epochs)
            return
        self._num_epochs = num_epochs
        self._metrics_col = TextColumn("")
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn(f"Epoch {epoch + 1}/{num_epochs}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            self._metrics_col,
            console=self._console,
            transient=True,
        )
        self._progress.start()
        self._epoch_task_id = self._progress.add_task("epoch", total=None)

    def on_batch_start(self, batch: int, num_batches: int) -> None:
        if self._fallback is not None:
            self._fallback.on_batch_start(batch, num_batches)
            return
        if self._progress is not None and self._epoch_task_id is not None:
            self._progress.update(self._epoch_task_id, total=num_batches)  # type: ignore[arg-type]

    def on_batch_end(self, batch: int, batch_energy: float) -> None:
        if self._fallback is not None:
            return
        if self._progress is not None and self._epoch_task_id is not None:
            # Build live metrics string
            parts = [f"energy={batch_energy:.4f}"]
            if (
                self._task_type == "classification"
                and self._history is not None
                and self._history._running_accuracy > 0
            ):
                parts.append(f"acc={self._history._running_accuracy:.1%}")
            self._metrics_col.text_format = "[dim]|[/dim] " + " [dim]|[/dim] ".join(
                parts
            )
            self._progress.advance(self._epoch_task_id)  # type: ignore[arg-type]

    def on_lr_reduced(self, old_lr: float, new_lr: float) -> None:
        if self._fallback is not None:
            self._fallback.on_lr_reduced(old_lr, new_lr)
            return
        self._console.print(
            f"[yellow]  LR reduced: {old_lr:.6f} -> {new_lr:.6f}[/yellow]"
        )

    def on_epoch_end(self, epoch: int, history: TrainHistory) -> None:
        if self._fallback is not None:
            self._fallback.on_epoch_end(epoch, history)
            return
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        energy = history.energy.per_epoch[-1] if history.energy.per_epoch else 0.0
        parts = [f"  Epoch {epoch + 1}/{self._num_epochs}", f"energy={energy:.6f}"]
        if history.train_accuracy:
            parts.append(f"acc={history.train_accuracy[-1]:.1%}")
        self._console.print(" | ".join(parts))

    def on_train_end(self, history: TrainHistory) -> None:
        if self._fallback is not None:
            self._fallback.on_train_end(history)
            return
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        self._console.print("[bold green]Training complete.[/bold green]")
        if history.energy.per_epoch:
            self._console.print(f"  Final energy: {history.energy.per_epoch[-1]:.6f}")
        if history.train_accuracy:
            self._console.print(f"  Final accuracy: {history.train_accuracy[-1]:.4f}")


# ---------------------------------------------------------------------------
# Runtime autograd guard
# ---------------------------------------------------------------------------


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
                "Latent variable has an autograd graph attached "
                "(grad_fn is not None). "
                "PCN training should not build autograd graphs.",
                RuntimeWarning,
                stacklevel=3,
            )
            break


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_pcn(
    model: PredictiveCodingNetwork,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    config: TrainConfig,
) -> TrainHistory:
    """Train a PredictiveCodingNetwork using local Hebbian-like update rules.

    Implements Algorithm 3 from arXiv:2506.06332v1. All operations run
    under torch.no_grad(). No autograd graph is built.

    The training loop has two phases per batch:
    1. **Inference:** Update latent variables for T_infer steps to minimize
       prediction errors, keeping weights frozen.
    2. **Learning:** Update weights for T_learn steps using batch-averaged
       local gradients, keeping latents frozen.

    Args:
        model: The PredictiveCodingNetwork to train.
        dataloader: PyTorch DataLoader providing (x_batch, y_batch) tuples.
        config: Training configuration (hyperparameters, callbacks, etc.).

    Returns:
        TrainHistory with losses, energies, and accuracies per epoch.
    """
    device = next(model.parameters()).device
    history = TrainHistory(config=config)
    callback = config.callback if config.callback is not None else RichCallback()
    first_batch = True

    # LR schedule: build lookup from (epoch, batch) -> lr
    current_lr_learn = config.lr_learn
    lr_step_lookup: dict[tuple[int, int], float] = {}
    if config.lr_learn_steps is not None:
        for ep, batch, lr in config.lr_learn_steps:
            lr_step_lookup[(ep, batch)] = lr

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
                if y_batch.dim() == 1 or (
                    y_batch.dim() == 2 and y_batch.dtype in (torch.long, torch.int)
                ):
                    num_classes: int = model._readout.weight.shape[0]
                    y_batch = (
                        F.one_hot(y_batch.long(), num_classes=num_classes)
                        .float()
                        .to(device)
                    )
                else:
                    # Already 2D float -- assume pre-encoded one-hot
                    y_batch = y_batch.to(device)
            else:
                y_batch = y_batch.to(device)

            # Initialize latents for this batch
            model.init_latents(B)

            # Build weight reference list for in-place updates
            # nn.ModuleList iteration returns nn.Module, not PCNLayer
            weights: list[Tensor] = [
                layer.weight
                for layer in model.layers  # type: ignore[union-attr]
            ] + [model._readout.weight]  # type: ignore[assignment]

            T_learn = config.T_learn if config.T_learn is not None else B

            callback.on_batch_start(batch_idx, len(dataloader))

            # === INFERENCE PHASE ===
            amp_ctx = (
                torch.amp.autocast(device_type="cuda")
                if config.use_mixed_precision and device.type == "cuda"
                else nullcontext()
            )

            with torch.no_grad(), amp_ctx:
                energy_window: deque[float] = deque(maxlen=config.energy_window_size)

                for t in range(config.T_infer):
                    # Runtime guard on first iteration only
                    if first_batch and t == 0:
                        _check_no_autograd(model)
                        first_batch = False

                    # Compute all errors from consistent snapshot (Pitfall 1)
                    errors = model.compute_errors(x_batch, y_batch)

                    # Compute energy for this step
                    step_energy = compute_energy(errors)
                    energy_window.append(step_energy)
                    callback.on_inference_step(t, step_energy)

                    # Build extended errors: [E^(0), ..., E^(L-1), E^(L)]
                    extended_errors: list[Tensor] = list(errors.errors)
                    if errors.top_error is not None:
                        extended_errors.append(errors.top_error)
                    else:
                        # Unsupervised: E^(L) = 0
                        extended_errors.append(torch.zeros_like(model.latents[-1]))

                    # Update latents (synchronous: all errors computed first)
                    for idx in range(len(model.latents)):
                        # Paper: G_X^(l+1) = E^(l+1) - H^(l) @ W^(l)
                        grad_x = extended_errors[idx + 1] - (
                            errors.gm_errors[idx] @ weights[idx]
                        )
                        model.latents[idx] -= config.lr_infer * grad_x

                    # Early stopping check
                    if (
                        config.early_stop_threshold is not None
                        and len(energy_window) >= 2
                        and abs(energy_window[-1] - energy_window[-2])
                        < config.early_stop_threshold
                    ):
                        break

            # Manual LR step schedule: check before learning phase
            step_lr = lr_step_lookup.get((epoch, batch_idx))
            if step_lr is not None and step_lr != current_lr_learn:
                old_lr = current_lr_learn
                current_lr_learn = step_lr
                callback.on_lr_reduced(old_lr, current_lr_learn)

            # === LEARNING PHASE (no autocast -- Pitfall 5) ===
            states = [x_batch, *model.latents]

            with torch.no_grad():
                for t in range(T_learn):
                    # Recompute errors (weights changed each step!)
                    errors = model.compute_errors(x_batch, y_batch)

                    step_energy = compute_energy(errors)
                    callback.on_learning_step(t, step_energy)

                    # Weight gradients with batch averaging (Pitfall 2)
                    for idx in range(len(model.layers)):
                        # Paper: G_W^(l) = -(1/B) * H^(l).T @ X^(l+1)
                        grad_W = -(errors.gm_errors[idx].T @ states[idx + 1]) / B
                        weights[idx].data -= current_lr_learn * grad_W

                    # Readout weight update
                    if errors.supervised_error is not None:
                        # Paper: G_W_out = (1/B) * E_sup.T @ X^(L)
                        grad_W_out = (errors.supervised_error.T @ model.latents[-1]) / B
                        weights[-1].data -= current_lr_learn * grad_W_out

            # Record batch energy (after learning)
            batch_energy = compute_energy(model.compute_errors(x_batch, y_batch))
            history.energy.per_batch.append(batch_energy)
            epoch_energy += batch_energy * B

            # Track accuracy for classification via free inference
            # (no supervised signal, same as test_pcn) to get honest accuracy
            if config.task == "classification":
                # Re-initialize latents and run unsupervised inference
                # to measure actual classification ability
                model.init_latents(B)
                with torch.no_grad(), amp_ctx:
                    for _t in range(config.effective_T_infer_test):
                        free_errors = model.compute_errors(x_batch, y=None)
                        free_extended: list[Tensor] = list(free_errors.errors)
                        free_extended.append(torch.zeros_like(model.latents[-1]))
                        for idx in range(len(model.latents)):
                            grad_x = free_extended[idx + 1] - (
                                free_errors.gm_errors[idx] @ weights[idx]
                            )
                            model.latents[idx] -= config.lr_infer * grad_x

                y_hat = model.predict()
                predicted = y_hat.argmax(dim=1)
                targets = y_batch.argmax(dim=1) if y_batch.dim() > 1 else y_batch
                epoch_correct += int((predicted == targets).sum().item())
            epoch_total += B
            if config.task == "classification" and epoch_total > 0:
                history._running_accuracy = epoch_correct / epoch_total

            callback.on_batch_end(batch_idx, batch_energy)

        # Epoch summary
        epoch_mean_energy = epoch_energy / epoch_total if epoch_total > 0 else 0.0
        history.energy.per_epoch.append(epoch_mean_energy)
        history.train_loss.append(epoch_mean_energy)
        if config.task == "classification" and epoch_total > 0:
            history.train_accuracy.append(epoch_correct / epoch_total)

        # Track LR for this epoch
        history.lr_learn_per_epoch.append(current_lr_learn)

        callback.on_epoch_end(epoch, history)

    callback.on_train_end(history)
    return history


# ---------------------------------------------------------------------------
# Test / evaluation loop
# ---------------------------------------------------------------------------


def test_pcn(
    model: PredictiveCodingNetwork,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    config: TrainConfig,
) -> dict[str, float]:
    """Evaluate a trained PCN on a test set.

    Runs inference WITHOUT the supervised signal -- the model must predict
    the label from the input alone using only bottom-up generative errors.
    Labels are held out and used only for scoring after inference completes.

    Uses ``config.effective_T_infer_test`` inference steps (defaults to
    ``T_infer * 10``).  Unsupervised inference converges slower than
    supervised inference because the top latent receives no label signal
    and must be driven entirely by bottom-up prediction errors.

    Args:
        model: The trained PredictiveCodingNetwork to evaluate.
        dataloader: PyTorch DataLoader providing (x_batch, y_batch) tuples.
        config: Training configuration (uses T_infer_test, lr_infer, task).

    Returns:
        Dictionary with 'accuracy' and 'energy' keys.
    """
    model.train(mode=False)
    device = next(model.parameters()).device
    correct = 0
    total = 0
    total_energy = 0.0
    T_test = config.effective_T_infer_test

    amp_ctx = (
        torch.amp.autocast(device_type="cuda")
        if config.use_mixed_precision and device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            B = x_batch.shape[0]
            x_batch = x_batch.view(B, -1).to(device)
            y_batch = y_batch.to(device)

            # Initialize latents for this batch
            model.init_latents(B)

            # Build weight references for latent gradient computation
            # nn.ModuleList iteration returns nn.Module, not PCNLayer
            weights: list[Tensor] = [
                layer.weight
                for layer in model.layers  # type: ignore[union-attr]
            ] + [model._readout.weight]  # type: ignore[assignment]

            # Inference WITHOUT supervised signal (y=None).
            # Latents settle from bottom-up prediction errors only --
            # the model must infer the answer, not be told it.
            with amp_ctx:
                energy_window: deque[float] = deque(maxlen=config.energy_window_size)

                for _t in range(T_test):
                    errors = model.compute_errors(x_batch, y=None)
                    step_energy = compute_energy(errors)
                    energy_window.append(step_energy)

                    # No supervised signal -> top_error is None -> zero
                    extended_errors: list[Tensor] = list(errors.errors)
                    extended_errors.append(torch.zeros_like(model.latents[-1]))

                    # Update latents
                    for idx in range(len(model.latents)):
                        grad_x = extended_errors[idx + 1] - (
                            errors.gm_errors[idx] @ weights[idx]
                        )
                        model.latents[idx] -= config.lr_infer * grad_x

                    # Early stopping
                    if (
                        config.early_stop_threshold is not None
                        and len(energy_window) >= 2
                        and abs(energy_window[-1] - energy_window[-2])
                        < config.early_stop_threshold
                    ):
                        break

            # Score: read prediction from settled latents, compare to held-out labels
            final_errors = model.compute_errors(x_batch, y=None)
            batch_energy = compute_energy(final_errors)
            total_energy += batch_energy * B

            if config.task == "classification":
                y_hat = model.predict()
                predicted = y_hat.argmax(dim=1)
                targets = y_batch.argmax(dim=1) if y_batch.dim() > 1 else y_batch
                correct += int((predicted == targets).sum().item())
            total += B

    accuracy = correct / total if config.task == "classification" and total > 0 else 0.0
    energy = total_energy / total if total > 0 else 0.0

    return {"accuracy": accuracy, "energy": energy}
