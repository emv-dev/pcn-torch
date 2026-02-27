"""Tests for PCN training loop correctness and integration.

Covers:
- TrainConfig validation
- TST-02: Gain-modulated error uses preactivation f'(a^(l)) * eps
- TST-03: No autograd graph is built during training
- TST-04: top_error = supervised_error @ W_out with correct shape
- Training integration: train_pcn and test_pcn on synthetic data
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pcn_torch.network import PredictiveCodingNetwork
from pcn_torch.trainer import (
    TrainCallback,
    TrainConfig,
    TrainHistory,
    train_pcn,
)
from pcn_torch.trainer import (
    test_pcn as evaluate_pcn,  # alias avoids pytest collection
)

if TYPE_CHECKING:
    from pcn_torch.layers import PCNLayer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_dataloader(
    n_samples: int = 16,
    input_dim: int = 20,
    num_classes: int = 5,
    batch_size: int = 8,
    seed: int = 42,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a small deterministic classification DataLoader."""
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def _make_regression_dataloader(
    n_samples: int = 16,
    input_dim: int = 20,
    output_dim: int = 5,
    batch_size: int = 8,
    seed: int = 42,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a small deterministic regression DataLoader."""
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples, output_dim)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def _fast_config(**overrides: object) -> TrainConfig:
    """Return a fast TrainConfig for structural tests (not convergence)."""
    defaults: dict[str, object] = {
        "T_infer": 3,
        "T_infer_test": 5,  # Keep test inference steps small for fast tests
        "T_learn": 2,
        "num_epochs": 1,
        "use_mixed_precision": False,
        "callback": TrainCallback(),  # silent no-op callback
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TrainConfig validation tests
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    """Verify default values match paper: T_infer=50, lr_infer=0.05, etc."""
    config = TrainConfig()
    assert config.task == "classification"
    assert config.T_infer == 50
    assert config.lr_infer == 0.05
    assert config.lr_learn == 0.005
    assert config.num_epochs == 4
    assert config.lr_learn_steps is None


def test_config_invalid_task() -> None:
    """ValueError for invalid task string."""
    with pytest.raises(ValueError, match="task must be"):
        TrainConfig(task="invalid")


def test_config_invalid_T_infer() -> None:
    """ValueError for T_infer < 1."""
    with pytest.raises(ValueError, match="T_infer must be"):
        TrainConfig(T_infer=0)


def test_config_invalid_T_learn() -> None:
    """ValueError for T_learn < 1 (when not None)."""
    with pytest.raises(ValueError, match="T_learn must be"):
        TrainConfig(T_learn=0)


def test_config_invalid_T_infer_test() -> None:
    """ValueError for T_infer_test < 1 (when not None)."""
    with pytest.raises(ValueError, match="T_infer_test must be"):
        TrainConfig(T_infer_test=0)


def test_config_effective_T_infer_test_default() -> None:
    """effective_T_infer_test defaults to T_infer * 10."""
    config = TrainConfig(T_infer=50)
    assert config.effective_T_infer_test == 500


def test_config_effective_T_infer_test_explicit() -> None:
    """effective_T_infer_test uses explicit T_infer_test when set."""
    config = TrainConfig(T_infer=50, T_infer_test=200)
    assert config.effective_T_infer_test == 200


# ---------------------------------------------------------------------------
# LR schedule config validation tests
# ---------------------------------------------------------------------------


def test_config_lr_learn_steps_default_none() -> None:
    """lr_learn_steps defaults to None (constant lr_learn)."""
    config = TrainConfig()
    assert config.lr_learn_steps is None


def test_config_lr_learn_steps_valid() -> None:
    """lr_learn_steps accepts valid (epoch, batch, lr) tuples."""
    steps: list[tuple[int, int, float]] = [(0, 0, 0.05), (0, 20, 0.005)]
    config = TrainConfig(lr_learn_steps=steps)
    assert config.lr_learn_steps == steps


def test_config_lr_learn_steps_empty_raises() -> None:
    """ValueError for empty lr_learn_steps list."""
    with pytest.raises(ValueError, match="lr_learn_steps"):
        TrainConfig(lr_learn_steps=[])


def test_config_lr_learn_steps_negative_epoch_raises() -> None:
    """ValueError for negative epoch in lr_learn_steps."""
    with pytest.raises(ValueError, match="epoch must be >= 0"):
        TrainConfig(lr_learn_steps=[(-1, 0, 0.01)])


def test_config_lr_learn_steps_negative_batch_raises() -> None:
    """ValueError for negative batch in lr_learn_steps."""
    with pytest.raises(ValueError, match="batch must be >= 0"):
        TrainConfig(lr_learn_steps=[(0, -1, 0.01)])


def test_config_lr_learn_steps_zero_lr_raises() -> None:
    """ValueError for lr <= 0 in lr_learn_steps."""
    with pytest.raises(ValueError, match="lr must be > 0"):
        TrainConfig(lr_learn_steps=[(0, 0, 0.0)])


# ---------------------------------------------------------------------------
# LR schedule behavior tests
# ---------------------------------------------------------------------------


def test_lr_steps_applied_at_correct_batch() -> None:
    """lr_learn_steps changes LR at the specified (epoch, batch)."""

    class LRTracker(TrainCallback):
        def __init__(self) -> None:
            self.reductions: list[tuple[float, float]] = []

        def on_lr_reduced(self, old_lr: float, new_lr: float) -> None:
            self.reductions.append((old_lr, new_lr))

    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = _make_dataloader()
    tracker = LRTracker()
    config = _fast_config(
        num_epochs=2,
        lr_learn=0.1,
        lr_learn_steps=[(0, 0, 0.1), (1, 0, 0.01)],
        callback=tracker,
    )
    history = train_pcn(net, dl, config)

    # LR should change at epoch 1 batch 0: 0.1 -> 0.01
    assert len(tracker.reductions) == 1
    assert tracker.reductions[0] == (0.1, 0.01)
    assert len(history.lr_learn_per_epoch) == 2


def test_lr_unchanged_when_no_steps() -> None:
    """LR stays constant when lr_learn_steps is None."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = _make_dataloader()
    config = _fast_config(
        num_epochs=3,
        lr_learn=0.5,
    )
    history = train_pcn(net, dl, config)

    assert len(history.lr_learn_per_epoch) == 3
    assert all(lr == 0.5 for lr in history.lr_learn_per_epoch)


def test_lr_learn_per_epoch_tracked() -> None:
    """history.lr_learn_per_epoch has one entry per epoch."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = _make_dataloader()
    config = _fast_config(num_epochs=3)
    history = train_pcn(net, dl, config)

    assert len(history.lr_learn_per_epoch) == 3


# ---------------------------------------------------------------------------
# TST-02: Gain-modulated error uses preactivation
# ---------------------------------------------------------------------------


def test_gain_modulated_error_uses_preactivation() -> None:
    """TST-02: Verify h^(l) = f'(a^(l)) * eps^(l) uses PREACTIVATION."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    net.init_latents(4)
    x = torch.randn(4, 20)

    errors = net.compute_errors(x)

    # Manual computation for layer 0
    states = [x, *net.latents]
    layer0: PCNLayer = net.layers[0]  # type: ignore[assignment]
    preact = states[1] @ layer0.weight.T
    eps = states[0] - layer0.activation_fn(preact)
    expected_h = layer0.activation_deriv(preact) * eps

    assert torch.allclose(errors.gm_errors[0], expected_h)


# ---------------------------------------------------------------------------
# TST-03: No autograd graph during training
# ---------------------------------------------------------------------------


def test_no_autograd_graph_during_training() -> None:
    """TST-03: Verify training runs without building autograd graphs."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = _make_dataloader(n_samples=16, input_dim=20, num_classes=5, batch_size=8)
    config = _fast_config()

    # Should complete without RuntimeWarning about autograd
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        train_pcn(net, dl, config)
        autograd_warnings = [x for x in w if "autograd" in str(x.message).lower()]
        assert len(autograd_warnings) == 0

    # Verify no latent has grad_fn after training
    for lat in net.latents:
        assert lat.grad_fn is None
        assert not lat.requires_grad


# ---------------------------------------------------------------------------
# TST-04: Correct W_out^T supervised error projection
# ---------------------------------------------------------------------------


def test_top_error_wout_projection() -> None:
    """TST-04: Verify top_error = supervised_error @ W_out."""
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
    # Verify shape: (B, d_L) -- for dims=[20, 10, 5], d_L = 5
    assert errors.top_error.shape == (4, 5)


# ---------------------------------------------------------------------------
# Training integration tests
# ---------------------------------------------------------------------------


def test_train_pcn_classification() -> None:
    """train_pcn completes on classification data with populated history."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = _make_dataloader()
    config = _fast_config()

    history = train_pcn(net, dl, config)

    assert isinstance(history, TrainHistory)
    assert len(history.energy.per_batch) > 0
    assert len(history.energy.per_epoch) == 1
    assert len(history.train_accuracy) == 1


def test_train_pcn_regression() -> None:
    """train_pcn completes with task='regression'; no accuracy tracked."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = _make_regression_dataloader()
    config = _fast_config(task="regression")

    history = train_pcn(net, dl, config)

    assert isinstance(history, TrainHistory)
    assert len(history.energy.per_batch) > 0
    assert len(history.energy.per_epoch) == 1
    # Regression does not track accuracy
    assert len(history.train_accuracy) == 0


def test_train_pcn_energy_per_batch_length() -> None:
    """Verify per_batch list has expected length: num_batches * num_epochs."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    n_samples = 16
    batch_size = 8
    num_epochs = 2
    dl = _make_dataloader(n_samples=n_samples, batch_size=batch_size)
    config = _fast_config(num_epochs=num_epochs)

    history = train_pcn(net, dl, config)

    expected_batches = (n_samples // batch_size) * num_epochs
    assert len(history.energy.per_batch) == expected_batches


def test_train_pcn_history_config_stored() -> None:
    """history.config is the same config object that was passed in."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = _make_dataloader()
    config = _fast_config()

    history = train_pcn(net, dl, config)

    assert history.config is config


def test_test_pcn_returns_metrics() -> None:
    """test_pcn returns dict with 'accuracy' and 'energy' keys as floats."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = _make_dataloader()
    config = _fast_config()

    result = evaluate_pcn(net, dl, config)

    assert isinstance(result, dict)
    assert "accuracy" in result
    assert "energy" in result
    assert isinstance(result["accuracy"], float)
    assert isinstance(result["energy"], float)


def test_train_pcn_weights_change() -> None:
    """Verify weights actually change after training."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = _make_dataloader()
    config = _fast_config()

    # Snapshot weights before training
    before = {name: param.clone() for name, param in net.named_parameters()}

    train_pcn(net, dl, config)

    # At least one weight should differ
    any_changed = False
    for name, param in net.named_parameters():
        if not torch.equal(param, before[name]):
            any_changed = True
            break
    assert any_changed, "No weights changed after training"


def test_train_pcn_mixed_precision_false() -> None:
    """Runs without error when use_mixed_precision=False (CPU test)."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    dl = _make_dataloader()
    config = _fast_config(use_mixed_precision=False)

    history = train_pcn(net, dl, config)

    assert isinstance(history, TrainHistory)
    assert len(history.energy.per_batch) > 0
