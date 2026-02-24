"""Tests for energy computation from PCN error signals."""

from __future__ import annotations

import torch

from pcn_torch.energy import compute_energy, compute_energy_per_layer
from pcn_torch.network import PCNErrors, PredictiveCodingNetwork


def _make_network_and_errors(
    *, with_target: bool = True
) -> tuple[PredictiveCodingNetwork, PCNErrors]:
    """Create a small network and compute errors for testing.

    Args:
        with_target: If True, compute errors with a target (supervised).

    Returns:
        Tuple of (network, errors).
    """
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    net.init_latents(4)
    x = torch.randn(4, 20)
    y = torch.randn(4, 5) if with_target else None
    errors = net.compute_errors(x, y)
    return net, errors


def test_compute_energy_returns_float() -> None:
    """Verify compute_energy returns a Python float, not a Tensor."""
    _, errors = _make_network_and_errors()
    energy = compute_energy(errors)
    assert isinstance(energy, float)


def test_compute_energy_non_negative() -> None:
    """Verify energy >= 0 (sum of squared norms is always non-negative)."""
    _, errors = _make_network_and_errors()
    energy = compute_energy(errors)
    assert energy >= 0.0


def test_compute_energy_without_supervised() -> None:
    """Verify compute_energy works when supervised_error is None."""
    _, errors = _make_network_and_errors(with_target=False)
    assert errors.supervised_error is None
    energy = compute_energy(errors)
    assert isinstance(energy, float)
    assert energy >= 0.0


def test_compute_energy_with_supervised() -> None:
    """Verify supervised error adds a positive term to energy."""
    torch.manual_seed(42)
    net = PredictiveCodingNetwork(dims=[20, 10, 5])
    net.init_latents(4)
    x = torch.randn(4, 20)
    y = torch.randn(4, 5)

    errors_no_y = net.compute_errors(x)
    energy_no_y = compute_energy(errors_no_y)

    errors_with_y = net.compute_errors(x, y)
    energy_with_y = compute_energy(errors_with_y)

    # Supervised error adds a positive term, so energy_with_y > energy_no_y
    # (unless supervised_error is exactly zero, which is astronomically unlikely)
    assert energy_with_y > energy_no_y


def test_compute_energy_manual_formula() -> None:
    """Verify compute_energy matches manual (1/B) * sum(0.5 * ||e||^2)."""
    _, errors = _make_network_and_errors()
    B = errors.errors[0].shape[0]

    manual_energy = 0.0
    for e in errors.errors:
        manual_energy += 0.5 * (e * e).sum().item() / B
    if errors.supervised_error is not None:
        sf = errors.supervised_error
        manual_energy += 0.5 * (sf * sf).sum().item() / B

    computed = compute_energy(errors)
    assert torch.allclose(
        torch.tensor(computed), torch.tensor(manual_energy), atol=1e-6
    )


def test_compute_energy_per_layer_count() -> None:
    """Verify per_layer length: num_layers without y, num_layers+1 with y."""
    _, errors_with_y = _make_network_and_errors(with_target=True)
    _, errors_no_y = _make_network_and_errors(with_target=False)

    per_layer_with_y = compute_energy_per_layer(errors_with_y)
    per_layer_no_y = compute_energy_per_layer(errors_no_y)

    num_layers = len(errors_with_y.errors)
    assert len(per_layer_no_y) == num_layers
    assert len(per_layer_with_y) == num_layers + 1


def test_compute_energy_per_layer_sum_matches_total() -> None:
    """Verify sum of per-layer energies approximately equals total energy."""
    _, errors = _make_network_and_errors()
    total = compute_energy(errors)
    per_layer = compute_energy_per_layer(errors)
    assert torch.allclose(
        torch.tensor(sum(per_layer)), torch.tensor(total), atol=1e-6
    )


def test_compute_energy_float16_safe() -> None:
    """Verify compute_energy returns finite values with float16 errors."""
    torch.manual_seed(42)
    # Create errors with float16 tensors
    errors_f16 = PCNErrors(
        errors=[torch.randn(4, 20).half(), torch.randn(4, 10).half()],
        gm_errors=[torch.randn(4, 20).half(), torch.randn(4, 10).half()],
        supervised_error=torch.randn(4, 5).half(),
        top_error=torch.randn(4, 5).half(),
    )
    energy = compute_energy(errors_f16)
    assert isinstance(energy, float)
    assert not torch.isnan(torch.tensor(energy))
    assert not torch.isinf(torch.tensor(energy))
