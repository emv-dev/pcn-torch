"""Energy computation from PCN error signals.

Computes the batch-averaged total energy (prediction error + supervised error)
following arXiv:2506.06332v1 Section 5.2:

    E_batch = (1/B) * sum_b [ (1/2) * sum_l ||eps_b^(l)||^2
                             + (1/2) * ||eps_sup_b||^2 ]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pcn_torch.network import PCNErrors


def compute_energy(errors: PCNErrors) -> float:
    """Compute batch-averaged total energy from error signals.

    Casts to float32 before squaring to avoid float16 overflow
    when running inside autocast regions.

    Args:
        errors: PCNErrors from model.compute_errors().

    Returns:
        Scalar energy value (Python float).
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


def compute_energy_per_layer(errors: PCNErrors) -> list[float]:
    """Compute per-layer energy contributions.

    Each entry is the batch-averaged energy for one layer:
        0.5 * (1/B) * sum_b ||eps_b^(l)||^2

    If a supervised error is present, it is appended as the final element.

    Args:
        errors: PCNErrors from model.compute_errors().

    Returns:
        List of per-layer energy values (Python floats).
    """
    B = errors.errors[0].shape[0]
    per_layer: list[float] = []
    for e in errors.errors:
        ef = e.float()
        per_layer.append(0.5 * (ef * ef).sum().item() / B)
    if errors.supervised_error is not None:
        sf = errors.supervised_error.float()
        per_layer.append(0.5 * (sf * sf).sum().item() / B)
    return per_layer
