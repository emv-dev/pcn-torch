"""Activation functions with hand-coded derivatives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from pcn_torch._types import ActivationPair


def _relu(x: Tensor) -> Tensor:
    return torch.relu(x)


def _relu_deriv(x: Tensor) -> Tensor:
    return (x > 0).float()


def _tanh(x: Tensor) -> Tensor:
    return torch.tanh(x)


def _tanh_deriv(x: Tensor) -> Tensor:
    t = torch.tanh(x)
    return 1.0 - t * t


def _sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def _sigmoid_deriv(x: Tensor) -> Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 - s)


def _identity(x: Tensor) -> Tensor:
    return x


def _identity_deriv(x: Tensor) -> Tensor:
    return torch.ones_like(x)


_ACTIVATIONS: dict[str, ActivationPair] = {
    "relu": (_relu, _relu_deriv),
    "tanh": (_tanh, _tanh_deriv),
    "sigmoid": (_sigmoid, _sigmoid_deriv),
    "identity": (_identity, _identity_deriv),
}


def get_activation(name: str) -> ActivationPair:
    """Look up an activation function and its derivative by name.

    Args:
        name: Activation name. One of: relu, tanh, sigmoid, identity.

    Returns:
        A tuple of (activation_fn, activation_deriv).

    Raises:
        ValueError: If the activation name is not recognized.
    """
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]
