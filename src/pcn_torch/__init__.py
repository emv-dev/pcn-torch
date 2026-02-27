"""Predictive Coding Networks in PyTorch."""

from pcn_torch._types import ActivationDeriv, ActivationFn, ActivationPair
from pcn_torch.activations import get_activation
from pcn_torch.energy import compute_energy, compute_energy_per_layer
from pcn_torch.layers import PCNLayer
from pcn_torch.network import PCNErrors, PredictiveCodingNetwork
from pcn_torch.trainer import (
    EnergyHistory,
    RichCallback,
    TrainCallback,
    TrainConfig,
    TrainHistory,
    test_pcn,
    train_pcn,
)

try:
    from pcn_torch._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "ActivationDeriv",
    "ActivationFn",
    "ActivationPair",
    "EnergyHistory",
    "PCNErrors",
    "PCNLayer",
    "PredictiveCodingNetwork",
    "RichCallback",
    "TrainCallback",
    "TrainConfig",
    "TrainHistory",
    "compute_energy",
    "compute_energy_per_layer",
    "get_activation",
    "test_pcn",
    "train_pcn",
]
