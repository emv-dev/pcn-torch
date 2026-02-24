"""Predictive Coding Networks in PyTorch."""

from pcn_torch._types import ActivationDeriv, ActivationFn, ActivationPair
from pcn_torch.activations import get_activation
from pcn_torch.layers import PCNLayer
from pcn_torch.network import PCNErrors, PredictiveCodingNetwork

try:
    from pcn_torch._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "ActivationDeriv",
    "ActivationFn",
    "ActivationPair",
    "PCNErrors",
    "PCNLayer",
    "PredictiveCodingNetwork",
    "get_activation",
]
