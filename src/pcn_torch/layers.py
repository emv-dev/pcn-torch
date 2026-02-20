"""PCNLayer -- a single predictive coding network layer."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pcn_torch.activations import get_activation


class PCNLayer(nn.Module):
    """A single layer in a Predictive Coding Network.

    Follows nn.Linear constructor convention: positional (in_features,
    out_features) with keyword activation. No bias term.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        activation: Activation function name. Default: "relu".
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act_fn, act_deriv = get_activation(activation)
        self.activation_fn = act_fn
        self.activation_deriv = act_deriv
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass: compute prediction and preactivation.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            A tuple of (prediction, preactivation), both of shape
            (batch_size, out_features).
        """
        preactivation = x @ self.weight.T
        prediction = self.activation_fn(preactivation)
        return prediction, preactivation

    def extra_repr(self) -> str:
        return (
            f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}"
        )
