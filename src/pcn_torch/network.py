"""PredictiveCodingNetwork -- the core model composing PCNLayers."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor, nn

from pcn_torch.activations import get_activation
from pcn_torch.layers import PCNLayer

if TYPE_CHECKING:
    from collections.abc import Sequence

_VALID_MODES = frozenset({"classification", "regression"})


class PCNErrors(NamedTuple):
    """Container for all error signals computed by a PCN forward pass.

    Attributes:
        errors: Prediction errors eps^(l) for l=0..L-1.
        gm_errors: Gain-modulated errors h^(l) for l=0..L-1.
        supervised_error: eps_sup = y_hat - y, or None if no target.
        top_error: eps^(L) = eps_sup @ W_out, or None if no target.
    """

    errors: list[Tensor]
    gm_errors: list[Tensor]
    supervised_error: Tensor | None
    top_error: Tensor | None


class PredictiveCodingNetwork(nn.Module):
    """A Predictive Coding Network composed of stacked PCNLayers.

    Manages latent variables, computes prediction errors and gain-modulated
    errors across all layers, and provides a linear readout for output
    predictions with supervised error projection.

    Construct from either a dimension list or pre-built PCNLayer instances
    (mutually exclusive).

    Args:
        dims: Sequence of layer dimensions, top-down from input to top latent.
            Layer l is PCNLayer(in_features=dims[l+1], out_features=dims[l]).
        layers: Pre-built PCNLayer instances with validated dimensions.
        activation: Global activation function name (used only with dims).
        output_dim: Output dimension for the readout layer. Defaults to the
            top latent dimension.
        mode: Either "classification" or "regression".

    Raises:
        ValueError: If both or neither of dims/layers provided, or if
            dims has fewer than 2 elements, or layers is empty, or adjacent
            layer dimensions are incompatible, or mode is invalid.
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

        if dims is not None and layers is not None:
            msg = "Provide either 'dims' or 'layers', not both."
            raise ValueError(msg)
        if dims is None and layers is None:
            msg = "Provide either 'dims' or 'layers'."
            raise ValueError(msg)
        if mode not in _VALID_MODES:
            msg = f"Invalid mode '{mode}'. Must be one of: {sorted(_VALID_MODES)}"
            raise ValueError(msg)

        self.mode = mode

        if dims is not None:
            if len(dims) < 2:
                msg = f"'dims' must have at least 2 elements, got {len(dims)}."
                raise ValueError(msg)
            # Validate activation name eagerly
            get_activation(activation)
            built_layers = [
                PCNLayer(
                    in_features=dims[idx + 1],
                    out_features=dims[idx],
                    activation=activation,
                )
                for idx in range(len(dims) - 1)
            ]
            self._dims = list(dims)
        else:
            assert layers is not None  # for type narrowing
            layer_list = list(layers)
            if len(layer_list) == 0:
                msg = "'layers' must not be empty."
                raise ValueError(msg)
            # Validate adjacent layer compatibility
            for i in range(len(layer_list) - 1):
                in_feat_curr = layer_list[i].weight.shape[1]
                out_feat_next = layer_list[i + 1].weight.shape[0]
                if in_feat_curr != out_feat_next:
                    msg = (
                        f"Dimension mismatch between layers {i} and {i + 1}: "
                        f"layer {i} in_features={in_feat_curr} != "
                        f"layer {i + 1} out_features={out_feat_next}."
                    )
                    raise ValueError(msg)
            # Reconstruct dims from weight shapes
            self._dims = [layer_list[0].weight.shape[0]]
            for layer in layer_list:
                self._dims.append(layer.weight.shape[1])
            built_layers = layer_list

        self.layers = nn.ModuleList(built_layers)

        # Readout: maps top latent (d_L) to output dimension
        d_top = self._dims[-1]
        effective_output_dim = output_dim if output_dim is not None else d_top
        self._readout = nn.Linear(d_top, effective_output_dim, bias=False)

        # Public mutable state
        self.latents: list[Tensor] = []
        self.errors: PCNErrors | None = None

    def init_latents(self, batch_size: int) -> None:
        """Initialize latent variables from N(0,1) for all internal layers.

        Creates latents for layers l=1..L (excluding the input layer).
        Latents are plain tensors, not nn.Parameters.

        Args:
            batch_size: Number of samples in the batch.
        """
        device = next(self.parameters()).device
        self.latents = [
            torch.randn(batch_size, self._dims[idx], device=device)
            for idx in range(1, len(self._dims))
        ]

    def predict(self) -> Tensor:
        """Compute output prediction from the top latent via the readout layer.

        Returns:
            Prediction tensor of shape (batch_size, output_dim).

        Raises:
            RuntimeError: If latents have not been initialized.
        """
        if not self.latents:
            msg = "Latents not initialized. Call init_latents() first."
            raise RuntimeError(msg)
        result: Tensor = self._readout(self.latents[-1])
        return result

    def compute_errors(self, x: Tensor, y: Tensor | None = None) -> PCNErrors:
        """Compute prediction errors and gain-modulated errors for all layers.

        Follows arXiv:2506.06332v1 equations:
        - Prediction error: eps^(l) = x^(l) - f(W^(l) @ x^(l+1))
        - Gain-modulated error: h^(l) = f'(a^(l)) * eps^(l)
        - Supervised error (if y given): eps_sup = y_hat - y
        - Top error (if y given): eps^(L) = eps_sup @ W_out

        Args:
            x: Input tensor of shape (batch_size, dims[0]).
            y: Optional target tensor of shape (batch_size, output_dim).

        Returns:
            PCNErrors with all computed error signals.

        Raises:
            RuntimeError: If latents have not been initialized.
        """
        if not self.latents:
            msg = "Latents not initialized. Call init_latents() first."
            raise RuntimeError(msg)

        # Build full state: x^(0) is the clamped input, then latents
        states = [x, *self.latents]

        errors: list[Tensor] = []
        gm_errors: list[Tensor] = []

        for idx in range(len(self.layers)):
            pcn_layer: PCNLayer = self.layers[idx]  # type: ignore[assignment]
            prediction, preactivation = pcn_layer(states[idx + 1])
            eps = states[idx] - prediction
            h_val: Tensor = pcn_layer.activation_deriv(preactivation) * eps
            errors.append(eps)
            gm_errors.append(h_val)

        supervised_error: Tensor | None = None
        top_error: Tensor | None = None

        if y is not None:
            y_hat = self.predict()
            supervised_error = y_hat - y
            # W_out^T projection: (B, d_out) @ (d_out, d_L) -> (B, d_L)
            top_error = supervised_error @ self._readout.weight

        result = PCNErrors(
            errors=errors,
            gm_errors=gm_errors,
            supervised_error=supervised_error,
            top_error=top_error,
        )
        self.errors = result
        return result
