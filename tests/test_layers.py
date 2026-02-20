"""Tests for PCNLayer construction, forward pass, and shapes."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pcn_torch import PCNLayer


class TestPCNLayerSubclass:
    """Test that PCNLayer is a proper nn.Module subclass."""

    def test_is_nn_module_subclass(self) -> None:
        assert issubclass(PCNLayer, nn.Module)

    def test_instance_is_nn_module(self) -> None:
        layer = PCNLayer(10, 5)
        assert isinstance(layer, nn.Module)


class TestPCNLayerConstruction:
    """Test PCNLayer construction with various arguments."""

    def test_default_construction(self) -> None:
        layer = PCNLayer(784, 500)
        assert layer.weight.shape == (500, 784)

    def test_activation_relu(self) -> None:
        layer = PCNLayer(10, 5, activation="relu")
        assert layer.weight.shape == (5, 10)

    def test_activation_tanh(self) -> None:
        layer = PCNLayer(10, 5, activation="tanh")
        assert layer.weight.shape == (5, 10)

    def test_activation_sigmoid(self) -> None:
        layer = PCNLayer(10, 5, activation="sigmoid")
        assert layer.weight.shape == (5, 10)

    def test_activation_identity(self) -> None:
        layer = PCNLayer(10, 5, activation="identity")
        assert layer.weight.shape == (5, 10)

    def test_invalid_activation_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            PCNLayer(10, 5, activation="nonexistent")


class TestPCNLayerForward:
    """Test PCNLayer forward pass shapes and behavior."""

    def test_forward_shapes(self) -> None:
        layer = PCNLayer(784, 500)
        x = torch.randn(32, 784)
        prediction, preactivation = layer(x)
        assert prediction.shape == (32, 500)
        assert preactivation.shape == (32, 500)

    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    def test_forward_various_batch_sizes(self, batch_size: int) -> None:
        layer = PCNLayer(100, 50)
        x = torch.randn(batch_size, 100)
        prediction, preactivation = layer(x)
        assert prediction.shape == (batch_size, 50)
        assert preactivation.shape == (batch_size, 50)

    def test_forward_returns_tuple(self) -> None:
        layer = PCNLayer(10, 5)
        x = torch.randn(4, 10)
        result = layer(x)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestPCNLayerParameters:
    """Test PCNLayer parameter registration and properties."""

    def test_weight_is_nn_parameter(self) -> None:
        layer = PCNLayer(10, 5)
        assert isinstance(layer.weight, nn.Parameter)

    def test_weight_in_parameters(self) -> None:
        layer = PCNLayer(10, 5)
        params = list(layer.parameters())
        assert len(params) == 1
        assert params[0] is layer.weight

    def test_no_bias_parameter(self) -> None:
        layer = PCNLayer(10, 5)
        param_names = [name for name, _ in layer.named_parameters()]
        assert not any("bias" in name for name in param_names)

    def test_extra_repr_contains_features(self) -> None:
        layer = PCNLayer(784, 500)
        repr_str = layer.extra_repr()
        assert "in_features" in repr_str
        assert "out_features" in repr_str
        assert "784" in repr_str
        assert "500" in repr_str
