"""Tests for activation functions and derivatives."""

from __future__ import annotations

import pytest
import torch

from pcn_torch import get_activation


class TestGetActivation:
    """Test the activation registry lookup."""

    @pytest.mark.parametrize("name", ["relu", "tanh", "sigmoid", "identity"])
    def test_returns_callable_pair(self, name: str) -> None:
        fn, deriv = get_activation(name)
        assert callable(fn)
        assert callable(deriv)

    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation 'unknown'"):
            get_activation("unknown")

    def test_error_message_lists_available(self) -> None:
        with pytest.raises(ValueError, match="Available:"):
            get_activation("nonexistent")


class TestRelu:
    """Test ReLU activation and its derivative."""

    def test_zeros_out_negatives(self) -> None:
        fn, _ = get_activation("relu")
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = fn(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        assert torch.allclose(result, expected)

    def test_derivative(self) -> None:
        _, deriv = get_activation("relu")
        x = torch.tensor([-2.0, -1.0, 0.5, 1.0, 2.0])
        result = deriv(x)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_derivative_at_zero_is_zero(self) -> None:
        """ReLU derivative at zero is 0 by convention (x > 0, not x >= 0)."""
        _, deriv = get_activation("relu")
        result = deriv(torch.tensor([0.0]))
        assert result.item() == 0.0


class TestTanh:
    """Test tanh activation and its derivative."""

    def test_output_range(self) -> None:
        fn, _ = get_activation("tanh")
        torch.manual_seed(42)
        x = torch.randn(1000)
        result = fn(x)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_derivative_matches_formula(self) -> None:
        """tanh'(x) = 1 - tanh(x)^2."""
        fn, deriv = get_activation("tanh")
        torch.manual_seed(42)
        x = torch.randn(100)
        result = deriv(x)
        t = torch.tanh(x)
        expected = 1.0 - t * t
        assert torch.allclose(result, expected)


class TestSigmoid:
    """Test sigmoid activation and its derivative."""

    def test_output_range(self) -> None:
        fn, _ = get_activation("sigmoid")
        torch.manual_seed(42)
        x = torch.randn(1000)
        result = fn(x)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_derivative_matches_formula(self) -> None:
        """sigmoid'(x) = s(x) * (1 - s(x))."""
        fn, deriv = get_activation("sigmoid")
        torch.manual_seed(42)
        x = torch.randn(100)
        result = deriv(x)
        s = torch.sigmoid(x)
        expected = s * (1.0 - s)
        assert torch.allclose(result, expected)


class TestIdentity:
    """Test identity activation and its derivative."""

    def test_fn_returns_input(self) -> None:
        fn, _ = get_activation("identity")
        torch.manual_seed(42)
        x = torch.randn(50)
        result = fn(x)
        assert torch.equal(result, x)

    def test_derivative_is_ones(self) -> None:
        _, deriv = get_activation("identity")
        torch.manual_seed(42)
        x = torch.randn(50)
        result = deriv(x)
        assert torch.equal(result, torch.ones_like(x))
