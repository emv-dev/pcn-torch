"""Tests for PredictiveCodingNetwork construction, latent init, errors, and readout."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pcn_torch import PCNErrors, PCNLayer, PredictiveCodingNetwork


class TestPCNErrorsNamedTuple:
    """Test that PCNErrors is a proper NamedTuple."""

    def test_is_named_tuple(self) -> None:
        dummy = PCNErrors(
            errors=[], gm_errors=[], supervised_error=None, top_error=None
        )
        assert hasattr(dummy, "_fields")
        assert dummy._fields == ("errors", "gm_errors", "supervised_error", "top_error")

    def test_tuple_unpacking(self) -> None:
        errs = [torch.zeros(2, 3)]
        gm = [torch.ones(2, 3)]
        sup = torch.randn(2, 5)
        top = torch.randn(2, 3)
        errors, gm_errors, supervised_error, top_error = PCNErrors(errs, gm, sup, top)
        assert errors is errs
        assert gm_errors is gm
        assert supervised_error is sup
        assert top_error is top

    def test_named_access(self) -> None:
        errs = [torch.zeros(2, 3)]
        result = PCNErrors(
            errors=errs, gm_errors=[], supervised_error=None, top_error=None
        )
        assert result.errors is errs
        assert result.gm_errors == []
        assert result.supervised_error is None
        assert result.top_error is None


class TestNetworkConstruction:
    """Test PredictiveCodingNetwork construction from dims and layers."""

    def test_from_dims_basic(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        assert len(net.layers) == 2

    def test_from_dims_cifar10(self) -> None:
        net = PredictiveCodingNetwork(dims=[3072, 500, 500, 10])
        assert len(net.layers) == 3
        # Layer 0: PCNLayer(in_features=500, out_features=3072) -> weight (3072, 500)
        assert net.layers[0].weight.shape == (3072, 500)
        # Layer 1: PCNLayer(in_features=500, out_features=500) -> weight (500, 500)
        assert net.layers[1].weight.shape == (500, 500)
        # Layer 2: PCNLayer(in_features=10, out_features=500) -> weight (500, 10)
        assert net.layers[2].weight.shape == (500, 10)

    def test_from_dims_layer_weight_shapes(self) -> None:
        dims = [200, 100, 50, 10]
        net = PredictiveCodingNetwork(dims=dims)
        for idx in range(len(dims) - 1):
            expected_shape = (dims[idx], dims[idx + 1])
            assert net.layers[idx].weight.shape == expected_shape

    def test_from_layers_basic(self) -> None:
        layers = [PCNLayer(50, 100), PCNLayer(10, 50)]
        net = PredictiveCodingNetwork(layers=layers)
        assert len(net.layers) == 2
        assert net.layers[0].weight.shape == (100, 50)
        assert net.layers[1].weight.shape == (50, 10)

    def test_from_layers_validates_dimensions(self) -> None:
        # Layer 0: in_features=50, Layer 1: out_features=30 (mismatch: 50 != 30)
        layers = [PCNLayer(50, 100), PCNLayer(10, 30)]
        with pytest.raises(ValueError, match="Dimension mismatch"):
            PredictiveCodingNetwork(layers=layers)

    def test_both_dims_and_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            PredictiveCodingNetwork(dims=[100, 50], layers=[PCNLayer(50, 100)])

    def test_neither_dims_nor_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="Provide either"):
            PredictiveCodingNetwork()

    def test_dims_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            PredictiveCodingNetwork(dims=[100])

    def test_empty_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            PredictiveCodingNetwork(layers=[])

    def test_mode_classification(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 10])
        assert net.mode == "classification"

    def test_mode_regression(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 10], mode="regression")
        assert net.mode == "regression"

    def test_mode_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            PredictiveCodingNetwork(dims=[100, 10], mode="unsupported")

    def test_layers_exposed_publicly(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        assert isinstance(net.layers, nn.ModuleList)

    def test_readout_is_internal(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        assert isinstance(net._readout, nn.Linear)
        assert net._readout.bias is None

    def test_custom_output_dim(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10], output_dim=20)
        assert net._readout.weight.shape == (20, 10)

    def test_default_output_dim(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        # Default output_dim = dims[-1] = 10
        assert net._readout.weight.shape == (10, 10)

    def test_global_activation(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10], activation="tanh")
        # Verify layers use tanh by checking activation output
        x = torch.tensor([-1.0, 0.0, 1.0])
        expected = torch.tanh(x)
        pcn_layer: PCNLayer = net.layers[0]  # type: ignore[assignment]
        actual = pcn_layer.activation_fn(x)
        assert torch.allclose(actual, expected)


class TestLatentInitialization:
    """Test latent variable initialization."""

    def test_init_latents_shapes(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(8)
        assert len(net.latents) == 2
        assert net.latents[0].shape == (8, 50)
        assert net.latents[1].shape == (8, 10)

    def test_init_latents_count(self) -> None:
        dims = [200, 100, 50, 10]
        net = PredictiveCodingNetwork(dims=dims)
        net.init_latents(4)
        assert len(net.latents) == len(dims) - 1

    def test_init_latents_are_not_parameters(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(4)
        param_tensors = {p.data_ptr() for p in net.parameters()}
        for latent in net.latents:
            assert latent.data_ptr() not in param_tensors

    def test_init_latents_device(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(4)
        param_device = next(net.parameters()).device
        for latent in net.latents:
            assert latent.device == param_device

    def test_init_latents_stored_publicly(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(4)
        assert hasattr(net, "latents")
        assert isinstance(net.latents, list)
        assert len(net.latents) == 2

    def test_init_latents_resets_on_new_batch(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(4)
        old_latents = [lat.clone() for lat in net.latents]
        net.init_latents(8)
        assert len(net.latents) == 2
        assert net.latents[0].shape == (8, 50)
        # Should be different tensors
        assert net.latents[0].data_ptr() != old_latents[0].data_ptr()

    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    def test_init_latents_various_batch_sizes(self, batch_size: int) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(batch_size)
        assert net.latents[0].shape == (batch_size, 50)
        assert net.latents[1].shape == (batch_size, 10)


class TestPredict:
    """Test the predict() method."""

    def test_predict_shape(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(8)
        pred = net.predict()
        assert pred.shape == (8, 10)

    def test_predict_uses_readout(self) -> None:
        torch.manual_seed(42)
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(4)
        pred = net.predict()
        expected = net.latents[-1] @ net._readout.weight.T
        assert torch.allclose(pred, expected)

    def test_predict_without_init_raises(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        with pytest.raises(RuntimeError, match="Latents not initialized"):
            net.predict()

    def test_predict_custom_output_dim(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10], output_dim=20)
        net.init_latents(4)
        pred = net.predict()
        assert pred.shape == (4, 20)


class TestComputeErrors:
    """Test the compute_errors() method."""

    def test_compute_errors_unsupervised_shapes(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(8)
        result = net.compute_errors(torch.randn(8, 100))
        assert len(result.errors) == 2
        assert result.errors[0].shape == (8, 100)
        assert result.errors[1].shape == (8, 50)
        assert len(result.gm_errors) == 2
        assert result.gm_errors[0].shape == (8, 100)
        assert result.gm_errors[1].shape == (8, 50)

    def test_compute_errors_unsupervised_no_supervised(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(4)
        result = net.compute_errors(torch.randn(4, 100))
        assert result.supervised_error is None
        assert result.top_error is None

    def test_compute_errors_supervised_shapes(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(8)
        result = net.compute_errors(torch.randn(8, 100), torch.randn(8, 10))
        assert result.supervised_error is not None
        assert result.supervised_error.shape == (8, 10)
        assert result.top_error is not None
        assert result.top_error.shape == (8, 10)

    def test_compute_errors_supervised_custom_output_dim(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10], output_dim=20)
        net.init_latents(4)
        result = net.compute_errors(torch.randn(4, 100), torch.randn(4, 20))
        assert result.supervised_error is not None
        assert result.supervised_error.shape == (4, 20)
        assert result.top_error is not None
        assert result.top_error.shape == (4, 10)  # projected back to d_L=10

    def test_compute_errors_stored_on_network(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        assert net.errors is None
        net.init_latents(4)
        result = net.compute_errors(torch.randn(4, 100))
        assert net.errors is result

    def test_compute_errors_prediction_error_formula(self) -> None:
        """Verify eps^(l) = x^(l) - f(W^(l) @ x^(l+1)) manually."""
        torch.manual_seed(42)
        net = PredictiveCodingNetwork(dims=[20, 10, 5])
        net.init_latents(4)

        x = torch.randn(4, 20)
        result = net.compute_errors(x)

        # Manual computation for layer 0
        states = [x, *net.latents]
        layer0: PCNLayer = net.layers[0]  # type: ignore[assignment]
        preact0 = states[1] @ layer0.weight.T
        pred0 = layer0.activation_fn(preact0)
        expected_eps0 = states[0] - pred0
        assert torch.allclose(result.errors[0], expected_eps0)

        # Manual computation for layer 1
        layer1: PCNLayer = net.layers[1]  # type: ignore[assignment]
        preact1 = states[2] @ layer1.weight.T
        pred1 = layer1.activation_fn(preact1)
        expected_eps1 = states[1] - pred1
        assert torch.allclose(result.errors[1], expected_eps1)

    def test_compute_errors_gain_modulation_formula(self) -> None:
        """Verify h^(l) = f'(a^(l)) * eps^(l) manually."""
        torch.manual_seed(42)
        net = PredictiveCodingNetwork(dims=[20, 10, 5])
        net.init_latents(4)

        x = torch.randn(4, 20)
        result = net.compute_errors(x)

        # Manual computation for layer 0
        states = [x, *net.latents]
        layer0: PCNLayer = net.layers[0]  # type: ignore[assignment]
        preact0 = states[1] @ layer0.weight.T
        pred0 = layer0.activation_fn(preact0)
        eps0 = states[0] - pred0
        deriv0 = layer0.activation_deriv(preact0)
        expected_h0 = deriv0 * eps0
        assert torch.allclose(result.gm_errors[0], expected_h0)

    def test_compute_errors_supervised_error_formula(self) -> None:
        """Verify eps_sup = y_hat - y manually."""
        torch.manual_seed(42)
        net = PredictiveCodingNetwork(dims=[20, 10, 5])
        net.init_latents(4)

        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        result = net.compute_errors(x, y)

        y_hat = net.latents[-1] @ net._readout.weight.T
        expected_sup = y_hat - y
        assert result.supervised_error is not None
        assert torch.allclose(result.supervised_error, expected_sup)

    def test_compute_errors_top_error_formula(self) -> None:
        """Verify top_error = supervised_error @ W_out (batch W_out^T projection)."""
        torch.manual_seed(42)
        net = PredictiveCodingNetwork(dims=[20, 10, 5])
        net.init_latents(4)

        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        result = net.compute_errors(x, y)

        y_hat = net.latents[-1] @ net._readout.weight.T
        sup_err = y_hat - y
        expected_top = sup_err @ net._readout.weight
        assert result.top_error is not None
        assert torch.allclose(result.top_error, expected_top)

    def test_compute_errors_without_init_raises(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        with pytest.raises(RuntimeError, match="Latents not initialized"):
            net.compute_errors(torch.randn(4, 100))


class TestParameterRegistration:
    """Test that all parameters are properly registered."""

    def test_all_layer_params_in_model(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        all_params = set(net.parameters())
        for layer_module in net.layers:
            for param in layer_module.parameters():
                assert param in all_params

    def test_readout_params_in_model(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        all_params = set(net.parameters())
        assert net._readout.weight in all_params

    def test_state_dict_has_layers_and_readout(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        sd = net.state_dict()
        assert "layers.0.weight" in sd
        assert "layers.1.weight" in sd
        assert "_readout.weight" in sd

    def test_latents_not_in_state_dict(self) -> None:
        net = PredictiveCodingNetwork(dims=[100, 50, 10])
        net.init_latents(4)
        sd = net.state_dict()
        assert not any("latent" in key for key in sd)
