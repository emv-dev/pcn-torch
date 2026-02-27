---
phase: 02-core-model
verified: 2026-02-23T00:00:00Z
status: passed
score: 5/5 must-haves verified
---

# Phase 2: Core Model Verification Report

**Phase Goal:** Developers can construct a full PredictiveCodingNetwork with configurable layers and obtain predictions and error signals
**Verified:** 2026-02-23
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | PredictiveCodingNetwork(dims=[3072,500,500,10]) constructs a valid network with 3 layers | VERIFIED | len(net.layers)==3; weight shapes [(3072,500),(500,500),(500,10)] confirmed by test and live execution |
| 2 | init_latents(batch_size) produces randomly initialized latent variables at each layer | VERIFIED | 3 latents with shapes [(8,500),(8,500),(8,10)]; plain tensors not nn.Parameters |
| 3 | compute_errors() returns prediction errors and gain-modulated errors with correct shapes | VERIFIED | 3 errors and 3 gm_errors returned; shapes match layer dims; formula verified by unit tests and live check |
| 4 | Readout layer is nn.Linear(..., bias=False) mapping top latent to output predictions | VERIFIED | net._readout is nn.Linear(10,10,bias=False); bias is None confirmed in code and test |
| 5 | Supervised error signal correctly projected back to latent space via W_out^T | VERIFIED | top_error = supervised_error @ self._readout.weight (line 206); torch.allclose passes |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/pcn_torch/network.py | PredictiveCodingNetwork class | VERIFIED | 216 lines, fully implemented; no stubs; exports PredictiveCodingNetwork and PCNErrors |
| src/pcn_torch/layers.py | PCNLayer (used by network) | VERIFIED | 55 lines; nn.Module subclass; no bias; xavier init; used by network.py constructor |
| src/pcn_torch/__init__.py | Public API exports | VERIFIED | Exports PredictiveCodingNetwork, PCNErrors, PCNLayer, get_activation, type aliases |
| tests/test_network.py | Tests covering all must-haves | VERIFIED | 374 lines; 47 tests covering construction, latents, predict, compute_errors, param registration |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| PredictiveCodingNetwork | PCNLayer | nn.ModuleList in __init__ | WIRED | self.layers = nn.ModuleList(built_layers) (line 121) |
| compute_errors | PCNLayer.forward | loop calling pcn_layer(states[...]) | WIRED | Lines 191-197; prediction and preactivation used for eps and h |
| compute_errors | predict() / _readout | self.predict() inside method | WIRED | Lines 203-206; y_hat computed, supervised_error and top_error returned |
| supervised_error | W_out^T projection | supervised_error @ self._readout.weight | WIRED | Line 206; weight matrix used directly (W_out^T = weight for Linear) |
| __init__.py | network.py | from pcn_torch.network import ... | WIRED | PCNErrors and PredictiveCodingNetwork in __all__ |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| MOD-02 | SATISFIED | Network construction from dims list implemented and tested (TestNetworkConstruction) |
| MOD-03 | SATISFIED | init_latents(batch_size) creates N(0,1) latents for layers 1..L (TestLatentInitialization) |
| MOD-04 | SATISFIED | compute_errors() computes eps and h for all layers; supervised path with y (TestComputeErrors) |
| MOD-05 | SATISFIED | Readout is nn.Linear(d_top, output_dim, bias=False); W_out^T projection verified |

### Anti-Patterns Found

None. No TODOs, FIXMEs, placeholder content, empty returns, or stub patterns found in any verified files.

### Human Verification Required

None. All goal criteria are verifiable programmatically and confirmed by the test suite.

## Test Suite Results

79 tests collected. 79 passed. 0 failed. 0 skipped. Runtime: 2.81s.

Relevant test classes in tests/test_network.py:
- TestNetworkConstruction (13 tests): covers must-have 1 including exact dims=[3072,500,500,10] case
- TestLatentInitialization (9 tests): covers must-have 2 across multiple batch sizes
- TestComputeErrors (10 tests): covers must-have 3; formula verified via torch.allclose
- TestPredict (4 tests): covers must-have 4; confirms readout shape and weight usage
- TestParameterRegistration (4 tests): confirms readout is registered in model parameters

## Must-Have Detail

### MH-1: Network construction with dims=[3072, 500, 500, 10]

Verified in src/pcn_torch/network.py lines 83-96.

For dims=[3072, 500, 500, 10] the constructor builds 3 PCNLayer instances with weight
shapes (3072,500), (500,500), (500,10). len(dims)-1 == 3 layers. Confirmed by
test_from_dims_cifar10 and live execution producing len(net.layers)==3 with correct shapes.

### MH-2: init_latents(batch_size) initializes random latents at each layer

Verified in src/pcn_torch/network.py lines 132-145.

Creates one latent per non-input layer using torch.randn(batch_size, self._dims[idx]).
Iterates idx from 1 to len(self._dims)-1 inclusive. Latents are plain tensors,
not nn.Parameters. For dims=[3072,500,500,10] and batch_size=8: 3 latents with
shapes (8,500), (8,500), (8,10).

### MH-3: compute_errors() returns errors and gm_errors with correct shapes

Verified in src/pcn_torch/network.py lines 162-215.

Prediction error: eps = states[idx] - prediction
  (implements eps^(l) = x^(l) - f(W^(l) x^(l+1)) from arXiv:2506.06332v1).
Gain-modulated error: h = activation_deriv(preactivation) * eps
  (implements h^(l) = f_prime(a^(l)) * eps^(l)).
PCNErrors NamedTuple holds errors, gm_errors, supervised_error, top_error.
test_compute_errors_prediction_error_formula and test_compute_errors_gain_modulation_formula
verify math via torch.allclose against manual computation.

### MH-4: Readout is nn.Linear(..., bias=False)

Verified in src/pcn_torch/network.py line 126.

    self._readout = nn.Linear(d_top, effective_output_dim, bias=False)

Where d_top = self._dims[-1] (top latent dimension). For dims=[3072,500,500,10]:
nn.Linear(10, 10, bias=False). test_readout_is_internal confirms bias is None.

### MH-5: Supervised error projected back via W_out^T

Verified in src/pcn_torch/network.py lines 202-206.

    top_error = supervised_error @ self._readout.weight

nn.Linear stores weight as shape (out_features, in_features) = (d_out, d_L).
Multiplying (B, d_out) @ (d_out, d_L) gives (B, d_L), the W_out^T projection.
test_compute_errors_top_error_formula confirms via torch.allclose against manual
computation: expected_top = sup_err @ net._readout.weight.

## Gaps Summary

No gaps. All 5 must-haves are fully implemented, substantive, and wired. The test suite
provides 47 tests in test_network.py covering construction, latent initialization,
prediction, error computation, and parameter registration. All 79 tests pass with no
failures or skips.

---

_Verified: 2026-02-23_
_Verifier: Claude (gsd-verifier)_
