---
status: complete
phase: 02-core-model
source: [02-01-SUMMARY.md]
started: 2026-02-24T01:00:00Z
updated: 2026-02-24T01:15:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Public API Imports
expected: `from pcn_torch import PredictiveCodingNetwork, PCNErrors` succeeds without errors
result: pass

### 2. Network Construction from Dims
expected: `PredictiveCodingNetwork(dims=[3072, 500, 500, 10])` creates a network with 3 layers. `len(net.layers) == 3` and layers are accessible as `net.layers`.
result: pass

### 3. Latent Initialization
expected: After `net.init_latents(batch_size=4)`, `net.latents` contains 3 tensors with shapes (4,500), (4,500), (4,10) respectively. Latents are random (different values each call).
result: pass

### 4. Error Computation (Unsupervised)
expected: `net.compute_errors(x)` with x of shape (4, 3072) returns a PCNErrors with `.errors` (3 tensors) and `.gm_errors` (3 tensors), while `.supervised_error` and `.top_error` are None.
result: pass

### 5. Error Computation (Supervised)
expected: `net.compute_errors(x, y)` with y of shape (4, 10) returns PCNErrors with all four fields populated. `.supervised_error` has shape (4, 10) and `.top_error` has shape (4, 10).
result: pass

### 6. Readout Prediction
expected: `net.predict()` returns a tensor of shape (4, 10) representing the output prediction from the top latent via the internal readout layer.
result: pass

### 7. Network Construction from Pre-built Layers
expected: Building PCNLayer instances manually and passing them via `PredictiveCodingNetwork(layers=[...])` constructs a valid network. Passing layers with mismatched dimensions raises ValueError.
result: pass

## Summary

total: 7
passed: 7
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
