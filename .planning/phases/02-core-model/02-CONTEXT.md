# Phase 2: Core Model - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Construct a full PredictiveCodingNetwork from configurable layers, with latent variable management, error computation, and readout. The network produces predictions and error signals. Training loops, energy tracking, and tests are Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Network construction API
- Accept both a dims list (convenience) and pre-built PCNLayer instances (full control)
- Global activation function with per-layer override capability
- Layers exposed publicly as `network.layers`

### Latent initialization
- Latents stored on the network as `network.latents` (public attribute, inspectable)
- Auto-reset per batch (paper-defined behavior)
- Paper-prescribed initialization only — no custom init function

### Error computation interface
- `compute_errors(x, y=None)` — pass input and optional target explicitly each call
- Returns errors for all layers at once (not per-layer methods)
- Errors stored on the network after computation (like latents) for inspection/debugging
- When `y` is provided, supervised error (target vs prediction projected via W_out^T) is computed and included

### Readout layer
- Readout is internal (not a public attribute)
- Separate `predict()` method for getting output predictions
- Explicit mode flag for classification vs regression — researcher to check the paper for how these differ (loss function, target handling)

### Claude's Discretion
- Return type for `compute_errors` (named tuple, dataclass, or similar structured type)
- Gain modulation implementation (follows paper formula exactly)
- Supervised error formula (follows paper exactly)
- Exact initialization distribution for latents (follows paper)

</decisions>

<specifics>
## Specific Ideas

- All math (gain formula, error computation, supervised error projection, latent init distribution) follows arXiv:2506.06332v1 exactly — no deviations
- API should be consistent: latents and errors both stored on network as public attributes for inspection
- Researcher must check paper for classification vs regression differences before planning

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-core-model*
*Context gathered: 2026-02-20*
