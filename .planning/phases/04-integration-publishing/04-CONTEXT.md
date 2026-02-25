# Phase 4: Integration + Publishing - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Make pcn-torch installable from PyPI, provide a working CIFAR-10 example that demonstrates the library end-to-end, and write a README that explains predictive coding and gets people started. All core functionality (layers, network, training, energy) is already built in Phases 1-3.

</domain>

<decisions>
## Implementation Decisions

### CIFAR-10 example behavior
- Fixed configuration, no CLI arguments — user runs `python examples/cifar10.py` and it trains
- Target 5-10 minutes on CPU — full training run with meaningful accuracy results
- Use RichCallback for live progress (progress bars, accuracy, energy per epoch)
- Print full summary after training: final test accuracy, total training time, energy convergence summary

### README depth & structure
- Tone: academic but pedagogical + practical — explain the theory accessibly, then show how to use it
- Include a brief explainer section (3-5 paragraphs) on what predictive coding networks are and how they differ from backprop, with link to the paper
- Quickstart code snippet + API overview table listing main classes/functions with brief descriptions
- Include a "Results" section with expected CIFAR-10 accuracy numbers and training metrics

### Public API surface
- `pcn_torch.__init__` exports everything useful: network, training functions, layers, types, energy, callbacks
- `__all__` defined explicitly for IDE autocompletion and clean public API
- Example script imports only what it needs (minimal imports for clarity)
- Submodule import paths: Claude's discretion

### Package & distribution
- PyTorch is a peer dependency — NOT in install_requires. README instructs "install PyTorch first"
- License: MIT
- Author: Enrique Mendez
- Version: v1.0.0 (not 0.1.0 — user wants to start at 1.0.0)
- Prepare for publishing only — build and verify with TestPyPI, real publish is a manual step after
- PyPI classifiers, keywords, project URLs: Claude's discretion

### Claude's Discretion
- Submodule import strategy (whether `from pcn_torch.energy import ...` also works alongside top-level)
- CIFAR-10 network architecture and hyperparameters (whatever achieves reasonable accuracy in 5-10 min)
- Exact README section ordering and formatting
- PyPI classifiers, keywords, and project URLs
- Loading skeleton / error handling in example

</decisions>

<specifics>
## Specific Ideas

- README tone described as "academic but pedagogical" — explain the theory in a way that teaches, not just references
- User wants version number visible (v1.0.0) — this is a confident first release, not a tentative 0.x
- RichCallback already built in Phase 3 should be showcased in the example

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-integration-publishing*
*Context gathered: 2026-02-25*
