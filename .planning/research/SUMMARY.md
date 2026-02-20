# Research Summary: pcn-torch

**Project:** pcn-torch
**Domain:** PyTorch ML research library (PyPI package)
**Researched:** 2026-02-20
**Confidence:** HIGH

---

## Executive Summary

`pcn-torch` is a first-of-its-kind pip-installable PyTorch library implementing Predictive Coding Networks (PCNs) from arXiv:2506.06332v1 (Stenlund, 2025). The competitive landscape has no PyPI-published PyTorch PCN library — every existing implementation is either a Jupyter notebook or requires cloning a repo. This means the bar for differentiation is low and the opportunity to become the canonical reference implementation is immediately available. The recommended approach is to build a small, clean, well-tested library with 6 source modules, full type hints, and energy trajectory tracking as its primary diagnostic feature. The model/trainer separation (network.py vs trainer.py) and explicit no-autograd constraint are the two architectural decisions that most distinguish this library from standard PyTorch work.

The stack is unambiguously modern Python 2026: `uv` for project management, `hatchling` + `hatch-vcs` for packaging, `Ruff` for linting, `mypy` for type checking, `pytest` for testing, and GitHub Actions with OIDC-based PyPI publishing. All tools are at stable current versions and represent community consensus for open-source Python libraries as of early 2026. No meaningful stack alternatives need evaluation — the choices are settled.

The dominant risk is implementation correctness, not tooling or packaging. Three algorithmic bugs — asynchronous latent updates, accidental autograd graph construction, and incorrect top-layer supervised error signal — each silently corrupt results or cause OOM crashes and would destroy credibility on first use. These must be caught with targeted unit tests before any experiment is run. The secondary risk is PyTorch's complicated PyPI distribution story: users must install PyTorch independently with their preferred CUDA variant, and the README must state this clearly.

---

## Key Findings

### Recommended Stack

The stack is fully determined with high confidence. Python >=3.10, PyTorch >=2.0, `uv` as the project manager, `hatchling` + `hatch-vcs` as the build backend, and `Ruff` + `mypy` + `pytest` as the quality toolchain. The build configuration lives entirely in a single `pyproject.toml` with a `src/` layout. Publishing to PyPI uses trusted publishers (OIDC, no API tokens). CI runs a Python 3.10-3.13 matrix via GitHub Actions. Documentation uses MkDocs-Material with `mkdocstrings[python]` for API auto-generation.

**Core technologies:**
- **PyTorch >=2.0**: Peer dependency. Library operates entirely under `torch.no_grad()` — no autograd use. nn.Module inheritance is used for parameter registration and serialization only.
- **uv >=0.10**: Project and environment management. Replaces pip, virtualenv, and pip-tools. 10-100x faster than pip; lockfile-based reproducible CI.
- **hatchling >=1.28 + hatch-vcs >=0.4**: Build backend with git-tag-based automatic versioning. Zero boilerplate beyond pyproject.toml.
- **Ruff >=0.15**: Single tool replacing Black, Flake8, isort, and pyupgrade. Configured for Python 3.10 target.
- **mypy >=1.19**: Full static typing with `disallow_untyped_defs = true`. All public APIs must have type annotations.
- **pytest >=9.0 + pytest-cov >=7.0**: Test runner with branch coverage. CPU-only PyTorch in CI is sufficient for v1 (no GPU needed for MLP-only tests).
- **MkDocs-Material >=9.7**: Documentation site. Google-style docstrings auto-generated as API reference. Note: team pivots to Zensical in late 2026 but the library is stable and production-ready now.

**Key constraint**: Declare `torch>=2.0` as a loose dependency. Do NOT pin a specific CUDA variant — users install PyTorch first, then `pip install pcn-torch`.

### Expected Features

The paper's reference notebook (Monadillo/pcn-intro) exists but is not a library. `pcn-torch` extracts that work into a properly packaged, pip-installable library. The feature bar is low for table stakes (every competitor fails on basics) but the differentiators are where this library earns its place.

**Must have (table stakes) — all 16 identified:**
- `pip install pcn-torch` on PyPI — no existing PCN library is pip-installable
- `nn.Module` subclasses (PCNLayer, PredictiveCodingNetwork) — non-negotiable for PyTorch ecosystem
- GPU/CPU device transparency via `.to(device)`
- Configurable architecture (dims, output_dim, activation_fn, activation_deriv)
- Configurable PCN hyperparameters (eta_infer, eta_learn, T_infer, T_learn)
- Batch training support (vectorized row-batch form per Algorithm 3)
- Xavier weight initialization (matching the paper exactly)
- Latent initialization N(0,1), freshly per forward pass
- Serialization via `state_dict()` / `load_state_dict()`
- Reproducibility utility (`set_seed()`)
- Mixed-precision support via `torch.autocast`
- Unit tests (pytest, 80%+ coverage target)
- Minimal working example (CIFAR-10 from the paper)
- Full type hints on all public APIs
- Google-style docstrings
- MIT license

**Should have (competitive differentiators):**
- Energy trajectory tracking — per-step, per-batch energy during both phases; first-class feature, not print statements
- Per-layer energy decomposition — diagnostic tool for identifying which layers struggle
- Inference convergence callbacks — early stopping on energy delta or latent update norm
- Separation of inference and learning APIs (`model.infer()` / `model.learn()`) — composable, inspectable
- Prediction/test-time inference — clean `model.predict(x)` with frozen weights
- Custom activation function support with explicit derivatives
- Comprehensive Trainer (`train_pcn()`) with logging integration
- Classification + regression support explicitly (one-hot vs raw targets)
- Latent state inspection methods
- Numerical stability safeguards (NaN/Inf detection, gradient clipping on latents)
- Pretrained weights registry for CIFAR-10 (`pcn_torch.load_pretrained('cifar10')`)
- Comprehensive documentation site

**Defer to v2+:**
- Convolutional / Recurrent / Transformer layer types
- Unsupervised learning mode
- Hybrid predictive coding (amortized inference)
- Distributed training / multi-GPU
- Automatic hyperparameter tuning
- Bias terms in layers (paper explicitly specifies no bias)
- Spiking neural network support

**Hard anti-features (never build):**
- Autograd-based weight updates — defeats PCN's local computation principle
- Standard PyTorch optimizers (Adam, SGD) — PCN uses its own Hebbian-like local update rules
- Built-in data loading — use PyTorch DataLoader directly

### Architecture Approach

The library has 6 source modules in a flat structure under `src/pcn_torch/`. The dependency graph is acyclic and linear: `_types.py` -> `activations.py` -> `layers.py` -> `network.py` -> `trainer.py`, with `energy.py` as a lateral utility consumed by `trainer.py`. This structure enables phased development and isolated unit testing of each component before the next is built.

The critical architectural invariant is the separation of model definition (`network.py`) from training logic (`trainer.py`). The training loop is not a method on the network — it is an external function `train_pcn()`. This follows the Lightning philosophy but does not depend on Lightning (the PCN training loop is too different from standard PyTorch training). All training operations run under `torch.no_grad()` with weight updates applied via direct tensor mutation, not through `torch.optim`.

**Major components:**
1. `_types.py` — Type aliases (ActivationFn, ActivationDeriv), shared constants (DEFAULT_T_INFER=50)
2. `activations.py` — Paired (activation, derivative) callables. Explicit hand-coded derivatives, not autograd. Lookup by name.
3. `layers.py` — `PCNLayer(nn.Module)`: weight matrix W, no bias. `forward(x_above)` returns `(x_hat, preactivation)`.
4. `network.py` — `PredictiveCodingNetwork(nn.Module)`: ModuleList of PCNLayers + nn.Linear readout (bias=False). `init_latents()`, `compute_errors()`. Does NOT contain training logic.
5. `energy.py` — `compute_energy()` and `compute_energy_per_layer()`. Used for monitoring, not gradient computation.
6. `trainer.py` — `train_pcn()` and `test_pcn()`. Two-phase loop (inference then learning). All updates under `torch.no_grad()`. Direct tensor mutation for weight updates.

**Three data flows are fully specified:**
- Inference phase: errors computed as snapshot, then latents updated (synchronous — critical invariant)
- Learning phase: latents frozen, weights updated from batch-averaged local gradients
- Prediction (test time): frozen weights, inference with E^(L)=0 for unknown targets

### Critical Pitfalls

The pitfalls research identified 18 pitfalls across algorithm, packaging, and numerical categories. The top 5 ranked by impact:

1. **Asynchronous latent updates** — The synchronous update scheme (compute ALL errors first, then update ALL latents) must be implemented as two explicit phases. Implementing as compute-and-update-per-layer produces wrong results that still train, making the bug hard to detect. Prevention: `compute_errors()` completes before any `x[l] -= eta * grad` executes. Validate with a unit test that checks update independence.

2. **Accidental autograd graph construction** — Any operation missing `torch.no_grad()` silently builds a computation graph. With T_infer=50 + T_learn=B steps, this exhausts GPU memory in a single batch on the 3.6M parameter CIFAR-10 model. Prevention: wrap both loops in `with torch.no_grad()`, assert `requires_grad=False` on latents, monitor `torch.cuda.memory_allocated()` during dev.

3. **Wrong top-layer error signal** — The supervised error must be projected back to latent space: `eps^(L) = eps_sup @ W_out`. Using `eps_sup` directly (wrong shape) or leaving `eps^(L) = 0` means the supervised signal never reaches the hierarchy. Prevention: explicit shape assertion `assert eps_L.shape == (B, d_L)`, test that `x^(L)` changes direction correctly after one supervised inference step.

4. **PyTorch CUDA vs CPU on PyPI** — `pip install pcn-torch` will install CPU-only PyTorch by default. Users wanting GPU must install their CUDA-specific PyTorch wheel first. Prevention: declare `torch>=2.0` with no CUDA pinning; README must say "Install PyTorch first."

5. **Incorrect batch averaging** — Missing `1/B` in weight gradient `grad_W = -(1/B) * H^T @ X` makes hyperparameters batch-size-dependent, breaking reproducibility of the paper's reported CIFAR-10 results. Prevention: apply `/B` only in the weight update, not inference update; unit test with duplicated data at two batch sizes.

**Additional pitfalls to address during implementation:**
- Readout layer must use `nn.Linear(..., bias=False)` — paper specifies no bias; default `bias=True` introduces a frozen random offset
- `T_learn` must default to `B` (batch size), not a fixed constant — the paper's per-sample semantics require this
- Mixed-precision energy monitoring must cast to float32 before squaring to avoid float16 overflow
- In-place weight updates: use `layer.W.data -= eta * grad` or wrap in `no_grad` — verify with post-step weight-change assertion

---

## Cross-Cutting Themes

Three themes appear consistently across all four research dimensions:

**1. "No autograd" is the library's core identity.** Every design decision downstream flows from the fact that PCN training is local, Hebbian, and explicitly gradient-free. The stack choice (no optimizer), architecture (trainer.py separate from network.py), features (explicit activation derivatives), and pitfalls (Pitfall 2) all converge on this. Any future feature addition must be evaluated against this principle first.

**2. Reproducibility is the primary credibility signal.** The paper reports 99.92% on CIFAR-10. The library's value proposition is being the reproducible reference implementation of that result. This drives: Xavier initialization (matching paper, not He), `T_learn=B` default (matching paper semantics), batch-averaged gradients (matching paper formula), and the pretrained weights registry. Researchers will validate the library by running the CIFAR-10 example first.

**3. The diagnostic tooling is the key differentiator.** No existing PCN library provides energy trajectory tracking, per-layer energy decomposition, or inference convergence callbacks. These features are explicitly motivated by the paper (energy trajectories as "sanity checks") and address a real gap in the competitive landscape. They are also architecturally central: `energy.py` as a first-class module (not embedded in the trainer) reflects this priority. This theme influences both the API design and the phase ordering.

---

## Key Findings

### Stack
The three most consequential stack decisions:

1. **`uv` over pip/Poetry**: Faster CI, unified virtual environment management, and lockfile-based reproducibility. The dependency pin strategy is critical: loose lower bounds in `[project.dependencies]` for torch/numpy (user installs their preferred build), exact pins in `uv.lock` for dev dependencies.

2. **`hatchling` + `hatch-vcs`**: Eliminates manual version bumping. All versions derive from git tags. Start at `0.1.0` to signal pre-stable API. The `src/pcn_torch/` layout prevents accidental source-tree imports during testing.

3. **`mypy` with `disallow_untyped_defs = true`**: Full type enforcement from day one. The library's small size (6 modules) makes this feasible without significant friction, and it significantly improves the user experience via IDE autocompletion.

### Features
The critical path to MVP in priority order:
1. `PCNLayer` and `PredictiveCodingNetwork` (nn.Module subclasses)
2. Configurable architecture + hyperparameters
3. Vectorized batch training support
4. Energy trajectory tracking (makes this library unique)
5. pip-installable package
6. CIFAR-10 example (the credibility proof)
7. Unit tests (80%+ coverage)
8. Type hints + docstrings

### Architecture
Build order follows the dependency chain exactly:
- Phase 1: `_types.py` -> `activations.py` (no dependencies, testable in isolation)
- Phase 2: `layers.py` -> `network.py` (depend on Phase 1)
- Phase 3: `energy.py` -> `trainer.py` (depend on Phase 2)
- Phase 4: `__init__.py`, `pyproject.toml`, `examples/`, `tests/`

### Pitfalls
The algorithmic pitfalls (1, 2, 3, 5) must be prevented with unit tests before any experiment runs. The packaging pitfalls (7, 8) must be checked before the first PyPI publish.

---

## Implications for Roadmap

Based on combined research, a 4-phase roadmap is recommended. The build order from ARCHITECTURE.md directly maps to phase structure.

### Phase 1: Foundation (Types, Activations, Package Scaffold)
**Rationale:** Zero logic risk. No external dependencies beyond PyTorch. Establishes the project structure and quality baseline before any algorithm is written. Catching type alias and activation derivative bugs early prevents cascading errors in later phases.
**Delivers:** `_types.py`, `activations.py`, `pyproject.toml` scaffold, CI pipeline (Ruff, mypy, pytest), pre-commit hooks, `src/` layout, `uv.lock`.
**Addresses:** Table stakes (type hints, pip-installable structure), anti-pitfall (establish `no_grad` convention in code style from day one).
**Avoids:** Pitfall 9 (missing metadata), Pitfall 10 (version numbering), Pitfall 8 (PyPI name — check availability before writing any code).
**Research flag:** No deeper research needed. Standard Python packaging patterns; well-documented.

### Phase 2: Core Model (PCNLayer, PredictiveCodingNetwork)
**Rationale:** These are the algorithm-critical components where the most dangerous pitfalls live. Implementing and testing them in isolation — before the training loop exists — makes it possible to verify correctness at the unit level before integration.
**Delivers:** `layers.py` (PCNLayer with Xavier init, no bias), `network.py` (PredictiveCodingNetwork with nn.ModuleList, `init_latents()`, `compute_errors()`), `tests/test_layers.py`, `tests/test_network.py`.
**Addresses:** Table stakes (nn.Module subclasses, configurable architecture, Xavier init, batch form), differentiators (separation of inference/learning APIs at the model level).
**Avoids:** Pitfall 3 (gain-modulated error: must use preactivation `a`, not latent `x`), Pitfall 6 (readout `nn.Linear(..., bias=False)`), Pitfall 4 (in-place weight updates — establish `layer.W.data -= eta * grad` pattern here).
**Research flag:** No deeper research needed. Paper provides complete algorithmic specification.

### Phase 3: Training Loop + Energy (trainer.py, energy.py)
**Rationale:** This is where the most complex algorithmic correctness requirements converge. The two-phase training loop (inference then learning) must implement the synchronous update scheme. Energy tracking is built in from the start (not bolted on later) because it is a primary differentiator.
**Delivers:** `energy.py` (`compute_energy()`, `compute_energy_per_layer()`), `trainer.py` (`train_pcn()`, `test_pcn()`), `tests/test_trainer.py`, `tests/test_energy.py`. Initial energy trajectory logging.
**Addresses:** Table stakes (batch training, mixed precision, serialization), differentiators (energy trajectory tracking, per-layer energy decomposition, inference convergence callbacks, numerical stability safeguards, `T_learn=B` default).
**Avoids:** Pitfall 1 (synchronous updates — `compute_errors()` completes before any latent update), Pitfall 2 (`torch.no_grad()` wrapping both loops), Pitfall 5 (`eps^(L) = eps_sup @ W_out` with shape assertion), Pitfall 16 (`1/B` in weight gradient only), Pitfall 17 (`T_learn=B` default).
**Research flag:** No deeper research needed — paper provides complete pseudocode (Algorithms 2-4). However, the test for Pitfall 1 (asynchronous update detection) is non-trivial to write and may need iteration.

### Phase 4: Integration, Packaging, and Documentation
**Rationale:** Polish and credibility. The CIFAR-10 example is the primary trust signal for potential adopters. Complete packaging with PyPI publishing establishes the library as a proper first-class package, not a research artifact.
**Delivers:** `__init__.py` (public API surface), `examples/cifar10.py` (reproducing 99.92% paper result), full documentation site (MkDocs-Material + mkdocstrings), pretrained weights on GitHub Releases, PyPI publish (v0.1.0), `set_seed()` reproducibility utility, README with PyTorch installation instructions.
**Addresses:** All remaining table stakes (minimal working examples, docstrings, license, pip-installable), differentiators (comprehensive docs, pretrained weights registry, logging integration entry points).
**Avoids:** Pitfall 7 (PyTorch PyPI dependency — README installation instructions), Pitfall 11 (weights on GitHub Releases, not in the package), Pitfall 9 (complete metadata validation before publish).
**Research flag:** No deeper research needed for packaging mechanics. The CIFAR-10 example requires access to the paper's hyperparameters (available in arXiv:2506.06332v1) and the pretrained weights from the Monadillo/pcn-intro notebook.

### Phase Ordering Rationale

- **Phases 1-3 follow the dependency graph exactly.** No module in a later phase can be implemented until its dependencies in earlier phases are complete and tested. This is not arbitrary — it is forced by the acyclic dependency chain `_types -> activations -> layers -> network -> trainer`.
- **Energy tracking (Phase 3) is promoted over packaging (Phase 4)** because it is a first-class architectural concern, not documentation. Building it late would require retrofitting it into `trainer.py`.
- **Pitfalls 1, 2, and 5 must be validated in Phase 3** before the CIFAR-10 experiment in Phase 4. Attempting to run the experiment with a buggy training loop would waste time and erode confidence.
- **PyPI name reservation** (checking availability at `https://pypi.org/project/pcn-torch/`) should happen before Phase 1 begins, not in Phase 4.

### Research Flags

Phases needing deeper research during planning: None. All four phases implement well-specified algorithms from the paper (arXiv:2506.06332v1) using established tooling patterns.

Phases with standard patterns (no research-phase needed):
- **Phase 1:** Python packaging with uv/hatchling is fully documented
- **Phase 2:** Algorithm is completely specified in the paper's pseudocode
- **Phase 3:** Algorithm specified in paper; pitfall prevention patterns are clear
- **Phase 4:** MkDocs-Material documentation and PyPI publishing are standard

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All tools at current stable versions (Feb 2026). Community consensus is clear. uv, Ruff, hatchling are the settled 2025-2026 standard. |
| Features | HIGH | Based on the paper directly (arXiv:2506.06332v1) and competitive gap analysis of 6 existing implementations. Table stakes and differentiators are well-defined. |
| Architecture | HIGH | Paper provides complete algorithmic pseudocode (Algorithms 1-4). Component boundaries follow natural separation of concerns. No ambiguity in data flow. |
| Pitfalls | HIGH | 18 pitfalls identified with concrete prevention strategies and warning signs. Top 5 are well-understood failure modes with clear mitigations. |

**Overall confidence:** HIGH

### Gaps to Address

Three minor gaps that may need attention during implementation:

- **Test-time inference with unknown targets**: The paper describes running inference with E^(L)=0 when the target is unknown (pure prediction mode), but the reference notebook only shows the supervised mode. The API design for `model.predict(x)` vs `model.test(x, y)` needs to be explicit in Phase 3 and validated before Phase 4's example.
- **Pretrained weights access**: The paper reports 99.92% CIFAR-10 accuracy. The Monadillo/pcn-intro notebook has pretrained weights. These weights need to be extracted and hosted on GitHub Releases during Phase 4. If the weights are not directly accessible, the CIFAR-10 example must train from scratch, which requires GPU resources to validate.
- **T_learn=B for large batches**: Defaulting T_learn=B means 500 learning steps at batch size 500. This is the paper's recommendation but creates a non-obvious performance characteristic (training time scales as O(B^2) per epoch for T_learn=B). The API should expose this clearly and the documentation should explain the tradeoff.

---

## Sources

### Primary (HIGH confidence)
- [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332) — Paper by Stenlund (2025). Algorithms 1-4, hyperparameters, CIFAR-10 results. Primary reference for all implementation decisions.
- [Monadillo/pcn-intro](https://github.com/Monadillo/pcn-intro) — Reference notebook. Complete CIFAR-10 example, pretrained weights.
- [PyTorch 2.10.0 docs](https://pytorch.org/docs/stable/) — nn.Module, torch.no_grad, autocast API.
- [uv documentation](https://docs.astral.sh/uv/) — Project management, lockfiles, publishing.
- [Python Packaging User Guide](https://packaging.python.org/en/latest/) — pyproject.toml, trusted publishers, build backends.
- [Ruff documentation](https://docs.astral.sh/ruff/) — Linter and formatter configuration.
- [hatchling/hatch-vcs](https://hatch.pypa.io/latest/) — Build backend and git-based versioning.

### Secondary (MEDIUM confidence)
- [PRECO](https://github.com/bjornvz/PRECO) — Competitive analysis. PyTorch PCN library, not pip-installable.
- [Torch2PC](https://github.com/RobertRosenbaum/Torch2PC) — Competitive analysis. Single-function API, Sequential-only.
- [JPC (JAX)](https://arxiv.org/html/2412.03676v1) — Most sophisticated competitor; JAX ecosystem, not PyTorch.
- [arXiv:2407.04117](https://arxiv.org/abs/2407.04117) — Survey on predictive coding; context for PRECO.
- "Towards Stable Learning in Predictive Coding Networks" (Alonso et al.) — Source for Pitfall 14 (PE concentration in deep networks).
- [Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/packaging-simple/) — Packaging conventions for scientific libraries.
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/) — OIDC-based publishing setup.

### Tertiary (context only)
- lucidrains package patterns (nGPT-pytorch, x-transformers) — Architecture pattern reference for minimal research packages.
- "Dying ReLU and Initialization" (Lu et al. 2019) — Background for Pitfall 12-13 (Xavier vs He, dead neurons).
- MkDocs-Material future transition to Zensical — Note for Phase 4; low urgency, library is stable now.

---
*Research completed: 2026-02-20*
*Ready for roadmap: yes*
