# Pitfalls Research: pcn-torch

## Implementation Pitfalls

### Pitfall 1: Asynchronous Latent Updates Breaking Convergence

- **Risk**: During the inference loop, if latent variables are updated one-at-a-time and each subsequent update reads the already-modified state of previously updated layers, the algorithm becomes asynchronous. This violates the snapshot-based synchronous update scheme specified in the paper (Algorithm 3, lines 4-11 vs 12-16). Asynchronous updates mean each latent is updated against a partially-modified energy landscape, which can prevent convergence or produce subtly wrong results that appear to train but underperform. This is the single most likely implementation bug because it is natural to compute-and-update in a single pass.
- **Warning Signs**: (1) Energy trajectories that oscillate or fail to monotonically decrease even with very small inference rates. (2) Final accuracy significantly below the paper's reported 99.92% on CIFAR-10 with identical hyperparameters. (3) Results that vary wildly depending on layer ordering.
- **Prevention**: Implement inference in two explicit phases per step: (a) compute ALL errors and gain-modulated errors for every layer using current state, storing them; (b) THEN update all latent variables using the stored snapshot. In code, this means `compute_errors()` must complete before any `x[l] -= eta * grad` executes. Write a unit test that verifies the inference update for layer `l` uses pre-update values of all other layers.
- **Phase**: Core implementation (inference loop). Must be correct from day one -- this is the algorithm's correctness invariant.

### Pitfall 2: Accidentally Enabling Autograd During Inference/Learning

- **Risk**: The entire PCN training loop operates under `torch.no_grad()`. If any operation inadvertently creates a computation graph (e.g., forgetting the context manager, calling a method that internally enables gradients, or creating a new tensor with `requires_grad=True`), PyTorch will silently build a graph that consumes GPU memory proportional to the number of operations. With T_infer=50 inference steps and T_learn=500 learning steps per batch, this means 550 accumulated graph nodes per weight matrix, leading to rapid OOM on even modest networks. The 3.6M parameter CIFAR-10 model would exhaust GPU memory within a single batch.
- **Warning Signs**: (1) GPU memory usage growing linearly with T_infer or T_learn. (2) OOM errors that disappear when batch size is reduced to 1. (3) Training being dramatically slower than expected (the reference implementation trains in 4 minutes on an L4 GPU for 4 epochs of CIFAR-10).
- **Prevention**: (1) Wrap both inference and learning loops in `with torch.no_grad():`. (2) Initialize latents with `requires_grad=False` explicitly. (3) Add an assertion or debug check: `assert not any(p.grad is not None for p in model.parameters())` after a training step. (4) Monitor GPU memory during development with `torch.cuda.memory_allocated()` and verify it stays constant during inference iterations.
- **Phase**: Core implementation. Validate during initial development and add as a CI regression test.

### Pitfall 3: Wrong Gain-Modulated Error Computation (Hadamard Product Order)

- **Risk**: The gain-modulated error `h^(l) = f'(a^(l)) * eps^(l)` uses the elementwise product of the activation derivative and the prediction error. A common mistake is to apply the derivative to the wrong tensor -- for example, computing `f'(x^(l))` instead of `f'(a^(l))` (derivative of the activation evaluated at the preactivation, not the latent state). With ReLU, `f'(a) = (a > 0).float()`, and since `a = W @ x_above` while `x` is the latent state, these are entirely different tensors. This error propagates into both the inference update (via the feedback term `W^T @ h`) and the weight update (via the outer product `h^T @ x`), corrupting both phases silently.
- **Warning Signs**: (1) Energy decreases during inference but increases during learning. (2) Weight gradients have incorrect magnitudes. (3) The sign of weight updates is sometimes wrong, causing energy to spike.
- **Prevention**: (1) The `compute_errors` method should return both `err` and `gm_err`, computing `gm_err = err * layer.activation_deriv(a)` where `a` is the preactivation returned by the layer's forward method. (2) Write a unit test with a small 2-layer network that computes the gain-modulated error analytically and compares against the implementation. (3) Name variables unambiguously: `preactivation` not `a`, to avoid confusion with activations.
- **Phase**: Core implementation. Test with known analytical gradients before running any experiments.

### Pitfall 4: In-Place Weight Updates Corrupting nn.Parameter State

- **Risk**: The paper's reference code updates weights via a `weights` list containing direct references to `layer.W` tensors and applies in-place subtraction: `weights[l] -= eta_learn * grad_Wl`. While this works correctly when done inside `torch.no_grad()`, several things can go wrong: (a) If `weights[l]` is accidentally a copy rather than a reference, the model parameters never change. (b) If someone later refactors to use PyTorch optimizers (which expect `.grad` attributes), the in-place approach conflicts. (c) In-place operations on `nn.Parameter` objects can cause issues with `torch.compile` or `torch.jit.script` in newer PyTorch versions.
- **Warning Signs**: (1) Weights remaining at their initialized values after training (check via `torch.allclose(model.layers[0].W, initial_W_copy)`). (2) Loss/energy not decreasing during the learning phase despite correct inference. (3) Errors when attempting to use `torch.compile()`.
- **Prevention**: (1) After the first training batch, assert that weights have actually changed: `assert not torch.allclose(W_before, W_after)`. (2) Use `layer.W.data -= eta * grad` or explicitly wrap in `with torch.no_grad():` and use `layer.W -= eta * grad`. (3) Document that pcn-torch uses manual weight updates, not PyTorch optimizers, and why.
- **Phase**: Core implementation. Add a smoke test that verifies weight change after one training step.

### Pitfall 5: Incorrect Top-Layer Error Signal for Supervised vs Unsupervised

- **Risk**: In the unsupervised case, `eps^(L) = 0` (the top layer has no prediction error). In the supervised case, `eps^(L) = W^out^T @ eps^sup`. If the code path does not correctly switch between these two modes, or defaults to one without the other, the network will either (a) never receive supervised signal (eps^(L) always zero), producing random outputs, or (b) apply a supervised signal when doing unsupervised reconstruction, corrupting the generative model. Since v1 is supervised-only, the immediate risk is getting the supervised error propagation wrong -- particularly computing `eps^(L) = eps^sup` directly instead of `eps^(L) = W^out^T @ eps^sup`, which loses the output weight transpose that projects the supervised error back into the latent space.
- **Warning Signs**: (1) Classification accuracy stuck at chance level (10% for CIFAR-10's 10 classes). (2) The top latent `x^(L)` not changing during inference (if eps^(L) is accidentally zero). (3) Dimension mismatch errors if eps^sup (dim d_out) is used where eps^(L) (dim d_L) is expected.
- **Prevention**: (1) Explicitly compute `eps_L = eps_sup @ W_out` (the row-batch form). (2) Add a shape assertion: `assert eps_L.shape == (B, d_L)`. (3) Add a test that verifies a 1-batch, 1-step supervised inference changes `x^(L)` in the correct direction.
- **Phase**: Core implementation, supervised extension. Must be validated before any experiment.

### Pitfall 6: Readout Layer Using Bias When Paper Specifies No Bias

- **Risk**: `nn.Linear` defaults to `bias=True`. The paper explicitly states the readout layer `W^out` has no bias: the prediction is `y_hat = X^(L) @ W^out^T`. If `bias=True` is used, the supervised error `eps^sup = y_hat - y` will include the bias contribution, and the manual weight update formula `W^out -= eta * (eps_sup^T @ X^(L)) / B` will not update the bias at all (since it is not in the formula). The bias becomes a frozen random offset in the output, degrading classification accuracy.
- **Warning Signs**: (1) Accuracy plateaus below expected levels. (2) Inspection of `model.readout.bias` reveals it exists and is non-zero but never changes.
- **Prevention**: Initialize the readout layer as `nn.Linear(dims[-1], output_dim, bias=False)`. Add an assertion: `assert model.readout.bias is None`.
- **Phase**: Core implementation (network architecture definition).

---

## Packaging Pitfalls

### Pitfall 7: PyTorch Dependency Specification on PyPI

- **Risk**: PyTorch has historically had a complicated relationship with PyPI. Specifying `torch>=2.0` in `pyproject.toml` dependencies will pull the CPU-only PyTorch from PyPI by default (since PyTorch now publishes CPU wheels to PyPI). Users who want CUDA support must configure extra index URLs (e.g., `https://download.pytorch.org/whl/cu121`). If `pcn-torch` pins a specific PyTorch version too tightly, it breaks for users on different CUDA versions. If it pins too loosely, it may pull incompatible versions. Additionally, PyTorch's local version specifiers (`+cpu`, `+cu121`) are not valid in PEP 440 dependency strings, so you cannot require a specific accelerator variant.
- **Warning Signs**: (1) Users reporting that `pip install pcn-torch` installs CPU-only PyTorch unexpectedly. (2) Dependency resolution failures when users have CUDA PyTorch already installed. (3) GitHub issues about import errors from version mismatches.
- **Prevention**: (1) Declare `torch>=2.0` as a dependency without pinning accelerator variant -- let users install their preferred PyTorch first. (2) Document in the README: "Install PyTorch first with your desired CUDA version, then install pcn-torch." (3) Consider making `torch` an optional dependency or using a flexible lower bound. (4) Test installation in CI with both CPU and CUDA PyTorch variants.
- **Phase**: Packaging and release. Address in `pyproject.toml` and README before first PyPI publish.

### Pitfall 8: Package Name Collision or Squatting on PyPI

- **Risk**: PyPI package names are first-come-first-served. If someone else has already registered `pcn-torch`, or a confusingly similar name like `pcntorch` or `pcn_torch`, the publish will fail or users will install the wrong package. PyPI normalizes names (dashes, underscores, and case are equivalent), so `pcn-torch`, `pcn_torch`, and `PCN-Torch` all resolve to the same name.
- **Warning Signs**: (1) `pip install pcn-torch` installs an unexpected package. (2) `twine upload` or `uv publish` fails with a 403 or conflict error.
- **Prevention**: (1) Check PyPI name availability immediately: visit `https://pypi.org/project/pcn-torch/` before any development. (2) Reserve the name early by publishing a minimal 0.0.1 placeholder if needed. (3) Ensure the import name (e.g., `import pcn_torch`) matches the distribution name logically.
- **Phase**: Project setup, before any code is written. Name check should be the first action.

### Pitfall 9: Missing or Incorrect Package Metadata

- **Risk**: Publishing to PyPI with incomplete `pyproject.toml` metadata results in a package that is hard to discover, appears unprofessional, or fails to install correctly. Common omissions: missing `python_requires`, missing `license`, no `classifiers`, wrong `packages` specification (so the source files are not included in the wheel), or forgetting to include `py.typed` for type checker support.
- **Warning Signs**: (1) `pip install pcn-torch` succeeds but `import pcn_torch` fails with `ModuleNotFoundError`. (2) The PyPI page shows no description, no license, or wrong Python version requirements. (3) The installed package is empty (no .py files).
- **Prevention**: (1) Use a build backend that auto-discovers packages (e.g., `hatchling`, `setuptools` with `find:` or `find_namespace:`). (2) Build the wheel locally first (`python -m build`) and inspect its contents with `unzip -l dist/*.whl`. (3) Test installation from the built wheel in a fresh virtual environment before publishing. (4) Fill in all standard metadata fields: `name`, `version`, `description`, `requires-python`, `license`, `authors`, `classifiers`, `urls`.
- **Phase**: Packaging. Validate before first publish and automate in CI.

### Pitfall 10: Version Numbering Mistakes

- **Risk**: Starting with version `1.0.0` implies API stability per SemVer conventions. For a new, experimental library, this sets incorrect expectations. Conversely, staying at `0.x.y` indefinitely signals perpetual instability, discouraging adoption. Another mistake: forgetting to bump the version before publishing, causing PyPI to reject the upload (PyPI does not allow re-uploading the same version, even if you delete it).
- **Warning Signs**: (1) PyPI upload fails with "File already exists." (2) Users complain about breaking changes in minor version bumps. (3) The version in `pyproject.toml` does not match the version in `__init__.py`, causing confusion.
- **Prevention**: (1) Start at `0.1.0` to signal "initial development, API may change." (2) Use a single source of truth for the version (e.g., `__version__` in `__init__.py` read by `pyproject.toml` via dynamic versioning, or hardcoded in `pyproject.toml` only). (3) Automate version bumping in the release workflow. (4) Follow PEP 440 format strictly.
- **Phase**: Packaging. Decide versioning strategy before first release.

### Pitfall 11: Publishing Secrets or Large Files to PyPI

- **Risk**: Accidentally including large files (trained model weights, datasets, `.git` directory) or sensitive files (API keys, private configs) in the published package. PyPI has a ~100MB upload limit, and once published, a version cannot be re-uploaded even after deletion. Model weight files (the CIFAR-10 trained model is 3.6M params = ~14MB in float32) should not be bundled in the package.
- **Warning Signs**: (1) `python -m build` produces a wheel file larger than expected. (2) The `.git/` directory or data files appear in the wheel. (3) PyPI rejects the upload due to size limits.
- **Prevention**: (1) Create a comprehensive `MANIFEST.in` and/or configure `[tool.setuptools.packages.find]` to exclude test data, notebooks, and model weights. (2) Add a `.gitignore` and verify the wheel contents before publishing. (3) Use `check-wheel-contents` tool in CI to catch stale or unexpected files. (4) Host trained weights on GitHub Releases or Hugging Face Hub, not in the PyPI package.
- **Phase**: Packaging. Configure exclusions before first build.

---

## Numerical / Training Pitfalls

### Pitfall 12: Xavier Initialization with ReLU (Should Use He Initialization)

- **Risk**: The paper specifies Xavier (Glorot) uniform initialization for all weight matrices. Xavier initialization was designed for symmetric activations (tanh, sigmoid) and assumes the activation preserves variance. ReLU kills ~50% of activations (all negative values become zero), effectively halving the output variance at each layer. Over multiple layers, this causes signal magnitudes to decay exponentially (vanishing signals) or, depending on the specific Xavier variant, to grow (exploding signals). He (Kaiming) initialization compensates by multiplying the variance by 2. However, since the paper explicitly uses Xavier and reports 99.92% accuracy, matching the paper is the priority for v1, and diverging to He initialization would make results non-reproducible against the reference.
- **Warning Signs**: (1) Latent activations collapsing to zero in deeper layers after initialization (before any training). (2) Prediction errors being near-zero at intermediate layers (the "PE concentration" problem). (3) Gradients for intermediate layer weights being orders of magnitude smaller than for boundary layers.
- **Prevention**: (1) For v1, use Xavier uniform exactly as the paper specifies -- matching the reference is more important than theoretical optimality. (2) Add a diagnostic that logs per-layer activation statistics (mean, std, fraction of zeros) after initialization and after the first inference pass. (3) Document this as a known theoretical concern and consider He initialization as a v2 option. (4) If users report training instability on deeper networks (beyond 3 layers), recommend trying He initialization.
- **Phase**: Core implementation (weight initialization). Monitor during initial experiments; consider alternatives for future versions.

### Pitfall 13: Dead ReLU Neurons Accumulating During Training

- **Risk**: ReLU outputs zero for all negative inputs, and its derivative is zero for negative preactivations. Once a neuron's preactivation `a = W @ x_above` becomes persistently negative across all training samples, it produces zero prediction, zero derivative, and therefore zero gain-modulated error. This neuron's incoming weights receive zero gradient and never recover -- the neuron is "dead." In PCNs this is compounded: dead neurons in layer `l` produce zero gain-modulated error `h^(l)`, which means the feedback term `W^(l)^T @ h^(l)` used in layer `l+1`'s inference update also becomes zero, potentially cascading the dead zone upward.
- **Warning Signs**: (1) Fraction of zero activations increasing over epochs (monitor `(a <= 0).float().mean()` per layer). (2) A significant fraction (>30%) of neurons consistently outputting zero across the entire test set. (3) Effective model capacity shrinking over time, with accuracy plateauing early.
- **Prevention**: (1) Monitor dead neuron fraction per layer during training. (2) If dead neurons exceed a threshold (e.g., 20% of a layer), log a warning. (3) For v1, accept this as inherent to ReLU and document it. (4) For future versions, consider LeakyReLU or ELU as alternative activations (the code already supports custom activation functions via `activation_fn` and `activation_deriv` parameters).
- **Phase**: Monitoring and diagnostics. Add dead-neuron tracking during experiments; not a blocker for v1.

### Pitfall 14: Prediction Error Concentration at Boundary Layers (Deep Networks)

- **Risk**: Research has shown that in deep PCNs (significantly more than 3 layers), prediction errors concentrate at the input (layer 0) and output (layer L) layers after inference, while intermediate layers see near-zero errors. Since weight updates are proportional to prediction errors (`grad_W = -(h^T @ x) / B`), intermediate layers receive negligible gradient updates and effectively stop learning. This is documented in "Towards Stable Learning in Predictive Coding Networks" (Alonso et al.) and "Stable and Scalable Deep Predictive Coding Networks with Meta Prediction Errors." The paper's 3-layer architecture largely avoids this problem due to shallow depth, but users who try to scale to 5+ layers will encounter it.
- **Warning Signs**: (1) Per-layer energy contributions showing orders-of-magnitude differences between boundary and intermediate layers. (2) Intermediate layer weights barely changing across epochs. (3) Adding more layers does not improve (or actively degrades) performance.
- **Prevention**: (1) Document in the README that the v1 architecture is validated for 3 latent layers and that deeper networks may require stabilization techniques. (2) Log per-layer energy contributions during training. (3) For future versions, consider length regularization, skip connections, or meta-prediction-error approaches. (4) Provide a `verbose` or `diagnostics` mode that reports per-layer statistics.
- **Phase**: Documentation and future roadmap. Not blocking for v1 (3-layer architecture), but must be documented as a known limitation.

### Pitfall 15: Mixed Precision (autocast) Causing Numerical Instability

- **Risk**: The reference implementation uses `autocast(device_type='cuda')` during inference. This casts matrix multiplications to float16 for speed. However, float16 has a max value of 65504 and limited precision (~3.3 decimal digits). In PCNs, prediction errors can grow large during early inference steps (before convergence), potentially overflowing float16. Additionally, the gain-modulated error computation involves elementwise products that can underflow in float16. If the energy monitoring code sums squared errors (for the energy trajectory), float16 accumulation will lose precision on large batches.
- **Warning Signs**: (1) `NaN` or `inf` values appearing in prediction errors or latent states. (2) Energy trajectories showing sudden jumps to very large values. (3) Results differing significantly between GPU (with autocast) and CPU (without autocast). (4) Warnings from PyTorch about "overflow" or "underflow" in autocast regions.
- **Prevention**: (1) Use `autocast` only during inference, not during learning (matching the reference code). (2) Compute energy monitoring quantities in float32 even inside autocast regions (use `.float()` before squaring and summing). (3) Test that results with and without autocast produce the same accuracy (within tolerance). (4) Consider using `bfloat16` instead of `float16` if targeting Ampere+ GPUs, as it has a wider dynamic range (same exponent as float32) and avoids overflow.
- **Phase**: Core implementation (inference loop). Test with and without autocast during initial validation.

### Pitfall 16: Incorrect Batch Averaging of Weight Gradients

- **Risk**: The weight gradient must be averaged over the batch: `grad_W = -(1/B) * H^T @ X`. If the `1/B` factor is omitted, the effective learning rate scales with batch size, meaning larger batches produce proportionally larger weight updates. This makes hyperparameters non-transferable across batch sizes and can cause divergence at larger batch sizes. Conversely, if `1/B` is applied during inference (to the latent gradients), it incorrectly dampens the inference updates, which should be per-sample.
- **Warning Signs**: (1) Training working at batch size 500 but diverging at batch size 1000 without adjusting learning rate. (2) Energy increasing during the learning phase. (3) Hyperparameters from the paper not reproducing expected results.
- **Prevention**: (1) Apply `/ B` only in the weight update, not in the inference update. (2) Add a unit test: run two training steps with batch sizes B1 and B2 using the same data (duplicated), verify that the weight updates are proportional (not B-dependent). (3) Comment the code clearly: `# Batch-averaged gradient (divide by B)` vs `# Per-sample gradient (no averaging)`.
- **Phase**: Core implementation (learning loop). Validate with the paper's hyperparameters on CIFAR-10.

### Pitfall 17: Learning Steps Not Matching Batch Size (T_learn = B)

- **Risk**: The paper specifies `T_learn = B` (number of learning steps equals batch size). This is not arbitrary -- it maintains the ratio of inference-to-learning updates that the per-sample algorithm would have. If `T_learn` is set to a fixed constant (e.g., 1 or 10) regardless of batch size, the balance between inference and learning is broken. Too few learning steps means the weights under-adapt to the inferred latent states; too many means the weights overfit to a single batch's latent configuration before seeing new data.
- **Warning Signs**: (1) Performance degrading when batch size changes. (2) Energy decreasing during inference but not during learning (too few learning steps). (3) Energy oscillating during learning (too many learning steps).
- **Prevention**: (1) Default `T_learn = B` in the API, making it automatically scale with batch size. (2) Document why `T_learn = B` with a reference to the paper's Section 4.2. (3) Allow users to override but warn in the docstring that changing this ratio may require re-tuning the learning rate.
- **Phase**: API design and core implementation. Must be the default behavior.

### Pitfall 18: Latent Initialization Scale Affecting Convergence

- **Risk**: Latents are initialized from `N(0, 1)`. If the scale is too large, initial prediction errors will be enormous, causing the first few inference steps to make large, potentially destabilizing updates. If the scale is too small (or zero), the ReLU activations may all be in the linear regime initially, reducing the expressiveness of early inference steps. The reference code uses `torch.randn(batch_size, d)` (standard normal, scale=1), which works for the paper's architecture. But if users change layer dimensions significantly, the optimal initialization scale may differ.
- **Warning Signs**: (1) Energy spiking at the first inference step then slowly recovering. (2) Latent states saturating (all positive or all negative) after one inference step. (3) Inference requiring many more steps than T_infer=50 to converge.
- **Prevention**: (1) Use `torch.randn` (scale=1) as the default, matching the reference. (2) Make the initialization scale a configurable parameter (e.g., `latent_init_scale=1.0`). (3) Log the initial energy at step 0 and the energy at step 1 as a basic sanity check.
- **Phase**: Core implementation. Expose as a configurable parameter but default to the paper's value.

---

## Summary

**Top 5 most critical pitfalls ranked by impact:**

1. **Pitfall 1: Asynchronous Latent Updates** -- If updates are not snapshot-based, the algorithm is fundamentally wrong. Every downstream result will be incorrect, and the bug may be hard to detect because the network can still "sort of" train. This is the highest-impact pitfall because it silently produces wrong results rather than crashing.

2. **Pitfall 2: Accidentally Enabling Autograd** -- Forgetting `torch.no_grad()` will cause OOM crashes, making the library completely unusable on GPU. This is high-impact because it is an immediate show-stopper rather than a subtle degradation.

3. **Pitfall 5: Wrong Top-Layer Error Signal** -- Computing `eps^(L)` incorrectly (missing the `W^out^T` projection, or leaving it at zero) means the supervised signal never reaches the latent hierarchy. The network will train its generative weights but produce random classifications. High-impact because it renders supervised learning non-functional.

4. **Pitfall 7: PyTorch Dependency on PyPI** -- The most common source of user friction for PyTorch-based libraries. If installation is painful, users abandon the library before ever trying it. This is the highest-impact packaging pitfall.

5. **Pitfall 16: Incorrect Batch Averaging** -- Missing or misplaced `1/B` makes hyperparameters batch-size-dependent, causing the paper's reported results to be non-reproducible. Since reproducing the paper's CIFAR-10 results is the library's primary credibility signal, this is critical.

---

*Research completed: 2026-02-20*
*Sources: arXiv 2506.06332v1 (Stenlund 2025), "Towards Stable Learning in Predictive Coding Networks" (Alonso et al.), "Stable and Scalable Deep Predictive Coding Networks with Meta Prediction Errors", PyTorch documentation, PyPI packaging guides, "Dying ReLU and Initialization" (Lu et al. 2019)*
