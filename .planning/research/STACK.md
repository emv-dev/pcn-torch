# Stack Research: pcn-torch

> **Research Date:** 2026-02-20
> **Researcher:** Claude Opus 4.6
> **Milestone:** Greenfield -- Standard 2025/2026 stack for a PyTorch ML research library published to PyPI
> **Target:** `pcn-torch` -- Predictive Coding Networks, MLP-only v1, classification + regression

---

## Recommended Stack

### Core

| Component | Choice | Version | Rationale | Confidence |
|-----------|--------|---------|-----------|------------|
| **Language** | Python | >=3.10,<3.15 | PyTorch 2.x requires >=3.10. Python 3.14 is the current stable feature release (3.14.3, Feb 2026). Supporting 3.10-3.13 covers the widest user base while 3.14 can be added once PyTorch wheels stabilize. | High |
| **Deep Learning Framework** | PyTorch | >=2.0,<3.0 | Project requirement. Latest stable is 2.10.0 (Jan 2026). Specifying >=2.0 keeps the library compatible with the broadest range of user environments, which is critical for a research library. | High |
| **Array Library** | NumPy | >=1.24 | Transitive dependency of PyTorch. NumPy 2.4.2 is current stable (Feb 2026). Using >=1.24 avoids ABI incompatibility issues while supporting both NumPy 1.x and 2.x. Do NOT pin an upper bound -- let PyTorch's own constraints handle this. | High |

### Development & Testing

| Component | Choice | Version | Rationale | Confidence |
|-----------|--------|---------|-----------|------------|
| **Project/Package Manager** | uv | >=0.10 | Astral's uv (0.10.4, Feb 2026) is the de facto standard for fast Python project management. Written in Rust, 10-100x faster than pip. Handles virtual environments, dependency resolution, lockfiles, and Python version management in one tool. Fully pip-compatible. | High |
| **Linter + Formatter** | Ruff | >=0.15 | Ruff 0.15.1 (Feb 2026) replaces Black, Flake8, isort, pyupgrade, and bandit in a single tool. Written in Rust, 50-150x faster than Flake8. Adopted by pandas, FastAPI, and PyTorch itself. Configurable via pyproject.toml. | High |
| **Type Checker** | mypy | >=1.19 | mypy 1.19.1 (Dec 2025) is the most mature and widely adopted static type checker. Better ecosystem support than Pyright for CI/CD pipelines. Configured via pyproject.toml. Note: Astral's `ty` is emerging as a faster alternative but is still too early for production use. | High |
| **Test Framework** | pytest | >=9.0 | pytest 9.1 (Feb 2026) is the undisputed standard. Native TOML config support, rich plugin ecosystem. Requires Python >=3.10 which aligns with our target. | High |
| **Test Coverage** | pytest-cov | >=7.0 | pytest-cov 7.0.0 (Sep 2025) wraps coverage.py for pytest integration. Generates coverage reports in multiple formats. Pair with coverage.py 7.13+ for accurate branch coverage. | High |
| **Pre-commit Hooks** | pre-commit | >=4.5 | pre-commit 4.5.0 (Dec 2025) manages Git hooks for running Ruff, mypy, and other checks before each commit. Ensures code quality without relying on CI alone. Use `pre-commit-uv` for faster hook installation. | High |
| **CI/CD** | GitHub Actions | N/A | Industry standard for open-source Python projects. Use a matrix strategy to test across Python 3.10-3.13. Supports trusted publishing to PyPI via OIDC (no tokens needed). | High |
| **Documentation** | MkDocs + Material | mkdocs-material >=9.7 | MkDocs-Material 9.7.2 (Feb 2026) provides a polished, modern documentation site with Markdown-based authoring. Easier setup than Sphinx for research libraries. Use `mkdocstrings[python]` for auto-generating API docs from docstrings. Note: MkDocs-Material will be minimally maintained from late 2026 as the team pivots to Zensical, but it is stable and production-ready now. | Medium |

### Packaging & Distribution

| Component | Choice | Version | Rationale | Confidence |
|-----------|--------|---------|-----------|------------|
| **Build Backend** | hatchling | >=1.28 | Hatchling 1.28.0 (Nov 2025) is the recommended build backend for pure-Python packages. Strict PEP 621 compliance, excellent plugin ecosystem (hatch-vcs for git-based versioning). Chosen over `uv_build` because hatchling is more mature with better support for dynamic versioning via `hatch-vcs`. Chosen over setuptools because hatchling requires zero boilerplate beyond pyproject.toml. | High |
| **Version Management** | hatch-vcs | >=0.4 | Derives version numbers from git tags automatically. Eliminates manual version bumping. Works seamlessly with hatchling. Follow semantic versioning: 0.1.0 for initial alpha, 1.0.0 for stable release. | High |
| **Config File** | pyproject.toml | N/A | Single configuration file for build system, project metadata, tool configs (Ruff, mypy, pytest, coverage). PEP 621 standard. No setup.py, setup.cfg, or MANIFEST.in needed. | High |
| **PyPI Publishing** | Trusted Publishers (OIDC) | N/A | GitHub Actions OIDC-based publishing to PyPI. No API tokens to manage or rotate. Used by >25% of PyPI uploads as of Oct 2025. Use `pypa/gh-action-pypi-publish@release/v1` in CI. | High |
| **Build Tool** | `uv build` | >=0.10 | Use `uv build` to create sdist and wheel. Faster than `python -m build`. Produces standard PEP 517 artifacts that hatchling processes. | High |

---

## What NOT to Use

| Tool | Why Not |
|------|---------|
| **setuptools / setup.py** | Legacy build system. Requires boilerplate (setup.py, setup.cfg, MANIFEST.in). Hatchling achieves the same result with pyproject.toml alone. Setuptools is only needed for C extensions, which pcn-torch does not have. |
| **Poetry** | Adds a proprietary lockfile format and non-standard dependency specification. Poetry 2.0 now supports PEP 621 but its resolver is significantly slower than uv. The Poetry ecosystem is fragmented (poetry-core vs poetry). |
| **Conda** | Unnecessary for a pure-Python library. Adds distribution complexity. Users who need Conda can install from PyPI via `pip install` inside Conda environments. |
| **Black** | Replaced by Ruff's formatter, which is 100x faster and configuration-compatible. Black is now in maintenance mode with the rise of Ruff. |
| **Flake8** | Replaced by Ruff's linter. Flake8 is Python-based and orders of magnitude slower. Ruff implements 800+ rules including all common Flake8 plugins. |
| **isort** | Replaced by Ruff's isort-compatible import sorting. One fewer tool to configure. |
| **Pyright / Pylance** | While faster than mypy and excellent for VS Code integration, Pyright is less commonly used in CI pipelines for open-source ML libraries. mypy has broader community adoption and better documentation for library authors. Developers can still use Pylance locally. |
| **tox** | Unnecessary when using `uv` + GitHub Actions matrix builds. Tox adds another configuration layer and is slower than running pytest directly in uv-managed environments. |
| **Sphinx** | Heavier setup, reStructuredText-based, steeper learning curve. MkDocs-Material provides a better developer experience for a research library where documentation is primarily prose + API reference. |
| **Flit** | Simpler than hatchling but lacks plugin support, dynamic versioning, and build hooks. Hatchling is strictly more capable with minimal additional complexity. |
| **pdm** | Good tool but less adoption than uv. Astral's uv has become the clear community winner for project management in 2025. |
| **`uv_build` (as build backend)** | Still relatively new (stable since late 2025). Lacks hatchling's plugin ecosystem, particularly hatch-vcs for git-based version management. Suitable for simple projects, but hatchling is more battle-tested for libraries published to PyPI. |
| **ty (Astral type checker)** | Extremely fast (10-60x faster than mypy) but still in early development as of Feb 2026. Not production-ready for CI enforcement. Worth watching for future adoption. |

---

## Key Considerations

### Dependency Strategy
- **Declare loose lower bounds** for `torch` and `numpy` in `[project.dependencies]`. Research library users have diverse environments; avoid pinning exact versions.
- **Pin dev dependencies exactly** in `uv.lock` for reproducible CI builds.
- **Do NOT vendor PyTorch** -- it is a peer dependency. Users must install it themselves or as an extra.
- Example dependency spec: `dependencies = ["torch>=2.0"]`

### Python Version Matrix
- **Test on:** Python 3.10, 3.11, 3.12, 3.13 in CI (GitHub Actions matrix).
- **Declare:** `requires-python = ">=3.10"` in pyproject.toml.
- **Defer Python 3.14:** Add support once PyTorch 2.11+ provides stable wheels (expected April 2026).

### Project Layout
- Use **src layout** (`src/pcn_torch/`) -- the modern standard that prevents accidental imports of the source tree during testing.
- Package name on PyPI: `pcn-torch` (hyphenated).
- Import name in Python: `pcn_torch` (underscored).

### Versioning Strategy
- Use **Semantic Versioning** (SemVer): `MAJOR.MINOR.PATCH`.
- Start at `0.1.0` (pre-1.0 signals "API may change").
- Derive versions from git tags via `hatch-vcs` -- no manual version bumps.

### CI/CD Pipeline (GitHub Actions)
1. **On every PR:** Lint (ruff check + ruff format --check), type check (mypy), test (pytest with coverage), across Python 3.10-3.13.
2. **On tag push (v*):** Build sdist + wheel with `uv build`, publish to PyPI via Trusted Publishers.
3. **Optional:** Nightly test against PyTorch nightly to catch breaking changes early.

### Documentation Approach
- Use **Google-style docstrings** for all public API functions and classes.
- Auto-generate API reference with `mkdocstrings[python]`.
- Host on GitHub Pages via `mkdocs gh-deploy` or Read the Docs.

### Testing Strategy
- Use `pytest` with `pytest-cov` for unit tests.
- Test against **CPU-only PyTorch** in CI (no GPU needed for MLP-only v1).
- Use `torch.manual_seed()` for reproducible tensor operations in tests.
- Target 80%+ code coverage for v1.

---

## Minimal pyproject.toml Skeleton

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "pcn-torch"
dynamic = ["version"]
description = "Predictive Coding Networks in PyTorch"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0",
    "pytest-cov>=7.0",
    "mypy>=1.19",
    "ruff>=0.15",
    "pre-commit>=4.5",
]
docs = [
    "mkdocs-material>=9.7",
    "mkdocstrings[python]>=0.27",
]

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.targets.wheel]
packages = ["src/pcn_torch"]

[tool.ruff]
target-version = "py310"
line-length = 88
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "TCH"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=pcn_torch --cov-report=term-missing"

[tool.coverage.run]
source = ["pcn_torch"]
branch = true
```

---

## Version Reference (as of 2026-02-20)

| Package | Latest Stable | Release Date | PyPI |
|---------|--------------|--------------|------|
| Python | 3.14.3 | 2026-02-03 | N/A |
| PyTorch | 2.10.0 | 2026-01-21 | [torch](https://pypi.org/project/torch/) |
| NumPy | 2.4.2 | 2026-02-01 | [numpy](https://pypi.org/project/numpy/) |
| uv | 0.10.4 | 2026-02-17 | [uv](https://pypi.org/project/uv/) |
| Ruff | 0.15.1 | 2026-02-12 | [ruff](https://pypi.org/project/ruff/) |
| mypy | 1.19.1 | 2025-12-15 | [mypy](https://pypi.org/project/mypy/) |
| pytest | 9.1.0 | 2026-02-15 | [pytest](https://pypi.org/project/pytest/) |
| pytest-cov | 7.0.0 | 2025-09-09 | [pytest-cov](https://pypi.org/project/pytest-cov/) |
| hatchling | 1.28.0 | 2025-11-27 | [hatchling](https://pypi.org/project/hatchling/) |
| pre-commit | 4.5.0 | 2025-12-16 | [pre-commit](https://pypi.org/project/pre-commit/) |
| mkdocs-material | 9.7.2 | 2026-02-18 | [mkdocs-material](https://pypi.org/project/mkdocs-material/) |

---

## Sources

- [PyTorch Releases](https://github.com/pytorch/pytorch/releases)
- [Python Packaging User Guide - pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Python Build Backends in 2025](https://medium.com/@dynamicy/python-build-backends-in-2025-what-to-use-and-why-uv-build-vs-hatchling-vs-poetry-core-94dd6b92248f)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [GitHub Actions PyPI Publishing Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/packaging-simple/)
- [Python Status of Versions](https://devguide.python.org/versions/)
- [NumPy News](https://numpy.org/news/)
- [Hatchling on PyPI](https://pypi.org/project/hatchling/)
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Astral ty announcement](https://astral.sh/blog/ty)
