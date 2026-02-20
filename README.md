# pcn-torch

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--release-orange)](https://github.com/emv-dev/pcn-torch)
[![arXiv](https://img.shields.io/badge/arXiv-2506.06332-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.06332)

A clean, PyTorch-native implementation of Predictive Coding Networks (PCNs) from [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332).

## What is this?

pcn-torch is a pip-installable PyTorch library implementing Predictive Coding Networks as described in Stenlund's "Introduction to Predictive Coding Networks for Machine Learning" (2025). It provides MLP-based PCN layers and networks for supervised learning, supporting both classification and regression tasks. All training uses local Hebbian-like update rules under `torch.no_grad()` -- no backpropagation.

## Project Planning

This project is planned and tracked in the `.planning/` directory.

| Document | Description |
|----------|-------------|
| [PROJECT.md](.planning/PROJECT.md) | Project definition, core value, constraints, and key decisions |
| [ROADMAP.md](.planning/ROADMAP.md) | 4-phase build plan from foundation to PyPI publishing |
| [REQUIREMENTS.md](.planning/REQUIREMENTS.md) | v1 and v2 requirements with traceability to phases |
| [STATE.md](.planning/STATE.md) | Current project state, progress, and session continuity |
| [Research Summary](.planning/research/SUMMARY.md) | Stack, features, architecture, and pitfalls research |
| [Architecture Research](.planning/research/ARCHITECTURE.md) | Module dependency graph and component design |
| [Features Research](.planning/research/FEATURES.md) | Table stakes, differentiators, and competitive analysis |
| [Pitfalls Research](.planning/research/PITFALLS.md) | 18 identified pitfalls with prevention strategies |
| [Stack Research](.planning/research/STACK.md) | Technology choices and version decisions |

## Roadmap Overview

1. **Foundation** -- Package scaffold, PCNLayer, types, activations
2. **Core Model** -- PredictiveCodingNetwork hierarchy, error computation, readout
3. **Training + Energy + Tests** -- Inference/learning loops, energy tracking, correctness tests
4. **Integration + Publishing** -- CIFAR-10 example, PyPI publish, README

## References

- **Paper:** Stenlund, M. (2025). "Introduction to Predictive Coding Networks for Machine Learning." [arXiv:2506.06332v1](https://arxiv.org/abs/2506.06332)
- **Reference implementation:** [github.com/Monadillo/pcn-intro](https://github.com/Monadillo/pcn-intro)
