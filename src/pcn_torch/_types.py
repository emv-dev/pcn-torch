"""Type aliases for activation functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from torch import Tensor

ActivationFn: TypeAlias = Callable[[Tensor], Tensor]
ActivationDeriv: TypeAlias = Callable[[Tensor], Tensor]
ActivationPair: TypeAlias = tuple[ActivationFn, ActivationDeriv]
