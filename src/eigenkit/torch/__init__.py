"""The PyTorch package: the PyTorch interface and the `nn.Module` circuit.

Importing this package registers the `"torch"` differentiation interface with the
core, so that `Circuit(..., interface="torch")` works, and it makes the
`nn.Module` circuit, `TorchCircuit`, available. This package is an optional layer
on top of the NumPy core, and everything in it needs PyTorch.
"""

from . import interface  # noqa: F401 — registers the "torch" interface on import
from .circuit import TorchCircuit


__all__ = ["TorchCircuit"]
