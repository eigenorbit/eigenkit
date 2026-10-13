"""The internal circuit layer of eigenkit.

This package re-exports the classes and functions that users need, and it
registers the default backends when it is imported.

The entry points are:

- `Circuit` creates a circuit that can be run.
- `CircuitBuilder` builds a circuit one gate at a time.
- `AngleEncoding`, `BasicVariational`, `StrongVariational`, `Reuploading`,
  `CustomBlock` and `assemble` combine blocks into a circuit.
- `X`, `Y`, `Z`, `all_z`, `from_string` and `Observable` build observables.
- `PennyLaneConfig`, `QiskitConfig` and the other configurations hold the
  settings of each backend.
- `register` gives a configuration a name, so that it can be used again.
- `BaseQuantumSolver` is the base class of every solver that computes its result
  in a single call.

The top-level `provider=` argument accepts a registered name, a configuration
dataclass, or a backend instance.
"""

from . import backends  # noqa: F401 — registers numpy eagerly, others lazily
from .backends import NumpyConfig, NumpySimulator
from .blocks import (
    S0,
    AmplitudeEncoding,
    AngleEncoding,
    BasicVariational,
    CustomBlock,
    Grover,
    QubitUnitary,
    Reuploading,
    StrongVariational,
    assemble,
)
from .builder import CircuitBuilder
from .circuit import Circuit
from .ir import CircuitSpec, Observable, adjoint
from .observables import X, Y, Z, all_z, from_string
from .provider import (
    CostWarning,
    JobHandle,
    register,
    register_lazy,
    registered_names,
)
from .solver import BaseQuantumSolver
from .variational import VariationalResult, minimize_expectation


__all__ = [
    # circuits
    "Circuit",
    # construction
    "CircuitBuilder",
    "AmplitudeEncoding",
    "AngleEncoding",
    "BasicVariational",
    "Grover",
    "QubitUnitary",
    "S0",
    "StrongVariational",
    "Reuploading",
    "CustomBlock",
    "assemble",
    # observables
    "X",
    "Y",
    "Z",
    "all_z",
    "from_string",
    "Observable",
    # circuit data + transforms
    "CircuitSpec",
    "adjoint",
    # variational optimization
    "minimize_expectation",
    "VariationalResult",
    # solver contract
    "BaseQuantumSolver",
    # backends and configs
    "BraketBackend",
    "BraketConfig",
    "BraketJobHandle",
    "CudaQuantumBackend",
    "CudaQuantumConfig",
    "IQMBackend",
    "IQMConfig",
    "IQMJobHandle",
    "NumpyConfig",
    "NumpySimulator",
    "PennyLaneBackend",
    "PennyLaneConfig",
    "QiskitAerBackend",
    "QiskitAerConfig",
    "QiskitBackend",
    "QiskitConfig",
    "QiskitRuntimeBackend",
    "QiskitRuntimeConfig",
    "QiskitRuntimeJobHandle",
    "QsimBackend",
    "QsimConfig",
    "QulacsBackend",
    "QulacsConfig",
    "TensorCircuitBackend",
    "TensorCircuitConfig",
    "register",
    "register_lazy",
    "registered_names",
    "CostWarning",
    "JobHandle",
]


# Backend classes/configs are re-exported lazily (via `backends.__getattr__`)
# so the core imports no torch / pennylane / qiskit SDK until one is accessed.
_LAZY_BACKEND_EXPORTS = frozenset(
    {
        "BraketBackend",
        "BraketConfig",
        "BraketJobHandle",
        "CudaQuantumBackend",
        "CudaQuantumConfig",
        "IQMBackend",
        "IQMConfig",
        "IQMJobHandle",
        "PennyLaneBackend",
        "PennyLaneConfig",
        "QiskitAerBackend",
        "QiskitAerConfig",
        "QiskitBackend",
        "QiskitConfig",
        "QiskitRuntimeBackend",
        "QiskitRuntimeConfig",
        "QiskitRuntimeJobHandle",
        "QsimBackend",
        "QsimConfig",
        "QulacsBackend",
        "QulacsConfig",
        "TensorCircuitBackend",
        "TensorCircuitConfig",
    }
)


def __getattr__(name: str):
    """Return the objects that are re-exported only when they are first accessed.

    These are the backend classes and configurations. Because of this, importing
    `core` does not import any framework until one is needed.
    """
    if name in _LAZY_BACKEND_EXPORTS:
        return getattr(backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
