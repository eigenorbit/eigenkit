"""Photonic circuits, which compute with linear optics.

This package is separate from the qubit core, and works with Quandela's Perceval
software. It does *not* use the qubit `Provider` protocol, because Fock space,
components that act on optical modes and photon statistics do not fit a
`CircuitSpec` with Pauli `Observable`s.

It has the same layers as the qubit core. `ir` holds the data, `executor` defines
the protocol, `gradients` computes the gradients, `backends/` holds the executors
that call a software development kit (SDK), and `torch` is the PyTorch interface.
Importing this package does not import Perceval or PyTorch. The Perceval SDK is
imported only when an executor runs, and the PyTorch interface (`PhotonicCircuit`)
is imported the first time the name is accessed. `PhotonicCircuit` is listed in
`__all__`, so `from eigenkit.photonics import *` *does* import PyTorch. If that
matters, import only the names that you use.
"""

from .backends.perceval_backend import PercevalBackend, PercevalCompiled
from .backends.perceval_remote import QuandelaCloudBackend
from .executor import Compiled, Executor
from .ir import (
    BeamSplitter,
    Component,
    FockProbability,
    Permutation,
    PhaseShifter,
    PhotonicCircuitSpec,
    PhotonicObservable,
    PhotonNumber,
)


__all__ = [
    "BeamSplitter",
    "Compiled",
    "Component",
    "Executor",
    "FockProbability",
    "PercevalBackend",
    "PercevalCompiled",
    "Permutation",
    "PhaseShifter",
    "PhotonicCircuit",
    "PhotonicCircuitSpec",
    "PhotonicObservable",
    "PhotonNumber",
    "QuandelaCloudBackend",
]


def __getattr__(name):
    # Lazily expose the torch interface so importing the package stays torch-free.
    if name == "PhotonicCircuit":
        from .torch import PhotonicCircuit

        return PhotonicCircuit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
