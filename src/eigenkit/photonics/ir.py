"""The intermediate representation (IR) of photonic circuits, which use linear optics.

This module holds only data, and uses neither Perceval nor NumPy. It is the
photonic counterpart of `core.ir`.

A photonic circuit is a sequence of linear-optical **components** that act on
`n_modes` optical modes. They transform an input **Fock state**, which gives the
number of photons in each mode. This model of computation is very different from
the qubit-gate `CircuitSpec`. The states live in Fock space, the "gates" are beam
splitters and phase shifters that act on modes, and the readout is a distribution
of photon numbers, not the expectation value of a Pauli operator. This is why the
photonic circuits have their own IR, instead of an adapted `CircuitSpec`.

It deliberately **reuses** the `Parameter` expression tree of the qubit IR
(`Constant`, `WeightRef`, `InputRef`, `Scale`, `Sum` and `Multiply`). The angles
and phases of the components are symbolic numbers, so a component can be trained
(with a `WeightRef`) or can encode data (with an `InputRef`) with no extra code.
The same `resolve` function evaluates the angles of both kinds of circuit. The
simulation in Fock space and the translation to Perceval are in
`backends/perceval_backend`, never in this module.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Union

# Reused verbatim from the qubit IR — the single strongest bit of shared substrate.
# Angles accept what the qubit IR accepts: an expression or a plain number.
from ..core.ir import Constant, ParameterLike, _coerce_parameter


def _freeze(obj, *fields) -> None:
    """Store each sequence field as a tuple, so a spec built from lists stays
    hashable and the gradient rules can splice its components like a tuple."""
    for name in fields:
        object.__setattr__(obj, name, tuple(getattr(obj, name)))


def _coerce(obj, *fields) -> None:
    """Store each angle field as a ``Parameter``, wrapping a plain number as a
    ``Constant``, so the executors and gradient rules only ever see expressions."""
    for name in fields:
        object.__setattr__(obj, name, _coerce_parameter(getattr(obj, name)))


# ---- components ----


@dataclass(frozen=True)
class PhaseShifter:
    """A phase shift `exp(i·phi·n̂)` on a single mode.

    `n̂` is the photon-number operator of the mode. `phi` is a `Parameter`
    expression or a plain number, and a plain number is stored as a `Constant`.
    """

    mode: int
    phi: ParameterLike

    def __post_init__(self) -> None:
        _coerce(self, "phi")


@dataclass(frozen=True)
class BeamSplitter:
    """A beam splitter on two modes.

    `modes` are two adjacent modes `(m, m + 1)`, because Perceval can place a beam
    splitter only on adjacent modes. `theta` sets the reflectivity, with the
    Perceval convention `R = cos²(theta/2)`, so `theta = π/2` gives a balanced
    50:50 beam splitter. `phi` is a relative phase. Each angle is a `Parameter`
    expression or a plain number, and a plain number is stored as a `Constant`.
    The exact unitary is set by the executor, which maps the component to Perceval's
    `pcvl.BS`, so this class only describes the component.
    """

    modes: tuple[int, int]
    theta: ParameterLike
    phi: ParameterLike = Constant(0.0)

    def __post_init__(self) -> None:
        _freeze(self, "modes")
        _coerce(self, "theta", "phi")
        # Checked here rather than left to Perceval, whose own checks are asserts:
        # under `python -O` a reversed pair silently acts as (m, m + 1).
        if len(self.modes) != 2 or self.modes[1] != self.modes[0] + 1:
            raise ValueError(
                f"BeamSplitter needs adjacent modes (m, m + 1), got {self.modes}"
            )


@dataclass(frozen=True)
class Permutation:
    """A reordering of the modes, which is a network of mode swaps given as one component.

    `perm[i]` is the output mode to which input mode `i` is sent. The component
    acts on the modes `0` to `len(perm) - 1`, so `perm` must reorder exactly these
    modes.
    """

    perm: tuple[int, ...]

    def __post_init__(self) -> None:
        _freeze(self, "perm")
        if sorted(self.perm) != list(range(len(self.perm))):
            raise ValueError(
                f"perm must reorder 0..{len(self.perm) - 1}, got {self.perm}"
            )


Component = Union[PhaseShifter, BeamSplitter, Permutation]


# ---- circuit ----


@dataclass(frozen=True)
class PhotonicCircuitSpec:
    """A description of a linear-optical circuit that does not depend on the backend.

    It is the photonic counterpart of `core.ir.CircuitSpec`. It cannot be changed
    after it is built, and it can be hashed and pickled.

    `input_state` is the Fock state sent into the circuit, as the number of photons
    in each mode. For example, `(1, 0, 1, 0)` means one photon in mode `0` and one
    in mode `2`. What is measured (the full distribution of photon numbers, the
    expected photon number of each mode, or a function of the detection pattern)
    is left to the executor for now. A `PhotonicMeasurement` type is planned.

    Every component must act on the modes `0` to `n_modes - 1`, and `input_state`
    must hold non-negative integers. Both are checked when the object is built, so
    a wrong circuit fails at that point, and not inside Perceval.
    """

    n_modes: int
    n_weights: int
    n_inputs: int
    components: tuple[Component, ...]
    input_state: tuple[int, ...]

    def __post_init__(self) -> None:
        _freeze(self, "components", "input_state")
        if len(self.input_state) != self.n_modes:
            raise ValueError(
                f"input_state has {len(self.input_state)} modes but n_modes="
                f"{self.n_modes}"
            )
        if any(not isinstance(n, numbers.Integral) or n < 0 for n in self.input_state):
            raise ValueError(
                f"photon counts must be non-negative integers: {self.input_state}"
            )
        for comp in self.components:
            if isinstance(comp, PhaseShifter):
                modes = (comp.mode,)
            elif isinstance(comp, BeamSplitter):
                modes = comp.modes
            elif isinstance(comp, Permutation):
                modes = comp.perm  # the executor applies it from mode 0
            else:
                raise TypeError(f"unknown photonic component: {type(comp).__name__}")
            if any(
                not isinstance(m, numbers.Integral) or not 0 <= m < self.n_modes
                for m in modes
            ):
                raise ValueError(f"{comp!r} acts outside modes 0..{self.n_modes - 1}")

    @property
    def n_photons(self) -> int:
        """The total number of photons sent into the circuit.

        This number sets the cost of the simulation in Fock space.
        """
        return sum(self.input_state)


# ---- observables (Fock-diagonal readouts) ----


@dataclass(frozen=True)
class PhotonNumber:
    """The expected number of photons in a mode: `<O> = <n̂_mode>`."""

    mode: int


@dataclass(frozen=True)
class FockProbability:
    """The probability of detecting exactly the output Fock state `target`.

    In formula form, `<O> = P(target)`.
    """

    target: tuple[int, ...]

    def __post_init__(self) -> None:
        _freeze(self, "target")


# Fock-diagonal observables — the readouts photonic QML actually uses (photon
# counts, detection-pattern probabilities). The gradient rules in gradients.py
# assume diagonality; off-diagonal observables are not supported yet.
PhotonicObservable = Union[PhotonNumber, FockProbability]
