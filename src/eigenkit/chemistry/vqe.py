"""VQE — the Variational Quantum Eigensolver, on the core spine.

Given a qubit Hamiltonian (an `Observable` plus its constant offset), VQE
minimizes ``<H>`` over a parametrized ansatz to estimate the ground-state
energy. It is a thin layer over `minimize_expectation`: build an ansatz, wrap
it in a numpy `Circuit` whose observable is the Hamiltonian, optimize. Pure
core (numpy/scipy), torch-free, backend-agnostic.

The default ansatz is a hardware-efficient one (RY/RZ rotations + a linear
CNOT entangler) starting from |0...0>, so it needs no Hartree-Fock
preparation. For a chemistry-specific ansatz, pass `uccsd_ansatz(...)` as
`ansatz`; it starts from the Hartree-Fock reference itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core._results import ValueEquality, freeze_arrays
from ..core.builder import CircuitBuilder
from ..core.circuit import Circuit, ProviderSpec
from ..core.ir import CircuitSpec, Observable
from ..core.solver import BaseQuantumSolver
from ..core.variational import Optimizer, minimize_expectation


def hardware_efficient_ansatz(n_qubits: int, layers: int) -> CircuitSpec:
    """A standard hardware-efficient ansatz.

    It starts with a layer of `RY` rotations, then repeats `layers` times a linear
    chain of `CNOT` gates followed by an `RY` and an `RZ` rotation on each qubit. It
    has `n_qubits + 2 * layers * n_qubits` trainable weights.

    Args:
        n_qubits (int): The number of qubits.
        layers (int): The number of repetitions.

    Returns:
        The ansatz, as a `CircuitSpec`.
    """
    qc = CircuitBuilder(n_qubits=n_qubits)
    weights = qc.add_weights(n_qubits + 2 * layers * n_qubits)
    index = 0
    for qubit in range(n_qubits):
        qc.ry(weights[index], qubit)
        index += 1
    for _ in range(layers):
        for qubit in range(n_qubits - 1):
            qc.cnot(qubit, qubit + 1)
        for qubit in range(n_qubits):
            qc.ry(weights[index], qubit)
            index += 1
            qc.rz(weights[index], qubit)
            index += 1
    return qc.build()


@dataclass(frozen=True, eq=False)
class VQEResult(ValueEquality):
    """Outcome of a run of the Variational Quantum Eigensolver (VQE).

    The result is read-only. Assigning to a field raises `FrozenInstanceError`, and
    writing into an array field raises `ValueError`. This prevents a mistyped `=`
    (instead of `==`) or an in-place edit from leaving a result whose scalar fields
    describe a different run from the array it holds. To get a writable copy of the
    array, use `np.array(result.field)`.

    Results compare by value, including their arrays. They are not hashable,
    because a result is a collection of numbers and not a key, and a correct hash
    of a float array is expensive to compute.

    Attributes:
        energy: The lowest energy found, with the offset added back. For a
            molecule, it is in Hartree.
        optimal_parameters: The ansatz weights that produced this energy.
        n_iterations: The number of optimiser iterations in the best restart. It
            is not summed over restarts.
        success: Whether the optimiser of that restart reported convergence.
    """

    energy: float
    optimal_parameters: np.ndarray
    n_iterations: int
    success: bool

    def __post_init__(self) -> None:
        """Make the array field read-only.

        Freezing the dataclass prevents a field from being reassigned, but it does not
        prevent writing into the array that the field holds, and a result that has been
        written into still reports the scalar values of the original run. The result
        therefore keeps its own copy, so the array it was built from can still be
        changed by the caller. To get a writable copy, use
        `np.array(result.optimal_parameters)`.
        """
        freeze_arrays(self, "optimal_parameters")


class VQE(BaseQuantumSolver):
    """Variational Quantum Eigensolver (VQE).

    Args:
        layers: The depth of the default hardware-efficient ansatz. The default is
            `2`.
        provider: The backend to use. The default is `"numpy"`, the built-in
            reference simulator.
        optimizer: The optimiser: the name of a `scipy.optimize.minimize` method,
            `"spsa"`, or a custom callable (see `minimize_expectation`). The
            default is `"L-BFGS-B"`.
        restarts: The number of runs from random starting points. The lowest
            energy is kept, because variational landscapes have local minima. The
            default is `1`.
        random_state: The base seed. Restart `r` uses `random_state + r`, and
            `None` leaves every restart unseeded. The default is `42`.
    """

    def __init__(
        self,
        layers: int = 2,
        *,
        provider: ProviderSpec = "numpy",
        optimizer: Optimizer = "L-BFGS-B",
        restarts: int = 1,
        random_state: int | None = 42,
    ):
        if restarts < 1:
            raise ValueError(f"restarts must be >= 1, got {restarts}")
        super().__init__(provider=provider)
        self.layers = layers
        self.optimizer = optimizer
        self.restarts = restarts
        self.random_state = random_state

    def solve(
        self,
        hamiltonian: Observable,
        n_qubits: int,
        *,
        offset: float = 0.0,
        ansatz: CircuitSpec | None = None,
    ) -> VQEResult:
        """Minimise the expectation value of the Hamiltonian over the ansatz.

        It runs `restarts` independent optimisations from different random starting
        points and keeps the lowest energy, because a variational landscape has local
        minima.

        Args:
            hamiltonian: The Hamiltonian to minimise, as a qubit `Observable`, usually
                the one returned by `jordan_wigner`.
            n_qubits: The number of qubits of the circuit. It must match the
                Hamiltonian.
            offset: The constant added to the minimised value. This is where the
                identity term returned by the mapping goes, which for a molecule
                includes the nuclear repulsion, in Hartree. If it is left out, the
                energy is wrong by exactly that constant. The default is `0.0`.
            ansatz: The circuit to optimise. The default, `None`, uses the built-in
                hardware-efficient ansatz with `layers` layers.

        Returns:
            A `VQEResult` with the best energy, its parameters, and whether that
                restart converged.
        """
        spec = (
            ansatz
            if ansatz is not None
            else hardware_efficient_ansatz(n_qubits, self.layers)
        )
        circuit = Circuit(
            spec, observables=hamiltonian, provider=self.provider, interface="numpy"
        )
        best = None
        for restart in range(self.restarts):
            result = minimize_expectation(
                circuit,
                optimizer=self.optimizer,
                random_state=(
                    None if self.random_state is None else self.random_state + restart
                ),
            )
            if best is None or result.optimal_value < best.optimal_value:
                best = result
        return VQEResult(
            energy=best.optimal_value + offset,
            optimal_parameters=best.optimal_parameters,
            n_iterations=best.n_iterations,
            success=best.success,
        )
