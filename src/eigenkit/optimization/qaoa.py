"""The Quantum Approximate Optimization Algorithm (QAOA) with the core of the library.

It builds the standard QAOA ansatz from the qubit Hamiltonian of a problem,
optimises the angles `(γ, β)` with `minimize_expectation`, and then reads the
distribution of bitstrings of the optimised state (`probabilities`) to find a
solution. It uses only NumPy and SciPy, needs no PyTorch, and works with any
backend that can return the probabilities of the basis states.
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


def qaoa_ansatz(hamiltonian: Observable, num_qubits: int, layers: int) -> CircuitSpec:
    """Build the standard QAOA circuit.

    QAOA is the Quantum Approximate Optimization Algorithm. The circuit starts in
    `|+⟩^n`, then applies `layers` repetitions of the cost unitary
    `exp(-i γ_l H_C)` and the mixer `exp(-i β_l Σ_i X_i)`. It has `2 * layers`
    weights, in the order `[γ_0, β_0, γ_1, β_1, ...]`.

    Args:
        hamiltonian (Observable): The cost Hamiltonian.
        num_qubits (int): The number of qubits, one per variable.
        layers (int): The depth of QAOA, which is the number of pairs of a cost layer
            and a mixer layer.

    Returns:
        The ansatz, as a `CircuitSpec` with `2 * layers` trainable angles.
    """
    qc = CircuitBuilder(n_qubits=num_qubits)
    angles = qc.add_weights(2 * layers)
    for qubit in range(num_qubits):
        qc.h(qubit)
    for layer in range(layers):
        gamma = angles[2 * layer]
        beta = angles[2 * layer + 1]
        # Cost layer: one Pauli exponential per Hamiltonian term.
        for term in hamiltonian.terms:
            qc.pauli_exponential(term.paulis, gamma, coefficient=term.coeff)
        # Mixer layer: exp(-i beta X_q) on each qubit.
        for qubit in range(num_qubits):
            qc.pauli_exponential((("X", qubit),), beta)
    return qc.build()


@dataclass(frozen=True, eq=False)
class OptimizationResult(ValueEquality):
    """The result of a run of the Quantum Approximate Optimization Algorithm (QAOA).

    The result is read-only. Assigning to a field raises `FrozenInstanceError`, and
    writing into an array field raises `ValueError`. This prevents a mistyped `=`
    (instead of `==`) or an in-place edit from leaving a result whose scalar fields
    describe a different run from the arrays it holds. To get a writable copy of an
    array, use `np.array(result.field)`.

    Results compare by value, including their arrays. They are not hashable, because
    a result is a collection of numbers and not a key, and a correct hash of a float
    array is expensive to compute.

    Attributes:
        best_bitstring: The configuration with the lowest energy that was found, as a
            tuple of bits.
        best_energy: Its energy for the problem being solved. This is the answer to
            report: QAOA samples, so the best sample is better than the average.
        expectation: The mean energy of the final state. Compare it with
            `best_energy` to see how concentrated the distribution is.
        optimal_parameters: The angles `γ` and `β` that the optimiser settled on.
        probabilities: The probability of every basis state, of length `2**n`.
        n_iterations: The number of iterations of the optimiser.
        success: Whether the optimiser reported convergence.
    """

    best_bitstring: tuple[int, ...]
    best_energy: float
    expectation: float  # optimized <H> + offset (variational energy estimate)
    optimal_parameters: np.ndarray
    probabilities: np.ndarray
    n_iterations: int
    success: bool

    def __post_init__(self) -> None:
        """Make the array fields read-only.

        Freezing the dataclass prevents a field from being reassigned, but it does not
        prevent writing into the array that the field holds, and a result that has been
        written into still reports the scalar values of the original run. The result
        therefore keeps its own copies, so the arrays it was built from can still be
        changed by the caller. To get a writable copy, use
        `np.array(result.optimal_parameters)`.
        """
        freeze_arrays(self, "optimal_parameters", "probabilities")


class QAOA(BaseQuantumSolver):
    """The Quantum Approximate Optimization Algorithm (QAOA) solver.

    It solves an `Ising` or `QUBO` problem.

    Args:
        layers: The depth `p` of QAOA. Default: `1`.
        provider: The backend. Default: `"numpy"`, the reference NumPy simulator.
        optimizer: A SciPy method name, `"spsa"`, or a custom callable (see
            `minimize_expectation`). Default: `"L-BFGS-B"`.
        random_state: The seed of the random initial angles. With `None`, they are
            not seeded. Default: `42`.
    """

    def __init__(
        self,
        layers: int = 1,
        *,
        provider: ProviderSpec = "numpy",
        optimizer: Optimizer = "L-BFGS-B",
        random_state: int | None = 42,
    ):
        super().__init__(provider=provider)
        self.layers = layers
        self.optimizer = optimizer
        self.random_state = random_state

    def solve(
        self,
        problem,
        *,
        initial_parameters: np.ndarray | None = None,
        top_k: int | None = None,
        **minimize_kwargs,
    ) -> OptimizationResult:
        """Optimise the QAOA angles and read the answer from the final state.

        Args:
            problem (QUBO | Ising): The problem to minimise.
            initial_parameters (np.ndarray | None): The initial angles `γ` and `β`, of
                shape `(2 * layers,)`. They are random when they are not given. Default:
                `None`.
            top_k (int | None): How many of the most probable bitstrings to score with
                the energy of the problem. The best of them is returned as
                `best_bitstring`, and the others are not returned. Default: `None`, which
                scores the `64` most probable bitstrings, or all of them when there are
                fewer.
            **minimize_kwargs (Any): Other keyword arguments, passed to
                `minimize_expectation`, for example `max_iterations`.

        Returns:
            An `OptimizationResult` with the best bitstring, its energy and the final
                distribution.

        Raises:
            ValueError: If `top_k` is smaller than `1`.
        """
        hamiltonian, offset = problem.to_hamiltonian()
        num_qubits = problem.num_vars
        spec = qaoa_ansatz(hamiltonian, num_qubits, self.layers)
        circuit = Circuit(
            spec, observables=hamiltonian, provider=self.provider, interface="numpy"
        )
        result = minimize_expectation(
            circuit,
            initial_parameters=initial_parameters,
            optimizer=self.optimizer,
            random_state=self.random_state,
            **minimize_kwargs,
        )
        probabilities = circuit.probabilities(result.optimal_parameters)

        # Decode: evaluate the problem objective on the most-probable
        # bitstrings and keep the lowest-energy one (standard QAOA post-
        # processing — the optimized state concentrates on good solutions).
        if top_k is None:
            top_k = min(2**num_qubits, 64)
        elif top_k < 1:
            # Zero would return `best_bitstring=None` with `best_energy=inf`, and
            # a negative value would silently truncate the ranking from the
            # wrong end.
            raise ValueError(
                f"top_k must be at least 1; got {top_k}. It is how many of the "
                f"most probable bitstrings to score, so there is no answer for "
                f"fewer than one."
            )
        ranked = np.argsort(probabilities)[::-1][:top_k]
        best_bits: tuple[int, ...] | None = None
        best_energy = float("inf")
        for index in ranked:
            bits = tuple(
                (int(index) >> (num_qubits - 1 - k)) & 1 for k in range(num_qubits)
            )
            energy = problem.energy(bits)
            if energy < best_energy:
                best_energy, best_bits = energy, bits

        return OptimizationResult(
            best_bitstring=best_bits,
            best_energy=best_energy,
            expectation=result.optimal_value + offset,
            optimal_parameters=result.optimal_parameters,
            probabilities=probabilities,
            n_iterations=result.n_iterations,
            success=result.success,
        )
