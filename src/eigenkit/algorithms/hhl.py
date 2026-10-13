from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..core._results import ValueEquality, freeze_arrays
from ..core.blocks import (
    AmplitudeEncoding,
    HamiltonianQPEBlock,
    QPEBlock,
    QubitUnitary,
    assemble,
)
from ..core.circuit import Circuit, ProviderSpec
from ..core.ir import CircuitSpec, Constant, Gate
from ..core.solver import BaseQuantumSolver
from ._hamiltonian_sim import (
    _walsh_hadamard,
    pauli_decompose,
    trotter_drift_bound,
)


# --- honest-mode Trotter-step selection -------------------------------------
# Drift bound target as a fraction of C; 1/4 leaves margin to the C/2 Nyquist-capture
# threshold.
_AUTO_TROTTER_TOLERANCE = 0.25
# The bound is leading-order: trustworthy while the per-step generator angle
# τ·Σ|c_k| ≲ 0.5, tenfold optimistic past ~3.
_TROTTER_ANGLE_GUARD = 0.5
# Cost is Θ(steps · 2^n_l · terms), so an unbounded selector can ask for tens of
# millions of gates; past this it clamps and warns.
_AUTO_TROTTER_GATE_BUDGET = 2_000_000
# Relative residual above which solve() says the answer is rough.
_RESIDUAL_WARN_THRESHOLD = 0.25


@dataclass(frozen=True, eq=False)
class HHLResult(ValueEquality):
    """Output of [`HHL.solve()`][eigenkit.algorithms.HHL.solve].

    The result is read-only. Assigning to a field raises `FrozenInstanceError`, and
    writing into an array field raises `ValueError`. This prevents a mistyped `=`
    (instead of `==`) or an in-place edit from leaving a result whose scalar fields
    describe a different run from the arrays it holds. To get a writable copy of an
    array, use `np.array(result.field)`.

    Results compare by value, including their arrays. They are not hashable,
    because a result is a collection of numbers and not a key, and a correct hash
    of a float array is expensive to compute.

    Attributes:
        solution: The best-fit `x` with `Ax ≈ b`. It is `float64` for real inputs,
            and `complex128` when `A` or `b` has a non-zero imaginary part.
        euclidean_norm: The norm `‖solution‖` of the returned solution. It is
            computed after the classical least-squares rescaling, so it is not a
            norm that HHL itself measures.
        residual: The relative error `‖Ax − b‖ / ‖b‖`, computed with the original
            `A` and `b`.
        success_probability: The probability that the ancilla is in `|1⟩`, summed
            over all states of the clock register.
        readout_probability: The probability of the joint event from which the
            solution is read: the ancilla in `|1⟩` and the clock register in
            `|0…0⟩`. This is the physical post-selection rate. It is never larger
            than `success_probability`, so use it to estimate the number of shots.
        raw_solution: The post-selected amplitudes as the circuit produced them.
            They are complex, and they come before the imaginary part is dropped
            and before the rescaling, so this is the only part of the output that
            comes directly from the quantum computation. The vector has unit norm,
            except when the size of the system is not a power of two. In that case
            it is a slice of a unit-norm state, and the Trotterised evolution leaks
            a small amount of weight into the zero padding.

    !!! note
        HHL produces the quantum state `|x⟩`, not a vector of numbers. The full
        array in this result is read from the simulator's statevector, which only a
        simulator can provide. On hardware, only expectation values can be
        measured, and extracting all `N` components of `|x⟩` costs `O(N)`
        measurements, which cancels the speedup. In practice, `|x⟩` is passed to
        another quantum routine, or used to estimate one observable. Recovering the
        sign of `1/λ` needs a separate protocol. All of this applies to both values
        of `mode`.

    !!! warning
        `solution` is completed by two classical steps that hardware cannot
        perform. For real inputs, the imaginary part is discarded. The magnitude
        and the sign then come from a least-squares projection,
        `x = raw · ⟨A·raw, b⟩ / ‖A·raw‖²`, which uses the full classical `A` and
        `b`. Only the direction of `raw` comes from the quantum computation. The
        rescaling can change `residual` by an order of magnitude, and for an
        indefinite `A` its sign correction is not reliable, because
        `⟨Ax, b⟩ ≥ 0` does not imply `⟨x, A⁻¹b⟩ ≥ 0`.
    """

    solution: np.ndarray
    euclidean_norm: float
    residual: float
    success_probability: float
    readout_probability: float = 0.0
    raw_solution: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Make the array fields read-only.

        Freezing the dataclass prevents a field from being reassigned, but it does not
        prevent writing into the array that the field holds, and a result that has been
        written into still reports the scalar values of the original run. The result
        therefore keeps its own copy, so the array it was built from can still be
        changed by the caller. To get a writable copy, use
        `np.array(result.solution)`.

        Both array fields are frozen. `raw_solution` is optional, and `freeze_arrays`
        would turn a `None` into a zero-dimensional object array, so it is only frozen
        when the run recorded one.
        """
        freeze_arrays(self, "solution")
        if self.raw_solution is not None:
            freeze_arrays(self, "raw_solution")


# Both paths prepare |b⟩ with AmplitudeEncoding and invert the eigenvalues with a
# uniformly controlled rotation (RY + CNOT); the reference path adds the dense
# evolution oracle on top.
_HHL_REQUIRED_GATES: frozenset[str] = frozenset(
    {"AmplitudeEncoding", "ControlledQubitUnitary", "H", "CPHASE", "SWAP", "RY", "CNOT"}
)
# The honest path Trotterises the evolution, so it needs elementary rotations and CNOTs
# only, with no ControlledQubitUnitary at all.  RX rotates Y factors into the Z basis.
_HHL_HONEST_REQUIRED_GATES: frozenset[str] = frozenset(
    {"AmplitudeEncoding", "H", "CPHASE", "SWAP", "RX", "RY", "RZ", "CNOT", "PhaseShift"}
)


class HHL(BaseQuantumSolver):
    r"""Quantum linear solver based on the algorithm of Harrow, Hassidim and Lloyd (HHL).

    Solves `Ax = b` for any invertible square matrix `A`. If `A` is Hermitian, it is
    used directly, even when it is not positive semidefinite. Any other matrix is
    converted to the normal equations `A†Ax = A†b`, which have the same solution and a
    Hermitian, positive semidefinite system matrix. The cost of this conversion is
    that the condition number `κ` is squared.

    The circuit itself inverts the eigenvalues. It applies controlled `RY`
    rotations to the ancilla, with one fixed angle for each state of the clock
    register, so the reciprocal `1/λ` of each eigenvalue is encoded in the
    amplitudes without any classical division by the eigenvalues. The readout is
    different: it uses the simulator's full statevector and keeps only the part
    where the ancilla is in `|1⟩` and the clock register is in `|0⟩`. HHL produces
    the quantum state `|x⟩`, so reading a classical vector from it is only possible
    on a simulator. See [`HHLResult`][eigenkit.algorithms.HHLResult].

    !!! warning
        The backend must provide `statevector()` on its compiled circuit. The
        built-in numpy simulator, PennyLane `default.qubit` and `lightning.qubit`,
        and Qiskit Aer all do. A custom backend registered with
        [`register()`][eigenkit.core.provider.register] must implement it,
        otherwise `solve()` raises `NotImplementedError`.

    **Scope and scaling.** HHL is efficient only when three conditions hold: `A` is
    a sum of a few short Pauli terms, `A` is well conditioned, and the state `|b⟩`
    can be prepared efficiently.

    - *Hamiltonian simulation.* `e^{iAt}` is Trotterised from the Pauli
      decomposition of `A`, so its cost follows the number of Pauli terms. This
      means sparsity in the Pauli basis, which is different from sparsity of the
      matrix. The module `algorithms/_hamiltonian_sim.py` explains what this means
      in practice. In short, a sparse matrix is not necessarily cheap to simulate
      here.
    - *State preparation.* `|b⟩` is prepared with
      [`AmplitudeEncoding`][eigenkit.core.blocks.AmplitudeEncoding], which is
      treated as an oracle of unit cost. Loading general amplitudes costs `O(N)`
      gates, so a real speedup assumes that `b` comes from a formula, from an
      earlier quantum routine, or from a quantum random-access memory (qRAM).
    - *Conditioning.* The condition number `κ` is never an input. It only bounds
      the achievable success probability and the size of the clock register.
    - *Inversion angles.* The gates are elementary (see
      `_uniformly_controlled_ry()`), so the circuit built in `"honest"` mode
      contains no unitary-matrix gate. However, the `2^n_l` angles, where `n_l` is
      the number of clock qubits, are listed one by one instead of being computed
      by the circuit. The classical description of the circuit therefore has size
      `Θ(2^n_l)`. The circuit can be built from elementary gates, but its
      description is not compact.

    Solvers for dense matrices, for example the one of Wossnig, Zhao and Prakash,
    reach a quadratic speedup when the spectral norm is bounded. They need a
    specialised memory architecture that this package does not provide.

    **Real and complex inputs.** For real `A` and `b`, the readout discards the
    imaginary part. The leading error of the Trotter eigenvectors enters the raw
    vector as a purely imaginary component, so discarding the imaginary part
    removes it. This matters at `trotter_order=1`, where it improves the
    convergence by one order: the measured error falls as `steps⁻²` for real
    inputs, and as `steps⁻¹` for a complex Hermitian `A` or a complex `b`, which
    keep that error. At `trotter_order=2`, the default, the first-order commutator
    already cancels by construction, so there is nothing left to remove, and real
    and complex inputs both converge as `steps⁻²`. This advantage was not designed.
    It depends on the test for real inputs in
    [`solve()`][eigenkit.algorithms.HHL.solve], so a change to that test changes
    the convergence rate at order 1.

    Args:
        n_eigenvalue_qubits: The number of qubits in the clock register.
            Eigenvalues are resolved into `2**n_eigenvalue_qubits` buckets. The
            value must be at least 1, and at least 2 in `"honest"` mode. More
            qubits give a finer resolution of the spectrum and a larger circuit.
            The default is `4`.
        t: The evolution time for `U = e^{iAt}`. If given, it must be positive and
            finite. If `None`, it is chosen automatically: from the spectrum in
            `"reference"` mode, and from a Gershgorin bound in `"honest"` mode. The
            default is `None`.
        epsilon: The relative eigenvalue cutoff, in `[0, 1)`. For a Hermitian `A`,
            buckets whose estimated eigenvalue is below `epsilon * λ_max` are
            treated as zero. For the normal equations, the cutoff applies to the
            singular values instead, and it is implemented as `epsilon²` on the
            squared spectrum. If no eigenvalue is above the cutoff, `solve()` raises
            an error. The default is `0.01`.
        provider: The backend to use: a registered name (`"numpy"`, `"pennylane"`,
            `"qiskit"` and so on), a config instance, or a backend object. It must
            provide `statevector()`. The default is `"numpy"`.
        mode: How the circuit is built. This argument is keyword-only.

            - `"reference"` (the default): the eigenvalues are computed classically
              with `eigvalsh`, `e^{iAt}` is a dense `QubitUnitary` computed with
              `expm`, and `t` is tuned to the known spectrum. This mode is fast and
              exact, but the `O(N³)` precomputation and the dense unitary mean that
              its cost does not follow the scaling of HHL.
            - `"honest"`: the circuit is built as it would be on hardware. No
              eigendecomposition sets any circuit parameter: `t` comes from a
              Gershgorin bound, the scaling constant `C` of the inversion comes
              from the resolution of the clock register (see
              `_honest_qpe_params()`), and `e^{iAt}` is Trotterised into elementary
              gates. This mode requires `n_eigenvalue_qubits >= 2`. Input
              validation still runs an `O(N³)` rank check. This check makes the
              solver safer to use and is not part of the algorithm; see `solve()`.
        trotter_steps: The number of Trotter steps for the base evolution time in
            `"honest"` mode. It is ignored in `"reference"` mode. With `"auto"`,
            the default, the solver picks the smallest number of steps for which a
            computable bound on the Trotter drift `‖A_eff − A‖` stays below
            `trotter_tolerance × C`, which is below the eigenvalue resolution of
            the clock register. This matters because, with a fixed step size, the
            circuit is exact HHL for an effective matrix `A_eff` whose eigenvectors
            are rotated even when its eigenvalues are exact. That changes the
            direction of the answer while every check based on the spectrum
            reports success. An integer turns off the automatic choice, and the
            solver warns if the drift bound then exceeds `C/2`. Above that value,
            an extreme eigenvalue can be captured by the Nyquist bucket or wrap
            around the branch cut. The automatic choice is limited by a gate
            budget, and the solver says so when the limit is reached. The default
            is `"auto"`.
        trotter_order: The Trotter order, 1 or 2. The default is `2`. The
            symmetric product of order 2 is a palindrome, so the first-order
            commutator cancels, the error falls from `O(τ)` to `O(τ²)`, and `A_eff`
            stays real for a real `A`. At order 1, `A_eff` picks up `Y`-type terms,
            and a real system returns a complex state. Order 1 is kept for cost
            comparisons.
        trotter_tolerance: The target for `trotter_steps="auto"`. This argument is
            keyword-only. The drift bound must not exceed `trotter_tolerance × C`.
            The default, `0.25`, keeps the drift at a quarter of a bucket, below
            the `C/2` threshold at which the Nyquist bucket can capture an
            eigenvalue. At order 2, the cost grows as `steps ∝ 1/√tolerance`.

    Raises:
        ValueError: If `n_eigenvalue_qubits` is below 1, if `t` is not a positive
            finite number, or if `epsilon` is outside `[0, 1)`.

    Example:
        This example solves a `2 × 2` Hermitian system:

        ```python
        import numpy as np
        from eigenkit import HHL

        A = np.array([[2, 1], [1, 2]], dtype=float)
        b = np.array([1, 0], dtype=float)
        result = HHL(n_eigenvalue_qubits=6).solve(A, b)
        print(result.solution)
        print(result.residual)
        ```

        This example solves the same system on a faster simulator:

        ```python
        from eigenkit import HHL, PennyLaneConfig

        solver = HHL(
            n_eigenvalue_qubits=6,
            provider=PennyLaneConfig(device_name="lightning.qubit"),
        )
        result = solver.solve(A, b)
        ```
    """

    def __init__(
        self,
        n_eigenvalue_qubits: int = 4,
        t: float | None = None,
        epsilon: float = 1e-2,
        provider: ProviderSpec = "numpy",
        *,
        mode: str = "reference",
        trotter_steps: int | Literal["auto"] = "auto",
        trotter_order: int = 2,
        trotter_tolerance: float = _AUTO_TROTTER_TOLERANCE,
    ) -> None:
        if n_eigenvalue_qubits < 1:
            raise ValueError("n_eigenvalue_qubits must be ≥ 1")
        if t is not None and (not math.isfinite(t) or t <= 0.0):
            raise ValueError(
                f"t must be a positive, finite number, got {t}. "
                "t=0 causes division by zero in eigenvalue reconstruction; "
                "a non-finite t propagates NaN/Inf into the evolution unitary."
            )
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        if mode not in ("reference", "honest"):
            raise ValueError(f"mode must be 'reference' or 'honest', got {mode!r}")
        if mode == "honest" and n_eigenvalue_qubits < 2:
            raise ValueError(
                "honest mode needs n_eigenvalue_qubits >= 2 so the clock register "
                "is two-sided (>= 4 buckets); the sign of an eigenvalue is otherwise "
                "unresolvable without diagonalising A."
            )
        if trotter_steps != "auto":
            if isinstance(trotter_steps, bool) or not isinstance(trotter_steps, int):
                raise ValueError(
                    f"trotter_steps must be a positive int or 'auto', got {trotter_steps!r}"
                )
            if trotter_steps < 1:
                raise ValueError(
                    f"trotter_steps must be >= 1 or 'auto', got {trotter_steps}"
                )
        if trotter_order not in (1, 2):
            raise ValueError(f"trotter_order must be 1 or 2, got {trotter_order}")
        if not 0.0 < trotter_tolerance <= 1.0:
            raise ValueError(
                f"trotter_tolerance must be in (0, 1], got {trotter_tolerance}"
            )
        super().__init__(provider=provider)
        self.n_eigenvalue_qubits = n_eigenvalue_qubits
        self.t = t
        self.epsilon = epsilon
        self.mode = mode
        self.trotter_steps = trotter_steps
        self.trotter_order = trotter_order
        self.trotter_tolerance = trotter_tolerance

    def solve(self, A: np.ndarray, b: np.ndarray) -> HHLResult:
        """Solve the linear system `Ax = b`.

        Args:
            A: A real or complex square matrix, which must be invertible. A
                non-Hermitian matrix is converted to the normal equations.
            b: The right-hand side, of length `A.shape[0]`.

        Returns:
            An `HHLResult`. Its `solution` is read from the simulator's statevector and
                completed with a classical least-squares rescaling. See the notes on
                [`HHLResult`][eigenkit.algorithms.HHLResult].

        Raises:
            ValueError: If `A` is not square or is singular, if `b` has the wrong
                length, is zero or is not finite, or if `t` causes aliasing in quantum
                phase estimation (QPE).
            NotImplementedError: If the backend does not implement `statevector()`, or
                if it returns something else, such as a density matrix.

        !!! note
            Validation runs one `O(N³)` singular value decomposition (SVD) to reject a
            singular `A` and to compute a condition number for the warnings. No
            quantity derived from it reaches `t`, `C`, the rotation angles or the
            number of Trotter steps, and changing it would leave the circuit exactly
            the same. `mode="honest"` therefore keeps its defining property, but
            `solve()` itself still does `O(N³)` work.
        """
        from ._provider_checks import check_required_gates

        required_gates = (
            _HHL_HONEST_REQUIRED_GATES if self.mode == "honest" else _HHL_REQUIRED_GATES
        )
        check_required_gates(
            self.provider,
            required_gates,
            algorithm="HHL",
            hint="Switch to a provider that supports these gates (e.g. provider='pennylane').",
        )
        A = np.asarray(A, dtype=complex)
        b = np.asarray(b, dtype=complex)
        sigma_max_input, kappa_input = _validate_inputs(A, b)

        A_h, b_h, is_hermitian = _make_hermitian(A, b, kappa_input)
        is_normal_eq = not is_hermitian

        N_sys = A_h.shape[0]
        N = 1 << math.ceil(math.log2(max(N_sys, 2)))
        A_p = np.zeros((N, N), dtype=complex)
        A_p[:N_sys, :N_sys] = A_h
        # A_h is Hermitian only to floating point, and residual asymmetry becomes an
        # imaginary Pauli coefficient downstream. Identity for exactly Hermitian input.
        A_p = 0.5 * (A_p + A_p.conj().T)
        b_p = np.zeros(N, dtype=complex)
        b_p[:N_sys] = b_h

        b_norm_padded = _safe_norm(b_p)
        if b_norm_padded == 0.0 or not math.isfinite(b_norm_padded):
            raise ValueError(
                "b is a zero vector, or its norm is not representable in float64 "
                "(entries above ~1e154). Rescale b before solving."
            )
        b_unit = _safe_unit(b_p)

        n_b = int(math.log2(N))
        n_l = self.n_eigenvalue_qubits
        n_buckets = 2**n_l

        epsilon_eff = self.epsilon**2 if is_normal_eq else self.epsilon
        is_real_input = _is_effectively_real(A) and _is_effectively_real(b)

        pauli_terms: list[tuple[complex, str]] | None = None
        trotter_steps = 1
        # A†A is PSD and solve() knows it, so the whole phase window can hold positive
        # eigenvalues; see _honest_qpe_params for what that trades.
        one_sided = self.mode == "honest" and is_normal_eq

        # np.square, not `** 2`: the latter raises OverflowError above ~1.34e154 where
        # np.square returns inf, which the isfinite() guards below expect.
        kappa_system = float(np.square(kappa_input)) if is_normal_eq else kappa_input
        sigma_max_system = (
            float(np.square(sigma_max_input)) if is_normal_eq else sigma_max_input
        )
        # σ_min of the circuit Hamiltonian, from the SVD validation already took. Both
        # floors that can swallow it need the circuit parameters, so the check waits.
        sigma_min_system = sigma_max_system / kappa_system if kappa_system > 0 else 0.0

        if self.mode == "honest":
            # Setup first: it rejects bad input on O(nnz) row sums, before the
            # decomposition is paid for.
            A_circuit, t, C_for_angles, lam_max_for_angles, kappa_capacity = (
                self._honest_setup(A_p, one_sided=one_sided)
            )
            pauli_terms = pauli_decompose(A_p)
            trotter_steps = self._resolve_trotter_steps(pauli_terms, t, C_for_angles, n_l)
            self._warn_on_unresolvable_spectrum(
                sigma_min_system,
                C_for_angles,
                lam_max_for_angles,
                epsilon_eff,
                n_l,
                one_sided=one_sided,
                is_normal_eq=is_normal_eq,
            )
        else:
            A_circuit, t, C_for_angles, lam_max_for_angles, kappa_capacity = (
                self._reference_setup(A_p, n_buckets, is_normal_eq, epsilon_eff)
            )

        rotation_angles = _compute_rotation_angles(
            n_buckets,
            t,
            C_for_angles,
            lam_max_for_angles,
            epsilon_eff,
            one_sided=one_sided,
        )

        sv = self._run_circuit(
            A_circuit,
            b_unit,
            n_b,
            n_l,
            t,
            rotation_angles,
            pauli_terms=pauli_terms,
            trotter_steps=trotter_steps,
        )

        b_amps, success_prob, readout_prob = _postselect(sv, n_b, n_l)

        if success_prob < 1e-12:
            warnings.warn(
                "HHL: success probability is near zero. "
                "Consider increasing n_eigenvalue_qubits or checking that A "
                f"is well-conditioned ({self._conditioning_hint(kappa_capacity)}).",
                RuntimeWarning,
                stacklevel=2,
            )
            # Same dtype as the normal path, so conditioning cannot change it.
            solution = np.zeros(A.shape[0], dtype=float if is_real_input else complex)
            return HHLResult(
                solution=solution,
                euclidean_norm=0.0,
                residual=1.0,
                success_probability=0.0,
                readout_probability=0.0,
                raw_solution=np.zeros(A.shape[0], dtype=complex),
            )

        if self.mode == "honest" and is_real_input:
            _warn_on_imaginary_readout(b_amps, self.trotter_order)

        # Real views for real inputs, so the returned dtype is float64.
        A_fit = A.real if is_real_input else A
        b_fit = b.real if is_real_input else b
        raw_quantum = b_amps[: A.shape[0]].copy()
        raw = raw_quantum.real.copy() if is_real_input else raw_quantum

        # The fit runs in a normalised frame: ‖A·raw‖² overflows to inf for ‖A‖ ≳ 1e154.
        # Both scales are non-zero: a zero A is rejected as singular, a zero b as a
        # zero vector.
        A_scale = float(np.max(np.abs(A_fit)))
        b_scale = float(np.max(np.abs(b_fit)))
        Ax_unit = (A_fit / A_scale) @ raw
        denom = float(np.real(np.dot(Ax_unit.conj(), Ax_unit)))
        raw_norm = _safe_norm(raw)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if denom < 1e-28 * max(raw_norm, 1.0) ** 2:
                solution = raw
            else:
                scale = np.dot(Ax_unit.conj(), b_fit / b_scale) / denom
                solution = raw * (scale * (b_scale / A_scale))
            euclidean_norm = _safe_norm(solution)
        b_norm = _safe_norm(b_fit)
        # ‖b‖ > 0 is the only guard needed, and it goes through _safe_norm: an absolute
        # floor rejects good answers at small scale, and an inf denominator would report
        # residual 0.0 for any answer at all.
        residual_is_meaningful = True
        if b_norm > 0.0:
            # errstate: the check below turns the overflow into one clear message.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                residual = float(
                    _safe_norm((A_fit / b_scale) @ solution - b_fit / b_scale)
                    / (b_norm / b_scale)
                )
        else:
            residual = 1.0
        if not math.isfinite(residual) or not math.isfinite(euclidean_norm):
            # Separation beyond what float64 spans, or a solution outside it. Reported,
            # because a NaN here would silently suppress the warning below.
            warnings.warn(
                f"HHL: ‖A‖ ≈ {A_scale:.3g} and ‖b‖ ≈ {b_norm:.3g} are too far apart to "
                "evaluate the residual in float64; it is reported as 1.0 and carries no "
                "information. Rescale the system to a workable range.",
                RuntimeWarning,
                stacklevel=2,
            )
            residual = 1.0
            residual_is_meaningful = False

        if residual_is_meaningful and residual > _RESIDUAL_WARN_THRESHOLD:
            warnings.warn(
                f"HHL: relative residual ‖Ax−b‖/‖b‖ = {residual:.3g} — a rough answer, "
                f"and with κ ≈ {kappa_system:.0f} the direction of x can be off by κ× that "
                "(a small residual bounds the backward error, not the forward one). "
                "Raise n_eigenvalue_qubits; in honest mode a finer Trotter grid is the "
                "other usual cause.",
                RuntimeWarning,
                stacklevel=2,
            )

        return HHLResult(
            solution=solution,
            euclidean_norm=euclidean_norm,
            residual=float(residual),
            success_probability=float(success_prob),
            readout_probability=float(readout_prob),
            raw_solution=raw_quantum,
        )

    def _conditioning_hint(self, kappa_capacity: float) -> str:
        """Text for the near-zero-probability warning.

        In honest mode ``kappa_capacity`` is ``‖A‖_bound / C``, which with an automatic ``t`` is
        identically ``(N−2)/2`` (or ``N−1`` one-sided) — the register's *capacity*, carrying no
        information about ``A``, and a best case at that, since it is only attained when
        ``λ_max`` saturates the Gershgorin bound.
        """
        if self.mode == "honest":
            return (
                f"the clock register resolves κ up to ≈ {kappa_capacity:.0f} at best — "
                "less when the Gershgorin bound is loose — and for a non-Hermitian A it "
                "is κ(A)² that has to fit"
            )
        return f"κ ≈ {kappa_capacity:.1f}"

    def _resolve_trotter_steps(
        self,
        pauli_terms: list[tuple[complex, str]],
        t: float,
        C: float,
        n_clock: int,
    ) -> int:
        """Number of Trotter steps for the base time ``t`` in honest mode.

        With ``trotter_steps="auto"`` this solves the drift criterion
        ``trotter_drift_bound(τ) ≤ trotter_tolerance · C`` for the smallest
        integer ``steps = ⌈t/τ⌉``, which is closed-form because the bound is
        ``τ·Γ₁`` at order 1 and ``τ²·Γ₂`` at order 2.  With an explicit
        ``trotter_steps`` it only checks the criterion and warns.

        Selection is additionally floored by the angle guard
        ``τ·Σ|c_k| ≤ _TROTTER_ANGLE_GUARD`` (outside which the bound itself stops
        being trustworthy) and capped by ``_AUTO_TROTTER_GATE_BUDGET``.

        Everything is evaluated on coefficients normalised to ``max|c_k| = 1``.  The
        criterion is homogeneous in that scale, so this changes no answer, and it keeps
        ``Γ₂ ∝ scale³`` inside float64: computed directly it overflows above
        ``‖A‖ ≈ 1e103`` and underflows to ``0.0`` below ~1e-110.
        """
        unit = max((abs(complex(c)) for c, _ in pauli_terms), default=0.0)
        if unit == 0.0:
            return 1 if not isinstance(self.trotter_steps, int) else self.trotter_steps
        unit_terms = [(complex(c) / unit, s) for c, s in pauli_terms]
        # Γ̂ on unit-scaled coefficients: Γ₁ = unit²·Γ̂₁, Γ₂ = unit³·Γ̂₂.
        gamma_hat = trotter_drift_bound(unit_terms, 1.0, self.trotter_order)
        coeff_l1_hat = float(sum(abs(complex(c)) for c, _ in unit_terms))

        def drift(steps: int) -> float:
            # Grouped so no intermediate leaves the scale of a physical eigenvalue.
            tau_unit = (t / steps) * unit
            if self.trotter_order == 1:
                return tau_unit * unit * gamma_hat
            return tau_unit * tau_unit * unit * gamma_hat

        if isinstance(self.trotter_steps, int):
            steps = self.trotter_steps
            if gamma_hat > 0.0 and drift(steps) > 0.5 * C:
                needed = self._steps_from_criterion(gamma_hat, coeff_l1_hat, unit, t, C)
                warnings.warn(
                    f"HHL(mode='honest'): the Trotter drift bound ‖A_eff − A‖ ≲ "
                    f"{drift(steps):.3g} exceeds C/2 = {0.5 * C:.3g} at "
                    f"trotter_steps={steps}, trotter_order={self.trotter_order}. The "
                    "circuit is then exact HHL for a perturbed A_eff: an extremal "
                    "eigenvalue can be captured by the sign-ambiguous Nyquist bucket, and "
                    "the eigenvectors rotate, corrupting the direction of the solution "
                    f"without changing its spectrum. Use trotter_steps={needed} or 'auto'.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return steps

        if gamma_hat <= 0.0:
            # All Pauli terms commute pairwise — the Trotter product is exact.
            return 1
        steps = self._steps_from_criterion(gamma_hat, coeff_l1_hat, unit, t, C)
        budget_steps = _max_steps_within_budget(pauli_terms, n_clock, self.trotter_order)
        if steps > budget_steps:
            warnings.warn(
                f"HHL(mode='honest'): trotter_steps='auto' wanted {steps} steps to hold "
                f"the drift under {self.trotter_tolerance:g}·C, past the "
                f"{_AUTO_TROTTER_GATE_BUDGET:,}-gate budget; clamped to {budget_steps}, "
                f"leaving a drift of {drift(budget_steps) / C:.2g}·C, so the solution "
                f"direction may be corrupted. Pass trotter_steps={steps} to accept the "
                "cost, or raise trotter_tolerance to accept the drift.",
                RuntimeWarning,
                stacklevel=3,
            )
            return budget_steps
        return steps

    def _steps_from_criterion(
        self, gamma_hat: float, coeff_l1_hat: float, unit: float, t: float, C: float
    ) -> int:
        """Smallest ``steps`` meeting the drift criterion and the angle guard.

        ``gamma_hat`` and ``coeff_l1_hat`` are evaluated on coefficients scaled to
        ``max|c_k| = 1``; ``unit`` is the scale that was divided out.  Solving

            order 1:  (t/steps)·unit²·Γ̂₁ ≤ tol·C
            order 2:  (t/steps)²·unit³·Γ̂₂ ≤ tol·C

        for ``steps`` and grouping the factors as ``(t·unit)`` and ``(unit/target)``
        keeps every intermediate at the scale of a physical eigenvalue ratio, so the
        result is exact at any magnitude of ``A``.
        """
        target = self.trotter_tolerance * C
        t_unit = t * unit  # dimensionless: t ∝ 1/‖A‖ and unit ∝ ‖A‖
        if self.trotter_order == 1:
            steps = math.ceil(t_unit * gamma_hat * (unit / target))
        else:
            steps = math.ceil(t_unit * math.sqrt(gamma_hat * (unit / target)))
        if coeff_l1_hat > 0.0:
            steps = max(steps, math.ceil(t_unit * coeff_l1_hat / _TROTTER_ANGLE_GUARD))
        return max(1, steps)

    @staticmethod
    def _warn_on_unresolvable_spectrum(
        sigma_min: float,
        C: float,
        lam_max_bound: float,
        epsilon: float,
        n_clock: int,
        *,
        one_sided: bool,
        is_normal_eq: bool,
    ) -> None:
        """Warn when the smallest eigenvalue falls below either read-out floor.

        Honest mode has **two** independent floors, and a component below either is attenuated
        or dropped — always the one with the largest ``1/λ`` weight, i.e. the direction ``x``
        depends on most:

        * the **register** floor: an eigenvalue rounds to bucket ``round(λ/C)``, so below
          ``C/2`` it lands in the rotation-free bucket 0 and is dropped outright.  Between
          ``C/2`` and ``C`` it is read as ``C``, under-weighting it by ``λ/C`` — at most 2×, a
          distortion the rough-residual warning picks up if it matters.  Only the hard cliff
          warns, because warning at ``λ ≲ C`` fires on answers still correct to four decimals.
        * the **epsilon** floor ``epsilon·‖A‖_bound``: the inversion table zeroes every bucket
          under it outright.

        Both floors are compared against the true ``σ_min``, and both scale with
        ``‖A‖_bound``, so the looseness of that bound is part of the answer.

        The recommendation matters as much as the warning: past ``κ ≈ 1/epsilon`` no clock
        register helps, since ``epsilon·‖A‖_bound`` does not shrink with ``n_clock``, so this
        names whichever floor binds and both when both do.
        """
        if sigma_min <= 0.0 or not math.isfinite(sigma_min):
            return
        epsilon_floor = epsilon * lam_max_bound
        # C/2 is where λ rounds into bucket 0, i.e. the hard cliff; the epsilon cutoff
        # is a hard cliff wherever it sits, so it is compared directly.
        register_binds = sigma_min < 0.5 * C
        epsilon_binds = sigma_min < epsilon_floor
        if not (register_binds or epsilon_binds):
            return

        remedies: list[str] = []
        if register_binds:
            # C ∝ 1/N, and the target is full resolution (C ≤ σ_min), not the bucket-0
            # cliff: landing on the cliff still costs half the component.
            ratio = lam_max_bound / sigma_min
            needed = ratio + 1.0 if one_sided else 2.0 * ratio + 2.0
            remedies.append(
                f"raise n_eigenvalue_qubits to ≥ {max(2, math.ceil(math.log2(needed)))}"
            )
        if epsilon_binds:
            # epsilon·‖A‖_bound does not depend on n_clock: only epsilon can move it.
            safe = 0.5 * sigma_min / lam_max_bound
            if is_normal_eq:
                safe = math.sqrt(safe)  # the user-facing epsilon is squared internally
            remedies.append(
                f"lower epsilon to ≲ {safe:.2g} (no register size fixes this)"
            )

        floors = " and ".join(
            part
            for part in (
                f"half the resolution floor, C/2 = {0.5 * C:.3g}, so it rounds into the "
                "rotation-free bucket 0"
                if register_binds
                else "",
                f"the epsilon cutoff {epsilon:g}·‖A‖_bound = {epsilon_floor:.3g}"
                if epsilon_binds
                else "",
            )
            if part
        )
        warnings.warn(
            f"HHL(mode='honest'): σ_min ≈ {sigma_min:.3g} is below {floors}, so that "
            "component of b is attenuated or dropped — and it carries the largest 1/λ "
            "weight, so it is what the direction of x depends on most. At "
            f"n_eigenvalue_qubits={n_clock}: " + "; ".join(remedies) + ".",
            RuntimeWarning,
            stacklevel=3,
        )

    def _reference_setup(
        self,
        A_p: np.ndarray,
        n_buckets: int,
        is_normal_eq: bool,
        epsilon_eff: float,
    ) -> tuple[np.ndarray, float, float, float, float]:
        """Reference-mode setup: classical eigendecomposition drives t and C.

        Returns ``(A_circuit, t, C_for_angles, lam_max_for_angles, kappa)``.
        This is the simulator-only path; it calls ``eigvalsh``.
        """
        eigs = np.linalg.eigvalsh(A_p)
        abs_eigs = np.abs(eigs)
        lam_max_abs = float(abs_eigs.max())

        threshold_abs = epsilon_eff * lam_max_abs
        sig_eigs = eigs[abs_eigs > threshold_abs]
        # The exclusion warning applies to Hermitian input too: the epsilon filter
        # drops small eigenvalues, and those carry the largest 1/λ weights, so the
        # returned vector solves a reduced-rank system either way.
        n_structural = int((abs_eigs > 1e-15 * lam_max_abs).sum())
        if len(sig_eigs) < n_structural:
            kind = "singular direction(s) of A" if is_normal_eq else "eigen-direction(s)"
            warnings.warn(
                f"HHL: {n_structural - len(sig_eigs)} {kind} "
                f"excluded by epsilon={self.epsilon}. "
                "The solution approximates a reduced-rank system. "
                "Decrease epsilon or improve conditioning to include them.",
                RuntimeWarning,
                stacklevel=3,
            )
        if len(sig_eigs) == 0:
            raise ValueError(
                "No eigenvalues above the epsilon threshold. "
                "A may be singular or ill-conditioned."
            )
        lam_min_abs = float(np.abs(sig_eigs).min())
        C_actual = lam_min_abs

        kappa = lam_max_abs / lam_min_abs
        if kappa > 100:
            warnings.warn(
                f"HHL: condition number κ = {kappa:.1f} is large. "
                "HHL accuracy degrades with κ: the success probability scales as "
                "O(1/κ) and the solution residual typically grows with κ. "
                f"Consider preconditioning A, or use a direct solver for κ > 100.",
                RuntimeWarning,
                stacklevel=3,
            )

        has_negative = bool((sig_eigs < 0).any())
        t_max = math.pi * (n_buckets - 2) / n_buckets if has_negative else math.pi
        if self.t is not None and self.t * lam_max_abs > t_max:
            safe_t = t_max / lam_max_abs
            raise ValueError(
                f"t={self.t!r} causes QPE aliasing for this spectrum: "
                f"t·λ_max = {self.t * lam_max_abs:.6g} > t_max = {t_max:.6g}. "
                f"Use t ≤ {safe_t:.6g} or leave t=None to set it automatically."
            )

        if self.t is None:
            A_circuit = A_p / lam_max_abs
            lam_ratios = np.abs(sig_eigs) / lam_max_abs
            t = _choose_t(lam_ratios, n_buckets, has_negative=has_negative)
            lam_max_for_angles = 1.0
            C_for_angles = C_actual / lam_max_abs
        else:
            A_circuit = A_p
            t = self.t
            lam_max_for_angles = lam_max_abs
            C_for_angles = C_actual
        return A_circuit, t, C_for_angles, lam_max_for_angles, kappa

    def _honest_setup(
        self, A_p: np.ndarray, *, one_sided: bool = False
    ) -> tuple[np.ndarray, float, float, float, float]:
        """Honest-mode setup: no eigendecomposition, no eigenvalue-tuned time.

        Returns ``(A_circuit, t, C_for_angles, lam_max_bound, kappa_capacity)``.
        ``t`` and ``C`` come from `_honest_qpe_params()` (Gershgorin bound +
        register resolution); ``A`` is evolved unnormalised (the Trotter block
        handles the scale) and the last element is the register's resolvable
        condition number, a post-hoc capacity figure — not an estimate of κ(A).

        ``one_sided=True`` declares the circuit Hamiltonian PSD (the
        normal-equations lift), which lets the whole phase window hold positive
        eigenvalues and halves ``C``.
        """
        t, C_for_angles, lam_max_bound = _honest_qpe_params(
            A_p, self.n_eigenvalue_qubits, one_sided=one_sided, t=self.t
        )
        return A_p, t, C_for_angles, lam_max_bound, lam_max_bound / C_for_angles

    def _build_circuit_spec(
        self,
        A: np.ndarray,
        b_unit: np.ndarray,
        n_b: int,
        n_l: int,
        t: float,
        rotation_angles: dict[int, float],
        pauli_terms: list[tuple[complex, str]] | None = None,
        trotter_steps: int | None = None,
    ) -> CircuitSpec:
        """Describe the full HHL circuit as a provider-agnostic CircuitSpec.

        Qubit layout:

        ```text
        [ancilla(0) | clock(1..n_l) | b_reg(n_l+1..n_l+n_b)]
        ```

        The QPE sub-blocks differ by ``mode`` (dense ``QPEBlock`` for
        ``"reference"``, Trotterised ``HamiltonianQPEBlock`` for ``"honest"``);
        everything else — the ``|b⟩`` preparation and the controlled-RY
        eigenvalue inversion — is identical.

        ``pauli_terms`` and ``trotter_steps`` carry down what
        [`solve()`][eigenkit.algorithms.HHL.solve] already
        computed: the decomposition is the honest path's dominant classical cost, and
        ``trotter_steps`` may have been auto-selected.  ``pauli_terms`` is re-derived
        if omitted; ``trotter_steps`` is not, because guessing it would mean silently
        building a grid the selector would have rejected.
        """
        n_total = 1 + n_l + n_b
        ancilla = 0
        clock = tuple(range(1, 1 + n_l))
        b_reg = tuple(range(1 + n_l, 1 + n_l + n_b))

        gates: list[Gate] = []

        for g in AmplitudeEncoding(b_unit, normalize=False).build(n_b).gates:
            gates.append(
                Gate(
                    name=g.name,
                    qubits=tuple(b_reg[q] for q in g.qubits),
                    params=g.params,
                    data=g.data,
                    adjoint=g.adjoint,
                )
            )

        gates.extend(
            self._qpe_gates(
                A,
                n_b,
                n_l,
                t,
                inverse=False,
                pauli_terms=pauli_terms,
                trotter_steps=trotter_steps,
            )
        )

        gates.extend(_uniformly_controlled_ry(rotation_angles, n_l, clock, ancilla))

        gates.extend(
            self._qpe_gates(
                A,
                n_b,
                n_l,
                t,
                inverse=True,
                pauli_terms=pauli_terms,
                trotter_steps=trotter_steps,
            )
        )

        return CircuitSpec(
            n_qubits=n_total,
            n_weights=0,
            n_inputs=0,
            gates=tuple(gates),
        )

    def _qpe_gates(
        self,
        A: np.ndarray,
        n_b: int,
        n_l: int,
        t: float,
        inverse: bool,
        pauli_terms: list[tuple[complex, str]] | None = None,
        trotter_steps: int | None = None,
    ) -> list[Gate]:
        """QPE (or inverse-QPE) gates for ``e^{iAt}``, remapped past the ancilla.

        Both blocks use the local layout ``[clock(0..n_l-1) | state(n_l..)]``;
        every qubit index is shifted by ``+1`` to skip the HHL ancilla (qubit 0).
        """
        if self.mode == "honest":
            if pauli_terms is None:
                pauli_terms = pauli_decompose(A)
            if trotter_steps is None:
                if not isinstance(self.trotter_steps, int):
                    raise ValueError(
                        "trotter_steps='auto' has to be resolved before the circuit is "
                        "built; solve() does that and passes the result down. Calling "
                        "this directly means passing an explicit trotter_steps, because "
                        "defaulting to 1 here would silently build the coarse grid the "
                        "selector exists to avoid."
                    )
                trotter_steps = self.trotter_steps
            frag = HamiltonianQPEBlock(
                pauli_terms,
                n_clock=n_l,
                time=t,
                steps=trotter_steps,
                order=self.trotter_order,
                inverse=inverse,
            ).build(n_l + n_b)
        else:
            import scipy.linalg

            U_base = scipy.linalg.expm(1j * A * t)
            oracle_spec = assemble([QubitUnitary(U_base, wires=tuple(range(n_b)))])
            frag = QPEBlock(oracle_spec, n_clock=n_l, inverse=inverse).build(n_l + n_b)

        return [
            Gate(
                name=g.name,
                qubits=tuple(q + 1 for q in g.qubits),
                params=g.params,
                data=g.data,
                adjoint=g.adjoint,
            )
            for g in frag.gates
        ]

    def _run_circuit(
        self,
        A: np.ndarray,
        b_unit: np.ndarray,
        n_b: int,
        n_l: int,
        t: float,
        rotation_angles: dict[int, float],
        pauli_terms: list[tuple[complex, str]] | None = None,
        trotter_steps: int | None = None,
    ) -> np.ndarray:
        """Build and execute the HHL circuit via the IR, returning the statevector."""
        spec = self._build_circuit_spec(
            A,
            b_unit,
            n_b,
            n_l,
            t,
            rotation_angles,
            pauli_terms=pauli_terms,
            trotter_steps=trotter_steps,
        )
        circuit = Circuit(spec, provider=self.provider)
        if not callable(getattr(circuit.compiled, "statevector", None)):
            raise NotImplementedError(
                "HHL requires statevector() support, but the selected provider "
                f"'{circuit.provider.name}' does not implement it. "
                "Use a statevector simulator such as provider='pennylane' "
                "(default.qubit or lightning.qubit)."
            )
        # Via the public statevector API, not circuit.compiled.
        sv = np.asarray(circuit.statevector(np.zeros(0, dtype=np.float64)))
        expected = 1 << (1 + n_l + n_b)
        if sv.ndim != 1 or sv.shape[0] != expected:
            raise NotImplementedError(
                "HHL requires a pure-statevector backend, but provider "
                f"'{circuit.provider.name}' returned shape {sv.shape} where a "
                f"statevector of length {expected} was expected. Density-matrix devices "
                "cannot supply the amplitude slice the solution is read from; use e.g. "
                "PennyLane 'default.qubit' or 'lightning.qubit'."
            )
        return sv


def _safe_norm(v: np.ndarray) -> float:
    """Euclidean norm that survives entries near the ends of float64.

    ``np.linalg.norm`` squares before summing, so it silently overflows above ~1.3e154 and
    underflows on subnormals.  Factoring the largest entry out first keeps the whole
    representable range usable.
    """
    peak = float(np.max(np.abs(v))) if v.size else 0.0
    if peak == 0.0 or not math.isfinite(peak):
        return peak if math.isfinite(peak) else float("inf")
    # An exact power of two, not `peak` itself: dividing a denormal array by a denormal
    # scalar goes through complex division and yields inf+nanj.  Clamped, because a
    # denormal peak asks for 2^1062 and ±1000 already reaches the normal range.
    shift = max(-1000, min(1000, -math.frexp(peak)[1]))
    scaled = float(np.linalg.norm(v * math.ldexp(1.0, shift)))
    return math.ldexp(scaled, -shift)


def _safe_unit(v: np.ndarray) -> np.ndarray:
    """``v / ‖v‖`` for vectors whose entries sit near the ends of float64.

    Dividing a denormal array by its own denormal norm goes through complex division and
    returns ``nan``; a huge one divides by ``inf`` and returns zeros.  Pulling the array
    into the normal range first, by an exact power of two, makes both operands ordinary.
    """
    norm = _safe_norm(v)
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("cannot normalise a zero or non-finite vector")
    shift = max(-1000, min(1000, -math.frexp(norm)[1]))
    scaled = v * math.ldexp(1.0, shift)
    return scaled / float(np.linalg.norm(scaled))


def _is_effectively_real(M: np.ndarray) -> bool:
    """Is ``M`` real up to floating-point noise *relative to its own scale*?

    Scaling to ``max|M|`` makes the answer invariant under ``M → αM``.  An absolute
    tolerance (``np.allclose(M.imag, 0)``, ``atol=1e-8``) calls a genuinely complex
    matrix of magnitude 1e-9 real.
    """
    scale = float(np.max(np.abs(M))) if M.size else 0.0
    if scale == 0.0:
        return True
    return bool(float(np.max(np.abs(M.imag))) <= 1e-12 * scale)


def _validate_inputs(A: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A.shape}")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError(
            f"b must be a 1-D vector of length {A.shape[0]}, got shape {b.shape}"
        )
    if not np.all(np.isfinite(A)):
        raise ValueError(
            "A contains non-finite values (NaN or Inf). "
            "HHL requires a finite invertible matrix."
        )
    if not np.all(np.isfinite(b)):
        raise ValueError(
            "b contains non-finite values (NaN or Inf). "
            "HHL requires a finite right-hand side vector."
        )
    # One SVD serves both the singularity guard and the conditioning hint; it decides
    # messages and warnings only, never a circuit parameter.
    if A.size == 0:
        raise ValueError("A is empty; HHL requires a square matrix of size >= 1.")
    singular_values = np.linalg.svd(A, compute_uv=False)
    sigma_max = float(singular_values.max())
    tol = sigma_max * max(A.shape) * float(np.finfo(float).eps)
    rank = int((singular_values > tol).sum())
    if rank < A.shape[0]:
        raise ValueError(
            f"A is singular (rank {rank} < {A.shape[0]}). "
            "HHL requires an invertible matrix."
        )
    return sigma_max, sigma_max / float(singular_values.min())


def _make_hermitian(
    A: np.ndarray, b: np.ndarray, kappa: float
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Lift non-Hermitian A to the normal equations A†Ax = A†b.

    Returns ``(A_h, b_h, is_hermitian)``. When A is already Hermitian the inputs
    pass through unchanged (``is_hermitian=True``); the caller derives
    ``is_normal_eq`` from this flag, so the Hermitian test lives in one place.

    The comparison uses an absolute tolerance scaled to `‖A‖`.  A purely relative
    test (`atol=0`) calls a matrix non-Hermitian when a *structural zero* faces a
    partner of size 1e-18, which routinely happens when `A` is assembled as
    `BᵀDB` or element by element.

    !!! warning
        Forming the normal equations squares the condition number:
        κ(A†A) = κ(A)².  Provide a Hermitian A directly when possible.
    """
    scale = float(np.max(np.abs(A))) if A.size else 0.0
    if np.allclose(A, A.conj().T, atol=1e-12 * scale, rtol=1e-10):
        return A, b, True
    AH = A.conj().T
    if kappa > 10.0:
        warnings.warn(
            f"HHL: A is not Hermitian (κ(A) ≈ {kappa:.1f}). "
            f"The normal equations A†Ax = A†b are formed automatically, "
            f"squaring the condition number to κ(A†A) ≈ {kappa**2:.0f}. "
            "Consider providing a Hermitian formulation of A directly, "
            "or preconditioning A before calling solve().",
            RuntimeWarning,
            stacklevel=3,
        )
    return AH @ A, AH @ b, False


def _compute_rotation_angles(
    n_buckets: int,
    t: float,
    C_actual: float,
    lam_max_abs: float,
    epsilon: float,
    *,
    one_sided: bool = False,
) -> dict[int, float]:
    """Controlled-RY angle for every clock bucket: ``θ_j = 2·asin(C/λ̃_j)``.

    Two-sided (the default) reads buckets above ``N/2`` as negative eigenvalues, which is
    what an indefinite spectrum needs.  Bucket ``N/2`` (Nyquist) is
    treated as positive; ``_choose_t`` / `_honest_qpe_params()` size ``t`` so
    that no true eigenvalue lands there.

    ``one_sided=True`` declares the spectrum non-negative (the normal-equations
    lift), so every bucket ``1 … N−1`` maps to a positive eigenvalue and the full
    ``[0, 2π)`` phase window is available — halving ``C`` for the same ``N``.

    !!! note
        The table has one entry per bucket, so the *classical description* of the
        inversion stage is ``Θ(2^{n_l})``.  See the note in [`HHL`][eigenkit.algorithms.HHL] on why
        that does not change the complexity class but does cost uniformity.
    """
    angles: dict[int, float] = {}
    threshold = epsilon * lam_max_abs
    half = n_buckets // 2
    for j in range(1, n_buckets):
        j_signed = j if one_sided or j <= half else j - n_buckets
        lam_j_est = 2.0 * math.pi * j_signed / (t * n_buckets)
        if abs(lam_j_est) < threshold:
            angles[j] = 0.0
        else:
            ratio = max(-1.0, min(C_actual / lam_j_est, 1.0))
            angles[j] = 2.0 * math.asin(ratio)
    return angles


def _warn_on_imaginary_readout(b_amps: np.ndarray, trotter_order: int) -> None:
    """Non-spectral Trotter-error detector for real inputs.

    For real ``A`` and ``b`` the exact HHL state is real up to a global phase.  Anything
    left over is Trotter error in the *eigenvectors* — the channel every spectrum-based
    check is blind to, since the fixed-``τ`` deformation can be exactly isospectral while
    rotating the eigenbasis by tens of degrees.  Costs ``O(N)`` and needs no spectrum.
    """
    amps = np.asarray(b_amps)
    if amps.size == 0:
        return
    pivot = int(np.argmax(np.abs(amps)))
    if abs(amps[pivot]) < 1e-14:
        return
    phased = amps * np.exp(-1j * np.angle(amps[pivot]))
    norm = float(np.linalg.norm(phased))
    if norm < 1e-14:
        return
    frac = float(np.linalg.norm(phased.imag) / norm)
    if frac <= 0.05:
        return
    warnings.warn(
        f"HHL(mode='honest'): the post-selected state is {frac:.1%} imaginary after the "
        "global phase is removed, but A and b are real, so the exact answer is real. That "
        "is Trotter error in the eigenvectors: the eigenvalues, and every spectrum-based "
        "check, still look correct. "
        + (
            "Use trotter_order=2, which removes this channel structurally."
            if trotter_order == 1
            else "Raise trotter_steps, or lower trotter_tolerance."
        ),
        RuntimeWarning,
        stacklevel=3,
    )


def _honest_phase_window(n_buckets: int, *, one_sided: bool = False) -> float:
    """Largest ``t·‖A‖`` the clock register can represent without aliasing.

    Two-sided: ``π(N−2)/N`` keeps every phase inside ``(−π, π)`` *and* off the
    sign-ambiguous Nyquist bucket ``N/2``, leaving a half-bucket guard band.
    One-sided (spectrum known non-negative): ``2π(N−1)/N`` uses the full circle.
    """
    if one_sided:
        return 2.0 * math.pi * (n_buckets - 1) / n_buckets
    return math.pi * (n_buckets - 2) / n_buckets


def _estimate_trotter_gate_count(
    pauli_terms: list[tuple[complex, str]] | tuple[tuple[float, str], ...],
    n_clock: int,
    steps: int,
    order: int,
) -> int:
    """Gate count of the two controlled-evolution ladders, without building them.

    A controlled Pauli exponential over ``w`` active (non-identity) qubits emits **two**
    basis-change gates per X *or* Y factor, ``2(w−1)`` ladder CNOTs, and the controlled RZ
    as ``RZ·CNOT·RZ·CNOT``.  An all-identity string emits one ``PhaseShift``.  Both QPE
    stages run ``steps·(2^{n_clock} − 1)`` Trotter steps of ``order``-many sweeps.

    X and Y cost the same because `_append_pauli_exp()`
    rotates a Y factor with a single ``RX(+pi/2)`` in and ``RX(-pi/2)`` out, not with the
    textbook two-gate ``H·S†`` / ``S·H`` pair.  Counting that pair's "four gates per Y"
    would overcount by 2 per Y factor per applied term and clamp
    `HHL._resolve_trotter_steps()` to a smaller grid than the budget allows.  Exactness
    against the emitted gate list is pinned by a test.
    """
    per_step = 0
    for _, pauli_string in pauli_terms:
        active = [ch for ch in pauli_string if ch != "I"]
        if not active:
            per_step += 1
            continue
        basis = 2 * sum(1 for ch in active if ch in ("X", "Y"))
        per_step += basis + 2 * (len(active) - 1) + 4
    sweeps = 1 if order == 1 else 2
    ladder_steps = steps * ((1 << n_clock) - 1)
    return 2 * sweeps * ladder_steps * per_step


def _max_steps_within_budget(
    pauli_terms: list[tuple[complex, str]] | tuple[tuple[float, str], ...],
    n_clock: int,
    order: int,
) -> int:
    """Largest ``steps`` whose estimated gate count fits the auto-selection budget."""
    unit = _estimate_trotter_gate_count(pauli_terms, n_clock, 1, order)
    if unit <= 0:
        return 1
    return max(1, _AUTO_TROTTER_GATE_BUDGET // unit)


def _gershgorin_spectral_radius(A: np.ndarray) -> float:
    """Upper bound on ``max|eigenvalue|`` via Gershgorin discs: the maximum absolute row sum.

    Costs ``O(nnz)`` — no eigendecomposition, which is what lets the honest path pick an
    evolution time without the ``O(N³)`` shortcut that would defeat its scaling.
    """
    return float(np.max(np.sum(np.abs(A), axis=1)))


def _honest_qpe_params(
    A: np.ndarray, n_clock: int, *, one_sided: bool = False, t: float | None = None
) -> tuple[float, float, float]:
    """Spectral-free ``(t, C, lam_max_bound)`` for the honest HHL path.

    No knowledge of the spectrum is used:

    * ``lam_max_bound`` — Gershgorin upper bound on ``|λ|_max``.
    * ``t`` — from that bound, sized for a two-sided phase window so it holds whether or
      not ``A`` is indefinite: ``t = π(N−2)/N / lam_max_bound`` keeps every phase inside
      ``(−π, π)`` and off the sign-ambiguous Nyquist bucket ``N/2``.
    * ``C`` — the rotation constant, the smallest eigenvalue the register resolves,
      ``2π/(t·N)``.  Any ``C ≤ λ_min`` is correct; this is the safe spectrum-free choice.
      ``lam_max_bound / C`` is the register's *capacity*, not an estimate of ``κ(A)``.

    ``one_sided=True`` is used only when the caller knows the spectrum is non-negative —
    currently the ``A†A`` lift.  The window becomes the full ``[0, 2π)``, so
    ``C = lam_max_bound/(N−1)`` instead of ``2·lam_max_bound/(N−2)``: about one extra bit
    of resolution.  It is not free — the ancilla amplitude is ``C/λ``, so a smaller ``C``
    lowers the post-selection rate and roughly doubles the auto-selected Trotter depth.
    What it buys is reach, for components a two-sided window would drop into bucket 0.

    !!! note
        The guard band is computed on ``A`` while QPE sees the Trotterised ``A_eff``, so a
        drift of order ``C`` can still cross the branch cut — which is what
        `HHL._resolve_trotter_steps()` bounds.  A one-sided *table* cannot invert a
        sign, but a phase drifting below zero wraps to bucket ``N−1`` and the component
        with the largest ``1/λ`` weight is handed the smallest.  Under
        ``trotter_steps="auto"`` that is unreachable: wrapping needs drift past ``C/2``
        and the selector holds it at ``C/4``.

    Requires ``n_clock ≥ 2`` (a two-sided register needs at least 4 buckets).
    """
    n_buckets = 2**n_clock
    if n_buckets < 4:
        raise ValueError(
            "honest HHL needs n_eigenvalue_qubits >= 2 so the clock register is "
            "two-sided (>= 4 buckets); the sign of an eigenvalue is otherwise "
            "unresolvable without diagonalising A."
        )
    lam_max_bound = _gershgorin_spectral_radius(A)
    if not math.isfinite(lam_max_bound) or lam_max_bound <= 0.0:
        raise ValueError(
            "A has a zero or non-finite Gershgorin radius; cannot choose an "
            "evolution time. A must be a finite, non-zero matrix."
        )
    window = _honest_phase_window(n_buckets, one_sided=one_sided)
    if t is None:
        t = window / lam_max_bound
    # 1e-12 relative slack: t = window/‖A‖_bound is the documented maximum, and
    # multiplying it back by ‖A‖_bound can land one ulp above the window.
    elif t * lam_max_bound > window * (1.0 + 1e-12):
        raise ValueError(
            f"t={t!r} causes QPE aliasing: t·‖A‖_bound = "
            f"{t * lam_max_bound:.6g} > t_max = {window:.6g}. "
            f"Use t ≤ {window / lam_max_bound:.6g} or leave t=None to set it "
            "automatically."
        )
    C = 2.0 * math.pi / (t * n_buckets)
    return t, C, lam_max_bound


def _postselect(
    sv: np.ndarray,
    n_b: int,
    n_l: int,
) -> tuple[np.ndarray, float, float]:
    """Post-select on ancilla = |1⟩ and project onto the clock = |0⟩ subspace.

    Qubit layout: [ancilla(MSB) | clock | b_reg(LSB)].

    Inverse QPE returns the clock exactly to |0⟩ only when every eigenvalue is
    representable in ``n_l`` bits; otherwise a small spread remains.  The |0⟩ slice is the
    one branch where the map is diagonal in the eigenbasis with real weights, so it is
    projected out and the rest discarded: the other branches are not copies of ``x`` but
    differently phased Dirichlet products mixing eigencomponents.

    It is not an *unbiased* slice: the weight kept for ``λ_j`` depends on how commensurate
    ``λ_j`` is with the bucket grid (exact on grid, ~9 % low at half-offset), so a small
    non-monotonic wobble in the residual as ``n_l`` grows is expected by construction.

    Returns:
        b_reg_amps: Normalised complex amplitudes of length ``2^n_b`` from the clock=|0⟩
            slice; complex amplitudes preserve the sign of ``1/λ_j``.
        success_prob: Weight on ancilla=|1⟩ over all clock states.
        readout_prob: Weight on ancilla=|1⟩ **and** clock=|0…0⟩ — the slice the solution
            is read from, and the true post-selection rate.  Always ≤ ``success_prob``.
    """
    n_total = 1 + n_l + n_b
    ancilla_mask = 1 << (n_total - 1)  # ancilla is MSB
    clock_mask = ((1 << n_l) - 1) << n_b  # clock sits above the b_reg (LSB) bits
    n_b_dim = 2**n_b

    indices = np.arange(len(sv))
    ancilla_one = (indices & ancilla_mask) != 0

    # Physical post-selection probability: weight on ancilla=|1⟩ (all clocks).
    success_prob = float(np.sum(np.abs(sv[ancilla_one]) ** 2))

    # Read the solution from the clock=|0⟩ slice. Each b-register value occurs
    # exactly once there, so this is a direct projection, not a sum over branches.
    selected = indices[ancilla_one & ((indices & clock_mask) == 0)]
    b_reg_amps = np.zeros(n_b_dim, dtype=complex)
    b_reg_amps[selected % n_b_dim] = sv[selected]

    # The joint (ancilla=1 ∧ clock=0) weight — what a device post-selects on.
    readout_prob = float(np.sum(np.abs(b_reg_amps) ** 2))

    norm = _safe_norm(b_reg_amps)
    if norm > 1e-14:
        b_reg_amps /= norm
    else:
        b_reg_amps = np.zeros(n_b_dim, dtype=complex)

    return b_reg_amps, success_prob, readout_prob


def _choose_t(
    lam_ratios: np.ndarray, n_buckets: int, has_negative: bool = False
) -> float:
    r"""Choose evolution time t to maximise QPE bucket alignment.

    Scores candidates \(t = 2\pi m / (N\,|\tilde\lambda_j|)\) by total squared
    fractional bucket error and returns the best.

    For sign-indefinite spectra t is capped at \(\pi (N-2)/N\) to keep
    eigenvalues off the Nyquist bucket N/2: that bucket maps to both +π/t
    and −π/t and is resolved as positive, so a large negative eigenvalue
    landing there gets the wrong 1/λ sign.
    """
    N = n_buckets
    if has_negative and N < 4:
        raise ValueError(
            "A has negative eigenvalues but the QPE register has only "
            f"{N} bucket(s); the single non-zero bucket is sign-ambiguous. "
            "Use n_eigenvalue_qubits >= 2."
        )
    t_max = math.pi * (N - 2) / N if has_negative else math.pi

    best_t = t_max
    best_score = _t_score(t_max, lam_ratios, N)

    for lam in lam_ratios:
        if abs(lam) < 1e-12:
            continue
        for m in range(1, N // 2):
            t_cand = 2.0 * math.pi * m / (N * abs(lam))
            if t_cand > t_max + 1e-12:
                continue
            score = _t_score(t_cand, lam_ratios, N)
            if score < best_score - 1e-12:
                best_score, best_t = score, t_cand

    return best_t


def _t_score(t: float, lam_ratios: np.ndarray, n_buckets: int) -> float:
    """Sum of squared fractional-bucket errors for all eigenvalues at time t.

    Bucket 0 has no controlled-RY rotation; eigenvalues rounding there are
    silently dropped.  Distance is measured to bucket 1 in that case so
    such candidates are penalised.
    """
    total = 0.0
    for lam in lam_ratios:
        bucket = n_buckets * abs(lam) * t / (2.0 * math.pi)
        nearest = round(bucket)
        if nearest == 0:
            nearest = 1
        frac = bucket - nearest
        total += frac * frac
    return total


def _uniformly_controlled_ry(
    rotation_angles: dict[int, float],
    n_clock: int,
    control_wires: tuple[int, ...],
    target: int,
) -> list[Gate]:
    """The eigenvalue inversion as ``2^{n_clock}`` RY gates and ``2^{n_clock}`` CNOTs.

    The inversion needs a different ``RY(θ_j)`` on the ancilla per clock bucket.  Emitting
    that literally — one multi-controlled rotation per bucket — is not elementary; the
    uniformly controlled decomposition of Möttönen, Vartiainen, Bergholm and Salomaa
    (quant-ph/0407010) replaces it exactly and without ancillas: interleave ``RY(α_i)`` on
    the target with one CNOT whose control is the clock qubit whose Gray-code bit flips at
    step ``i``.  The angles follow from ``θ = M α`` with
    ``M_{ij} = (−1)^{popcount(i & gray(j))}``, and since ``M Mᵀ = 2^{n_clock} I`` the
    inverse is ``α = Mᵀθ / 2^{n_clock}``.  ``M`` is a Hadamard matrix whose columns are
    permuted into Gray order, so that product is a Walsh–Hadamard transform of ``θ`` read
    at ``gray(j)`` — ``Θ(2^{n_clock}·n_clock)``.  Forming ``M`` instead would cost
    ``Θ(4^{n_clock})`` in time and memory to produce a ``Θ(2^{n_clock})`` gate list, which
    is 576 MiB at ``n_clock = 13``, and ``mode="reference"`` shares this inversion.

    Two elementary gates per bucket, and no dense gate anywhere in the honest circuit.  The
    complexity class is unchanged and the angles are still enumerated bucket by bucket, so
    the circuit's *classical description* remains ``Θ(2^{n_clock})``: synthesisable, but not
    a uniform circuit family.
    """
    n_buckets = 1 << n_clock
    theta = np.zeros(n_buckets, dtype=float)
    for bucket, angle in rotation_angles.items():
        theta[bucket] = angle
    if not np.any(theta):
        return []

    buckets = np.arange(n_buckets)
    gray = buckets ^ (buckets >> 1)
    # θ is real, so the transform is too; `_walsh_hadamard` works in complex.
    alpha = _walsh_hadamard(theta).real[gray] / n_buckets

    gates: list[Gate] = []
    for step in range(n_buckets):
        gates.append(
            Gate(name="RY", qubits=(target,), params=(Constant(float(alpha[step])),))
        )
        # The control is the bit that flips on the way to the next Gray code; on the last
        # step that is the wrap-around to gray[0], which closes the product.
        flipped = int(gray[step] ^ gray[(step + 1) % n_buckets])
        bit = flipped.bit_length() - 1
        # Clock qubit 0 holds the most significant bucket bit (`format(j, "0nb")`).
        gates.append(Gate(name="CNOT", qubits=(control_wires[n_clock - 1 - bit], target)))
    return gates
