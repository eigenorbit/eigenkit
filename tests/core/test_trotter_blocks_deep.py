"""Unitary-level tests for the blocks the honest HHL circuit is assembled from.

Everything here is checked as a **matrix**, against ``scipy.linalg.expm`` or against an
independently built product formula — not against another block in the library.  That
matters because the honest path's correctness argument is a chain of exact identities:

* one Pauli exponential is ``V† RZ V`` with ``V`` the basis change and CNOT ladder;
* one Trotter step is the ordered product of those, and the *order* is a contract;
* the controlled evolution for clock qubit ``k`` is that same step unitary raised to
  ``steps·2^{n_clock−1−k}`` — a fixed ``τ``, never a re-Trotterised longer time;
* the inverse block is the exact **adjoint** of the forward one, which at order 1
  requires emitting the factors in reverse (negating ``τ`` alone is a different
  unitary).

Each of those links gets its own test, so a break localises instead of surfacing as a
slightly worse HHL residual.
"""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest
import scipy.linalg as sla

from eigenkit.algorithms._hamiltonian_sim import (
    pauli_decompose,
    pauli_string_matrix,
)
from eigenkit.algorithms.hhl import _estimate_trotter_gate_count
from eigenkit.core.blocks import (
    HamiltonianQPEBlock,
    QPEBlock,
    QubitUnitary,
    TrotterBlock,
    _append_controlled_trotter,
    _append_pauli_exp,
    _build_iqft,
    _build_qft,
    assemble,
)
from eigenkit.core.builder import CircuitBuilder
from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import CircuitSpec


P = pauli_string_matrix

# Elementary gates only: no dense unitary primitive may appear in the honest path.
_DENSE_GATES = {"QubitUnitary", "ControlledQubitUnitary", "AmplitudeEncoding"}


def _spec(gates, n: int) -> CircuitSpec:
    return CircuitSpec(n_qubits=n, n_weights=0, n_inputs=0, gates=tuple(gates))


def _unitary(spec: CircuitSpec) -> np.ndarray:
    """Full unitary of a spec via the numpy backend (qubit 0 = most significant)."""
    circ = Circuit(spec, provider="numpy")
    dim = 2**spec.n_qubits
    cols = []
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        cols.append(circ.statevector(np.zeros(0), initial_state=e))
    return np.column_stack(cols)


def _builder_unitary(build, n: int) -> np.ndarray:
    qb = CircuitBuilder(n_qubits=n)
    build(qb)
    return _unitary(_spec(qb.build().gates, n))


def _phase_aligned(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``U`` rotated so its global phase matches ``V``'s."""
    idx = np.unravel_index(np.argmax(np.abs(V)), V.shape)
    if abs(U[idx]) < 1e-14:
        return U
    return U * np.exp(1j * (np.angle(V[idx]) - np.angle(U[idx])))


def _step_unitary(terms, tau: float, order: int) -> np.ndarray:
    """One Trotter step exactly as :class:`TrotterBlock` orders the factors.

    The first term in the list is applied **first**, i.e. it is the rightmost matrix
    factor.  The identity term is included here (it is a global phase for the plain
    block and a real clock phase for the controlled one).
    """
    dim = 2 ** len(terms[0][1])
    seq = (
        [(c, s, tau) for c, s in terms]
        if order == 1
        else [(c, s, tau / 2) for c, s in terms]
        + [(c, s, tau / 2) for c, s in reversed(terms)]
    )
    W = np.eye(dim, dtype=complex)
    for c, s, dt in seq:
        W = sla.expm(1j * complex(c).real * P(s) * dt) @ W
    return W


def _real_terms(A: np.ndarray) -> tuple[tuple[float, str], ...]:
    """Decomposition with the coefficients cast to ``float``.

    ``pauli_decompose`` returns ``complex``; ``TrotterBlock`` and
    ``HamiltonianQPEBlock`` normalise that to ``float`` in ``__post_init__``, and the
    private ladder helpers below are documented to take the normalised form.
    """
    return tuple(
        (complex(c).real, s) for c, s in pauli_decompose(np.asarray(A, dtype=complex))
    )


def _random_hermitian(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return m + m.conj().T


# 2I + X: a single non-identity term, so every Trotter product is exact.
A_EXACT = np.array([[2.0, 1.0], [1.0, 2.0]])
# Heisenberg + field: five terms, two anticommuting families — Trotterisation bites.
A_HEISENBERG = (P("XX") + P("YY") + P("ZZ") + P("ZI") + P("IZ")).real


# ---------------------------------------------------------------------------
# one Pauli exponential
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("string", ["I", "X", "Y", "Z"])
@pytest.mark.parametrize("coeff,dt", [(1.0, 0.3), (-0.7, 1.1), (2.5, -0.4)])
def test_single_qubit_pauli_exponential_matches_expm(string, coeff, dt):
    got = _builder_unitary(lambda qb: _append_pauli_exp(qb, coeff, string, dt), 1)
    want = sla.expm(1j * coeff * P(string) * dt)
    # The all-identity string is a pure global phase and is deliberately not emitted.
    np.testing.assert_allclose(_phase_aligned(got, want), want, atol=1e-12)


@pytest.mark.parametrize(
    "string",
    ["II", "IX", "XI", "XX", "XY", "YX", "YY", "YZ", "ZY", "ZZ", "XZ", "ZX", "IY", "YI"],
)
def test_two_qubit_pauli_exponential_matches_expm(string):
    coeff, dt = 0.83, 0.37
    got = _builder_unitary(lambda qb: _append_pauli_exp(qb, coeff, string, dt), 2)
    want = sla.expm(1j * coeff * P(string) * dt)
    np.testing.assert_allclose(_phase_aligned(got, want), want, atol=1e-12)


@pytest.mark.parametrize("string", ["XYZ", "YZY", "ZIX", "YYY", "IZI"])
def test_three_qubit_pauli_exponential_matches_expm(string):
    coeff, dt = -1.2, 0.21
    got = _builder_unitary(lambda qb: _append_pauli_exp(qb, coeff, string, dt), 3)
    want = sla.expm(1j * coeff * P(string) * dt)
    np.testing.assert_allclose(_phase_aligned(got, want), want, atol=1e-12)


@pytest.mark.parametrize("string", ["I", "X", "Y", "Z", "XY", "ZZ", "YZ"])
def test_controlled_pauli_exponential_is_exact_including_the_phase(string):
    """No global-phase slack in the controlled form — HHL's clock register reads it.

    The basis change and the ladder are *not* controlled: they conjugate the single
    controlled RZ and cancel when the control is |0⟩.  The identity string becomes a
    ``PhaseShift`` on the control, which is exactly why the estimated phase reflects
    the whole eigenvalue and not just the traceless part of ``A``.
    """
    coeff, dt = 0.61, 0.44
    n_state = len(string)
    state = tuple(range(1, 1 + n_state))
    got = _builder_unitary(
        lambda qb: _append_pauli_exp(qb, coeff, string, dt, control=0, state_wires=state),
        1 + n_state,
    )
    target = sla.expm(1j * coeff * P(string) * dt)
    want = np.block(
        [
            [np.eye(2**n_state), np.zeros((2**n_state,) * 2)],
            [np.zeros((2**n_state,) * 2), target],
        ]
    )
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_controlled_identity_term_is_a_clock_phase_not_a_no_op():
    """Dropping it would shift every estimated eigenvalue by ``Tr(A)/N``."""
    got = _builder_unitary(
        lambda qb: _append_pauli_exp(qb, 2.0, "II", 0.5, control=0, state_wires=(1, 2)),
        3,
    )
    want = np.diag([1.0] * 4 + [np.exp(1j * 2.0 * 0.5)] * 4)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_state_wires_maps_pauli_positions_onto_circuit_qubits():
    """Position ``i`` of the string acts on ``state_wires[i]`` — not on qubit ``i``."""
    coeff, dt = 0.9, 0.3
    # "XZ" on wires (2, 0) of a 3-qubit circuit == the string "ZIX" on (0, 1, 2).
    got = _builder_unitary(
        lambda qb: _append_pauli_exp(qb, coeff, "XZ", dt, state_wires=(2, 0)), 3
    )
    want = sla.expm(1j * coeff * P("ZIX") * dt)
    np.testing.assert_allclose(_phase_aligned(got, want), want, atol=1e-12)


def test_pauli_exponential_emits_only_elementary_gates():
    qb = CircuitBuilder(n_qubits=4)
    _append_pauli_exp(qb, 0.5, "XYZ", 0.2, control=3, state_wires=(0, 1, 2))
    names = {g.name for g in qb.build().gates}
    assert names <= {"H", "RX", "RZ", "CNOT", "PhaseShift"}
    assert not (names & _DENSE_GATES)


# ---------------------------------------------------------------------------
# TrotterBlock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("steps", [1, 3])
def test_trotter_block_is_the_ordered_product_it_documents(order, steps):
    """Exact identity, not an approximation: the block *is* ``W(τ)^steps``.

    Everything downstream — the ``A_eff`` frame, the drift bound, the step selector —
    assumes precisely this, so it is checked for a non-commuting Hamiltonian at
    ``τ`` large enough that any ordering mistake shows up at ``O(1)``.
    """
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    time = 1.3
    frag = TrotterBlock(terms, time=time, steps=steps, order=order).build(2)
    got = _unitary(_spec(frag.gates, 2))
    want = np.linalg.matrix_power(_step_unitary(terms, time / steps, order), steps)
    np.testing.assert_allclose(_phase_aligned(got, want), want, atol=1e-11)


def test_trotter_block_applies_the_first_term_rightmost():
    """The list order is a numerical contract; this pins its direction."""
    terms = [(1.0, "X"), (1.0, "Z")]
    frag = TrotterBlock(terms, time=0.7, steps=1, order=1).build(1)
    got = _unitary(_spec(frag.gates, 1))
    first_rightmost = sla.expm(1j * P("Z") * 0.7) @ sla.expm(1j * P("X") * 0.7)
    first_leftmost = sla.expm(1j * P("X") * 0.7) @ sla.expm(1j * P("Z") * 0.7)
    np.testing.assert_allclose(
        _phase_aligned(got, first_rightmost), first_rightmost, atol=1e-12
    )
    assert not np.allclose(first_rightmost, first_leftmost, atol=1e-6)


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_trotter_block_is_exact_for_a_commuting_hamiltonian(steps):
    terms = pauli_decompose(A_EXACT.astype(complex))
    frag = TrotterBlock(terms, time=0.9, steps=steps, order=1).build(1)
    got = _unitary(_spec(frag.gates, 1))
    want = sla.expm(1j * A_EXACT * 0.9)
    np.testing.assert_allclose(_phase_aligned(got, want), want, atol=1e-12)


@pytest.mark.parametrize("order,expected_slope", [(1, -1.0), (2, -2.0)])
def test_trotter_error_falls_at_the_advertised_rate(order, expected_slope):
    """Order 2 is the shipped default *because* of this exponent."""
    A = A_HEISENBERG.astype(complex)
    terms = pauli_decompose(A)
    time = 1.0
    exact = sla.expm(1j * A * time)
    steps_list = [4, 8, 16, 32]
    errors = []
    for steps in steps_list:
        frag = TrotterBlock(terms, time=time, steps=steps, order=order).build(2)
        got = _phase_aligned(_unitary(_spec(frag.gates, 2)), exact)
        errors.append(float(np.linalg.norm(got - exact, 2)))
    slope = float(np.polyfit(np.log(steps_list), np.log(errors), 1)[0])
    assert slope == pytest.approx(expected_slope, abs=0.15), (slope, errors)


def test_order_two_beats_order_one_at_equal_step_count():
    A = A_HEISENBERG.astype(complex)
    terms = pauli_decompose(A)
    exact = sla.expm(1j * A * 1.0)
    errs = {}
    for order in (1, 2):
        frag = TrotterBlock(terms, time=1.0, steps=4, order=order).build(2)
        got = _phase_aligned(_unitary(_spec(frag.gates, 2)), exact)
        errs[order] = float(np.linalg.norm(got - exact, 2))
    assert errs[2] < errs[1]


def test_trotter_block_validates_its_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        TrotterBlock((), time=1.0)
    with pytest.raises(ValueError, match="Pauli strings must be non-empty"):
        TrotterBlock(((1.0, ""),), time=1.0)
    with pytest.raises(ValueError, match="same length"):
        TrotterBlock(((1.0, "XX"), (1.0, "X")), time=1.0)
    with pytest.raises(ValueError, match="invalid Pauli string"):
        TrotterBlock(((1.0, "XW"),), time=1.0)
    with pytest.raises(ValueError, match="not real"):
        TrotterBlock(((1.0 + 1.0j, "X"),), time=1.0)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        TrotterBlock(((1.0, "X"),), time=1.0, steps=0)
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        TrotterBlock(((1.0, "X"),), time=1.0, order=3)
    with pytest.raises(ValueError, match="needs 2 qubits"):
        TrotterBlock(((1.0, "XX"),), time=1.0).build(1)


def test_trotter_block_imaginary_tolerance_is_relative():
    """A Hermitian ``A`` of magnitude 1e10 leaves ~1e-6 of float noise in ``Im c``."""
    TrotterBlock(((1e10 + 1e-6j, "X"),), time=1.0)  # accepted
    with pytest.raises(ValueError, match="not real"):
        TrotterBlock(((1e10 + 1e3j, "X"),), time=1.0)


def test_trotter_block_serialisation_round_trip():
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    block = TrotterBlock(terms, time=-0.75, steps=3, order=2)
    clone = TrotterBlock.from_dict(block.to_dict())
    assert clone.to_dict() == block.to_dict()
    assert clone.build(2).gates == block.build(2).gates


# ---------------------------------------------------------------------------
# the controlled Trotter ladder, and why `reverse` exists
# ---------------------------------------------------------------------------


def test_reverse_flag_makes_the_order_one_ladder_an_exact_adjoint():
    """At order 1 the step is ``e^{X_L}···e^{X_1}``; its inverse reverses the factors.

    Negating ``τ`` alone yields ``e^{−X_L}···e^{−X_1}`` — a different unitary — so this
    is the identity the whole uncompute stage rests on.
    """
    terms = _real_terms(A_HEISENBERG)
    tau, n_steps = 0.4, 2
    state = (1, 2)

    def forward(qb):
        _append_controlled_trotter(qb, 0, terms, tau, n_steps, 1, state)

    def reversed_back(qb):
        _append_controlled_trotter(qb, 0, terms, -tau, n_steps, 1, state, reverse=True)

    def naive_back(qb):
        _append_controlled_trotter(qb, 0, terms, -tau, n_steps, 1, state)

    F = _builder_unitary(forward, 3)
    np.testing.assert_allclose(
        _builder_unitary(reversed_back, 3) @ F, np.eye(8), atol=1e-11
    )
    assert not np.allclose(_builder_unitary(naive_back, 3) @ F, np.eye(8), atol=1e-3)


def test_order_two_ladder_is_self_adjoint_at_negative_tau_so_reverse_is_a_no_op():
    """Reversing a palindrome would produce ``G_L…G_1G_1…G_L`` and break uncompute."""
    terms = _real_terms(A_HEISENBERG)
    state = (1, 2)

    def build(reverse):
        return lambda qb: _append_controlled_trotter(
            qb, 0, terms, -0.4, 2, 2, state, reverse=reverse
        )

    np.testing.assert_allclose(
        _builder_unitary(build(True), 3), _builder_unitary(build(False), 3), atol=1e-13
    )
    F = _builder_unitary(
        lambda qb: _append_controlled_trotter(qb, 0, terms, 0.4, 2, 2, state), 3
    )
    np.testing.assert_allclose(
        _builder_unitary(build(True), 3) @ F, np.eye(8), atol=1e-11
    )


def test_controlled_ladder_leaves_the_state_alone_when_the_control_is_zero():
    terms = _real_terms(A_HEISENBERG)
    U = _builder_unitary(
        lambda qb: _append_controlled_trotter(qb, 0, terms, 0.4, 3, 1, (1, 2)), 3
    )
    np.testing.assert_allclose(U[:4, :4], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(U[:4, 4:], np.zeros((4, 4)), atol=1e-12)


def test_controlled_ladder_applies_exactly_n_steps_of_the_fixed_step():
    """``n_steps`` steps of size ``τ`` — never one step of size ``n_steps·τ``."""
    terms = _real_terms(A_HEISENBERG)
    tau, n_steps = 0.35, 3
    U = _builder_unitary(
        lambda qb: _append_controlled_trotter(qb, 0, terms, tau, n_steps, 1, (1, 2)), 3
    )
    want = np.linalg.matrix_power(_step_unitary(terms, tau, 1), n_steps)
    np.testing.assert_allclose(U[4:, 4:], want, atol=1e-11)
    # The single-long-step alternative is a genuinely different matrix.
    assert not np.allclose(want, _step_unitary(terms, tau * n_steps, 1), atol=1e-3)


# ---------------------------------------------------------------------------
# QFT / IQFT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_is_the_dft_matrix_in_the_library_layout(n):
    got = _builder_unitary(lambda qb: _build_qft(qb, n), n)
    N = 2**n
    j = np.arange(N)
    dft = np.exp(2j * math.pi * np.outer(j, j) / N) / math.sqrt(N)
    np.testing.assert_allclose(got, dft, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_iqft_is_the_exact_adjoint_of_qft(n):
    Q = _builder_unitary(lambda qb: _build_qft(qb, n), n)
    Qi = _builder_unitary(lambda qb: _build_iqft(qb, n), n)
    np.testing.assert_allclose(Qi, Q.conj().T, atol=1e-12)
    np.testing.assert_allclose(Qi @ Q, np.eye(2**n), atol=1e-12)


def test_qft_uses_only_elementary_gates():
    qb = CircuitBuilder(n_qubits=4)
    _build_qft(qb, 4)
    _build_iqft(qb, 4)
    names = {g.name for g in qb.build().gates}
    assert names <= {"H", "CPHASE", "SWAP"}


# ---------------------------------------------------------------------------
# HamiltonianQPEBlock
# ---------------------------------------------------------------------------


def _hqpe_unitary(terms, n_clock, time, steps, order=1, inverse=False):
    n_state = len(terms[0][1])
    frag = HamiltonianQPEBlock(
        terms,
        n_clock=n_clock,
        time=time,
        steps=steps,
        order=order,
        inverse=inverse,
    ).build(n_clock + n_state)
    return _unitary(_spec(frag.gates, n_clock + n_state))


def _ideal_qpe_unitary(W: np.ndarray, n_clock: int) -> np.ndarray:
    """Textbook QPE on a dense oracle, which raises it to powers by ``matrix_power``."""
    n_state = W.shape[0].bit_length() - 1
    oracle = assemble([QubitUnitary(W, wires=tuple(range(n_state)))])
    frag = QPEBlock(oracle, n_clock=n_clock).build(n_clock + n_state)
    return _unitary(_spec(frag.gates, n_clock + n_state))


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("n_clock,steps", [(2, 1), (3, 2)])
def test_qpe_is_exact_phase_estimation_of_the_single_step_unitary(order, n_clock, steps):
    """The load-bearing identity of the honest path, tested on a **non-commuting** A.

    Because ``τ`` is held fixed, clock qubit ``k`` evolves for ``steps·2^{…}`` steps of
    the *same* ``W``, so the block must equal textbook QPE on the dense matrix
    ``W(τ)^steps`` — exactly, not approximately.  That is what turns the whole Trotter
    error into the single spectral substitution ``A → A_eff`` the drift bound measures.
    A per-power re-Trotterisation would fail this at ``O(τ²)``.
    """
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    time = 0.8
    W = np.linalg.matrix_power(_step_unitary(terms, time / steps, order), steps)
    got = _hqpe_unitary(terms, n_clock, time, steps, order)
    np.testing.assert_allclose(got, _ideal_qpe_unitary(W, n_clock), atol=1e-10)


@pytest.mark.parametrize("order", [1, 2])
def test_qpe_matches_the_dense_block_exactly_for_a_commuting_hamiltonian(order):
    terms = pauli_decompose(A_EXACT.astype(complex))
    got = _hqpe_unitary(terms, 3, 0.7, 2, order)
    want = _ideal_qpe_unitary(sla.expm(1j * A_EXACT * 0.7), 3)
    np.testing.assert_allclose(got, want, atol=1e-11)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("steps", [1, 3])
def test_inverse_block_is_the_exact_adjoint_of_the_forward_one(order, steps):
    """Not "approximately uncomputes": the two unitaries multiply to the identity.

    The order-1 inverse pass must be a true adjoint, and it is,
    at both orders and for a non-commuting Hamiltonian, which is what lets the
    clock=|0⟩ read-out mean what :func:`_postselect` says it means.
    """
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    n_clock, time = 2, 0.6
    F = _hqpe_unitary(terms, n_clock, time, steps, order, inverse=False)
    B = _hqpe_unitary(terms, n_clock, time, steps, order, inverse=True)
    dim = 2 ** (n_clock + 2)
    np.testing.assert_allclose(B @ F, np.eye(dim), atol=1e-10)
    np.testing.assert_allclose(F @ B, np.eye(dim), atol=1e-10)
    np.testing.assert_allclose(B, F.conj().T, atol=1e-10)


def test_qpe_recovers_an_on_grid_phase_exactly():
    """A diagonal ``A`` whose phases are exact multiples of ``2π/N``: one clean bucket."""
    n_clock = 3
    N = 2**n_clock
    # Phases 2π·2/N and 2π·5/N.
    lam = np.array([2.0, 5.0])
    time = 2.0 * math.pi / N
    terms = pauli_decompose(np.diag(lam).astype(complex))
    U = _hqpe_unitary(terms, n_clock, time, 1)
    for state_index, bucket in ((0, 2), (1, 5)):
        psi = np.zeros(2 ** (n_clock + 1), dtype=complex)
        psi[state_index] = 1.0  # clock |0…0⟩, state |state_index⟩
        out = U @ psi
        probs = np.abs(out.reshape(N, 2)[:, state_index]) ** 2
        assert np.argmax(probs) == bucket
        assert probs[bucket] == pytest.approx(1.0, abs=1e-10)


def test_identity_term_lands_on_the_clock_as_a_phase():
    """Without it the estimated eigenvalue is the traceless part of ``A``."""
    n_clock = 3
    N = 2**n_clock
    A = np.diag([3.0, 3.0])  # = 3·I, purely a trace
    time = 2.0 * math.pi / N  # phase 3 buckets
    U = _hqpe_unitary(pauli_decompose(A.astype(complex)), n_clock, time, 1)
    psi = np.zeros(2 ** (n_clock + 1), dtype=complex)
    psi[0] = 1.0
    probs = np.abs((U @ psi).reshape(N, 2)[:, 0]) ** 2
    assert np.argmax(probs) == 3
    assert probs[3] == pytest.approx(1.0, abs=1e-10)


def test_clock_qubit_zero_carries_the_highest_power():
    """Big-endian QPE: clock qubit ``k`` controls ``e^{iA·2^{n_clock−1−k}t}``.

    A flipped convention still produces a valid-looking distribution, just with the
    bucket bits reversed, so it has to be checked against an absolute reference.
    """
    n_clock = 3
    terms = ((1.0, "Z"),)
    counts = collections.Counter()
    frag = HamiltonianQPEBlock(terms, n_clock=n_clock, time=0.5, steps=1, order=1).build(
        n_clock + 1
    )
    for g in frag.gates:
        # The controlled RZ pairs are the only gates touching a clock qubit and the
        # state qubit together.
        if g.name == "CNOT" and g.qubits[0] < n_clock <= g.qubits[1]:
            counts[g.qubits[0]] += 1
    # Two CNOTs per controlled RZ, one controlled RZ per Trotter step: clock qubit 0
    # runs 2^{n_clock-1} steps, qubit 1 runs 2^{n_clock-2}, and so on.
    assert counts[0] == 2 * 2 ** (n_clock - 1)
    assert counts[1] == 2 * 2 ** (n_clock - 2)
    assert counts[n_clock - 1] == 2


def test_qpe_block_emits_no_dense_gate_and_no_matrix_power():
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    for inverse in (False, True):
        frag = HamiltonianQPEBlock(
            terms, n_clock=3, time=0.5, steps=2, order=2, inverse=inverse
        ).build(5)
        names = {g.name for g in frag.gates}
        assert not (names & _DENSE_GATES), names
        assert names <= {"H", "RX", "RZ", "CNOT", "CPHASE", "SWAP", "PhaseShift"}
        assert all(not g.data for g in frag.gates)


def test_qpe_block_validates_its_inputs():
    with pytest.raises(ValueError, match="n_clock must be >= 1"):
        HamiltonianQPEBlock(((1.0, "X"),), n_clock=0, time=1.0)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        HamiltonianQPEBlock(((1.0, "X"),), n_clock=2, time=1.0, steps=0)
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        HamiltonianQPEBlock(((1.0, "X"),), n_clock=2, time=1.0, order=5)
    with pytest.raises(ValueError, match="non-empty"):
        HamiltonianQPEBlock((), n_clock=2, time=1.0)
    with pytest.raises(ValueError, match="needs 4 qubits"):
        HamiltonianQPEBlock(((1.0, "XX"),), n_clock=2, time=1.0).build(3)


def test_qpe_block_required_qubits():
    block = HamiltonianQPEBlock(((1.0, "XX"),), n_clock=3, time=1.0)
    assert block.required_qubits() == 5


def test_qpe_block_serialisation_round_trip():
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    for inverse in (False, True):
        block = HamiltonianQPEBlock(
            terms, n_clock=3, time=-0.4, steps=2, order=2, inverse=inverse
        )
        clone = HamiltonianQPEBlock.from_dict(block.to_dict())
        assert clone.to_dict() == block.to_dict()
        assert clone.build(5).gates == block.build(5).gates


# ---------------------------------------------------------------------------
# the gate-count estimator that gates auto-selection
# ---------------------------------------------------------------------------


def _emitted_trotter_gates(terms, n_clock: int, steps: int, order: int) -> int:
    """Gates emitted by both QPE stages with the (I)QFT scaffolding removed."""
    n_state = len(terms[0][1])
    total = sum(
        len(
            HamiltonianQPEBlock(
                terms,
                n_clock=n_clock,
                time=1.0,
                steps=steps,
                order=order,
                inverse=inverse,
            )
            .build(n_clock + n_state)
            .gates
        )
        for inverse in (False, True)
    )
    # Per stage: n_clock outer H, plus the (I)QFT's n_clock H + n(n−1)/2 CPHASE +
    # n//2 SWAP.
    scaffold = 2 * (2 * n_clock + n_clock // 2 + n_clock * (n_clock - 1) // 2)
    return total - scaffold


@pytest.mark.parametrize(
    "terms,n_clock,steps,order",
    [
        # No Y: only I, X and Z factors.
        (((2.0, "I"), (1.0, "X")), 4, 1, 1),
        (((2.0, "I"), (1.0, "X")), 3, 4, 2),
        (((1.0, "ZZ"), (0.5, "XI")), 3, 2, 1),
        (((1.0, "XZX"),), 2, 1, 2),
        # With Y: each factor costs 2 basis-change gates, the same as X.
        (((1.0, "Y"),), 3, 1, 1),
        (((1.0, "YY"),), 3, 1, 1),
        (((1.0, "XY"),), 3, 2, 2),
        (((1.0, "YZY"),), 2, 1, 1),
        (_real_terms(A_HEISENBERG), 3, 1, 2),
    ],
)
def test_estimator_is_exact(terms, n_clock, steps, order):
    """The budget clamp is only meaningful if the estimate tracks reality.

    :func:`_append_pauli_exp` spends a single ``RX(±pi/2)`` each way on a Y factor, the
    same 2 gates an X factor costs, not the 4 of the textbook ``H·S†`` / ``S·H`` pair.
    Pricing Y at 4 would overcount by up to +43 % on a Y-heavy Hamiltonian and make
    ``trotter_steps="auto"`` clamp to a grid up to 44 % coarser than the gate budget
    actually allows.
    """
    assert _estimate_trotter_gate_count(terms, n_clock, steps, order) == (
        _emitted_trotter_gates(terms, n_clock, steps, order)
    )


@pytest.mark.parametrize(
    "terms,n_clock,steps,order",
    [
        (((1.0, "Y"),), 3, 1, 1),
        (((1.0, "YY"),), 3, 1, 1),
        (((1.0, "XY"),), 3, 2, 2),
        (_real_terms(A_HEISENBERG), 3, 1, 2),
    ],
)
def test_estimator_never_undercounts(terms, n_clock, steps, order):
    """Weaker than exactness, and kept deliberately.

    Exactness is what the test above demands, but *this* is the property the budget
    relies on: an overestimate only clamps early, while an underestimate would let
    ``"auto"`` build a circuit past the declared budget.  Keeping it stated separately
    means a future change to the emitted gate mix fails loudly on the right axis.
    """
    assert _estimate_trotter_gate_count(terms, n_clock, steps, order) >= (
        _emitted_trotter_gates(terms, n_clock, steps, order)
    )


@pytest.mark.parametrize("n_clock", [4, 6])
def test_the_budget_clamp_sits_against_the_real_gate_count(n_clock, monkeypatch):
    """Checked from both sides: the chosen count fits, one more step does not.

    An overpriced estimate clamps early: pricing Y at 4 gates would clamp the Heisenberg
    witness at ``n_l=9`` to 25 steps and 1,737,400 real gates, leaving 13 % of the
    2,000,000-gate budget unspent for no reason.  The budget is lowered here so the
    assertion can be made against gates that are actually emitted rather than 2 M-gate
    fragments; the shipped constant is checked separately below, through an estimate the
    test above proves exact.
    """
    from eigenkit.algorithms import hhl

    budget = 50_000
    monkeypatch.setattr(hhl, "_AUTO_TROTTER_GATE_BUDGET", budget)
    terms = _real_terms(A_HEISENBERG)
    steps = hhl._max_steps_within_budget(terms, n_clock, 2)
    assert steps > 1  # otherwise the clamp is at its floor and proves nothing
    assert _emitted_trotter_gates(terms, n_clock, steps, 2) <= budget
    assert _emitted_trotter_gates(terms, n_clock, steps + 1, 2) > budget


def test_the_shipped_budget_is_spent_not_padded():
    """The same two-sided check at the real 2,000,000-gate budget.

    Uses the estimate rather than building the fragments — twelve million gate objects
    would dominate the suite's runtime — which is sound only because
    :func:`test_estimator_is_exact` pins the estimate to the emitted count.
    """
    from eigenkit.algorithms.hhl import (
        _AUTO_TROTTER_GATE_BUDGET,
        _max_steps_within_budget,
    )

    terms = _real_terms(A_HEISENBERG)
    for n_clock in (8, 9, 10):
        steps = _max_steps_within_budget(terms, n_clock, 2)
        assert steps > 1
        assert (
            _estimate_trotter_gate_count(terms, n_clock, steps, 2)
            <= _AUTO_TROTTER_GATE_BUDGET
        )
        assert (
            _estimate_trotter_gate_count(terms, n_clock, steps + 1, 2)
            > _AUTO_TROTTER_GATE_BUDGET
        )
        # And at least 85 % of the budget is actually used, not left on the table.
        spent = _estimate_trotter_gate_count(terms, n_clock, steps, 2)
        assert spent > 0.85 * _AUTO_TROTTER_GATE_BUDGET, (n_clock, steps, spent)


def test_estimator_scales_linearly_in_steps_and_as_two_to_the_clock():
    terms = ((2.0, "I"), (1.0, "X"))
    base = _estimate_trotter_gate_count(terms, 4, 1, 1)
    assert _estimate_trotter_gate_count(terms, 4, 5, 1) == 5 * base
    assert _estimate_trotter_gate_count(terms, 4, 1, 2) == 2 * base
    # (2^5 − 1)/(2^4 − 1) exactly, since the ladder length is the only n_clock term.
    assert _estimate_trotter_gate_count(terms, 5, 1, 1) * (2**4 - 1) == base * (2**5 - 1)
