"""Edge cases, cost contracts and the gaps that are still open on the honest path.

Complements ``test_hhl_honest_properties.py`` — one test per *known* failure mode — with
the cases nothing else covers:

* what ``solve()`` does to inputs it does not own (mutation, dtype, memory layout);
* the boundaries: a 1x1 system, a padded one, the smallest register, a cutoff that culls
  the whole spectrum, and both ends of float64;
* the cost contracts a reader would otherwise have to measure — how the circuit, the
  classical assembly and the term list grow;
* the places where the shipped behaviour still differs from ideal HHL.

Tests asserting a property the code does **not** have yet carry
``pytest.mark.xfail(strict=True)`` with the reason spelled out, so the suite stays honest
about what is open and turns green by itself the day the gap closes.
"""

from __future__ import annotations

import ast
import collections
import inspect
import json
import pathlib
import re
import tracemalloc
import warnings

import numpy as np
import pytest
import scipy.linalg as sla

from eigenkit import HHL
from eigenkit.algorithms import hhl as H
from eigenkit.algorithms._hamiltonian_sim import (
    pauli_decompose,
    pauli_string_matrix,
    trotter_drift_bound,
)


P = pauli_string_matrix

TRI4 = np.array([[2.0, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]])
BAND4 = np.array([[4.0, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]])
# Three large commuting terms and one small anticommuting one: the drift bound sees
# only the small term while the per-step angle is set by the large ones.
A_ANGLE_GUARD = (
    6.0 * P("ZI").real + 3.0 * P("IZ").real + 1.0 * P("ZZ").real + 0.1 * P("XI").real
)


def _honest(n_l: int = 6, **kwargs) -> HHL:
    return HHL(n_eigenvalue_qubits=n_l, mode="honest", provider="numpy", **kwargs)


def _cos(u, v) -> float:
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    return float(abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)))


def _solve_capturing(A, b, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _honest(**kwargs).solve(A, b)
    return result, [str(w.message) for w in caught]


def _spec_for(A, b_unit, n_l: int, **solver_kwargs):
    """The assembled circuit for ``A`` at ``n_l`` clock qubits, without running it."""
    solver = _honest(n_l=n_l, **solver_kwargs)
    Ac = np.asarray(A, dtype=complex)
    A_circuit, t, C, lam, _ = solver._honest_setup(Ac)
    terms = pauli_decompose(A_circuit)
    steps = solver._resolve_trotter_steps(terms, t, C, n_l)
    angles = H._compute_rotation_angles(2**n_l, t, C, lam, solver.epsilon)
    n_b = int(np.log2(Ac.shape[0]))
    spec = solver._build_circuit_spec(
        A_circuit,
        np.asarray(b_unit, dtype=complex),
        n_b,
        n_l,
        t,
        angles,
        pauli_terms=terms,
        trotter_steps=steps,
    )
    return spec, steps


def _effective_generator(pauli_terms, tau: float) -> np.ndarray:
    """``Log(W)/(i*tau)`` for one second-order Trotter step — the generator QPE sees."""
    dim = 2 ** len(pauli_terms[0][1])
    step = np.eye(dim, dtype=complex)
    for coefficient, string in list(pauli_terms) + list(reversed(pauli_terms)):
        step = sla.expm(0.5j * complex(coefficient).real * tau * P(string)) @ step
    return sla.logm(step) / (1j * tau)


# ---------------------------------------------------------------------------
# the input contract: solve() borrows A and b, it does not own them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["honest", "reference"])
def test_solve_leaves_its_arguments_untouched(mode):
    """A solver that edits the caller's arrays corrupts whatever else holds them.

    ``solve()`` casts, pads, symmetrises and (for a non-Hermitian input) squares ``A``;
    every one of those has an in-place spelling that would be invisible from the result.
    """
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    Ac = np.array([[2.0, 1.0 - 1j], [1.0 + 1j, 3.0]])
    bc = np.array([1.0, 1j])
    A_before, b_before = A.copy(), b.copy()
    Ac_before, bc_before = Ac.copy(), bc.copy()

    HHL(n_eigenvalue_qubits=4, mode=mode, provider="numpy").solve(A, b)
    HHL(n_eigenvalue_qubits=4, mode=mode, provider="numpy").solve(Ac, bc)

    assert np.array_equal(A, A_before)
    assert np.array_equal(b, b_before)
    assert np.array_equal(Ac, Ac_before)
    assert np.array_equal(bc, bc_before)


def test_plain_lists_and_integer_arrays_are_accepted():
    """``np.asarray`` is the only entry check, so the documented "array-like" has to hold
    for the two spellings a user reaches for first."""
    from_lists = _honest(n_l=5).solve([[2, 1], [1, 2]], [1, 0])
    from_ints = _honest(n_l=5).solve(
        np.array([[2, 1], [1, 2]], dtype=np.int64), np.array([1, 0], dtype=np.int64)
    )
    from_floats = _honest(n_l=5).solve(
        np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0])
    )
    assert from_lists.solution.dtype == np.float64
    np.testing.assert_allclose(from_lists.solution, from_floats.solution, atol=1e-12)
    np.testing.assert_allclose(from_ints.solution, from_floats.solution, atol=1e-12)


def test_memory_layout_does_not_change_the_answer():
    """Fortran order, a strided view and a read-only buffer all describe the same matrix.

    Anything that reads ``A.data`` or writes through a view would disagree here.
    """
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    reference = _honest(n_l=5).solve(A, b).solution

    fortran = _honest(n_l=5).solve(np.asfortranarray(A), b).solution
    strided = _honest(n_l=5).solve(np.repeat(A, 2, axis=0)[::2], b).solution
    frozen = A.copy()
    frozen.flags.writeable = False
    read_only = _honest(n_l=5).solve(frozen, b).solution

    for other in (fortran, strided, read_only):
        np.testing.assert_array_equal(other, reference)


def test_one_solver_instance_serves_several_systems():
    """No per-solve state may survive on the solver: the auto step count, the Pauli terms
    and the evolution time are all derived from ``A``, and caching any of them on ``self``
    would make the second answer depend on the first."""
    solver = _honest(n_l=5)
    first = solver.solve(TRI4, np.array([1.0, 0.0, 0.0, 0.0]))
    other = solver.solve(BAND4, np.ones(4))
    again = solver.solve(TRI4, np.array([1.0, 0.0, 0.0, 0.0]))

    np.testing.assert_array_equal(first.solution, again.solution)
    assert first.success_probability == again.success_probability
    assert not np.allclose(other.solution, first.solution)


# ---------------------------------------------------------------------------
# boundaries of the system itself
# ---------------------------------------------------------------------------


def test_a_one_by_one_system_is_padded_and_solved_exactly():
    """``N = 1`` has zero qubits; the padding to 2x2 is what makes it representable.

    Worth pinning because the padded block is *singular*, and the honest path derives its
    evolution time from a Gershgorin bound over the padded matrix.
    """
    result = _honest(n_l=5).solve(np.array([[2.0]]), np.array([1.0]))
    np.testing.assert_allclose(result.solution, [0.5], atol=1e-9)
    assert result.residual < 1e-9


@pytest.mark.parametrize(
    "A, b",
    [
        (np.array([[4.0, 1, 0], [1, 4, 1], [0, 1, 4]]), np.array([1.0, 0.0, 0.0])),
        (
            np.diag([4.0, 3, 2, 5, 6])
            + np.diag([1.0, 1, 1, 1], 1)
            + np.diag([1.0, 1, 1, 1], -1),
            np.ones(5),
        ),
    ],
)
def test_padding_leaks_only_a_negligible_weight_into_the_zero_block(A, b):
    """A non-power-of-two system is padded with zeros, and the padded directions have
    eigenvalue 0 — bucket 0, no rotation — so they should carry no read-out weight.

    ``HHLResult.raw_solution`` is documented as unit-norm "except for a non-power-of-two
    system, where the Trotterised evolution leaks a little weight into the zero padding".
    This says how little: the returned slice still holds essentially all of it.
    """
    result = _honest(n_l=6).solve(A, b)
    leaked = 1.0 - float(np.linalg.norm(result.raw_solution)) ** 2
    assert 0.0 <= leaked < 1e-4
    assert _cos(result.solution, np.linalg.solve(A, b)) > 0.999


def test_the_identity_is_a_single_commuting_term_and_is_solved_exactly():
    """One Pauli term means the Trotter product is the exact exponential, so nothing but
    the QPE grid can be wrong — and ``lambda = 1`` sits on it."""
    terms = pauli_decompose(np.eye(4, dtype=complex))
    assert [s for _, s in terms] == ["II"]

    result, warned = _solve_capturing(np.eye(4), np.array([1.0, 0.0, 0.0, 0.0]), n_l=5)
    assert warned == []
    assert result.residual < 1e-9
    np.testing.assert_allclose(result.solution, [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_b_along_an_eigenvector_is_returned_exactly():
    """The single-eigenvalue case: no superposition over buckets, so the answer is exact
    whenever that one eigenvalue is on the grid."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    result = _honest(n_l=5).solve(A, np.array([1.0, 1.0]))
    np.testing.assert_allclose(result.solution, [1 / 3, 1 / 3], atol=1e-9)


# ---------------------------------------------------------------------------
# boundaries of the register and of the cutoff
# ---------------------------------------------------------------------------


def test_the_smallest_register_says_so_instead_of_answering_quietly():
    """``n_l = 2`` gives four buckets, so ``C`` is comparable to ``norm(A)`` and every
    eigenvalue below ``C/2`` is dropped outright.  The answer is then meaningless, and the
    thing that must not happen is for it to arrive without a word."""
    result, warned = _solve_capturing(
        np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0]), n_l=2
    )
    assert result.residual > 0.25
    assert any("rotation-free bucket 0" in w for w in warned)
    assert any("relative residual" in w for w in warned)


def test_epsilon_zero_keeps_every_bucket():
    """``epsilon=0`` is in range and has to mean "no cutoff", not "cut everything"."""
    angles = H._compute_rotation_angles(32, 1.0, 0.2, 3.0, 0.0)
    assert all(theta != 0.0 for theta in angles.values())

    result, warned = _solve_capturing(
        np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0]), n_l=5, epsilon=0.0
    )
    assert result.residual < 1e-9
    assert not any("epsilon cutoff" in w for w in warned)


def test_a_cutoff_that_culls_the_whole_spectrum_still_returns_a_finite_result():
    """``epsilon * norm(A)_bound`` is measured against the *Gershgorin* bound, so a loose
    bound lets the cutoff climb past the true ``lambda_max`` and zero every bucket a real
    eigenvalue lands in.  The result is worthless — but it must be finite, warned about,
    and not an exception out of an empty rotation table.
    """
    A = np.array([[1.0, 1.0], [1.0, -1.0]])  # row sums 2, spectrum +-sqrt(2)
    result, warned = _solve_capturing(A, np.array([1.0, 0.0]), n_l=5, epsilon=0.99)
    assert np.all(np.isfinite(result.solution))
    assert any("epsilon cutoff" in w for w in warned)


@pytest.mark.parametrize("n_l", [4, 5, 6, 7])
def test_a_degenerate_gershgorin_tight_bottom_is_safe_at_every_register_size(n_l):
    """The narrow class that is genuinely dangerous: a *degenerate*
    extremum saturating the Gershgorin bound.  The first-order protection theorem does not
    hold there — the pair splits through the off-diagonal element and drifts outward at
    ``Theta(tau)`` — so it is the case where Nyquist capture is actually reachable.

    Block-diagonal ``A`` of that shape (one isolated coordinate at ``-Lambda`` plus a 3x3
    non-negative block with constant row sums) is what "several independent systems solved
    jointly" produces.
    """
    A = np.zeros((4, 4))
    A[0, 0] = -4.0
    A[1:, 1:] = -np.array([[0.0, 2, 2], [2, 0, 2], [2, 2, 0]])
    assert np.allclose(np.sort(np.linalg.eigvalsh(A)), [-4.0, -4.0, 2.0, 2.0])
    assert H._gershgorin_spectral_radius(A) == 4.0  # tight on the degenerate bottom

    result, warned = _solve_capturing(A, np.ones(4), n_l=n_l)
    assert _cos(result.solution, np.linalg.solve(A, np.ones(4))) > 0.999
    assert warned == []


# ---------------------------------------------------------------------------
# both ends of float64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-150, 1e-8, 1e8, 1e150])
def test_a_common_rescale_leaves_the_answer_where_it_was(scale):
    """``(alpha A) x = alpha b`` has the same ``x``.  Every threshold on the honest path —
    the Pauli pruning, the Hermiticity test, the drift criterion, the real-input test — is
    relative for this reason, and a single absolute one would show up here."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    base = _honest(n_l=5).solve(A, b)
    scaled = _honest(n_l=5).solve(A * scale, b * scale)
    np.testing.assert_allclose(scaled.solution, base.solution, rtol=1e-12)
    assert scaled.success_probability == pytest.approx(
        base.success_probability, rel=1e-12
    )


def test_a_and_b_scaled_apart_still_give_the_right_magnitude():
    """``norm(A) ~ 1e150`` against ``norm(b) ~ 1e-150`` puts ``norm(A x)**2`` past float64
    on the way to the least-squares rescale; the fit runs in a normalised frame precisely
    so that the magnitude survives."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]]) * 1e150
    b = np.array([1.0, 0.0]) * 1e-150
    result = _honest(n_l=5).solve(A, b)
    expected = np.linalg.solve(np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(result.solution / 1e-300, expected, rtol=1e-9)
    assert np.isfinite(result.residual)


# ---------------------------------------------------------------------------
# cost contracts — what grows, and how fast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_l", [4, 5, 6])
def test_the_circuit_doubles_with_every_clock_qubit(n_l):
    """The controlled-evolution ladder is ``sum_k steps*2^k = steps*(2^n_l - 1)`` Trotter
    steps, so at a fixed step count the whole circuit doubles per clock qubit.  That is
    the ``Theta(2^n_l) = Theta(kappa_G / eps_phase)`` every QPE-based HHL pays, and
    pinning it keeps a future "optimisation" from quietly changing the exponent."""
    b = np.array([1.0, 0.0, 0.0, 0.0])
    with warnings.catch_warnings():
        # A fixed step count is the point here; whether it is fine enough is not.
        warnings.simplefilter("ignore", RuntimeWarning)
        small, _ = _spec_for(TRI4, b, n_l, trotter_steps=2)
        large, _ = _spec_for(TRI4, b, n_l + 1, trotter_steps=2)
    ratio = len(large.gates) / len(small.gates)
    assert 1.9 < ratio < 2.1


def test_resolution_and_success_probability_move_as_the_theory_says():
    """``C = 2*pi/(t*N)`` halves per clock qubit, so the residual falls with it and the
    ancilla amplitude ``C/lambda`` — hence the probability — falls by four.

    The second half is the part users get wrong: buying accuracy costs shots
    quadratically, and on a device that, not the qubit count, is what bounds
    ``n_eigenvalue_qubits``.
    """
    b = np.array([1.0, 0.0, 0.0, 0.0])
    residuals, probabilities = [], []
    for n_l in (4, 5, 6, 7):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _honest(n_l=n_l).solve(BAND4, b)
        residuals.append(result.residual)
        probabilities.append(result.success_probability)

    for coarse, fine in zip(residuals, residuals[1:]):
        assert 1.5 < coarse / fine < 3.0
    for coarse, fine in zip(probabilities, probabilities[1:]):
        assert 3.0 < coarse / fine < 5.0


@pytest.mark.parametrize("dim", [8, 16, 32, 64])
def test_a_dense_matrix_costs_a_quadratic_number_of_terms(dim):
    """``L = N(N+1)/2`` for a dense real symmetric ``A`` — exactly the strings with an even
    number of ``Y`` factors survive.

    ``L`` is the honest cost metric, and it is what makes the *classical* preparation bind
    before the simulation does: the drift bound behind ``trotter_steps="auto"`` is
    ``Theta(L**2) = Theta(N**4)``, so it — not ``pauli_decompose``, which is
    ``Theta(N**2 log N)`` — is the dominant classical step today.
    """
    rng = np.random.default_rng(dim)
    M = rng.standard_normal((dim, dim))
    terms = pauli_decompose((M + M.T) / 2)
    assert len(terms) == dim * (dim + 1) // 2


def test_the_inversion_assembly_is_linear_in_the_gates_it_emits():
    """``α = Mᵀθ / 2^n_l`` is a Walsh–Hadamard transform read in Gray order, not a dense
    matrix product.  Forming ``M`` would cost 144 MiB at ``n_l = 12`` and quadruple per
    clock qubit, to produce a gate list of only ``2 · 2^n_l``; ``mode="reference"`` shares
    this inversion, so the cost would not be confined to the honest path.

    The budget below leaves room for the angle vector and the gate list and nothing else.
    """
    n_l = 12
    angles = H._compute_rotation_angles(2**n_l, 1.0, 0.1, 1.0, 0.0)
    tracemalloc.start()
    try:
        H._uniformly_controlled_ry(angles, n_l, tuple(range(1, 1 + n_l)), 0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 2**20


# ---------------------------------------------------------------------------
# what still differs from ideal HHL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "A, b",
    [
        (np.array([[2.0, 1.0], [1.0, 0.0]]), np.array([1.0, 1.0])),
        (np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0])),
        (TRI4, np.array([1.0, 0.0, 0.0, 0.0])),
    ],
)
def test_the_reported_readout_probability_reconstructs_the_norm(A, b):
    """HHL's own statement of ``norm(x)`` is ``sqrt(P) * norm(b) / C`` — and ``P`` is the
    *joint* ``(ancilla=1 and clock=0)`` event, which is what ``readout_probability``
    reports.

    So the quantum output plus one reported number recovers the magnitude without the
    classical least-squares fit; ``success_probability`` is the larger, wrong number for
    this and overstates ``norm(x)``.  Both halves are asserted, because the two fields
    differ by only 5-25 % and are easy to mix up.
    """
    n_l = 6
    result = _honest(n_l=n_l).solve(A, b)
    _, C, _ = H._honest_qpe_params(np.asarray(A, dtype=complex), n_l)

    exact_norm = float(np.linalg.norm(np.linalg.solve(A, b)))
    from_readout = np.sqrt(result.readout_probability) * np.linalg.norm(b) / C
    from_success = np.sqrt(result.success_probability) * np.linalg.norm(b) / C

    assert from_readout == pytest.approx(exact_norm, rel=0.1)
    assert from_success > from_readout
    assert result.readout_probability <= result.success_probability


def test_a_small_residual_does_not_certify_the_direction():
    """The residual is a *backward* error: ``norm(Ax-b)/norm(b) = 0.03`` on a kappa ~ 48
    system still leaves the direction 35 degrees off, and nothing warns, because the
    warning threshold is a residual threshold.

    Pinned as a witness rather than filed as a defect: honest mode never learns kappa, so
    the forward error is not available to warn about.  The test exists so the gap stays
    visible, and so nobody "fixes" it by lowering the residual threshold — which would
    fire on answers correct to four decimals.
    """
    A = np.array([[0.29, 0.47], [0.47, 0.69]])  # spectrum ~ {-0.021, 1.001}
    b = np.array([0.59, 0.81])
    result, warned = _solve_capturing(A, b, n_l=6)

    assert np.linalg.cond(A) > 40
    assert result.residual < 0.05
    assert _cos(result.solution, np.linalg.solve(A, b)) < 0.9
    assert warned == []


@pytest.mark.xfail(
    strict=True,
    reason="With an explicit trotter_steps the drift bound is applied outside the "
    "per-step angle guard tau*sum|c_k| <= 0.5 that trotter_steps='auto' enforces. Past "
    "it the leading-order bound under-reports the true drift (1.5x on this witness), so "
    "the C/2 capture warning stays silent while the real drift is already over C/2.",
)
def test_an_explicit_step_count_outside_the_angle_guard_is_warned_about():
    """Witness: ``6*ZI + 3*IZ + ZZ + 0.1*XI`` — three large commuting terms and one small
    anticommuting one, ``kappa = 5``.

    ``Gamma_2`` sees only the small term, so the bound is 0.149 against ``C/2 = 0.163``
    and nothing is said, while the true ``norm(A_eff - A)`` is 0.226 — past the threshold
    at which an extremal eigenvalue can be captured by the Nyquist bucket.  The per-step
    angle is 3.04, six times the guard the selector holds itself to.

    ``trotter_steps="auto"`` is unaffected; the companion test below shows why.
    """
    n_l = 6
    terms = pauli_decompose(np.asarray(A_ANGLE_GUARD, dtype=complex))
    t, C, _ = H._honest_qpe_params(np.asarray(A_ANGLE_GUARD, dtype=complex), n_l)

    true_drift = float(np.linalg.norm(_effective_generator(terms, t) - A_ANGLE_GUARD, 2))
    bound = trotter_drift_bound(terms, t, 2)
    angle = t * sum(abs(complex(c)) for c, _ in terms)

    assert angle > 6 * H._TROTTER_ANGLE_GUARD  # far outside the bound's validity
    assert bound < 0.5 * C < true_drift  # silent, and yet past the threshold

    _, warned = _solve_capturing(A_ANGLE_GUARD, np.ones(4), n_l=n_l, trotter_steps=1)
    assert any("drift" in w for w in warned)


def test_the_auto_selector_covers_the_case_the_explicit_path_misses():
    """Companion to the xfail above: on the same witness ``"auto"`` is held by its
    angle-guard floor, and the step count it picks brings the *true* drift — not just the
    bound — well under ``C/2``."""
    n_l = 6
    solver = _honest(n_l=n_l)
    terms = pauli_decompose(np.asarray(A_ANGLE_GUARD, dtype=complex))
    t, C, _ = H._honest_qpe_params(np.asarray(A_ANGLE_GUARD, dtype=complex), n_l)
    steps = solver._resolve_trotter_steps(terms, t, C, n_l)

    coefficient_l1 = sum(abs(complex(c)) for c, _ in terms)
    assert steps >= np.ceil(t * coefficient_l1 / H._TROTTER_ANGLE_GUARD)

    tau = t / steps
    true_drift = float(
        np.linalg.norm(_effective_generator(terms, tau) - A_ANGLE_GUARD, 2)
    )
    assert true_drift < 0.5 * C

    result, warned = _solve_capturing(A_ANGLE_GUARD, np.ones(4), n_l=n_l)
    assert warned == []
    assert _cos(result.solution, np.linalg.solve(A_ANGLE_GUARD, np.ones(4))) > 0.999


# ---------------------------------------------------------------------------
# the normal-equations lift — the only path that turns the one-sided window on
# ---------------------------------------------------------------------------


def test_the_normal_equations_lift_solves_a_non_hermitian_system():
    """A non-Hermitian ``A`` is solved as ``A†A x = A†b``, which shares the solution.

    Quiet at ``kappa(A) = 3.3``, because ``kappa**2 = 11`` still fits a 64-bucket
    register — the lift is only usable while the *squared* condition number does.
    """
    A = np.array([[3.0, 1.0], [0.0, 2.0]])
    b = np.array([1.0, 1.0])
    result, warned = _solve_capturing(A, b, n_l=6)
    assert np.linalg.cond(A) ** 2 < 20
    assert _cos(result.solution, np.linalg.solve(A, b)) > 0.99
    assert warned == []


def test_the_squared_condition_number_is_what_has_to_fit_the_register():
    """The other side of the same lift: ``kappa(A) = 49`` is inside the register's reach,
    ``kappa(A)**2 = 2404`` is not, and the answer is destroyed.

    Three warnings have to arrive together — the lift itself, the resolution floor the
    squared spectrum falls through, and the rough residual — because each names a
    different remedy and only the first explains where the factor of ``kappa**2`` is from.
    """
    A = np.array([[5.0, 4.9], [0.0, 0.2]])
    b = np.array([1.0, 1.0])
    result, warned = _solve_capturing(A, b, n_l=6)

    assert np.linalg.cond(A) == pytest.approx(49.0, rel=0.05)
    assert _cos(result.solution, np.linalg.solve(A, b)) < 0.5
    assert any(
        "not Hermitian" in w and "squaring the condition number" in w for w in warned
    )
    assert any("rotation-free bucket 0" in w for w in warned)
    assert any("relative residual" in w for w in warned)


def test_the_one_sided_window_never_wraps_the_smallest_eigenvalue():
    """One-sided reads *every* bucket ``1 .. N-1`` as a positive eigenvalue, so a phase
    drifting below zero does not become a small negative one — it wraps to bucket ``N-1``
    and the component with the largest ``1/lambda`` weight is handed the smallest.

    :func:`_honest_qpe_params` claims that is unreachable under ``trotter_steps="auto"``,
    since wrapping needs drift past ``C/2`` and the selector holds it at ``C/4``.  This
    checks the claim on the effective generator the circuit actually runs, not on ``A``.
    """
    A = np.array([[3.0, 1.0], [0.0, 2.0]])
    n_l = 6
    n_buckets = 2**n_l
    lifted = np.asarray(A.T @ A, dtype=complex)

    solver = _honest(n_l=n_l)
    t, C, _ = H._honest_qpe_params(lifted, n_l, one_sided=True)
    terms = pauli_decompose(lifted)
    steps = solver._resolve_trotter_steps(terms, t, C, n_l)

    effective = np.sort(np.linalg.eigvalsh(_effective_generator(terms, t / steps)))
    exact = np.sort(np.linalg.eigvalsh(lifted))
    buckets = np.round(effective * t * n_buckets / (2 * np.pi)).astype(int)

    assert np.array_equal(
        buckets, np.round(exact * t * n_buckets / (2 * np.pi)).astype(int)
    )
    assert np.all(buckets >= 1)  # nothing fell into the rotation-free bucket
    assert np.all(buckets <= n_buckets - 1)  # nothing wrapped past the top
    assert np.all(np.diff(buckets) > 0)  # the ordering of 1/lambda survives


def test_a_hermitian_psd_input_is_not_given_the_one_sided_window():
    """Documented asymmetry, pinned so nobody assumes it is symmetric: the extra bit of
    resolution is switched on by ``is_normal_eq``, not by the spectrum, because deciding
    that a Hermitian ``A`` is PSD is exactly the spectral work honest mode refuses to do.

    A directly supplied PSD matrix therefore pays the two-sided ``C``, twice as coarse.
    """
    A = np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=complex)
    assert np.all(np.linalg.eigvalsh(A) > 0)

    _, C_two_sided, _ = H._honest_qpe_params(A, 6, one_sided=False)
    _, C_one_sided, _ = H._honest_qpe_params(A, 6, one_sided=True)
    assert C_two_sided / C_one_sided == pytest.approx(2.0, rel=0.05)

    solver = _honest(n_l=6)
    _, _, C_used, _, _ = solver._honest_setup(A)
    assert C_used == pytest.approx(C_two_sided)


def test_honest_mode_is_never_better_than_the_reference_path():
    """Sanity direction for the whole branch: honest mode replaces an exact dense ``expm``
    oracle and a classically tuned ``t`` with a Trotter product and a Gershgorin bound, so
    it can only lose accuracy — and the gap is what the honest path costs.  An honest
    result that *beats* reference means one of the two is not solving the stated system.
    """
    cases = [
        (TRI4, np.array([1.0, 0.0, 0.0, 0.0])),
        (BAND4, np.ones(4)),
        (np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0])),
    ]
    for A, b in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            honest = _honest(n_l=6).solve(A, b)
            reference = HHL(
                n_eigenvalue_qubits=6, mode="reference", provider="numpy"
            ).solve(A, b)
        assert reference.residual <= honest.residual + 1e-9
        assert _cos(honest.solution, np.linalg.solve(A, b)) > 0.99


def _emittable_warning_patterns() -> list[re.Pattern[str]]:
    """Every warning `hhl.py` can produce, as a regex over the message it builds.

    A recorded warning carries interpolated values, so it cannot be looked for in the
    source as a literal, and neither can a fixed-length prefix of it: two of the three
    messages on the example page interpolate inside their first 40 characters —
    `σ_min ≈ {sigma_min:.3g}` at character 20 and `residual = {residual:.3g}` at 38.

    Reading the f-strings settles it: a literal segment has to appear verbatim, and
    anything interpolated — a value, a `join`, a conditional clause — matches anything.
    That checks the shape of the whole message rather than its head, and stays quiet when
    an interpolated part is reworded.
    """

    def pattern(node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return re.escape(node.value)
        if isinstance(node, ast.JoinedStr):
            return "".join(pattern(value) for value in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return pattern(node.left) + pattern(node.right)
        return "(.+?)"

    tree = ast.parse(inspect.getsource(H))
    return [
        re.compile(pattern(node.args[0]), re.S)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "warn"
        and node.args
    ]


_PROSE = re.compile(r"[A-Za-z_][A-Za-z_-]*(?: [A-Za-z_][A-Za-z_-]*)*")


def _prose_fragments(text: str) -> list[str]:
    """The runs of words in `text` long enough to be prose rather than a value.

    Twelve characters is below the shortest run that is really prose in any current
    message, and above the longest that is part of a formatted number's units.
    """
    return [run.group() for run in _PROSE.finditer(text) if len(run.group()) >= 12]


def _document_text(path: pathlib.Path) -> str:
    """The prose and code of a page or a notebook, as text a warning can be found in."""
    if path.suffix != ".ipynb":
        return path.read_text(encoding="utf-8")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


@pytest.mark.parametrize(
    "document",
    ["docs/examples/hhl-honest.md", "examples/notebooks/hhl-honest.ipynb"],
)
def test_the_example_quotes_warnings_the_code_can_still_emit(document):
    """The published example shows its own captured output, which is what makes it worth
    reading — and what makes it go stale silently when a warning is reworded.

    The notebook is generated from the page, so the two normally agree by construction;
    it is checked as well because it is the copy that gets edited by hand.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    if not (root / "docs").is_dir():
        pytest.skip("docs/ and examples/ are not in the sdist")
    text = _document_text(root / document)

    quoted = re.findall(r"RuntimeWarning: (HHL.+)", text)
    assert quoted, f"no recorded HHL warning found in {document}"

    patterns = _emittable_warning_patterns()
    # The string *constants* of the module, not its text: adjacent literals are one
    # constant to the parser but two lines to a reader, so a phrase that spans the wrap
    # is not a substring of the source.
    literals = "\n".join(
        node.value
        for node in ast.walk(ast.parse(inspect.getsource(H)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )

    def prose_is_in_the_source(found: re.Match[str]) -> bool:
        """Every interpolated fragment that reads as prose has to come from the module.

        A fragment assembled a few lines above the `warn` call and interpolated in —
        `floors` and `remedies` in the σ_min warning — is `.+?` from the call site, so
        the frame alone does not pin it.  Checked against the module's string constants
        rather than its text, so wrapping a literal across two lines is not a change.
        """
        return all(
            fragment in literals
            for capture in found.groups()
            if capture
            for fragment in _prose_fragments(capture)
        )

    stale = []
    for message in sorted(set(quoted)):
        matches = [found for found in (p.fullmatch(message) for p in patterns) if found]
        if not matches:
            stale.append("no warning has this shape: " + message)
        elif not any(prose_is_in_the_source(found) for found in matches):
            stale.append("interpolated wording is not in the source: " + message)
    assert not stale, "the example quotes what hhl.py cannot emit:\n  " + "\n  ".join(
        stale
    )


def test_the_honest_circuit_holds_no_gate_the_reference_one_needs():
    """The point of honest mode, as a set relation: the honest spec is elementary rotations
    and CNOTs, while reference needs a dense oracle and a multi-controlled primitive.
    Anything reintroducing either into honest mode fails here rather than in a residual
    three files away.
    """
    spec, _ = _spec_for(TRI4, np.array([1.0, 0.0, 0.0, 0.0]), 4)
    names = collections.Counter(g.name for g in spec.gates)
    assert names["QubitUnitary"] == 0
    assert names["ControlledQubitUnitary"] == 0
    assert set(names) <= H._HHL_HONEST_REQUIRED_GATES | {"CNOT"}
    assert "ControlledQubitUnitary" in H._HHL_REQUIRED_GATES
