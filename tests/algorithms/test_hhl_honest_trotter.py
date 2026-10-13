"""Trotter-error control in HHL(mode="honest").

The honest path holds the Trotter step size ``τ`` fixed across every controlled
power, so QPE is *exact* phase estimation of the single step unitary and the whole
circuit is ideal HHL for an effective Hamiltonian ``A_eff = Log(W)/(iτ)``.  The
entire Trotter error is therefore one spectral substitution ``A → A_eff`` — and its
dominant component is a rotation of the *eigenvectors*, which every
eigenvalue-based diagnostic is blind to.

These tests cover the machinery that bounds that substitution: the drift bound, the
``trotter_steps="auto"`` selector built on it, and end-to-end witnesses on systems
where the substitution is large enough to decide the answer.
"""

from __future__ import annotations

import math
import warnings
from unittest import mock

import numpy as np
import pytest
import scipy.linalg as sla

from eigenkit.algorithms._hamiltonian_sim import (
    anticommuting_weight_profile,
    pauli_decompose,
    pauli_string_matrix,
    trotter_drift_bound,
)
from eigenkit.algorithms.hhl import (
    _RESIDUAL_WARN_THRESHOLD,
    HHL,
    _compute_rotation_angles,
    _honest_qpe_params,
)
from eigenkit.core.ir import Gate


P = pauli_string_matrix

# The canonical Heisenberg witness: κ = 3, five Pauli terms, local — every
# condition the honest path advertises as favourable.  At trotter_steps=1,
# trotter_order=1 its eigenvalues are exact to 1e-15 while its eigenvectors are
# rotated by 59°, so it isolates the error channel no spectral check can see.
A_HEISENBERG = (P("XX") + P("YY") + P("ZZ") + P("ZI") + P("IZ")).real
# Gershgorin-tight on a negative eigenvalue: λ_min = −Λ = −4 sits at the centre of
# the last safe bucket, so a Trotter drift of order C flips the sign of 1/λ_min.
A_NYQUIST = -np.array([[0.0, 1, 1, 2], [1, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 0]])


def _cos(u, v) -> float:
    # Each vector is scaled by its own largest entry first: the extreme-scale tests
    # below reach solutions of magnitude ~1e-171, where squaring inside a plain
    # np.linalg.norm underflows to zero and the cosine comes back as nan.
    u = np.asarray(u).real.ravel().astype(float)
    v = np.asarray(v).real.ravel().astype(float)
    u = u / (np.max(np.abs(u)) or 1.0)
    v = v / (np.max(np.abs(v)) or 1.0)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def _step_unitary(terms, tau, order):
    """The Trotter step exactly as blocks.py builds it (first term = rightmost)."""
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


def _a_eff(terms, tau, order):
    """The Hermitian generator QPE actually estimates."""
    M = sla.logm(_step_unitary(terms, tau, order)) / (1j * tau)
    return (M + M.conj().T) / 2


# ---------------------------------------------------------------------------
# the drift bound itself
# ---------------------------------------------------------------------------


def test_anticommuting_profile_hand_checked():
    # IZ and ZI commute with each other and with ZZ; both anticommute with XX and
    # YY; XX and YY commute.  In the shipped (lexicographic) order the list is
    # IZ, XX, YY, ZI, ZZ, so the strict-upper-triangle anticommuting partners are
    # IZ→{XX, YY}, XX→{ZI}, YY→{ZI}, ZI→{}, ZZ→{}.
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    assert [s for _, s in terms] == ["IZ", "XX", "YY", "ZI", "ZZ"]
    np.testing.assert_allclose(
        anticommuting_weight_profile(terms), [2.0, 1.0, 1.0, 0.0, 0.0], atol=1e-12
    )


def test_gamma2_constants_are_pinned():
    """Pin the 1/3 and 1/6 of Γ₂, not just its τ² scaling.

    Order 2 is the shipped default and Γ₂ drives every auto-selected step count, but
    the τ² scaling and the upper-bound property both survive a wrong constant, which
    would just make every circuit deeper. Only the values themselves catch that.

    For two anticommuting terms `[(a, "X"), (b, "Z")]`: R = (|ab|, 0), S = (|b|, 0),
    so Γ₂ = (1/3)|b|·|ab| + (1/6)|a|·|ab| = |ab|(2|b| + |a|) / 6.
    """
    for a, b in ((1.0, 1.0), (0.7, 0.4), (2.0, 0.25), (-1.5, 0.8)):
        expected = abs(a * b) * (2 * abs(b) + abs(a)) / 6.0
        assert trotter_drift_bound([(a, "X"), (b, "Z")], 1.0, 2) == pytest.approx(
            expected
        )
    # And the order-1 constant likewise: Γ₁ = |ab| exactly for one anticommuting pair.
    assert trotter_drift_bound([(0.7, "X"), (0.4, "Z")], 1.0, 1) == pytest.approx(0.28)


def test_drift_bound_is_permutation_invariant():
    # Reordering the terms changes the achieved Δ₁ but not the bound: commuting
    # pairs contribute zero wherever they sit, and the bound sums |c_k c_l| over
    # unordered anticommuting pairs.
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    assert trotter_drift_bound(terms[::-1], 0.3, 1) == pytest.approx(
        trotter_drift_bound(terms, 0.3, 1)
    )


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("seed", [0, 1])
def test_drift_bound_upper_bounds_the_real_thing(order, seed):
    """The bound must not under-report the actual ‖A_eff − A‖.

    It is a leading-order Trotter bound, so it is asserted only in the regime the
    selector drives towards: per-step generator angle τ·Σ|c_k| ≤ 0.5, with 1 % slack.
    """
    rng = np.random.default_rng(seed)
    for dim in (2, 4):
        M = rng.standard_normal((dim, dim))
        A = (M + M.T).astype(complex)
        terms = pauli_decompose(A)
        coeff_l1 = sum(abs(complex(c)) for c, _ in terms)
        tau = 0.5 / coeff_l1  # sit exactly on the angle guard
        bound = trotter_drift_bound(terms, tau, order)
        actual = float(np.linalg.norm(_a_eff(terms, tau, order) - A, 2))
        assert bound >= actual * 0.99


# ---------------------------------------------------------------------------
# trotter_steps="auto"
# ---------------------------------------------------------------------------


def test_auto_is_the_default():
    solver = HHL(mode="honest")
    assert solver.trotter_steps == "auto"
    assert solver.trotter_order == 2


def test_auto_leaves_commuting_systems_at_one_step():
    # A diagonal A needs no refinement; auto must not charge for accuracy that a
    # single step already delivers exactly.
    A = np.diag([2.0, 3.0, 1.0, 4.0]).astype(complex)
    solver = HHL(n_eigenvalue_qubits=6, mode="honest", provider="numpy")
    t, C, _ = _honest_qpe_params(A, 6)
    assert solver._resolve_trotter_steps(pauli_decompose(A), t, C, 6) == 1


def test_auto_refines_a_non_commuting_system():
    A = A_HEISENBERG.astype(complex)
    solver = HHL(n_eigenvalue_qubits=6, mode="honest", provider="numpy")
    t, C, _ = _honest_qpe_params(A, 6)
    terms = pauli_decompose(A)
    steps = solver._resolve_trotter_steps(terms, t, C, 6)
    assert steps > 1
    # The chosen grid meets the stated criterion ...
    assert trotter_drift_bound(terms, t / steps, 2) <= solver.trotter_tolerance * C
    # ... and one step coarser does not, i.e. the selector is not over-provisioning.
    assert trotter_drift_bound(
        terms, t / (steps - 1), 2
    ) > solver.trotter_tolerance * C or steps == math.ceil(
        t * sum(abs(complex(c)) for c, _ in terms) / 0.5
    )


def test_auto_step_count_grows_with_clock_qubits():
    # C halves per clock qubit, so the drift target tightens and the grid must get
    # finer: at order 2 steps ∝ √N, which is what turns the Trotter cost from
    # Θ(2^n_l) into Θ(2^{1.5 n_l}).
    A = A_HEISENBERG.astype(complex)
    solver = HHL(mode="honest", provider="numpy")
    terms = pauli_decompose(A)
    counts = []
    for n_l in (4, 6, 8):
        t, C, _ = _honest_qpe_params(A, n_l)
        counts.append(solver._resolve_trotter_steps(terms, t, C, n_l))
    assert counts[0] <= counts[1] <= counts[2]
    assert counts[2] > counts[0]


def test_explicit_coarse_steps_warn_instead_of_failing_silently():
    # A single step leaves the drift far above C/2 on this matrix, so an explicit
    # trotter_steps=1 has to be warned about.
    A = A_HEISENBERG.astype(complex)
    solver = HHL(n_eigenvalue_qubits=6, mode="honest", provider="numpy", trotter_steps=1)
    t, C, _ = _honest_qpe_params(A, 6)
    with pytest.warns(RuntimeWarning, match="Trotter drift bound"):
        assert solver._resolve_trotter_steps(pauli_decompose(A), t, C, 6) == 1


def test_explicit_fine_steps_do_not_warn():
    A = A_HEISENBERG.astype(complex)
    solver = HHL(
        n_eigenvalue_qubits=6, mode="honest", provider="numpy", trotter_steps=200
    )
    t, C, _ = _honest_qpe_params(A, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert solver._resolve_trotter_steps(pauli_decompose(A), t, C, 6) == 200


def test_auto_clamps_to_the_gate_budget_with_a_warning():
    # A large clock register on a many-term matrix asks for a circuit that cannot
    # be built; the selector must clamp and say what it wanted.
    rng = np.random.default_rng(3)
    M = rng.standard_normal((8, 8))
    A = (M + M.T).astype(complex)
    solver = HHL(mode="honest", provider="numpy")
    t, C, _ = _honest_qpe_params(A, 12)
    with pytest.warns(RuntimeWarning, match="gate budget"):
        steps = solver._resolve_trotter_steps(pauli_decompose(A), t, C, 12)
    assert steps >= 1


def test_trotter_tolerance_validated():
    with pytest.raises(ValueError, match="trotter_tolerance"):
        HHL(mode="honest", trotter_tolerance=0.0)
    with pytest.raises(ValueError, match="trotter_tolerance"):
        HHL(mode="honest", trotter_tolerance=2.0)


def test_trotter_grid_arguments_are_validated():
    for bad in ("fine", 2.5, 0, -1):
        with pytest.raises(ValueError, match="trotter_steps"):
            HHL(mode="honest", trotter_steps=bad)
    with pytest.raises(ValueError, match="trotter_order"):
        HHL(mode="honest", trotter_order=3)


def test_tighter_tolerance_costs_more_steps():
    A = A_HEISENBERG.astype(complex)
    terms = pauli_decompose(A)
    t, C, _ = _honest_qpe_params(A, 7)
    loose = HHL(mode="honest", provider="numpy", trotter_tolerance=1.0)
    tight = HHL(mode="honest", provider="numpy", trotter_tolerance=0.05)
    assert tight._resolve_trotter_steps(terms, t, C, 7) > loose._resolve_trotter_steps(
        terms, t, C, 7
    )


# ---------------------------------------------------------------------------
# end-to-end witnesses
# ---------------------------------------------------------------------------


def test_heisenberg_witness_is_solved_at_defaults():
    """Five anticommuting terms, κ = 3: aligned to 0.999 with no warning to excuse it."""
    x_true = np.linalg.solve(A_HEISENBERG, np.ones(4))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # and with no warning to excuse it
        res = HHL(n_eigenvalue_qubits=5, mode="honest", provider="numpy").solve(
            A_HEISENBERG, np.ones(4)
        )
    assert _cos(res.solution, x_true) > 0.999
    assert res.residual < 0.05
    # readout_probability is the joint (ancilla=1 ∧ clock=0) sub-event of the ancilla one
    assert 0.0 < res.readout_probability <= res.success_probability + 1e-12


def test_heisenberg_quality_holds_as_clock_qubits_grow():
    # t grows with N, so a fixed step count would let τ = t/steps grow with it and the
    # quality fall as the register gets finer. The selector has to keep it flat.
    x_true = np.linalg.solve(A_HEISENBERG, np.ones(4))
    quality = [
        _cos(
            HHL(n_eigenvalue_qubits=n_l, mode="honest", provider="numpy")
            .solve(A_HEISENBERG, np.ones(4))
            .solution,
            x_true,
        )
        for n_l in (3, 4, 5)
    ]
    assert min(quality) > 0.99


def test_nyquist_capture_witness_is_solved_at_defaults():
    """λ_min = −Λ sits at the centre of the last safe bucket, so a drift of order C
    pushes it onto the sign-ambiguous Nyquist bucket and inverts the sign of 1/λ."""
    x_true = np.linalg.solve(A_NYQUIST, np.ones(4))
    res = HHL(n_eigenvalue_qubits=5, mode="honest", provider="numpy").solve(
        A_NYQUIST, np.ones(4)
    )
    assert _cos(res.solution, x_true) > 0.99


def test_field_sweep_never_returns_an_antiparallel_answer():
    # A mis-signed 1/λ shows up as an answer pointing the opposite way to the truth,
    # so the field strength is swept and alignment required throughout.
    for h in (0.0, 0.25, 2.0):  # h=1.0 is the Heisenberg witness, covered above
        A = (P("XX") + P("YY") + P("ZZ") + h * (P("ZI") + P("IZ"))).real
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        res = HHL(n_eigenvalue_qubits=5, mode="honest", provider="numpy").solve(
            A, np.ones(4)
        )
        assert _cos(res.solution, np.linalg.solve(A, np.ones(4))) > 0.99


# ---------------------------------------------------------------------------
# the non-spectral diagnostic
# ---------------------------------------------------------------------------


def test_imaginary_readout_is_reported_at_order_one():
    # A = I + X + Z. At order 1 the effective Hamiltonian picks up a Y term, which
    # is imaginary Hermitian, so a real system returns a genuinely complex state —
    # a Trotter error no eigenvalue check can see.
    A = np.array([[2.0, 1.0], [1.0, 0.0]])
    with pytest.warns(RuntimeWarning, match="imaginary"):
        HHL(
            n_eigenvalue_qubits=6,
            mode="honest",
            provider="numpy",
            trotter_steps=1,
            trotter_order=1,
        ).solve(A, np.array([1.0, 1.0]))


def test_no_imaginary_readout_at_order_two():
    # Order 2 is a palindrome, so A_eff stays real for a real A and the imaginary
    # channel closes entirely.  (A coarse grid still earns the drift warning; this
    # test is only about the imaginary one.)
    A = np.array([[2.0, 1.0], [1.0, 0.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HHL(
            n_eigenvalue_qubits=6,
            mode="honest",
            provider="numpy",
            trotter_steps=1,
            trotter_order=2,
        ).solve(A, np.array([1.0, 1.0]))
    assert not any("imaginary" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# one-sided phase window for the normal-equations lift
# ---------------------------------------------------------------------------


def test_one_sided_window_halves_C():
    A = np.diag([1.0, 0.5, 0.25, 2.0]).astype(complex)
    for n_l in (4, 6, 8):
        N = 2**n_l
        _, c_two, lam = _honest_qpe_params(A, n_l, one_sided=False)
        t_one, c_one, _ = _honest_qpe_params(A, n_l, one_sided=True)
        assert c_two == pytest.approx(2.0 * lam / (N - 2))
        assert c_one == pytest.approx(lam / (N - 1))
        assert c_one < c_two
        # The full circle is used, and no phase reaches 2π.
        assert t_one * lam == pytest.approx(2.0 * math.pi * (N - 1) / N)
        assert t_one * lam < 2.0 * math.pi


def test_one_sided_rotation_angles_are_all_positive():
    # With a PSD spectrum every bucket means a positive eigenvalue: no bucket is
    # read as negative, so no sign of 1/λ can be inverted by a wrapped phase.
    A = np.diag([1.0, 0.5]).astype(complex)
    n_l, N = 5, 32
    t, C, lam = _honest_qpe_params(A, n_l, one_sided=True)
    angles = _compute_rotation_angles(N, t, C, lam, epsilon=0.0, one_sided=True)
    assert all(theta > 0 for theta in angles.values())
    assert angles[1] == pytest.approx(math.pi)  # C/λ̃₁ = 1 exactly, no clamping
    # The two-sided table, by contrast, reads the upper half as negative.
    two = _compute_rotation_angles(N, t, C, lam, epsilon=0.0, one_sided=False)
    assert two[N - 1] < 0


def test_normal_equations_use_the_one_sided_window():
    # A non-Hermitian input is lifted to A†A, which solve() knows is PSD, so the honest
    # path spends the whole phase window on it: one extra bit of eigenvalue resolution,
    # paid for with a lower post-selection rate. A Hermitian input must stay two-sided,
    # since nothing rules out a negative eigenvalue there.
    import eigenkit.algorithms.hhl as hhl_mod

    real_params = hhl_mod._honest_qpe_params
    seen: list[bool] = []

    def spy(A_p, n_clock, *, one_sided=False, t=None):
        seen.append(one_sided)
        return real_params(A_p, n_clock, one_sided=one_sided, t=t)

    A = np.array([[1.0, 2.0], [0.0, 1.0]])  # non-Hermitian -> A†A
    b = np.array([1.0, 1.0])
    with mock.patch.object(hhl_mod, "_honest_qpe_params", spy):
        res = HHL(
            n_eigenvalue_qubits=7, mode="honest", provider="numpy", trotter_steps=80
        ).solve(A, b)
        assert seen == [True]
        seen.clear()
        HHL(n_eigenvalue_qubits=7, mode="honest", provider="numpy").solve(
            np.array([[2.0, 1.0], [1.0, 2.0]]), b
        )
        assert seen == [False]
    assert _cos(res.solution, np.linalg.solve(A, b)) > 0.99


# ---------------------------------------------------------------------------
# read-out contract
# ---------------------------------------------------------------------------


def test_raw_solution_is_the_unrescaled_quantum_output():
    # The returned `solution` is `raw.real` times a classical least-squares scale.
    # `raw_solution` exposes what the circuit actually produced, so the claim can be
    # checked and not only documented.
    A = np.array([[2.0, 1.0], [1.0, 0.0]])
    b = np.array([1.0, 1.0])
    res = HHL(n_eigenvalue_qubits=6, mode="honest", provider="numpy").solve(A, b)
    raw = res.raw_solution
    assert raw is not None
    assert np.iscomplexobj(raw)  # unit-norm and complex, whatever the input dtype
    assert np.linalg.norm(raw) == pytest.approx(1.0)
    # solution is parallel (or anti-parallel) to Re(raw): only the direction is
    # quantum, the magnitude and sign come from the classical fit.
    assert abs(_cos(raw.real, res.solution)) == pytest.approx(1.0, abs=1e-9)
    scale = np.dot(A @ raw.real, b) / float(np.dot(A @ raw.real, A @ raw.real))
    np.testing.assert_allclose(res.solution, raw.real * scale, atol=1e-12)


def test_near_hermitian_matrix_is_not_pushed_through_normal_equations():
    # A structural zero facing a 1e-18 partner is Hermitian for every practical
    # purpose. A purely relative symmetry test would call it non-Hermitian and push it
    # through the normal equations, squaring κ to 2500.
    A = np.diag([1.0, 0.02])
    A[1, 0] = 1e-18
    res = HHL(n_eigenvalue_qubits=7, mode="honest", provider="numpy").solve(A, np.ones(2))
    assert _cos(res.solution, np.linalg.solve(A, np.ones(2))) > 0.99


def test_unresolvable_spectrum_names_the_floor_that_binds():
    """Honest mode has two floors under the smallest eigenvalue, and the warning has to
    name whichever one binds: a register floor C that more clock qubits lower, and an
    epsilon floor that they do not.  Both are checked against the true σ_min, and each
    remedy is only offered for the floor it actually moves.
    """
    warn = HHL._warn_on_unresolvable_spectrum
    kw = dict(n_clock=8, one_sided=False, is_normal_eq=False)
    # Comfortably above both floors: silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn(0.5, 0.01, 1.0, 0.001, **kw)
    # Just under C but above C/2: bucket 1, read as C, still a good answer — quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn(0.009, 0.01, 1.0, 0.0001, **kw)
    # Below C/2: rounds into the rotation-free bucket 0. The remedy is clock qubits.
    with pytest.warns(RuntimeWarning, match="rounds into the") as rec:
        warn(0.004, 0.01, 1.0, 0.0001, **kw)
    assert "raise n_eigenvalue_qubits" in str(rec[0].message)
    assert "lower epsilon" not in str(rec[0].message)
    # Below the epsilon floor only: no register size fixes it, and it must say so.
    with pytest.warns(RuntimeWarning, match="epsilon cutoff") as rec:
        warn(0.005, 0.001, 1.0, 0.01, **kw)
    assert "lower epsilon" in str(rec[0].message)
    assert "no register size fixes this" in str(rec[0].message)
    # Below both: both remedies.
    with pytest.warns(RuntimeWarning) as rec:
        warn(0.0005, 0.01, 1.0, 0.01, **kw)
    assert "raise n_eigenvalue_qubits" in str(rec[0].message)
    assert "lower epsilon" in str(rec[0].message)


def test_loose_gershgorin_epsilon_cull_is_not_silent():
    """A Gershgorin bound loose by 1.2× pushes σ_min below `epsilon·‖A‖_bound` while
    κ = 90 stays inside the register's capacity, so at n_l = 8 bucket 1 is culled by
    epsilon alone — reference mode solves the same system exactly. The residual stays
    small enough to pass its own check, so only naming the epsilon floor catches it.
    """
    A = np.array(
        [[0.155932372085, 0.349626855693], [0.349626855693, 0.855178739026]]
    )  # eigenvalues {1, 1/90}, Gershgorin bound 1.205× λ_max
    b = np.array([0.336013275261, 0.941857249719])
    with pytest.warns(RuntimeWarning, match="epsilon cutoff"):
        res = HHL(n_eigenvalue_qubits=8, mode="honest", provider="numpy").solve(A, b)
    assert (
        res.residual < _RESIDUAL_WARN_THRESHOLD
    )  # the residual alone would not catch it
    # And the remedy the warning gives actually works.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fixed = HHL(
            n_eigenvalue_qubits=8, mode="honest", provider="numpy", epsilon=0.004
        ).solve(A, b)
    assert _cos(fixed.solution, np.linalg.solve(A, b)) > 0.999


def test_no_culling_noise_on_a_well_conditioned_system_at_high_n_l():
    # diag(2, 1) at n_l=8 returns cos = 1.000000. The angle table has zeroed buckets at
    # that register size whatever the spectrum, so their presence alone must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = HHL(n_eigenvalue_qubits=8, mode="honest", provider="numpy").solve(
            np.diag([2.0, 1.0]), np.array([1.0, 1.0])
        )
    assert res.residual < 0.01


def test_pauli_decomposition_runs_once_per_solve():
    # It is the dominant classical cost of the honest path, so solve() computes it once
    # and passes it down to the circuit builder.
    import eigenkit.algorithms.hhl as hhl_mod

    calls = []
    original = hhl_mod.pauli_decompose

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    hhl_mod.pauli_decompose = counting
    try:
        HHL(n_eigenvalue_qubits=4, mode="honest", provider="numpy").solve(
            np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0])
        )
    finally:
        hhl_mod.pauli_decompose = original
    assert len(calls) == 1


def test_term_order_is_pinned():
    """Honest-mode results depend on the order the Pauli factors are applied.

    The first-order Trotter error is ``-(iτ/2) Σ_{k<l} c_k c_l [P_k, P_l]`` with
    ``k < l`` in *application* order, so permuting this list changes every
    honest-mode answer at a fixed ``trotter_steps``.  The order is lexicographic
    over ``I < X < Y < Z`` and is part of the numerical contract.
    """
    rng = np.random.default_rng(4)
    M = rng.standard_normal((4, 4))
    A = (M + M.T).astype(complex)
    strings = [s for _, s in pauli_decompose(A)]
    # A real symmetric A has no support on strings with an odd number of Y factors,
    # so the surviving subset is this, in lexicographic order over I < X < Y < Z.
    assert strings == [
        "II",
        "IX",
        "IZ",
        "XI",
        "XX",
        "XZ",
        "YY",
        "ZI",
        "ZX",
        "ZZ",
    ]
    # And the Heisenberg witness, whose ordering decides the sign pattern of the
    # first-order commutator sum: 40 of its 120 orderings make Δ₁ vanish exactly
    # (total-S_z symmetry), and lexicographic is not one of them.
    golden = pauli_decompose(A_HEISENBERG.astype(complex))
    assert [s for _, s in golden] == ["IZ", "XX", "YY", "ZI", "ZZ"]
    np.testing.assert_allclose(
        [complex(c).real for c, _ in golden], [1.0] * 5, atol=1e-12
    )


def test_trotter_block_applies_the_first_term_rightmost():
    """The other half of the ordering contract: application orientation.

    ``TrotterBlock`` emits the terms in list order, and gates appended later act
    later, so the first list entry is the *rightmost* factor of the step unitary.
    Every drift bound and every pinned honest-mode number is stated against that
    orientation; silently reversing it would flip the sign of Δ₁.
    """
    from eigenkit.core.blocks import TrotterBlock, assemble
    from eigenkit.core.circuit import Circuit

    terms = [(0.7, "X"), (0.4, "Z")]  # X and Z anticommute, so the order shows up
    tau = 0.9
    spec = assemble([TrotterBlock(tuple(terms), time=tau, steps=1, order=1)])
    got = np.asarray(Circuit(spec, provider="numpy").statevector(np.zeros(0)))

    def _aligned(u, v):
        # Equal up to the global phase the identity-free product leaves free.
        return abs(abs(np.vdot(v, u)) - 1.0) < 1e-9

    e0 = np.array([1.0, 0.0], dtype=complex)
    assert _aligned(got, _step_unitary(terms, tau, 1) @ e0)
    assert not _aligned(got, _step_unitary(terms[::-1], tau, 1) @ e0)


def test_lexicographic_order_reproduces_the_documented_witness_value():
    """End-to-end canary on the term order.

    At the pinned lexicographic order with ``trotter_order=1, trotter_steps=1`` the
    Heisenberg witness returns cos = 0.641 — still poor, which is the point the auto
    selector exists for.  The number is *bad on purpose*: its job is to fail loudly if
    the decomposition order ever changes, because 40 of the 120 orderings would turn
    this into an exact answer and the rest into something else again.  It is also the
    end-to-end fingerprint of the first-order inverse QPE, which is an adjoint only
    when it reverses the factor order as well as negating τ.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = HHL(
            n_eigenvalue_qubits=6,
            mode="honest",
            provider="numpy",
            trotter_steps=1,
            trotter_order=1,
        ).solve(A_HEISENBERG, np.ones(4))
    x_true = np.linalg.solve(A_HEISENBERG, np.ones(4))
    assert _cos(res.solution, x_true) == pytest.approx(0.64089, abs=2e-3)


def test_density_matrix_backend_raises_the_documented_error():
    # default.mixed passes the `callable(statevector)` check but returns a 2-D object,
    # which cannot supply the amplitude slice the solution is read from.
    pytest.importorskip("pennylane")
    from eigenkit import PennyLaneConfig as pennylane_config

    with pytest.raises(NotImplementedError, match="pure-statevector"):
        HHL(
            n_eigenvalue_qubits=4,
            mode="honest",
            provider=pennylane_config(device_name="default.mixed"),
        ).solve(np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0]))


@pytest.mark.parametrize("scale", [1e-110, 1e-9, 1e150])
def test_auto_step_count_is_scale_invariant(scale):
    """The drift criterion is homogeneous in the coefficient scale, so `steps` must not
    depend on the units of A.  Γ₂ grows as the cube of that scale, and these three
    scales straddle both ends of float64: computed on unnormalised coefficients it
    overflows above ~1e103 and underflows to zero below ~1e-110, where a non-commuting
    Hamiltonian reads as a commuting one.
    """
    A = (scale * A_HEISENBERG).astype(complex)
    solver = HHL(n_eigenvalue_qubits=6, mode="honest", provider="numpy")
    t, C, _ = _honest_qpe_params(A, 6)
    reference = solver._resolve_trotter_steps(
        pauli_decompose(A_HEISENBERG.astype(complex)),
        *_honest_qpe_params(A_HEISENBERG.astype(complex), 6)[:2],
        6,
    )
    assert solver._resolve_trotter_steps(pauli_decompose(A), t, C, 6) == reference


def test_genuinely_complex_small_scale_system_stays_complex():
    """A relative realness test, not an absolute one.

    `np.allclose(A.imag, 0)` has an absolute atol of 1e-8, which classifies a genuinely
    complex Hermitian A of scale 1e-9 as real — dropping the imaginary part of the
    answer and evaluating `residual` against `A.real`, where it reads ~1e-16 for a
    solution whose true residual is 0.5.
    """
    A = 1e-9 * np.array([[2.0, 1j], [-1j, 2.0]])
    b = np.array([1.0, 0.0])
    res = HHL(n_eigenvalue_qubits=6, mode="honest", provider="numpy").solve(A, b)
    assert np.iscomplexobj(res.solution)
    true_residual = np.linalg.norm(A @ res.solution - b) / np.linalg.norm(b)
    assert res.residual == pytest.approx(true_residual, rel=1e-6)
    assert _cos(np.asarray(res.solution).real, np.linalg.solve(A, b).real) > 0.99


@pytest.mark.parametrize(
    "a_scale,b_scale",
    [(1.0, 1e-16), (1e155, 1.0), (1e155, 1e16)],  # ‖b‖ floor, ‖A·raw‖² overflow, both
)
def test_residual_is_scale_free(a_scale, b_scale):
    """The residual is evaluated in a normalised frame, so it means the same thing at
    every magnitude: an absolute floor on ‖b‖ or ‖x‖ misreads good small-scale answers,
    and ‖A·raw‖² overflows above ‖A‖ ≈ 1e154."""
    A = a_scale * np.array([[2.0, 1.0], [1.0, 0.0]])
    b = b_scale * np.array([1.0, 0.0])
    res = HHL(n_eigenvalue_qubits=6, provider="numpy").solve(A, b)
    true_residual = np.linalg.norm(A @ res.solution - b) / np.linalg.norm(b)
    assert res.residual == pytest.approx(true_residual, rel=1e-6)
    assert res.residual < 0.05
    assert _cos(res.solution, np.linalg.solve(A, b)) > 0.99


def test_register_capacity_warning_uses_the_two_sided_capacity():
    """A two-sided register resolves κ up to (N−2)/2, not N, so both the threshold and
    the `n_eigenvalue_qubits` the message recommends follow the two-sided capacity.
    """
    b = np.array([1.0, 1.0]) / math.sqrt(2.0)
    # Inside capacity (7 at n_l=4): quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HHL(n_eigenvalue_qubits=4, mode="honest", provider="numpy").solve(
            np.diag([1.0, 1.0 / 6.0]), b
        )
    # Beyond it: warns, and the recommendation actually fixes the system.
    A = np.diag([1.0, 1.0 / 24.0])
    with pytest.warns(RuntimeWarning, match="rounds into the"):
        HHL(n_eigenvalue_qubits=4, mode="honest", provider="numpy").solve(A, b)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = HHL(n_eigenvalue_qubits=6, mode="honest", provider="numpy").solve(A, b)
    assert res.residual < 0.1


# ---------------------------------------------------------------------------
# the inversion stage
# ---------------------------------------------------------------------------


def test_inversion_is_elementary_and_costs_two_gates_per_bucket():
    """The eigenvalue inversion emits 2^n_l RY and 2^n_l CNOT — nothing else.

    The uniformly controlled decomposition (Möttönen et al., quant-ph/0407010) gives
    each bucket its own angle with two elementary gates and no ancilla, which is what
    keeps every stage of the honest circuit free of multi-controlled and dense gates.
    """
    from eigenkit.algorithms.hhl import _uniformly_controlled_ry

    for n_clock in (2, 3, 5):
        n_buckets = 1 << n_clock
        angles = {j: 0.3 * (j + 1) for j in range(1, n_buckets)}
        gates = _uniformly_controlled_ry(angles, n_clock, tuple(range(1, 1 + n_clock)), 0)
        names = [g.name for g in gates]
        assert names.count("RY") == n_buckets
        assert names.count("CNOT") == n_buckets
        assert set(names) == {"RY", "CNOT"}
    # An all-zero table (every bucket culled) needs no gates at all.
    assert _uniformly_controlled_ry({1: 0.0, 2: 0.0}, 2, (1, 2), 0) == []


def test_inversion_applies_the_requested_rotation_to_every_bucket():
    """Block j of the stage's unitary must be exactly RY(θ_j).

    The uniformly controlled form has that property only if the Gray-code walk and the
    angle transform are both right, so the stage is reconstructed and compared against
    the definition.
    """
    from eigenkit.algorithms.hhl import _uniformly_controlled_ry
    from eigenkit.core.circuit import Circuit
    from eigenkit.core.ir import CircuitSpec

    def ry(t):
        c, s = math.cos(t / 2.0), math.sin(t / 2.0)
        return np.array([[c, -s], [s, c]], dtype=complex)

    n_clock = 3
    n_buckets = 1 << n_clock
    rng = np.random.default_rng(0)
    theta = {j: float(rng.uniform(-math.pi, math.pi)) for j in range(1, n_buckets)}
    # Layout [target(MSB) | clock] matches HHL's [ancilla | clock | b_reg] with no b_reg.
    gates = _uniformly_controlled_ry(theta, n_clock, tuple(range(1, 1 + n_clock)), 0)
    spec = CircuitSpec(n_qubits=1 + n_clock, n_weights=0, n_inputs=0, gates=tuple(gates))

    # Recover the unitary column by column from statevector runs on basis states.
    dim = 1 << (1 + n_clock)
    columns = []
    for basis in range(dim):
        prep = tuple(
            Gate(name="X", qubits=(q,))
            for q in range(1 + n_clock)
            if (basis >> (n_clock - q)) & 1
        )
        full = CircuitSpec(
            n_qubits=1 + n_clock, n_weights=0, n_inputs=0, gates=prep + spec.gates
        )
        columns.append(
            np.asarray(Circuit(full, provider="numpy").statevector(np.zeros(0)))
        )
    unitary = np.array(columns).T

    for bucket in range(n_buckets):
        # target is the MSB, so the two amplitudes for clock=bucket sit `n_buckets` apart
        idx = [bucket, bucket + n_buckets]
        block = unitary[np.ix_(idx, idx)]
        np.testing.assert_allclose(block, ry(theta.get(bucket, 0.0)), atol=1e-12)


@pytest.mark.parametrize("mode", ["honest", "reference"])
def test_declared_gate_set_matches_what_each_mode_emits(mode):
    """The provider gate check must not pass a provider the circuit will then break on,
    so each declared set is compared against what that mode actually emits. A complex
    Hermitian A carries Y strings, which is what pulls RX into the honest circuit.
    """
    from eigenkit.algorithms.hhl import (
        _HHL_HONEST_REQUIRED_GATES,
        _HHL_REQUIRED_GATES,
    )

    A = np.array([[2.0, 1j], [-1j, 2.0]])
    solver = HHL(n_eigenvalue_qubits=4, mode=mode, provider="numpy", trotter_steps=2)
    t, C, lam = _honest_qpe_params(A, 4)
    spec = solver._build_circuit_spec(
        A,
        np.array([1.0, 0.0]),
        1,
        4,
        t,
        _compute_rotation_angles(16, t, C, lam, 0.01),
        pauli_terms=pauli_decompose(A),
        trotter_steps=2,
    )
    declared = _HHL_HONEST_REQUIRED_GATES if mode == "honest" else _HHL_REQUIRED_GATES
    assert {g.name for g in spec.gates} <= declared
