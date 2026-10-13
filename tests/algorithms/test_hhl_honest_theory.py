"""Does the honest circuit compute what HHL says it computes?

The tests above this one check that each part is built correctly.  These check the
assembled circuit against the *algorithm*, by predicting its output from theory and
comparing amplitudes — not residuals:

* on grid, the post-selected state must be ``Σ_j β_j (C/λ_j) u_j`` exactly, and the
  ancilla weight must be ``Σ_j |β_j|² (C/λ_j)²`` exactly;
* off grid, it must equal the Fejér-averaged transfer function
  ``Σ_j β_j w_j v_j^eff``, ``w_j = Σ_m |D_m(φ_j)|² sin(θ_m/2)``, applied to the
  eigenvectors of the **effective** generator ``A_eff = Log(W)/(iτ)``;
* the Trotter error must be exactly that one substitution ``A → A_eff``, whose size the
  shipped drift bound has to dominate.

The last point is the easy one to miss: the
fixed-``τ`` deformation can be *exactly isospectral* — eigenvalues right to 1e-15 —
while rotating the eigenbasis by 50°, so every eigenvalue-based check passes and the
answer is still wrong.  The tests here measure the rotation directly.
"""

from __future__ import annotations

import math
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

# 2I + X: commuting terms, so every Trotter product is exact and the only error
# channels left are the ones QPE itself has.
A_COMMUTING = np.array([[2.0, 1.0], [1.0, 2.0]])
TRI4 = np.array([[2.0, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]])
A_HEISENBERG = (P("XX") + P("YY") + P("ZZ") + P("ZI") + P("IZ")).real
# Gershgorin-tight on λ_min = −4: the extremal eigenvalue sits at the centre of the
# last safe bucket, so a Trotter drift of order C flips the sign of its inverse.
A_NYQUIST = -np.array([[0.0, 1, 1, 2], [1, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 0]])


def _cos(u, v) -> float:
    u = np.asarray(u).real.ravel().astype(float)
    v = np.asarray(v).real.ravel().astype(float)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def _step_unitary(terms, tau: float, order: int) -> np.ndarray:
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


def _a_eff(terms, tau: float, order: int) -> np.ndarray:
    """The Hermitian generator QPE actually estimates, ``Log(W)/(iτ)``."""
    M = sla.logm(_step_unitary(terms, tau, order)) / (1j * tau)
    return (M + M.conj().T) / 2


def _eigenvector_tilt(A: np.ndarray, B: np.ndarray) -> float:
    """Largest angle between corresponding eigenvectors of ``A`` and ``B``, degrees."""
    _, Ua = np.linalg.eigh(np.asarray(A, dtype=complex))
    _, Ub = np.linalg.eigh(np.asarray(B, dtype=complex))
    overlap = np.abs(np.diag(Ua.conj().T @ Ub)).min()
    return float(np.degrees(np.arccos(np.clip(overlap, -1.0, 1.0))))


def _internals(A, b, n_l: int, *, steps=None, order: int = 2, epsilon: float = 1e-2):
    """Run the honest pipeline and expose the raw post-selected slice.

    Deliberately goes through the same private helpers ``solve()`` uses, so the
    predictions below are compared against the shipped circuit rather than a replica.
    """
    kwargs = dict(
        n_eigenvalue_qubits=n_l,
        mode="honest",
        trotter_order=order,
        epsilon=epsilon,
        provider="numpy",
    )
    if steps is not None:
        kwargs["trotter_steps"] = steps
    solver = HHL(**kwargs)
    Ac = np.asarray(A, dtype=complex)
    A_circuit, t, C, lam_bound, _ = solver._honest_setup(Ac)
    terms = pauli_decompose(A_circuit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        used_steps = solver._resolve_trotter_steps(terms, t, C, n_l)
    angles = H._compute_rotation_angles(2**n_l, t, C, lam_bound, epsilon)
    n_b = int(math.log2(Ac.shape[0]))
    b_unit = np.asarray(b, dtype=complex) / np.linalg.norm(b)
    sv = solver._run_circuit(
        A_circuit,
        b_unit,
        n_b,
        n_l,
        t,
        angles,
        pauli_terms=terms,
        trotter_steps=used_steps,
    )
    amps, success, readout = H._postselect(sv, n_b, n_l)
    return dict(
        amps=amps,
        success=success,
        readout=readout,
        t=t,
        C=C,
        lam_bound=lam_bound,
        terms=terms,
        steps=used_steps,
        angles=angles,
        b_unit=b_unit,
        order=order,
    )


def _dirichlet_weights(phi: float, n_buckets: int, angles: dict[int, float]) -> float:
    """``w = Σ_m |D_m(φ)|² sin(θ_m/2)`` — the Fejér average of the inversion table."""
    k = np.arange(n_buckets)
    m = np.arange(n_buckets)
    amplitudes = np.exp(1j * np.outer(k, phi - 2 * np.pi * m / n_buckets)).sum(axis=0)
    amplitudes = amplitudes / n_buckets
    f = np.array([math.sin(angles.get(int(i), 0.0) / 2) for i in m])
    return float(np.sum(np.abs(amplitudes) ** 2 * f))


def _transfer_prediction(d: dict, n_l: int) -> np.ndarray:
    """``x_raw ∝ Σ_j β_j w_j v_j^eff``, evaluated in the effective eigenbasis."""
    A_eff = _a_eff(d["terms"], d["t"] / d["steps"], d["order"])
    lam_eff, V = np.linalg.eigh(A_eff)
    beta = V.conj().T @ d["b_unit"]
    out = np.zeros(V.shape[0], dtype=complex)
    for j, lam in enumerate(lam_eff):
        w = _dirichlet_weights(float(lam) * d["t"], 2**n_l, d["angles"])
        out = out + beta[j] * w * V[:, j]
    return out


# ---------------------------------------------------------------------------
# HHL's own amplitude formula, on grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_l", [5, 7])
def test_on_grid_amplitudes_reproduce_hhl_theory_to_machine_precision(n_l):
    """The post-selected state *is* ``Σ_j β_j (C/λ_j) u_j``, component by component.

    ``2I + X`` has commuting terms (Trotter exact) and, with the honest ``t``, its
    spectrum {1, 3} lands on buckets ``λ(N−2)/(2Λ)`` — integers.  So there is no Trotter
    error, no Dirichlet leakage and no bucket rounding: any deviation is a genuine
    defect in the inversion or the read-out, not an approximation.
    """
    b = np.array([1.0, 0.0])
    d = _internals(A_COMMUTING, b, n_l)
    lam, U = np.linalg.eigh(A_COMMUTING.astype(complex))
    # Confirm the premise: every eigenvalue is exactly on a bucket centre.
    buckets = lam * d["t"] * 2**n_l / (2 * math.pi)
    np.testing.assert_allclose(buckets, np.round(buckets), atol=1e-10)

    beta = U.conj().T @ d["b_unit"]
    ideal = U @ (beta * (d["C"] / lam))
    ideal /= np.linalg.norm(ideal)
    got = d["amps"]
    phase = np.vdot(ideal, got)
    phase /= abs(phase)
    assert np.linalg.norm(got / phase - ideal) < 1e-12


@pytest.mark.parametrize("n_l", [5, 7])
def test_on_grid_success_probability_reproduces_hhl_theory_exactly(n_l):
    """``P = Σ_j |β_j|² (C/λ_j)²`` — the ratio must be 1, not merely close."""
    d = _internals(A_COMMUTING, np.array([1.0, 0.0]), n_l)
    lam, U = np.linalg.eigh(A_COMMUTING.astype(complex))
    beta = U.conj().T @ d["b_unit"]
    predicted = float(np.sum(np.abs(beta) ** 2 * (d["C"] / lam) ** 2))
    assert d["success"] / predicted == pytest.approx(1.0, rel=1e-9)
    # On grid the inverse QPE returns the clock exactly to |0⟩, so the joint event
    # carries the whole ancilla weight.
    assert d["readout"] / d["success"] == pytest.approx(1.0, rel=1e-9)


def test_off_grid_leaks_weight_out_of_the_read_out_slice():
    """The two probabilities separate exactly when the spectrum is off grid.

    ``readout_probability`` is the physical post-selection rate; reporting
    ``success_probability`` for it would overstate the achievable shot count.
    """
    d = _internals(np.diag([1.0, 0.3]), np.array([1.0, 1.0]), 5)
    assert d["readout"] < d["success"]
    assert 0.0 < d["readout"] / d["success"] < 1.0


# ---------------------------------------------------------------------------
# the exact transfer function, off grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "A,b,n_l,order",
    [
        (A_COMMUTING, np.array([1.0, 0.0]), 4, 2),
        (np.diag([1.0, 0.25]), np.array([1.0, 1.0]), 5, 2),
        (TRI4, np.array([1.0, 0.0, 0.0, 0.0]), 4, 2),
        (TRI4, np.array([1.0, 0.0, 0.0, 0.0]), 5, 1),
        (A_HEISENBERG, np.ones(4), 4, 2),
        (A_HEISENBERG, np.ones(4), 4, 1),
    ],
)
def test_the_circuit_realises_the_predicted_transfer_function(A, b, n_l, order):
    """Every error channel at once, predicted from theory and matched to 1e-9.

    ``x_raw ∝ Σ_j β_j w_j v_j^eff`` holds only if three separate things are right: the
    forward pass estimates ``A_eff``'s phases, the inversion table is applied as
    ``sin(θ_m/2)`` per bucket, and the backward pass is the **exact adjoint** of the
    forward one.  If the uncompute were merely the negative-time product, the
    clock=|0⟩ slice would pick up extra phases and this would fail at order 1.
    """
    d = _internals(A, b, n_l, order=order)
    predicted = _transfer_prediction(d, n_l)
    got = d["amps"][: A.shape[0]]
    predicted /= np.linalg.norm(predicted)
    got = got / np.linalg.norm(got)
    assert abs(np.vdot(predicted, got)) == pytest.approx(1.0, abs=1e-9)


def test_the_inversion_weight_is_exact_on_grid_and_biased_off_it():
    """The documented Dirichlet table: on grid ``w·x = 1``, off grid it is a *bias*.

    A component half a bucket off is under-weighted by ~9 %, one at 0.4 buckets by
    91 %, and a scalar rescale cannot undo it because the factor differs per
    eigenvalue.  This is the sharp content of ``N ≳ κ_G + 2``.
    """
    n_buckets = 32
    angles = {
        j: 2 * math.asin(1.0 / (j if j <= n_buckets // 2 else j - n_buckets))
        for j in range(1, n_buckets)
    }
    expected = {
        1.0: 1.0000,
        1.5: 0.9098,
        3.5: 0.9826,
        0.8: 0.7014,
        0.6: 0.3349,
        0.4: 0.0883,
    }
    for x, want in expected.items():
        phi = 2 * math.pi * x / n_buckets
        got = _dirichlet_weights(phi, n_buckets, angles) * x
        assert got == pytest.approx(want, abs=5e-4), x


def test_bucket_zero_leakage_grows_as_the_component_approaches_the_floor():
    """Why the register-floor warning triggers at ``C/2`` and not later.

    Inside the first bucket the loss into the rotation-free bucket 0 grows
    monotonically as ``λ/C → 0``, reaching a quarter of the component's weight at
    ``0.6·C`` and over half at ``0.4·C`` — and it is the component with the largest
    ``1/λ`` weight, i.e. the direction the answer depends on most.
    """
    n_buckets = 32

    def leakage(x: float) -> float:
        phi = 2 * math.pi * x / n_buckets
        k = np.arange(n_buckets)
        return float(abs(np.exp(1j * k * phi).sum() / n_buckets) ** 2)

    inside_first_bucket = [leakage(x) for x in (0.9, 0.8, 0.6, 0.4, 0.2)]
    assert inside_first_bucket == sorted(inside_first_bucket)
    assert leakage(0.6) > 0.25 and leakage(0.4) > 0.5
    # A component several buckets up keeps essentially all of its weight, and an
    # on-grid one loses nothing at all.
    assert leakage(2.5) < 0.02
    assert leakage(3.0) < 1e-20


# ---------------------------------------------------------------------------
# the A_eff frame: what the Trotter error actually is
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("steps", [1, 4])
def test_qpe_buckets_are_those_of_the_step_unitary_not_of_A(order, steps):
    """Fixed ``τ`` ⇒ the estimated phases are exactly ``arg eig(W)``, to 1e-12.

    This is the identity that makes the whole Trotter error a single substitution
    ``A → A_eff`` instead of a per-power error that compounds.
    """
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    t, _, _ = H._honest_qpe_params(A_HEISENBERG.astype(complex), 6)
    tau = t / steps
    W = np.linalg.matrix_power(_step_unitary(terms, tau, order), steps)
    A_eff = _a_eff(terms, tau, order)
    phases_from_eff = np.sort(np.linalg.eigvalsh(A_eff) * t)
    phases_from_W = np.sort(np.angle(np.linalg.eigvals(W)))
    np.testing.assert_allclose(phases_from_eff, phases_from_W, atol=1e-12)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_drift_bound_dominates_the_true_substitution(order, seed):
    """``trotter_drift_bound`` must be an upper bound on ``‖A_eff − A‖``, not a fit.

    Every safety claim in honest mode — no Nyquist capture, no branch-cut wraparound,
    bounded eigenvector rotation — is a statement about ``‖A_eff − A‖`` reached through
    this bound, so an optimistic bound is a silent correctness bug.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    A = m + m.conj().T
    terms = pauli_decompose(A)
    A_from_terms = sum(c * P(s) for c, s in terms)
    for tau in (0.01, 0.03, 0.1):
        true_drift = float(np.linalg.norm(_a_eff(terms, tau, order) - A_from_terms, 2))
        bound = trotter_drift_bound(terms, tau, order)
        assert true_drift <= bound * (1 + 1e-9), (tau, true_drift, bound)


def test_the_trotter_error_can_be_exactly_isospectral_yet_rotate_the_eigenbasis():
    """The blind spot of eigenvalue-only checks, reproduced as a measurement.

    On the Heisenberg witness at ``order=1, steps=1`` the eigenvalues of ``A_eff``
    match ``A``'s to machine precision while the eigenvectors are rotated by ~50°.  So
    no bucket check, no ``κ_G/N`` check and no Nyquist check can detect this error
    class — which is why the step selector bounds ``‖A_eff − A‖`` instead, and why the
    imaginary-read-out detector exists.
    """
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    t, C, _ = H._honest_qpe_params(A_HEISENBERG.astype(complex), 6)
    A_eff = _a_eff(terms, t, 1)  # steps = 1

    spectral_error = float(
        np.max(
            np.abs(
                np.sort(np.linalg.eigvalsh(A_eff))
                - np.sort(np.linalg.eigvalsh(A_HEISENBERG.astype(complex)))
            )
        )
    )
    assert spectral_error < 1e-12  # a spectrum-based check sees nothing at all
    assert _eigenvector_tilt(A_HEISENBERG, A_eff) > 40.0  # yet the basis has moved
    # And the drift is enormous compared with the resolution the register claims.
    assert np.linalg.norm(A_eff - A_HEISENBERG, 2) > 10 * C


def test_the_shipped_defaults_keep_the_effective_generator_close_to_A():
    """The same witness under ``trotter_steps='auto'``: drift inside ``C/4``, tilt ≪ 1°."""
    for n_l in (4, 6):
        terms = pauli_decompose(A_HEISENBERG.astype(complex))
        t, C, _ = H._honest_qpe_params(A_HEISENBERG.astype(complex), n_l)
        solver = HHL(n_eigenvalue_qubits=n_l, mode="honest", provider="numpy")
        steps = solver._resolve_trotter_steps(terms, t, C, n_l)
        A_eff = _a_eff(terms, t / steps, 2)
        assert np.linalg.norm(A_eff - A_HEISENBERG, 2) <= 0.25 * C
        assert _eigenvector_tilt(A_HEISENBERG, A_eff) < 1.0


def test_order_two_keeps_the_effective_generator_real_for_a_real_matrix():
    """At order 1, ``i[X, Z] = 2Y`` puts imaginary Hermitian terms into ``A_eff``.

    Those are what rotate the eigenvectors of a real system into complex ones, and the
    symmetric product cancels them structurally — the reason order 2 is the default.
    """
    A = np.array([[2.0, 1.0], [1.0, 0.0]])  # = I + X + Z
    terms = pauli_decompose(A.astype(complex))
    tau = 0.5
    eff_one = _a_eff(terms, tau, 1)
    eff_two = _a_eff(terms, tau, 2)
    assert np.max(np.abs(eff_one.imag)) > 0.1
    assert np.max(np.abs(eff_two.imag)) < 1e-12
    assert [s for _, s in pauli_decompose(eff_one, rtol=1e-6) if "Y" in s]
    assert not [s for _, s in pauli_decompose(eff_two, rtol=1e-6) if "Y" in s]


def test_a_finer_grid_shrinks_the_eigenvector_tilt_at_the_documented_rate():
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    t = 1.0
    tilts = {
        order: [
            _eigenvector_tilt(A_HEISENBERG, _a_eff(terms, t / steps, order))
            for steps in (4, 8, 16, 32)
        ]
        for order in (1, 2)
    }
    for order, expected in ((1, -1.0), (2, -2.0)):
        slope = float(np.polyfit(np.log([4, 8, 16, 32.0]), np.log(tilts[order]), 1)[0])
        assert slope == pytest.approx(expected, abs=0.2), (order, tilts[order])


# ---------------------------------------------------------------------------
# against reference mode, across providers, and against itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "A,b",
    [
        (A_COMMUTING, np.array([1.0, 0.0])),
        (np.diag([1.0, 0.25]), np.array([1.0, 1.0])),
        (A_HEISENBERG, np.ones(4)),
        (A_NYQUIST, np.ones(4)),
        (TRI4, np.array([1.0, 0.0, 0.0, 0.0])),
        (np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([1.0, 1.0])),  # indefinite
        (np.array([[2.0, 1j], [-1j, 2.0]]), np.array([1.0, 0.0])),  # complex Hermitian
    ],
)
def test_honest_mode_agrees_with_reference_mode(A, b):
    """Reference mode diagonalises classically, so it is the ground truth here.

    The witnesses span the hard cases for honest mode — commuting and anticommuting
    terms, a Gershgorin-tight negative extremum, an indefinite and a complex Hermitian
    system — on all of which reference mode is exact, so honest mode has to agree.
    """
    expected = np.linalg.solve(np.asarray(A, dtype=complex), np.asarray(b, dtype=complex))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        honest = HHL(n_eigenvalue_qubits=6, mode="honest", provider="numpy").solve(A, b)
        reference = HHL(n_eigenvalue_qubits=6, provider="numpy").solve(A, b)
    truth = expected.real if np.max(np.abs(expected.imag)) < 1e-12 else expected
    assert abs(np.vdot(_normalise(honest.solution), _normalise(truth))) > 0.99
    assert (
        abs(np.vdot(_normalise(honest.solution), _normalise(reference.solution))) > 0.99
    )
    assert honest.residual < 0.05


def _normalise(v):
    v = np.asarray(v).ravel()
    return v / np.linalg.norm(v)


def test_the_result_is_bitwise_reproducible():
    """Honest mode has no sampling anywhere; two identical calls must agree exactly."""
    first, second = (
        HHL(n_eigenvalue_qubits=5, mode="honest", provider="numpy").solve(
            TRI4, np.array([1.0, 0.0, 0.0, 0.0])
        )
        for _ in range(2)
    )
    assert np.array_equal(first.solution, second.solution)
    assert np.array_equal(first.raw_solution, second.raw_solution)
    assert first.success_probability == second.success_probability
    assert first.readout_probability == second.readout_probability


@pytest.mark.parametrize("other", ["pennylane", "qiskit"])
def test_providers_agree_on_the_same_circuit(other):
    """The IR is provider agnostic; a backend-specific gate convention would show here."""
    if other == "qiskit":
        pytest.importorskip("qiskit")
    results = {
        provider: HHL(n_eigenvalue_qubits=5, mode="honest", provider=provider).solve(
            TRI4, np.array([1.0, 0.0, 0.0, 0.0])
        )
        for provider in ("numpy", other)
    }
    a, b = (np.asarray(r.solution) for r in results.values())
    np.testing.assert_allclose(a, b, atol=1e-12)
    probs = [r.readout_probability for r in results.values()]
    assert probs[0] == pytest.approx(probs[1], rel=1e-10)


def test_the_full_circuit_spec_round_trips_through_serialisation():
    """The honest blocks are registered, so a saved bundle rebuilds the same gates."""
    from eigenkit.core.blocks import HamiltonianQPEBlock, TrotterBlock

    terms = tuple(
        (complex(c).real, s) for c, s in pauli_decompose(A_HEISENBERG.astype(complex))
    )
    for block in (
        TrotterBlock(terms, time=0.7, steps=3, order=2),
        HamiltonianQPEBlock(terms, n_clock=3, time=0.7, steps=3, order=2, inverse=True),
    ):
        clone = type(block).from_dict(block.to_dict())
        assert clone.build(5).gates == block.build(5).gates
