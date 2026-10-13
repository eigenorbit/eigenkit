"""End-to-end tests for HHL(mode="honest") — the hardware-faithful path.

These check that the honest path (Gershgorin time + register-resolution C +
Trotterised HamiltonianQPEBlock) actually solves Ax=b, that it never touches
the classical spectral shortcuts (eigvalsh / eigenvalue-tuned time), and that
its circuit contains no dense e^{iAt} oracle.
"""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest

from eigenkit.algorithms.hhl import HHL, _gershgorin_spectral_radius


def _abs_cos(u, v) -> float:
    """|cos| — sign-insensitive, unlike the signed `_cos` in test_hhl_honest_trotter."""
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    return abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))


def _solve_honest(A, b, n_l=6, steps=40, order=2):
    return HHL(
        n_eigenvalue_qubits=n_l,
        provider="numpy",
        mode="honest",
        trotter_steps=steps,
        trotter_order=order,
    ).solve(A, b)


# ---- honesty invariants ----


def test_honest_does_not_call_eigvalsh_or_choose_t():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    with (
        mock.patch("numpy.linalg.eigvalsh") as spy_eig,
        mock.patch("eigenkit.algorithms.hhl._choose_t") as spy_t,
    ):
        _solve_honest(A, b, n_l=5, steps=8)
    spy_eig.assert_not_called()
    spy_t.assert_not_called()


def test_reference_does_call_eigvalsh():
    # Guard-rail: the spy actually detects the classical shortcut in reference.
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    real_eigvalsh = np.linalg.eigvalsh
    with mock.patch("numpy.linalg.eigvalsh", side_effect=real_eigvalsh) as spy:
        HHL(n_eigenvalue_qubits=5, provider="numpy").solve(A, b)
    assert spy.called


def test_reference_circuit_uses_dense_oracle():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    solver = _make(mode="reference", n_l=4)
    spec = solver._build_circuit_spec(
        A, np.array([1.0, 0.0]), n_b=1, n_l=4, t=0.5, rotation_angles={}
    )
    dense = [g for g in spec.gates if g.name == "ControlledQubitUnitary"]
    # The evolution oracle is a controlled dense unitary on the whole b register.
    assert dense
    assert any(len(g.qubits) >= 1 + 1 for g in dense)  # control(s) + b-reg target
    assert "RZ" not in {g.name for g in spec.gates}  # no Trotter rotations


# ---- construction / validation ----


def _make(mode="honest", n_l=4, steps=1, order=1):
    return HHL(
        n_eigenvalue_qubits=n_l,
        provider="numpy",
        mode=mode,
        trotter_steps=steps,
        trotter_order=order,
    )


def test_default_mode_is_reference():
    assert HHL().mode == "reference"


def test_honest_requires_two_clock_qubits():
    with pytest.raises(ValueError, match="n_eigenvalue_qubits >= 2"):
        HHL(n_eigenvalue_qubits=1, mode="honest")


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        HHL(mode="quantum-supremacy")


def _safe_explicit_t(A, n_l, fraction=0.5):
    """A user-supplied evolution time comfortably inside the non-aliasing band.

    The honest path caps ``t`` at ``t_max = π·(N−2)/N / ‖A‖_bound``; taking a
    fraction of that keeps every phase strictly inside ``(−π, π)``.
    """
    n_buckets = 2**n_l
    lam_bound = _gershgorin_spectral_radius(np.asarray(A, dtype=complex))
    t_max = math.pi * (n_buckets - 2) / n_buckets / lam_bound
    return fraction * t_max


# ---- explicit evolution time (the t-provided arm of _honest_setup) ----


def test_honest_explicit_time_solves():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    t = _safe_explicit_t(A, n_l=6)
    res = HHL(
        n_eigenvalue_qubits=6,
        provider="numpy",
        mode="honest",
        trotter_steps=40,
        trotter_order=2,
        t=t,
    ).solve(A, b)
    x_true = np.linalg.solve(A, b)
    assert _abs_cos(res.solution, x_true) > 0.99
    assert res.residual < 0.1


def test_honest_explicit_time_too_large_aliases():
    # t chosen so that t·‖A‖_bound exceeds the two-sided phase window -> reject.
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="causes QPE aliasing"):
        HHL(
            n_eigenvalue_qubits=6,
            provider="numpy",
            mode="honest",
            t=10.0,
        ).solve(A, b)


def test_honest_setup_explicit_time_C_is_register_bin():
    # With a user t, C is the first frequency bin 2π/(t·N) — not a spectral value.
    A = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=complex)
    n_l = 5
    n_buckets = 2**n_l
    t = _safe_explicit_t(A, n_l)
    solver = HHL(n_eigenvalue_qubits=n_l, provider="numpy", mode="honest", t=t)
    _, t_out, C, lam_bound, kappa = solver._honest_setup(A)
    assert t_out == pytest.approx(t)
    assert C == pytest.approx(2.0 * math.pi / (t * n_buckets))
    # The reported condition number is the post-hoc diagnostic ‖A‖_bound / C.
    assert kappa == pytest.approx(lam_bound / C)


# ---- wider system shapes and dtypes ----


def test_honest_solves_4x4_diagonal():
    # Two b-register qubits: exercises the multi-qubit clock/state wiring.
    A = np.diag([2.0, 3.0, 1.0, 4.0])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    res = _solve_honest(A, b, n_l=7, steps=1, order=1)  # diagonal -> Trotter exact
    x_true = np.linalg.solve(A, b)
    assert res.solution.shape == (4,)
    assert _abs_cos(res.solution, x_true) > 0.99
    assert res.residual < 0.05


def test_honest_solves_3x3_padded():
    # Non-power-of-two system is zero-padded to 4×4 internally; the extra
    # dimension must not corrupt the recovered 3-vector.
    A = np.diag([3.0, 2.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    res = _solve_honest(A, b, n_l=7, steps=1, order=1)
    x_true = np.linalg.solve(A, b)
    assert res.solution.shape == (3,)
    assert _abs_cos(res.solution, x_true) > 0.99
    assert res.residual < 0.1


def test_honest_solves_complex_hermitian():
    # A = 2I − Y. The Y terms are what the basis change RX(±π/2) rotates, and a sign
    # error there returns the conjugate of the answer: same real part, so only a
    # complex system catches it.
    A = np.array([[2.0, 1j], [-1j, 2.0]])
    b = np.array([1.0, 0.0])
    res = _solve_honest(A, b)
    x_true = np.linalg.solve(A, b)
    assert np.iscomplexobj(res.solution)
    assert _abs_cos(res.solution, x_true) > 0.99
    assert res.residual < 0.1


def test_honest_real_input_solution_has_negligible_imaginary_part():
    # For real A and b the physical answer is real, and the honest read-out must not
    # invent an imaginary component.  `solution` is float64 for real inputs, so the
    # property is asserted on `raw_solution`: the complex post-selected vector before
    # the .real cast, which is where a spurious imaginary part would show up.
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    res = _solve_honest(A, b)
    assert res.solution.dtype == np.float64
    raw = np.asarray(res.raw_solution)
    pivot = raw[np.argmax(np.abs(raw))]
    phased = raw * np.exp(-1j * np.angle(pivot))
    assert np.abs(phased.imag).max() < 1e-9


# ---- negative / robustness paths through the honest branch ----


def test_honest_rejects_singular_A():
    A = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-deficient
    b = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="singular"):
        _solve_honest(A, b)


def test_honest_near_zero_probability_returns_zeroed_result():
    # When the ancilla almost never lands in |1⟩, solve() must warn and return a
    # well-formed zeroed result instead of dividing by ~0.  We force the
    # degenerate post-selection to isolate this contract deterministically.
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    solver = HHL(n_eigenvalue_qubits=4, provider="numpy", mode="honest", trotter_steps=3)
    with mock.patch(
        "eigenkit.algorithms.hhl._postselect",
        return_value=(np.zeros(2, dtype=complex), 0.0, 0.0),
    ):
        with pytest.warns(RuntimeWarning, match="success probability is near zero"):
            res = solver.solve(A, b)
    assert res.success_probability == 0.0
    assert res.euclidean_norm == 0.0
    assert res.residual == 1.0
    assert res.solution.shape == (2,)
    # Real inputs -> real zero vector on the degenerate path.
    assert not np.iscomplexobj(res.solution)


def test_honest_requires_statevector_provider():
    # The honest read-out reads the full statevector; a backend whose compiled
    # circuit cannot supply one must fail loudly, not silently misbehave.
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])

    class _NoStatevectorCompiled:
        statevector = None  # not callable

    class _FakeProvider:
        name = "no-statevector"

    class _FakeCircuit:
        def __init__(self, spec, provider=None):
            self.compiled = _NoStatevectorCompiled()
            self.provider = _FakeProvider()

    solver = HHL(n_eigenvalue_qubits=4, provider="numpy", mode="honest", trotter_steps=2)
    with mock.patch("eigenkit.algorithms.hhl.Circuit", _FakeCircuit):
        with pytest.raises(NotImplementedError, match="statevector"):
            solver.solve(A, b)


def test_honest_full_circuit_including_inversion_has_no_dense_oracle():
    # The inversion rotations are included here, so the assertion covers the whole
    # assembled circuit and not just the QPE stage.
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    solver = _make(mode="honest", n_l=4, steps=3)
    angles = {1: 0.5, 3: -0.5, 5: 0.25}  # a few non-trivial inversion rotations
    spec = solver._build_circuit_spec(
        A / 4.0, np.array([1.0, 0.0]), n_b=1, n_l=4, t=0.5, rotation_angles=angles
    )
    names = {g.name for g in spec.gates}
    # Nothing dense anywhere: the evolution is Trotterised (RZ + CNOT + H) and the
    # inversion is a uniformly controlled rotation (RY + CNOT), so unlike the
    # reference path the assembled honest circuit carries no unitary-matrix gate of
    # any kind.
    assert "QubitUnitary" not in names
    assert "ControlledQubitUnitary" not in names
    assert {"RZ", "RY", "H", "CNOT"} <= names
