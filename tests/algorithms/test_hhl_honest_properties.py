"""One test per known failure mode of the honest HHL path.

Each test names a way the honest path can go wrong and asserts the property that rules
it out, so a failure reads as that specific failure mode, not as "some residual moved".

Deliberately end-to-end: these go through the public :meth:`HHL.solve`, because several
of the failure modes live in the *seams* between correct components (a bound computed on
``A`` while the circuit runs ``A_eff``, a probability measured on one event and reported
for another, a validation call that has no place on the honest path).
"""

from __future__ import annotations

import collections
import pathlib
import warnings

import numpy as np
import pytest

from eigenkit import HHL
from eigenkit.algorithms import hhl as H
from eigenkit.algorithms._hamiltonian_sim import (
    pauli_decompose,
    pauli_string_matrix,
)


P = pauli_string_matrix

TRI4 = np.array([[2.0, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]])
# Κ = 3, five Pauli terms, local — the honest path's best case.
A_HEISENBERG = (P("XX") + P("YY") + P("ZZ") + P("ZI") + P("IZ")).real
# Gershgorin-tight on λ_min = −4, so the extremal eigenvalue starts at
# the centre of the last safe bucket and a drift of order C captures it at Nyquist.
A_NYQUIST = -np.array([[0.0, 1, 1, 2], [1, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 0]])


def _honest(n_l: int = 6, **kwargs) -> HHL:
    return HHL(n_eigenvalue_qubits=n_l, mode="honest", provider="numpy", **kwargs)


def _cos(u, v) -> float:
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    if np.max(np.abs(u.imag)) < 1e-12 and np.max(np.abs(v.imag)) < 1e-12:
        u, v = u.real, v.real
    return float(abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)))


def _signed_cos(u, v) -> float:
    u = np.asarray(u).real.ravel().astype(float)
    v = np.asarray(v).real.ravel().astype(float)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def _solve_capturing(A, b, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _honest(**kwargs).solve(A, b)
    return result, [str(w.message) for w in caught]


def _random_hermitian(rng, n: int, kappa: float, indefinite: bool) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigenvalues = np.geomspace(1.0, 1.0 / kappa, n)
    if indefinite:
        eigenvalues = eigenvalues * rng.choice([-1.0, 1.0], size=n)
        if np.all(eigenvalues > 0):
            eigenvalues[0] *= -1
    return Q @ np.diag(eigenvalues) @ Q.T


# ---------------------------------------------------------------------------
# The eigenvector error, not the eigenvalue error, is what breaks the answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_l", [3, 4, 5, 6, 7])
def test_the_heisenberg_witness_is_solved_at_every_register_size(n_l):
    """Aligned at every ``n_l`` from 3 to 7, with zero warnings.

    ``trotter_steps=1`` with ``t = π(N−2)/(NΛ)`` pairs the largest admissible ``t`` with
    the smallest step count, i.e. the largest ``τ`` the code can produce.  Worse,
    ``t`` *grows* with ``N``, so at a fixed step count adding clock qubits makes the
    answer worse — the opposite of the user's mental model.
    """
    expected = np.linalg.solve(A_HEISENBERG, np.ones(4))
    result, messages = _solve_capturing(A_HEISENBERG, np.ones(4), n_l=n_l)
    assert _signed_cos(result.solution, expected) > 0.999, messages
    assert result.residual < 0.03
    assert not messages


def test_quality_does_not_degrade_as_the_clock_register_grows():
    """The monotonicity the docs promise, on a witness a fixed step count violates."""
    expected = np.linalg.solve(A_HEISENBERG, np.ones(4))
    cosines = [
        _signed_cos(_honest(n_l=n_l).solve(A_HEISENBERG, np.ones(4)).solution, expected)
        for n_l in (3, 4, 5, 6)
    ]
    assert min(cosines) > 0.999
    # No systematic decline: the last is at least as good as the first.
    assert cosines[-1] >= cosines[0] - 1e-3


@pytest.mark.parametrize("h", [0.0, 0.25, 0.75, 1.0, 2.0, 4.0])
def test_the_field_sweep_never_returns_an_antiparallel_answer(h):
    """A mis-signed ``1/λ`` returns an answer pointing the wrong way, with no warning
    at all; the field sweep has to stay aligned at every strength."""
    A = (P("XX") + P("YY") + P("ZZ") + h * (P("ZI") + P("IZ"))).real
    if abs(np.linalg.det(A)) < 1e-9:
        pytest.skip("singular at this field strength")
    expected = np.linalg.solve(A, np.ones(4))
    result, messages = _solve_capturing(A, np.ones(4))
    assert _signed_cos(result.solution, expected) > 0.99, (h, messages)


def test_the_answer_survives_an_arbitrary_permutation_of_the_pauli_terms():
    """The reproducibility hazard of a coarse Trotter grid.

    The first-order Trotter error depends on the *application order* of the terms, and
    that order is whatever ``itertools.product("IXYZ", …)`` yields — physically
    arbitrary.  On a coarse grid, simply reordering the list can turn a poor answer for
    the witness into an exact one, which leaves the answer at the mercy of an
    implementation detail.  With the step count derived from a bound, the answer has to
    be correct for *every* ordering, not just the lucky ones.
    """
    expected = np.linalg.solve(A_HEISENBERG, np.ones(4))
    original = pauli_decompose
    orderings = {
        "lexicographic": lambda terms: terms,
        "reversed": lambda terms: terms[::-1],
        "commuting families first": lambda terms: sorted(
            terms, key=lambda term: any(ch in "XY" for ch in term[1])
        ),
        "shuffled": lambda terms: [terms[i] for i in (2, 0, 4, 1, 3)],
    }
    try:
        for name, reorder in orderings.items():

            def patched(M, _reorder=reorder, **kw):
                return _reorder(original(M, **kw))

            H.pauli_decompose = patched
            result, messages = _solve_capturing(A_HEISENBERG, np.ones(4), n_l=5)
            assert _signed_cos(result.solution, expected) > 0.99, (name, messages)
            assert result.residual < 0.05, name
    finally:
        H.pauli_decompose = original


def test_the_step_count_rises_with_the_clock_register():
    """``τ = t/steps`` and ``t`` grows with ``N``, so ``steps`` has to grow too.

    Holding ``steps`` fixed would make more clock qubits mean less accuracy.
    """
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    counts = []
    with warnings.catch_warnings():
        # n_l = 9 wants more steps than the gate budget allows and says so; that
        # clamp has its own test, and it does not change the trend measured here.
        warnings.simplefilter("ignore", RuntimeWarning)
        for n_l in (3, 5, 7, 9):
            t, C, _ = H._honest_qpe_params(A_HEISENBERG.astype(complex), n_l)
            counts.append(_honest(n_l=n_l)._resolve_trotter_steps(terms, t, C, n_l))
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


# ---------------------------------------------------------------------------
# The guard band is computed on A while QPE sees A_eff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_l", [4, 5, 6])
def test_the_nyquist_witness_is_solved_at_defaults(n_l):
    """Solved at the defaults, where an unbounded drift would fail silently.

    ``Λ`` is a Gershgorin bound on ``A``, but QPE estimates ``A_eff``.  With half a
    bucket of guard band — which *halves* with every clock qubit — a Trotter drift that
    does not shrink with ``N`` eventually pushes ``λ_min`` onto the sign-ambiguous
    Nyquist bucket, and then past the logarithm's branch cut.
    """
    expected = np.linalg.solve(A_NYQUIST, np.ones(4))
    result, messages = _solve_capturing(A_NYQUIST, np.ones(4), n_l=n_l)
    assert _signed_cos(result.solution, expected) > 0.99, messages
    assert result.residual < 0.05


@pytest.mark.parametrize("n_l", [4, 5, 6, 7, 8])
def test_no_bucket_capture_or_wraparound_at_the_selected_step_count(n_l):
    """Checked on the buckets themselves, without running the circuit.

    At ``trotter_order=1, trotter_steps=1`` and ``n_l = 7`` the estimated spectrum of
    this matrix loses its extremal negative eigenvalue entirely: the phase crosses ``−π``
    and re-enters from the positive side (measured buckets ``[-17, 13, 37, 64]`` where
    the exact ones are ``[-63, -25, 16, 40]``).  With the drift held under ``C/4`` that
    is unreachable.
    """
    import scipy.linalg as sla

    A = A_NYQUIST.astype(complex)
    terms = pauli_decompose(A)
    N = 2**n_l
    t, C, _ = H._honest_qpe_params(A, n_l)
    with warnings.catch_warnings():
        # At n_l = 8 the selector hits the gate budget and warns; the clamped grid is
        # what the circuit would use, so that is what gets checked for capture.
        warnings.simplefilter("ignore", RuntimeWarning)
        steps = _honest(n_l=n_l)._resolve_trotter_steps(terms, t, C, n_l)

    W = np.eye(4, dtype=complex)
    tau = t / steps
    sequence = [(c, s, tau / 2) for c, s in terms] + [
        (c, s, tau / 2) for c, s in reversed(terms)
    ]
    for c, s, dt in sequence:
        W = sla.expm(1j * complex(c).real * P(s) * dt) @ W
    W = np.linalg.matrix_power(W, steps)

    exact = np.sort(np.round(np.linalg.eigvalsh(A) * t * N / (2 * np.pi)).astype(int))
    trotterised = np.sort(
        np.round(np.angle(np.linalg.eigvals(W)) * N / (2 * np.pi)).astype(int)
    )
    assert np.max(np.abs(trotterised)) < N // 2, trotterised  # no Nyquist capture
    np.testing.assert_array_equal(exact, trotterised)  # and no wraparound either


def test_a_degenerate_gershgorin_tight_bottom_is_still_safe():
    """The genuinely dangerous class.

    For a real symmetric ``A`` with a *simple* Gershgorin-tight extremum the
    first-order shift vanishes identically.  The protection fails on a **degenerate**
    tight bottom — typical of a reducible, block-structured ``A``, e.g. several
    independent systems solved jointly — where the pair splits and drifts outward at
    ``Θ(τ)``.  Built exactly that way, its capture map has a diagonal that says
    "every added clock qubit needs a doubling of ``steps``".
    """
    # One isolated coordinate at −Λ, plus a 3×3 non-negative block with constant row
    # sums (so its top eigenvalue is exactly that sum), the whole thing negated.
    block = np.array([[0.0, 1.0, 3.0], [1.0, 2.0, 1.0], [3.0, 1.0, 0.0]])
    row_sum = float(block.sum(axis=1)[0])
    A = np.zeros((4, 4))
    A[0, 0] = row_sum
    A[1:, 1:] = block
    A = -A
    spectrum = np.linalg.eigvalsh(A)
    # Premise of the witness: Gershgorin is tight, and the bottom is degenerate.
    assert abs(spectrum[0]) == pytest.approx(H._gershgorin_spectral_radius(A), rel=1e-9)
    assert spectrum[1] == pytest.approx(spectrum[0], abs=1e-9)

    expected = np.linalg.solve(A, np.ones(4))
    for n_l in (4, 5, 6):
        result, messages = _solve_capturing(A, np.ones(4), n_l=n_l)
        assert _signed_cos(result.solution, expected) > 0.95, (n_l, messages)
        assert result.residual < 0.05, (n_l, messages)


def test_an_explicit_coarse_grid_warns_and_names_a_sufficient_count():
    """Opting out of the selector is allowed, but not silently."""
    result, messages = _solve_capturing(A_NYQUIST, np.ones(4), n_l=6, trotter_steps=1)
    drift_warnings = [m for m in messages if "Trotter drift bound" in m]
    assert len(drift_warnings) == 1
    assert "Nyquist" in drift_warnings[0]
    suggested = int(drift_warnings[0].split("trotter_steps=")[-1].split()[0].rstrip("."))
    better, quiet = _solve_capturing(
        A_NYQUIST, np.ones(4), n_l=6, trotter_steps=suggested
    )
    assert not [m for m in quiet if "Trotter drift bound" in m]
    assert better.residual < result.residual


# ---------------------------------------------------------------------------
# What the read-out actually reports
# ---------------------------------------------------------------------------


def test_the_raw_quantum_vector_is_exposed_alongside_the_fitted_one():
    """``solution`` is finished by two classical steps hardware cannot perform.

    The direction is ``Re(raw)``; the magnitude *and sign* come from a least-squares
    projection using the full classical ``A`` and ``b``, which on a poor raw vector can
    turn a residual worse than returning zero into a small one.  So the fit stays, and
    the unfitted vector is exposed beside it and documented as such.
    """
    result = _honest().solve(TRI4, np.array([1.0, 0.0, 0.0, 0.0]))
    assert result.raw_solution is not None
    assert result.raw_solution.dtype == np.complex128
    assert np.linalg.norm(result.raw_solution) == pytest.approx(1.0, rel=1e-9)
    # The fitted solution is a real multiple of the raw one's real part.
    ratios = result.solution / result.raw_solution.real
    assert np.allclose(ratios, ratios[0], rtol=1e-9)
    assert "least-squares" in H.HHLResult.__doc__
    assert "raw_solution" in H.HHLResult.__doc__


def test_euclidean_norm_is_the_norm_of_what_is_returned():
    """``euclidean_norm`` is defined as ``‖solution‖``, and the docstring says which
    quantity that is.  It is not ``√P·‖b‖/C``, and a field documented as one quantity
    while computing another misreports the magnitude.
    """
    for A, b in (
        (np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0])),
        (np.array([[2.0, 1.0], [1.0, 0.0]]), np.array([1.0, 1.0])),
        (TRI4, np.array([1.0, 0.0, 0.0, 0.0])),
    ):
        result = _honest().solve(A, b)
        assert result.euclidean_norm == pytest.approx(
            float(np.linalg.norm(result.solution)), rel=1e-12
        )
    assert "of the returned solution" in H.HHLResult.__doc__


def test_the_physical_post_selection_rate_is_reported_separately():
    """``success_probability`` measures ancilla=|1⟩ over *all* clock states,
    while the solution is read from the joint ``(ancilla=1, clock=0)`` slice — an
    overstatement of up to ×21 on a degenerate spectrum.  So both are reported.
    """
    for A, b, n_l in (
        (np.diag([1.0, 0.3]), np.array([1.0, 1.0]), 5),
        (A_HEISENBERG, np.ones(4), 5),
        (TRI4, np.array([1.0, 0.0, 0.0, 0.0]), 5),
    ):
        result = _honest(n_l=n_l).solve(A, b)
        assert 0.0 < result.readout_probability <= result.success_probability
    assert "readout_probability" in H.HHLResult.__doc__
    assert "use it to estimate the number of shots" in H.HHLResult.__doc__


@pytest.mark.parametrize("order,expected_slopes", [(1, (-2.0, -1.0)), (2, (-2.0, -2.0))])
def test_convergence_rate_for_real_and_complex_inputs(order, expected_slopes):
    """Measures the real/complex asymmetry the ``.real`` cast creates.

    For real ``A, b`` the leading eigenvector error enters the raw vector as a purely
    imaginary component, so dropping the imaginary part cancels it and the error falls
    as ``steps⁻²``.  A complex input keeps that channel — **at order 1**, where it is
    the leading term.  At order 2 the first-order commutator cancels structurally, so
    both converge as ``steps⁻²`` and the asymmetry disappears.

    The class docstring's "Real and complex inputs" paragraph is checked against these
    same numbers by :func:`test_the_documented_convergence_rates_name_the_right_order`,
    which requires the ``steps⁻¹`` case to be attributed to ``trotter_order=1`` and not
    to ``trotter_order=2``, the one order where it does not happen.
    """
    steps_grid = [4, 8, 16]
    slopes = []
    for b in (
        np.ones(4),
        np.ones(4) + 1j * np.array([0.3, -0.2, 0.5, 0.1]),
    ):
        expected = np.linalg.solve(A_HEISENBERG, b)
        errors = []
        for steps in steps_grid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                got = np.asarray(
                    _honest(n_l=3, trotter_steps=steps, trotter_order=order)
                    .solve(A_HEISENBERG, b)
                    .solution
                )
            u = got / np.linalg.norm(got)
            v = expected / np.linalg.norm(expected)
            phase = np.vdot(v, u)
            errors.append(float(np.linalg.norm(u - v * phase / abs(phase))))
        slopes.append(float(np.polyfit(np.log(steps_grid), np.log(errors), 1)[0]))
    assert slopes[0] == pytest.approx(expected_slopes[0], abs=0.25), slopes
    assert slopes[1] == pytest.approx(expected_slopes[1], abs=0.25), slopes


def _passage(text: str, opening: str, closing: str) -> str:
    """The passage between two markers, whitespace collapsed to single spaces."""
    collapsed = " ".join(text.split())
    assert opening in collapsed, f"passage not found: {opening}"
    return collapsed.split(opening, 1)[1].split(closing, 1)[0]


def _hhl_docstring_passage() -> tuple[str, str, str]:
    return (
        _passage(HHL.__doc__, "**Real and complex inputs.**", "Args:"),
        "`steps⁻¹`",
        "both converge as `steps⁻²`",
    )


def _hhl_chapter_passage() -> tuple[str, str, str]:
    docs = pathlib.Path(__file__).resolve().parents[2] / "docs"
    if not docs.is_dir():
        pytest.skip("docs/ is not in the sdist")
    chapter = docs / "algorithms/hhl.md"
    return (
        _passage(
            chapter.read_text(encoding="utf-8"),
            "**Real inputs converge one order faster",
            "- **",
        ),
        "`steps⁻¹`",
        "both converge as `steps⁻²`",
    )


@pytest.mark.parametrize(
    "load",
    [
        pytest.param(_hhl_docstring_passage, id="hhl-docstring"),
        pytest.param(_hhl_chapter_passage, id="hhl-chapter"),
    ],
)
def test_the_documented_convergence_rates_name_the_right_order(load):
    """Every place that states the rates has to attribute ``steps⁻¹`` to the order it
    happens at.

    Measured by the test above: ``(−2, −1)`` real/complex at ``trotter_order=1`` and
    ``(−2, −2)`` at ``trotter_order=2``.  Naming order 2 for the ``steps⁻¹`` case, i.e.
    exactly the one order where real and complex agree, would send a reader to order 1
    to avoid a penalty that only order 1 has.  Both the class docstring and the chapter
    are parametrised here, because a passage no test reads can carry that error
    unnoticed.
    """
    passage, slow, resolved = load()

    def position(token: str) -> int:
        assert token in passage, f"{token} is missing from:\n{passage}"
        return passage.index(token)

    # The steps⁻¹ claim must sit on the order-1 side of the passage, and order 2 must be
    # described as removing the asymmetry rather than causing it.
    order_one, order_two = position("`trotter_order=1`"), position("`trotter_order=2`")
    assert order_one < position(slow) < order_two, passage
    assert resolved in passage[order_two:], passage


# ---------------------------------------------------------------------------
# Order 1 turns a real system's solution complex
# ---------------------------------------------------------------------------


def test_order_two_is_the_default():
    solver = HHL(n_eigenvalue_qubits=4, mode="honest")
    assert solver.trotter_order == 2
    assert solver.trotter_steps == "auto"


def test_a_real_system_returns_a_real_state_at_the_defaults():
    """At order 1 on a coarse grid the post-selected state can be largely imaginary
    after the global phase is removed, for a real ``A`` and a real ``b`` whose exact
    answer is real.  The defaults have to return a real state."""
    result = _honest().solve(np.array([[2.0, 1.0], [1.0, 0.0]]), np.array([1.0, 1.0]))
    raw = result.raw_solution
    pivot = int(np.argmax(np.abs(raw)))
    phased = raw * np.exp(-1j * np.angle(raw[pivot]))
    imaginary_fraction = np.linalg.norm(phased.imag) / np.linalg.norm(phased)
    assert imaginary_fraction < 1e-3


def test_the_imaginary_channel_is_detected_when_it_is_switched_back_on():
    """The non-spectral diagnostic: the one check that can see an isospectral
    eigenvector rotation, since it needs no spectrum at all."""
    _, messages = _solve_capturing(
        np.array([[2.0, 1.0], [1.0, 0.0]]),
        np.array([1.0, 1.0]),
        n_l=6,
        trotter_steps=1,
        trotter_order=1,
    )
    detector = [m for m in messages if "imaginary after the global phase" in m]
    assert len(detector) == 1
    assert "Trotter error in the eigenvectors" in detector[0]
    assert "trotter_order=2" in detector[0]


# ---------------------------------------------------------------------------
# The answer must not depend on the physical units of A
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_honest_mode_is_scale_invariant_end_to_end(scale):
    """An absolute ``atol`` would cost ``1.05e-10 · (I + 0.9X)`` its X term and the
    answer its direction, and below ``5e-11`` would raise ``ValueError``.  Reference mode
    normalises by ``λ_max`` and is invariant by construction; the honest path never
    computes ``λ_max``, so every threshold on it has to be relative instead.
    """
    A = np.array([[1.0, 0.9], [0.9, 1.0]])
    b = np.array([1.0, 0.0])
    baseline = _honest(n_l=5).solve(A, b)
    scaled = _honest(n_l=5).solve(scale * A, b)
    assert len(pauli_decompose((scale * A).astype(complex))) == 2
    # Same direction, and the magnitude tracks 1/scale exactly.
    assert _signed_cos(scaled.solution, baseline.solution) > 1 - 1e-9
    np.testing.assert_allclose(scaled.solution * scale, baseline.solution, rtol=1e-9)
    assert scaled.residual == pytest.approx(baseline.residual, rel=1e-9)


def test_a_scaled_system_builds_the_identical_circuit():
    """Invariance of the *answer* could come from luck; invariance of the circuit
    is the actual property."""
    specs = []
    for scale in (1.0, 1e6):
        solver = _honest(n_l=4)
        Ac = np.asarray(scale * TRI4, dtype=complex)
        A_circuit, t, C, lam, _ = solver._honest_setup(Ac)
        terms = pauli_decompose(A_circuit)
        steps = solver._resolve_trotter_steps(terms, t, C, 4)
        angles = H._compute_rotation_angles(16, t, C, lam, solver.epsilon)
        specs.append(
            (
                steps,
                [s for _, s in terms],
                tuple(round(angles[j], 12) for j in sorted(angles)),
            )
        )
    assert specs[0] == specs[1]


# ---------------------------------------------------------------------------
# The inversion stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_l", [2, 4, 6])
def test_the_inversion_holds_no_multi_controlled_primitive(n_l):
    """Not ``2^{n_l} − 1`` ``ControlledQubitUnitary`` gates with ``n_l`` controls each.

    The inversion is the Möttönen uniformly controlled rotation: ``2^{n_l}`` RY +
    ``2^{n_l}`` CNOT, no ancillas, no dense gate — roughly ``n_l`` times cheaper than the
    multi-controlled table and, more to the point, elementary.  The classical
    *description* is still ``Θ(2^{n_l})``; that is a uniformity cost, not a
    complexity-class one, and the docstring says so.
    """
    solver = _honest(n_l=n_l)
    Ac = np.asarray(TRI4, dtype=complex)
    A_circuit, t, C, lam, _ = solver._honest_setup(Ac)
    terms = pauli_decompose(A_circuit)
    steps = solver._resolve_trotter_steps(terms, t, C, n_l)
    angles = H._compute_rotation_angles(2**n_l, t, C, lam, solver.epsilon)
    spec = solver._build_circuit_spec(
        A_circuit,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        2,
        n_l,
        t,
        angles,
        pauli_terms=terms,
        trotter_steps=steps,
    )
    names = collections.Counter(g.name for g in spec.gates)
    assert names["ControlledQubitUnitary"] == 0
    assert names["QubitUnitary"] == 0
    assert names["RY"] == 2**n_l  # one per bucket, and nothing else emits RY
    assert set(names) <= H._HHL_HONEST_REQUIRED_GATES | {"CNOT"}
    assert "its description is not compact" in " ".join(HHL.__doc__.split())


def test_the_declared_gate_set_is_what_honest_mode_emits():
    """A declared set that is too small breaks a strict provider; too large and the
    solver refuses backends that would have worked."""
    solver = _honest(n_l=4)
    Ac = np.asarray(A_HEISENBERG, dtype=complex)
    A_circuit, t, C, lam, _ = solver._honest_setup(Ac)
    terms = pauli_decompose(A_circuit)
    steps = solver._resolve_trotter_steps(terms, t, C, 4)
    angles = H._compute_rotation_angles(16, t, C, lam, solver.epsilon)
    spec = solver._build_circuit_spec(
        A_circuit,
        np.ones(4, dtype=complex) / 2.0,
        2,
        4,
        t,
        angles,
        pauli_terms=terms,
        trotter_steps=steps,
    )
    emitted = {g.name for g in spec.gates}
    assert emitted <= H._HHL_HONEST_REQUIRED_GATES
    assert "ControlledQubitUnitary" not in H._HHL_HONEST_REQUIRED_GATES


def test_the_epsilon_cull_is_visible_in_the_gate_count():
    """The ``ε`` filter zeroes symmetric bucket pairs, and with the
    default ``ε = 0.01`` it starts biting at ``n_l = 8`` — for *every* matrix, at
    exactly the buckets carrying the largest ``1/λ`` weights."""
    n_l = 9
    n_buckets = 2**n_l
    t, C, lam = H._honest_qpe_params(TRI4.astype(complex), n_l)
    culled = [
        j
        for j, theta in H._compute_rotation_angles(n_buckets, t, C, lam, 1e-2).items()
        if theta == 0.0
    ]
    assert culled  # the cull is real at this register size
    # Symmetric in the bucket sign, and only the smallest |j_signed| are hit.
    signed = sorted({abs(j if j <= n_buckets // 2 else j - n_buckets) for j in culled})
    assert signed == list(range(1, len(signed) + 1))
    # And it is vacuous at small registers, as documented (N > 2/epsilon + 2).
    t4, c4, lam4 = H._honest_qpe_params(TRI4.astype(complex), 4)
    assert all(
        theta != 0.0
        for theta in H._compute_rotation_angles(16, t4, c4, lam4, 1e-2).values()
    )


# ---------------------------------------------------------------------------
# Honest mode runs no classical spectral routine
# ---------------------------------------------------------------------------


def test_honest_mode_calls_no_classical_spectral_routine_but_the_rank_check():
    """Exactly one SVD, in input validation, and nothing spectral anywhere else.

    ``matrix_rank`` resolves the linear-algebra names outside ``numpy.linalg._linalg``,
    so patching that namespace alone misses its ``svd``; this test patches **both**.
    """
    import numpy.linalg as numpy_linalg
    import numpy.linalg._linalg as numpy_linalg_impl

    watched = ("svd", "eigvalsh", "eigh", "eig", "matrix_power", "matrix_rank", "inv")
    calls: collections.Counter = collections.Counter()
    saved = []
    for module in (numpy_linalg, numpy_linalg_impl):
        for name in watched:
            original = getattr(module, name, None)
            if original is None:
                continue
            saved.append((module, name, original))

            def wrap(label, func):
                def wrapper(*args, **kwargs):
                    calls[label] += 1
                    return func(*args, **kwargs)

                return wrapper

            setattr(module, name, wrap(name, original))
    try:
        calls.clear()
        _honest(n_l=4).solve(TRI4, np.array([1.0, 0.0, 0.0, 0.0]))
        honest_calls = dict(calls)

        calls.clear()
        HHL(n_eigenvalue_qubits=4, provider="numpy").solve(
            TRI4, np.array([1.0, 0.0, 0.0, 0.0])
        )
        reference_calls = dict(calls)
    finally:
        for module, name, original in saved:
            setattr(module, name, original)

    assert honest_calls == {"svd": 1}, honest_calls
    # Reference mode is where the classical shortcuts live, by design.
    assert reference_calls.get("eigvalsh", 0) >= 1
    assert reference_calls.get("matrix_power", 0) >= 1
    docs = " ".join(HHL.solve.__doc__.split())
    assert "(SVD) to reject a singular" in docs
    assert "changing it would leave the circuit exactly the same" in docs


def test_no_spectral_quantity_from_validation_reaches_the_circuit():
    """The defining property of honest mode, stated as a test: the circuit is
    byte-identical when the validation SVD is unavailable."""
    solver = _honest(n_l=4)
    Ac = np.asarray(TRI4, dtype=complex)
    A_circuit, t, C, lam, _ = solver._honest_setup(Ac)
    terms = pauli_decompose(A_circuit)
    steps = solver._resolve_trotter_steps(terms, t, C, 4)
    angles = H._compute_rotation_angles(16, t, C, lam, solver.epsilon)
    b_unit = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    build = lambda: solver._build_circuit_spec(  # noqa: E731
        A_circuit, b_unit, 2, 4, t, angles, pauli_terms=terms, trotter_steps=steps
    )
    baseline = build()

    original_svd = np.linalg.svd

    def forbidden(*args, **kwargs):
        raise AssertionError("the circuit build reached np.linalg.svd")

    np.linalg.svd = forbidden
    try:
        assert build().gates == baseline.gates
    finally:
        np.linalg.svd = original_svd


# ---------------------------------------------------------------------------
# A 1e-18 asymmetry must not square the condition number
# ---------------------------------------------------------------------------


def test_a_floating_point_asymmetry_does_not_trigger_the_normal_equations():
    """With ``atol=0`` the Hermitian test would be purely relative, so a structural zero
    facing a 1e-18 partner would read as "not Hermitian" → normal equations → ``κ²`` →
    a lost answer.  A realistic trigger is assembling ``A`` as ``BᵀDB`` or element by
    element.
    """
    A = np.diag([1.0, 0.02])
    A[1, 0] = 1e-18
    expected = np.linalg.solve(A, np.ones(2))
    asymmetric, messages = _solve_capturing(A, np.ones(2), n_l=7)
    exact = _honest(n_l=7).solve(np.diag([1.0, 0.02]), np.ones(2))

    assert not [m for m in messages if "not Hermitian" in m]
    assert _signed_cos(asymmetric.solution, expected) > 0.999
    assert asymmetric.residual == pytest.approx(exact.residual, rel=1e-6)
    assert asymmetric.success_probability == pytest.approx(
        exact.success_probability, rel=1e-6
    )


# ---------------------------------------------------------------------------
# The dtype contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["honest", "reference"])
def test_real_inputs_return_float64_and_complex_ones_complex128(mode):
    """The documented dtype contract, in both modes: ``float64`` for a real system,
    ``complex128`` as soon as ``A`` or ``b`` is complex."""
    solver = HHL(n_eigenvalue_qubits=5, mode=mode, provider="numpy")
    real = solver.solve(np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0]))
    assert real.solution.dtype == np.float64

    complex_A = solver.solve(np.array([[2.0, 1j], [-1j, 2.0]]), np.array([1.0, 0.0]))
    assert complex_A.solution.dtype == np.complex128

    complex_b = solver.solve(
        np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0 + 1j, 0.0])
    )
    assert complex_b.solution.dtype == np.complex128


def test_the_near_zero_probability_branch_keeps_the_same_dtype():
    """The fallback and the normal path of one function must agree on the dtype,
    otherwise the conditioning of ``A`` would decide it."""
    A = np.diag([1.0, 1e-9])  # κ = 1e9: nothing survives the register floor
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _honest(n_l=2).solve(A, np.array([0.0, 1.0]))
    assert any("near zero" in str(w.message) for w in caught)
    assert result.solution.dtype == np.float64
    assert result.success_probability == 0.0
    assert result.readout_probability == 0.0
    assert result.raw_solution is not None and result.raw_solution.dtype == np.complex128


# ---------------------------------------------------------------------------
# A randomised sweep for systematic silent failures
# ---------------------------------------------------------------------------


def test_no_silent_failures_on_a_randomised_sweep():
    """Indefinite and ill-conditioned systems are where an unbounded Trotter error
    returns answers pointing the opposite way behind a healthy success probability and
    no warning.  "Silently bad" is the criterion here: ``cos < 0.9``, zero warnings,
    ``P > 1e-6``.
    """
    rng = np.random.default_rng(20260807)
    rows = []
    for dim in (2, 4):
        for kappa in (2.0, 10.0):
            for indefinite in (False, True):
                for _ in range(2):
                    A = _random_hermitian(rng, dim, kappa, indefinite)
                    b = rng.standard_normal(dim)
                    expected = np.linalg.solve(A, b)
                    result, messages = _solve_capturing(A, b, n_l=6)
                    rows.append(
                        (
                            _signed_cos(result.solution, expected),
                            result.residual,
                            result.success_probability,
                            len(messages),
                            dict(dim=dim, kappa=kappa, indefinite=indefinite),
                        )
                    )
    silent = [r for r in rows if r[0] < 0.9 and r[3] == 0 and r[2] > 1e-6]
    assert not silent, silent
    cosines = np.array([r[0] for r in rows])
    assert cosines.min() > 0.9, rows[int(np.argmin(cosines))]
    assert float(np.median(cosines)) > 0.999


def test_a_bigger_clock_register_improves_the_answer():
    """More clock qubits must improve the answer, as the docs promise, on all but at
    most one system of the sweep.  What decides it is the Trotter grid, not the ``ε``
    filter: a step count that grows with the register keeps the improvement monotone.
    """
    rng = np.random.default_rng(31337)
    improved_cos = improved_residual = total = 0
    for kappa in (10.0, 50.0):
        for indefinite in (False, True):
            for _ in range(3):
                A = _random_hermitian(rng, 2, kappa, indefinite)
                b = rng.standard_normal(2)
                expected = np.linalg.solve(A, b)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    coarse = _honest(n_l=4).solve(A, b)
                    fine = _honest(n_l=6).solve(A, b)
                total += 1
                improved_cos += (
                    _signed_cos(fine.solution, expected)
                    >= _signed_cos(coarse.solution, expected) - 1e-9
                )
                improved_residual += fine.residual <= coarse.residual + 1e-9
    assert improved_cos >= total - 1, (improved_cos, total)
    assert improved_residual >= total - 1, (improved_residual, total)


# ---------------------------------------------------------------------------
# Smaller contracts
# ---------------------------------------------------------------------------


def test_the_decomposition_runs_once_per_solve():
    """The decomposition is the dominant classical cost — 2.8 s at dimension 128
    against 0.0013 s for the validation SVD — so it must run exactly once."""
    original = pauli_decompose
    count = 0

    def counting(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    H.pauli_decompose = counting
    try:
        _honest(n_l=4).solve(TRI4, np.array([1.0, 0.0, 0.0, 0.0]))
    finally:
        H.pauli_decompose = original
    assert count == 1


def test_gate_remapping_preserves_every_field():
    """Shifting qubit indices past the ancilla must carry every other field along,
    ``adjoint`` and ``params`` included, in ``_qpe_gates`` and in the
    ``AmplitudeEncoding`` loop alike.  A dropped field stays harmless only for as long
    as nothing in those fragments uses it.
    """
    from eigenkit.core.blocks import AmplitudeEncoding, HamiltonianQPEBlock

    solver = _honest(n_l=3)
    Ac = np.asarray(TRI4, dtype=complex)
    terms = pauli_decompose(Ac)
    t, C, _ = H._honest_qpe_params(Ac, 3)
    steps = solver._resolve_trotter_steps(terms, t, C, 3)

    for inverse in (False, True):
        source = HamiltonianQPEBlock(
            terms, n_clock=3, time=t, steps=steps, order=2, inverse=inverse
        ).build(5)
        mapped = solver._qpe_gates(
            Ac, 2, 3, t, inverse=inverse, pauli_terms=terms, trotter_steps=steps
        )
        assert len(mapped) == len(source.gates)
        for original_gate, mapped_gate in zip(source.gates, mapped):
            assert mapped_gate.name == original_gate.name
            assert mapped_gate.params == original_gate.params
            assert mapped_gate.data == original_gate.data
            assert mapped_gate.adjoint == original_gate.adjoint
            assert mapped_gate.qubits == tuple(q + 1 for q in original_gate.qubits)

    b_unit = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
    encoding = AmplitudeEncoding(b_unit, normalize=False).build(2).gates
    angles = H._compute_rotation_angles(8, t, C, H._gershgorin_spectral_radius(Ac), 1e-2)
    spec = solver._build_circuit_spec(
        Ac, b_unit, 2, 3, t, angles, pauli_terms=terms, trotter_steps=steps
    )
    for original_gate, mapped_gate in zip(encoding, spec.gates[: len(encoding)]):
        assert mapped_gate.name == original_gate.name
        assert mapped_gate.params == original_gate.params
        assert mapped_gate.data == original_gate.data
        assert mapped_gate.adjoint == original_gate.adjoint
        assert mapped_gate.qubits == tuple(q + 4 for q in original_gate.qubits)


def test_a_single_non_identity_term_is_trotterised_exactly():
    """``[[2,1],[1,2]]`` is not an approximate case for having "an off-diagonal X
    term".  It is ``2I + X`` — one non-identity term, so the product formula is exact
    and the selector correctly asks for a single step.
    """
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    terms = pauli_decompose(A.astype(complex))
    assert [s for _, s in terms] == ["I", "X"]
    t, C, _ = H._honest_qpe_params(A.astype(complex), 6)
    assert _honest()._resolve_trotter_steps(terms, t, C, 6) == 1
    result = _honest().solve(A, np.array([1.0, 0.0]))
    assert _signed_cos(result.solution, np.linalg.solve(A, np.array([1.0, 0.0]))) > 0.9999


def test_the_conditioning_hint_does_not_pass_register_capacity_off_as_kappa():
    """In honest mode ``(N−2)/2`` is a property of the register, not of ``A``, so the
    near-zero-probability warning states it as the register's capacity, not as ``κ``."""
    A = np.diag([1.0, 1e-9])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _honest(n_l=7).solve(A, np.array([0.0, 1.0]))
    near_zero = [m for m in (str(w.message) for w in caught) if "near zero" in m]
    assert near_zero
    assert "resolves κ up to" in near_zero[0]
    assert "Gershgorin bound is loose" in near_zero[0]
