"""End-to-end QAOA on the core spine: build ansatz -> optimize -> decode.

Torch-free and backend-agnostic (numpy adjoint, PennyLane parameter-shift)."""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.optimization import QAOA, QUBO, max_cut
from eigenkit.optimization.problems import Ising


@pytest.fixture(params=["numpy", "pennylane"])
def provider(request):
    if request.param == "pennylane":
        pytest.importorskip("pennylane")
    return request.param


def _optimal_indices(problem):
    n = problem.num_vars
    energies = []
    for x in range(2**n):
        bits = tuple((x >> (n - 1 - k)) & 1 for k in range(n))
        energies.append(problem.energy(bits))
    energies = np.array(energies)
    return np.flatnonzero(np.isclose(energies, energies.min()))


class TestQAOA:
    def test_finds_max_cut(self, provider):
        # 4-cycle MaxCut; the decode recovers an optimal cut on every backend.
        problem = max_cut([(0, 1), (1, 2), (2, 3), (3, 0)])
        result = QAOA(layers=2, provider=provider, random_state=1).solve(problem)
        _, optimal_energy = problem.brute_force()
        assert np.isclose(result.best_energy, optimal_energy)
        assert problem.energy(result.best_bitstring) == result.best_energy

    def test_solves_qubo(self, provider):
        # Minimize x0 + x1 - 2 x0 x1  (min -? ) over {0,1}^2.
        Q = np.array([[1.0, -2.0], [0.0, 1.0]])
        problem = QUBO(Q)
        result = QAOA(layers=2, provider=provider, random_state=3).solve(problem)
        _, optimal_energy = problem.brute_force()
        assert np.isclose(result.best_energy, optimal_energy)

    def test_concentrates_probability_on_optimum(self):
        # QAOA should shape the state toward optimal bitstrings — the optimal
        # set should carry well more than its uniform share of probability.
        problem = max_cut([(0, 1), (1, 2), (2, 3), (3, 0)])
        result = QAOA(layers=2, provider="numpy", random_state=1).solve(problem)
        optimal = _optimal_indices(problem)
        optimal_probability = float(result.probabilities[optimal].sum())
        uniform_share = len(optimal) / 2**problem.num_vars
        assert optimal_probability > 2 * uniform_share

    def test_result_shapes(self):
        problem = max_cut([(0, 1), (1, 2)])
        result = QAOA(layers=1, provider="numpy").solve(problem)
        assert result.probabilities.shape == (2**3,)
        assert np.isclose(result.probabilities.sum(), 1.0)
        assert result.optimal_parameters.shape == (2,)
        assert len(result.best_bitstring) == 3


class TestRunsOnAnyStatevectorBackend:
    """QAOA reads its answer out of `circuit.probabilities()`.

    Not every backend implements that natively, but qiskit, qiskit-aer,
    qulacs and qsim return a perfectly good statevector for the same circuit.
    `Circuit.probabilities` derives it from the statevector when there is no
    native implementation, so QAOA runs on every one of them.
    """

    @staticmethod
    def _triangle():
        # A 3-cycle max-cut: every optimum has energy -1.
        return Ising(
            h=np.zeros(3),
            J=np.triu(np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]), 1),
        )

    @pytest.mark.parametrize(
        "provider,module",
        [
            ("numpy", None),
            ("pennylane", "pennylane"),
            ("qiskit", "qiskit"),
            ("qiskit-aer", "qiskit_aer"),
            ("qulacs", "qulacs"),
            ("qsim", "qsimcirq"),
            # No statevector, but a native probabilities().
            ("braket", "braket"),
        ],
    )
    def test_finds_the_true_optimum(self, provider, module):
        if module is not None:
            pytest.importorskip(module)
        problem = self._triangle()
        _, exact = problem.brute_force()
        result = QAOA(layers=2, random_state=0, provider=provider).solve(problem)
        assert result.best_energy == pytest.approx(exact, abs=1e-9)
        assert problem.energy(result.best_bitstring) == pytest.approx(
            result.best_energy, abs=1e-9
        )

    def test_a_backend_without_a_statevector_says_so(self, monkeypatch):
        pytest.importorskip("braket")
        from eigenkit.core.backends.braket_backend import BraketCompiled

        # Braket has neither a statevector nor, once this is removed, a
        # native probabilities(): the case the message is for.
        monkeypatch.delattr(BraketCompiled, "probabilities")
        with pytest.raises(NotImplementedError, match="statevector"):
            QAOA(layers=1, random_state=0, provider="braket").solve(self._triangle())
