"""Tests for the Ising / QUBO models and their qubit Hamiltonians."""

from __future__ import annotations

import numpy as np

from eigenkit.optimization import QUBO, Ising, max_cut


def _all_bitstrings(n):
    return [tuple((x >> (n - 1 - k)) & 1 for k in range(n)) for x in range(2**n)]


class TestQUBOIsing:
    def test_qubo_to_ising_energies_match(self):
        # The conversion must preserve the objective on every bitstring.
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((4, 4))
        qubo = QUBO(Q, constant=0.7)
        ising = qubo.to_ising()
        for bits in _all_bitstrings(4):
            assert np.isclose(qubo.energy(bits), ising.energy(bits))

    def test_hamiltonian_offset_and_expectation(self):
        ising = Ising(
            h=np.array([0.5, 0.0]), J=np.array([[0.0, 1.0], [0.0, 0.0]]), constant=2.0
        )
        observable, offset = ising.to_hamiltonian()
        assert offset == 2.0
        # <H> on |00> (s = +1, +1) = 0.5*1 + 1.0*1 = 1.5; energy = 1.5 + offset.
        assert np.isclose(ising.energy((0, 0)), 1.5 + 2.0)

    def test_brute_force_minimum(self):
        # H = Z0 Z1: minimum -1 when the spins differ.
        ising = Ising(h=np.zeros(2), J=np.array([[0.0, 1.0], [0.0, 0.0]]))
        bits, energy = ising.brute_force()
        assert np.isclose(energy, -1.0)
        assert bits in {(0, 1), (1, 0)}


class TestMaxCut:
    def test_triangle(self):
        # Triangle: best cut separates one node, cutting 2 of 3 edges.
        problem = max_cut([(0, 1), (1, 2), (2, 0)])
        _, energy = problem.brute_force()
        assert np.isclose(-energy, 2.0)  # cut value = -energy

    def test_square_cycle(self):
        # 4-cycle: max cut = 4 (alternating coloring cuts every edge).
        problem = max_cut([(0, 1), (1, 2), (2, 3), (3, 0)])
        bits, energy = problem.brute_force()
        assert np.isclose(-energy, 4.0)
        assert bits in {(0, 1, 0, 1), (1, 0, 1, 0)}

    def test_weighted(self):
        problem = max_cut([(0, 1), (1, 2)], weights=[3.0, 1.0])
        _, energy = problem.brute_force()
        assert np.isclose(-energy, 4.0)  # cut both edges
