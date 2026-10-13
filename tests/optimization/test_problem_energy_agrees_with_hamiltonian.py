"""The energy a problem reports and the operator QAOA minimises must agree.

`Ising` documents `J` as strictly upper-triangular -- "a symmetric matrix and
its upper half give the same problem" -- and `to_hamiltonian` honours that,
reading only `i < j`. `Ising.energy` must too: a plain `s @ J @ s` reads the
whole matrix, so a symmetric `J` would count every pair twice and a non-zero
diagonal would add a constant the operator never sees.

The bitstring ranking survives a uniform doubling, so such a mismatch would go
unnoticed in the assignment QAOA finds -- it would show up only as an energy
twice the one actually optimised, with `result.expectation` unable to reach
`result.best_energy`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from eigenkit.optimization.problems import QUBO, Ising


def _hamiltonian_energy(problem, bits) -> float:
    observable, offset = problem.to_hamiltonian()
    spins = 1 - 2 * np.asarray(bits, dtype=np.float64)
    total = offset
    for term in observable.terms:
        value = term.coeff
        for _, qubit in term.paulis:
            value *= spins[qubit]
        total += value
    return total


def _all_bits(n):
    return list(itertools.product([0, 1], repeat=n))


class TestIsing:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_symmetric_and_upper_triangular_j_agree(self, n):
        """The equivalence the docstring promises."""
        rng = np.random.default_rng(n)
        upper = np.triu(rng.normal(size=(n, n)), k=1)
        symmetric = upper + upper.T
        h = rng.normal(size=n)
        for bits in _all_bits(n):
            assert Ising(h=h, J=upper).energy(bits) == pytest.approx(
                Ising(h=h, J=symmetric).energy(bits)
            )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_the_diagonal_is_ignored(self, n):
        rng = np.random.default_rng(n + 10)
        J = np.triu(rng.normal(size=(n, n)), k=1)
        h = rng.normal(size=n)
        with_diagonal = J.copy()
        np.fill_diagonal(with_diagonal, rng.normal(size=n))
        for bits in _all_bits(n):
            assert Ising(h=h, J=J).energy(bits) == pytest.approx(
                Ising(h=h, J=with_diagonal).energy(bits)
            )

    @pytest.mark.parametrize("symmetric", [False, True])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_energy_matches_the_hamiltonian_on_every_assignment(self, n, symmetric):
        rng = np.random.default_rng(n * 3 + int(symmetric))
        upper = np.triu(rng.normal(size=(n, n)), k=1)
        J = upper + upper.T if symmetric else upper
        problem = Ising(h=rng.normal(size=n), J=J, constant=float(rng.normal()))
        for bits in _all_bits(n):
            assert problem.energy(bits) == pytest.approx(
                _hamiltonian_energy(problem, bits), abs=1e-12
            )

    def test_brute_force_finds_the_hamiltonian_minimum(self):
        rng = np.random.default_rng(7)
        upper = np.triu(rng.normal(size=(4, 4)), k=1)
        problem = Ising(h=rng.normal(size=4), J=upper + upper.T)
        _, energy = problem.brute_force()
        assert energy == pytest.approx(
            min(_hamiltonian_energy(problem, b) for b in _all_bits(4)), abs=1e-12
        )


class TestQubo:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_qubo_energy_matches_its_ising_form(self, n):
        rng = np.random.default_rng(n + 100)
        problem = QUBO(Q=rng.normal(size=(n, n)), constant=float(rng.normal()))
        ising = problem.to_ising()
        for bits in _all_bits(n):
            assert problem.energy(bits) == pytest.approx(ising.energy(bits), abs=1e-12)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_qubo_energy_matches_its_hamiltonian(self, n):
        rng = np.random.default_rng(n + 200)
        problem = QUBO(Q=rng.normal(size=(n, n)), constant=float(rng.normal()))
        for bits in _all_bits(n):
            assert problem.energy(bits) == pytest.approx(
                _hamiltonian_energy(problem, bits), abs=1e-12
            )
