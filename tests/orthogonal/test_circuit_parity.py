"""The key test of the circuit view: on the unary basis an `RBSNetwork` is exactly
`orthogonal_matrix`, on every provider that returns a statevector."""

import numpy as np
import pytest

from eigenkit import Circuit
from eigenkit.core.blocks import assemble
from eigenkit.core.builder import CircuitBuilder
from eigenkit.orthogonal import (
    RBSNetwork,
    Topology,
    brick,
    butterfly,
    diagonal,
    orthogonal_matrix,
    pyramid,
    x,
)


PROVIDERS = ["numpy", "qiskit", "pennylane"]

TOPOLOGIES = [
    butterfly(2),
    butterfly(4),
    butterfly(8),
    pyramid(2),
    pyramid(3),
    pyramid(5),
    pyramid(4, reflection=True),
    x(2),
    x(4),
    x(6),
    diagonal(4),
    diagonal(7),
    brick(5, 3),
    Topology(4, (((0, 3),), ((1, 2),), ((0, 1), (2, 3)))),
]
IDS = [f"{t.name}{t.n}{'r' if t.reflection else ''}" for t in TOPOLOGIES]


def _basis_state(n, wire):
    builder = CircuitBuilder(n_qubits=n)
    builder.x(wire)
    return builder.to_block()


def _unary(sv, n):
    return np.array([sv[2 ** (n - 1 - wire)] for wire in range(n)])


def _require(provider):
    if provider != "numpy":
        pytest.importorskip(provider)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_qubit_zero_is_the_most_significant_bit(provider):
    _require(provider)
    circuit = Circuit(assemble([_basis_state(3, 0)]), provider=provider)
    sv = circuit.statevector(np.zeros(0))
    assert np.argmax(np.abs(sv)) == 4


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("t", TOPOLOGIES, ids=IDS)
def test_unary_amplitudes_equal_the_orthogonal_matrix(t, provider):
    _require(provider)
    theta = np.random.default_rng(t.n_angles).normal(scale=1.5, size=t.n_angles)
    W = orthogonal_matrix(t, theta)
    for column in range(t.n):
        circuit = Circuit(
            assemble([_basis_state(t.n, column), RBSNetwork(t)]), provider=provider
        )
        sv = np.asarray(circuit.statevector(theta))
        amplitudes = _unary(sv, t.n)
        np.testing.assert_allclose(amplitudes, W[:, column], atol=1e-10)
        assert np.sum(np.abs(sv) ** 2) - np.sum(amplitudes**2) == pytest.approx(
            0.0, abs=1e-10
        )


def test_one_rbs_gate_leaves_the_other_basis_states_alone():
    t = Topology(2, (((0, 1),),))
    circuit = Circuit(assemble([RBSNetwork(t)]))
    sv = circuit.statevector(np.array([0.8]))
    np.testing.assert_allclose(sv, [1, 0, 0, 0], atol=1e-12)
    both = Circuit(assemble([_basis_state(2, 0), _basis_state(2, 1), RBSNetwork(t)]))
    np.testing.assert_allclose(
        both.statevector(np.array([0.8])), [0, 0, 0, 1], atol=1e-12
    )
