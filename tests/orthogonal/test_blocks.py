"""`RBSNetwork` as a block, `to_block` and `to_circuit`: shapes, the gate
decomposition, relocation in `assemble`, round trips and gradients."""

import json

import numpy as np
import pytest

from eigenkit import Circuit
from eigenkit.core.blocks import AngleEncoding, assemble
from eigenkit.core.ir import Constant, Negate, WeightRef
from eigenkit.orthogonal import (
    RBSNetwork,
    UnaryLoader,
    butterfly,
    orthogonal_matrix,
    orthogonal_matrix_jacobian,
    pyramid,
    to_block,
    to_circuit,
    unary_loader_angles,
    unary_probabilities,
    unary_readout,
    x,
)


def test_a_trainable_network_takes_one_weight_per_gate():
    t = butterfly(8)
    spec = assemble([RBSNetwork(t)])
    assert spec.n_qubits == 8
    assert spec.n_weights == t.n_angles
    assert spec.n_inputs == 0
    assert len(spec.gates) == 8 * t.n_angles


def test_each_rbs_gate_is_the_documented_decomposition():
    spec = assemble([RBSNetwork(x(2))])
    names = [(g.name, g.qubits) for g in spec.gates]
    assert names == [
        ("H", (0,)),
        ("CNOT", (0, 1)),
        ("H", (1,)),
        ("RY", (0,)),
        ("RY", (1,)),
        ("H", (1,)),
        ("CNOT", (0, 1)),
        ("H", (0,)),
    ]
    assert spec.gates[3].params == (Negate(WeightRef(0)),)
    assert spec.gates[4].params == (WeightRef(0),)


def test_the_reflection_adds_a_z_gate_on_the_last_wire():
    spec = assemble([RBSNetwork(pyramid(3, reflection=True))])
    assert spec.gates[-1].name == "Z"
    assert spec.gates[-1].qubits == (2,)


def test_weights_are_relocated_after_another_block():
    t = pyramid(3)
    spec = assemble([AngleEncoding(in_features=3), RBSNetwork(t), RBSNetwork(t)])
    ry = [g for g in spec.gates if g.name == "RY"]
    refs = {p.index for g in ry[3:] for p in g.params if isinstance(p, WeightRef)}
    assert refs == set(range(2 * t.n_angles))
    assert spec.n_weights == 2 * t.n_angles


def test_a_smaller_network_leaves_the_extra_qubits_alone():
    spec = assemble([AngleEncoding(in_features=5), RBSNetwork(pyramid(3))])
    assert spec.n_qubits == 5
    assert max(q for g in spec.gates[5:] for q in g.qubits) == 2


def test_a_circuit_that_is_too_small_is_rejected():
    with pytest.raises(ValueError, match="needs 4 qubits"):
        RBSNetwork(pyramid(4)).build(3)


def test_fixed_angles_become_constants():
    block = to_block(pyramid(3), np.array([0.1, -0.2, 0.3]))
    spec = assemble([block])
    assert spec.n_weights == 0
    assert spec.gates[3].params == (Constant(-0.1),)
    assert spec.gates[4].params == (Constant(0.1),)


def test_the_wrong_number_of_fixed_angles_is_rejected():
    with pytest.raises(ValueError, match="needs 3 angles"):
        RBSNetwork(pyramid(3), (0.1, 0.2))


def test_a_non_finite_angle_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        RBSNetwork(pyramid(2), (float("nan"),))


@pytest.mark.parametrize("angles", [None, (0.5, -0.25, 1.0)])
def test_the_network_round_trips_through_json(angles):
    block = RBSNetwork(pyramid(3, reflection=True), angles)
    again = RBSNetwork.from_dict(json.loads(json.dumps(block.to_dict())))
    assert again == block


def test_both_blocks_are_registered_for_bundles():
    from eigenkit.qml._serialization import _BLOCK_TYPES

    assert _BLOCK_TYPES["RBSNetwork"] is RBSNetwork
    assert _BLOCK_TYPES["UnaryLoader"] is UnaryLoader


@pytest.mark.parametrize("layout", ["diagonal", "parallel", "semi_diagonal"])
def test_a_circuit_spec_with_both_blocks_round_trips(layout):
    from eigenkit.qml._serialization import (
        deserialize_circuit_spec,
        serialize_circuit_spec,
    )

    blocks = [
        UnaryLoader(4, layout=layout),
        RBSNetwork(pyramid(4, reflection=True)),
        to_block(butterfly(4), np.random.default_rng(0).normal(size=4)),
    ]
    spec = json.loads(json.dumps(serialize_circuit_spec(blocks, unary_readout(4))))
    again, _ = deserialize_circuit_spec(spec)
    assert again == blocks
    assert assemble(again).gates == assemble(blocks).gates


def test_a_restored_circuit_spec_computes_the_same_probabilities():
    from eigenkit.qml._serialization import (
        deserialize_circuit_spec,
        serialize_circuit_spec,
    )

    t = pyramid(4)
    theta = np.random.default_rng(1).normal(size=t.n_angles)
    spec = serialize_circuit_spec([UnaryLoader(4), RBSNetwork(t)], unary_readout(4))
    blocks, observables = deserialize_circuit_spec(json.loads(json.dumps(spec)))
    circuit = Circuit(blocks, observables=observables)
    xs = np.random.default_rng(2).normal(size=(5, 4))
    p = unary_probabilities(circuit(theta, unary_loader_angles(xs)))
    expected = (xs @ orthogonal_matrix(t, theta).T) ** 2
    np.testing.assert_allclose(p, expected / (xs**2).sum(1, keepdims=True), atol=1e-12)


@pytest.mark.parametrize("loader", ["diagonal", "parallel", "semi_diagonal"])
def test_to_circuit_computes_the_squared_rotated_vector(loader):
    t = butterfly(8)
    rng = np.random.default_rng(3)
    theta = rng.normal(size=t.n_angles)
    xs = rng.normal(size=(4, 8))
    circuit = to_circuit(t, theta, loader=loader)
    p = unary_probabilities(circuit(np.zeros(0), unary_loader_angles(xs, loader)))
    expected = (xs @ orthogonal_matrix(t, theta).T) ** 2 / np.sum(xs**2, axis=1)[:, None]
    np.testing.assert_allclose(p, expected, atol=1e-10)


def test_to_circuit_without_a_loader_holds_only_the_network():
    circuit = to_circuit(pyramid(3), np.zeros(3), loader=None)
    assert circuit.spec.n_inputs == 0
    np.testing.assert_allclose(circuit(np.zeros(0)), [1.0, 1.0, 1.0])


@pytest.mark.parametrize("provider", ["qiskit", "pennylane"])
def test_to_circuit_agrees_with_numpy_on_another_provider(provider):
    pytest.importorskip(provider)
    t = x(4)
    theta = np.random.default_rng(5).normal(size=t.n_angles)
    angles = unary_loader_angles(np.array([0.3, -1.2, 0.7, 2.0]))
    reference = to_circuit(t, theta)(np.zeros(0), angles)
    other = to_circuit(t, theta, provider=provider)(np.zeros(0), angles)
    np.testing.assert_allclose(other, reference, atol=1e-10)


@pytest.mark.parametrize("provider", ["numpy", "qiskit"])
def test_the_weight_gradient_matches_the_matrix_derivative(provider):
    # qiskit has no jacobian of its own, so this runs the core parameter-shift rule,
    # which is exact for the two RY rotations each angle feeds.
    if provider != "numpy":
        pytest.importorskip(provider)
    t = pyramid(4)
    theta = np.random.default_rng(6).normal(size=t.n_angles)
    xv = np.array([0.4, -0.9, 1.3, 0.2])
    circuit = Circuit(
        assemble([UnaryLoader(4), RBSNetwork(t)]),
        observables=unary_readout(4),
        provider=provider,
    )
    jac = circuit.jacobian(theta, unary_loader_angles(xv))
    W = orthogonal_matrix(t, theta)
    dW = orthogonal_matrix_jacobian(t, theta)
    unit = xv / np.linalg.norm(xv)
    # <Z_i> = 1 - 2 (W u)_i^2, so d<Z_i>/dθ_k = -4 (W u)_i (dW_k u)_i.
    expected = -4 * (W @ unit)[:, None] * np.einsum("kij,j->ik", dW, unit)
    np.testing.assert_allclose(jac, expected, atol=1e-10)


def test_to_block_rejects_angles_of_the_wrong_shape():
    with pytest.raises(ValueError, match="needs 4 angles"):
        to_block(butterfly(4), np.zeros((2, 2)))


def test_the_loader_and_readout_accept_numpy_integers():
    loader = UnaryLoader(np.int64(4), "parallel")
    assert type(loader.n) is int
    assert len(unary_readout(np.int64(3))) == 3
