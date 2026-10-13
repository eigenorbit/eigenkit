"""Tests for the quantum reservoir module: driver dispatch (parallel vs
sequential), the three policies, feature extractors (validated against the
trusted circuit forward path), and the ridge readout."""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble
from eigenkit.core.circuit import Circuit
from eigenkit.core.observables import Z, all_z, from_string
from eigenkit.reservoir import (
    Carry,
    LeakyCarry,
    LinearReadout,
    Stateless,
    amplitude_features,
    expectation_features,
    pauli_z_features,
    probability_features,
    run_reservoir,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def reservoir_2q():
    """A 2-qubit reservoir consuming F=2 features/step (n_inputs == F)."""
    ir = assemble(
        [AngleEncoding(in_features=2), BasicVariational(depth=2, rotation="RY")]
    )
    circuit = Circuit(ir, all_z(2), "numpy", "numpy")
    weights = np.random.default_rng(7).standard_normal(ir.n_weights)
    return circuit, weights


@pytest.fixture
def observables():
    # A Hamiltonian (X/Y/Z + coeffs) plus two simpler terms — a list, so the
    # circuit forward returns one column per observable.
    return [
        from_string("0.5 Z0 X1 - 0.3 Y0 + 0.7 Z1"),
        Z(0).to_observable(),
        from_string("X0 X1"),
    ]


# ---- features validated against the circuit's own forward path ----


class TestExpectationFeatures:
    def test_matches_circuit_forward(self, reservoir_2q, observables, rng):
        circuit, weights = reservoir_2q
        circuit = Circuit(circuit.spec, observables, "numpy", "numpy")
        inputs = rng.standard_normal((5, 2))
        states = circuit.statevector(weights, inputs)[:, None, :]  # (5, 1, 4)

        feats = expectation_features(states, observables)[:, 0, :]
        direct = circuit(weights, inputs)  # (5, 3)
        assert feats.shape == (5, 3)
        assert np.allclose(feats, direct, atol=1e-10)

    def test_accepts_hamiltonian_string(self, reservoir_2q, rng):
        circuit, weights = reservoir_2q
        states = circuit.statevector(weights, rng.standard_normal((4, 2)))[:, None, :]
        feats = expectation_features(states, "Z0 + Z1")
        assert feats.shape == (4, 1, 1)

    def test_pauli_z_matches_all_z_forward(self, reservoir_2q, rng):
        circuit, weights = reservoir_2q  # observables are all_z(2)
        inputs = rng.standard_normal((6, 2))
        states = circuit.statevector(weights, inputs)[:, None, :]
        feats = pauli_z_features(states)[:, 0, :]
        assert feats.shape == (6, 2)
        assert np.allclose(feats, circuit(weights, inputs), atol=1e-10)


class TestReadoutFeatures:
    def test_probability_features_normalized(self, reservoir_2q, rng):
        circuit, weights = reservoir_2q
        states = circuit.statevector(weights, rng.standard_normal((5, 2)))[:, None, :]
        probs = probability_features(states)
        assert probs.shape == (5, 1, 4)
        assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-10)

    def test_amplitude_features_shape(self, reservoir_2q, rng):
        circuit, weights = reservoir_2q
        states = circuit.statevector(weights, rng.standard_normal((5, 2)))[:, None, :]
        amps = amplitude_features(states)
        assert amps.shape == (5, 1, 8)  # re ‖ im over 2**2


# ---- driver: stateless / parallel path ----


class TestStatelessPolicy:
    def test_window1_equals_independent_evaluation(self, reservoir_2q, rng):
        circuit, weights = reservoir_2q
        seq = rng.standard_normal((3, 4, 2))  # (B, T, F)
        states = run_reservoir(circuit, Stateless(window=1), seq, weights)
        assert states.shape == (3, 4, 4)
        # Memoryless: each (b,t) equals evolving that input alone, from |0…0⟩.
        B, T, F = seq.shape
        ref = circuit.statevector(weights, seq.reshape(B * T, F)).reshape(B, T, -1)
        assert np.allclose(states, ref, atol=1e-12)

    def test_window_assembly_and_padding(self):
        seq = np.arange(2 * 3 * 1, dtype=np.float64).reshape(2, 3, 1)  # (B=2,T=3,F=1)
        expanded = Stateless(window=2).expand_inputs(seq)
        assert expanded.shape == (2, 3, 2)
        # First step is front zero-padded; later steps are [x_{t-1}, x_t].
        assert np.allclose(expanded[:, 0, :], np.stack([np.zeros(2), seq[:, 0, 0]], 1))
        assert np.allclose(expanded[:, 1, 0], seq[:, 0, 0])
        assert np.allclose(expanded[:, 1, 1], seq[:, 1, 0])

    def test_windowed_run_matches_manual(self, rng):
        # Circuit consumes window*F = 2 inputs.
        ir = assemble(
            [AngleEncoding(in_features=2), BasicVariational(depth=1, rotation="RY")]
        )
        circuit = Circuit(ir, all_z(2), "numpy", "numpy")
        weights = rng.standard_normal(ir.n_weights)
        seq = rng.standard_normal((3, 5, 1))

        states = run_reservoir(circuit, Stateless(window=2), seq, weights)
        expanded = Stateless(window=2).expand_inputs(seq)
        B, T, _ = expanded.shape
        ref = circuit.statevector(weights, expanded.reshape(B * T, 2)).reshape(B, T, -1)
        assert np.allclose(states, ref, atol=1e-12)

    def test_n_inputs_mismatch_raises(self, rng):
        ir = assemble(
            [AngleEncoding(in_features=1), BasicVariational(depth=1, rotation="RY")]
        )
        circuit = Circuit(ir, all_z(1), "numpy", "numpy")  # n_inputs == 1
        weights = rng.standard_normal(ir.n_weights)
        seq = rng.standard_normal((2, 4, 1))
        with pytest.raises(ValueError, match="n_inputs"):
            run_reservoir(circuit, Stateless(window=2), seq, weights)  # window*F = 2


# ---- driver: sequential carry path ----


class TestCarryPolicies:
    def test_carry_matches_manual_chain(self, reservoir_2q, rng):
        circuit, weights = reservoir_2q
        seq = rng.standard_normal((3, 6, 2))
        states = run_reservoir(circuit, Carry(), seq, weights)
        assert states.shape == (3, 6, 4)

        prev = None
        ref = []
        for t in range(seq.shape[1]):
            prev = circuit.statevector(weights, seq[:, t, :], initial_state=prev)
            ref.append(prev)
        assert np.allclose(states, np.stack(ref, axis=1), atol=1e-12)

    def test_leaky_alpha_one_equals_carry(self, reservoir_2q, rng):
        circuit, weights = reservoir_2q
        seq = rng.standard_normal((2, 5, 2))
        carried = run_reservoir(circuit, Carry(), seq, weights)
        leaky = run_reservoir(circuit, LeakyCarry(alpha=1.0), seq, weights)
        assert np.allclose(carried, leaky, atol=1e-12)

    def test_leaky_states_stay_normalized(self, reservoir_2q, rng):
        circuit, weights = reservoir_2q
        seq = rng.standard_normal((2, 7, 2))
        states = run_reservoir(circuit, LeakyCarry(alpha=0.6), seq, weights)
        norms = np.sum(np.abs(states) ** 2, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_leaky_differs_from_carry(self, reservoir_2q, rng):
        circuit, weights = reservoir_2q
        seq = rng.standard_normal((2, 5, 2))
        carried = run_reservoir(circuit, Carry(), seq, weights)
        leaky = run_reservoir(circuit, LeakyCarry(alpha=0.3), seq, weights)
        assert not np.allclose(carried, leaky, atol=1e-3)

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            LeakyCarry(alpha=1.5)


# ---- ridge readout ----


class TestLinearReadout:
    def test_recovers_linear_map(self, rng):
        X = rng.standard_normal((80, 4))
        coef = rng.standard_normal(4)
        y = X @ coef + 0.25
        readout = LinearReadout(ridge=1e-10).fit(X, y)
        assert np.allclose(readout.predict(X), y, atol=1e-4)

    def test_multioutput(self, rng):
        X = rng.standard_normal((80, 3))
        coef = rng.standard_normal((3, 2))
        Y = X @ coef
        readout = LinearReadout(ridge=1e-10).fit(X, Y)
        preds = readout.predict(X)
        assert preds.shape == (80, 2)
        assert np.allclose(preds, Y, atol=1e-4)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            LinearReadout().predict(np.zeros((2, 3)))


# ---- end-to-end smoke ----


def test_end_to_end_reservoir_pipeline(reservoir_2q, rng):
    circuit, weights = reservoir_2q
    seq = rng.standard_normal((4, 8, 2))
    states = run_reservoir(circuit, Carry(), seq, weights)
    feats = pauli_z_features(states)  # (4, 8, 2)
    B, T, n = feats.shape
    targets = seq[:, :, 0].reshape(B * T)  # toy: regress the first input channel
    readout = LinearReadout(ridge=1e-3).fit(feats.reshape(B * T, n), targets)
    preds = readout.predict(feats.reshape(B * T, n))
    assert preds.shape == (B * T,)
