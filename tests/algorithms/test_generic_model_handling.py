"""What the Generic estimators do with the model they are handed.

Two properties, both about state staying on its own side of a boundary.

  * `fit()` probes the model's output shape with a single sample, in eval
    mode. In training mode any batch-statistics layer rejects that probe: a
    standard MLP with `BatchNorm1d` -- the most ordinary model a user brings
    to a Generic estimator -- could not be fitted at all, and a model
    containing `MinMaxNorm` would have its running statistics seeded from that
    one probe row.
  * `reset_parameters` is how `fit()` re-initialises after seeding, so
    `TorchCircuit` defines it like every other torch layer. Without it,
    `random_state` would vary the classical weights and leave the quantum ones
    bit-identical.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from eigenkit import GenericClassifier, GenericRegressor
from eigenkit.torch.circuit import TorchCircuit
from eigenkit.torch.layers import Basic, MinMaxNorm


def _data(n_features=3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, n_features))
    return X, X[:, 0]


class TestModelsWithBatchStatistics:
    def test_a_batchnorm_regressor_fits(self):
        X, y = _data()
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 1))
        fitted = GenericRegressor(model=model, epochs=1).fit(X, y)
        assert fitted.predict(X).shape == (20,)

    def test_a_batchnorm_classifier_fits(self):
        X, y = _data()
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        fitted = GenericClassifier(model=model, epochs=1).fit(X, (y > 0).astype(int))
        assert fitted.predict(X).shape == (20,)

    def test_the_probe_does_not_seed_running_statistics(self):
        X, y = _data(n_features=2)
        norm = MinMaxNorm(feature_dim=2)
        model = nn.Sequential(norm, nn.Linear(2, 1))
        GenericRegressor(model=model, epochs=0).fit(X, y)
        fitted_norm = next(
            m
            for m in GenericRegressor(model=model, epochs=0).fit(X, y).model_.modules()
            if isinstance(m, MinMaxNorm)
        )
        assert not bool(fitted_norm.any_batch_tracked), (
            "the shape probe established running statistics from one row"
        )


class TestRandomStateReachesTheQuantumWeights:
    @staticmethod
    def _quantum_weights(random_state):
        X, y = _data(n_features=2)
        torch.manual_seed(999)
        model = nn.Sequential(Basic(n_qubits=2, depth=1), nn.Linear(2, 1))
        fitted = GenericRegressor(model=model, epochs=0, random_state=random_state).fit(
            X, y
        )
        return [
            p.detach().clone()
            for m in fitted.model_.modules()
            if isinstance(m, Basic)
            for p in m.parameters()
        ]

    def test_different_seeds_give_different_quantum_weights(self):
        first, second = self._quantum_weights(1), self._quantum_weights(2)
        assert first, "no quantum parameters found"
        assert not all(torch.equal(a, b) for a, b in zip(first, second))

    def test_the_same_seed_reproduces_them(self):
        first, second = self._quantum_weights(7), self._quantum_weights(7)
        assert all(torch.equal(a, b) for a, b in zip(first, second))


class TestTorchCircuitResetParameters:
    def test_it_redraws_the_weights(self):
        from eigenkit.core.blocks import BasicVariational, assemble

        circuit = TorchCircuit(assemble([BasicVariational(depth=1)]), provider="numpy")
        before = circuit.weights.detach().clone()
        circuit.reset_parameters()
        assert not torch.equal(before, circuit.weights.detach())

    def test_it_preserves_shape_and_dtype(self):
        from eigenkit.core.blocks import BasicVariational, assemble

        circuit = TorchCircuit(assemble([BasicVariational(depth=1)]), provider="numpy")
        shape, dtype = circuit.weights.shape, circuit.weights.dtype
        circuit.reset_parameters()
        assert circuit.weights.shape == shape
        assert circuit.weights.dtype == dtype

    def test_it_is_seeded_by_the_global_rng(self):
        from eigenkit.core.blocks import BasicVariational, assemble

        circuit = TorchCircuit(assemble([BasicVariational(depth=1)]), provider="numpy")
        torch.manual_seed(3)
        circuit.reset_parameters()
        first = circuit.weights.detach().clone()
        torch.manual_seed(3)
        circuit.reset_parameters()
        assert torch.equal(first, circuit.weights.detach())
