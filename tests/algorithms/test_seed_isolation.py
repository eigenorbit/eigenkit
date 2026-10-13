"""An estimator's seed must not reach the caller's RNG.

`torch.manual_seed` sets the process-wide generator, on the CPU and on every
CUDA device. The estimators call it while building a model, inside
`isolated_rng`; without that, any `fit()` or `load()` would silently reseed
whatever else the process was doing. These tests pin both halves: the
caller's stream survives, and the estimator stays exactly as reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from eigenkit import HQNC, HQNR, GenericRegressor


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available (torch.cuda.is_available() is False)",
)


def _tiny_classification():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(8, 2))
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


def _tiny_regression():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(8, 2))
    return X, X[:, 0]


def _hqnc():
    return HQNC(in_features=2, n_qubits=2, depth=1, epochs=1, provider="numpy")


def _hqnr():
    return HQNR(in_features=2, n_qubits=2, depth=1, epochs=1, provider="numpy")


class TestCallerStreamSurvives:
    def test_fit_leaves_the_cpu_stream_untouched(self):
        X, y = _tiny_classification()

        torch.manual_seed(1234)
        expected = torch.randn(3).tolist()

        torch.manual_seed(1234)
        _hqnc().fit(X, y)
        actual = torch.randn(3).tolist()

        assert actual == expected

    def test_a_regressor_fit_leaves_the_cpu_stream_untouched(self):
        X, y = _tiny_regression()

        torch.manual_seed(99)
        expected = torch.randn(3).tolist()

        torch.manual_seed(99)
        _hqnr().fit(X, y)
        actual = torch.randn(3).tolist()

        assert actual == expected

    def test_load_leaves_the_cpu_stream_untouched(self, tmp_path):
        X, y = _tiny_classification()
        model = _hqnc()
        model.fit(X, y)
        path = tmp_path / "model.eigenkit"
        model.save(path)

        torch.manual_seed(7)
        expected = torch.randn(3).tolist()

        torch.manual_seed(7)
        HQNC.load(path)
        actual = torch.randn(3).tolist()

        assert actual == expected


class TestSeedingStillWorks:
    def test_the_same_random_state_still_gives_the_same_model(self):
        X, y = _tiny_classification()

        def weights():
            model = HQNC(
                in_features=2,
                n_qubits=2,
                depth=1,
                epochs=1,
                provider="numpy",
                random_state=11,
            )
            model.fit(X, y)
            return [p.detach().clone() for p in model._model.parameters()]

        first, second = weights(), weights()
        assert all(torch.equal(a, b) for a, b in zip(first, second))

    def test_a_different_random_state_gives_a_different_model(self):
        X, y = _tiny_classification()

        def weights(random_state):
            model = HQNC(
                in_features=2,
                n_qubits=2,
                depth=1,
                epochs=1,
                provider="numpy",
                random_state=random_state,
            )
            model.fit(X, y)
            return [p.detach().clone() for p in model._model.parameters()]

        assert not all(torch.equal(a, b) for a, b in zip(weights(11), weights(12)))

    def test_the_caller_cannot_change_the_result_by_reseeding(self):
        # The estimator's own random_state decides, not whatever the caller
        # last set: isolation must not cost reproducibility.
        X, y = _tiny_classification()

        def weights(caller_seed):
            torch.manual_seed(caller_seed)
            model = HQNC(
                in_features=2,
                n_qubits=2,
                depth=1,
                epochs=1,
                provider="numpy",
                random_state=5,
            )
            model.fit(X, y)
            return [p.detach().clone() for p in model._model.parameters()]

        assert all(torch.equal(a, b) for a, b in zip(weights(1), weights(2)))


class TestUnseededStaysUnseeded:
    """`random_state=None` must not be reproducible.

    The fork restores the RNG state on exit, so wrapping an *unseeded* call in
    it would hand every `fit()` the same starting state and the same model —
    reproducibility the caller explicitly declined. Only a call that seeds is
    isolated; there is nothing to contain in one that does not.
    """

    def test_two_unseeded_fits_give_different_models(self):
        X, y = _tiny_classification()

        def weights():
            model = HQNC(
                in_features=2,
                n_qubits=2,
                depth=1,
                epochs=1,
                provider="numpy",
                random_state=None,
            )
            model.fit(X, y)
            return [p.detach().clone() for p in model._model.parameters()]

        assert not all(torch.equal(a, b) for a, b in zip(weights(), weights()))

    def test_an_unseeded_fit_advances_the_caller_stream(self):
        # The other side of the same coin: an unseeded fit draws from the
        # caller's stream, so what follows it must not repeat.
        X, y = _tiny_classification()

        torch.manual_seed(1234)
        untouched = torch.randn(3).tolist()

        torch.manual_seed(1234)
        HQNC(
            in_features=2,
            n_qubits=2,
            depth=1,
            epochs=1,
            provider="numpy",
            random_state=None,
        ).fit(X, y)

        assert torch.randn(3).tolist() != untouched


@requires_cuda
class TestOnCuda:
    """`torch.manual_seed` seeds every CUDA device, so the fork must too."""

    def test_fit_leaves_the_cuda_stream_untouched(self):
        X, y = _tiny_classification()

        torch.manual_seed(2024)
        expected = torch.randn(3, device="cuda").tolist()

        torch.manual_seed(2024)
        _hqnc().fit(X, y)
        actual = torch.randn(3, device="cuda").tolist()

        assert actual == expected

    def test_the_cuda_rng_state_is_restored_byte_for_byte(self):
        X, y = _tiny_classification()
        torch.manual_seed(3)
        before = torch.cuda.get_rng_state().clone()

        _hqnc().fit(X, y)

        assert torch.equal(torch.cuda.get_rng_state(), before)

    def test_a_model_on_cuda_is_still_reproducible(self):
        X, y = _tiny_regression()

        def weights():
            net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)).cuda()
            model = GenericRegressor(
                model=net, in_features=2, epochs=2, batch_size=4, random_state=13
            )
            model.fit(X, y)
            return [p.detach().cpu().clone() for p in model._model.parameters()]

        first, second = weights(), weights()
        assert all(torch.equal(a, b) for a, b in zip(first, second))

    def test_weights_initialised_on_the_device_are_still_pinned(self):
        # `.cuda()` moves CPU-initialised weights; `device="cuda"` draws them
        # from the CUDA generator instead. Only the second path exercises the
        # device RNG that `torch.manual_seed` also sets.
        X, y = _tiny_regression()

        def weights():
            net = nn.Sequential(
                nn.Linear(2, 4, device="cuda"),
                nn.ReLU(),
                nn.Linear(4, 1, device="cuda"),
            )
            model = GenericRegressor(
                model=net, in_features=2, epochs=2, batch_size=4, random_state=17
            )
            model.fit(X, y)
            return [p.detach().cpu().clone() for p in model._model.parameters()]

        first, second = weights(), weights()
        assert all(torch.equal(a, b) for a, b in zip(first, second))

    def test_a_cuda_fit_leaves_the_cuda_stream_untouched(self):
        X, y = _tiny_regression()
        net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)).cuda()

        torch.manual_seed(55)
        expected = torch.randn(3, device="cuda").tolist()

        torch.manual_seed(55)
        GenericRegressor(
            model=net, in_features=2, epochs=2, batch_size=4, random_state=13
        ).fit(X, y)
        actual = torch.randn(3, device="cuda").tolist()

        assert actual == expected
