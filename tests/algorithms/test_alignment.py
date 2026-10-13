"""Kernel alignment: training a QuantumKernel's weights against its Gram matrix."""

import numpy as np
import pytest

from eigenkit import QSVC, QuantumKernel, align_kernel
from eigenkit.core.optimizers import spsa
from eigenkit.qml import kernel_target_alignment, svc_dual_loss


def _bandwidth_map(qc, x, w):
    """RX(w0 * x) per qubit plus a CNOT: the weight scales the encoding, so
    alignment has a real effect on the kernel."""
    from eigenkit.core.ir import Multiply

    for q in range(len(x)):
        qc.rx(Multiply(w[0], x[q]), qubit=q)
    qc.cnot(0, 1)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, size=(24, 2))
    y = (np.sin(X[:, 0]) * np.cos(X[:, 1]) > 0).astype(int)
    return X, y


def _kernel(weights=(3.0,)):
    return QuantumKernel(
        n_features=2, feature_map=_bandwidth_map, n_weights=1, weights=list(weights)
    )


def test_spsa_minimizes_a_quadratic():
    result = spsa(
        lambda x: float(np.sum((x - 1.0) ** 2)),
        np.zeros(3),
        maxiter=300,
        learning_rate=0.2,
        perturbation=0.1,
    )
    np.testing.assert_allclose(result.x, 1.0, atol=0.05)
    assert result.success and result.fun < result.initial_fun
    assert len(result.history) == 300
    assert result.best_fun <= min(result.history)


def test_spsa_calibration_takes_a_sensible_first_step():
    steps = []
    spsa(
        lambda x: 1000.0 * float(np.sum(x**2)),
        np.full(2, 0.5),
        maxiter=1,
        learning_rate=None,
        callback=lambda k, x, value: steps.append(x),
    )
    moved = np.linalg.norm(steps[0] - 0.5)
    assert 0.1 < moved < 2.0, "calibration rescales a steep objective to a ~2*pi/10 step"


def test_minimize_expectation_accepts_spsa():
    from eigenkit import Circuit, Variational, Z, minimize_expectation

    circuit = Circuit([Variational(depth=1)], observables=Z(0))
    seen = []
    result = minimize_expectation(
        circuit, optimizer="spsa", max_iterations=200, callback=seen.append
    )
    assert result.optimal_value < -0.9
    assert len(seen) == 200


@pytest.mark.parametrize("loss", ["svc", "kta"])
def test_alignment_lowers_the_loss(data, loss):
    X, y = data
    result = align_kernel(_kernel(), X, y, loss=loss, maxiter=25)
    assert result.loss < result.initial_loss
    assert result.success
    assert result.weights.shape == (1,)
    assert len(result.loss_history) == 25
    np.testing.assert_allclose(result.kernel.weights, result.weights)
    assert result.loss == pytest.approx(min(result.loss_history + (result.initial_loss,)))


def test_the_svc_loss_matches_the_dual_objective(data):
    from sklearn.svm import SVC

    X, y = data
    K = _kernel().compute_kernel_matrix(X)
    svc = SVC(kernel="precomputed").fit(K, y)
    alpha_y = np.zeros(len(y))
    alpha_y[svc.support_] = svc.dual_coef_[0]
    expected = np.abs(alpha_y).sum() - 0.5 * alpha_y @ K @ alpha_y
    assert svc_dual_loss(K, y) == pytest.approx(expected)


def test_the_svc_loss_rejects_more_than_two_classes():
    with pytest.raises(ValueError, match="binary labels"):
        svc_dual_loss(np.eye(6), np.array([0, 1, 2] * 2))


def test_kta_is_one_for_the_ideal_kernel():
    y = np.array([0, 0, 1, 1])
    ideal = np.where(y[:, None] == y[None, :], 1.0, -1.0)
    assert kernel_target_alignment(ideal, y) == pytest.approx(1.0)


def test_scipy_optimizer_and_callback(data):
    X, y = data
    seen = []
    result = align_kernel(
        _kernel(),
        X,
        y,
        loss="kta",
        optimizer="COBYLA",
        maxiter=15,
        callback=lambda i, w, value: seen.append(value),
    )
    assert result.loss <= result.initial_loss
    assert len(seen) == result.n_evaluations


def test_trained_kernel_feeds_qsvc(data):
    X, y = data
    result = align_kernel(_kernel(), X, y, maxiter=10)
    model = QSVC(kernel=result.kernel).fit(X, y)
    assert 0.0 <= model.score(X, y) <= 1.0


def test_a_kernel_without_weights_is_rejected(data):
    X, y = data
    with pytest.raises(ValueError, match="no trainable weights"):
        align_kernel(QuantumKernel(n_features=2), X, y)


def test_unknown_loss_is_rejected(data):
    X, y = data
    with pytest.raises(ValueError, match="loss must be one of"):
        align_kernel(_kernel(), X, y, loss="hinge")
