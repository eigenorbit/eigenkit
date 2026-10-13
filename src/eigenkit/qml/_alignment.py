"""Train the weights of a quantum kernel with kernel alignment.

[`align_kernel()`][eigenkit.qml.align_kernel] optimises the trainable weights of
a [`QuantumKernel`][eigenkit.qml.QuantumKernel] to minimise a loss computed from
its training Gram matrix. There are two built-in losses:

- `"svc"` is the optimal value of the dual problem of the support vector machine
  (SVM) on `K_theta`:
  `F(theta) = sum_i alpha_i - 1/2 sum_ij alpha_i alpha_j y_i y_j K_theta(x_i, x_j)`,
  where `alpha` is the dual solution. It is an upper bound on the generalisation
  error, and quantum kernel alignment (QKA) minimises it (Glick et al.,
  arXiv:2105.03406). It has the same value as `SVCLoss` in
  qiskit-machine-learning.
- `"kta"` is minus the kernel-target alignment
  `<K, Y>_F / (||K||_F ||Y||_F)`, where `Y_ij = +1` for a pair of the same class
  and `-1` otherwise (Cristianini et al. 2001). It needs no SVM fit at each
  evaluation.

Each evaluation of the loss computes a whole kernel matrix, so the default
optimiser is Simultaneous Perturbation Stochastic Approximation (SPSA). It
estimates the gradient from two evaluations per iteration, whatever the number of
weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ._kernels import QuantumKernel


def svc_dual_loss(K: np.ndarray, y: np.ndarray, C: float = 1.0) -> float:
    """Return the optimal dual objective of a support vector machine (SVM).

    The SVM is fitted on a precomputed kernel. This is the `"svc"` loss.

    Args:
        K: The training Gram matrix, of shape `(n, n)`.
        y: The labels, of shape `(n,)`. They must have exactly two classes.
        C: The regularisation parameter of the SVM.

    Returns:
        The value `sum |alpha_i y_i| - 1/2 (alpha y)^T K_sv (alpha y)`, summed
            over the support vectors. Lower is better.

    Raises:
        ValueError: If `y` does not have exactly two classes. Use `loss="kta"`
            for more classes.
    """
    from sklearn.svm import SVC

    if len(np.unique(y)) != 2:
        raise ValueError("the 'svc' loss needs binary labels; use loss='kta'.")
    svc = SVC(kernel="precomputed", C=C).fit(K, y)
    dual = svc.dual_coef_[0]  # alpha_i * y_i on the support vectors
    support = svc.support_
    K_sv = K[np.ix_(support, support)]
    return float(np.abs(dual).sum() - 0.5 * dual @ K_sv @ dual)


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Return the kernel-target alignment `<K, Y>_F / (||K||_F ||Y||_F)`.

    `Y_ij = +1` when `y_i == y_j` and `-1` otherwise. For binary labels in
    `{-1, +1}`, this is `y y^T`.

    Args:
        K: The Gram matrix, of shape `(n, n)`.
        y: The labels, of shape `(n,)`.

    Returns:
        A value in `[-1, 1]`. A higher value means that the kernel already
            separates the classes.
    """
    y = np.asarray(y)
    Y = np.where(y[:, None] == y[None, :], 1.0, -1.0)
    return float(np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y)))


_LOSSES: dict[str, Callable[..., float]] = {
    "svc": svc_dual_loss,
    "kta": lambda K, y, C=1.0: -kernel_target_alignment(K, y),
}


@dataclass(frozen=True)
class KernelAlignmentResult:
    """Outcome of [`align_kernel()`][eigenkit.qml.align_kernel].

    Attributes:
        weights: The trained weights.
        kernel: A copy of the input kernel with `weights`, ready for
            `QSVC(kernel=...)`.
        loss: The loss at `weights`.
        initial_loss: The loss at the starting weights.
        loss_history: The loss after every iteration of Simultaneous Perturbation
            Stochastic Approximation (SPSA), or at every evaluation for a SciPy
            optimiser. With SPSA, `weights` is the point with the lowest loss
            seen, including the starting weights.
        n_evaluations: The number of kernel matrices computed.
        success: Whether the loss is lower than at the starting point.
    """

    weights: np.ndarray
    kernel: QuantumKernel
    loss: float
    initial_loss: float
    loss_history: tuple[float, ...] = field(default_factory=tuple)
    n_evaluations: int = 0
    success: bool = True


def align_kernel(
    kernel: QuantumKernel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    loss: str | Callable[[np.ndarray, np.ndarray], float] = "svc",
    optimizer: str | Callable = "spsa",
    maxiter: int = 50,
    C: float = 1.0,
    initial_weights: np.ndarray | None = None,
    learning_rate: float | None = None,
    perturbation: float = 0.2,
    random_state: int | None = 0,
    callback: Callable[[int, np.ndarray, float], None] | None = None,
) -> KernelAlignmentResult:
    """Train the weights of a kernel so that it fits the labels.

    Args:
        kernel: A `QuantumKernel` whose feature map has trainable weights.
        X: The training samples, of shape `(n, n_features)`.
        y: The training labels, of shape `(n,)`.
        loss: The loss to minimise. It is `"svc"` (the dual objective of the
            support vector machine, which is the loss of quantum kernel alignment),
            `"kta"` (minus the kernel-target alignment), or a callable
            `loss(K, y) -> float`.
        optimizer: The optimiser. It can be `"spsa"`, the default, which selects
            Simultaneous Perturbation Stochastic Approximation (SPSA). It can also
            be the name of a `scipy.optimize.minimize` method such as `"COBYLA"`,
            or a callable `optimizer(fun, x0) -> x`.
        maxiter: The number of SPSA iterations, or the `maxiter` option of SciPy.
        C: The regularisation parameter of the support vector machine, for the
            `"svc"` loss.
        initial_weights: The starting point. By default, it is the kernel's own
            weights, or zeros if the kernel has none.
        learning_rate: The initial step size of SPSA. `None` calibrates it so that
            the first step moves the weights by about `2π / 10`. The `"svc"` loss
            grows with the number of samples, so a fixed step that suits one
            dataset size can be far too large for another.
        perturbation: The initial perturbation size of SPSA. The default, `0.2`,
            is the value that qiskit-algorithms uses for its calibration. Much
            smaller values give estimates of the gradient that are too local to
            cross a flat region of the loss.
        random_state: The seed for the perturbation directions of SPSA.
        callback: A function called as `callback(iteration, weights, loss)` after
            every SPSA iteration or SciPy evaluation.

    Returns:
        A [`KernelAlignmentResult`][eigenkit.qml._alignment.KernelAlignmentResult].

    Raises:
        ValueError: If the kernel has no trainable weights, or if `loss` is an
            unknown name.

    Example:
        ```python
        import numpy as np
        from eigenkit import QSVC, QuantumKernel, align_kernel

        def my_map(qc, x, w):
            for q in range(len(x)):
                qc.ry(w[0], q)
                qc.rz(x[q], q)

        rng = np.random.default_rng(0)
        X_train = rng.uniform(0, np.pi, (8, 2))
        y_train = np.array([0, 1] * 4)

        kernel = QuantumKernel(n_features=2, feature_map=my_map, n_weights=1)
        result = align_kernel(kernel, X_train, y_train, maxiter=20)
        model = QSVC(kernel=result.kernel).fit(X_train, y_train)
        ```
    """
    if kernel.n_weights == 0:
        raise ValueError(
            "this kernel has no trainable weights; build its feature map with "
            "weights (n_weights > 0) to align it."
        )
    if isinstance(loss, str):
        if loss not in _LOSSES:
            raise ValueError(
                f"loss must be one of {sorted(_LOSSES)} or a callable; got {loss!r}"
            )
        loss_name = loss

        def loss_fn(K, labels):
            return _LOSSES[loss_name](K, labels, C=C)
    else:
        loss_fn = loss

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    x0 = kernel._weights_or_default(initial_weights).copy()
    evaluations = 0
    history: list[float] = []

    def objective(weights) -> float:
        nonlocal evaluations
        evaluations += 1
        value = float(loss_fn(kernel.compute_kernel_matrix(X, weights=weights), y))
        if not (isinstance(optimizer, str) and optimizer.lower() == "spsa"):
            history.append(value)
            if callback is not None:
                callback(evaluations, np.asarray(weights, dtype=np.float64), value)
        return value

    if isinstance(optimizer, str) and optimizer.lower() == "spsa":
        from eigenkit.core.optimizers import spsa

        result = spsa(
            objective,
            x0,
            maxiter=maxiter,
            learning_rate=learning_rate,
            perturbation=perturbation,
            random_state=random_state,
            callback=callback,
        )
        # The loss is deterministic on a simulator, so keep the best iterate:
        # SPSA's last step can overshoot a minimum it already passed.
        final, history = result.best_x, list(result.history)
        initial_loss, final_loss = result.initial_fun, result.best_fun
    elif isinstance(optimizer, str):
        from scipy.optimize import minimize

        initial_loss = objective(x0)
        result = minimize(objective, x0, method=optimizer, options={"maxiter": maxiter})
        final, final_loss = result.x, float(result.fun)
    else:
        initial_loss = objective(x0)
        final = np.asarray(optimizer(objective, x0), dtype=np.float64)
        final_loss = objective(final)

    final = np.asarray(final, dtype=np.float64)
    return KernelAlignmentResult(
        weights=final,
        kernel=kernel.with_weights(final),
        loss=final_loss,
        initial_loss=initial_loss,
        loss_history=tuple(history),
        n_evaluations=evaluations,
        success=final_loss < initial_loss,
    )
