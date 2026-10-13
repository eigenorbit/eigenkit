"""A pure-numpy ridge (Tikhonov) linear readout — the classic QRC head.

Frozen-reservoir QRC trains *only* the readout, and the canonical readout is a
linear layer fit by ridge regression. That has a closed-form solution, so no
iterative optimizer (and no torch/sklearn) is needed — this stays framework-free
and lives in the numpy domain ring. For a trainable non-linear readout, use the
estimator layer (an MLP head), not this.
"""

from __future__ import annotations

import numpy as np


class LinearReadout:
    """Ridge regression in closed form, with a bias term.

    It solves `W = (XᵀX + λI)⁻¹ XᵀY`, and the bias column is not regularised. `Y`
    can have one output or several. `ridge` is the strength `λ` of the L2 penalty,
    which adds `λ‖W‖²` to the squared error of the fit. A small positive value
    also keeps the solution numerically stable when the reservoir features are
    collinear.

    Args:
        ridge: The strength `λ` of the L2 penalty. Raise it when the readout
            overfits or the feature matrix is nearly singular, and lower it when the
            readout underfits. The default is `1e-06`.
    """

    def __init__(self, ridge: float = 1e-6):
        if ridge < 0:
            raise ValueError("ridge must be >= 0")
        self.ridge = ridge
        self.coef_: np.ndarray | None = None  # (n_features + 1, n_targets)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearReadout":
        """Fit the readout to `X` and `y`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.
            y (np.ndarray): The targets, of shape `(n_samples,)` for one output, or
                `(n_samples, n_outputs)` for several.

        Returns:
            This readout, so that calls can be chained.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        single = y.ndim == 1
        if single:
            y = y[:, None]
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # bias column
        gram = Xb.T @ Xb
        reg = self.ridge * np.eye(Xb.shape[1])
        reg[-1, -1] = 0.0  # do not penalize the bias
        self.coef_ = np.linalg.solve(gram + reg, Xb.T @ y)
        self._single_output = single
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target for every sample in `X`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            One prediction per sample, with one column per output when the readout was
                fitted on several outputs.
        """
        if self.coef_ is None:
            raise RuntimeError("LinearReadout is not fitted; call fit() first")
        X = np.asarray(X, dtype=np.float64)
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        out = Xb @ self.coef_
        return out[:, 0] if self._single_output else out
