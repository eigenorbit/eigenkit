"""Optimisers that need no gradient, for objectives that are noisy and expensive.

`spsa` implements Simultaneous Perturbation Stochastic Approximation (SPSA)
(Spall 1992). It estimates the gradient from two evaluations of the objective per
iteration, whatever the number of parameters. This is why it is the usual choice
when every evaluation is a whole kernel matrix or a paid job on hardware. It uses
only NumPy.

Example:
    ```python
    import numpy as np
    from eigenkit.core.optimizers import spsa

    result = spsa(lambda x: float(np.sum((x - 1.0) ** 2)), np.zeros(3), maxiter=200)
    print(np.round(result.x, 1))   # [0.9 0.9 0.9]
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SPSAResult:
    """Outcome of [`spsa()`][eigenkit.core.optimizers.spsa].

    Attributes:
        x: The final parameters.
        fun: The value of the objective at `x`.
        n_iterations: The number of iterations run.
        n_evaluations: The number of evaluations of the objective, including the
            one at the final `x`.
        history: The value of the objective after every iteration, in order.
        success: Whether the final value of the objective is lower than the
            starting value.
        initial_fun: The value of the objective at `x0`.
        best_x: The point with the lowest value of the objective seen, including
            `x0`. The last point of Simultaneous Perturbation Stochastic
            Approximation (SPSA) can move past a minimum that it already reached,
            so on a deterministic objective this is the point to keep.
        best_fun: The value of the objective at `best_x`.
    """

    x: np.ndarray
    fun: float
    n_iterations: int
    n_evaluations: int
    history: tuple[float, ...] = field(default_factory=tuple)
    success: bool = True
    initial_fun: float = float("nan")
    best_x: np.ndarray | None = None
    best_fun: float = float("nan")


def spsa(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    maxiter: int = 100,
    learning_rate: float | None = 0.05,
    perturbation: float = 0.05,
    alpha: float = 0.602,
    gamma: float = 0.101,
    random_state: int | None = 0,
    callback: Callable[[int, np.ndarray, float], None] | None = None,
) -> SPSAResult:
    """Minimise `fun` with Simultaneous Perturbation Stochastic Approximation (SPSA).

    Each iteration draws a random direction `delta` with entries `±1`, evaluates
    `fun` at `x ± c_k * delta`, and takes a step against the resulting estimate of
    the gradient. The step size and the perturbation size decrease as
    `a / (k + 1) ** alpha` and `c / (k + 1) ** gamma`, with the standard exponents
    of Spall.

    Args:
        fun: The objective to minimise. It maps a 1-D parameter vector to a float.
        x0: The starting parameters.
        maxiter: The number of iterations to run.
        learning_rate: The initial step size `a`. `None` calibrates it from a few
            gradient estimates at `x0`, so that the first step moves the
            parameters by about `2π / 10`, as the SPSA of qiskit-algorithms does.
            These evaluations are counted in `n_evaluations`.
        perturbation: The initial perturbation size `c`.
        alpha: The exponent with which the step size decreases.
        gamma: The exponent with which the perturbation size decreases.
        random_state: The seed for the perturbation directions.
        callback: A function called after every iteration as
            `callback(iteration, x, fun(x))`.

    Returns:
        An [`SPSAResult`][eigenkit.core.optimizers.SPSAResult]. The evaluations
            for the history are extra calls to `fun`, made so that the run can be
            plotted, and `n_evaluations` counts them.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x0, dtype=np.float64).copy()
    start = float(fun(x))
    n_evaluations = 1
    if learning_rate is None:
        learning_rate, used = _calibrate(fun, x, perturbation, rng)
        n_evaluations += used
    history: list[float] = []
    best_x, best_fun = x.copy(), start
    for k in range(maxiter):
        a_k = learning_rate / (k + 1) ** alpha
        c_k = perturbation / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=x.shape)
        f_plus = float(fun(x + c_k * delta))
        f_minus = float(fun(x - c_k * delta))
        # delta entries are ±1, so 1 / delta == delta.
        x = x - a_k * (f_plus - f_minus) / (2.0 * c_k) * delta
        value = float(fun(x))
        n_evaluations += 3
        history.append(value)
        if value < best_fun:
            best_x, best_fun = x.copy(), value
        if callback is not None:
            callback(k, x.copy(), value)
    final = history[-1] if history else start
    return SPSAResult(
        x=x,
        fun=final,
        n_iterations=maxiter,
        n_evaluations=n_evaluations,
        history=tuple(history),
        success=final < start,
        initial_fun=start,
        best_x=best_x,
        best_fun=best_fun,
    )


def _calibrate(fun, x, perturbation: float, rng, steps: int = 5):
    """Step size that makes SPSA's first update about `2 * pi / 10` long.

    Returns:
        `(learning_rate, evaluations_used)`.
    """
    target = 2.0 * np.pi / 10.0
    magnitude = 0.0
    for _ in range(steps):
        delta = rng.choice([-1.0, 1.0], size=x.shape)
        diff = float(fun(x + perturbation * delta)) - float(fun(x - perturbation * delta))
        magnitude += abs(diff / (2.0 * perturbation))
    magnitude /= steps
    if magnitude < 1e-12:
        return target, 2 * steps
    return target / magnitude, 2 * steps
