"""Variational optimisation: minimise `<H>(θ)` over the parameters of a circuit.

This is the training engine, independent of any framework, that the Variational
Quantum Eigensolver (VQE) in `chemistry`, the quantum approximate optimisation
algorithm (QAOA) in `optimization`, and any workflow of the form "prepare a
parametrised state, measure `<H>`, optimise `θ`" share. It uses only NumPy and
SciPy. It runs the NumPy forward pass of the core `Circuit` (`compiled.run`) and
its analytic NumPy gradients (`_gradient_jacobians`: the backend's own gradient
when it has one, such as the adjoint method of the NumPy simulator, and the
parameter-shift rule otherwise). So it runs on **any** backend (NumPy,
PennyLane, Qiskit, Braket, CUDA Quantum) without PyTorch. Even on PennyLane it
uses the NumPy path, not the PyTorch QNode. SciPy is imported only at the first
call.

Backends that cannot differentiate (`gradient_method="none"`, for example a
quantum processing unit (QPU) whose configuration leaves `enable_gradients` off)
must use an optimiser that needs no gradient, such as `"COBYLA"`,
`"Nelder-Mead"`, `"Powell"` or `"spsa"`. These only need the forward evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Union

import numpy as np

from ._results import ValueEquality, freeze_arrays


if TYPE_CHECKING:
    from .circuit import Circuit


# scipy.optimize.minimize methods that do not use a jacobian.
GRADIENT_FREE_METHODS = frozenset({"nelder-mead", "powell", "cobyla", "cobyqa"})

# A custom optimizer: `optimizer(fun, x0, jac_or_None) -> final_parameters`,
# or `-> (final_parameters, converged)` when it wants to report convergence
# itself. Without the second form there is no channel for it to say, so
# `minimize_expectation` infers `success` from whether the cost improved.
Optimizer = Union[str, Callable[..., np.ndarray]]


@dataclass(frozen=True, eq=False)
class VariationalResult(ValueEquality):
    """Outcome of [`minimize_expectation()`][eigenkit.core.variational.minimize_expectation].

    The result is read-only. Assigning to a field raises `FrozenInstanceError`, and
    writing into an array field raises `ValueError`. This prevents a mistyped `=`
    (instead of `==`) or an in-place edit from leaving a result whose scalar fields
    describe a different run from the array it holds. To get a writable copy of the
    array, use `np.array(result.field)`.

    Results compare by value, including their arrays. They are not hashable,
    because a result is a collection of numbers and not a key, and a correct hash
    of a float array is expensive to compute.

    Attributes:
        optimal_value: The lowest expectation value reached. When the observable
            is a Hamiltonian, it is the estimated ground-state energy.
        optimal_parameters: The weights that produced this value, of shape
            `(n_weights,)`.
        n_iterations: The number of iterations reported by the SciPy optimiser.
            When there is no such count (a custom callable, `"spsa"`, or a SciPy
            method that does not report one), it is the number of cost
            evaluations, as in `n_evaluations`.
        n_evaluations: The number of circuit evaluations that the optimiser made.
            With a SciPy method it is usually higher than `n_iterations`, because
            a line search costs several evaluations per step. Evaluations made
            only for bookkeeping are not counted, so a custom optimiser that never
            called the cost function reports `0`.
        success: Whether the run converged. A SciPy optimiser reports it. A
            custom callable reports it only if it returns
            `(parameters, converged)`, and otherwise it is inferred from whether
            the final cost is lower than the starting cost. A `False` with a
            plausible energy usually means that the maximum number of iterations
            was reached.
        message: The termination message of the optimiser. For a custom callable
            that reported nothing, it gives the cost before and after the run.
        history: The expectation value at every evaluation, in order. Plot it to
            see whether the run converged or stalled. It is a tuple, so nothing
            can be appended to it after the run.
    """

    optimal_value: float
    optimal_parameters: np.ndarray
    n_iterations: int
    n_evaluations: int
    success: bool
    message: str
    history: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Make the array field read-only.

        Freezing the dataclass prevents a field from being reassigned, but it does not
        prevent writing into the array that the field holds, and a result that has been
        written into still reports the scalar values of the original run. The result
        therefore keeps its own copy, so the array it was built from can still be
        changed by the caller. To get a writable copy, use
        `np.array(result.optimal_parameters)`.
        """
        freeze_arrays(self, "optimal_parameters")


def minimize_expectation(
    circuit: Circuit,
    *,
    initial_parameters: np.ndarray | None = None,
    optimizer: Optimizer = "L-BFGS-B",
    max_iterations: int | None = None,
    tol: float | None = None,
    random_state: int | None = 42,
    callback: Callable[[np.ndarray], None] | None = None,
) -> VariationalResult:
    """Minimise `<H>(θ)` for a circuit whose parameters are all variational.

    Args:
        circuit: A core NumPy `Circuit` whose single observable is the Hamiltonian
            `H` and whose `spec.n_inputs == 0` (every parameter is a variational
            weight). It works with any provider.
        initial_parameters: The starting `θ`, of shape `(n_weights,)`. When it is
            not given, it is drawn at random with the seed `random_state`.
        optimizer: The name of a `scipy.optimize.minimize` method (the default is
            `"L-BFGS-B"`), `"spsa"` for
            [`spsa()`][eigenkit.core.optimizers.spsa], or a callable
            `optimizer(cost, theta0, gradient) -> parameters` for a custom or
            framework optimiser. SPSA needs no gradient: each iteration evaluates
            the circuit twice for the gradient estimate and once more to record the
            history. A callable can instead return `(parameters, converged)` to
            report convergence itself. Otherwise, `success` is inferred from
            whether the final cost is lower than the starting cost, because a
            vector alone says nothing about convergence.
        max_iterations: The maximum number of iterations. It is passed to SciPy as
            `options={"maxiter": ...}`, and to `spsa()` as `maxiter`, which is
            `100` when `max_iterations` is `None`.
        tol: The tolerance, passed to `scipy.optimize.minimize`.
        random_state: The seed for the random starting point, which also seeds the
            perturbations of `"spsa"`. `None` draws an unseeded starting point.
        callback: An optional function called after every iteration with the
            current parameters. It is passed to SciPy, and `"spsa"` also calls it.

    Returns:
        A [`VariationalResult`][eigenkit.core.variational.VariationalResult]
            with the optimal value, the parameters, and the history of the cost at
            every evaluation.

    Example:
        ```python
        from eigenkit import Circuit, Variational
        from eigenkit.core.ir import Observable
        from eigenkit.core.variational import minimize_expectation

        # One qubit, one trainable rotation, H = Z. The minimum is -1.
        circuit = Circuit(
            blocks=[Variational(depth=1)],
            observables=Observable.from_string("Z0"),
            provider="numpy",
        )

        result = minimize_expectation(circuit, random_state=42)

        print(round(result.optimal_value, 6))        # -1.0
        print(result.success, result.n_iterations)   # True 4
        ```
    """
    spec = circuit.spec
    if spec.n_inputs != 0:
        raise ValueError(
            "minimize_expectation expects a pure variational circuit "
            f"(n_inputs == 0); got n_inputs={spec.n_inputs}. Data-dependent "
            "training is the torch ring's job."
        )
    if len(circuit.observables) != 1:
        raise ValueError(
            "minimize_expectation expects exactly one observable (the "
            f"Hamiltonian H); got {len(circuit.observables)}."
        )

    n_weights = spec.n_weights
    if initial_parameters is None:
        theta0 = np.random.default_rng(random_state).standard_normal(n_weights)
    else:
        theta0 = np.asarray(initial_parameters, dtype=np.float64)
        if theta0.shape != (n_weights,):
            raise ValueError(
                f"initial_parameters must have shape ({n_weights},); got {theta0.shape}"
            )

    empty_inputs = np.empty(0, dtype=np.float64)
    history: list[float] = []
    eval_count = 0

    def cost(theta) -> float:
        nonlocal eval_count
        eval_count += 1
        value = float(
            circuit.compiled.run(np.asarray(theta, dtype=np.float64), empty_inputs)[0]
        )
        history.append(value)
        return value

    def evaluate(theta) -> float:
        """Run the circuit without counting it as one of the optimizer's calls.

        `history` and `n_evaluations` describe what the optimizer did. The
        bookkeeping evaluations around it — the starting cost and the final
        cost on the custom-optimizer branch — are ours, not its.

        Args:
            theta: Parameter vector to evaluate.

        Returns:
            The expectation value at `theta`.
        """
        return float(
            circuit.compiled.run(np.asarray(theta, dtype=np.float64), empty_inputs)[0]
        )

    def gradient(theta) -> np.ndarray:
        weight_jac, _ = circuit._gradient_jacobians(
            np.asarray(theta, dtype=np.float64), empty_inputs
        )
        return weight_jac[0]

    no_gradients = circuit.provider.gradient_method == "none"

    if isinstance(optimizer, str) and optimizer.lower() == "spsa":
        from .optimizers import spsa

        def optimizer(fun, x0, _gradient):
            result = spsa(
                fun,
                x0,
                maxiter=100 if max_iterations is None else max_iterations,
                learning_rate=None,
                random_state=random_state,
                callback=None if callback is None else lambda _k, x, _f: callback(x),
            )
            return result.best_x, result.best_fun < result.initial_fun

    if callable(optimizer):
        returned = optimizer(cost, theta0, None if no_gradients else gradient)
        reported: bool | None = None
        if isinstance(returned, tuple) and len(returned) == 2:
            candidate, flag = returned
            if isinstance(flag, (bool, np.bool_)):
                returned, reported = candidate, bool(flag)
        final = np.asarray(returned, dtype=np.float64)
        if final.shape != (n_weights,):
            raise ValueError(
                "custom optimizer must return a parameter vector of shape "
                f"({n_weights},), optionally paired with a convergence flag; "
                f"got {final.shape}"
            )
        # A callable that returns only a vector has no way to report
        # convergence, so `success=True` would be our assumption rather than
        # its answer — and it would read as true for an optimizer that did
        # nothing, and for one that made the cost worse. Infer it instead: did
        # the cost improve on where it started? Neither evaluation counts as
        # the optimizer's work.
        start_value = evaluate(theta0)
        final_value = evaluate(final)
        converged = reported if reported is not None else final_value < start_value
        return VariationalResult(
            optimal_value=final_value,
            optimal_parameters=final,
            n_iterations=eval_count,
            n_evaluations=eval_count,
            success=converged,
            message=(
                "custom optimizer reported convergence"
                if reported is True
                else "custom optimizer reported no convergence"
                if reported is False
                else f"custom optimizer returned; cost {start_value:.6g} -> "
                f"{final_value:.6g}"
            ),
            history=tuple(history),
        )

    gradient_free = optimizer.lower() in GRADIENT_FREE_METHODS
    if no_gradients and not gradient_free:
        raise ValueError(
            f"backend '{circuit.provider.name}' cannot differentiate "
            f"(gradient_method='none'); use a gradient-free optimizer "
            f"(e.g. 'COBYLA') instead of {optimizer!r}, or set "
            f"enable_gradients=True in the device config to opt in."
        )

    from scipy.optimize import minimize as scipy_minimize

    options = {"maxiter": max_iterations} if max_iterations is not None else None
    result = scipy_minimize(
        cost,
        theta0,
        method=optimizer,
        jac=None if gradient_free else gradient,
        tol=tol,
        options=options,
        callback=callback,
    )
    return VariationalResult(
        optimal_value=float(result.fun),
        optimal_parameters=np.asarray(result.x, dtype=np.float64),
        n_iterations=int(getattr(result, "nit", eval_count)),
        n_evaluations=eval_count,
        success=bool(result.success),
        message=str(result.message),
        history=tuple(history),
    )
