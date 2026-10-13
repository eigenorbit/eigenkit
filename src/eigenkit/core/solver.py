"""The base class every quantum solver inherits.

It lives in `core` rather than in `algorithms` because the four solvers that
inherit it sit in three different packages, and `optimization` and `chemistry`
depend on `core` alone. In `algorithms`, inheriting it would make those two
packages depend on `algorithms`, which they otherwise never touch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .circuit import ProviderSpec


class BaseQuantumSolver(ABC):
    """Base class for quantum solvers that compute a result in a single call.

    Unlike [`BaseQuantumEstimator`][eigenkit.qml.BaseQuantumEstimator], a solver
    has no training phase and no fitted state, and it does not follow the
    scikit-learn estimator interface. It receives a mathematical problem directly
    in `solve()` and returns a structured result object.

    Four classes inherit from it: [`HHL`][eigenkit.algorithms.HHL],
    [`IQAE`][eigenkit.algorithms.IQAE], [`QAOA`][eigenkit.optimization.QAOA] and
    [`VQE`][eigenkit.chemistry.VQE].
    [`PortfolioRiskEstimator`][eigenkit.algorithms.PortfolioRiskEstimator] does
    not, by design. It is a scikit-learn-style estimator with `fit()`,
    `estimate()` and `var_at()`, and it has no `solve()` method. As a result,
    `isinstance(x, BaseQuantumSolver)` tells you whether an object can be given a
    problem directly, and it returns `False` for the estimator.

    Each subclass sets its own `provider` default in its signature and passes it to
    this class. All of them use `"numpy"`. Two of the modes of `IQAE`, a callable
    oracle and `shot_mode="provider"`, need `provider="pennylane"`.

    Args:
        provider: The backend to use: a registered name (`"numpy"`, `"pennylane"`,
            `"qiskit"` and so on), a config instance, or a backend object. The
            default is `"numpy"`, the built-in simulator that the core installation
            includes.

    Example:
        ```python
        from eigenkit.core import BaseQuantumSolver
        from eigenkit.optimization import QAOA

        isinstance(QAOA(), BaseQuantumSolver)   # True
        ```
    """

    def __init__(self, provider: ProviderSpec = "numpy") -> None:
        self.provider = provider

    @abstractmethod
    def solve(self, *args, **kwargs):
        """Compute and return the solution.

        Each subclass declares the signature that its problem needs. `HHL` takes a
        matrix and a right-hand side, `IQAE` takes an oracle, `QAOA` takes a problem,
        and `VQE` takes a Hamiltonian and the number of qubits.
        """
