"""General-purpose quantum algorithms.

This package contains three algorithms:

- [`HHL`][eigenkit.algorithms.HHL] solves linear systems with the algorithm of
  Harrow, Hassidim and Lloyd (HHL).
- [`IQAE`][eigenkit.algorithms.IQAE] runs Iterative Quantum Amplitude
  Estimation (IQAE).
- [`PortfolioRiskEstimator`][eigenkit.algorithms.PortfolioRiskEstimator]
  estimates the Value at Risk (VaR) and the Conditional Value at Risk (CVaR) of
  a portfolio with IQAE.

`HHL` and `IQAE` inherit from
[`BaseQuantumSolver`][eigenkit.core.solver.BaseQuantumSolver], the base class
for solvers that work in a single call and keep no fitted state. The Quantum
Approximate Optimization Algorithm (`QAOA`) and the Variational Quantum
Eigensolver (`VQE`) also inherit from it, in their own packages. The base class
is defined in `core` because `QAOA` and `VQE` depend only on `core`.
`PortfolioRiskEstimator` does not inherit from it, by design. It is a
scikit-learn-style estimator, registered in the estimator registry of
`eigenkit.qml` so that it can be saved and loaded.
It therefore needs scikit-learn and is imported on first use; `HHL` and `IQAE`
need only the core install.
"""

from typing import TYPE_CHECKING

from .hhl import HHL, HHLResult
from .iqae import IQAE, IQAEResult


if TYPE_CHECKING:
    from .portfolio_risk import PortfolioRiskEstimator, PortfolioRiskResult


def __getattr__(name: str):
    if name in ("PortfolioRiskEstimator", "PortfolioRiskResult"):
        from . import portfolio_risk

        return getattr(portfolio_risk, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "HHL",
    "HHLResult",
    "IQAE",
    "IQAEResult",
    "PortfolioRiskEstimator",
    "PortfolioRiskResult",
]
