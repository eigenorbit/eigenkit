"""Tests for the `BaseQuantumSolver` contract.

The base class lives in `core`, so solvers in packages that depend on `core`
alone can inherit it, and all four do. The roster below is what keeps that
true: a fifth solver added outside it fails here, and the estimator that must
stay outside is pinned by name.
"""

from __future__ import annotations

import pytest

from eigenkit.algorithms import HHL, IQAE, PortfolioRiskEstimator
from eigenkit.chemistry import VQE
from eigenkit.core import BaseQuantumSolver
from eigenkit.optimization import QAOA


def _solvers():
    """Return one instance of every class that is meant to be a solver."""
    return [
        HHL(),
        IQAE(epsilon=0.05, alpha=0.05, n_qubits=2),
        QAOA(),
        VQE(),
    ]


@pytest.mark.parametrize("solver", _solvers(), ids=lambda s: type(s).__name__)
def test_every_solver_inherits_the_base(solver):
    assert isinstance(solver, BaseQuantumSolver)
    assert issubclass(type(solver), BaseQuantumSolver)


@pytest.mark.parametrize("solver", _solvers(), ids=lambda s: type(s).__name__)
def test_the_base_holds_the_provider(solver):
    # `__init__` sets it once, in the base, for all four.
    assert hasattr(solver, "provider")


def test_each_solver_keeps_its_own_provider_default():
    # Every solver defaults to the backend a bare install already has. IQAE's
    # callable oracle and shot_mode="provider" need provider="pennylane",
    # which the caller asks for.
    assert HHL().provider == "numpy"
    assert QAOA().provider == "numpy"
    assert VQE().provider == "numpy"
    assert IQAE(epsilon=0.05, alpha=0.05, n_qubits=2).provider == "numpy"


def test_the_portfolio_estimator_is_deliberately_not_a_solver():
    # It is an sklearn-style estimator: fit / predict / estimate, no solve.
    estimator = PortfolioRiskEstimator()
    assert not isinstance(estimator, BaseQuantumSolver)
    assert not hasattr(estimator, "solve")


def test_a_subclass_without_solve_cannot_be_instantiated():
    class Forgetful(BaseQuantumSolver):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Forgetful()
