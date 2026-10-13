"""One name and one default for seeding, across the whole public API.

Every seeded public API takes `random_state`, the name the sklearn-compatible
estimators use, and every default is 42, with `None` meaning "do not pin this".

`eigenkit.benchmark` is exempt on purpose: its `seed` is a run index
that labels the rows of `result.table()` and selects among them in
`best_model(seed="last")`.
"""

from __future__ import annotations

import ast
import inspect
import io
import os

import numpy as np
import pytest

from eigenkit.algorithms import IQAE, PortfolioRiskEstimator
from eigenkit.chemistry import VQE
from eigenkit.core import minimize_expectation
from eigenkit.datasets import load_dataset
from eigenkit.datasets._loaders import load_sklearn_dataset, load_url_dataset
from eigenkit.datasets._synthetic import (
    generate_checkerboard,
    generate_circles,
    generate_moons,
    generate_regression,
    generate_timeseries,
)
from eigenkit.optimization import QAOA


SRC = os.path.join("src", "eigenkit")

SEEDED_FUNCTIONS = [
    load_dataset,
    load_sklearn_dataset,
    load_url_dataset,
    generate_moons,
    generate_circles,
    generate_checkerboard,
    generate_regression,
    generate_timeseries,
    minimize_expectation,
]
CONSTRUCTORS = [QAOA, VQE, IQAE, PortfolioRiskEstimator]


def _seed_parameters():
    """Yield every `seed` / `random_state` parameter declared under `src/`."""
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name).replace("\\", "/")
            tree = ast.parse(io.open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                args = node.args
                for arg in args.posonlyargs + args.args + args.kwonlyargs:
                    if arg.arg in ("seed", "random_state"):
                        yield path, node.name, arg.arg


class TestOneName:
    def test_nothing_outside_benchmark_declares_seed(self):
        offenders = [
            f"{path}::{func}"
            for path, func, arg in _seed_parameters()
            if arg == "seed" and "/benchmark/" not in path
        ]
        assert offenders == []

    def test_the_benchmark_run_index_keeps_its_own_word(self):
        # Not an RNG parameter: it labels result rows and selects among them.
        kept = [
            f"{path}::{func}"
            for path, func, arg in _seed_parameters()
            if arg == "seed" and "/benchmark/" in path
        ]
        assert kept, "the benchmark should still speak of a seed"


class TestOneDefault:
    @pytest.mark.parametrize("func", SEEDED_FUNCTIONS, ids=lambda f: f.__name__)
    def test_a_seeded_function_defaults_to_42(self, func):
        assert inspect.signature(func).parameters["random_state"].default == 42

    @pytest.mark.parametrize("cls", CONSTRUCTORS, ids=lambda c: c.__name__)
    def test_a_constructor_defaults_to_42(self, cls):
        assert inspect.signature(cls.__init__).parameters["random_state"].default == 42

    def test_no_default_is_none(self):
        for obj in SEEDED_FUNCTIONS + [c.__init__ for c in CONSTRUCTORS]:
            parameter = inspect.signature(obj).parameters["random_state"]
            assert parameter.default is not None


class TestNoneIsAccepted:
    def test_none_leaves_a_dataset_split_unseeded(self):
        first = load_dataset("moons", random_state=None)
        second = load_dataset("moons", random_state=None)
        assert not np.array_equal(first["X_train"], second["X_train"])

    def test_an_integer_pins_a_dataset_split(self):
        first = load_dataset("moons", random_state=3)
        second = load_dataset("moons", random_state=3)
        assert np.array_equal(first["X_train"], second["X_train"])

    def test_none_is_accepted_by_every_constructor(self):
        QAOA(random_state=None)
        VQE(random_state=None)
        IQAE(epsilon=0.05, alpha=0.05, n_qubits=2, random_state=None)
        PortfolioRiskEstimator(random_state=None)


class TestOnlyRandomStateIsAccepted:
    def test_seed_is_not_a_parameter(self):
        with pytest.raises(TypeError, match="seed"):
            load_dataset("moons", seed=7)
        with pytest.raises(TypeError, match="seed"):
            QAOA(seed=5)
