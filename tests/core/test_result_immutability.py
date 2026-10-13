"""Every solver result is frozen, and the roster says which types those are.

The results span `algorithms`, `chemistry`, `optimization` and `core`, and
all of them are frozen. Were some mutable, the same slip — a `=` where `==`
was meant — would raise on one solver's result and silently corrupt
another's.

`BenchmarkResult` is deliberately outside this rule: it is a container with
methods that aggregates on demand, not a value returned from one call.
"""

from __future__ import annotations

import ast
import dataclasses
import io
import os

import numpy as np
import pytest

import eigenkit
from eigenkit.algorithms.hhl import HHLResult
from eigenkit.algorithms.iqae import IQAEResult
from eigenkit.algorithms.portfolio_risk import PortfolioRiskResult
from eigenkit.chemistry.vqe import VQEResult
from eigenkit.core._results import Converged
from eigenkit.core.variational import VariationalResult
from eigenkit.optimization.qaoa import OptimizationResult


RESULT_NAMES = [
    "HHLResult",
    "IQAEResult",
    "PortfolioRiskResult",
    "VQEResult",
    "VariationalResult",
    "OptimizationResult",
]

RESULTS = [
    HHLResult,
    IQAEResult,
    PortfolioRiskResult,
    VQEResult,
    VariationalResult,
    OptimizationResult,
]


def _make(cls):
    """Build an instance with a plausible value for every field."""
    filler = {
        float: 0.0,
        int: 0,
        bool: True,
        str: "",
        "np.ndarray": np.zeros(2),
        "tuple[int, ...]": (0, 1),
        "tuple[float, float]": (0.0, 1.0),
    }
    values = {}
    for f in dataclasses.fields(cls):
        annotation = f.type
        if annotation in filler:
            values[f.name] = filler[annotation]
        elif "ndarray" in str(annotation):
            values[f.name] = np.zeros(2)
        elif "tuple" in str(annotation):
            values[f.name] = (0,)
        elif "list" in str(annotation):
            values[f.name] = []
        elif "bool" in str(annotation):
            values[f.name] = True
        elif "int" in str(annotation):
            values[f.name] = 0
        elif "str" in str(annotation):
            values[f.name] = ""
        else:
            values[f.name] = 0.0
    return cls(**values)


@pytest.mark.parametrize("cls", RESULTS, ids=lambda c: c.__name__)
def test_every_result_type_is_frozen(cls):
    params = cls.__dataclass_params__
    assert params.frozen, f"{cls.__name__} is a result type and must be frozen"


@pytest.mark.parametrize("cls", RESULTS, ids=lambda c: c.__name__)
def test_assigning_to_a_field_raises(cls):
    result = _make(cls)
    field = dataclasses.fields(cls)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, field, 0.0)


@pytest.mark.parametrize("cls", RESULTS, ids=lambda c: c.__name__)
def test_writing_into_an_array_field_raises(cls):
    # `frozen` alone stops rebinding, not mutation of what a field points at,
    # and a result written into keeps reporting the scalars of the run it no
    # longer holds. `__post_init__` closes that by making the arrays read-only.
    result = _make(cls)
    arrays = [
        f.name
        for f in dataclasses.fields(cls)
        if isinstance(getattr(result, f.name), np.ndarray)
    ]
    for name in arrays:
        with pytest.raises(ValueError, match="read-only"):
            getattr(result, name)[0] = 7.0


@pytest.mark.parametrize("cls", RESULTS, ids=lambda c: c.__name__)
def test_no_field_is_a_mutable_container(cls):
    # A list would slip through the read-only treatment, which is why
    # VariationalResult.history is a tuple.
    result = _make(cls)
    mutable = [
        f.name
        for f in dataclasses.fields(cls)
        if isinstance(getattr(result, f.name), (list, dict, set))
    ]
    assert mutable == []


def test_copying_an_array_field_gives_something_writable():
    from eigenkit.algorithms.hhl import HHLResult as R

    result = R(np.zeros(3), 0.0, 0.0, 0.0)
    editable = np.array(result.solution)
    editable[0] = 7.0
    assert editable[0] == 7.0
    assert result.solution[0] == 0.0


@pytest.mark.parametrize("cls", RESULTS, ids=lambda c: c.__name__)
def test_constructing_a_result_leaves_the_caller_array_writable(cls):
    """A result freezes its own copy, never the array it was handed.

    Freezing in place would reach outside the result: a custom optimizer that
    returns the vector it was given hands `minimize_expectation` the caller's
    `initial_parameters` back, and freezing that would stop the caller writing
    to their own array for the rest of the program.
    """
    array_fields = [f.name for f in dataclasses.fields(cls) if "ndarray" in str(f.type)]
    if not array_fields:
        pytest.skip(f"{cls.__name__} carries no array field")

    template = _make(cls)
    mine = {name: np.zeros(2) for name in array_fields}
    result = cls(
        **{
            f.name: mine.get(f.name, getattr(template, f.name))
            for f in dataclasses.fields(cls)
        }
    )

    for name, array in mine.items():
        assert array.flags.writeable, f"{cls.__name__} froze the caller's {name}"
        array[0] = 7.0
        assert not getattr(result, name).flags.writeable
        assert getattr(result, name)[0] == 0.0, f"{cls.__name__}.{name} aliases the input"


def test_a_custom_optimizer_returning_its_input_does_not_freeze_it():
    """The path through which the aliasing is reachable from the public API."""
    from eigenkit.core import Circuit, Observable
    from eigenkit.core.ir import CircuitSpec, Gate, WeightRef
    from eigenkit.core.variational import minimize_expectation

    spec = CircuitSpec(
        n_qubits=1,
        n_weights=1,
        n_inputs=0,
        gates=(Gate("RY", (0,), (WeightRef(0),)),),
    )
    circuit = Circuit(spec, Observable.z(0), "numpy", "numpy")

    initial = np.array([0.3])
    result = minimize_expectation(
        circuit, initial_parameters=initial, optimizer=lambda cost, x0, grad: x0
    )

    assert initial.flags.writeable
    initial[0] = 7.0
    assert not result.optimal_parameters.flags.writeable
    assert result.optimal_parameters[0] == 0.3


def test_no_result_type_outside_the_roster_is_left_mutable():
    """Catch a seventh result type added without being frozen."""
    # Anchored on the package itself, not on the working directory: a
    # cwd-relative "src/eigenkit" walks nothing when pytest runs from
    # anywhere but the repo root, and the guard then passes having checked no
    # classes at all — exactly when a new unfrozen result type would slip by.
    package_root = os.path.dirname(eigenkit.__file__)
    offenders = []
    inspected = []
    for root, dirs, files in os.walk(package_root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            tree = ast.parse(io.open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef) or not node.name.endswith("Result"):
                    continue
                if node.name == "BenchmarkResult":
                    continue  # a container with methods, not a returned value
                inspected.append(node.name)
                frozen = any(
                    kw.arg == "frozen" and ast.unparse(kw.value) == "True"
                    for dec in node.decorator_list
                    for kw in getattr(dec, "keywords", [])
                )
                if not frozen:
                    offenders.append(f"{path}::{node.name}")
    # An empty walk would report success without checking anything, so pin
    # that the roster was actually found.
    assert set(RESULT_NAMES) <= set(inspected), (
        f"only inspected {sorted(inspected)}; the walk missed part of the package"
    )
    assert offenders == []


class TestEqualityAndHashing:
    """`==` works on all six; none of them claims to be hashable.

    A dataclass compares a tuple of its fields, which raises ValueError the
    moment one is a numpy array — four of the six carry one. And
    `frozen=True, eq=True` generates a `__hash__` as well, so those four would
    advertise hashability while `hash()` raises on the array inside.
    `ValueEquality` settles both, the same way for every result type.
    """

    @pytest.mark.parametrize("cls", RESULTS, ids=lambda c: c.__name__)
    def test_two_results_with_the_same_values_are_equal(self, cls):
        assert _make(cls) == _make(cls)

    @pytest.mark.parametrize("cls", RESULTS, ids=lambda c: c.__name__)
    def test_a_difference_in_any_field_is_seen(self, cls):
        first, second = _make(cls), _make(cls)
        for field in dataclasses.fields(cls):
            value = getattr(second, field.name)
            if isinstance(value, np.ndarray):
                changed = value + 1.0
            elif isinstance(value, bool):
                changed = not value
            elif isinstance(value, (int, float)):
                changed = value + 1
            elif isinstance(value, str):
                changed = value + "x"
            elif isinstance(value, tuple):
                changed = tuple(v + 1 for v in value)
            else:
                continue
            other = dataclasses.replace(second, **{field.name: changed})
            assert first != other, f"a change to {cls.__name__}.{field.name} went unseen"

    @pytest.mark.parametrize("cls", RESULTS, ids=lambda c: c.__name__)
    def test_comparing_against_another_type_is_false_not_an_error(self, cls):
        assert (_make(cls) == object()) is False

    @pytest.mark.parametrize("cls", RESULTS, ids=lambda c: c.__name__)
    def test_no_result_claims_to_be_hashable(self, cls):
        # The failure has to be honest: `__hash__ is None`, not a generated
        # hash that raises on the array inside.
        assert cls.__hash__ is None
        with pytest.raises(TypeError, match=cls.__name__):
            hash(_make(cls))

    def test_equality_reaches_inside_an_array_field(self):
        from eigenkit.algorithms.hhl import HHLResult as R

        base = R(np.array([1.0, 2.0]), 2.23, 1e-9, 0.5)
        same = R(np.array([1.0, 2.0]), 2.23, 1e-9, 0.5)
        other = R(np.array([1.0, 9.0]), 2.23, 1e-9, 0.5)
        assert base == same
        assert base != other

    def test_two_runs_of_the_same_solve_compare_equal(self):
        from eigenkit import HHL

        a = np.array([[1.0, 0.5], [0.5, 1.5]])
        b = np.array([1.0, 0.0])
        assert HHL(n_eigenvalue_qubits=3).solve(a, b) == HHL(n_eigenvalue_qubits=3).solve(
            a, b
        )


class TestConvergenceContract:
    """`Converged` answers whether a result has a convergence story at all.

    Four of the six ran an iterative loop and know whether they met their
    target. HHLResult and PortfolioRiskResult do not, and carry neither field
    rather than a stub: HHL runs one circuit, and the portfolio estimator
    aggregates runs it does not own. Asking with `isinstance` is what lets one
    loop read both kinds without special-casing either.
    """

    ITERATIVE = [IQAEResult, VQEResult, VariationalResult, OptimizationResult]
    ONE_SHOT = [HHLResult, PortfolioRiskResult]

    @pytest.mark.parametrize("cls", ITERATIVE, ids=lambda c: c.__name__)
    def test_an_iterative_result_satisfies_the_protocol(self, cls):
        assert isinstance(_make(cls), Converged)

    @pytest.mark.parametrize("cls", ONE_SHOT, ids=lambda c: c.__name__)
    def test_a_one_shot_result_does_not(self, cls):
        assert not isinstance(_make(cls), Converged)

    @pytest.mark.parametrize("cls", ITERATIVE, ids=lambda c: c.__name__)
    def test_both_members_are_readable(self, cls):
        result = _make(cls)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.success, bool)

    def test_iqae_reads_its_round_count_under_the_shared_name(self):
        result = IQAEResult(0.5, (0.4, 0.6), n_rounds=4, n_oracle_calls=99, success=True)
        assert result.n_iterations == 4
        assert result.n_rounds == 4

    def test_issubclass_is_not_available_and_that_is_the_trade(self):
        # Both members are attributes, which is what a caller wants to read.
        # A runtime-checkable protocol supports issubclass only when every
        # member is a method, so isinstance is the supported question.
        with pytest.raises(TypeError, match="non-method members"):
            issubclass(IQAEResult, Converged)

    def test_a_stranger_with_the_right_shape_also_satisfies_it(self):
        # Structural, not declared: a third-party result works without
        # importing anything from us.
        class Foreign:
            n_iterations = 3
            success = True

        assert isinstance(Foreign(), Converged)
