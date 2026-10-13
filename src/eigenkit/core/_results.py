"""Value equality for results that carry arrays.

A dataclass generates `__eq__` by comparing a tuple of its fields. That breaks
the moment one of them is a numpy array: `array == array` returns an array, and
the tuple comparison then raises *ValueError: the truth value of an array with
more than one element is ambiguous*. Four of the six result types carry one, so
the generated `==` would raise on them and work on the other two.

The result types are frozen, and `frozen=True` with the generated `eq=True`
also generates `__hash__`. Those four would then advertise hashability —
`__hash__ is not None`, `isinstance(x, Hashable)` true — while `hash()` would
still raise on the array inside.

`ValueEquality` settles both, the same way for all six: compare by value with
arrays handled properly, and do not pretend to be hashable. Hashing a result
is not useful — nothing in the library does it — and a hash over an array
cannot be made correct cheaply: `np.array_equal` calls `0.0` and `-0.0` equal
while their bytes differ, so a hash built from `tobytes()` would give two equal
objects two different hashes, which is the one thing a hash may never do.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np


def _fields_equal(left, right) -> bool:
    """Compare one field, using `np.array_equal` when either side is an array.

    Args:
        left: The field's value on the left-hand result.
        right: The same field's value on the right-hand result.

    Returns:
        Whether the two are equal.
    """
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(left == right)


class ValueEquality:
    """Compare by field value, arrays included; refuse to hash.

    Mix into a result dataclass declared with `eq=False`, which leaves both
    `__eq__` and `__hash__` to this class. It holds no fields of its own, so it
    does not disturb the dataclass's field order.

    NaN follows the usual rule and is not equal to itself, as it is under a
    generated dataclass `__eq__`.
    """

    def __eq__(self, other: object) -> bool:
        """Return whether every field of two results of the same type matches.

        Args:
            other (object): The value to compare against.

        Returns:
            `True` when both are the same result type and every field is
            equal; `NotImplemented` when the types differ, so Python can fall
            back to the other operand.
        """
        if other.__class__ is not self.__class__:
            return NotImplemented
        return all(
            _fields_equal(getattr(self, f.name), getattr(other, f.name))
            for f in dataclasses.fields(self)
        )

    # A result is a bag of numbers, not a key. Saying so is better than
    # advertising a hash that raises on the array inside.
    __hash__ = None


@runtime_checkable
class Converged(Protocol):
    """A result that ran an iterative loop and knows whether it met its target.

    Four of the six result types do: `IQAEResult`, `VQEResult`,
    `VariationalResult` and `OptimizationResult`. `HHLResult` and
    `PortfolioRiskResult` do not, and deliberately carry neither field —
    HHL runs one circuit rather than a loop, and the portfolio estimator
    aggregates runs it does not own. `isinstance(result, Converged)` is how to
    ask before reaching for either attribute:

    ```python
    if isinstance(result, Converged) and not result.success:
        print(f"stopped after {result.n_iterations} iterations without converging")
    ```

    Both members are attributes rather than methods, which is what a caller
    wants to read. That costs `issubclass()`: a runtime-checkable protocol
    supports it only while every member is a method. `isinstance()` works,
    and it is the question worth asking about a result you were handed.
    """

    n_iterations: int
    success: bool


def freeze_arrays(result: object, *names: str) -> None:
    """Replace each named array field with a read-only copy the result owns.

    Call from a result's `__post_init__`. Freezing in place would be cheaper,
    but the array a result is constructed with is not always its own: a custom
    optimizer that returns the vector it was handed, for instance, gives
    `minimize_expectation` the caller's `initial_parameters` back, and marking
    that read-only would stop the caller writing to their own array long after
    the run. Copying first keeps the freeze inside the result.

    Rebinding goes through `object.__setattr__` because the result types are
    declared `frozen=True`, which is what makes the field itself unassignable.

    Args:
        result: The result instance, mid-construction.
        *names: The array fields to freeze.
    """
    for name in names:
        owned = np.array(getattr(result, name))
        owned.setflags(write=False)
        object.__setattr__(result, name, owned)
