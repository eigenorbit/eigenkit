"""The executor protocol for photonic circuits, the linear-optics counterpart of
`core.provider`.

A photonic **executor** compiles a
[`PhotonicCircuitSpec`][eigenkit.photonics.ir.PhotonicCircuitSpec] into a
`Compiled` handle that can be reused. This is a *different contract* from the
qubit `core.provider.Provider`: there is no `CircuitSpec`, no Pauli `Observable`
and no array of expectation values. This is why photonics is a separate package,
and not a qubit backend.

The executors themselves are in `photonics/backends/` (for example, the Perceval
backend, which uses Perceval's simulation of linear optical systems (SLOS)).
This module does not depend on any framework (neither Perceval nor PyTorch). It
defines only the contract, and a small registry of names whose executors are
built on first use, like the one in `core.provider`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Compiled(Protocol):
    """A handle to a compiled photonic circuit, cheap to call many times.

    Its contents are internal to the executor that built it.
    """

    def run(self, weights, inputs) -> dict:
        """Evaluate the circuit for one sample.

        Args:
            weights (np.ndarray): The 1-D weight vector, of length `spec.n_weights`.
            inputs (np.ndarray): The 1-D input vector, of length `spec.n_inputs`.

        Returns:
            The output Fock distribution, as a dictionary
                `{photon_counts_tuple: probability}`.
        """
        ...

    def expectation(self, weights, inputs, observable) -> float:
        """Evaluate one observable that is diagonal in the Fock basis, for one sample.

        Args:
            weights (np.ndarray): The 1-D weight vector.
            inputs (np.ndarray): The 1-D input vector.
            observable (PhotonicObservable): A `PhotonNumber` or a `FockProbability`.

        Returns:
            The expectation value of `observable`.
        """
        ...

    def jacobian(
        self,
        weights,
        inputs,
        observable,
        *,
        method: str = "parameter_shift",
        need_weight_grad: bool = True,
        need_input_grad: bool = True,
    ):
        """Compute the gradient of one observable with respect to weights and inputs.

        Args:
            weights (np.ndarray): The 1-D weight vector.
            inputs (np.ndarray): The 1-D input vector.
            observable (PhotonicObservable): A `PhotonNumber` or a `FockProbability`.
            method (str): How to differentiate, for example `"parameter_shift"`.
            need_weight_grad (bool): If True, compute the gradient with respect to
                the weights. If False, the implementation may skip it and return
                `None` in its place. The PyTorch bridge takes this value from
                `ctx.needs_input_grad`.
            need_input_grad (bool): The same as `need_weight_grad`, for the gradient
                with respect to the inputs.

        Returns:
            weight_jac (np.ndarray | None): The derivatives `d<O>/d weights`, or
                `None` if they were not requested.
            input_jac (np.ndarray | None): The derivatives `d<O>/d inputs`, or `None`
                if they were not requested.
        """
        ...


@runtime_checkable
class Executor(Protocol):
    """An object that compiles a `PhotonicCircuitSpec` into a `Compiled` handle.

    The handle can be reused for many evaluations.
    """

    def compile(self, spec) -> Compiled:
        """Prepare `spec` so that it can be evaluated many times.

        Args:
            spec (PhotonicCircuitSpec): The circuit to compile.

        Returns:
            A `Compiled` handle for `spec`.
        """
        ...


# ---- minimal lazy name registry (mirrors core.provider) ----

_LAZY = {}
_CACHE = {}


def register_lazy(name, loader):
    """Register a function that builds an executor, without building it yet.

    The first registration of a name is kept, and later ones are ignored.

    Args:
        name (str): The name that `get` looks up.
        loader (Callable[[], Executor]): A function with no arguments that builds
            the executor. It is called the first time the name is looked up.
    """
    _LAZY.setdefault(name, loader)


def get(name):
    """Return the executor registered under `name`, building it on first use.

    Args:
        name (str): A name passed to `register_lazy`.

    Returns:
        executor (Executor): The executor. It is kept after the first call, and the
            same object is returned by later calls.

    Raises:
        KeyError: If nothing is registered under `name`.
    """
    if name not in _CACHE:
        if name not in _LAZY:
            raise KeyError(
                f"no photonic executor registered as {name!r}; "
                f"registered: {sorted(_LAZY)}"
            )
        _CACHE[name] = _LAZY[name]()
    return _CACHE[name]


def registered_names():
    """Return the names of all registered executors, in sorted order."""
    return sorted(set(_LAZY) | set(_CACHE))
