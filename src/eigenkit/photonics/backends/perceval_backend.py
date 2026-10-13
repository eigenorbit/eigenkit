"""The Perceval executor for photonic circuits (experimental).

It translates a
[`PhotonicCircuitSpec`][eigenkit.photonics.ir.PhotonicCircuitSpec] into a
Perceval `Circuit`, and computes the output distribution of photon numbers with an
exact Perceval simulator, by default simulation of linear optical systems (SLOS).
It is the linear-optics counterpart of a qubit `Provider`, but it does **not**
implement the qubit `Provider` protocol: there is no `CircuitSpec`, no Pauli
`Observable` and no array of expectation values. This is why photonics is a
separate package.

`perceval` is imported only *inside* the methods, when they run, so importing this
module (and therefore `import eigenkit`) never imports the Perceval software
development kit (SDK). The data module `photonics.ir` never uses Perceval at all.

The work is split between compilation and evaluation, as in the qubit providers:
`compile()` is called once, and `run(weights, inputs)` at every call. For now, the
Perceval circuit is built again at every call, after the `Parameter` expressions
are evaluated to numbers. Building it once with symbolic `pcvl.P` parameters, and
setting their values at each call, is not done yet. (This is the saving that the
qubit providers get from the split between compilation and evaluation.)
"""

from __future__ import annotations

import functools

import numpy as np

from ...core.ir import resolve
from ..executor import register_lazy
from ..ir import BeamSplitter, Permutation, PhaseShifter, PhotonicCircuitSpec


FockDistribution = dict  # {tuple[int, ...] photon-counts -> float probability}


@functools.cache
def _check_backend_name(name: str) -> None:
    """Reject names perceval does not know (it would silently fall back to SLOS)
    and sampling-only backends, which have no ``prob_distribution``."""
    import perceval as pcvl

    available = pcvl.BackendFactory.list()
    if name not in available:
        raise ValueError(
            f"unknown perceval backend {name!r}; available: {sorted(available)}"
        )
    if not hasattr(pcvl.BackendFactory.get_backend(name), "prob_distribution"):
        raise ValueError(
            f"perceval backend {name!r} only samples, but PercevalBackend returns the "
            "output probability distribution; use 'SLOS' (the default) or 'Naive'"
        )


def _check_lengths(spec, weights, inputs) -> None:
    """One sample per call: ``weights`` / ``inputs`` must be 1-D and match the spec,
    or extra entries would be silently ignored by ``resolve``."""
    for label, values, expected in (
        ("weights", weights, spec.n_weights),
        ("inputs", inputs, spec.n_inputs),
    ):
        shape = np.shape(values)
        if shape != (expected,):
            raise ValueError(
                f"{label} has shape {shape} but the spec needs ({expected},); photonic "
                "circuits take one 1-D sample per call"
            )


def build_perceval_circuit(spec, weights, inputs):
    """Translate a `PhotonicCircuitSpec` into a Perceval `Circuit`.

    The `Parameter` angles are evaluated to numbers first. The local and the
    remote executors both use this function, so they build exactly the same
    circuit. (The live test of the remote executor compares sampled and exact
    results on the *same* circuit.)

    Args:
        spec (PhotonicCircuitSpec): The circuit to translate.
        weights (np.ndarray): The 1-D weight vector used to evaluate the angles.
        inputs (np.ndarray): The 1-D input vector used to evaluate the angles.

    Returns:
        circuit (perceval.Circuit): The circuit, on `spec.n_modes` modes.

    Raises:
        ValueError: If `weights` or `inputs` is not 1-D with the length that the
            spec declares.
    """
    import perceval as pcvl

    _check_lengths(spec, weights, inputs)
    circ = pcvl.Circuit(spec.n_modes)
    for comp in spec.components:
        if isinstance(comp, PhaseShifter):
            circ.add(comp.mode, pcvl.PS(float(resolve(comp.phi, weights, inputs))))
        elif isinstance(comp, BeamSplitter):
            theta = float(resolve(comp.theta, weights, inputs))
            phi = float(resolve(comp.phi, weights, inputs))
            # Our single `phi` maps to perceval's top-right phase; the exact BS unitary
            # (Rx convention, R = cos^2(theta/2)) is pinned by the parity tests.
            circ.add(tuple(comp.modes), pcvl.BS(theta=theta, phi_tr=phi))
        elif isinstance(comp, Permutation):
            circ.add(0, pcvl.PERM(list(comp.perm)))
        else:  # pragma: no cover - guarded by the Component union
            raise TypeError(f"unknown photonic component: {type(comp).__name__}")
    return circ


def fock_tuple(state, n_modes: int) -> tuple[int, ...]:
    """Convert a Perceval `BasicState`, or any indexable object, to a tuple of photon counts.

    Args:
        state (BasicState | Sequence[int]): The number of photons in each mode,
            indexed by mode.
        n_modes (int): The number of modes to read.

    Returns:
        The number of photons in each mode, as a tuple of integers.
    """
    return tuple(int(state[i]) for i in range(n_modes))


class PercevalCompiled:
    """A reusable handle to a compiled photonic circuit, cheap to call many times.

    It is the linear-optics counterpart of `CompiledCircuit`.
    `PercevalBackend.compile` builds it.

    Args:
        spec (PhotonicCircuitSpec): The circuit that this handle evaluates.
        backend_name (str): The Perceval backend, as for `PercevalBackend`.
            Default: `"SLOS"`.

    Raises:
        ValueError: If `backend_name` is unknown or only samples.
    """

    def __init__(self, spec: PhotonicCircuitSpec, backend_name: str = "SLOS") -> None:
        _check_backend_name(backend_name)
        self._spec = spec
        self._backend_name = backend_name

    def _build_circuit(self, weights, inputs):
        return build_perceval_circuit(self._spec, weights, inputs)

    def run(self, weights, inputs) -> FockDistribution:
        """Return the output distribution of photon numbers.

        The distribution is a dictionary `{photon_counts_tuple: probability}`. Fock
        states with probability zero may be left out, because Perceval's
        `prob_distribution` is sparse.

        Args:
            weights (np.ndarray): The 1-D weight vector, of length `spec.n_weights`.
            inputs (np.ndarray): The 1-D input vector, of length `spec.n_inputs`.

        Raises:
            ValueError: If `weights` or `inputs` has the wrong shape.
        """
        import perceval as pcvl

        circ = self._build_circuit(weights, inputs)
        backend = pcvl.BackendFactory.get_backend(self._backend_name)
        backend.set_circuit(circ)
        backend.set_input_state(pcvl.BasicState(list(self._spec.input_state)))
        dist = backend.prob_distribution()
        return {
            fock_tuple(state, self._spec.n_modes): float(prob)
            for state, prob in dist.items()
        }

    def expectation(self, weights, inputs, observable) -> float:
        """Return the expectation value of a `PhotonicObservable`, as a single number.

        The observable is diagonal in the Fock basis.

        Args:
            weights (np.ndarray): The 1-D weight vector.
            inputs (np.ndarray): The 1-D input vector.
            observable (PhotonicObservable): A `PhotonNumber` or a `FockProbability`.

        Raises:
            ValueError: If `observable` can never match the output of the circuit.
        """
        from ..gradients import _check_observable, _expectation

        _check_observable(self._spec, observable)
        return _expectation(observable, self.run(weights, inputs))

    def jacobian(
        self,
        weights,
        inputs,
        observable,
        *,
        method="parameter_shift",
        need_weight_grad=True,
        need_input_grad=True,
    ):
        """Return `(d<O>/d weights, d<O>/d inputs)` for an observable.

        The observable is diagonal in the Fock basis. Only one sample is accepted
        (1-D weights and inputs).

        Args:
            weights (np.ndarray): The 1-D weight vector.
            inputs (np.ndarray): The 1-D input vector.
            observable (PhotonicObservable): A `PhotonNumber` or a `FockProbability`.
            method (str): `"parameter_shift"` for the exact photonic parameter-shift
                rule (in `gradients.py`), or `"finite_difference"` for central finite
                differences, the reference method. Default: `"parameter_shift"`.
            need_weight_grad (bool): If True, compute the gradient with respect to
                the weights. If False, it is skipped, and `None` is returned in its
                place.
            need_input_grad (bool): If True, compute the gradient with respect to
                the inputs. If False, it is skipped, and `None` is returned in its
                place.

        Raises:
            ValueError: If `method` is neither of these two values.
        """
        from .. import gradients

        executor = PercevalBackend(self._backend_name)
        weights = np.asarray(weights, dtype=float)
        inputs = np.asarray(inputs, dtype=float)
        flags = {"need_weight_grad": need_weight_grad, "need_input_grad": need_input_grad}
        if method == "parameter_shift":
            return gradients.photonic_parameter_shift_jacobian(
                executor, self._spec, weights, inputs, observable, **flags
            )
        if method == "finite_difference":
            return gradients.finite_difference_jacobian(
                executor, self._spec, weights, inputs, observable, **flags
            )
        raise ValueError(f"unknown gradient method: {method!r}")


class PercevalBackend:
    """The photonic executor that runs on a local Perceval simulator.

    `compile(spec)` returns a `PercevalCompiled`.

    Args:
        backend_name (str): A Perceval backend that computes the full output
            distribution. `"SLOS"` (the default) and `"Naive"` are exact, while
            `"MPS"` and `"NaiveApprox"` approximate the distribution, and so the
            gradients too. Backends that only sample, such as
            `"CliffordClifford2017"`, are rejected, because `run` returns
            probabilities, not samples.

    Raises:
        ValueError: If `backend_name` is unknown or only samples.
    """

    def __init__(self, backend_name: str = "SLOS") -> None:
        _check_backend_name(backend_name)
        self.backend_name = backend_name

    def compile(self, spec: PhotonicCircuitSpec) -> PercevalCompiled:
        """Wrap `spec` in a reusable handle. Perceval runs at each call of the handle.

        Args:
            spec (PhotonicCircuitSpec): The circuit to compile.

        Returns:
            The `PercevalCompiled` handle for `spec`.
        """
        return PercevalCompiled(spec, backend_name=self.backend_name)


register_lazy("perceval", PercevalBackend)
