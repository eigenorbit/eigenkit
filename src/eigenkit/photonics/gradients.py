"""Gradients of photonic circuits: the photonic parameter-shift rule, and finite
differences as a simple reference method.

The qubit rule shifts the angle by `±π/2`. It holds for a Pauli rotation because a
Pauli generator has the eigenvalues `±½`, which give a single frequency. The
generator of a linear-optical component instead has *equally spaced* eigenvalues.
For a phase shifter (and for the `phi` of a beam splitter), the generator is the
number operator `n̂`, whose eigenvalues are integers. The expectation value of any
observable that is diagonal in the Fock basis is then a trigonometric polynomial
in the angle, of degree at most `N`, where `N` is the total number of photons. Its
derivative is given *exactly* by the general shift rule for equally spaced
frequencies (Wierichs et al., 2022), which uses `2N` terms. At `N = 1`, this rule
is the usual `±π/2` rule.

The `theta` of a beam splitter needs more care. Its generator is a generator of
the special unitary group SU(2), on the two modes that the beam splitter acts on.
The eigenvalues of this generator are integers or half-integers, depending on whether
the number of photons `n_ij` in those two modes is even or odd:
`U(theta + 2π) = (-1)^n_ij`. This is a sign that depends on the photon-number
sector, not a global phase. In a circuit with two modes, `n_ij = N` always, so the
frequencies are integers at most `N`, and the rule above applies. With more
modes, the components around the beam splitter create coherent combinations of sectors
with even and odd photon numbers, and half-integer frequencies appear. The rule is
then applied in the angle `theta / 2`, with order `2N` (`4N` evaluations).

This module is the linear-optics counterpart of
`core.provider.parameter_shift_jacobian`. It reuses the `chain_factor` and
`chain_factor_input` functions of the qubit intermediate representation to
turn the derivative with respect to a component angle into derivatives with
respect to the weights and inputs that the angle depends on. The executor reuses
the same `Parameter` expression tree.

Reference: the photonic parameter-shift rule, arXiv:2410.02726 (Phys. Rev. A 111,
032429).

Scope: one sample at a time (1-D weights and inputs), and observables that are
diagonal in the Fock basis.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import replace

import numpy as np

from ..core.ir import (
    Constant,
    Sum,
    chain_factor,
    chain_factor_input,
    collect_input_dependencies,
    collect_weight_dependencies,
)
from .ir import BeamSplitter, FockProbability, Permutation, PhaseShifter, PhotonNumber


def _check_observable(spec, observable, *, min_photons=None) -> None:
    """Reject an observable that can never match ``spec``'s output: it would read
    0.0 with a zero gradient (or fail deep inside the sum) instead of failing here.

    ``min_photons`` is the fewest photons an output may carry; it defaults to
    ``spec.n_photons``, since linear optics conserves photon number. A sampled
    backend that post-selects on fewer detected photons passes its threshold.
    """
    if isinstance(observable, PhotonNumber):
        mode = observable.mode
        if not isinstance(mode, numbers.Integral) or not 0 <= mode < spec.n_modes:
            raise ValueError(
                f"PhotonNumber mode {mode!r} is outside 0..{spec.n_modes - 1}"
            )
    elif isinstance(observable, FockProbability):
        target = observable.target
        low = spec.n_photons if min_photons is None else min_photons
        if (
            len(target) != spec.n_modes
            or any(n < 0 for n in target)
            or not low <= sum(target) <= spec.n_photons
        ):
            photons = (
                f"{spec.n_photons}"
                if low == spec.n_photons
                else f"{low}..{spec.n_photons}"
            )
            raise ValueError(
                f"FockProbability target {target} needs {spec.n_modes} modes and "
                f"{photons} photons (linear optics conserves photon number)"
            )
    else:
        raise TypeError(f"unknown photonic observable: {type(observable).__name__}")


def _expectation(observable, dist) -> float:
    """Expectation of a Fock-diagonal observable over an output distribution."""
    if isinstance(observable, PhotonNumber):
        mode = observable.mode
        return float(sum(prob * state[mode] for state, prob in dist.items()))
    if isinstance(observable, FockProbability):
        return float(dist.get(observable.target, 0.0))
    raise TypeError(f"unknown photonic observable: {type(observable).__name__}")


def _equidistant_shift_terms(order: int) -> list[tuple[float, float]]:
    """(shift, coefficient) pairs of the ``2*order``-term equidistant
    parameter-shift rule for integer frequencies ``{1..order}``. Reduces to the
    ±π/2 rule at ``order == 1``."""
    two_r = 2 * order
    terms = []
    for mu in range(1, two_r + 1):
        x = (2 * mu - 1) * math.pi / two_r
        y = (-1) ** (mu - 1) / (4 * order * math.sin(x / 2) ** 2)
        terms.append((x, y))
    return terms


def _half_angle_shift_terms(order: int) -> list[tuple[float, float]]:
    """(shift, coefficient) pairs for an angle whose frequencies are multiples of
    ½ up to ``order``: the order-``2*order`` rule in ``angle / 2``, rescaled back
    to the angle (shifts doubled, coefficients halved)."""
    return [(2 * x, y / 2) for x, y in _equidistant_shift_terms(2 * order)]


def _parametric_slots(spec):
    """Yield ``(component_index, attribute_name, parameter_expr)`` for every
    differentiable component angle."""
    for index, comp in enumerate(spec.components):
        if isinstance(comp, PhaseShifter):
            yield index, "phi", comp.phi
        elif isinstance(comp, BeamSplitter):
            yield index, "theta", comp.theta
            yield index, "phi", comp.phi
        elif isinstance(comp, Permutation):
            continue
        else:  # pragma: no cover - guarded by the Component union
            raise TypeError(f"unknown photonic component: {type(comp).__name__}")


def _with_shifted_angle(spec, index, attr, shift):
    """A copy of ``spec`` with component ``index``'s ``attr`` angle offset by a
    constant ``shift`` (wrapping the original expression, so it still resolves
    against the same weights/inputs)."""
    comp = spec.components[index]
    shifted = Sum((getattr(comp, attr), Constant(shift)))
    new_comp = replace(comp, **{attr: shifted})
    components = spec.components[:index] + (new_comp,) + spec.components[index + 1 :]
    return replace(spec, components=components)


def photonic_parameter_shift_jacobian(
    executor,
    spec,
    weights,
    inputs,
    observable,
    *,
    order=None,
    need_weight_grad=True,
    need_input_grad=True,
):
    """Compute the exact `d<O>/d(weights)` and `d<O>/d(inputs)` with the photonic
    parameter-shift rule.

    Args:
        executor (Executor): The executor that evaluates each shifted circuit, for
            example `PercevalBackend()`.
        spec (PhotonicCircuitSpec): The circuit to differentiate.
        weights (np.ndarray): The 1-D weight vector.
        inputs (np.ndarray): The 1-D input vector.
        observable (PhotonicObservable): An observable that is diagonal in the Fock
            basis.
        order (int | None): The degree of the trigonometric polynomial to
            reconstruct. The default is the total number of photons, which is a
            safe upper bound: the rule is exact whenever the true degree is at most
            `order`. In a circuit with more than two modes, the `theta` of a beam
            splitter is reconstructed in `theta / 2` with order `2 * order` (see
            the module docstring). `order=1` gives the simple `±π/2` estimate,
            which is in general *wrong* with more than one photon. The tests use it
            to show that the rule with more terms is necessary.
        need_weight_grad (bool): If True, compute the gradient with respect to the
            weights. If False, its shifted evaluations are skipped, and `None` is
            returned in its place.
        need_input_grad (bool): If True, compute the gradient with respect to the
            inputs. If False, its shifted evaluations are skipped, and `None` is
            returned in its place.

    Returns:
        weight_jac (np.ndarray | None): The gradient with respect to the weights,
            of length `spec.n_weights`, or `None` if it was not requested.
        input_jac (np.ndarray | None): The gradient with respect to the inputs, of
            length `spec.n_inputs`, or `None` if it was not requested.

    Raises:
        ValueError: If `observable` can never match the output of the circuit.
    """
    _check_observable(spec, observable)
    order = spec.n_photons if order is None else order
    weight_jac = np.zeros(spec.n_weights) if need_weight_grad else None
    input_jac = np.zeros(spec.n_inputs) if need_input_grad else None
    if order < 1:
        return weight_jac, input_jac  # vacuum / no photons: no dynamics

    integer_terms = _equidistant_shift_terms(order)
    half_angle_terms = _half_angle_shift_terms(order)
    for index, attr, expr in _parametric_slots(spec):
        wdeps = collect_weight_dependencies(expr) if need_weight_grad else ()
        xdeps = collect_input_dependencies(expr) if need_input_grad else ()
        if not wdeps and not xdeps:
            continue  # constant (or unwanted) angle contributes nothing

        # A beam splitter's theta picks up half-integer frequencies once the circuit
        # has spectator modes; in a two-mode circuit the cheaper integer rule is exact.
        half_angle = attr == "theta" and spec.n_modes > 2
        terms = half_angle_terms if half_angle else integer_terms

        # d<O>/d(this component's angle), via the equidistant shift rule.
        d_angle = 0.0
        for shift, coeff in terms:
            shifted = _with_shifted_angle(spec, index, attr, shift)
            dist = executor.compile(shifted).run(weights, inputs)
            d_angle += coeff * _expectation(observable, dist)

        # Chain rule onto the weights/inputs this angle actually depends on.
        for i in wdeps:
            weight_jac[i] += float(chain_factor(expr, i, weights, inputs)) * d_angle
        for j in xdeps:
            input_jac[j] += float(chain_factor_input(expr, j, weights, inputs)) * d_angle
    return weight_jac, input_jac


def finite_difference_jacobian(
    executor,
    spec,
    weights,
    inputs,
    observable,
    *,
    eps=1e-6,
    need_weight_grad=True,
    need_input_grad=True,
):
    """Compute the gradients with central finite differences, as a simple reference.

    It shifts the weights and inputs themselves, so the chain rule is included
    automatically. It is the reference that checks the exact rule, in the same way
    as the NumPy simulator is the reference for qubit circuits.

    Args:
        executor (Executor): The executor that evaluates the circuit, for example
            `PercevalBackend()`.
        spec (PhotonicCircuitSpec): The circuit to differentiate.
        weights (np.ndarray): The 1-D weight vector.
        inputs (np.ndarray): The 1-D input vector.
        observable (PhotonicObservable): An observable that is diagonal in the Fock
            basis.
        eps (float): The step of the central difference. Default: `1e-6`.
        need_weight_grad (bool): If True, compute the gradient with respect to the
            weights. If False, it is skipped, and `None` is returned in its place.
        need_input_grad (bool): If True, compute the gradient with respect to the
            inputs. If False, it is skipped, and `None` is returned in its place.

    Returns:
        weight_jac (np.ndarray | None): The gradient with respect to the weights,
            as for `photonic_parameter_shift_jacobian`.
        input_jac (np.ndarray | None): The gradient with respect to the inputs, as
            for `photonic_parameter_shift_jacobian`.

    Raises:
        ValueError: If `observable` can never match the output of the circuit.
    """
    _check_observable(spec, observable)
    compiled = executor.compile(spec)

    def exp_at(w, x):
        return _expectation(observable, compiled.run(w, x))

    weights = np.asarray(weights, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    weight_jac = np.zeros(len(weights)) if need_weight_grad else None
    input_jac = np.zeros(len(inputs)) if need_input_grad else None
    if need_weight_grad:
        for i in range(len(weights)):
            plus, minus = weights.copy(), weights.copy()
            plus[i] += eps
            minus[i] -= eps
            weight_jac[i] = (exp_at(plus, inputs) - exp_at(minus, inputs)) / (2 * eps)
    if need_input_grad:
        for j in range(len(inputs)):
            plus, minus = inputs.copy(), inputs.copy()
            plus[j] += eps
            minus[j] -= eps
            input_jac[j] = (exp_at(weights, plus) - exp_at(weights, minus)) / (2 * eps)
    return weight_jac, input_jac
