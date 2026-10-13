"""The core `Circuit`, which needs no framework, compiles once and accepts batches.

- The circuit is compiled once, in `__init__`, and every call (`__call__`) reuses
  the compiled handle.
- `inputs` can be 1-D (one sample) or 2-D (a batch, with the samples along the
  first axis). The array is always passed with its shape unchanged to
  `compiled.run` or `compiled.run_torch`, and the backend handles the shape
  itself. Backends that cannot run a batch at once use `batch_call`.
- The observables can be given in several forms. `None` measures `Z` on every
  qubit. Otherwise, pass an `Observable`, a list of them, or a `PauliExpr`.

This module holds the NumPy path: forward values computed with NumPy only, and
analytic jacobians from the parameter-shift rule or the chain rule. Automatic
differentiation in a framework (PyTorch now, JAX later) is handled by a
registered *interface* (see `interfaces.py`), so this module imports no framework
and stays in the lightweight core. The PyTorch `nn.Module` wrapper, `TorchCircuit`,
is in the PyTorch package (`eigenkit.torch`).
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, Union

import numpy as np

from .draw import draw_text
from .interfaces import get_interface
from .ir import (
    MULTI_FREQUENCY_GATES,
    CircuitSpec,
    Observable,
    chain_factor,
    chain_factor_input,
    collect_input_dependencies,
    collect_weight_dependencies,
    find_input_shift_unsafe_gates,
    find_parameter_shift_unsafe_gates,
    inputs_needing_four_term_shift,
    shift_gate_angle,
    weights_needing_four_term_shift,
)
from .provider import (
    FOUR_TERM_RULE,
    TWO_TERM_RULE,
    CompiledCircuit,
    Provider,
    parameter_shift_input_jacobian,
    parameter_shift_jacobian,
    resolve_provider,
    validate_circuit,
)


Interface = Literal["numpy", "torch"]
ObservableLike = Union[Observable, Sequence[Observable], "PauliExpr", None]  # noqa: F821
# `provider=` accepts a string (registry lookup), a config with
# .make_backend() (e.g. PennyLaneConfig), or a Backend instance (advanced).
ProviderSpec = Union[str, Provider, object]


def _to_numpy(value) -> np.ndarray:
    """Coerce a numpy array or framework tensor to a float64 numpy array.

    Duck-typed on `.detach` (torch / jax-style) so the core never imports a
    framework just to read values out of a tensor.
    """
    detach = getattr(value, "detach", None)
    if detach is not None:
        return detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def _normalize_observables(
    observables: ObservableLike,
    n_qubits: int,
) -> tuple[list[Observable], bool]:
    """Return `(list_of_Observable, was_scalar)`.

    `was_scalar` lets the outer layer squeeze a 1-element result back to a
    scalar.
    """
    from .observables import PauliExpr  # local import to avoid a cycle

    if observables is None:
        return ([Observable.z(q) for q in range(n_qubits)], False)
    if isinstance(observables, PauliExpr):
        return ([observables.to_observable()], True)
    if isinstance(observables, Observable):
        return ([observables], True)
    result: list[Observable] = []
    for obs in observables:
        if isinstance(obs, PauliExpr):
            result.append(obs.to_observable())
        elif isinstance(obs, Observable):
            result.append(obs)
        else:
            raise TypeError(f"not an observable: {type(obs)}")
    return (result, False)


def _reject_empty(observables: list[Observable]) -> None:
    """Refuse an observable with no Pauli terms.

    An empty `Observable` is a legitimate value — `jordan_wigner` returns one
    with an offset when the operator reduces to a multiple of the identity —
    but it is not something a circuit can measure, and the backends do not
    agree on what to answer: three would return 0.0 and the rest would raise
    three different exception types. It is rejected once here rather than
    eight times, differently, further down.

    Args:
        observables (list[Observable]): The normalized observables.

    Raises:
        ValueError: If any of them carries no Pauli terms.
    """
    for position, observable in enumerate(observables):
        if not observable.terms:
            where = f" at index {position}" if len(observables) > 1 else ""
            raise ValueError(
                f"observable{where} has no Pauli terms; there is nothing to "
                f"measure. A constant operator carries its value beside the "
                f"observable — the chemistry entry points return it as `offset`."
            )


class Circuit:
    """A compiled circuit, ready to evaluate.

    It holds a `CircuitSpec`, the observables to measure and a compiled backend
    handle. The circuit is compiled once, when the instance is created, and every
    call reuses the compiled handle. The instance does not store weights: you pass
    the `weights` at each call.

    This is the NumPy interface. The top-level `Circuit(...)` function of the
    package returns this class for `interface="numpy"` and a `TorchCircuit` for
    `interface="torch"`.

    Args:
        spec: The circuit to run, usually built with `assemble(blocks)`.
        observables: What to measure. `None` measures `Z` on every qubit and returns
            one value per qubit. A single `Observable` or `PauliExpr` returns a
            single value, and a list of them returns one value for each. Default:
            `None`.
        provider: The backend that runs the circuit. It can be a registered name
            (`"numpy"`, `"pennylane"` and others), a configuration instance, or a
            provider object. Default: `"numpy"`.
        interface: How a call is evaluated. With `"numpy"`, a call runs the
            NumPy path and returns NumPy values. With `"torch"`, each call is
            passed to the registered PyTorch interface, and `TorchCircuit` builds
            its core circuit in this way. The top-level `Circuit` function always
            passes `"numpy"`. Default: `"numpy"`.

    Example:
        ```python
        import numpy as np
        from eigenkit import Circuit, Encoding, Variational, all_z

        circuit = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1)],
            observables=all_z(2),
        )

        # The circuit stores no weights, so you pass them at every call.
        weights = np.zeros(circuit.spec.n_weights)
        values = circuit(weights, np.array([0.1, 0.2]))

        # Gradients, without PyTorch.
        grad = circuit.jacobian(weights, np.array([0.1, 0.2]))
        ```
    """

    def __init__(
        self,
        spec: CircuitSpec,
        observables: ObservableLike = None,
        provider: ProviderSpec = "numpy",
        interface: Interface = "numpy",
    ):
        observable_list, scalar = _normalize_observables(observables, spec.n_qubits)
        _reject_empty(observable_list)
        self.spec = spec
        self.observables = observable_list
        self._scalar = scalar
        self.provider: Provider = resolve_provider(provider)
        self.interface = interface
        validate_circuit(spec, self.provider.capabilities)
        # Compile once; reuse for every call.
        self.compiled: CompiledCircuit = self.provider.compile(spec, observable_list)
        # Decide jacobian dispatch up front. Naive parameter-shift is only
        # correct when each gate angle is a direct WeightRef that no other
        # angle shares; anything wrapping a weight (Scale/Multiply/Sum/Negate,
        # e.g. `Reuploading(trainable_scale=True)`) or reusing one needs the
        # chain-rule path, which perturbs each parametric gate's angle
        # individually and weights it by the symbolic chain factor.
        self._has_unsafe_gates: bool = bool(find_parameter_shift_unsafe_gates(spec))
        # Weights and inputs feeding a multi-frequency gate (a controlled
        # rotation): the two-term shift rule is not exact for those, so the
        # shift jacobians switch to the four-term rule for them alone.
        self._four_term_weights = weights_needing_four_term_shift(spec)
        self._four_term_inputs = inputs_needing_four_term_shift(spec)
        # Same check for inputs: when every input-dependent gate is a direct
        # InputRef used once, the naive input-shift jacobian is correct and
        # avoids `shift_gate_angle` (which wraps in `Sum`), so backends that
        # don't support `Sum` (notably cudaq) still get input grads.
        self._has_unsafe_input_gates: bool = bool(find_input_shift_unsafe_gates(spec))
        # Cache of `(gate_index, shift_value)` -> angle-shifted compiled circuit,
        # lazily populated on the first backward pass through chain-rule.
        self._chain_rule_cache: dict[tuple[int, float], CompiledCircuit] = {}

    # ---- dispatch ----

    def __call__(self, weights, inputs=None):
        if self.interface == "numpy":
            if inputs is None:
                inputs = self._default_inputs()
            return self._run_numpy(weights, inputs)
        if self.interface == "torch":
            # Framework autodiff is delegated to the registered interface,
            # which owns tensor<->numpy conversion and the autograd wrapping
            # (and builds a default inputs tensor when `inputs is None`).
            return get_interface("torch").run_circuit(self, weights, inputs)
        raise ValueError(f"unknown interface {self.interface!r}")

    def _default_inputs(self):
        """Supply an empty inputs vector when the circuit consumes none.
        Raise early with a clear message when inputs are actually needed."""
        if self.spec.n_inputs != 0:
            raise ValueError(
                f"circuit expects {self.spec.n_inputs} input value(s) but "
                f"none were provided"
            )
        return np.empty(0, dtype=np.float64)

    def _check_length(self, given: int, expected: int, name: str) -> None:
        """Reject a parameter vector the circuit cannot consume.

        Args:
            given (int): Length the caller passed.
            expected (int): Length the compiled spec needs.
            name (str): `"weights"` or `"inputs"`, used in the message.

        Raises:
            ValueError: If the two differ.
        """
        if given != expected:
            raise ValueError(
                f"this circuit takes {expected} {name}, got {given}. "
                f"(n_qubits={self.spec.n_qubits}, n_weights={self.spec.n_weights}, "
                f"n_inputs={self.spec.n_inputs})"
            )

    def _check_parameter_lengths(self, weights, inputs) -> None:
        """Reject a `weights` or `inputs` the circuit cannot consume.

        Checking `ndim` alone is not enough. Without a length check, extra
        features would be read past silently -- a 5-feature input to a
        3-feature circuit would return the 3-feature answer with no indication
        -- and too few would surface as `IndexError: index 3 is out of
        bounds`, which names neither the circuit nor the expected size.

        Lives here, not on the numpy path, because `__call__` dispatches to an
        interface that resolves its own default `inputs` first: the guard has
        to run after that, and every interface needs it, not just numpy. Reads
        `np.shape`, so a framework tensor is measured where it lies -- no
        conversion and no device transfer.

        Args:
            weights (Any): Weight vector or batch; numpy or framework-native.
            inputs (Any): Input vector or batch, same.

        Raises:
            ValueError: If either trailing axis is not the length the spec needs.
        """
        self._check_length(np.shape(weights)[-1], self.spec.n_weights, "weights")
        self._check_length(np.shape(inputs)[-1], self.spec.n_inputs, "inputs")

    # ---- numpy path ----

    def _run_numpy(self, weights, inputs) -> np.ndarray | float:
        weights_np = np.asarray(weights, dtype=np.float64)
        inputs_np = np.asarray(inputs, dtype=np.float64)
        if inputs_np.ndim not in (1, 2):
            raise ValueError(
                f"inputs must be 1D (single sample) or 2D (batch), got ndim={inputs_np.ndim}"
            )
        self._check_parameter_lengths(weights_np, inputs_np)
        output = self.compiled.run(weights_np, inputs_np)
        if not self._scalar:
            return output
        if inputs_np.ndim == 1:
            return float(output[0])
        return output[:, 0]

    # ---- gradient jacobians (parameter-shift; chain-rule when needed) ----

    def _gradient_jacobians(
        self,
        weights_np: np.ndarray,
        inputs_np: np.ndarray,
        *,
        need_input_grad: bool = False,
        need_weight_grad: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return `(weight_jacobian, input_jacobian_or_None)`.

        `weight_jacobian` has shape `(n_observables, n_weights)`;
        `input_jacobian` has shape `(n_observables, n_inputs)` when
        `need_input_grad` is True, otherwise None. With `need_weight_grad`
        False (frozen weights in a torch backward) the parameter-shift paths
        skip the weight shifts and return None in its place.

        Provider-native gradients come first: any backend whose compiled
        circuit exposes a ``jacobian`` (the numpy sim's adjoint, PennyLane's
        own diff engine, ...) uses it — computed however that backend's device
        config specifies. We only fall back to our backend-agnostic
        parameter-shift when the provider supplies none. We extend a backend's
        capabilities, never override them.

        The parameter-shift fallback itself has two paths: the fast naive
        weight-only path when (a) only weight grads are needed and (b) every
        gate angle is either a direct WeightRef or independent of all weights;
        else the chain-rule path computes weight and input jacobians in a
        single pass over gates, reusing each perturbed-circuit forward.

        For a 2-D ``inputs`` batch the jacobians gain a leading batch axis —
        ``(B, n_observables, n_weights)`` and ``(B, n_observables, n_inputs)``.

        Every gradient path — `jacobian()`, the torch backward and the
        variational spine — comes through here, so this is where a backend that
        refuses gradients (``gradient_method="none"``) is held to it.
        """
        if self.provider.gradient_method == "none":
            raise RuntimeError(
                f"backend '{self.provider.name}' has gradients disabled "
                f"(gradient_method='none'): a parameter-shift pass on hardware "
                f"costs 2 * n_weights paid jobs. Train on a simulator and deploy "
                f"the weights, or set enable_gradients=True in the device config "
                f"to opt in."
            )
        if inputs_np.ndim == 2:
            return self._batched_gradient_jacobians(
                weights_np,
                inputs_np,
                need_input_grad=need_input_grad,
                need_weight_grad=need_weight_grad,
            )
        provider_jacobian = getattr(self.compiled, "jacobian", None)
        if provider_jacobian is not None:
            return provider_jacobian(
                weights_np, inputs_np, need_input_grad=need_input_grad
            )
        if not need_input_grad and not self._has_unsafe_gates:
            weight_jac = parameter_shift_jacobian(
                self.compiled, weights_np, inputs_np, self._four_term_weights
            )
            return weight_jac, None
        if (
            need_input_grad
            and not self._has_unsafe_gates
            and not self._has_unsafe_input_gates
        ):
            # Both jacobians via naive shifts. Avoids `shift_gate_angle`
            # so backends without `Sum` support (cudaq) still work.
            weight_jac = (
                parameter_shift_jacobian(
                    self.compiled, weights_np, inputs_np, self._four_term_weights
                )
                if need_weight_grad
                else None
            )
            input_jac = parameter_shift_input_jacobian(
                self.compiled, weights_np, inputs_np, self._four_term_inputs
            )
            return weight_jac, input_jac
        return self._chain_rule_jacobians(
            weights_np,
            inputs_np,
            need_input_grad=need_input_grad,
            need_weight_grad=need_weight_grad,
        )

    def _batched_gradient_jacobians(
        self,
        weights_np: np.ndarray,
        inputs_np: np.ndarray,
        *,
        need_input_grad: bool = False,
        need_weight_grad: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Jacobians for a 2-D ``inputs`` batch, with a leading batch axis.

        A backend whose compiled circuit advertises a batch-capable ``jacobian``
        (``supports_batched_jacobian``, e.g. the numpy sim's adjoint) computes
        the whole batch in one sweep; every other backend falls back to looping
        the single-sample path and stacking — same result, just without the
        per-sample vectorization. (Note: a backend can batch the *forward* pass
        natively yet still have a single-sample ``jacobian``, so this is a
        distinct, jacobian-specific flag — not ``capabilities.supports_batching``.)
        """
        provider_jacobian = getattr(self.compiled, "jacobian", None)
        if provider_jacobian is not None and getattr(
            self.compiled, "supports_batched_jacobian", False
        ):
            return provider_jacobian(
                weights_np, inputs_np, need_input_grad=need_input_grad
            )
        weight_jacs: list[np.ndarray | None] = []
        input_jacs: list[np.ndarray | None] = []
        for row in inputs_np:
            weight_jac, input_jac = self._gradient_jacobians(
                weights_np,
                row,
                need_input_grad=need_input_grad,
                need_weight_grad=need_weight_grad,
            )
            weight_jacs.append(weight_jac)
            input_jacs.append(input_jac)
        batched_weight = (
            None if weight_jacs[0] is None else np.stack(weight_jacs)
        )  # (B, n_obs, n_weights)
        batched_input = (
            np.stack(input_jacs) if need_input_grad else None  # (B, n_obs, n_inputs)
        )
        return batched_weight, batched_input

    def _chain_rule_jacobians(
        self,
        weights_np: np.ndarray,
        inputs_np: np.ndarray,
        *,
        need_input_grad: bool = False,
        need_weight_grad: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Chain-rule-aware parameter-shift jacobians.

        For each parametric gate `g` with angle `theta_g = f(weights, inputs)`:

            d<O>/dw_i = sum_g (dtheta_g/dw_i) * (d<O>/dtheta_g)
            d<O>/dx_j = sum_g (dtheta_g/dx_j) * (d<O>/dtheta_g)

        `d<O>/dtheta_g` comes from parameter-shift on a circuit where `g`'s
        angle is wrapped in `Sum(original, Constant(shift))`: shifts of
        +/- pi/2, and also +/- 3*pi/2 for the gates in `MULTI_FREQUENCY_GATES`
        (`CRY`), whose four-term rule needs both. Perturbed compiled circuits
        are cached per `(gate_index, shift)`, so the cost is paid once across
        all backward calls on this instance.

        `dtheta_g/dw_i` and `dtheta_g/dx_j` are computed symbolically via
        `chain_factor` / `chain_factor_input` at the current point. Both
        jacobians share the same `gate_angle_grad`, so adding the input
        jacobian costs only the extra chain-factor evaluations.
        """
        n_obs = len(self.observables)
        n_weights = len(weights_np)
        weight_jac = (
            np.zeros((n_obs, n_weights), dtype=np.float64) if need_weight_grad else None
        )
        input_jac = (
            np.zeros((n_obs, len(inputs_np)), dtype=np.float64)
            if need_input_grad
            else None
        )

        for gate_index, gate in enumerate(self.spec.gates):
            if not gate.params:
                continue
            angle = gate.params[0]
            weight_deps = (
                collect_weight_dependencies(angle) if need_weight_grad else set()
            )
            input_deps = collect_input_dependencies(angle) if need_input_grad else set()
            if not weight_deps and not input_deps:
                continue
            weight_factors: dict[int, float] = {}
            for weight_index in weight_deps:
                factor = chain_factor(angle, weight_index, weights_np, inputs_np)
                if factor != 0.0:
                    weight_factors[weight_index] = float(factor)
            input_factors: dict[int, float] = {}
            for input_index in input_deps:
                factor = chain_factor_input(angle, input_index, weights_np, inputs_np)
                if factor != 0.0:
                    input_factors[input_index] = float(factor)
            if not weight_factors and not input_factors:
                continue

            # Same rule choice as the naive path, but per gate rather than per
            # weight: here the gate whose angle is being shifted is known, so
            # the four-term rule is used exactly for the multi-frequency ones.
            rule = FOUR_TERM_RULE if gate.name in MULTI_FREQUENCY_GATES else TWO_TERM_RULE
            gate_angle_grad = None  # shape (n_obs,)
            for shift, coefficient in rule:
                term = coefficient * self._perturbed_compiled(gate_index, shift).run(
                    weights_np, inputs_np
                )
                gate_angle_grad = (
                    term if gate_angle_grad is None else gate_angle_grad + term
                )

            if weight_jac is not None:
                for weight_index, factor in weight_factors.items():
                    weight_jac[:, weight_index] += factor * gate_angle_grad
            if input_jac is not None:
                for input_index, factor in input_factors.items():
                    input_jac[:, input_index] += factor * gate_angle_grad

        return weight_jac, input_jac

    def _perturbed_compiled(
        self,
        gate_index: int,
        shift_value: float,
    ) -> CompiledCircuit:
        """Lazily build (and cache) a CompiledCircuit identical to this
        one except that one gate's angle is shifted by `shift_value`."""
        key = (gate_index, shift_value)
        cached = self._chain_rule_cache.get(key)
        if cached is not None:
            return cached
        perturbed_spec = shift_gate_angle(self.spec, gate_index, shift_value)
        # A hardware handle compiles its shifted copies itself: on its own
        # device and qubits, so every gradient term comes from the same
        # hardware, and without repeating its cost warnings.
        compile_shifted = getattr(self.compiled, "compile_shifted", None)
        if compile_shifted is not None:
            compiled = compile_shifted(perturbed_spec)
        else:
            compiled = self.provider.compile(perturbed_spec, self.observables)
        self._chain_rule_cache[key] = compiled
        return compiled

    # ---- public gradients (numpy interface, no torch) ----

    def jacobian(self, weights, inputs=None, *, input_grad=False):
        """Return the gradient of the expectation values with respect to the weights.

        It returns `d<O>/d(weights)`, of shape `(n_weights,)` for a single
        observable or `(n_observables, n_weights)` for a list of observables, like
        the output of `__call__()`. Pass a single 1-D `inputs` sample. For a batch,
        call it once for each row.

        The NumPy interface can compute every gradient without PyTorch. This method
        uses the same sequence of gradient methods as the variational optimisation
        tools and the PyTorch interface. It uses the backend's own differentiation
        when the backend has one (the analytic adjoint method of the NumPy
        simulator, the differentiation of PennyLane, and so on), and otherwise the
        parameter-shift rule, which works on any backend.

        With `input_grad=True`, it returns `(weight_jacobian, input_jacobian)`,
        where `input_jacobian` is `d<O>/d(inputs)`, with the same leading shape.

        Args:
            weights (np.ndarray): The weight vector, of shape `(n_weights,)`.
            inputs (np.ndarray | None): One 1-D sample, of shape `(n_inputs,)`. This
                method takes a single sample, so for a batch, call it once for each
                row. `None` works only for a circuit that takes no inputs, and it
                raises a `ValueError` for any other circuit. Default: `None`.
            input_grad (bool): If True, also return the jacobian with respect to the
                inputs. Default: `False`.

        Returns:
            jacobian (np.ndarray | tuple[np.ndarray, np.ndarray]): The weight
                jacobian, or `(weight_jacobian, input_jacobian)` when `input_grad`
                is True.

        Raises:
            RuntimeError: On a hardware backend whose configuration leaves
                `enable_gradients` off (`gradient_method="none"`).
        """
        if inputs is None:
            inputs = self._default_inputs()
        weights_np = _to_numpy(weights)
        inputs_np = _to_numpy(inputs)
        if inputs_np.ndim != 1:
            raise ValueError(
                "jacobian() takes a single 1-D inputs sample; map over rows for a batch"
            )
        weight_jac, input_jac = self._gradient_jacobians(
            weights_np, inputs_np, need_input_grad=input_grad
        )
        if self._scalar:
            weight_jac = weight_jac[0]
            if input_jac is not None:
                input_jac = input_jac[0]
        if input_grad:
            return weight_jac, input_jac
        return weight_jac

    # ---- non-Pauli measurements ----

    def _require_single_sample(self, inputs_np: np.ndarray, method: str) -> None:
        """Reject a batch where only one sample is meaningful.

        `probabilities` and `zero_state_probability` return one distribution
        and one scalar respectively, so a 2-D `inputs` has nowhere to go.
        Handed to the backend, it would produce silent nonsense: a 2-qubit
        circuit with a 3-row batch would come back shape (16,) summing to 4.0
        on numpy and (2, 4) summing to 2.0 on pennylane.

        Args:
            inputs_np (np.ndarray): The coerced inputs.
            method (str): Name used in the message.

        Raises:
            ValueError: If `inputs_np` is not one sample.
        """
        if inputs_np.ndim > 1:
            raise ValueError(
                f"{method}() takes one sample; got a {inputs_np.ndim}-D inputs "
                f"array of shape {inputs_np.shape}. Map over the rows, or use "
                f"zero_state_probabilities() for a batch of all-zeros "
                f"probabilities."
            )

    def _require_backend(self, name: str, hint: str = ""):
        """Return the backend's `name` method, or say which backend lacks it.

        Not every `CompiledCircuit` implements the optional half of the
        protocol — statevector access, async submission, dry runs — so
        each of those entry points asks for its method here rather than
        repeating the same check and message.

        Args:
            name (str): The method to look up on the compiled handle.
            hint (str): What to do instead, appended to the message.

        Returns:
            The bound method.

        Raises:
            NotImplementedError: If the active backend does not implement it.
        """
        method = getattr(self.compiled, name, None)
        if method is None:
            message = f"backend '{self.provider.name}' does not implement {name}()."
            raise NotImplementedError(f"{message} {hint}" if hint else message)
        return method

    def zero_state_probability(self, weights, inputs=None) -> float:
        """Return the probability of measuring the all-zeros state after the circuit.

        It is useful for quantities based on fidelity: quantum kernels built from
        inversion circuits, swap tests and the Bures distance. Each backend
        computes it in its own way. PennyLane returns `qml.probs(...)[0]`, the
        Qiskit and NumPy simulators return the squared amplitude of the all-zeros
        basis state, Braket uses its own probability result, and IQM and Qiskit
        Runtime estimate it from shots. This method does not go through the
        `Observable` path, because the value is not the expectation value of a
        Pauli operator.

        Args:
            weights (np.ndarray): The weight vector, of shape `(n_weights,)`.
            inputs (np.ndarray | None): One 1-D sample. `None` works only for a
                circuit that takes no inputs, and it raises a `ValueError` for any
                other circuit. Default: `None`.

        Returns:
            The probability, a float in `[0, 1]`.

        Raises:
            NotImplementedError: If the active backend does not provide
                `zero_state_probability`.
        """
        probability = self._require_backend("zero_state_probability")
        if inputs is None:
            inputs = self._default_inputs()
        inputs_np = np.asarray(_to_numpy(inputs))
        self._require_single_sample(inputs_np, "zero_state_probability")
        return float(probability(_to_numpy(weights), inputs_np))

    def zero_state_probabilities(self, weights, inputs) -> np.ndarray:
        """Compute `zero_state_probability()` for every row of a batch of inputs.

        `inputs` has shape `(B, n_inputs)`, and the result is an array of shape
        `(B,)` of all-zeros probabilities. Backends that provide a batched
        `zero_state_probabilities` (such as the NumPy simulator) compute the whole
        batch at once. The other backends call the single-sample method once for
        each row. This is what lets [`QuantumKernel`][eigenkit.qml.QuantumKernel]
        build a fidelity matrix without a Python loop over pairs of samples.

        Args:
            weights (np.ndarray): The weight vector, of shape `(n_weights,)`.
            inputs (np.ndarray): A batch, of shape `(B, n_inputs)`.

        Returns:
            One all-zeros probability for each row, as an array of shape `(B,)`.
        """
        probability = self._require_backend("zero_state_probability")
        weights_np = _to_numpy(weights)
        inputs_np = np.asarray(_to_numpy(inputs), dtype=np.float64)
        if inputs_np.ndim != 2:
            raise ValueError(
                f"zero_state_probabilities expects a batch of shape "
                f"(B, n_inputs); got a {inputs_np.ndim}-D array of shape "
                f"{inputs_np.shape}. Use zero_state_probability() for one sample."
            )
        batched = getattr(self.compiled, "zero_state_probabilities", None)
        if batched is not None:
            return np.asarray(batched(weights_np, inputs_np), dtype=np.float64)
        return np.array(
            [float(probability(weights_np, row)) for row in inputs_np],
            dtype=np.float64,
        )

    def probabilities(self, weights, inputs=None) -> np.ndarray:
        """Return the measurement distribution over all `2**n_qubits` basis states.

        The basis states are ordered with qubit `0` as the most significant bit.
        The distribution is used to decode samples. For example, the quantum
        approximate optimisation algorithm (QAOA) reads the most probable
        bitstrings from the optimised state.

        Args:
            weights (np.ndarray): The weight vector, of shape `(n_weights,)`.
            inputs (np.ndarray | None): One 1-D sample. `None` works only for a
                circuit that takes no inputs, and it raises a `ValueError` for any
                other circuit. Default: `None`.

        Returns:
            The probabilities of all basis states, of shape `(2**n_qubits,)`.

        Raises:
            NotImplementedError: On a backend that has neither a `probabilities()`
                method nor a state vector to compute the probabilities from.
        """
        # Derived from the statevector when the backend has no native
        # `probabilities`. Only four backends implement one, so without this
        # QAOA -- which reads its answer out of this -- would raise
        # NotImplementedError on qiskit, qiskit-aer, qulacs, qsim and braket,
        # four of which return a perfectly good statevector for the same
        # circuit. Deriving it here covers every backend that has one.
        native = getattr(self.compiled, "probabilities", None)
        if native is None and not hasattr(self.compiled, "statevector"):
            self._require_backend(
                "probabilities", "It has no statevector() to derive them from."
            )
        if inputs is None:
            inputs = self._default_inputs()
        inputs_np = np.asarray(_to_numpy(inputs))
        self._require_single_sample(inputs_np, "probabilities")
        if native is not None:
            return np.asarray(native(_to_numpy(weights), inputs_np), dtype=np.float64)
        return np.abs(np.asarray(self.statevector(weights, inputs_np))) ** 2

    def sample(self, weights, inputs=None) -> dict[str, int]:
        """Return the raw measurement histogram, which maps each bitstring to a count.

        It is what a sampling backend measured, before any averaging. These are
        the counts behind `probabilities()`, and it is the only form that stays
        usable when the circuit has too many qubits for a dense distribution. In
        each bitstring, qubit `0` is the most significant bit, so `"100"` means
        that qubit `0` was measured in state `1`.

        Only sampling backends can return a histogram, because it needs a finite
        number of shots. Exact statevector simulators raise an error instead of
        making up a histogram. Use `probabilities()` with those.

        Args:
            weights (np.ndarray): The weight vector, of shape `(n_weights,)`.
            inputs (np.ndarray | None): One 1-D sample. `None` works only for a
                circuit that takes no inputs, and it raises a `ValueError` for any
                other circuit. Default: `None`.

        Returns:
            counts (dict[str, int]): The number of shots for each measured
                bitstring. The counts add up to the number of shots set for the
                backend.

        Raises:
            NotImplementedError: On backends that do not sample. This includes
                every exact simulator, which has no shots to report.
        """
        sampler = self._require_backend(
            "sample",
            "It returns exact values rather than shots; use probabilities().",
        )
        if inputs is None:
            inputs = self._default_inputs()
        inputs_np = np.asarray(_to_numpy(inputs))
        self._require_single_sample(inputs_np, "sample")
        return dict(sampler(_to_numpy(weights), inputs_np))

    def statevector(self, weights, inputs=None, initial_state=None) -> np.ndarray:
        r"""Return the full complex statevector after the circuit runs.

        It is a flat array of length `2**n_qubits`, ordered with qubit `0` as the
        most significant bit. A 2-D `inputs` batch returns an array of shape
        `(B, 2**n_qubits)`. With `initial_state`, a normalised flat statevector in
        the same order, the evolution starts from that state instead of `|0…0⟩`.
        Reading a state out and passing it back in this way is the basic operation
        behind the state carried over in quantum reservoirs and behind amplitude
        encoding.

        Args:
            weights (np.ndarray): The weight vector, of shape `(n_weights,)`.
            inputs (np.ndarray | None): A 1-D sample, or a 2-D batch. `None` works
                only for a circuit that takes no inputs, and it raises a
                `ValueError` for any other circuit. Default: `None`.
            initial_state (np.ndarray | None): The state to start the evolution
                from, instead of `|0…0⟩`. It is a normalised flat statevector in the
                same order. Default: `None`.

        Returns:
            The statevector, of shape `(2**n_qubits,)`, or `(B, 2**n_qubits)` for
                a batch.

        Raises:
            NotImplementedError: On backends without access to the statevector,
                which is the case for most quantum processing unit (QPU) and cloud
                backends.
        """
        self._require_backend("statevector")
        if inputs is None:
            inputs = self._default_inputs()
        return np.asarray(
            self.compiled.statevector(
                _to_numpy(weights), _to_numpy(inputs), initial_state=initial_state
            ),
            dtype=np.complex128,
        )

    # ---- async submission (QPU / cloud) ----

    def submit(self, weights, inputs=None):
        """Submit a job without waiting for it, and return a `JobHandle`.

        `handle.result()` returns the expectation value of each observable.

        Args:
            weights (np.ndarray): The weight vector, of shape `(n_weights,)`.
            inputs (np.ndarray | None): One 1-D sample. `None` works only for a
                circuit that takes no inputs, and it raises a `ValueError` for any
                other circuit. Default: `None`.

        Returns:
            handle (JobHandle): A handle for the submitted job.

        Raises:
            NotImplementedError: If the `CompiledCircuit` of the backend does not
                provide `submit`. Currently, Braket, Qiskit Runtime and IQM provide
                it.
        """
        self._require_backend("submit", "Use .__call__() instead.")
        if inputs is None:
            inputs = self._default_inputs()
        weights_np = _to_numpy(weights)
        inputs_np = _to_numpy(inputs)
        if inputs_np.ndim != 1:
            raise ValueError(
                "submit() accepts only 1-D inputs (single sample); "
                "loop in Python for batches"
            )
        return self.compiled.submit(weights_np, inputs_np)

    # ---- inspection ----

    def dry_run_info(self) -> dict:
        """Describe the prepared circuit in a dictionary, without submitting it.

        It is available only on backends whose `CompiledCircuit` provides a
        `dry_run_info` method. Currently, these are Braket (on its local simulator
        as well as on cloud devices), Qiskit Runtime and IQM. It is useful to check
        the size of the circuit, the gate counts after transpilation, the number of
        shots and the number of circuits that each call runs, before you submit to
        a cloud simulator or a quantum processing unit (QPU). It does not estimate a
        price.
        """
        return self._require_backend(
            "dry_run_info",
            "Only cloud/QPU backends (Braket, Qiskit Runtime, IQM) implement this.",
        )()

    def resources(self, **kwargs: Any) -> dict:
        """Report the resources that the circuit uses once transpiled for its target.

        It reports the native gate counts, the number of two-qubit gates and the
        two-qubit depth, the physical qubits chosen by the layout, the number of
        two-qubit gates added by routing (compared with the same circuit on the
        native gates with all-to-all connectivity), and the success probability
        predicted by the error rates of the target. Nothing is submitted.

        Args:
            **kwargs (Any): Arguments passed to the backend, for example
                `measured=False` on Qiskit Runtime for the bare estimator circuit.

        Returns:
            A dictionary. See `QiskitRuntimeCompiled.resources()` for its keys.

        Raises:
            NotImplementedError: On backends that do not transpile for a device,
                which includes every simulator. To inspect a device locally, use
                Qiskit Runtime or IQM with a fake `backend_instance`.
        """
        return self._require_backend(
            "resources",
            "Only transpiling hardware backends (Qiskit Runtime, IQM) implement this.",
        )(**kwargs)

    def transpiled_circuit(self, **kwargs: Any) -> Any:
        """Return the `qiskit.QuantumCircuit` that a hardware backend submits.

        The circuit is transpiled for the target of the backend. Use it to draw the
        circuit (`.draw("mpl")`) or to inspect it.

        Raises:
            NotImplementedError: On backends that do not transpile for a device.
        """
        return self._require_backend(
            "transpiled_circuit",
            "Only transpiling hardware backends (Qiskit Runtime, IQM) implement this.",
        )(**kwargs)

    def draw(self, *, decimals: int = 3) -> str:
        """Return a text drawing of the circuit, over several lines.

        The drawing does not depend on the provider: it is built from the
        `CircuitSpec`, not from the compiled backend object. See `eigenkit.core.draw`
        for the details of the layout.

        Args:
            decimals: The number of significant digits shown for a numeric gate
                angle. Default: `3`.

        Returns:
            The drawing, as a single string that contains newlines.
        """
        return draw_text(self.spec, decimals=decimals)

    def draw_mpl(self, **kwargs):
        """Draw the circuit as a matplotlib figure, and return `(fig, ax)`.

        Gates are drawn as labelled boxes and controls as filled dots, and each
        measured wire (from `self.observables`) ends with a measurement symbol.
        As with `draw()`, the picture is built from the `CircuitSpec` and the
        observables, not from the compiled backend, so it does not depend on the
        provider.

        Pass `style="eigen"` for the brand colours (each gate is coloured by its
        kind: rotations, `H` and block gates, and measurements), or
        `style="roles"` to colour the outline of each gate by its role (encoding,
        trainable or fixed). The default is black on white. Each style has a
        `"-dark"` variant (for example `"eigen-dark"`) for dark backgrounds, and
        `"eigen-universal"` is a single image that works on both.

        matplotlib is an optional dependency. Install it with `eigenkit[viz]`.

        Args:
            **kwargs (Any): Arguments passed to
                [`draw_mpl()`][eigenkit.core.draw_mpl.draw_mpl]: `decimals`,
                `show_params`, `show_measurements`, `fontsize`, `style`, `legend`
                and `ax`.

        Returns:
            fig_ax (tuple): The matplotlib `(fig, ax)` pair.
        """
        from .draw_mpl import draw_mpl

        return draw_mpl(self.spec, observables=self.observables, **kwargs)
