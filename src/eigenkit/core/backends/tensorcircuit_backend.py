"""The tensorcircuit-ng backend, for tensor-network simulation in JAX.

It uses the native automatic differentiation of JAX. tensorcircuit-ng is an optional extra (`pip install eigenkit[tensorcircuit]`).
Its packages (tensorcircuit and JAX) are imported only when needed, so
`import eigenkit` does not import them. `compile()` raises `ImportError` if they
are missing.

It declares `gradient_method = "adjoint"`: the compiled circuit has a `jacobian`
method, computed by reverse-mode automatic differentiation in JAX
(`jax.jacrev`), so the gradient code of `Circuit` uses it directly. Both the
NumPy variational optimisation (used by the Variational Quantum Eigensolver and
the Quantum Approximate Optimization Algorithm) and the parameter-shift fallback
of the PyTorch interface get their gradients through `_gradient_jacobians`,
which calls this `jacobian`.

The simulation contracts the tensor network exactly, through the
`Circuit.expectation` method of tensorcircuit. The batch axis is vectorised with
`jax.vmap`, and the whole forward and gradient computation is compiled with
`jax.jit`. This suits **repeated** workloads (training loops, sweeps, restarts
with an ansatz of fixed shape), where the one-time compilation is spread over
many evaluations. For a single solve, the NumPy reference simulator is faster,
because the compilation takes most of the time. The simulation is still an exact
contraction, so the gain is a constant factor: it does not make the cost
sub-exponential.

The supported gates are `RX`, `RY`, `RZ`, `H`, `X`, `Z`, `CNOT` (controlled-NOT)
and `SWAP`. To add more, extend `_apply` and `SUPPORTED`. The observables are sums
of Pauli operators. The 64-bit mode of JAX is turned on at first use, for the
`complex128` statevector.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # by registered name
    circuit = Circuit(blocks=blocks, observables=all_z(4), provider="tensorcircuit")

    # or with the config, when a device setting matters
    from eigenkit.configs import TensorCircuitConfig

    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=TensorCircuitConfig(),
    )
    ```

    The simulation uses tensor networks on JAX, with batching through
    `jax.vmap`. This backend needs the `tensorcircuit` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..ir import CircuitSpec, Observable, resolve
from ..provider import Capabilities


# Cached SDK handles, populated by `_require_tc` on first use.
_JAX = None
_JNP = None
_TC = None


def _require_tc():
    """Import tensorcircuit + jax lazily, configure jax for the complex128
    statevector, and cache the module handles. Raise a clear ``ImportError``
    (with the install hint) if the extra is not installed."""
    global _JAX, _JNP, _TC
    if _TC is not None:
        return _JAX, _JNP, _TC
    try:
        import jax
        import jax.numpy as jnp
        import tensorcircuit as tc
    except ImportError as e:
        raise ImportError(
            "tensorcircuit-ng is not installed. Install the extra with "
            "`pip install eigenkit[tensorcircuit]`."
        ) from e
    jax.config.update("jax_enable_x64", True)  # complex128 amplitudes need x64
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    _JAX, _JNP, _TC = jax, jnp, tc
    return jax, jnp, tc


# Every name here must be an `ir.GateName` and must have a branch in `_apply`;
# a name the IR never emits could never match. The IR spells a two-qubit CZ as
# `MCZ` and a Toffoli as `MCX`, both variable-arity; `Capabilities` declares
# support per gate name and cannot say "MCZ, but only on two qubits", so
# declaring them would make this backend accept a circuit at validation and
# fail at compile.
SUPPORTED = frozenset({"RX", "RY", "RZ", "H", "X", "Z", "CNOT", "SWAP"})


def _apply(c, gate, weights, inputs) -> None:
    """Apply one IR gate to a tc ``Circuit`` (angles resolved via the IR's own
    backend-agnostic ``resolve``, so they stay jax-traced under jit/vmap)."""
    name, q = gate.name, gate.qubits
    if name == "RX":
        c.rx(q[0], theta=resolve(gate.params[0], weights, inputs))
    elif name == "RY":
        c.ry(q[0], theta=resolve(gate.params[0], weights, inputs))
    elif name == "RZ":
        c.rz(q[0], theta=resolve(gate.params[0], weights, inputs))
    elif name == "H":
        c.h(q[0])
    elif name == "X":
        c.x(q[0])
    elif name == "Z":
        c.z(q[0])
    elif name == "CNOT":
        c.cnot(q[0], q[1])
    elif name == "SWAP":
        c.swap(q[0], q[1])
    else:
        raise NotImplementedError(
            f"tensorcircuit backend does not support gate {name!r} "
            f"(supported: {sorted(SUPPORTED)})"
        )


@dataclass(frozen=True)
class TensorCircuitConfig:
    """Configuration for the tensorcircuit backend.

    It has no settings yet: the contraction path is the default of tensorcircuit,
    and the batch axis is vectorised with `jax.vmap`. It exists for consistency
    with the configurations of the other backends, and so that
    `provider=TensorCircuitConfig()` works without a `register` call.
    """

    def make_backend(self) -> "TensorCircuitBackend":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `TensorCircuitBackend` for this configuration.
        """
        return TensorCircuitBackend(config=self)


class TensorCircuitCompiled:
    """A circuit prepared for tensorcircuit-ng, compiled with `jax.jit` on first use.

    It is internal:
    [`TensorCircuitBackend.compile()`][eigenkit.core.backends.tensorcircuit_backend.TensorCircuitBackend.compile]
    builds it and returns it, and it is never built directly. `jacobian()` uses
    reverse-mode automatic differentiation in JAX, and the batch axis is
    vectorised with `jax.vmap`, so the one-time compilation cost is spread over a
    training loop or a sweep.
    """

    # The batched jacobian is a jax.vmap over the single-sample jacrev, so the
    # dispatch can hand us the whole 2-D batch in one call.
    supports_batched_jacobian = True

    def __init__(self, circuit: CircuitSpec, observables: Sequence[Observable]):
        """Prepare the circuit for repeated evaluation.

        `TensorCircuitBackend.compile()` builds this object, so there is no reason
        to build one directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.
        """
        jax, jnp, tc = _require_tc()
        self._n = circuit.n_qubits
        self._gates = circuit.gates
        # (coeff, ((op, qubit), ...)) per term, per observable. Pauli gate
        # objects are built fresh inside `forward` — tc consumes a gate's
        # nodes during expectation, so a shared instance corrupts the network.
        obs_terms = [
            [(float(t.coeff), tuple(t.paulis)) for t in o.terms] for o in observables
        ]
        pget = {"X": tc.gates.x, "Y": tc.gates.y, "Z": tc.gates.z}
        n, gates = self._n, self._gates

        def forward(weights, inputs):
            c = tc.Circuit(n)
            for g in gates:
                _apply(c, g, weights, inputs)
            outs = []
            for terms in obs_terms:
                total = 0.0
                for coeff, paulis in terms:
                    ops = tuple((pget[op](), [qubit]) for op, qubit in paulis)
                    total = total + coeff * jnp.real(c.expectation(*ops))
                outs.append(total)
            return jnp.stack(outs)

        self._run1 = jax.jit(forward)
        self._runB = jax.jit(jax.vmap(forward, in_axes=(None, 0)))
        self._jw1 = jax.jit(jax.jacrev(forward, argnums=0))
        self._jx1 = jax.jit(jax.jacrev(forward, argnums=1))
        self._jwB = jax.jit(jax.vmap(jax.jacrev(forward, argnums=0), in_axes=(None, 0)))
        self._jxB = jax.jit(jax.vmap(jax.jacrev(forward, argnums=1), in_axes=(None, 0)))

    def run(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return one expectation value per observable.

        `weights` must be 1-D. The result has shape `(n_obs,)` when `inputs` is
        1-D, and `(B, n_obs)` when `inputs` is 2-D.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features: 1-D for one sample, or 2-D
                with a leading batch axis.

        Returns:
            The expectation values, one for each observable.
        """
        _, jnp, _ = _require_tc()
        w = jnp.asarray(np.asarray(weights, np.float64))
        x = jnp.asarray(np.asarray(inputs, np.float64))
        return np.asarray(self._runB(w, x) if x.ndim == 2 else self._run1(w, x))

    def jacobian(self, weights, inputs, *, need_input_grad: bool = False):
        """Compute the jacobians with reverse-mode automatic differentiation in JAX.

        The shapes are the same as for the adjoint method of the NumPy simulator:
        `(n_obs, n_weights)` and `(n_obs, n_inputs)`, with a leading batch axis
        for 2-D inputs.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features: 1-D for one sample, or 2-D
                with a leading batch axis.
            need_input_grad (bool): If True, also compute the jacobian with
                respect to the inputs. Default: `False`.

        Returns:
            weight_jacobian (np.ndarray): The jacobian with respect to the weights.
            input_jacobian (np.ndarray | None): The jacobian with respect to the
                inputs, or `None` when `need_input_grad` is `False`.
        """
        _, jnp, _ = _require_tc()
        w = jnp.asarray(np.asarray(weights, np.float64))
        x = jnp.asarray(np.asarray(inputs, np.float64))
        if x.ndim == 2:
            weight_jac = np.asarray(self._jwB(w, x))
            input_jac = np.asarray(self._jxB(w, x)) if need_input_grad else None
        else:
            weight_jac = np.asarray(self._jw1(w, x))
            input_jac = np.asarray(self._jx1(w, x)) if need_input_grad else None
        return weight_jac, input_jac

    def probabilities(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Return the full measurement distribution over the `2**n` basis states.

        Qubit `0` is the most significant bit. The Quantum Approximate
        Optimization Algorithm (QAOA) uses it to decode its result. It builds the
        dense state, so it is only practical for a small number of qubits.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features of one sample (1-D).

        Returns:
            The probabilities, of shape `(2**n_qubits,)`.
        """
        _, jnp, tc = _require_tc()
        w = jnp.asarray(np.asarray(weights, np.float64))
        x = jnp.asarray(np.asarray(inputs, np.float64))
        c = tc.Circuit(self._n)
        for g in self._gates:
            _apply(c, g, w, x)
        return np.asarray(jnp.abs(c.state()) ** 2)


class TensorCircuitBackend:
    """The provider for tensorcircuit-ng, tensor-network simulation on JAX.

    It is registered as `"tensorcircuit"` once the extra is installed. It is
    internal: you select this backend with `provider="tensorcircuit"` or by
    passing a
    [`TensorCircuitConfig`][eigenkit.core.backends.tensorcircuit_backend.TensorCircuitConfig],
    not by building this class.
    """

    gradient_method = "adjoint"  # ships its own reverse-mode (jax) jacobian

    def __init__(
        self,
        config: "TensorCircuitConfig | None" = None,
        name: str = "tensorcircuit",
    ):
        """Build the backend from a device configuration.

        Args:
            config (TensorCircuitConfig): The device settings. Default: `None`,
                which uses `TensorCircuitConfig()`.
            name (str): The registry name, used in error messages. Default:
                `"tensorcircuit"`.
        """
        self.name = name
        self.config = config or TensorCircuitConfig()

    @property
    def capabilities(self) -> Capabilities:
        """What this backend supports.

        It runs the subset of gates that this backend translates to tensorcircuit
        (`RX`, `RY`, `RZ`, `H`, `X`, `Z`, `CNOT` (controlled-NOT) and `SWAP`),
        evaluates a batch in one call through `jax.vmap`, returns exact
        expectation values only, and has no limit on the number of qubits. See
        [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        return Capabilities(
            supported_gates=SUPPORTED,
            supports_batching=True,  # native parameter batch via jax.vmap
            supports_shots=False,
            max_qubits=None,
        )

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> TensorCircuitCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        A `Circuit` calls it once and reuses the result for every call, so the
        expensive work belongs here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.

        Returns:
            The prepared circuit, as a `TensorCircuitCompiled` object.
        """
        return TensorCircuitCompiled(circuit, observables)
