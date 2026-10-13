"""The qsim backend, through Cirq.

qsim is a fast Schrödinger statevector simulator from Google, and `qsimcirq`
connects it to Cirq. Both are in the optional `[qsim]` extra
(`pip install eigenkit[qsim]`) and are imported only when needed, so the rest of
the library works without them.

The circuit is translated once into a **parametrised** `cirq.Circuit`, whose
angles are SymPy symbols `w{i}` and `x{i}`. Each call sets the values of the
symbols and runs the circuit on a `qsimcirq.QSimSimulator`. Expectation values
use the native qsim method `simulate_expectation_values` (each `Observable` of
the library becomes a `cirq.PauliSum`), and statevectors come from
`simulate(...).final_state_vector`.

It declares `gradient_method = "parameter_shift"`: qsim has no cheap analytic
reverse-mode jacobian here, so the gradient code of the library differentiates
with the parameter-shift rule.

**Precision.** qsim is a **single-precision** simulator (32-bit floating point),
and the pip wheel has no double-precision (64-bit) build. The outputs are
returned as `complex128`, but they carry an accumulated error of about `1e-6`, so
they agree with a double-precision backend to about `1e-4`, not to machine
precision. This difference comes from the precision and is not a bug, as with
the single-precision path of CUDA Quantum. It does mean that qsim is the wrong
choice when you need exact agreement rather than speed.

Bit order: Cirq orders the statevector with the **first** qubit of `qubit_order`
as the most significant bit. This backend passes
`[LineQubit(0), …, LineQubit(n-1)]`, so index `0` is qubit `0`, the most
significant bit. This is the same convention as in NumPy and PennyLane. No
reordering is needed, and the backend comparison tests check this against the
NumPy reference. The number of threads follows `OMP_NUM_THREADS`, unless a
configuration sets it.

On macOS, importing this module sets `KMP_DUPLICATE_LIB_OK=TRUE` (unless it is
already set). qsimcirq and PyTorch each include their own OpenMP runtime, and
without this setting the second one to start aborts the process.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # by registered name
    circuit = Circuit(blocks=blocks, observables=all_z(4), provider="qsim")

    # or with the config, when a device setting matters
    from eigenkit.configs import QsimConfig

    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=QsimConfig(threads=8),
    )
    ```

    qsim is fast and single-precision, with an error of about `1e-6`. This
    backend needs the `qsim` extra.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..ir import (
    Constant,
    Gate,
    InputRef,
    Multiply,
    Negate,
    Observable,
    Parameter,
    Scale,
    Sum,
    WeightRef,
)
from ..provider import Capabilities, CompiledCircuit, batch_call


# The qsimcirq and torch wheels for macOS each bundle libomp; whichever starts
# second aborts with "OMP: Error #15" unless this is set before it loads. The
# flag is process-wide; drop it once the wheels stop shipping separate copies.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# Gate set qsim/Cirq supports through this adapter. Data gates (AmplitudeEncoding / QubitUnitary)
# are intentionally excluded — validate_circuit rejects them with a clear error.
SUPPORTED = frozenset(
    {
        "RX",
        "RY",
        "RZ",
        "H",
        "X",
        "Z",
        "S",
        "Sdg",
        "CNOT",
        "SWAP",
        "PhaseShift",
        "CRY",
        "CPHASE",
        "MCZ",
        "MCX",
    }
)


def _require_qsim() -> None:
    try:
        import cirq  # noqa: F401
        import qsimcirq  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "qsim is not installed. Install it via `pip install eigenkit[qsim]`."
        ) from e


@dataclass(frozen=True)
class QsimConfig:
    """Device configuration for the qsim backend.

    Args:
        threads: The size of the internal thread pool of qsim. `None` reads
            `OMP_NUM_THREADS` instead, which defaults to `1`, so qsim uses one
            core unless you set it here or in the environment. Set it explicitly
            before you report any timing. Default: `None`.
    """

    threads: int | None = None

    def make_backend(self) -> "QsimBackend":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `QsimBackend` for this configuration.
        """
        return QsimBackend(config=self)


def _resolve_symbolic(parameter: Parameter, weight_symbols, input_symbols):
    """Translate an IR Parameter into a sympy expression over the ``w{i}`` / ``x{i}`` symbols."""
    if isinstance(parameter, Constant):
        return parameter.value
    if isinstance(parameter, WeightRef):
        return weight_symbols[parameter.index]
    if isinstance(parameter, InputRef):
        return input_symbols[parameter.index]
    if isinstance(parameter, Scale):
        return parameter.factor * _resolve_symbolic(
            parameter.expression, weight_symbols, input_symbols
        )
    if isinstance(parameter, Negate):
        return -_resolve_symbolic(parameter.expression, weight_symbols, input_symbols)
    if isinstance(parameter, Sum):
        total = _resolve_symbolic(parameter.expressions[0], weight_symbols, input_symbols)
        for e in parameter.expressions[1:]:
            total = total + _resolve_symbolic(e, weight_symbols, input_symbols)
        return total
    if isinstance(parameter, Multiply):
        return _resolve_symbolic(
            parameter.a, weight_symbols, input_symbols
        ) * _resolve_symbolic(parameter.b, weight_symbols, input_symbols)
    raise TypeError(parameter)


def _apply_gate(cirq, ops, gate: Gate, qubits, weight_symbols, input_symbols):
    # NOTE: the phase-gate exponents below divide by ``math.pi``, never ``sympy.pi``. Dividing a
    # plain float by ``sympy.pi`` yields a sympy expression with no free symbols, which cirq's
    # resolver skips (nothing to substitute) and qsim's translator then rejects outright
    # ("unparseable op: Z**1.047/pi"). A float divisor keeps constant angles numeric and still
    # produces a resolvable expression for symbolic ones.
    v = [_resolve_symbolic(p, weight_symbols, input_symbols) for p in gate.params]
    q = [qubits[i] for i in gate.qubits]
    name = gate.name
    if name == "RX":
        ops.append(cirq.rx(v[0]).on(q[0]))
    elif name == "RY":
        ops.append(cirq.ry(v[0]).on(q[0]))
    elif name == "RZ":
        ops.append(cirq.rz(v[0]).on(q[0]))
    elif name == "H":
        ops.append(cirq.H(q[0]))
    elif name == "X":
        ops.append(cirq.X(q[0]))
    elif name == "Z":
        ops.append(cirq.Z(q[0]))
    elif name == "S":
        ops.append(cirq.S(q[0]))
    elif name == "Sdg":
        ops.append((cirq.S**-1).on(q[0]))
    elif name == "CNOT":
        ops.append(cirq.CNOT(q[0], q[1]))
    elif name == "SWAP":
        ops.append(cirq.SWAP(q[0], q[1]))
    elif name == "PhaseShift":  # diag(1, e^{i phi}) = ZPowGate(exponent=phi/pi)
        ops.append(cirq.ZPowGate(exponent=v[0] / math.pi).on(q[0]))
    elif name == "CPHASE":
        ops.append(cirq.CZPowGate(exponent=v[0] / math.pi).on(q[0], q[1]))
    elif name == "CRY":
        ops.append(cirq.ry(v[0]).controlled().on(q[0], q[1]))
    elif name == "MCZ":
        ops.append(cirq.Z(q[-1]).controlled_by(*q[:-1]))
    elif name == "MCX":
        ops.append(cirq.X(q[-1]).controlled_by(*q[:-1]))
    else:
        raise NotImplementedError(f"gate '{name}' not implemented in the qsim backend")


class QsimCompiled(CompiledCircuit):
    """A circuit prepared for qsim, as a Cirq circuit.

    It is internal:
    [`QsimBackend.compile()`][eigenkit.core.backends.qsim_backend.QsimBackend.compile]
    builds it and returns it, and it is never built directly.
    """

    def __init__(
        self,
        circuit,
        observables: Sequence[Observable],
        config: QsimConfig,
    ):
        """Prepare the circuit for repeated evaluation.

        `QsimBackend.compile()` builds this object, so there is no reason to
        build one directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.
            config (QsimConfig): The device settings.
        """
        _require_qsim()
        import cirq
        import qsimcirq
        import sympy

        self._n_qubits = circuit.n_qubits
        self._qubits = [cirq.LineQubit(i) for i in range(circuit.n_qubits)]
        weight_symbols = [sympy.Symbol(f"w{i}") for i in range(circuit.n_weights)]
        input_symbols = [sympy.Symbol(f"x{i}") for i in range(circuit.n_inputs)]
        self._weight_names = [f"w{i}" for i in range(circuit.n_weights)]
        self._input_names = [f"x{i}" for i in range(circuit.n_inputs)]

        ops: list = []
        for gate in circuit.gates:
            _apply_gate(cirq, ops, gate, self._qubits, weight_symbols, input_symbols)
        self._circuit = cirq.Circuit(ops)
        self._psums = [self._obs_to_paulisum(cirq, obs) for obs in observables]
        self._n_obs = len(observables)

        t = (
            config.threads
            if config.threads is not None
            else int(os.environ.get("OMP_NUM_THREADS", "1"))
        )
        self._sim = qsimcirq.QSimSimulator(
            qsim_options=qsimcirq.QSimOptions(cpu_threads=max(1, t))
        )

    def _obs_to_paulisum(self, cirq, obs: Observable):
        pmap = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
        strings = []
        for term in obs.terms:
            m = {self._qubits[q]: pmap[op] for op, q in term.paulis}
            strings.append(cirq.PauliString(m, coefficient=complex(term.coeff)))
        return cirq.PauliSum.from_pauli_strings(strings)

    def _resolver(self, weights, inputs):
        import cirq

        d = {name: float(weights[i]) for i, name in enumerate(self._weight_names)}
        d.update({name: float(inputs[i]) for i, name in enumerate(self._input_names)})
        return cirq.ParamResolver(d)

    def run(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return one expectation value per observable.

        It follows the `CompiledCircuit` contract: the result has shape `(n_obs,)`
        when `weights` and `inputs` are both 1-D, and `(B, n_obs)` otherwise.

        Args:
            weights (np.ndarray): The parameter vector: 1-D for one sample, or 2-D
                with a leading batch axis.
            inputs (np.ndarray): The input features, with the same 1-D or 2-D
                shape convention as `weights`.

        Returns:
            The expectation values, one for each observable.
        """
        return batch_call(self._run_one, weights, inputs)

    def _run_one(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        vals = self._sim.simulate_expectation_values(
            self._circuit,
            observables=self._psums,
            qubit_order=self._qubits,
            param_resolver=self._resolver(weights, inputs),
        )
        return np.array([float(np.real(v)) for v in vals], dtype=np.float64)

    def statevector(
        self, weights: np.ndarray, inputs: np.ndarray, initial_state=None
    ) -> np.ndarray:
        """Return the full complex statevector after the circuit has run.

        The result is a flat array of `2**n_qubits` amplitudes, with qubit `0` as
        the most significant bit.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features: 1-D for one sample, or 2-D
                with a leading batch axis.
            initial_state (np.ndarray | None): Must be `None`. This backend can
                only start from `|0…0⟩`. Default: `None`.

        Returns:
            The statevector, of shape `(2**n_qubits,)`, or `(B, 2**n_qubits)` for
                a batch.

        Raises:
            NotImplementedError: If `initial_state` is not `None`.
        """
        if initial_state is not None:
            raise NotImplementedError(
                "initial_state injection is not wired for the qsim backend"
            )
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        if weights.ndim == 1 and inputs.ndim == 2:
            return np.stack([self._statevector_one(weights, row) for row in inputs])
        return self._statevector_one(weights, inputs)

    def _statevector_one(self, weights, inputs) -> np.ndarray:
        result = self._sim.simulate(
            self._circuit,
            qubit_order=self._qubits,
            param_resolver=self._resolver(weights, inputs),
        )
        # Cirq orders qubit_order[0] as MSB -> already the library's qubit-0-MSB flat order.
        return np.asarray(result.final_state_vector, dtype=np.complex128)

    def zero_state_probability(self, weights: np.ndarray, inputs: np.ndarray) -> float:
        """Return the probability of measuring the state `|0…0⟩`.

        Fidelity kernels and swap tests are built on it. It does not go through
        `Observable`, because it is not the expectation value of a Pauli operator.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features of one sample (1-D).

        Returns:
            The probability, a float in `[0, 1]`.
        """
        sv = self._statevector_one(
            np.asarray(weights, dtype=np.float64), np.asarray(inputs, dtype=np.float64)
        )
        return float(np.abs(sv[0]) ** 2)


class QsimBackend:
    """The provider for qsim, the statevector simulator from Google, through Cirq.

    It is registered as `"qsim"`. It is internal: you select this backend with
    `provider="qsim"` or by passing a
    [`QsimConfig`][eigenkit.core.backends.qsim_backend.QsimConfig], not by
    building this class.
    """

    gradient_method = "parameter_shift"

    def __init__(self, config: QsimConfig | None = None, name: str = "qsim"):
        """Build the backend from a device configuration.

        Args:
            config (QsimConfig): The device settings. Default: `None`, which uses
                `QsimConfig()`.
            name (str): The registry name, used in error messages. Default:
                `"qsim"`.
        """
        self.name = name
        self.config = config or QsimConfig()

    @property
    def capabilities(self) -> Capabilities:
        """What this backend supports.

        It runs the subset of gates that can be translated to Cirq, returns exact
        expectation values only, does not evaluate a batch in one call, and has no
        limit on the number of qubits. A circuit that uses a gate outside this
        subset is rejected before compilation, with a message that names the
        gate. See [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        return Capabilities(
            supported_gates=SUPPORTED,
            supports_batching=False,
            supports_shots=False,
            max_qubits=None,
        )

    def compile(self, circuit, observables: Sequence[Observable]) -> QsimCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        A `Circuit` calls it once and reuses the result for every call, so the
        expensive work belongs here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.

        Returns:
            The prepared circuit, as a `QsimCompiled` object.
        """
        return QsimCompiled(circuit, observables, self.config)
