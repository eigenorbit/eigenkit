"""Quantum kernel matrices `K[i, j] = |<psi(x_i)|psi(x_j)>|**2`.

On a backend that gives access to the statevector, the whole Gram matrix comes
from one batched evolution of the forward feature map `U(x)` and a single matrix
product. This takes `O(N)` state evolutions instead of the `O(N**2)` fidelities
of inversion circuits, and gives the same values up to floating-point rounding.
Backends without access to the statevector, such as backends with shots or a
quantum processing unit (QPU), use the inversion circuit instead. It applies
`U(x_i)`, then the adjoint of `U(x_j)`, and reads the probability of the
all-zeros state, which is the fidelity `K[i, j]` of the two prepared states. On
a hardware backend that batches natively (Qiskit Runtime, IQM), every pair goes
into one batched call, so the whole matrix is submitted as a single job. A local
simulator receives one row of pairs per call. For a symmetric Gram matrix, only
the pairs above the diagonal are run.

The feature map is one of the two built-in maps, or any circuit that you write:

- `"Angle"`: angle encoding on every qubit (`RX` for the first repetition, `RZ`
  for the others), with no entanglement, so the state is a product state.
- `"AngleCNOT"`: the same as `"Angle"`, with a ladder of controlled-NOT (CNOT)
  gates before each repetition after the first.
- A custom map: a `CircuitSpec`, a `CircuitBuilder`, a block, a list of blocks,
  or a callable `emit(builder, x, w)` that writes `U(x; w)` onto a
  `CircuitBuilder`. The inverse half is derived with `adjoint()`, and the two
  halves share the trainable weights, so a variational kernel needs no inversion
  written by hand.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from eigenkit.core.blocks import _relocate_parameter, assemble
from eigenkit.core.builder import CircuitBuilder
from eigenkit.core.circuit import Circuit, ProviderSpec
from eigenkit.core.ir import CircuitSpec, Observable, adjoint
from eigenkit.core.provider import resolve_provider, validate_circuit


if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


FeatureMapLike = (
    Any  # str | CircuitSpec | CircuitBuilder | Block | Sequence[Block] | Callable
)


def _emit_feature_map(qc, x, n_features: int, feature_map: str, reps: int) -> None:
    """Emit the forward feature map U(x) onto ``qc``.

    First rep is an RX angle embedding; each additional rep adds an optional
    CNOT entangling ladder (``feature_map == "AngleCNOT"``) followed by an RZ
    embedding. ``x`` is the sequence of input handles bound to the data.
    """
    for q in range(n_features):
        qc.rx(x[q], qubit=q)
    for _ in range(reps - 1):
        if feature_map == "AngleCNOT":
            for q in range(n_features - 1):
                qc.cnot(q, q + 1)
        for q in range(n_features):
            qc.rz(x[q], qubit=q)


def _build_feature_map_spec(
    n_features: int,
    feature_map: str,
    reps: int,
) -> CircuitSpec:
    """Build the forward-only built-in feature map U(x) with ``n_features`` inputs."""
    qc = CircuitBuilder(n_qubits=n_features)
    x = qc.add_inputs(n_features)
    _emit_feature_map(qc, x, n_features, feature_map, reps)
    return qc.build()


def _inversion_spec(forward: CircuitSpec) -> CircuitSpec:
    """`U(x_1; w)` followed by `U(x_2; w)` adjoint, as one spec.

    Allocates `2 * n_inputs` inputs: `[0:n]` hold `x_1` and `[n:2n]` hold
    `x_2`. Only the input refs of the adjoint half are shifted; the weight refs
    are left alone, so both halves read the same trainable weights. `assemble`
    cannot build this, because it offsets weights per block as well.
    """
    n = forward.n_inputs
    inverse_half = tuple(
        replace(
            gate,
            params=tuple(_relocate_parameter(p, 0, n) for p in gate.params),
        )
        for gate in adjoint(forward).gates
    )
    return CircuitSpec(
        n_qubits=forward.n_qubits,
        n_weights=forward.n_weights,
        n_inputs=2 * n,
        gates=forward.gates + inverse_half,
    )


def _is_block(value: object) -> bool:
    return callable(getattr(value, "build", None)) and callable(
        getattr(value, "required_qubits", None)
    )


def _resolve_feature_map(
    feature_map: FeatureMapLike,
    n_features: int,
    reps: int,
    n_weights: int | None,
    n_qubits: int | None,
) -> CircuitSpec:
    """Turn any accepted `feature_map` into the forward `CircuitSpec` U(x; w)."""
    if isinstance(feature_map, str):
        if feature_map not in QuantumKernel._VALID_FEATURE_MAPS:
            raise ValueError(
                f"Unknown feature_map: {feature_map!r}. Expected one of "
                f"{QuantumKernel._VALID_FEATURE_MAPS}, or a custom map: a "
                "CircuitSpec, a CircuitBuilder, a block or list of blocks, or a "
                "callable emit(builder, x, w)."
            )
        spec = _build_feature_map_spec(n_features, feature_map, reps)
    elif isinstance(feature_map, CircuitSpec):
        spec = feature_map
    elif isinstance(feature_map, CircuitBuilder):
        spec = feature_map.build()
    elif _is_block(feature_map):
        spec = assemble([feature_map])
    elif (
        isinstance(feature_map, Sequence)
        and feature_map
        and all(_is_block(b) for b in feature_map)
    ):
        spec = assemble(list(feature_map))
    elif callable(feature_map):
        builder = CircuitBuilder(n_qubits=n_qubits or n_features)
        x = builder.add_inputs(n_features)
        w = builder.add_weights(n_weights or 0)
        feature_map(builder, x, w)
        spec = builder.build()
    else:
        raise TypeError(
            f"feature_map of type {type(feature_map).__name__} is not supported. "
            "Pass 'Angle', 'AngleCNOT', a CircuitSpec, a CircuitBuilder, a block or "
            "list of blocks, or a callable emit(builder, x, w)."
        )

    if spec.n_inputs != n_features:
        raise ValueError(
            f"the feature map reads {spec.n_inputs} input(s) but the kernel was "
            f"built for n_features={n_features}. Each feature must be one input."
        )
    if n_weights is not None and spec.n_weights != n_weights:
        raise ValueError(
            f"n_weights={n_weights} but the feature map declares "
            f"{spec.n_weights} weight(s)."
        )
    return spec


def _pair_indices(n_x: int, n_y: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Row/column indices of the kernel entries a run has to measure.

    A symmetric Gram matrix (`n_y is None`) needs only the pairs above the
    diagonal: the diagonal is a fidelity of a state with itself, exactly 1.
    """
    if n_y is None:
        return np.triu_indices(n_x, k=1)
    rows, cols = np.indices((n_x, n_y))
    return rows.ravel(), cols.ravel()


def _assemble_matrix(
    values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n_x: int,
    n_y: int | None,
) -> np.ndarray:
    """Place measured pair values back into a kernel matrix."""
    if values.shape != rows.shape:
        raise ValueError(
            f"expected {len(rows)} kernel entries, got {values.size}; "
            "wrong job id or n_x/n_y?"
        )
    if n_y is None:
        K = np.eye(n_x, dtype=np.float64)
        K[rows, cols] = values
        K[cols, rows] = values
        return K
    K = np.empty((n_x, n_y), dtype=np.float64)
    K[rows, cols] = values
    return K


def project_psd(K: np.ndarray) -> np.ndarray:
    """Return the nearest positive semidefinite matrix to a symmetric `K`.

    Shot noise can make a measured Gram matrix slightly indefinite, and a support
    vector machine solver handles such a matrix poorly. Setting the negative
    eigenvalues to zero gives the nearest positive semidefinite matrix in the
    Frobenius norm.

    Args:
        K (np.ndarray): A square, symmetric kernel matrix.

    Returns:
        The projected matrix, of the same shape.
    """
    K = 0.5 * (K + K.T)
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    return (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T


class KernelJob:
    """A kernel matrix submitted to a backend, to be collected later.

    [`QuantumKernel.submit_kernel_matrix()`][eigenkit.qml.QuantumKernel.submit_kernel_matrix]
    returns it. The handle records which pair each submitted circuit measured, so
    `result()` returns the assembled matrix instead of a flat list of
    probabilities. `save()` writes this information together with the job id, so
    that the matrix can be collected from a new session with
    [`QuantumKernel.retrieve_kernel_matrix()`][eigenkit.qml.QuantumKernel.retrieve_kernel_matrix].
    """

    def __init__(
        self,
        handle,
        n_x: int,
        n_y: int | None,
        psd_projection: bool = False,
    ):
        self._handle = handle
        self.n_x = n_x
        self.n_y = n_y
        self.psd_projection = psd_projection
        self._rows, self._cols = _pair_indices(n_x, n_y)

    @property
    def n_circuits(self) -> int:
        """The number of circuits that the job runs, one for each measured entry."""
        return len(self._rows)

    def job_id(self) -> str:
        """Return the identifier of the job on the backend."""
        return self._handle.job_id()

    def status(self) -> str:
        """Return the current status of the job, as a string from the backend."""
        return self._handle.status()

    def result(self) -> np.ndarray:
        """Wait until the job finishes, and return the kernel matrix.

        Returns:
            The kernel matrix, of shape `(n_x, n_x)` for a Gram matrix and
                `(n_x, n_y)` otherwise.
        """
        values = np.asarray(self._handle.result(), dtype=np.float64)
        K = _assemble_matrix(values, self._rows, self._cols, self.n_x, self.n_y)
        if self.psd_projection and self.n_y is None:
            K = project_psd(K)
        return K

    def save(self, path: str | Path) -> Path:
        """Write the job id and the layout of the pairs to a JSON file.

        The setting of `psd_projection` is written too.

        Args:
            path: Where to write the file. Its suffix is replaced by `.json`, or
                `.json` is added when it has none.

        Returns:
            The path of the written file.
        """
        path = Path(path)
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        path.write_text(
            json.dumps(
                {
                    "job_id": self.job_id(),
                    "n_x": self.n_x,
                    "n_y": self.n_y,
                    "psd_projection": self.psd_projection,
                },
                indent=2,
            )
        )
        return path

    def __repr__(self) -> str:
        shape = (self.n_x, self.n_x if self.n_y is None else self.n_y)
        return f"KernelJob(job_id={self.job_id()!r}, shape={shape}, n_circuits={self.n_circuits})"


class QuantumKernel:
    """Quantum kernel matrix `K[i, j] = fidelity(U(x_i), U(x_j))`.

    Args:
        n_features: The number of features. For the built-in maps, it is also the
            number of qubits.
        feature_map: The feature map. It is `"Angle"`,
            `"AngleCNOT"`, or a custom forward map `U(x; w)`. A custom map is a
            `CircuitSpec`, a `CircuitBuilder`, a block, a list of blocks, or a
            callable `emit(builder, x, w)` that writes the map onto a
            `CircuitBuilder`, given the input references `x` and the weight
            references `w`. A custom map must read exactly `n_features` inputs.
            Default: `"AngleCNOT"`.
        reps: The number of repetitions of a built-in feature map. `reps=1` uses
            only the `RX` encoding, and `reps>=2` adds repetitions of the `RZ`
            encoding (with ladders of controlled-NOT (CNOT) gates for
            `"AngleCNOT"`). It is ignored for a custom map. Default: `2`.
        provider: The backend. It takes the same options as
            `Circuit(provider=...)`. Default: `"numpy"`, which needs no extra.
        weights: The trainable weights of the feature map, of shape
            `(n_weights,)`. `None` uses zeros. Maps without weights ignore them.
        scale: The bandwidth. Every input is multiplied by it before it reaches
            the circuit, so `scale` sets how far apart two samples are in Hilbert
            space. Default: `1.0`.
        n_weights: The number of weights that a callable `feature_map` reserves.
            For any other map, it is read from the circuit and, if given, checked.
        n_qubits: The number of qubits of a callable `feature_map`. Default:
            `n_features`.
        psd_projection: If True, a sampled Gram matrix is projected onto the
            nearest positive semidefinite matrix. It only affects backends that
            use shots. Default: `False`.

    Example:
        ```python
        import numpy as np
        from eigenkit import QuantumKernel

        rng = np.random.default_rng(0)
        X_train, X_test = rng.uniform(0, np.pi, (8, 4)), rng.uniform(0, np.pi, (3, 4))

        qk = QuantumKernel(n_features=4, feature_map="AngleCNOT", reps=2)
        K_train = qk.compute_kernel_matrix(X_train)
        K_test = qk.compute_kernel_matrix(X_test, X_train)

        # A custom feature map with a trainable weight.
        def zz_map(qc, x, w):
            for q in range(len(x)):
                qc.ry(w[0], q)
                qc.h(q)
                qc.rz(x[q], q)
            qc.pauli_exponential([("Z", 0), ("Z", 1)], x[0])

        trainable = QuantumKernel(n_features=2, feature_map=zz_map, n_weights=1)
        K = trainable.compute_kernel_matrix(X_train[:, :2], weights=[0.3])
        ```
    """

    _VALID_FEATURE_MAPS = ("Angle", "AngleCNOT")

    def __init__(
        self,
        n_features: int,
        feature_map: FeatureMapLike = "AngleCNOT",
        reps: int = 2,
        provider: ProviderSpec = "numpy",
        *,
        weights: Sequence[float] | np.ndarray | None = None,
        scale: float = 1.0,
        n_weights: int | None = None,
        n_qubits: int | None = None,
        psd_projection: bool = False,
    ):
        if reps < 1:
            raise ValueError(f"reps must be >= 1, got {reps}.")

        self.n_features = n_features
        self.feature_map = feature_map
        self.reps = reps
        self.provider = provider
        self.scale = float(scale)
        self.psd_projection = psd_projection
        self._n_weights_arg = n_weights
        self._n_qubits_arg = n_qubits

        self._spec = _resolve_feature_map(
            feature_map, n_features, reps, n_weights, n_qubits
        )
        self._kernel_spec = _inversion_spec(self._spec)
        self.weights = self._check_weights(weights)

        # Resolve the provider and check the gate set now, so a bad provider
        # or an unsupported gate fails at construction. Nothing is compiled
        # yet: compiling against a QPU transpiles and warns about cost, and a
        # kernel that is only cloned by GridSearchCV should never pay for it.
        self._provider = resolve_provider(provider)
        validate_circuit(self._kernel_spec, self._provider.capabilities)

        self._inversion: Circuit | None = None
        self._forward: Circuit | None = None
        self._feature_map_circuit: Circuit | None = None
        self._fast_path_probed = False

    # ---- construction helpers ----

    def _check_weights(self, weights) -> np.ndarray | None:
        if weights is None:
            return None
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape != (self.n_weights,):
            raise ValueError(
                f"this kernel's feature map has {self.n_weights} weight(s); "
                f"got weights of shape {weights.shape}."
            )
        return weights

    def _weights_or_default(self, weights: np.ndarray | None = None) -> np.ndarray:
        if weights is not None:
            return self._check_weights(weights)
        if self.weights is not None:
            return self.weights
        return np.zeros(self.n_weights, dtype=np.float64)

    def _scaled(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X if self.scale == 1.0 else X * self.scale

    def _clone_kwargs(self) -> dict:
        # A custom map is passed on as its resolved spec, so cloning never
        # re-runs user code.
        feature_map = (
            self.feature_map if isinstance(self.feature_map, str) else self._spec
        )
        return {
            "n_features": self.n_features,
            "feature_map": feature_map,
            "reps": self.reps,
            "provider": self.provider,
            "weights": None if self.weights is None else self.weights.copy(),
            "scale": self.scale,
            "psd_projection": self.psd_projection,
        }

    def with_weights(self, weights: np.ndarray) -> "QuantumKernel":
        """Return a copy of this kernel with different trainable weights.

        Args:
            weights: The new weights, of shape `(n_weights,)`.

        Returns:
            A new kernel, which is not compiled yet.
        """
        return type(self)(**{**self._clone_kwargs(), "weights": weights})

    def __sklearn_clone__(self) -> "QuantumKernel":
        """Return a new copy of the kernel, not compiled yet, for `sklearn.base.clone`.

        `GridSearchCV`, `cross_val_score` and `Pipeline` use it. The copy has the
        same feature map, provider, weights, scale and `psd_projection`.
        """
        return type(self)(**self._clone_kwargs())

    def __getstate__(self) -> dict:
        # Compiled circuits hold backend objects (devices, services, transpiled
        # circuits) that need not survive a pickle; they are rebuilt on use.
        state = self.__dict__.copy()
        for key in ("_inversion", "_forward", "_feature_map_circuit", "_provider"):
            state[key] = None
        state["_fast_path_probed"] = False
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._provider = resolve_provider(self.provider)

    def to_json_dict(self) -> dict:
        """Return a JSON description of the kernel, for `.eigenkit` bundles.

        A custom feature map is stored as its `CircuitSpec`. Secret fields of the
        provider, such as access tokens, are never written.

        Returns:
            The description, as a dictionary.
        """
        from eigenkit.core.wire import circuit_spec_to_json

        from ._serialization import serialize_provider_param

        feature_map = (
            self.feature_map
            if isinstance(self.feature_map, str)
            else {"circuit_spec": circuit_spec_to_json(self._spec)}
        )
        return {
            "__quantum_kernel__": 1,
            "n_features": self.n_features,
            "feature_map": feature_map,
            "reps": self.reps,
            "provider": serialize_provider_param(self.provider),
            "weights": None if self.weights is None else self.weights.tolist(),
            "scale": self.scale,
            "psd_projection": self.psd_projection,
        }

    @classmethod
    def from_json_dict(
        cls, data: dict, provider: ProviderSpec | None = None
    ) -> "QuantumKernel":
        """Rebuild a kernel written by `to_json_dict()`.

        Args:
            data: The stored description.
            provider: A provider that replaces the stored one. It is needed when
                the stored provider held a secret, or to move a saved model onto
                another backend.

        Returns:
            The rebuilt kernel.
        """
        from eigenkit.core.wire import circuit_spec_from_json

        from ._serialization import deserialize_provider_param

        feature_map = data["feature_map"]
        if isinstance(feature_map, dict):
            feature_map = circuit_spec_from_json(feature_map["circuit_spec"])
        return cls(
            n_features=data["n_features"],
            feature_map=feature_map,
            reps=data.get("reps", 2),
            provider=provider
            if provider is not None
            else deserialize_provider_param(data.get("provider", "numpy")),
            weights=data.get("weights"),
            scale=data.get("scale", 1.0),
            psd_projection=data.get("psd_projection", False),
        )

    def __repr__(self) -> str:
        fmap = (
            repr(self.feature_map)
            if isinstance(self.feature_map, str)
            else f"<custom: {self._spec.n_qubits} qubits, {len(self._spec.gates)} gates>"
        )
        parts = [f"n_features={self.n_features}", f"feature_map={fmap}"]
        if isinstance(self.feature_map, str):
            parts.append(f"reps={self.reps}")
        parts.append(f"provider={self.provider!r}")
        if self.n_weights:
            parts.append(f"n_weights={self.n_weights}")
        if self.scale != 1.0:
            parts.append(f"scale={self.scale:g}")
        return f"QuantumKernel({', '.join(parts)})"

    # ---- public circuit access ----

    @property
    def n_qubits(self) -> int:
        """The number of qubits of the feature-map circuit."""
        return self._spec.n_qubits

    @property
    def n_weights(self) -> int:
        """The number of trainable weights in the feature map."""
        return self._spec.n_weights

    @property
    def feature_map_spec(self) -> CircuitSpec:
        """The forward feature map `U(x; w)`, as a `CircuitSpec`."""
        return self._spec

    @property
    def kernel_spec(self) -> CircuitSpec:
        """The inversion circuit `U(x_1; w) U(x_2; w)^dagger`, as a `CircuitSpec`.

        It has `2 * n_features` inputs.
        """
        return self._kernel_spec

    @property
    def circuit(self) -> Circuit:
        """The compiled inversion circuit.

        A backend with shots runs it for each kernel entry. It is compiled when it
        is first accessed.
        """
        if self._inversion is None:
            self._inversion = Circuit(
                self._kernel_spec, provider=self._provider, interface="numpy"
            )
        return self._inversion

    @property
    def feature_map_circuit(self) -> Circuit:
        """The compiled forward feature map `U(x; w)`.

        It is compiled when it is first accessed.
        """
        if self._forward is None:
            self._forward = Circuit(
                self._spec, provider=self._provider, interface="numpy"
            )
        return self._forward

    def draw(self, *, inversion: bool = False, decimals: int = 3) -> str:
        """Return a text drawing of the feature map, or of the inversion circuit.

        Args:
            inversion: If True, draw `U(x_1) U(x_2)^dagger` instead of `U(x)`.
            decimals: The number of significant digits of constant angles.

        Returns:
            The drawing, as a string of several lines.
        """
        from eigenkit.core.draw import draw_text

        return draw_text(
            self._kernel_spec if inversion else self._spec, decimals=decimals
        )

    def draw_mpl(
        self, *, inversion: bool = False, **kwargs: Any
    ) -> "tuple[Figure, Axes]":
        """Draw the feature map, or the inversion circuit, with Matplotlib.

        The inversion circuit is drawn with a measurement on every qubit, and the
        forward map without one. This method needs the `viz` extra.

        Args:
            inversion: If True, draw `U(x_1) U(x_2)^dagger` instead of `U(x)`.
            **kwargs (Any): Arguments passed to
                [`draw_mpl()`][eigenkit.core.draw_mpl.draw_mpl], for example
                `style="eigen"`.

        Returns:
            The Matplotlib figure and axes, as the pair `(fig, ax)`.
        """
        from eigenkit.core.draw_mpl import draw_mpl

        spec = self._kernel_spec if inversion else self._spec
        observables = (
            [Observable.z(q) for q in range(spec.n_qubits)] if inversion else None
        )
        kwargs.setdefault("show_measurements", inversion)
        return draw_mpl(spec, observables=observables, **kwargs)

    # ---- evaluation ----

    def _statevector_feature_map(self) -> Circuit | None:
        """The feature-map circuit when this backend can evolve a state, else
        `None`. Probed once, on first use.

        A backend whose compiled handle has no `statevector` at all (Qiskit
        Runtime, IQM) is ruled out without compiling the forward map, so a
        hardware kernel transpiles one circuit and warns about cost once.
        Otherwise the decision is made by trying it, not by `hasattr`: a
        shots-configured backend has a `statevector` method that raises when
        called ("requires analytic mode").

        Deferred rather than done at construction, because the probe *runs a
        circuit*. Any exception means no fast path: the result is a capability
        hint, not a computed value.
        """
        if not self._fast_path_probed:
            self._fast_path_probed = True
            self._feature_map_circuit = None
            if getattr(self.circuit.compiled, "statevector", None) is not None:
                forward = self.feature_map_circuit
                try:
                    forward.statevector(
                        self._weights_or_default(), np.zeros(forward.spec.n_inputs)
                    )
                except Exception:  # noqa: BLE001 — a hint, so any failure means "no"
                    self._feature_map_circuit = None
                else:
                    self._feature_map_circuit = forward
        return self._feature_map_circuit

    def kernel_value(
        self, x1: np.ndarray, x2: np.ndarray, *, weights: np.ndarray | None = None
    ) -> float:
        """Return the kernel value of one pair `(x1, x2)`, from the inversion circuit."""
        joint = np.concatenate([self._scaled(x1), self._scaled(x2)])
        return self.circuit.zero_state_probability(
            self._weights_or_default(weights), joint
        )

    def _joint_inputs(self, X: np.ndarray, Y: np.ndarray | None):
        """Every pair a run measures, as `(B, 2 * n_features)` circuit inputs."""
        n_y = None if Y is None else Y.shape[0]
        rows, cols = _pair_indices(X.shape[0], n_y)
        other = X if Y is None else Y
        return np.concatenate([X[rows], other[cols]], axis=1), rows, cols

    def compute_kernel_matrix(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        *,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the kernel matrix between the rows of `X` and the rows of `Y`.

        Args:
            X: The samples, of shape `(n_X, n_features)`.
            Y: The second set of samples, of shape `(n_Y, n_features)`. If `None`,
                the method returns `K(X, X)`.
            weights: The feature-map weights for this evaluation. By default, they
                are the kernel's own `weights`, or zeros if it has none.

        Returns:
            An array of shape `(n_X, n_Y)`.
        """
        X_arr = self._scaled(X)
        Y_arr = None if Y is None else self._scaled(Y)
        w = self._weights_or_default(weights)

        # Fast path: every entry is a state overlap, so evolve each feature
        # state once and read the whole Gram matrix off a single contraction.
        feature_map_circuit = self._statevector_feature_map()
        if feature_map_circuit is not None:
            psi_x = feature_map_circuit.statevector(w, X_arr)
            psi_y = psi_x if Y_arr is None else feature_map_circuit.statevector(w, Y_arr)
            return np.abs(psi_x.conj() @ psi_y.T) ** 2

        # Inversion circuit: every pair in one job on hardware; a local
        # simulator gets a row's worth of pairs per call so memory stays
        # O(n_y). A Gram matrix measures only i < j.
        n_x = X_arr.shape[0]
        n_y = None if Y_arr is None else Y_arr.shape[0]
        joints, rows, cols = self._joint_inputs(X_arr, Y_arr)
        values = np.empty(0, dtype=np.float64)
        if len(joints):
            hardware = hasattr(self.circuit.compiled, "submit_zero_state_probabilities")
            step = len(joints) if hardware else (n_y or n_x)
            values = np.concatenate(
                [
                    self.circuit.zero_state_probabilities(w, joints[i : i + step])
                    for i in range(0, len(joints), step)
                ]
            )
        K = _assemble_matrix(values, rows, cols, n_x, n_y)
        if self.psd_projection and n_y is None:
            K = project_psd(K)
        return K

    # ---- hardware workflow ----

    def estimate_cost(self, X: np.ndarray, Y: np.ndarray | None = None) -> dict:
        """Return what computing `K(X, Y)` would need on the backend of this kernel.

        Nothing is submitted. On a sampling backend, the inversion circuit is
        compiled (and transpiled, for hardware) to report its size.

        Args:
            X: The samples, of shape `(n_X, n_features)`.
            Y: The second set of samples, of shape `(n_Y, n_features)`, or `None`
                for the Gram matrix `K(X, X)`.

        Returns:
            A dictionary with the keys `method` (`"statevector"` or
                `"inversion"`), `n_circuits`, `n_jobs`, `shots` and `total_shots`.
                With the `"statevector"` method, it also has `n_state_evolutions`.
                When the backend reports them, it also has the transpiled
                two-qubit gate count `n_2q_gates`, the `depth` and the
                `backend_name`.
        """
        n_x = np.asarray(X).shape[0]
        n_y = None if Y is None else np.asarray(Y).shape[0]
        if self._statevector_feature_map() is not None:
            n_states = n_x + (0 if n_y is None else n_y)
            return {
                "method": "statevector",
                "n_state_evolutions": n_states,
                "n_circuits": 0,
                "n_jobs": 0,
                "shots": None,
                "total_shots": 0,
            }
        rows, _ = _pair_indices(n_x, n_y)
        compiled = self.circuit.compiled
        batched = getattr(compiled, "zero_state_probabilities", None) is not None
        info: dict = {
            "method": "inversion",
            "n_circuits": int(len(rows)),
            "n_jobs": 1 if batched else int(len(rows)),
        }
        shots = getattr(getattr(compiled, "_config", None), "shots", None)
        info["shots"] = shots
        info["total_shots"] = None if shots is None else int(shots) * int(len(rows))
        resources = getattr(compiled, "resources", None)
        if resources is not None:
            report = resources()
            info["n_2q_gates"] = report.get("n_2q_gates")
            info["depth"] = report.get("depth")
            info["backend_name"] = report.get("backend_name")
        return info

    def submit_kernel_matrix(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        *,
        weights: np.ndarray | None = None,
    ) -> KernelJob:
        """Submit `K(X, Y)` as one asynchronous job, and return its handle.

        This is for the queues of a quantum processing unit (QPU), which can take
        minutes or hours. Print the job id of the handle or call its `save()`, and
        collect the matrix later with `job.result()` or, from a new session, with
        `retrieve_kernel_matrix()`.

        Args:
            X: The samples, of shape `(n_X, n_features)`.
            Y: The second set of samples, of shape `(n_Y, n_features)`, or `None`
                for the Gram matrix.
            weights: The feature-map weights. By default, they are the kernel's
                own.

        Returns:
            A [`KernelJob`][eigenkit.qml._kernels.KernelJob].

        Raises:
            NotImplementedError: On backends without asynchronous submission,
                which includes every local simulator. Use
                `compute_kernel_matrix()` on them.
        """
        submit = getattr(self.circuit.compiled, "submit_zero_state_probabilities", None)
        if submit is None:
            raise NotImplementedError(
                f"backend '{self.circuit.provider.name}' does not submit kernel "
                "jobs asynchronously; use compute_kernel_matrix() instead."
            )
        X_arr = self._scaled(X)
        Y_arr = None if Y is None else self._scaled(Y)
        joints, _, _ = self._joint_inputs(X_arr, Y_arr)
        if not len(joints):
            raise ValueError(
                "no kernel entries to measure; a 1-sample Gram matrix is [[1.0]]"
            )
        handle = submit(self._weights_or_default(weights), joints)
        return KernelJob(
            handle,
            n_x=X_arr.shape[0],
            n_y=None if Y_arr is None else Y_arr.shape[0],
            psd_projection=self.psd_projection,
        )

    def retrieve_kernel_matrix(
        self,
        job: str | Path,
        n_x: int | None = None,
        n_y: int | None = None,
    ) -> KernelJob:
        """Reconnect to a kernel job submitted earlier, possibly in another session.

        Args:
            job: The JSON file written by `KernelJob.save()`, or a job id on its
                own.
            n_x: The number of rows of the matrix. It is required with a job id on
                its own.
            n_y: The number of columns, for a rectangular matrix. Leave it as
                `None` for a Gram matrix.

        Returns:
            A [`KernelJob`][eigenkit.qml._kernels.KernelJob]. Call `.result()` on
                it to get the matrix.
        """
        psd = self.psd_projection
        path = Path(str(job))
        if path.suffix == ".json" and path.exists():
            saved = json.loads(path.read_text())
            job_id, n_x, n_y = saved["job_id"], saved["n_x"], saved["n_y"]
            psd = saved.get("psd_projection", psd)
        else:
            job_id = str(job)
            if n_x is None:
                raise ValueError(
                    "pass n_x (and n_y for a rectangular matrix) with a bare job id."
                )
        retrieve = getattr(
            self.circuit.compiled, "retrieve_zero_state_probabilities", None
        )
        if retrieve is None:
            raise NotImplementedError(
                f"backend '{self.circuit.provider.name}' cannot re-attach to a job by id."
            )
        return KernelJob(retrieve(job_id), n_x=n_x, n_y=n_y, psd_projection=psd)


def _estimator_kernel(estimator, n_features: int, *, warn: bool = True) -> QuantumKernel:
    """The kernel a QSVC/QSVR fits with: its `kernel` parameter when given,
    otherwise one built from `feature_map` / `reps` / `provider`.

    Args:
        estimator: The QSVC or QSVR being fitted.
        n_features: Columns of the training data.
        warn: Warn when `kernel` is given alongside non-default values of the
            parameters it overrides.

    Raises:
        TypeError: If `kernel` is not a `QuantumKernel`.
        ValueError: If the kernel was built for a different number of features.
    """
    kernel = getattr(estimator, "kernel", None)
    if kernel is None:
        return QuantumKernel(
            n_features=n_features,
            feature_map=estimator.feature_map,
            reps=estimator.reps,
            provider=estimator.provider,
        )
    name = type(estimator).__name__
    if not isinstance(kernel, QuantumKernel):
        raise TypeError(
            f"{name}(kernel=...) expects a QuantumKernel, got {type(kernel).__name__}."
        )
    if kernel.n_features != n_features:
        raise ValueError(
            f"the kernel was built for n_features={kernel.n_features}, but X has "
            f"{n_features} columns."
        )
    if warn:
        import inspect
        import warnings

        defaults = inspect.signature(type(estimator).__init__).parameters
        overridden = [
            key
            for key in ("feature_map", "reps", "provider")
            if key in defaults and getattr(estimator, key) != defaults[key].default
        ]
        if overridden:
            warnings.warn(
                f"{name}(kernel=...) ignores {', '.join(overridden)}; set them on "
                "the QuantumKernel instead.",
                UserWarning,
                stacklevel=3,
            )
    return kernel
