"""Shared pieces of the sampling-hardware kernel path.

Two things the Qiskit Runtime and IQM backends both need:

- a job handle that reduces a submitted sampling job to all-zeros
  probabilities, which is what an inversion-circuit kernel reads;
- a report of what a transpiled circuit costs on its target — two-qubit gate
  count and depth, the physical qubits it landed on, how much of that is
  routing, and the success probability the target's own error rates predict.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np


def job_result(job, timeout: float | None):
    """`job.result()`, with the timeout passed when the job type accepts one.

    Runtime jobs (`RuntimeJobV2`) and IQM jobs take `timeout=`; the local
    `PrimitiveJob` a fake backend returns does not.
    """
    try:
        accepts_timeout = "timeout" in inspect.signature(job.result).parameters
    except (TypeError, ValueError):
        accepts_timeout = False
    return job.result(timeout=timeout) if accepts_timeout else job.result()


class ProbabilityJobHandle:
    """A submitted sampling job whose result is one all-zeros probability per
    submitted circuit.

    Returned by `submit_zero_state_probabilities()` on the Qiskit Runtime and
    IQM compiled circuits, and by their `retrieve_zero_state_probabilities()`
    for a job submitted earlier.
    """

    def __init__(
        self,
        job,
        reduce: Callable[[Any], np.ndarray],
        poll_timeout_seconds: float | None,
    ):
        """Wrap a job.

        Args:
            job: The SDK job object.
            reduce: Turns the job's result into the probability array.
            poll_timeout_seconds: How long `result()` waits before giving up.
        """
        self._job = job
        self._reduce = reduce
        self._poll_timeout_seconds = poll_timeout_seconds

    def status(self) -> str:
        """Job state as the SDK reports it, e.g. `"QUEUED"` or `"DONE"`. Does
        not block."""
        state = self._job.status()
        return state.name if hasattr(state, "name") else str(state)

    def cancel(self) -> None:
        """Ask the provider to cancel the job. Does not block."""
        self._job.cancel()

    def job_id(self) -> str:
        """The provider's job id."""
        return str(self._job.job_id())

    def result(self) -> np.ndarray:
        """Block until the job finishes; return one probability per circuit."""
        return self._reduce(job_result(self._job, self._poll_timeout_seconds))


def zero_fraction(counts: dict[str, int], n_qubits: int) -> float:
    """Share of shots that measured the all-zeros bitstring."""
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError("the job returned an empty histogram")
    return counts.get("0" * n_qubits, 0) / total


def _two_qubit(instruction) -> bool:
    operation = instruction.operation
    return operation.num_qubits == 2 and operation.name not in ("barrier", "delay")


def _estimated_success_probability(circuit, backend) -> float | None:
    """Product of `1 - error` over every operation the target has an error
    rate for, readout included. `None` when the target reports none."""
    target = getattr(backend, "target", None)
    if target is None:
        return None
    probability = 1.0
    found = False
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        try:
            properties = target[name][qargs]
        except KeyError:
            continue
        error = getattr(properties, "error", None)
        if error is None:
            continue
        found = True
        probability *= 1.0 - float(error)
    return probability if found else None


def _basis_gates(backend, exclude: frozenset[str]) -> list[str]:
    """The target's native gate names that `transpile(basis_gates=...)` can
    use without a coupling map: standard gates only, so device-specific
    operations such as IQM's MOVE drop out."""
    from qiskit.circuit.library import get_standard_gate_name_mapping

    standard = get_standard_gate_name_mapping()
    names = getattr(backend, "operation_names", None) or []
    return sorted(name for name in names if name in standard and name not in exclude) + [
        "measure"
    ]


def transpiled_resources(
    transpiled,
    backend,
    logical,
    *,
    optimization_level: int,
    seed_transpiler: int | None,
    exclude_from_basis: frozenset[str] = frozenset({"delay", "reset", "measure"}),
) -> dict:
    """What `transpiled` costs on `backend`.

    Args:
        transpiled: The circuit as it will be submitted.
        backend: Its target.
        logical: The same circuit before transpilation, used for the routing
            baseline: transpiled to the target's native gates with all-to-all
            connectivity, so the difference to `transpiled` is what the
            device's connectivity adds.
        optimization_level: Used for the baseline as well.
        seed_transpiler: Used for the baseline as well.
        exclude_from_basis: Operations left out of the baseline's basis.

    Returns:
        A dict with `gate_counts`, `n_2q_gates`, `two_qubit_gate_counts`,
        `depth`, `depth_2q`, `n_2q_gates_unrouted`, `routing_overhead_2q`,
        `physical_qubits`, `n_physical_qubits_device` and
        `estimated_success_probability`.
    """
    from qiskit import transpile

    two_qubit_counts: dict[str, int] = {}
    for instruction in transpiled.data:
        if _two_qubit(instruction):
            name = instruction.operation.name
            two_qubit_counts[name] = two_qubit_counts.get(name, 0) + 1
    n_2q = sum(two_qubit_counts.values())

    unrouted = transpile(
        logical,
        basis_gates=_basis_gates(backend, exclude_from_basis),
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    n_2q_unrouted = sum(1 for instruction in unrouted.data if _two_qubit(instruction))

    layout = getattr(transpiled, "layout", None)
    physical_qubits = None
    if layout is not None:
        try:
            physical_qubits = list(layout.initial_index_layout(filter_ancillas=True))
        except (AttributeError, TypeError):
            physical_qubits = None

    return {
        "gate_counts": dict(transpiled.count_ops()),
        "n_2q_gates": n_2q,
        "two_qubit_gate_counts": two_qubit_counts,
        "depth": transpiled.depth(),
        "depth_2q": transpiled.depth(_two_qubit),
        "n_2q_gates_unrouted": n_2q_unrouted,
        "routing_overhead_2q": n_2q - n_2q_unrouted,
        "physical_qubits": physical_qubits,
        "n_physical_qubits_device": getattr(backend, "num_qubits", None),
        "estimated_success_probability": _estimated_success_probability(
            transpiled, backend
        ),
    }
