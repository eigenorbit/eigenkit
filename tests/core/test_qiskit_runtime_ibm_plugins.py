"""IBM translation plugins: real IBM backends transpile through them.

A real `IBMBackend` routes the translation stage through the plugins that
qiskit-ibm-runtime registers (`ibm_backend`, `ibm_dynamic_circuits`, ...).
The fake backends do not, so the rest of the Qiskit Runtime suite never loads
them. This file drives them explicitly against `FakeFez`, offline, so a
qiskit-ibm-runtime release whose plugins cannot import on the pinned qiskit
fails here instead of on hardware.

qiskit-ibm-runtime 0.44.x imports `calc_final_ops` from qiskit at module
level; it exists only from qiskit 2.0, so on qiskit 1.x every IBM plugin
fails to import.
"""

from __future__ import annotations

from importlib.metadata import requires

import pytest


pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")

from packaging.requirements import Requirement  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeFez  # noqa: E402


@pytest.mark.parametrize("translation_method", ["ibm_backend", "ibm_dynamic_circuits"])
def test_transpiles_through_ibm_translation_plugin(translation_method):
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()

    pass_manager = generate_preset_pass_manager(
        optimization_level=1, backend=FakeFez(), translation_method=translation_method
    )
    transpiled = pass_manager.run(circuit)

    assert set(transpiled.count_ops()) <= {"rz", "sx", "x", "measure", "barrier"}


def _extra_requirement(extra: str, name: str) -> Requirement:
    for raw in requires("eigenkit") or []:
        req = Requirement(raw)
        if req.name == name and req.marker and req.marker.evaluate({"extra": extra}):
            return req
    raise AssertionError(f"{name} is not declared in the {extra!r} extra")


def test_qiskit_extra_excludes_runtime_releases_that_need_qiskit_2():
    """A fresh resolve of the `qiskit` extra must not pair qiskit 1.x with a
    qiskit-ibm-runtime whose plugins need qiskit 2. Only 0.44.x does that:
    later releases require qiskit 2 themselves, so they are safe to allow."""
    qiskit = _extra_requirement("qiskit", "qiskit")
    runtime = _extra_requirement("qiskit", "qiskit-ibm-runtime")
    if not qiskit.specifier.contains("1.4.1"):
        pytest.skip("the qiskit extra requires qiskit 2; 0.44.x is safe there")
    assert not runtime.specifier.contains("0.44.0")
    assert not runtime.specifier.contains("0.44.1")
