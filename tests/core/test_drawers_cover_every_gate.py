"""Both drawers must render every gate, on every qubit it acts on.

A circuit diagram is documentation people trust to understand what their
circuit does, so one that disagrees with the IR is a correctness bug.

Two failures matter. A drawer that does not know a multi-qubit gate draws it
as a single box on `qubits[0]` with its remaining wires left blank, and for a
controlled rotation puts the label on the *control* while the qubit it actually
rotates reads as idle. A drawer that raises `NotImplementedError` for a gate
cannot draw any circuit that uses it -- QFT, Grover, HHL and UCCSD among them.

Both drawers are swept over the whole gate table, so a gate either one does not
handle fails here.
"""

from __future__ import annotations

import importlib.util

import pytest

from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import (
    ALL_GATES,
    CircuitSpec,
    Constant,
    Gate,
    Observable,
    PauliTerm,
)


# The text drawer needs nothing beyond the core, so it is checked even on an
# install without matplotlib -- which is exactly what the
# `test-missing-importorskip` CI job builds. Only the matplotlib class is
# gated; a module-level `importorskip` would have skipped the text sweep too.
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
requires_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB, reason="matplotlib is not installed"
)


OBSERVABLE = Observable(terms=(PauliTerm(1.0, (("Z", 0),)),))
_UNITARY_1Q = ((0.0, 1.0), (1.0, 0.0))
_AMPLITUDES = (0.5, 0.5, 0.5, 0.5)

# One runnable instance of every gate in the vocabulary, on 3 qubits.
GATE_FIXTURES: dict[str, Gate] = {
    "RX": Gate("RX", (0,), (Constant(0.3),)),
    "RY": Gate("RY", (0,), (Constant(0.3),)),
    "RZ": Gate("RZ", (0,), (Constant(0.3),)),
    "PhaseShift": Gate("PhaseShift", (0,), (Constant(0.3),)),
    "CRY": Gate("CRY", (0, 2), (Constant(0.3),)),
    "CPHASE": Gate("CPHASE", (0, 2), (Constant(0.3),)),
    "H": Gate("H", (0,)),
    "X": Gate("X", (0,)),
    "Z": Gate("Z", (0,)),
    "S": Gate("S", (0,)),
    "Sdg": Gate("Sdg", (0,)),
    "CNOT": Gate("CNOT", (0, 2)),
    "SWAP": Gate("SWAP", (0, 2)),
    "MCZ": Gate("MCZ", (0, 2)),
    "MCX": Gate("MCX", (0, 2)),
    "QubitUnitary": Gate("QubitUnitary", (0,), data=(_UNITARY_1Q,)),
    "AmplitudeEncoding": Gate("AmplitudeEncoding", (0, 1), data=(_AMPLITUDES,)),
    "ControlledQubitUnitary": Gate(
        "ControlledQubitUnitary", (0, 2), data=(_UNITARY_1Q, None)
    ),
}
IDS = sorted(GATE_FIXTURES)


def _pyplot():
    """pyplot on a headless backend, imported only where matplotlib is present."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _circuit(gate: Gate) -> Circuit:
    spec = CircuitSpec(n_qubits=3, n_weights=0, n_inputs=0, gates=(gate,))
    return Circuit(spec, OBSERVABLE, "numpy", "numpy")


def test_the_fixture_table_covers_the_whole_vocabulary():
    """Otherwise a new gate is silently exempt from this file."""
    assert set(GATE_FIXTURES) == set(ALL_GATES)


class TestTextDrawer:
    @pytest.mark.parametrize("name", IDS)
    def test_it_renders(self, name):
        art = _circuit(GATE_FIXTURES[name]).draw()
        assert art.strip(), f"{name} rendered nothing"

    @pytest.mark.parametrize("name", IDS)
    def test_every_qubit_the_gate_acts_on_is_marked(self, name):
        """The failure this guards: a two-qubit gate drawn on one wire, with
        the other left looking idle."""
        gate = GATE_FIXTURES[name]
        rows = [ln for ln in _circuit(gate).draw().splitlines() if ln.startswith("q_")]
        assert len(rows) == 3
        for qubit in set(gate.qubits):
            body = rows[qubit].split(":", 1)[1]
            assert body.strip("─ \t"), (
                f"{name} acts on q{qubit} but that wire is empty:\n" + "\n".join(rows)
            )

    @pytest.mark.parametrize("name", IDS)
    def test_untouched_qubits_stay_empty(self, name):
        gate = GATE_FIXTURES[name]
        rows = [ln for ln in _circuit(gate).draw().splitlines() if ln.startswith("q_")]
        for qubit in set(range(3)) - set(gate.qubits):
            body = rows[qubit].split(":", 1)[1]
            assert not body.strip("─ \t"), f"{name} marked untouched q{qubit}"

    def test_a_controlled_rotation_labels_the_qubit_it_rotates(self):
        """`CRY(control=2, target=0)` puts the label on q0, not the control q2."""
        gate = Gate("CRY", (2, 0), (Constant(1.0),))
        rows = [ln for ln in _circuit(gate).draw().splitlines() if ln.startswith("q_")]
        assert "CRY" in rows[0], "label not on the rotated qubit:\n" + "\n".join(rows)
        assert "CRY" not in rows[2], "label on the control"

    def test_an_adjoint_gate_is_distinguishable(self):
        plain = _circuit(Gate("QubitUnitary", (0,), data=(_UNITARY_1Q,))).draw()
        dagger = _circuit(
            Gate("QubitUnitary", (0,), data=(_UNITARY_1Q,), adjoint=True)
        ).draw()
        assert plain != dagger, "U and U-dagger render identically"


@requires_matplotlib
class TestMatplotlibDrawer:
    @pytest.mark.parametrize("name", IDS)
    def test_it_renders(self, name):
        plt = _pyplot()
        figure, axes = _circuit(GATE_FIXTURES[name]).draw_mpl()
        try:
            assert axes.patches or axes.lines or axes.texts, f"{name} drew nothing"
        finally:
            plt.close(figure)

    def test_drawing_repeatedly_does_not_leak_figures(self):
        plt = _pyplot()
        plt.close("all")
        circuit = _circuit(Gate("CNOT", (0, 2)))
        owned = plt.figure()
        try:
            for _ in range(30):
                circuit.draw_mpl()
            assert len(plt.get_fignums()) == 1, "diagrams were left open"
        finally:
            plt.close("all")
            del owned

    def test_drawing_does_not_steal_the_active_figure(self):
        plt = _pyplot()
        plt.close("all")
        mine = plt.figure()
        try:
            _circuit(Gate("CNOT", (0, 2))).draw_mpl()
            assert plt.gcf() is mine
        finally:
            plt.close("all")
