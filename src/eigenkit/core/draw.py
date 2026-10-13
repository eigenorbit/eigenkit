"""Text drawer for `CircuitSpec`, which does not depend on the provider.

It draws the circuit as a string of several lines:

```text
q_0: ──RX(x_0)──RZ(w_0)──●──H──
                         │
q_1: ──RY(x_1)───────────X─────
```

A single-qubit gate is drawn as a `NAME(param)` label on its wire. A
controlled-NOT (CNOT) gate is drawn as `●` on the control and `X` on the target,
with `│` connectors between the wires that it spans.

Parameter expressions are drawn as symbols:

```text
Constant(0.5)              -> "0.5"
WeightRef(3)               -> "w_3"
InputRef(0)                -> "x_0"
Scale(InputRef(0), 2*pi)   -> "6.28*x_0"
Negate(WeightRef(0))       -> "-w_0"
Sum((w_0, x_0))            -> "w_0+x_0"
Multiply(w_0, x_0)         -> "w_0*x_0"
```
"""

from __future__ import annotations

from .ir import (
    CircuitSpec,
    Constant,
    Gate,
    InputRef,
    Multiply,
    Negate,
    Parameter,
    Scale,
    Sum,
    WeightRef,
)


def draw_text(spec: CircuitSpec, *, decimals: int = 3) -> str:
    """Draw `spec` as text on several lines.

    The text is made of American Standard Code for Information Interchange (ASCII)
    characters and a few box-drawing characters.

    Args:
        spec: The `CircuitSpec` to draw.
        decimals: The number of significant digits of constant angles. Default:
            `3`.

    Returns:
        The drawing, as a string. Print it to see the circuit.
    """
    n = spec.n_qubits
    if n == 0:
        return "(empty circuit: 0 qubits)"

    # 1. Assign each gate to a column.
    placements, n_cols = _layout(spec)

    # 2. Build the (col, qubit) -> label grid + the vertical spans linking the
    # qubits of one multi-qubit gate.
    #
    # Every multi-qubit gate marks each wire it acts on. A single box on
    # `qubits[0]` would show a SWAP touching one wire, and a controlled
    # rotation with its label on the *control* while the qubit it actually
    # rotates reads as idle. A diagram is documentation, so one that disagrees
    # with the IR is wrong in the way that matters most.
    label_grid: dict[tuple[int, int], str] = {}
    spans: list[tuple[int, int, int]] = []  # (col, lo, hi)
    for col, gate in placements:
        marks = _qubit_marks(gate, decimals=decimals)
        for qubit, mark in marks.items():
            label_grid[(col, qubit)] = mark
        if len(marks) > 1:
            spans.append((col, min(marks), max(marks)))

    # 3. Column widths: max label width per column (minimum 1).
    col_widths = [1] * n_cols
    for (col, _), label in label_grid.items():
        col_widths[col] = max(col_widths[col], _display_width(label))

    # 4. Render row-by-row.
    qubit_width = len(str(n - 1))
    label_prefix_width = len(f"q_{0:>{qubit_width}}: ")

    lines: list[str] = []
    for q in range(n):
        prefix = f"q_{q:>{qubit_width}}: "
        parts = [prefix]
        for col in range(n_cols):
            width = col_widths[col]
            label = label_grid.get((col, q))
            if label is None:
                cell = "─" * width
            else:
                pad = width - _display_width(label)
                left = pad // 2
                right = pad - left
                cell = "─" * left + label + "─" * right
            parts.append("──" + cell)
        if n_cols > 0:
            parts.append("──")
        lines.append("".join(parts))

        # Connector row between qubit q and q+1.
        if q < n - 1:
            connector_prefix = " " * label_prefix_width
            row = [connector_prefix]
            for col in range(n_cols):
                width = col_widths[col]
                has_span = any(c == col and lo <= q < hi for c, lo, hi in spans)
                if has_span:
                    pad = width - 1
                    left = pad // 2
                    right = pad - left
                    cell = " " * left + "│" + " " * right
                else:
                    cell = " " * width
                row.append("  " + cell)
            if n_cols > 0:
                row.append("  ")
            lines.append("".join(row).rstrip() or connector_prefix)

    return "\n".join(lines)


def _layout(spec: CircuitSpec) -> tuple[list[tuple[int, Gate]], int]:
    """Pack gates into columns left-to-right without overlap.

    A multi-qubit gate occupies the smallest column where all its qubits
    (and any qubits it spans) are free.
    """
    n = spec.n_qubits
    next_col = [0] * n
    placements: list[tuple[int, Gate]] = []

    for gate in spec.gates:
        if len(gate.qubits) == 1:
            q = gate.qubits[0]
            col = next_col[q]
            placements.append((col, gate))
            next_col[q] = col + 1
        else:
            lo = min(gate.qubits)
            hi = max(gate.qubits)
            col = max(next_col[lo : hi + 1])
            placements.append((col, gate))
            for q in range(lo, hi + 1):
                next_col[q] = col + 1

    n_cols = max(next_col) if next_col else 0
    return placements, n_cols


def _qubit_marks(gate: Gate, decimals: int) -> dict[int, str]:
    """What to print on each qubit this gate touches.

    Returns a mapping from qubit index to its cell text, so a multi-qubit gate
    marks every wire it acts on rather than only the first. Controls are `●`,
    an open control (`control_values` entry of 0) is `○`, a CNOT target is `X`,
    a SWAP end is `×`, and anything else gets the gate's label.

    Args:
        gate (Gate): The gate to place.
        decimals (int): Significant digits for a numeric angle.

    Returns:
        `{qubit: cell text}` for every qubit the gate acts on.
    """
    qubits = gate.qubits
    name = gate.name
    label = _gate_label(gate, decimals=decimals)

    if name == "CNOT":
        return {qubits[0]: "●", qubits[1]: "X"}
    if name == "SWAP":
        return {qubits[0]: "×", qubits[1]: "×"}
    if name == "MCX":
        return {**{q: "●" for q in qubits[:-1]}, qubits[-1]: "X"}
    if name == "MCZ":
        return {**{q: "●" for q in qubits[:-1]}, qubits[-1]: "Z"}
    if name in ("CRY", "CPHASE"):
        # The label belongs on the qubit the rotation acts on, not the control.
        return {qubits[0]: "●", qubits[1]: label}
    if name == "ControlledQubitUnitary":
        # `data[1]` holds the control values when present; 0 means an open
        # control, drawn hollow the way every other renderer draws it.
        n_target = _controlled_unitary_target_count(gate)
        controls = qubits[:-n_target] if n_target else qubits[:-1]
        targets = qubits[len(controls) :]
        values = gate.data[1] if len(gate.data) > 1 and gate.data[1] else None
        marks = {}
        for i, q in enumerate(controls):
            open_control = values is not None and i < len(values) and not values[i]
            marks[q] = "○" if open_control else "●"
        for q in targets:
            marks[q] = label
        return marks
    return {q: label for q in qubits}


def _controlled_unitary_target_count(gate: Gate) -> int:
    """How many of a ControlledQubitUnitary's qubits the matrix acts on."""
    try:
        size = len(gate.data[0])
    except (IndexError, TypeError):
        return 1
    count = max(size.bit_length() - 1, 1)
    return min(count, max(len(gate.qubits) - 1, 1))


def _gate_label(gate: Gate, decimals: int) -> str:
    # The dagger is part of what the gate is: without it `U` and `U-dagger`
    # would render identically, and a Grover iterate would read as its state
    # preparation applied twice.
    name = f"{gate.name}†" if gate.adjoint else gate.name
    if not gate.params:
        return name
    return f"{name}({_param_repr(gate.params[0], decimals=decimals)})"


def _param_repr(p: Parameter, decimals: int) -> str:
    if isinstance(p, Constant):
        return f"{p.value:.{decimals}g}"
    if isinstance(p, WeightRef):
        return f"w_{p.index}"
    if isinstance(p, InputRef):
        return f"x_{p.index}"
    if isinstance(p, Scale):
        return f"{p.factor:.{decimals}g}*{_param_repr(p.expression, decimals)}"
    if isinstance(p, Negate):
        inner = _param_repr(p.expression, decimals)
        # Wrap composite inner expressions so "-a+b" doesn't read as "(-a)+b".
        if isinstance(p.expression, (Sum, Scale, Multiply)):
            inner = f"({inner})"
        return f"-{inner}"
    if isinstance(p, Sum):
        return "+".join(_param_repr(e, decimals) for e in p.expressions)
    if isinstance(p, Multiply):
        a = _param_repr(p.a, decimals)
        b = _param_repr(p.b, decimals)
        if isinstance(p.a, Sum):
            a = f"({a})"
        if isinstance(p.b, Sum):
            b = f"({b})"
        return f"{a}*{b}"
    raise TypeError(f"unknown parameter kind: {type(p).__name__}")


def _display_width(text: str) -> int:
    """Visual width in monospace cells.

    The few box-drawing / bullet characters we emit (`●`, `X`, `│`) are
    all single-cell, so `len()` is correct. Kept as a hook in case we
    ever add wide characters.
    """
    return len(text)
