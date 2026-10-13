"""Matplotlib drawer for `CircuitSpec`, which does not depend on the provider.

It draws a circuit diagram: gates as boxes with labels, controls as filled dots,
the targets of controlled-NOT (`CNOT`) and multi-controlled `X` (`MCX`) gates as
`⊕`, `SWAP` gates as crossed wires and, optionally, measurement meters at the end
of the wires that the observables act on.

Like the text drawer in `draw.py`, the picture is built from the `CircuitSpec`
intermediate representation (IR) and the observables of the circuit, never from
the compiled backend object. The drawing is therefore the same whichever provider
ran the circuit. It uses `draw._layout` to place the gates in columns, so the
matplotlib and the text drawings stay consistent.

matplotlib is an optional dependency, and it is imported only when `draw_mpl` is
called. A core installation does not install it. Install it with
`eigenkit[viz]`.

The styles come in light and dark pairs, selected by name:

- `"default"`: black on white.
- `"eigen"`: the brand palette. Each gate is coloured by its kind: violet for
  rotations, cyan for `H` and the block gates, and blue for the meters.
- `"roles"`: the outline of each gate is coloured by its role (encoding,
  trainable or fixed), with a legend.

Each of them has a `"-dark"` variant (`"default-dark"`, `"eigen-dark"`,
`"roles-dark"`) for dark backgrounds. These variants have a transparent canvas and
light strokes, so they work on any dark shade. There is also `"eigen-universal"`:
a single transparent image of filled gate chips, which can be read on both light
and dark pages.

The styles are plain dictionaries in `_STYLES`, where a new theme can be added.
The wires are drawn only in the gaps between gates, so no wire shows through the
label of a gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Sequence

from .draw import _layout, _param_repr
from .ir import (
    SINGLE_QUBIT_ROTATIONS,
    TWO_QUBIT_ROTATIONS,
    CircuitSpec,
    Gate,
    Observable,
    collect_input_dependencies,
    collect_weight_dependencies,
)


if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_EXTRAS_HINT = "Install with: pip install eigenkit[viz]"

# Single-qubit gates that render as a plain (non-parametric) labeled box.
_PLAIN_SINGLE_QUBIT: frozenset[str] = frozenset({"H", "X", "Z", "S", "Sdg"})
# Everything drawn as a box sitting on one wire.
_SINGLE_QUBIT_BOX: frozenset[str] = SINGLE_QUBIT_ROTATIONS | _PLAIN_SINGLE_QUBIT

# Data-carrying gates act on all their qubits at once, so they get a box on
# each rather than a marker on one. `ControlledQubitUnitary` is handled apart
# because its leading qubits are controls.
_MULTI_QUBIT_BOX: frozenset[str] = frozenset({"AmplitudeEncoding", "QubitUnitary"})

# Styles are plain dicts so new themes can be added without touching the
# drawing code. Every drawing primitive reads its colours/weights from the
# resolved style dict. Keys:
#   background      figure facecolor for a new figure ("none" => transparent);
#                   omitted => matplotlib's default (white). Only applied when
#                   draw_mpl creates the figure (not for a caller-supplied ax).
#   box_facecolor / box_edgecolor / text_color
#                   gate boxes, unless `gate_colors` or `categorical` says
#                   otherwise; `box_facecolor` also fills the open target circle.
#   control_color   control dots, CNOT/MCX targets, SWAP crosses and the
#                   vertical links.
#   meter_color     the meter dial, needle and pivot.
#   gate_colors     optional {"rotation", "block", "meter"} -> {"edge", "face",
#                   "text"}: colour each box by the kind of gate (see
#                   `_gate_kind`); the meter's "text" is its dial.
#   categorical     when True, each box's outline is coloured by the gate's role
#                   via `category_colors` (see `_gate_category`) and a legend is
#                   drawn.
#   category_colors {"encoding", "trainable", "fixed"} -> colour.
# All other keys are self-describing.

# The default black-on-white look: light, airy, normal-weight sans text.
_DEFAULT_STYLE: dict = {
    "wire_color": "#1a1a1a",
    "wire_width": 1.1,
    "box_facecolor": "white",
    "box_edgecolor": "#1a1a1a",
    "box_linewidth": 1.3,
    "cross_width": 1.1,  # the CNOT/MCX target "+" and SWAP "x" strokes
    "text_color": "#1a1a1a",  # gate labels (inside boxes)
    "label_color": "#1a1a1a",  # qubit-index labels (on the white canvas)
    "control_color": "#1a1a1a",
    "meter_color": "#1a1a1a",
    "font_family": "sans-serif",
    "font_weight": "normal",
}

# Brand palette. A gate's colour says what kind of gate it is: violet for
# rotations, cyan for `H` and the block gates, blue for the meters.
_INK = "#0D1016"  # gate text, meter dial, wires and qubit labels on a light page

# Light page: dark outlines over solid pastel fills, black text.
_EIGEN_STYLE: dict = {
    **_DEFAULT_STYLE,
    "wire_color": _INK,
    "wire_width": 1.3,
    "box_edgecolor": _INK,
    "box_linewidth": 1.5,
    "cross_width": 1.5,
    "text_color": _INK,
    "label_color": _INK,
    "control_color": "#2C90B0",
    "meter_color": _INK,
    "gate_colors": {
        "rotation": {"edge": "#6644A4", "face": "#D3AAFF", "text": _INK},
        "block": {"edge": "#286477", "face": "#7DE0FF", "text": _INK},
        "meter": {"edge": "#285CA5", "face": "#AEC2FF", "text": _INK},
    },
}

# Dark page (designed on #0D1016, transparent canvas): dark fills, a muted
# outline and a bright label in each gate's colour.
_EIGEN_DARK_STYLE: dict = {
    **_EIGEN_STYLE,
    "background": "none",
    "box_facecolor": "none",  # the open target circle
    "box_edgecolor": "#94A3B8",
    "wire_color": "#94A3B8",
    "text_color": "#B489FF",
    "label_color": "#C2CAD6",
    "control_color": "#4FC8D3",
    "meter_color": "#1880E3",
    "gate_colors": {
        "rotation": {"edge": "#804BA2", "face": "#20192C", "text": "#B489FF"},
        "block": {"edge": "#2D81A2", "face": "#14222E", "text": "#4FC8D3"},
        "meter": {"edge": "#115FA9", "face": "#0F1B2C", "text": "#1880E3"},
    },
}

# One image for light and dark pages: opaque chips with black text, which read
# on any background, and mid-tone wires and links that keep their contrast on
# both white and near-black.
_EIGEN_UNIVERSAL_STYLE: dict = {
    **_EIGEN_STYLE,
    "background": "none",
    "box_facecolor": "none",  # the open target circle
    "box_edgecolor": "#7D8798",
    "wire_color": "#7D8798",
    "label_color": "#7D8798",
    "control_color": "#2C90B0",
    "gate_colors": {
        "rotation": {"edge": "#C08EFF", "face": "#C08EFF", "text": _INK},
        "block": {"edge": "#5CCFF5", "face": "#5CCFF5", "text": _INK},
        "meter": {"edge": "#8FA9FF", "face": "#8FA9FF", "text": _INK},
    },
}

# Role style: white boxes whose outline colour encodes the gate's role — data
# encoding, trainable, or fixed/structural — to read a PQC at a glance. Fixed
# gates are muted so the data and the trainable gates stand out; controls and
# targets keep a colour of their own.
_ROLES_STYLE: dict = {
    **_DEFAULT_STYLE,
    "wire_width": 1.3,
    "box_edgecolor": "#94A3B8",  # the meter, like the fixed gates
    "box_linewidth": 1.8,  # a touch bolder so the role colour reads clearly
    "cross_width": 1.8,
    "control_color": "#285CA5",
    "categorical": True,
    "category_colors": {
        "encoding": "#6644A4",  # gates that load input data
        "trainable": "#1F8FAD",  # gates with trainable weights
        "fixed": "#94A3B8",  # constant / structural gates
    },
}

# --- dark-background variants ---
# The canvas and the box interiors are *transparent* (`background` /
# `box_facecolor` = "none") so the styles sit on any dark shade — the wires are
# broken under the boxes, so nothing shows through a gate's label.
_DARK_INK = "#eef1f6"  # near-white gate text
_DARK_MUTED = "#c8cfdb"  # light slate: neutral outlines/labels/links

_DEFAULT_DARK_STYLE: dict = {
    **_DEFAULT_STYLE,
    "background": "none",
    "box_facecolor": "none",
    "box_edgecolor": _DARK_MUTED,
    "text_color": _DARK_INK,
    "label_color": _DARK_MUTED,
    "control_color": _DARK_MUTED,
    "meter_color": _DARK_MUTED,
    "wire_color": "#8a94a6",
}

_ROLES_DARK_STYLE: dict = {
    **_ROLES_STYLE,
    "background": "none",
    "box_facecolor": "none",
    "box_edgecolor": "#7D8798",  # the meter, like the fixed gates
    "wire_color": "#94A3B8",
    "text_color": "#C2CAD6",
    "label_color": "#94A3B8",
    "control_color": "#1880E3",
    "meter_color": "#C2CAD6",
    "category_colors": {
        "encoding": "#B489FF",
        "trainable": "#4FC8D3",
        "fixed": "#7D8798",
    },
}

_STYLES: dict[str, dict] = {
    "default": _DEFAULT_STYLE,
    "eigen": _EIGEN_STYLE,
    "roles": _ROLES_STYLE,
    "default-dark": _DEFAULT_DARK_STYLE,
    "eigen-dark": _EIGEN_DARK_STYLE,
    "roles-dark": _ROLES_DARK_STYLE,
    "eigen-universal": _EIGEN_UNIVERSAL_STYLE,
}


def _resolve_style(style) -> dict:
    """Return the style dict for a style name (or pass a dict through)."""
    if isinstance(style, dict):
        return style
    try:
        return _STYLES[style]
    except KeyError:
        raise ValueError(
            f"unknown style {style!r}; choose from {sorted(_STYLES)}"
        ) from None


# Geometry, in data units (one wire = one unit tall). With
# `set_aspect("equal")` and the figure sized at INCHES_PER_UNIT, point-based
# font sizes map predictably onto the data grid (see _char_width).
_INCHES_PER_UNIT = 0.72
_BOX_HEIGHT = 0.88
_MIN_BOX_W = 0.86
_BOX_PAD = 0.42  # horizontal breathing room around a gate label
_BOX_PAD_PARAM = 0.34  # ...and around the (smaller) angle label
_PARAM_FONT_RATIO = 0.64  # angle label size relative to the gate name
_CONTROL_RADIUS = 0.14
_TARGET_RADIUS = 0.21
_SWAP_HALF = 0.16
_COLUMN_GAP = 0.6  # clear space between adjacent columns
_METER_W = 0.88
_METER_H = 0.84
_DEFAULT_FONTSIZE = 13.0
_LEGEND_SWATCH = 0.42  # swatch size for the role legend
_LEGEND_GAP = 0.55  # space after each legend entry
_LEGEND_ROW = 1.0  # vertical space reserved below the circuit for the legend


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            f"matplotlib is required for draw_mpl(). {_EXTRAS_HINT}"
        ) from exc
    return plt


def _char_width(fontsize: float) -> float:
    """Approximate width of one character, in data units.

    A point is 1/72 inch; one data unit is `_INCHES_PER_UNIT` inches, so a
    character roughly `0.6 * fontsize` points wide spans this many units.
    Used only to size boxes around their labels — exact metrics are not
    needed, just enough headroom that text does not spill out.
    """
    return fontsize * 0.6 / (72.0 * _INCHES_PER_UNIT)


def _box_width(label: str, param: Optional[str], fontsize: float) -> float:
    """Width (data units) of a gate box holding `label` and optional `param`.

    Shared by the column-layout pass and the box renderer so a box is always
    placed in a slot exactly as wide as the box drawn into it.
    """
    w = max(len(label) * _char_width(fontsize) + _BOX_PAD, _MIN_BOX_W)
    if param:
        w = max(
            w, len(param) * _char_width(fontsize * _PARAM_FONT_RATIO) + _BOX_PAD_PARAM
        )
    return w


def measured_qubits(observables: Iterable[Observable]) -> set[int]:
    """Return every qubit that any of `observables` acts on (their Pauli support).

    Args:
        observables (Sequence[Observable]): The observables to measure.

    Returns:
        The index of every qubit that the observables act on.
    """
    qubits: set[int] = set()
    for obs in observables:
        for term in obs.terms:
            for _op, qubit in term.paulis:
                qubits.add(qubit)
    return qubits


def _box_label(
    gate: Gate, *, decimals: int, show_params: bool
) -> tuple[str, Optional[str]]:
    """Return `(name, param_text_or_None)` for a gate rendered as a box.

    `CRY` shows its rotation (`RY`) in the box with the control drawn
    separately; `MCZ` shows `Z` on its target. Everything else keeps its
    gate name.
    """
    name = gate.name
    if name == "CRY":
        name = "RY"
    elif name == "MCZ":
        name = "Z"
    if show_params and gate.params:
        return name, _param_repr(gate.params[0], decimals)
    return name, None


def _new_figure(plt, **kwargs):
    """A figure that belongs to the caller, not to pyplot's global manager.

    `plt.subplots` registers the figure with pyplot, so every diagram would
    stay open until explicitly closed -- drawing in a loop would hit
    matplotlib's "more than 20 figures" warning and hold the memory -- and it
    would also become the active figure, so the caller's next `plt.plot`
    would land on the circuit diagram. Constructing `Figure` directly avoids
    both; the returned object still saves and displays the same way.

    Args:
        plt: The pyplot module, already imported by the caller.
        **kwargs: Passed to `Figure`.

    Returns:
        The `(figure, axes)` pair.
    """
    from matplotlib.figure import Figure

    figure = Figure(**kwargs)
    return figure, figure.subplots()


def _controlled_unitary_target_count(gate: Gate) -> int:
    """How many of a ControlledQubitUnitary's qubits its matrix acts on."""
    try:
        size = len(gate.data[0])
    except (IndexError, TypeError):
        return 1
    count = max(size.bit_length() - 1, 1)
    return min(count, max(len(gate.qubits) - 1, 1))


def _box_qubit(gate: Gate) -> Optional[int]:
    """The qubit a gate's labelled box sits on, or None if it draws no box.

    Single-qubit boxes sit on their one qubit; `CRY`/`MCZ` draw their box on
    the target (the last qubit). Everything else (controls, targets, swaps)
    draws no box.
    """
    if gate.name in _SINGLE_QUBIT_BOX:
        return gate.qubits[0]
    if gate.name in ("CRY", "MCZ", "CPHASE"):
        return gate.qubits[-1]
    if gate.name in _MULTI_QUBIT_BOX or gate.name == "ControlledQubitUnitary":
        return gate.qubits[-1]
    return None


def draw_mpl(
    spec: CircuitSpec,
    *,
    observables: Optional[Sequence[Observable]] = None,
    decimals: int = 3,
    show_params: bool = True,
    show_measurements: bool = True,
    fontsize: float = _DEFAULT_FONTSIZE,
    style: str = "default",
    legend: bool = True,
    ax: "Optional[Axes]" = None,
) -> "tuple[Figure, Axes]":
    """Draw `spec` as a matplotlib circuit diagram.

    Args:
        spec: The circuit to draw.
        observables: The observables whose Pauli support sets the wires that end
            with a measurement meter. Pass `None`, or `show_measurements=False`,
            to draw no meters. Default: `None`.
        decimals: The number of significant digits of constant gate angles.
            Default: `3`.
        show_params: If `True`, the angle of each parametrised gate is written in
            its box (for example `RX` above `w_0`). Default: `True`.
        show_measurements: If `True` and `observables` is given, a meter is drawn
            on each measured wire. Default: `True`.
        fontsize: The font size of the gate names. The angle labels are smaller.
            Default: `13.0`.
        style: The name of a style: `"default"` (black on white), `"eigen"` (the
            brand palette, where gates are coloured by kind: rotations, `H` and
            block gates, and meters), `"roles"` (gate outlines coloured by role:
            encoding, trainable or fixed) or `"eigen-universal"` (one image for
            light and dark pages). Each of them except the last has a `"-dark"`
            variant (for example `"eigen-dark"`) for dark backgrounds. A style
            dictionary can also be passed. Default: `"default"`.
        legend: If `True`, the `"roles"` styles draw a legend that maps each
            colour to its role. Styles that do not colour gates by role ignore it.
            Default: `True`.
        ax: An existing matplotlib `Axes` to draw into. If it is not given, a new
            figure and axes are created, with a size that fits the circuit. The
            `background` of a style is applied only to a figure that `draw_mpl`
            creates. When you pass your own `ax`, set the face colour of its
            figure yourself: the `"-dark"` styles draw in light colours and need a
            dark background to be readable. Default: `None`.

    Returns:
        The `(figure, axes)` pair. Save the figure with `figure.savefig(...)`. A
            figure that `draw_mpl` creates is not registered with pyplot, so
            `plt.show()` does not display it.
    """
    plt = _load_matplotlib()
    style = _resolve_style(style)
    n = spec.n_qubits

    if n == 0:
        fig, axes = (
            (ax.figure, ax) if ax is not None else _new_figure(plt, figsize=(3, 1))
        )
        axes.axis("off")
        axes.text(0.5, 0.5, "(empty circuit: 0 qubits)", ha="center", va="center")
        return fig, axes

    placements, n_cols = _layout(spec)

    meter_qubits: set[int] = set()
    if show_measurements and observables:
        meter_qubits = measured_qubits(observables)

    # --- per-column geometry ---
    # Classify the box-drawn gates once: their column, the qubit the box sits
    # on, and the box width. The same widths size each column's slot below and
    # mask the wires further down, so the box, its slot, and its wire gap can
    # never disagree.
    boxes: list[tuple[int, int, float]] = []
    for col, gate in placements:
        bq = _box_qubit(gate)
        if bq is None:
            continue
        name, param = _box_label(gate, decimals=decimals, show_params=show_params)
        boxes.append((col, bq, _box_width(name, param, fontsize)))

    # Each column is only as wide as its widest box, so a single wide gate
    # (e.g. PhaseShift) does not stretch the whole diagram. Columns are then
    # laid out left to right, separated by _COLUMN_GAP.
    col_width = [_MIN_BOX_W] * n_cols
    for col, _bq, bw in boxes:
        col_width[col] = max(col_width[col], bw)

    centers: list[float] = []
    cursor = 0.0
    for c in range(n_cols):
        cursor += col_width[c] / 2
        centers.append(cursor)
        cursor += col_width[c] / 2 + _COLUMN_GAP

    def col_x(col: int) -> float:
        return centers[col]

    # The meters sit in their own slot to the right of the last gate column.
    if meter_qubits:
        meter_x = cursor + _METER_W / 2
        right_edge = meter_x + _METER_W / 2
    elif n_cols:
        meter_x = 0.0
        right_edge = centers[-1] + col_width[-1] / 2
    else:
        meter_x = 0.0
        right_edge = 0.4

    # --- wire occluders: x-spans a box/meter masks, per qubit ---
    # Wires are drawn only in the gaps, so a box with a transparent interior
    # never has the wire showing through its label.
    occluders: dict[int, list] = {q: [] for q in range(n)}
    for col, bq, bw in boxes:
        occluders[bq].append((col_x(col) - bw / 2, col_x(col) + bw / 2))
    for q in meter_qubits:
        occluders[q].append((meter_x - _METER_W / 2, meter_x + _METER_W / 2))

    # --- role legend (categorical styles only) ---
    used_categories: list[str] = []
    if legend and style.get("categorical"):
        used_categories = _used_categories(placements)
    legend_space = _LEGEND_ROW if used_categories else 0.0
    legend_width = _legend_width(used_categories, fontsize) if used_categories else 0.0

    # --- canvas extents ---
    x_left = -0.95  # room for the left-hand qubit labels
    x_right = max(right_edge, legend_width) + 0.2
    wire_x0 = -0.45
    wire_x1 = (right_edge if used_categories else x_right) - 0.05

    if ax is None:
        fig_w = (x_right - x_left) * _INCHES_PER_UNIT
        fig_h = (n + 0.2 + legend_space) * _INCHES_PER_UNIT
        # A higher-than-default dpi keeps text and curves crisp on screen and
        # when saved; callers can still override via figure.savefig(dpi=...).
        fig, ax = _new_figure(plt, figsize=(fig_w, fig_h), dpi=250)
        # Dark styles set a canvas colour. `axis("off")` skips the axes patch,
        # so the figure facecolor is what shows (and what savefig uses unless
        # the caller passes transparent=True).
        if style.get("background"):
            fig.set_facecolor(style["background"])
    else:
        fig = ax.figure

    ax.axis("off")

    # --- wires + left-hand qubit labels ---
    # Each wire is drawn in the gaps left by the boxes/meters on that qubit, so
    # a box never shows the wire crossing its label.
    for q in range(n):
        for gx0, gx1 in _gaps(wire_x0, wire_x1, occluders[q]):
            ax.plot(
                [gx0, gx1],
                [q, q],
                color=style["wire_color"],
                linewidth=style["wire_width"],
                zorder=1,
            )
        _text(
            ax,
            -0.6,
            q,
            str(q),
            style,
            fontsize=fontsize,
            ha="right",
            color=style["label_color"],
        )

    # --- gates ---
    for col, gate in placements:
        x = col_x(col)
        _draw_gate(
            ax,
            gate,
            x,
            style,
            decimals=decimals,
            show_params=show_params,
            fontsize=fontsize,
        )

    # --- terminal measurement meters ---
    for q in sorted(meter_qubits):
        _add_meter(ax, meter_x, q, style)

    # --- role legend, below the circuit ---
    if used_categories:
        _draw_category_legend(
            ax, used_categories, style, y=n - 0.4 + 0.55, fontsize=fontsize
        )

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(n - 0.4 + legend_space, -0.6)  # inverted: qubit 0 on top
    ax.set_aspect("equal")

    return fig, ax


# ---- per-gate drawing ----


def _draw_gate(
    ax: "Axes",
    gate: Gate,
    x: float,
    style: dict,
    *,
    decimals: int,
    show_params: bool,
    fontsize: float,
) -> None:
    name = gate.name
    lo, hi = min(gate.qubits), max(gate.qubits)
    accent = _box_colors(gate, style)

    if name == "CNOT":
        ctrl, tgt = gate.qubits
        _add_vertical(ax, x, ctrl, tgt, style)
        _add_control(ax, x, ctrl, style)
        _add_target(ax, x, tgt, style)
        return

    if name == "MCX":
        *ctrls, tgt = gate.qubits
        _add_vertical(ax, x, lo, hi, style)
        for c in ctrls:
            _add_control(ax, x, c, style)
        _add_target(ax, x, tgt, style)
        return

    if name == "MCZ":
        *ctrls, tgt = gate.qubits
        _add_vertical(ax, x, lo, hi, style)
        for c in ctrls:
            _add_control(ax, x, c, style)
        label, param = _box_label(gate, decimals=decimals, show_params=show_params)
        _add_box(ax, x, tgt, label, param, style, fontsize=fontsize, accent=accent)
        return

    if name == "CRY":
        ctrl, tgt = gate.qubits
        _add_vertical(ax, x, ctrl, tgt, style)
        _add_control(ax, x, ctrl, style)
        label, param = _box_label(gate, decimals=decimals, show_params=show_params)
        _add_box(ax, x, tgt, label, param, style, fontsize=fontsize, accent=accent)
        return

    if name == "SWAP":
        a, b = gate.qubits
        _add_vertical(ax, x, a, b, style)
        _add_swap(ax, x, a, style)
        _add_swap(ax, x, b, style)
        return

    if name == "CPHASE":
        ctrl, tgt = gate.qubits
        _add_vertical(ax, x, ctrl, tgt, style)
        _add_control(ax, x, ctrl, style)
        label, param = _box_label(gate, decimals=decimals, show_params=show_params)
        _add_box(ax, x, tgt, label, param, style, fontsize=fontsize, accent=accent)
        return

    if name == "ControlledQubitUnitary":
        n_target = _controlled_unitary_target_count(gate)
        ctrls, targets = gate.qubits[:-n_target], gate.qubits[-n_target:]
        values = gate.data[1] if len(gate.data) > 1 and gate.data[1] else None
        _add_vertical(ax, x, lo, hi, style)
        for i, c in enumerate(ctrls):
            open_control = values is not None and i < len(values) and not values[i]
            _add_control(ax, x, c, style, filled=not open_control)
        label, param = _box_label(gate, decimals=decimals, show_params=show_params)
        for q in targets:
            _add_box(ax, x, q, label, param, style, fontsize=fontsize, accent=accent)
        return

    if name in _MULTI_QUBIT_BOX:
        # A data-carrying gate acts on all of its qubits at once; draw a box on
        # each and link them, rather than raising, so that QFT, Grover, HHL and
        # UCCSD circuits can be drawn.
        label, param = _box_label(gate, decimals=decimals, show_params=show_params)
        if lo != hi:
            _add_vertical(ax, x, lo, hi, style)
        for q in gate.qubits:
            _add_box(ax, x, q, label, param, style, fontsize=fontsize, accent=accent)
        return

    if name in _SINGLE_QUBIT_BOX:
        label, param = _box_label(gate, decimals=decimals, show_params=show_params)
        _add_box(
            ax, x, gate.qubits[0], label, param, style, fontsize=fontsize, accent=accent
        )
        return

    raise NotImplementedError(
        f"draw_mpl: gate {name!r} has no drawing rule. Every name in "
        f"`ir.ALL_GATES` should be covered; this is a gap in the drawer, not "
        f"in your circuit."
    )


# ---- gate kinds (the eigen styles) ----

# Parametric rotations; everything else drawn as a box is a block gate.
_ROTATION_GATES: frozenset[str] = SINGLE_QUBIT_ROTATIONS | TWO_QUBIT_ROTATIONS


def _gate_kind(gate: Gate) -> str:
    """Classify a boxed gate as 'rotation' or 'block' for `gate_colors`."""
    return "rotation" if gate.name in _ROTATION_GATES else "block"


def _box_colors(gate: Gate, style: dict) -> tuple:
    """`(edge, face, text)` for the boxes of `gate` under `style`."""
    if style.get("categorical"):
        edge = style["category_colors"][_gate_category(gate)]
        return edge, style["box_facecolor"], style["text_color"]
    if "gate_colors" in style:
        colors = style["gate_colors"][_gate_kind(gate)]
        return colors["edge"], colors["face"], colors["text"]
    return style["box_edgecolor"], style["box_facecolor"], style["text_color"]


# ---- gate roles (categorical styling) ----

_CATEGORY_ORDER = ("encoding", "trainable", "fixed")
_CATEGORY_LABELS = {
    "encoding": "Encoding",
    "trainable": "Trainable",
    "fixed": "Fixed",
}


def _gate_category(gate: Gate) -> str:
    """Classify a gate as 'encoding', 'trainable', or 'fixed'.

    A gate that loads input data (its angle depends on an `InputRef`) is
    'encoding'; one that depends only on trainable weights is 'trainable';
    everything else (non-parametric or constant-angle) is 'fixed'. Input takes
    priority, so a trainable-scaled encoding gate (`weight * input`) reads as
    encoding.
    """
    has_input = any(collect_input_dependencies(p) for p in gate.params)
    if has_input:
        return "encoding"
    has_weight = any(collect_weight_dependencies(p) for p in gate.params)
    if has_weight:
        return "trainable"
    return "fixed"


def _used_categories(placements: list) -> list[str]:
    """Roles present in the circuit, in canonical legend order."""
    present = {_gate_category(gate) for _col, gate in placements}
    return [cat for cat in _CATEGORY_ORDER if cat in present]


def _legend_width(used: list, fontsize: float) -> float:
    """Total data-unit width of the role legend (for canvas sizing)."""
    char_w = _char_width(fontsize * 0.9)
    return sum(
        _LEGEND_SWATCH + 0.16 + len(_CATEGORY_LABELS[cat]) * char_w + _LEGEND_GAP
        for cat in used
    )


def _draw_category_legend(
    ax: "Axes", used: list, style: dict, *, y: float, fontsize: float
) -> None:
    """Draw a left-aligned row of swatch + role-name entries below the circuit."""
    from matplotlib.patches import FancyBboxPatch

    fs = fontsize * 0.9
    char_w = _char_width(fs)
    sw = _LEGEND_SWATCH
    cx = 0.0
    for cat in used:
        color = style["category_colors"][cat]
        ax.add_patch(
            FancyBboxPatch(
                (cx, y - sw / 2),
                sw,
                sw,
                boxstyle=_BOXSTYLE,
                facecolor=style["box_facecolor"],
                edgecolor=color,
                linewidth=style["box_linewidth"],
                mutation_aspect=1.0,
                zorder=3,
            )
        )
        label = _CATEGORY_LABELS[cat]
        _text(ax, cx + sw + 0.16, y, label, style, fontsize=fs, ha="left")
        cx += sw + 0.16 + len(label) * char_w + _LEGEND_GAP


# ---- primitive shapes ----


def _text(
    ax: "Axes",
    x: float,
    y: float,
    s: str,
    style: dict,
    *,
    fontsize: float,
    ha: str = "center",
    color: Optional[str] = None,
    zorder: int = 5,
) -> None:
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color if color is not None else style["text_color"],
        family=style["font_family"],
        fontweight=style["font_weight"],
        zorder=zorder,
    )


_CORNER_RADIUS = 0.13  # rounded-box corner radius (data units)
_BOXSTYLE = f"round,pad=0,rounding_size={_CORNER_RADIUS}"


def _gaps(lo: float, hi: float, intervals: list) -> list:
    """Sub-intervals of [lo, hi] not covered by `intervals` (e.g. wire pieces
    left after removing the spans masked by gate boxes)."""
    segs = []
    cursor = lo
    for a, b in sorted(intervals):
        a, b = max(a, lo), min(b, hi)
        if a >= b:  # interval clamped away (lies outside [lo, hi]) — skip it
            continue
        if a > cursor:
            segs.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < hi:
        segs.append((cursor, hi))
    return segs


def _rounded_box(
    ax: "Axes", x: float, y: float, w: float, h: float, style: dict, colors: tuple
) -> None:
    """Draw a rounded gate/meter box in `colors` = `(edge, face, text)`."""
    from matplotlib.patches import FancyBboxPatch

    edge, face, _text_color = colors
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=_BOXSTYLE,
            facecolor=face,
            edgecolor=edge,
            linewidth=style["box_linewidth"],
            mutation_aspect=1.0,
            zorder=3,
        )
    )


def _add_box(
    ax: "Axes",
    x: float,
    y: float,
    label: str,
    param: Optional[str],
    style: dict,
    *,
    fontsize: float,
    accent: tuple,
) -> None:
    w = _box_width(label, param, fontsize)
    _rounded_box(ax, x, y, w, _BOX_HEIGHT, style, accent)
    color = accent[2]

    if param:
        # y grows downward (axis is inverted), so the name sits above the
        # angle value.
        _text(ax, x, y - 0.18, label, style, fontsize=fontsize, color=color)
        _text(
            ax,
            x,
            y + 0.20,
            param,
            style,
            fontsize=fontsize * _PARAM_FONT_RATIO,
            color=color,
        )
    else:
        _text(ax, x, y, label, style, fontsize=fontsize, color=color)


def _add_control(
    ax: "Axes", x: float, y: float, style: dict, *, filled: bool = True
) -> None:
    """Draw a control dot. `filled=False` draws the hollow marker that means
    the control activates on |0>, which `control_values` can ask for."""
    from matplotlib.patches import Circle

    c = style["control_color"]
    face = c if filled else style.get("box_facecolor", "none")
    ax.add_patch(Circle((x, y), _CONTROL_RADIUS, facecolor=face, edgecolor=c, zorder=4))


def _add_target(ax: "Axes", x: float, y: float, style: dict) -> None:
    """The CNOT/MCX target: an open circle with a cross through it (``⊕``)."""
    from matplotlib.patches import Circle

    r = _TARGET_RADIUS
    lw = style["cross_width"]
    c = style["control_color"]
    ax.add_patch(
        Circle(
            (x, y),
            r,
            facecolor=style["box_facecolor"],
            edgecolor=c,
            linewidth=lw,
            zorder=4,
        )
    )
    ax.plot([x - r, x + r], [y, y], color=c, linewidth=lw, zorder=5)
    ax.plot([x, x], [y - r, y + r], color=c, linewidth=lw, zorder=5)


def _add_swap(ax: "Axes", x: float, y: float, style: dict) -> None:
    d = _SWAP_HALF
    lw = style["cross_width"]
    c = style["control_color"]
    ax.plot([x - d, x + d], [y - d, y + d], color=c, linewidth=lw, zorder=4)
    ax.plot([x - d, x + d], [y + d, y - d], color=c, linewidth=lw, zorder=4)


def _add_vertical(ax: "Axes", x: float, y0: float, y1: float, style: dict) -> None:
    c = style["control_color"]
    ax.plot([x, x], [y0, y1], color=c, linewidth=style["wire_width"], zorder=2)


def _add_meter(ax: "Axes", x: float, y: float, style: dict) -> None:
    """A measurement meter: a rounded box with a gauge dial and a needle.

    Drawn for a downward-growing (inverted) y-axis, so the dial dome opens
    *upward* on screen (arc angles 180..360) and the needle points to the
    upper-right.
    """
    from matplotlib.patches import Arc, Circle, FancyArrowPatch

    w, h = _METER_W, _METER_H
    lw = style["cross_width"]
    if "gate_colors" in style:
        meter = style["gate_colors"]["meter"]
        colors = (meter["edge"], meter["face"], meter["text"])
    else:
        colors = (style["box_edgecolor"], style["box_facecolor"], style["meter_color"])
    color = colors[2]

    _rounded_box(ax, x, y, w, h, style, colors)

    pivot = (x, y + h * 0.18)  # dial centre, below the box centre on screen
    ax.add_patch(
        Arc(
            pivot,
            w * 0.62,
            h * 0.62,
            angle=0.0,
            theta1=180.0,
            theta2=360.0,
            edgecolor=color,
            linewidth=lw,
            zorder=5,
        )
    )
    tip = (x + w * 0.24, y - h * 0.20)
    ax.add_patch(
        FancyArrowPatch(
            pivot,
            tip,
            arrowstyle="-|>",
            mutation_scale=10.0,
            linewidth=lw,
            color=color,
            zorder=6,
        )
    )
    ax.add_patch(
        Circle(
            pivot,
            0.024,
            facecolor=color,
            edgecolor=color,
            zorder=6,
        )
    )
