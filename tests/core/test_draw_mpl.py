"""matplotlib drawer for CircuitSpec.

Structural tests only: we assert on the patches/texts the drawer emits, not
on pixels. matplotlib is forced onto the headless Agg backend so these run
without a display.
"""

from __future__ import annotations

import pytest

from eigenkit import Circuit, CircuitBuilder, Encoding, Variational, all_z
from eigenkit.core.draw_mpl import draw_mpl, measured_qubits
from eigenkit.core.ir import Observable


pytest.importorskip("matplotlib")  # skips entire module when matplotlib is absent

import matplotlib.pyplot as plt  # noqa: E402


plt.switch_backend("Agg")  # headless: no display needed for structural tests


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _texts(ax) -> list[str]:
    return [t.get_text() for t in ax.texts]


class TestReturnValue:
    def test_returns_fig_and_ax(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.rx(0.5, qubit=0)
        fig, ax = draw_mpl(qc.build())
        assert isinstance(fig, plt.Figure)
        assert ax in fig.axes

    def test_draws_into_supplied_ax(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.h(qubit=0)
        fig, ax = plt.subplots()
        out_fig, out_ax = draw_mpl(qc.build(), ax=ax)
        assert out_ax is ax
        assert out_fig is fig


class TestWiresAndLabels:
    def test_one_wire_per_qubit(self):
        qc = CircuitBuilder(n_qubits=3)
        qc.h(qubit=0)
        _, ax = draw_mpl(qc.build())
        # One horizontal wire Line2D per qubit (plus any gate strokes).
        wire_ys = {
            line.get_ydata()[0]
            for line in ax.lines
            if line.get_ydata()[0] == line.get_ydata()[-1]
        }
        assert {0, 1, 2}.issubset(wire_ys)

    def test_qubit_index_labels(self):
        qc = CircuitBuilder(n_qubits=2)
        texts = _texts(draw_mpl(qc.build())[1])
        assert "0" in texts and "1" in texts

    def test_gaps_helper_splits_around_occluders(self):
        from eigenkit.core.draw_mpl import _gaps

        assert _gaps(0.0, 10.0, []) == [(0.0, 10.0)]
        assert _gaps(0.0, 10.0, [(3.0, 5.0)]) == [(0.0, 3.0), (5.0, 10.0)]
        assert _gaps(0.0, 10.0, [(2.0, 4.0), (6.0, 8.0)]) == [
            (0.0, 2.0),
            (4.0, 6.0),
            (8.0, 10.0),
        ]
        # intervals outside [lo, hi] clamp away and never produce a segment
        # past the bounds
        assert _gaps(0.0, 10.0, [(12.0, 15.0)]) == [(0.0, 10.0)]
        assert _gaps(0.0, 10.0, [(-3.0, -1.0)]) == [(0.0, 10.0)]
        # an interval covering the whole span leaves no wire
        assert _gaps(0.0, 10.0, [(-1.0, 11.0)]) == []


class TestSingleQubitBoxes:
    def test_rotation_shows_name_and_param(self):
        qc = CircuitBuilder(n_qubits=1)
        w = qc.add_weights(1)
        qc.rx(w[0], qubit=0)
        texts = _texts(draw_mpl(qc.build())[1])
        assert "RX" in texts
        assert "w_0" in texts

    def test_show_params_false_hides_angle(self):
        qc = CircuitBuilder(n_qubits=1)
        w = qc.add_weights(1)
        qc.rx(w[0], qubit=0)
        texts = _texts(draw_mpl(qc.build(), show_params=False)[1])
        assert "RX" in texts
        assert "w_0" not in texts

    def test_plain_gate_has_no_param_line(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.h(qubit=0)
        texts = _texts(draw_mpl(qc.build())[1])
        # "H" box label, plus the "0" wire label — nothing else.
        assert "H" in texts

    def test_phase_shift_renders(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.phase_shift(0.5, qubit=0)
        texts = _texts(draw_mpl(qc.build())[1])
        assert "PhaseShift" in texts


class TestMultiQubitGates:
    def test_cnot_has_control_and_target(self):
        from matplotlib.patches import Circle

        qc = CircuitBuilder(n_qubits=2)
        qc.cnot(0, 1)
        _, ax = draw_mpl(qc.build())
        circles = [p for p in ax.patches if isinstance(p, Circle)]
        # One filled control dot + one open target circle.
        assert len(circles) == 2

    def test_cry_shows_ry_box(self):
        qc = CircuitBuilder(n_qubits=2)
        w = qc.add_weights(1)
        qc.cry(w[0], control=0, target=1)
        texts = _texts(draw_mpl(qc.build())[1])
        assert "RY" in texts and "w_0" in texts

    def test_mcz_shows_z_box(self):
        qc = CircuitBuilder(n_qubits=3)
        qc.mcz(0, 1, 2)
        texts = _texts(draw_mpl(qc.build())[1])
        assert "Z" in texts

    def test_swap_renders(self):
        qc = CircuitBuilder(n_qubits=2)
        qc.swap(0, 1)
        # Should not raise; produces line strokes for the two crosses.
        fig, ax = draw_mpl(qc.build())
        assert len(ax.lines) >= 2


class TestMeasurements:
    def test_meter_drawn_on_measured_wires(self):
        from matplotlib.patches import Arc

        qc = CircuitBuilder(n_qubits=2)
        qc.h(qubit=0)
        spec = qc.build()
        _, ax = draw_mpl(spec, observables=[Observable.z(0), Observable.z(1)])
        arcs = [p for p in ax.patches if isinstance(p, Arc)]
        assert len(arcs) == 2  # one meter per measured wire

    def test_show_measurements_false(self):
        from matplotlib.patches import Arc

        qc = CircuitBuilder(n_qubits=1)
        qc.h(qubit=0)
        _, ax = draw_mpl(
            qc.build(), observables=[Observable.z(0)], show_measurements=False
        )
        assert not [p for p in ax.patches if isinstance(p, Arc)]

    def test_measured_qubits_union(self):
        obs = [Observable.from_string("Z0 X2"), Observable.z(3)]
        assert measured_qubits(obs) == {0, 2, 3}


def _edge_colors(ax) -> set:
    from matplotlib.patches import FancyBboxPatch

    return {
        tuple(round(c, 3) for c in p.get_edgecolor())
        for p in ax.patches
        if isinstance(p, FancyBboxPatch)
    }


def _rgba(color) -> tuple:
    from matplotlib.colors import to_rgba

    return tuple(round(c, 3) for c in to_rgba(color))


class TestStyles:
    def test_eigen_colours_gates_by_kind(self):
        from eigenkit.core.draw_mpl import _EIGEN_STYLE

        qc = CircuitBuilder(n_qubits=1)
        qc.h(0)
        qc.ry(0.3, qubit=0)
        _, ax = draw_mpl(qc.build(), observables=[Observable.z(0)], style="eigen")
        kinds = _EIGEN_STYLE["gate_colors"]
        edges = _edge_colors(ax)
        for kind in ("rotation", "block", "meter"):
            assert _rgba(kinds[kind]["edge"]) in edges, kind

    def test_rotations_include_two_qubit_rotations(self):
        from eigenkit.core.draw_mpl import _gate_kind

        qc = CircuitBuilder(n_qubits=2)
        qc.cry(0.3, 0, 1)
        qc.h(0)
        cry, h = qc.build().gates
        assert _gate_kind(cry) == "rotation"
        assert _gate_kind(h) == "block"

    def test_no_style_draws_gradients(self):
        from matplotlib.collections import LineCollection

        qc = CircuitBuilder(n_qubits=2)
        qc.h(0)
        qc.cnot(0, 1)
        for name in ("eigen", "eigen-dark", "eigen-universal"):
            _, ax = draw_mpl(qc.build(), style=name)
            assert not ax.images, name
            assert not [c for c in ax.collections if isinstance(c, LineCollection)]

    def test_controls_take_the_control_colour(self):
        from matplotlib.patches import Circle

        from eigenkit.core.draw_mpl import _STYLES

        qc = CircuitBuilder(n_qubits=2)
        x = qc.add_inputs(1)
        qc.cry(x[0], 0, 1)
        for name in ("eigen", "roles", "roles-dark"):
            _, ax = draw_mpl(qc.build(), style=name)
            dots = [p for p in ax.patches if isinstance(p, Circle)]
            assert dots
            want = _rgba(_STYLES[name]["control_color"])
            assert all(_rgba(d.get_edgecolor()) == want for d in dots), name

    def test_qubit_labels_are_not_white(self):
        qc = CircuitBuilder(n_qubits=2)
        _, ax = draw_mpl(qc.build(), style="eigen")
        label_colors = {t.get_color() for t in ax.texts if t.get_text() in ("0", "1")}
        assert label_colors and "white" not in label_colors

    def test_unknown_style_raises(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.h(0)
        with pytest.raises(ValueError, match="unknown style"):
            draw_mpl(qc.build(), style="neon")

    def test_circuit_method_forwards_style(self):
        from eigenkit.core.draw_mpl import _EIGEN_STYLE

        circuit = Circuit(
            CircuitBuilder(n_qubits=2).ry(0.3, qubit=0).cnot(0, 1).build(),
            observables=all_z(2),
            provider="numpy",
        )
        _, ax = circuit.draw_mpl(style="eigen")
        rotation = _EIGEN_STYLE["gate_colors"]["rotation"]["edge"]
        assert _rgba(rotation) in _edge_colors(ax)


class TestRolesStyle:
    def test_gate_category_classification(self):
        from eigenkit.core.draw_mpl import _gate_category

        qc = CircuitBuilder(n_qubits=1)
        x = qc.add_inputs(1)
        w = qc.add_weights(1)
        qc.ry(x[0], qubit=0)  # encoding
        qc.rz(w[0], qubit=0)  # trainable
        qc.rx(0.5, qubit=0)  # fixed (constant angle)
        qc.h(0)  # fixed (non-parametric)
        gates = qc.build().gates
        assert _gate_category(gates[0]) == "encoding"
        assert _gate_category(gates[1]) == "trainable"
        assert _gate_category(gates[2]) == "fixed"
        assert _gate_category(gates[3]) == "fixed"

    def test_trainable_scaled_encoding_reads_as_encoding(self):
        from eigenkit.core.draw_mpl import _gate_category
        from eigenkit.core.ir import Multiply

        qc = CircuitBuilder(n_qubits=1)
        x = qc.add_inputs(1)
        w = qc.add_weights(1)
        qc.ry(Multiply(w[0], x[0]), qubit=0)
        assert _gate_category(qc.build().gates[0]) == "encoding"

    def test_role_colors_applied_to_box_outlines(self):
        from eigenkit.core.draw_mpl import _ROLES_STYLE

        qc = CircuitBuilder(n_qubits=1)
        x = qc.add_inputs(1)
        w = qc.add_weights(1)
        qc.ry(x[0], qubit=0)  # encoding box
        qc.rz(w[0], qubit=0)  # trainable box
        _, ax = draw_mpl(qc.build(), style="roles")
        edge_colors = {tuple(round(c, 3) for c in p.get_edgecolor()) for p in ax.patches}
        cc = _ROLES_STYLE["category_colors"]
        from matplotlib.colors import to_rgba

        assert tuple(round(c, 3) for c in to_rgba(cc["encoding"])) in edge_colors
        assert tuple(round(c, 3) for c in to_rgba(cc["trainable"])) in edge_colors

    def test_legend_drawn_for_present_roles_only(self):
        qc = CircuitBuilder(n_qubits=1)
        x = qc.add_inputs(1)
        qc.ry(x[0], qubit=0)  # only encoding present
        qc.h(0)  # and fixed
        texts = _texts(draw_mpl(qc.build(), style="roles")[1])
        assert "Encoding" in texts
        assert "Fixed" in texts
        assert "Trainable" not in texts  # no trainable gate -> not in legend

    def test_legend_can_be_disabled(self):
        qc = CircuitBuilder(n_qubits=1)
        x = qc.add_inputs(1)
        qc.ry(x[0], qubit=0)
        texts = _texts(draw_mpl(qc.build(), style="roles", legend=False)[1])
        assert "Encoding" not in texts


class TestDarkStyles:
    def test_dark_styles_resolve_and_render(self):
        for name in ("default-dark", "eigen-dark", "roles-dark"):
            qc = CircuitBuilder(n_qubits=2)
            x = qc.add_inputs(1)
            qc.ry(x[0], qubit=0)
            qc.cnot(0, 1)
            fig, _ax = draw_mpl(qc.build(), style=name)
            assert fig is not None

    def test_dark_style_has_transparent_canvas(self):
        # Dark styles use a transparent canvas so they sit on any dark shade.
        qc = CircuitBuilder(n_qubits=1)
        qc.h(0)
        fig, _ax = draw_mpl(qc.build(), style="default-dark")
        assert fig.get_facecolor() == (0.0, 0.0, 0.0, 0.0)

    def test_light_style_keeps_default_figure_background(self):
        # No `background` key -> figure facecolor untouched (matplotlib white).
        qc = CircuitBuilder(n_qubits=1)
        qc.h(0)
        fig, _ax = draw_mpl(qc.build(), style="default")
        from matplotlib.colors import to_rgba

        assert fig.get_facecolor() == to_rgba("white")

    def test_dark_styles_have_no_light_interior_fill(self):
        # Box-like interiors (gate boxes, the meter box, the open ⊕ target
        # circle) are transparent or dark, so the light strokes and labels read
        # on a dark page. We isolate the target circle by its radius: control
        # dots and the meter pivot are Circles too, but those are intentionally
        # opaque foreground marks, not interiors.
        from matplotlib.patches import Circle, FancyBboxPatch

        from eigenkit.core.draw_mpl import _TARGET_RADIUS

        def luminance(rgba):
            r, g, b, _a = rgba
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        for name in ("default-dark", "eigen-dark", "roles-dark"):
            qc = CircuitBuilder(n_qubits=2)
            qc.h(0)
            qc.cnot(0, 1)
            _, ax = draw_mpl(qc.build(), observables=[Observable.z(0)], style=name)
            fills = [
                p.get_facecolor()
                for p in ax.patches
                if isinstance(p, FancyBboxPatch)
                or (
                    isinstance(p, Circle)
                    and round(p.get_radius(), 3) == round(_TARGET_RADIUS, 3)
                )
            ]
            assert fills, f"{name}: expected at least one interior patch"
            assert all(fc[3] == 0.0 or luminance(fc) < 0.15 for fc in fills), name


class TestUniversalStyle:
    def test_transparent_background_for_both_themes(self):
        # A single transparent image is what makes it work on light and dark.
        qc = CircuitBuilder(n_qubits=1)
        qc.h(0)
        fig, _ax = draw_mpl(qc.build(), style="eigen-universal")
        assert fig.get_facecolor() == (0.0, 0.0, 0.0, 0.0)

    def test_boxes_are_opaque_chips_with_dark_text(self):
        from matplotlib.patches import FancyBboxPatch

        from eigenkit.core.draw_mpl import _EIGEN_UNIVERSAL_STYLE

        qc = CircuitBuilder(n_qubits=1)
        qc.h(0)
        _, ax = draw_mpl(qc.build(), style="eigen-universal")
        boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
        assert boxes and all(p.get_facecolor()[3] == 1.0 for p in boxes)
        ink = _EIGEN_UNIVERSAL_STYLE["gate_colors"]["block"]["text"]
        assert ink in {t.get_color() for t in ax.texts if t.get_text() == "H"}


class TestEmpty:
    def test_zero_qubits_does_not_raise(self):
        from eigenkit.core.ir import CircuitSpec

        spec = CircuitSpec(n_qubits=0, n_weights=0, n_inputs=0, gates=())
        fig, ax = draw_mpl(spec)
        assert "empty" in " ".join(_texts(ax)).lower()


class TestCircuitMethod:
    def test_circuit_draw_mpl_draws_meters_from_observables(self):
        from matplotlib.patches import Arc

        circuit = Circuit(
            CircuitBuilder(n_qubits=2).ry(0.3, qubit=0).cnot(0, 1).build(),
            observables=all_z(2),
            provider="numpy",
        )
        fig, ax = circuit.draw_mpl()
        arcs = [p for p in ax.patches if isinstance(p, Arc)]
        assert len(arcs) == 2

    def test_blocks_circuit_renders(self):
        circuit = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1, rotation="RY")],
            observables=all_z(2),
            provider="numpy",
        )
        fig, ax = circuit.draw_mpl()
        texts = _texts(ax)
        assert any("RY" in t for t in texts)
