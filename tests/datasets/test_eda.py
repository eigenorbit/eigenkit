"""Tests for dataset EDA report API."""

from __future__ import annotations

import numpy as np
import pytest


matplotlib = pytest.importorskip("matplotlib")

from eigenkit.datasets import (  # noqa: E402
    FONTSIZE_AXES,
    FONTSIZE_LEGEND,
    FONTSIZE_TICKS,
    STYLE_COLORS,
    DatasetReport,
    load_dataset,
    rc_params,
)


matplotlib.use("Agg")


def test_dataset_report_from_builtin_dataset():
    data = load_dataset("iris", split=False, random_state=42)
    report = DatasetReport.from_data(data)

    figs = report.plot_overview(max_features=3)
    assert {
        "data_health",
        "target",
        "distributions",
        "correlation",
        "feature_target",
    } <= set(figs.keys())

    summary = report.summarize()
    assert summary["task"] == "classification"
    assert summary["n_samples"] == 150
    assert "text" in summary


def test_dataset_report_from_custom_arrays():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)).astype(np.float32)
    y = (X[:, 0] + 0.2 * X[:, 1] > 0).astype(int)

    report = DatasetReport.from_data(X=X, y=y, feature_names=[f"f{i}" for i in range(5)])
    summary = report.summarize()

    assert summary["task"] == "classification"
    assert "class_imbalance_ratio" in summary
    fig = report.plot(kind="correlation")
    assert fig is not None


def test_timeseries_report_has_timeseries_outputs():
    data = load_dataset("sine_wave", split=False, random_state=42)
    report = DatasetReport.from_data(data, task="timeseries", time_index=data["X"])

    figs = report.plot_overview()
    assert "timeseries_target" in figs

    summary = report.summarize()
    assert "timeseries" in summary
    assert "recommended_lags" in summary["timeseries"]


def test_timeseries_short_series_fallback():
    t = np.arange(8)
    y = np.sin(t).astype(np.float32)
    report = DatasetReport.from_data(X=t, y=y, task="timeseries", time_index=t)

    summary = report.summarize()
    assert "timeseries" in summary
    assert isinstance(summary["timeseries"]["recommended_lags"], list)
    fig = report.plot(kind="timeseries_target")
    assert fig is not None


def test_rc_params_are_explicit_and_side_effect_free():
    before_font_size = matplotlib.rcParams["font.size"]
    before_axes_labelsize = matplotlib.rcParams["axes.labelsize"]
    before_legend_fontsize = matplotlib.rcParams["legend.fontsize"]
    before_xtick_labelsize = matplotlib.rcParams["xtick.labelsize"]
    before_ytick_labelsize = matplotlib.rcParams["ytick.labelsize"]

    params = rc_params()

    assert params["axes.labelsize"] == FONTSIZE_AXES
    assert params["axes.titlesize"] == FONTSIZE_AXES
    assert params["legend.fontsize"] == FONTSIZE_LEGEND
    assert params["font.size"] == FONTSIZE_TICKS
    assert params["xtick.labelsize"] == FONTSIZE_TICKS
    assert params["ytick.labelsize"] == FONTSIZE_TICKS

    # rc_params() should only return style settings and not mutate global rcParams.
    assert matplotlib.rcParams["font.size"] == before_font_size
    assert matplotlib.rcParams["axes.labelsize"] == before_axes_labelsize
    assert matplotlib.rcParams["legend.fontsize"] == before_legend_fontsize
    assert matplotlib.rcParams["xtick.labelsize"] == before_xtick_labelsize
    assert matplotlib.rcParams["ytick.labelsize"] == before_ytick_labelsize


def test_rc_params_support_light_and_dark_themes():
    light_params = rc_params(theme="light")
    dark_params = rc_params(theme="dark")

    assert light_params["figure.facecolor"] == "white"
    assert light_params["legend.facecolor"] == "white"
    assert light_params["text.color"] == "black"

    assert dark_params["figure.facecolor"] == "none"
    assert dark_params["legend.facecolor"] == "none"
    assert dark_params["text.color"] == "white"


def test_dataset_report_from_data_rejects_unknown_theme():
    data = load_dataset("iris", split=False, random_state=42)
    try:
        DatasetReport.from_data(data, theme="sepia")
    except ValueError as exc:
        assert "theme must be either 'light' or 'dark'" in str(exc)
    else:
        raise AssertionError("DatasetReport.from_data should reject unsupported themes.")


def test_promoted_plotting_names_are_exported_from_the_package():
    """`rc_params` and `STYLE_COLORS` are the public plotting surface.

    Eight worked examples import them.
    """
    import eigenkit.datasets as datasets

    for name in (
        "rc_params",
        "STYLE_COLORS",
        "FONTSIZE_AXES",
        "FONTSIZE_LEGEND",
        "FONTSIZE_TICKS",
    ):
        assert name in datasets.__all__
        assert hasattr(datasets, name)


def test_style_colors_cannot_be_mutated():
    """The palette is a tuple, so no caller can change every later figure."""
    assert isinstance(STYLE_COLORS, tuple)
    assert len(STYLE_COLORS) == 8
    with pytest.raises(AttributeError):
        STYLE_COLORS.append("#000000")


def test_style_colors_is_the_prop_cycle_rc_params_installs():
    """Indexing the palette and letting the cycle run give the same colours."""
    cycle = rc_params()["axes.prop_cycle"].by_key()["color"]
    assert tuple(cycle) == STYLE_COLORS


def test_plot_text_font_draws_underscores():
    """matplotlib's cmr10 keeps TeX's encoding and draws "_" as a dot accent, so
    titles and metric names such as "error_rate" would render as "error˙rate"."""
    import matplotlib
    from matplotlib import font_manager

    with matplotlib.rc_context(rc_params()):
        path = font_manager.findfont(font_manager.FontProperties())
    assert "cmr10" not in path.lower(), path
