"""Helpers for the visual analysis of a dataset, and for reproducible summaries.

This module has one main entry point,
[`DatasetReport`][eigenkit.datasets.eda.DatasetReport]. It converts built-in and
custom datasets to the same format, and then:

- it draws quick overview plots of the distributions, the correlations, and the
  relation between the features and the target;
- it draws diagnostics that are specific to time series;
- it writes a reproducible summary, as text and as a dictionary, without using a
  language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None

STYLE_COLORS = (
    "#CE2173",
    "#AA5EFF",
    "#7D4ECF",
    "#43CDFF",
    "#3333FB",
    "#11E57A",
    "#E6C541",
    "#94A3B8",
)
"""The plot palette of the library, in the order of the colour cycle.

`rc_params()` sets it as `axes.prop_cycle`, so a plot drawn inside that context
uses these colours without naming them. Index it when a series needs a fixed
colour, and not the next one in the cycle. For example, use `STYLE_COLORS[0]` for
a training curve and `STYLE_COLORS[1]` for validation, so that the two curves
keep their colours across figures. It is a tuple, so that an `append` by mistake
cannot change how every later plot in the process looks.
"""

FONTSIZE_AXES = 20
"""The font size, in points, that `rc_params()` gives the axis labels and titles."""

FONTSIZE_LEGEND = 18
"""The font size, in points, that `rc_params()` gives the legend entries."""

FONTSIZE_TICKS = 16
"""The font size, in points, of the tick labels, which is also the base `font.size`."""


@dataclass
class DatasetReport:
    """An exploratory data analysis (EDA) report for a built-in or custom dataset.

    Args:
        X (np.ndarray): The feature matrix, of shape `(n_samples, n_features)`.
        y (np.ndarray): The target vector, of shape `(n_samples,)`.
        feature_names (list[str]): Readable names of the features, in the order of
            the columns of `X`.
        target_name (str): The name of the target in the plots and the summary.
        task (str): The type of task, usually `"classification"`, `"regression"` or
            `"timeseries"`.
        metadata (Any, optional): The metadata of the dataset, for example a
            `DatasetInfo`.
        time_index (Any, optional): The time axis for the time-series plots. It can
            be numeric (array-like) or a `pd.DatetimeIndex`.
        theme (str): The theme of the plots, `"light"` or `"dark"`.

    Notes:
        The recommended way to build a report is `from_data()`, which accepts the
        dictionaries returned by `load_dataset(...)` as well as your own `X` and
        `y`.

    Example:
        ```python
        from eigenkit.datasets import DatasetReport, load_dataset

        data = load_dataset("iris", split=False)
        report = DatasetReport.from_data(data)
        figs = report.plot_overview()
        summary = report.summarize()
        ```
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    task: str
    metadata: Any | None = None
    time_index: Any | None = None
    theme: str = "light"

    @classmethod
    def from_data(
        cls,
        data: dict[str, Any] | None = None,
        *,
        X: Any | None = None,
        y: Any | None = None,
        feature_names: list[str] | None = None,
        target_name: str = "target",
        task: str | None = None,
        time_index: Any | None = None,
        theme: str = "light",
    ) -> "DatasetReport":
        """Build a report from a dataset dictionary, or from your own arrays or tensors.

        Args:
            data (dict[str, Any], optional): A dataset dictionary in the format of
                eigenkit, either `{"X", "y"}` or
                `{"X_train", "y_train", "X_test", "y_test"}`. The training and test sets
                are combined.
            X (Any, optional): Your own feature matrix, as a NumPy array, a pandas
                object, a PyTorch tensor or another array-like.
            y (Any, optional): Your own target vector, in the same formats as `X`.
            feature_names (list[str], optional): The names of the features. If they are
                not given, the names come from the metadata, or they are `feature_0`,
                `feature_1` and so on.
            target_name (str): The name of the target in the plots and the summary.
            task (str, optional): The type of task. If it is not given, it is taken
                from the metadata, or guessed from `y`.
            time_index (Any, optional): The time axis for the time-series analysis.
            theme (str): The theme of the plots, `"light"` or `"dark"`.

        Returns:
            The report, ready for plotting and for the summary.

        Raises:
            ValueError: If the inputs are missing or do not agree, for example if the
                shapes do not match, or if `feature_names` does not have one name per
                feature.
        """
        if data is not None:
            X_arr, y_arr, metadata = _extract_arrays_from_dataset_dict(data)
            inferred_task = (
                getattr(metadata, "task", None) if metadata is not None else None
            )
            inferred_feature_names = (
                getattr(metadata, "feature_names", None) if metadata is not None else None
            )
            if feature_names is None and inferred_feature_names:
                feature_names = list(inferred_feature_names)
            if task is None:
                task = inferred_task
            if time_index is None and "X" in data and task == "timeseries":
                time_index = np.asarray(data["X"]).reshape(-1)
        else:
            if X is None or y is None:
                raise ValueError(
                    "Either `data` dict or both `X` and `y` must be provided."
                )
            X_arr = _to_numpy(X)
            y_arr = _to_numpy(y).reshape(-1)
            metadata = None

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        if len(feature_names) != X_arr.shape[1]:
            raise ValueError("feature_names length must match number of features.")

        if task is None:
            task = _infer_task(y_arr)

        parsed_time_index = _parse_time_index(time_index)
        normalized_theme = _normalize_theme(theme)

        return cls(
            X=X_arr,
            y=y_arr,
            feature_names=feature_names,
            target_name=target_name,
            task=task,
            metadata=metadata,
            time_index=parsed_time_index,
            theme=normalized_theme,
        )

    def plot_overview(
        self,
        *,
        max_features: int | None = None,
        bins: int = 30,
        max_points: int = 2000,
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Build the default set of plots of the exploratory data analysis.

        Args:
            max_features (int, optional): The largest number of features shown in the
                panels about features. `None` shows every feature.
            bins (int): The number of histogram bins in the distribution plots.
            max_points (int): The largest number of points in the scatter plots. Larger
                datasets are subsampled.
            path (str | Path, optional): A directory where every overview plot is saved
                as a PNG file. With `None`, the plots are not saved.

        Returns:
            A dictionary from each kind of plot to its matplotlib figure. The
                `"timeseries_target"` plot is included only for a time-series task.
        """
        figs: dict[str, Any] = {}
        figs["data_health"] = self.plot(kind="data_health")
        figs["target"] = self.plot(kind="target", bins=bins)
        figs["distributions"] = self.plot(
            kind="distributions", max_features=max_features, bins=bins
        )
        figs["correlation"] = self.plot(kind="correlation")
        figs["feature_target"] = self.plot(
            kind="feature_target", max_features=max_features, max_points=max_points
        )

        if self.task == "timeseries":
            figs["timeseries_target"] = self.plot(kind="timeseries_target")

        if path is not None:
            output_dir = Path(path)
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, fig in figs.items():
                fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")

        return figs

    def plot(self, kind: str, **kwargs: Any) -> Any:
        """Build one plot, chosen by its kind.

        Args:
            kind (str): The kind of plot: `"data_health"`, `"target"`,
                `"distributions"`, `"correlation"`, `"feature_target"` or
                `"timeseries_target"`.
            **kwargs (Any): Options of the plot, such as `bins`, `max_features` and
                `max_points`.

        Returns:
            The matplotlib figure.

        Raises:
            ValueError: If `kind` is not one of the kinds above.
        """
        if kind == "data_health":
            return self._plot_data_health()
        if kind == "target":
            return self._plot_target_distribution(bins=kwargs.get("bins", 30))
        if kind == "distributions":
            return self._plot_feature_distributions(
                max_features=kwargs.get("max_features"),
                bins=kwargs.get("bins", 30),
            )
        if kind == "correlation":
            return self._plot_correlation()
        if kind == "feature_target":
            return self._plot_feature_target_relation(
                max_features=kwargs.get("max_features"),
                max_points=kwargs.get("max_points", 2000),
            )
        if kind == "timeseries_target":
            return self._plot_timeseries_target_analysis()
        raise ValueError(f"Unknown plot kind: {kind!r}")

    def summarize(self) -> dict[str, Any]:
        """Build a reproducible summary of the dataset, with recommendations.

        Returns:
            A dictionary with statistics about the shape and the quality of the data,
                signals specific to the task when there are any, and a readable summary
                under the `"text"` key.
        """
        if np.issubdtype(self.X.dtype, np.floating):
            missing_ratio = float(np.isnan(self.X).mean())
        else:
            missing_ratio = 0.0
        n_samples, n_features = self.X.shape

        summary: dict[str, Any] = {
            "task": self.task,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "missing_ratio": missing_ratio,
            "highly_correlated_pairs": self._highly_correlated_pairs(),
            "outlier_fraction_top_features": self._outlier_fraction_top_features(),
            "recommendations": [],
        }

        recommendations = summary["recommendations"]
        if missing_ratio > 0.05:
            recommendations.append("Missing values exceed 5%; consider imputation.")
        if summary["highly_correlated_pairs"]:
            recommendations.append(
                "Highly correlated feature pairs detected; consider feature selection."
            )
        if self.task == "classification":
            # `np.unique`, not `np.bincount`: bincount indexes by label value,
            # so it invents zero-count classes for every unused integer below
            # the maximum and rejects negative labels outright. A balanced set
            # labelled {0, 2} would report an imbalance ratio of 50, and the
            # {-1, 1} convention would crash `summarize()`.
            labels, counts = np.unique(self.y, return_counts=True)
            ratio = float(counts.max() / max(counts.min(), 1))
            summary["class_labels"] = labels.tolist()
            summary["class_counts"] = counts.tolist()
            summary["class_imbalance_ratio"] = ratio
            if ratio > 2.0:
                recommendations.append(
                    "Class imbalance detected; consider class weighting or resampling."
                )
        elif self.task in {"regression", "timeseries"}:
            summary["target_mean"] = float(np.mean(self.y))
            summary["target_std"] = float(np.std(self.y))

        if self.task == "timeseries":
            ts = self._timeseries_insights()
            summary["timeseries"] = ts
            if ts["is_likely_non_stationary"]:
                recommendations.append(
                    "Series looks non-stationary; consider differencing or detrending."
                )
            if ts["seasonality_strength"] > 0.2:
                recommendations.append(
                    "Seasonality signal detected; include seasonal features or lags."
                )

        summary["text"] = self._summary_text(summary)
        return summary

    def _plot_data_health(self) -> Any:
        """Plot per-feature missing ratio and basic dataset size context.

        Returns:
            Any:
                Matplotlib figure.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        with matplotlib.rc_context(rc_params(theme=self.theme)):
            n_samples, n_features = self.X.shape
            missing_by_feature = (
                np.isnan(self.X).mean(axis=0)
                if np.issubdtype(self.X.dtype, np.floating)
                else np.zeros(n_features, dtype=float)
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(self.feature_names, missing_by_feature)
            ax.set_title(f"Missing Ratio by Feature (n={n_samples}, d={n_features})")
            ax.set_ylabel("Missing ratio")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            return fig

    def _plot_target_distribution(self, bins: int) -> Any:
        """Plot target distribution for classification or regression-like tasks.

        Args:
            bins (int):
                Number of histogram bins for continuous targets.

        Returns:
            Any:
                Matplotlib figure.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        with matplotlib.rc_context(rc_params(theme=self.theme)):
            fig, ax = plt.subplots(figsize=(8, 4))
            if self.task == "classification":
                classes, counts = np.unique(self.y.astype(int), return_counts=True)
                ax.bar(classes.astype(str), counts)
                ax.set_title("Target Class Distribution")
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
            else:
                ax.hist(self.y, bins=bins, alpha=0.85)
                ax.set_title("Target Distribution")
                ax.set_xlabel(self.target_name)
                ax.set_ylabel("Count")
                if sns is not None:
                    sns.kdeplot(self.y, ax=ax, color="black")
            fig.tight_layout()
            return fig

    def _plot_feature_distributions(self, max_features: int | None, bins: int) -> Any:
        """Plot feature distributions as a compact subplot grid.

        Args:
            max_features (int, optional):
                Optional cap on shown features. ``None`` shows all.
            bins (int):
                Histogram bin count.

        Returns:
            Any:
                Matplotlib figure.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        with matplotlib.rc_context(rc_params(theme=self.theme)):
            chosen = self._top_feature_indices(max_features=max_features)
            n = len(chosen)
            ncols = min(3, max(1, n))
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(18, 5 * nrows),
                squeeze=False,
            )
            axes_flat = axes.flatten()

            for i, idx in enumerate(chosen):
                ax = axes_flat[i]
                vals = self.X[:, idx]
                clean_vals = (
                    vals[~np.isnan(vals)]
                    if np.issubdtype(vals.dtype, np.floating)
                    else vals
                )
                if sns is not None and np.issubdtype(vals.dtype, np.floating):
                    sns.histplot(clean_vals, kde=True, ax=ax, bins=bins)
                else:
                    ax.hist(clean_vals, bins=bins, alpha=0.85)
                mean_val = float(np.mean(clean_vals))
                std_val = float(np.std(clean_vals))
                ax.set_title(
                    f"Distribution: {self.feature_names[idx]}\n"
                    f"(mean: {mean_val:.2f}, std: {std_val:.2f})"
                )
                ax.set_xlabel("")

            for j in range(n, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.tight_layout()
            return fig

    def _plot_correlation(self) -> Any:
        """Plot Pearson correlation heatmap.

        For regression/time-series tasks, target is included as the last
        row/column.

        Returns:
            Any:
                Matplotlib figure.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        with matplotlib.rc_context(rc_params(theme=self.theme)):
            if self.task in {"regression", "timeseries"}:
                matrix = np.column_stack([self.X, self.y])
                labels = [*self.feature_names, self.target_name]
                title = "Feature + target correlation (Pearson)"
            else:
                matrix = self.X
                labels = self.feature_names
                title = "Feature Correlation (Pearson)"

            corr = np.corrcoef(matrix, rowvar=False)
            fig, ax = plt.subplots(figsize=(7, 6))
            corr_cmap = LinearSegmentedColormap.from_list(
                "ss_palette_corr",
                ["#FFFFFF", "#433499"],
            )
            heatmap = ax.imshow(corr, cmap=corr_cmap, vmin=-1.0, vmax=1.0)
            n = len(labels)
            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(n))
            ax.set_xticklabels(labels, fontsize=12, rotation=45, ha="right")
            ax.set_yticklabels(labels, fontsize=12)
            ax.set_title(title)
            cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r"Correlation, $\rho$", fontsize=14)
            cbar.ax.tick_params(labelsize=12)
            fig.tight_layout()
            return fig

    def _plot_feature_target_relation(
        self, max_features: int | None, max_points: int
    ) -> Any:
        """Plot feature-target relationship as subplot grid.

        Args:
            max_features (int, optional):
                Optional cap on feature count. ``None`` shows all.
            max_points (int):
                Max sample count for scatter-heavy regression panels.

        Returns:
            Any:
                Matplotlib figure.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        with matplotlib.rc_context(rc_params(theme=self.theme)):
            chosen = self._top_feature_indices(max_features=max_features, by_target=True)
            n = len(chosen)
            ncols = min(3, max(1, n))
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(18, 5 * nrows),
                squeeze=False,
            )
            axes_flat = axes.flatten()

            idx_points = np.arange(len(self.y))
            if len(idx_points) > max_points:
                idx_points = np.linspace(0, len(self.y) - 1, max_points).astype(int)

            for i, feature_idx in enumerate(chosen):
                ax = axes_flat[i]
                xv = self.X[idx_points, feature_idx]
                yv = self.y[idx_points]
                if self.task == "classification":
                    classes = np.unique(yv.astype(int))
                    for c in classes:
                        class_vals = xv[yv.astype(int) == c]
                        if sns is not None:
                            sns.histplot(
                                class_vals,
                                kde=True,
                                bins=20,
                                stat="count",
                                alpha=0.35,
                                ax=ax,
                                label=f"class {int(c)}",
                            )
                        else:
                            ax.hist(
                                class_vals, bins=20, alpha=0.5, label=f"class {int(c)}"
                            )
                    ax.legend(loc="best")
                    ax.set_ylabel("Count")
                else:
                    if sns is not None:
                        sns.regplot(
                            x=xv,
                            y=yv,
                            ax=ax,
                            scatter_kws={"s": 14, "alpha": 0.45},
                            line_kws={"color": "tab:red"},
                            ci=None,
                        )
                    else:
                        ax.scatter(xv, yv, s=12, alpha=0.5)
                    ax.set_ylabel(self.target_name)
                ax.set_xlabel(self.feature_names[feature_idx])
                ax.set_title(f"Feature vs target: {self.feature_names[feature_idx]}")

            for j in range(n, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.tight_layout()
            return fig

    def _plot_timeseries_target_analysis(self) -> Any:
        """Plot time-series target diagnostics.

        If ``time_index`` is not provided, falls back to a simple
        ``plt.plot(target)`` chart.

        Returns:
            Any:
                Matplotlib figure.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        with matplotlib.rc_context(rc_params(theme=self.theme)):
            if self.time_index is None:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(self.y, linewidth=1.0)
                ax.set_title("Target over sample index")
                ax.set_xlabel("sample")
                ax.set_ylabel(self.target_name)
                fig.tight_layout()
                return fig

            time_axis = self.time_index
            period = self._infer_seasonal_period()
            rolling_windows = [7, 14, 30]

            fig, axes = plt.subplots(3, 2, figsize=(13, 10))
            ax_line, ax_roll = axes[0]
            ax_season, ax_shift = axes[1]
            ax_acf, ax_pacf = axes[2]

            ax_line.plot(time_axis, self.y, linewidth=1.0)
            ax_line.set_title("Target over time")

            ax_roll.plot(time_axis, self.y, alpha=0.35, label="target")
            for w in rolling_windows:
                if len(self.y) >= w:
                    roll = _rolling_mean(self.y, w)
                    ax_roll.plot(time_axis, roll, label=f"roll_mean_{w}")
            ax_roll.legend(loc="best")
            ax_roll.set_title("Rolling means")

            season_idx = np.arange(len(self.y)) % max(period, 1)
            grouped = [self.y[season_idx == i] for i in range(max(period, 1))]
            grouped = [g for g in grouped if len(g) > 0]
            ax_season.boxplot(grouped, showfliers=False)
            ax_season.set_title("Seasonality profile")
            ax_season.set_xlabel("Season slot")

            split = len(self.y) // 2
            ax_shift.hist(self.y[:split], bins=20, alpha=0.6, label="early")
            ax_shift.hist(self.y[split:], bins=20, alpha=0.6, label="late")
            ax_shift.legend(loc="best")
            ax_shift.set_title("Distribution shift (early vs late)")

            max_lag = min(40, len(self.y) - 1)
            acf = _acf(self.y, max_lag=max_lag)
            pacf = _pacf(self.y, max_lag=max_lag)
            ax_acf.stem(range(len(acf)), acf)
            ax_acf.set_title("ACF")
            ax_pacf.stem(range(len(pacf)), pacf)
            ax_pacf.set_title("PACF")

            fig.tight_layout()
            return fig

    def _top_feature_indices(
        self, *, max_features: int | None, by_target: bool = False
    ) -> list[int]:
        """Return selected feature indices for plotting.

        Args:
            max_features (int, optional):
                Optional cap. ``None`` means all features.
            by_target (bool):
                If ``True`` and task is regression/timeseries, rank by
                absolute feature-target correlation; otherwise rank by
                variance.

        Returns:
            list[int]:
                Ordered list of selected feature indices.

        Raises:
            ValueError:
                If ``max_features`` is provided and is non-positive.
        """
        n_features = self.X.shape[1]
        if max_features is None:
            return list(range(n_features))

        if max_features <= 0:
            raise ValueError("max_features must be positive when provided.")

        k = min(max_features, n_features)
        if n_features <= k:
            return list(range(n_features))

        if by_target and self.task in {"regression", "timeseries"}:
            scores = []
            for i in range(n_features):
                corr = _safe_corr(self.X[:, i], self.y)
                scores.append(abs(corr))
            return list(np.argsort(scores)[::-1][:k])

        vars_ = np.nanvar(self.X, axis=0)
        return list(np.argsort(vars_)[::-1][:k])

    def _highly_correlated_pairs(self, threshold: float = 0.9) -> list[dict[str, Any]]:
        """Find highly correlated feature pairs.

        Args:
            threshold (float):
                Absolute Pearson threshold for reporting a pair.

        Returns:
            list[dict[str, Any]]:
                Pair records with keys ``feature_a``, ``feature_b``, ``corr``.
        """
        if self.X.shape[1] < 2:
            return []
        corr = np.corrcoef(self.X, rowvar=False)
        pairs: list[dict[str, Any]] = []
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[1]):
                val = float(corr[i, j])
                if abs(val) >= threshold:
                    pairs.append(
                        {
                            "feature_a": self.feature_names[i],
                            "feature_b": self.feature_names[j],
                            "corr": val,
                        }
                    )
        return pairs

    def _outlier_fraction_top_features(self) -> list[dict[str, Any]]:
        """Estimate outlier fraction (IQR rule) for top features.

        Returns:
            list[dict[str, Any]]:
                Per-feature outlier fractions.
        """
        top_indices = self._top_feature_indices(max_features=min(5, self.X.shape[1]))
        out: list[dict[str, Any]] = []
        for idx in top_indices:
            x = self.X[:, idx]
            if not np.issubdtype(x.dtype, np.floating):
                continue
            q1, q3 = np.nanpercentile(x, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                frac = 0.0
            else:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                frac = float(np.mean((x < lo) | (x > hi)))
            out.append({"feature": self.feature_names[idx], "fraction": frac})
        return out

    def _timeseries_insights(self) -> dict[str, Any]:
        """Compute deterministic time-series insight heuristics.

        Returns:
            dict[str, Any]:
                Signals including trend/seasonality strengths, recommended lags,
                and a coarse non-stationarity flag.
        """
        y = self.y
        if len(y) < 4:
            return {
                "seasonality_strength": 0.0,
                "trend_strength": 0.0,
                "recommended_lags": [],
                "is_likely_non_stationary": False,
            }

        period = self._infer_seasonal_period()
        trend = _rolling_mean(y, window=max(3, min(30, len(y) // 10)))
        resid = y - trend
        var_y = float(np.var(y)) + 1e-12
        if len(y) < 2 * period:
            # With fewer than two periods, every residue class holds at most
            # one point, so the "seasonal component" reproduces the series and
            # the strength saturates at 1.0 -- a monotone 8-point ramp would
            # report full seasonality and a recommendation to add seasonal
            # features. Classes past the end are also empty, and each would
            # emit a numpy mean-of-empty-slice warning.
            seasonality_strength = 0.0
        else:
            season_component = np.array(
                [np.mean(y[np.arange(len(y)) % period == i]) for i in range(period)]
            )
            season_signal = season_component[np.arange(len(y)) % period]
            seasonality_strength = float(np.var(season_signal) / var_y)
        trend_strength = float(np.var(trend) / var_y)

        acf_vals = _acf(y, max_lag=min(30, len(y) - 1))
        lag_indices = np.argsort(np.abs(acf_vals[1:]))[::-1][:3] + 1

        first_half = y[: len(y) // 2]
        second_half = y[len(y) // 2 :]
        non_stationary = abs(float(np.mean(first_half) - np.mean(second_half))) > 0.25 * (
            float(np.std(y)) + 1e-12
        )

        return {
            "seasonality_strength": seasonality_strength,
            "trend_strength": trend_strength,
            "recommended_lags": lag_indices.astype(int).tolist(),
            "is_likely_non_stationary": bool(non_stationary),
            "residual_std": float(np.std(resid)),
        }

    def _infer_seasonal_period(self) -> int:
        """Infer a practical seasonal period from available time index.

        Returns:
            int:
                Period used for seasonal grouping (fallback is 12).
        """
        if self.time_index is None:
            return 12
        if pd is not None and isinstance(self.time_index, pd.DatetimeIndex):
            freq = self.time_index.inferred_freq
            if freq is None:
                return 12
            freq = freq.upper()
            if freq.startswith("H"):
                return 24
            if freq.startswith("D"):
                return 7
            if freq.startswith("M"):
                return 12
            return 12
        return 12

    def _summary_text(self, summary: dict[str, Any]) -> str:
        """Compose compact human-readable summary sentence block.

        Args:
            summary (dict[str, Any]):
                Structured output of `summarize()`.

        Returns:
            str:
                Human-readable one-line summary.
        """
        pieces = [
            f"Task: {summary['task']}",
            f"Samples: {summary['n_samples']}, features: {summary['n_features']}",
            f"Missing ratio: {summary['missing_ratio']:.3f}",
        ]
        if summary["highly_correlated_pairs"]:
            pieces.append(
                f"High correlations: {len(summary['highly_correlated_pairs'])} feature pairs."
            )
        if self.task == "classification" and "class_imbalance_ratio" in summary:
            pieces.append(
                f"Class imbalance ratio: {summary['class_imbalance_ratio']:.2f}"
            )
        if self.task == "timeseries":
            ts = summary["timeseries"]
            pieces.append(
                "Time-series: "
                f"seasonality={ts['seasonality_strength']:.2f}, "
                f"trend={ts['trend_strength']:.2f}, "
                f"lags={ts['recommended_lags']}."
            )
        if summary["recommendations"]:
            pieces.append("Recommendations: " + " ".join(summary["recommendations"]))
        return " ".join(pieces)


def _extract_arrays_from_dataset_dict(
    data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Any | None]:
    """Extract full ``X/y`` arrays from a eigenkit dataset dict.

    Args:
        data (dict[str, Any]):
            Dataset dict with either full data keys ``X/y`` or split keys
            ``X_train/y_train/X_test/y_test``.

    Returns:
        tuple[np.ndarray, np.ndarray, Any | None]:
            ``(X, y, metadata)``.

    Raises:
        ValueError:
            If dict does not match supported layouts.
    """
    metadata = data.get("metadata")
    if "X" in data and "y" in data:
        X = _to_numpy(data["X"])
        y = _to_numpy(data["y"]).reshape(-1)
        return X, y, metadata
    if {"X_train", "X_test", "y_train", "y_test"} <= data.keys():
        X_train = _to_numpy(data["X_train"])
        X_test = _to_numpy(data["X_test"])
        y_train = _to_numpy(data["y_train"]).reshape(-1)
        y_test = _to_numpy(data["y_test"]).reshape(-1)
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X, y, metadata
    raise ValueError(
        "Unsupported dataset dict format. Expected {X,y} or "
        "{X_train,y_train,X_test,y_test}."
    )


def _to_numpy(value: Any) -> np.ndarray:
    """Convert supported tensor/table inputs to numpy array.

    Args:
        value (Any):
            Input value (numpy/pandas/torch/array-like).

    Returns:
        np.ndarray:
            Converted array.
    """
    if isinstance(value, np.ndarray):
        return value
    if pd is not None:
        if isinstance(value, pd.DataFrame):
            return value.to_numpy()
        if isinstance(value, pd.Series):
            return value.to_numpy()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _infer_task(y: np.ndarray) -> str:
    """Heuristically infer task type from target values.

    Args:
        y (np.ndarray):
            Target values.

    Returns:
        str:
            ``"classification"`` or ``"regression"``.
    """
    if np.issubdtype(y.dtype, np.integer):
        return "classification"
    if len(np.unique(y)) <= 20 and np.allclose(y, np.round(y)):
        return "classification"
    return "regression"


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute robust Pearson correlation for two vectors.

    Args:
        a (np.ndarray):
            First vector.
        b (np.ndarray):
            Second vector.

    Returns:
        float:
            Correlation value, or ``0.0`` for zero-variance inputs.
    """
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Compute centered rolling mean with edge padding.

    Args:
        y (np.ndarray):
            Input series.
        window (int):
            Window size.

    Returns:
        np.ndarray:
            Smoothed series with same length as input.
    """
    if window <= 1:
        return y.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _acf(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function up to ``max_lag``.

    Args:
        y (np.ndarray):
            Input series.
        max_lag (int):
            Maximum lag.

    Returns:
        np.ndarray:
            ACF vector with length ``max_lag + 1``.
    """
    y0 = y - np.mean(y)
    denom = np.dot(y0, y0) + 1e-12
    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(y0[:-lag], y0[lag:]) / denom
    return acf


def _pacf(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute partial autocorrelation via lagged least squares.

    Args:
        y (np.ndarray):
            Input series.
        max_lag (int):
            Maximum lag.

    Returns:
        np.ndarray:
            PACF vector with length ``max_lag + 1``.
    """
    pacf = np.empty(max_lag + 1, dtype=float)
    pacf[0] = 1.0
    for lag in range(1, max_lag + 1):
        X = np.column_stack([y[lag - j - 1 : -(j + 1)] for j in range(lag)])
        target = y[lag:]
        X = np.column_stack([np.ones(len(X)), X])
        coeffs, *_ = np.linalg.lstsq(X, target, rcond=None)
        pacf[lag] = coeffs[-1]
    return pacf


def _parse_time_index(time_index: Any | None) -> np.ndarray | None:
    """Normalize optional time index without implicit datetime conversion.

    Args:
        time_index (Any, optional):
            Time index value.

    Returns:
        np.ndarray | None:
            ``None`` if missing; otherwise 1-D numeric array or
            unchanged ``pd.DatetimeIndex``.
    """
    if time_index is None:
        return None
    if pd is not None and isinstance(time_index, pd.DatetimeIndex):
        return time_index
    return np.asarray(time_index).reshape(-1)


def _normalize_theme(theme: str) -> str:
    """Normalize EDA theme name and validate supported values."""
    normalized = theme.strip().lower()
    if normalized not in {"light", "dark"}:
        raise ValueError("theme must be either 'light' or 'dark'.")
    return normalized


def rc_params(theme: str = "light") -> dict:
    """Return the matplotlib settings of the library for one plot, as rcParams.

    Every figure that the library draws uses these settings:
    `DatasetReport.plot_overview()`, the benchmark curves and the plots in the worked
    examples. This is what makes them look like one set. Pass the result to
    `matplotlib.rc_context()` and draw as usual. The settings apply only inside that
    block, because the returned dictionary is a new copy and the global
    `matplotlib.rcParams` is never changed.

    The two themes differ only in what has to change for a dark page. `"dark"` makes
    the backgrounds of the figure, the axes and the legend transparent, and makes the
    text, the ticks and the spines white. The sizes, the spines, the grid and
    [`STYLE_COLORS`][eigenkit.datasets.STYLE_COLORS] are the same in both themes, so
    a figure saved with each theme differs only in its background and in the colour
    of its text and lines.

    Args:
        theme (str): `"light"` or `"dark"`. The case and any surrounding spaces are
            ignored.

    Returns:
        The rcParams to draw with, ready for `matplotlib.rc_context()`.

    Raises:
        ValueError: If `theme` is neither `"light"` nor `"dark"`.

    Example:
        ```python
        import matplotlib
        import matplotlib.pyplot as plt
        from eigenkit.datasets import STYLE_COLORS, rc_params

        with matplotlib.rc_context(rc_params(theme="light")):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(history, color=STYLE_COLORS[0], label="train loss")
            ax.legend()
            fig.savefig("loss.png", bbox_inches="tight")
        ```
    """
    import matplotlib as mpl

    normalized_theme = _normalize_theme(theme)
    text_color = "black" if normalized_theme == "light" else "white"
    plot_background = "white" if normalized_theme == "light" else "none"
    legend_background = "white" if normalized_theme == "light" else "none"
    legend_frame_alpha = 1.0 if normalized_theme == "light" else 0.0

    params = {
        "font.size": FONTSIZE_TICKS,
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "figure.dpi": 200,
        "savefig.dpi": 450,
        "axes.labelsize": FONTSIZE_AXES,
        "axes.titlesize": FONTSIZE_AXES,
        "xtick.labelsize": FONTSIZE_TICKS,
        "ytick.labelsize": FONTSIZE_TICKS,
        "legend.fontsize": FONTSIZE_LEGEND,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "axes.prop_cycle": mpl.cycler(color=STYLE_COLORS),
        "figure.facecolor": plot_background,
        "axes.facecolor": plot_background,
        "savefig.facecolor": plot_background,
        "legend.facecolor": legend_background,
        "legend.framealpha": legend_frame_alpha,
        "legend.frameon": False,
        "text.color": text_color,
        "axes.labelcolor": text_color,
        "axes.titlecolor": text_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "axes.edgecolor": text_color,
        "legend.edgecolor": text_color,
    }

    # STIX, which matplotlib ships, rather than its Computer Modern TTF: cmr10
    # keeps TeX's encoding, so "_" in a title or a metric name renders as a
    # dot accent ("error˙rate").
    params["font.family"] = "serif"
    params["font.serif"] = ["STIXGeneral", "DejaVu Serif"]
    params["mathtext.fontset"] = "stix"

    return params
