"""BenchmarkResult — container and formatters for compare() output."""

from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..datasets.eda import STYLE_COLORS, _normalize_theme, rc_params
from ._metrics import (
    BUILTIN_METRIC_DIRECTIONS,
    MetricDirection,
    validate_metric_directions,
)


if TYPE_CHECKING:
    import pandas as pd


_EXTRAS_HINT = "Install with: pip install eigenkit[benchmark]"


def _load_matplotlib(function_name: str) -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rc_context
    except ImportError as exc:
        raise ImportError(
            f"matplotlib is required for {function_name}(). {_EXTRAS_HINT}"
        ) from exc
    return rc_context, plt


_TIME_METRICS: frozenset[str] = frozenset({"fit_time", "predict_time"})

_METRIC_TITLES = {
    "accuracy": "Accuracy",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "roc_auc": "ROC-AUC",
    "r2": "R²",
    "mae": "MAE",
    "rmse": "RMSE",
    "mse": "MSE",
    "fit_time": "Fit time (s)",
    "predict_time": "Predict time (s)",
}


def _format_cell(metric: str, mean: float, std: float) -> str:
    if metric in _TIME_METRICS:
        # Two significant figures: kernel fits often take milliseconds, which
        # a fixed two decimals would round to "0.00 ± 0.00".
        return f"{mean:.2g} ± {std:.2g}"
    return f"{mean:.3f} ± {std:.3f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _restore_prediction_arrays(predictions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    restored: dict[str, dict[str, Any]] = {}
    for model, values in predictions.items():
        restored[model] = {
            key: np.asarray(value) if value is not None else None
            for key, value in values.items()
        }
    return restored


@dataclass
class BenchmarkResult:
    """Results of a benchmark comparison.

    It stores the raw metric values of every seed, and aggregates them when you
    display them with `table()`, plot them with `plot()`, or analyse them with
    `to_dataframe()`.

    Each entry in `raw` is a dict with the keys `"model"`, `"seed"`, `"metric"` and
    `"value"`.

    `fitted_models` holds every fitted copy produced by
    [`compare()`][eigenkit.benchmark.compare], keyed by the name of the model in the
    benchmark. The names usually match the class names, and repeated names are
    numbered (for example, `SVC_0` and `SVC_1`). If `compare(models={...})`
    receives a dict, its keys are used instead.

    Example:
        ```python
        from sklearn.svm import SVC
        from eigenkit.qml import QSVC
        from eigenkit.benchmark import compare

        results = compare(
            models={
                "Quantum SVC": QSVC(
                    feature_map="Angle", reps=1, probability=True
                ),
                "Regularized SVC": SVC(C=0.3),
                "Flexible SVC": SVC(C=3.0),
            },
            dataset="iris",
            metrics=["accuracy"],
            n_seeds=1,
        )
        print(list(results.fitted_models))
        # ['Quantum SVC', 'Regularized SVC', 'Flexible SVC']
        results.plot(save_path="svc_comparison.png")
        df = results.to_dataframe()
        X_new = [[5.1, 3.5, 1.4, 0.2]]

        # Access the fitted estimators
        winner = results.best_model(metric="accuracy")
        preds = results.predict(X_new)
        quantum_fits = results.fitted_models["Quantum SVC"]

        # Run inference on new data, with all models or a subset
        out = results.inference(X_new)
        out = results.inference(X_new, models=["Quantum SVC"], seed=0)
        ```
    """

    raw: list[dict[str, Any]] = field(default_factory=list)
    fitted_models: dict[str, list[Any]] = field(default_factory=dict)
    predictions: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_artifacts: dict[str, list[str | None]] = field(default_factory=dict)
    metric_directions: dict[str, MetricDirection] = field(default_factory=dict)
    loss_curves: dict[str, list[dict[str, list[float]] | None]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.metric_directions = validate_metric_directions(self.metric_directions)
        if self.raw:
            raw_metrics = {str(entry["metric"]) for entry in self.raw}
            unknown = sorted(
                name
                for name in self.metric_directions
                if name not in BUILTIN_METRIC_DIRECTIONS and name not in raw_metrics
            )
            if unknown:
                raise ValueError(f"Unknown metric direction name(s): {unknown}.")

    def _raw_by_model(self) -> dict[str, dict[str, list[float]]]:
        """`{model: {metric: [value per seed]}}`, ungrouped and unaggregated."""
        grouped: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for entry in self.raw:
            grouped[entry["model"]][entry["metric"]].append(float(entry["value"]))
        return grouped

    def _summarize(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Aggregate raw results into ``{model: {metric: (mean, std)}}``."""
        grouped: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for entry in self.raw:
            grouped[entry["model"]][entry["metric"]].append(float(entry["value"]))

        summary: dict[str, dict[str, tuple[float, float]]] = {}
        for model, metrics in grouped.items():
            summary[model] = {}
            for metric, all_values in metrics.items():
                # A metric can be NaN without raising -- `roc_auc_score` returns
                # NaN on a degenerate split rather than failing -- and one NaN
                # seed would make the whole mean NaN, so a model scoring 0.99 on
                # four of five seeds would lose to one scoring 0.60. Aggregate over the
                # seeds that produced a number; a metric with none is left out
                # entirely, which `table()` already renders as a dash.
                values = [v for v in all_values if not math.isnan(v)]
                if not values:
                    continue
                mean = sum(values) / len(values)
                if len(values) > 1:
                    std = math.sqrt(
                        sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                    )
                else:
                    std = 0.0
                summary[model][metric] = (mean, std)
        return summary

    def _ordered_models(self) -> list[str]:
        seen: list[str] = []
        for entry in self.raw:
            m = entry["model"]
            if m not in seen:
                seen.append(m)
        return seen

    def _ordered_metrics(self) -> list[str]:
        seen: list[str] = []
        for entry in self.raw:
            m = entry["metric"]
            if m not in seen:
                seen.append(m)
        return seen

    def _metric_values_by_seed(self, model: str, metric: str) -> dict[int, float]:
        values: dict[int, float] = {}
        for entry in self.raw:
            if entry["model"] == model and entry["metric"] == metric:
                values[int(entry["seed"])] = float(entry["value"])
        return values

    def _metric_direction(self, metric: str) -> MetricDirection:
        if metric in self.metric_directions:
            return self.metric_directions[metric]
        if metric in BUILTIN_METRIC_DIRECTIONS:
            return BUILTIN_METRIC_DIRECTIONS[metric]
        raise ValueError(
            f"No direction registered for metric {metric!r}. "
            "Custom metrics must be created with custom_metric(...)."
        )

    def _get_fitted(self, name: str, seed: str | int) -> Any:
        """Return one fitted estimator for *name* at the requested *seed*.

        A seed the model failed on holds a `None` slot, so that position keeps
        meaning "the run for seed n" rather than "the n-th run that worked".
        `"last"` steps back over those to the newest estimator that exists;
        an explicit index says the run failed instead of returning `None` for
        the caller to hit as an `AttributeError` inside `predict`.
        """
        models_list = self.fitted_models[name]
        if not models_list:
            raise ValueError(
                f"No fitted estimator stored for {name!r}. Re-run compare() to "
                f"populate fitted_models."
            )
        if seed == "last":
            for model in reversed(models_list):
                if model is not None:
                    return model
            raise ValueError(
                f"{name!r} failed on every seed, so there is no fitted "
                f"estimator for it. The failures were reported as warnings "
                f"during compare()."
            )
        if isinstance(seed, int):
            if seed < 0 or seed >= len(models_list):
                raise IndexError(f"seed {seed} out of range (0..{len(models_list) - 1}).")
            model = models_list[seed]
            if model is None:
                raise ValueError(
                    f"{name!r} failed on seed {seed}, so there is no fitted "
                    f"estimator for it. Pass seed='last' for its newest "
                    f"successful run, or another index."
                )
            return model
        raise TypeError(f"seed must be 'last' or an int, got {seed!r}.")

    def _primary_metric(self) -> str:
        """Return the first non-timing metric present in the results."""
        metrics = self._ordered_metrics()
        if not metrics:
            raise ValueError(
                "No metrics were recorded — all requested metrics were skipped "
                "(e.g. roc_auc requires probability=True). "
                "Check that at least one metric is computable for your models."
            )
        for m in metrics:
            if m not in _TIME_METRICS:
                return m
        return metrics[0]

    def _best_model_name(self, metric: str) -> str:
        summary = self._summarize()
        candidates = {
            model: metrics[metric][0]
            for model, metrics in summary.items()
            if metric in metrics
        }
        if not candidates:
            raise ValueError(f"No results for metric {metric!r}.")
        if self._metric_direction(metric) == "lower":
            return min(candidates, key=candidates.get)
        return max(candidates, key=candidates.get)

    def best_model(self, metric: str | None = None, seed: str | int = "last") -> Any:
        """Return the fitted estimator with the best mean score on `metric`.

        When `metric` is `None`, the default, the first metric recorded in `raw` that
        is not a timing metric is used. With the default metrics, this is usually
        `"accuracy"` for classification and `"r2"` for regression. If custom metrics
        were passed in a different order, the first of them is used.

        Args:
            metric: The metric used for ranking, for example `"accuracy"` or `"r2"`.
                `None` selects the primary metric of the task. Built-in metrics use
                their registered direction, and custom metrics use the direction given
                to `custom_metric(...)`.
            seed: The seed whose estimator is returned for the winning model.
                `"last"`, the default, returns the estimator of the last seed. An `int`
                is a 0-based index into the list of seeds, in the order they ran.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If no fitted models are stored, or if `metric` is not in the
                results.
            IndexError: If `seed` is out of range.
        """
        if not self.fitted_models:
            raise ValueError(
                "No fitted models stored. Re-run compare() to populate fitted_models."
            )

        if metric is None:
            metric = self._primary_metric()

        summary = self._summarize()
        # `_summarize` already drops NaN seeds, so a surviving value is a real
        # number. The explicit guard is kept anyway because `min`/`max` over a
        # NaN is order-dependent -- a broken model would win or lose purely on
        # where it sits in `raw`.
        candidates = {
            name: summary[name][metric][0]
            for name in summary
            if metric in summary[name]
            and name in self.fitted_models
            and not math.isnan(summary[name][metric][0])
        }
        if not candidates:
            raise ValueError(
                f"No model has a usable {metric!r} score to rank by. "
                f"Available metrics: {self._ordered_metrics()}. A metric that "
                f"was NaN on every seed -- roc_auc on a degenerate split, for "
                f"instance -- is not ranked."
            )

        if self._metric_direction(metric) == "lower":
            best_name = min(candidates, key=candidates.__getitem__)
        else:
            best_name = max(candidates, key=candidates.__getitem__)
        return self._get_fitted(best_name, seed)

    def predict(self, X: Any, metric: str | None = None) -> Any:
        """Predict with the best model ranked by `metric`.

        This is a shortcut for `best_model()`. It calls `predict()` on the estimator
        returned by `best_model(metric=metric)`.

        Args:
            X: The input features.
            metric: The metric used to select the best model. `None`, the default,
                selects the primary metric of the task: `"accuracy"` for
                classification and `"r2"` for regression.

        Returns:
            The predictions of the best estimator's `predict()` method.
        """
        return self.best_model(metric=metric).predict(X)

    def inference(
        self,
        X: Any,
        models: list[str] | None = None,
        seed: str | int = "last",
    ) -> dict[str, dict[str, Any]]:
        """Run every fitted model, or a subset of them, on new data.

        Args:
            X: The input features to predict on.
            models: The names of the models to run. `None`, the default, runs all
                fitted models. Use the exact keys of `fitted_models`: the names based on
                the class for a list input (for example, `"SVC"` or `"SVC_0"`), or your
                own keys for a dict input.
            seed: The seed whose estimator is used for each model. `"last"`, the
                default, uses the estimator of the last seed. An `int` is a 0-based
                index into the list of seeds, in the order they ran.

        Returns:
            A dict keyed by model name. Each value contains two entries:

                - `"predictions"`: the array returned by `estimator.predict(X)`.
                - `"probabilities"`: the array returned by `estimator.predict_proba(X)`
                    for classifiers that support it, and `None` for regressors and for
                    classifiers without `predict_proba`.

        Raises:
            ValueError: If no fitted models are stored, or if an unknown model name is
                requested.
            IndexError: If `seed` is out of range for any model.

        Example:
            ```python
            out = results.inference(X_new)
            out = results.inference(X_new, models=["Flexible SVC"], seed=0)
            preds = out["Flexible SVC"]["predictions"]
            probs = out["Flexible SVC"]["probabilities"]  # None for regressors
            ```
        """
        if not self.fitted_models:
            raise ValueError(
                "No fitted models stored. Re-run compare() to populate fitted_models."
            )

        target_names: list[str] = (
            models if models is not None else list(self.fitted_models.keys())
        )
        unknown = [n for n in target_names if n not in self.fitted_models]
        if unknown:
            raise ValueError(
                f"Unknown model name(s): {unknown}. "
                f"Available: {list(self.fitted_models.keys())}"
            )

        output: dict[str, dict[str, Any]] = {}
        for name in target_names:
            estimator = self._get_fitted(name, seed)
            predictions = estimator.predict(X)
            probabilities: Any = None
            if hasattr(estimator, "predict_proba"):
                try:
                    probabilities = estimator.predict_proba(X)
                except (NotImplementedError, AttributeError, ValueError) as exc:
                    warnings.warn(
                        f"{name}: predict_proba failed ({type(exc).__name__}: {exc}); "
                        "probabilities will be None.",
                        UserWarning,
                        stacklevel=2,
                    )
            output[name] = {"predictions": predictions, "probabilities": probabilities}

        return output

    _MIN_PAIRS_FOR_SIGNIFICANCE = 6

    def _significance_pair_counts(self, metric: str, best: str) -> dict[str, int]:
        """Paired-seed count for each comparison against *best*.

        A Wilcoxon signed-rank test over n paired seeds cannot produce a
        two-sided p below `2 ** (1 - n)`, so 5 pairs bottom out at 0.0625 and
        nothing can be marked however large the difference; 6 reach 0.03125.
        What matters is therefore the number of pairs *per comparison*, not the
        run's nominal `n_seeds`: seed counts differ between models whenever one
        of them failed on a seed, and `significance_test` intersects the seeds
        of each pair.

        The count is an upper bound on the effective n, because
        `scipy.stats.wilcoxon` discards zero differences by default
        (`zero_method="wilcox"`).
        """
        best_values = self._metric_values_by_seed(best, metric)
        counts: dict[str, int] = {}
        for model in self._ordered_models():
            if model == best:
                continue
            other = self._metric_values_by_seed(model, metric)
            if other:
                counts[model] = len(set(best_values) & set(other))
        return counts

    def _significance_annotations(self, metric: str) -> dict[str, str]:
        summary = self._summarize()
        if not summary:
            return {}
        try:
            best = self._best_model_name(metric)
        except ValueError:
            return {}

        # Keyed on the pairs each comparison actually has. Taking `max` over
        # every model's seed count would ask the wrong question: with one model
        # on 6 seeds and another on 3, nothing would warn and the 3-seed pair
        # still could not be marked -- which is exactly the "unmarked table reads
        # as no significant difference" failure the warning exists to prevent.
        pair_counts = self._significance_pair_counts(metric, best)
        floor = self._MIN_PAIRS_FOR_SIGNIFICANCE
        if pair_counts and max(pair_counts.values()) < floor:
            most_pairs = max(pair_counts.values())
            warnings.warn(
                f"show_significance needs at least {floor} paired seeds to "
                f"reach p < 0.05 with a Wilcoxon signed-rank test; the best-"
                f"supported comparison against {best!r} has {most_pairs}, so "
                f"no result can be marked significant regardless of the "
                f"difference. Raise n_seeds, or read the means and spreads.",
                UserWarning,
                stacklevel=3,
            )
            return {}
        underpowered = sorted(m for m, n in pair_counts.items() if n < floor)
        if underpowered:
            warnings.warn(
                f"These comparisons against {best!r} have fewer than {floor} "
                f"paired seeds, so they cannot be marked significant however "
                f"large the difference: "
                f"{', '.join(f'{m} ({pair_counts[m]})' for m in underpowered)}. "
                f"An unmarked pair here is untested, not equal.",
                UserWarning,
                stacklevel=3,
            )

        try:
            matrix = self.significance_test(metric)
        except ValueError:
            return {}

        annotations: dict[str, str] = {}
        pvalues = [
            float(matrix.loc[best, other])
            for other in matrix.index
            if other != best and not math.isnan(float(matrix.loc[best, other]))
        ]
        if any(p < 0.01 for p in pvalues):
            annotations[best] = "**"
        elif any(p < 0.05 for p in pvalues):
            annotations[best] = "*"
        return annotations

    def table(self, show_significance: bool = False) -> str:
        """Return the results as a text table.

        The table is drawn with Unicode box-drawing characters and needs no optional
        dependency.

        Args:
            show_significance: If `True`, mark the best model with `*` or `**` when a
                Wilcoxon signed-rank test against another model gives `p < 0.05` or
                `p < 0.01`. The test needs at least `6` seeds shared by both models, so
                with fewer seeds no model is marked, and a warning explains why.
                The test needs pandas, from the `benchmark` extra.

        Returns:
            A string with one row per model and one column per metric. Each value is
                shown as `mean ± std`.

        Raises:
            ImportError: If `show_significance` is `True` and pandas is not installed.
        """
        if not self.raw:
            return "(no results)"

        summary = self._summarize()
        models = self._ordered_models()
        metrics = self._ordered_metrics()

        headers = ["Model"] + [_METRIC_TITLES.get(m, m) for m in metrics]
        rows: list[list[str]] = []
        annotations: dict[str, dict[str, str]] = {}
        if show_significance:
            annotations = {
                metric: self._significance_annotations(metric) for metric in metrics
            }

        for model in models:
            row = [model]
            for metric in metrics:
                cell = summary.get(model, {}).get(metric)
                if cell is None:
                    row.append("—")
                    continue
                marker = annotations.get(metric, {}).get(model, "")
                row.append(f"{_format_cell(metric, *cell)}{marker}")
            rows.append(row)

        widths = [
            max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
        ]

        def hline(left: str, mid: str, right: str) -> str:
            return left + mid.join("─" * (w + 2) for w in widths) + right

        def row_line(cells: list[str]) -> str:
            parts = [f" {c.ljust(widths[i])} " for i, c in enumerate(cells)]
            return "│" + "│".join(parts) + "│"

        lines = [
            hline("┌", "┬", "┐"),
            row_line(headers),
            hline("├", "┼", "┤"),
            *(row_line(r) for r in rows),
            hline("└", "┴", "┘"),
        ]
        return "\n".join(lines)

    def plot(self, save_path: str | None = None, theme: str = "light"):
        """Plot a grouped bar chart with error bars.

        Args:
            save_path: If given, the figure is saved to this path instead of being
                shown.
            theme: The plot theme, `"light"` (the default) or `"dark"`.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        rc_context, plt = _load_matplotlib("plot")

        if not self.raw:
            raise ValueError("No results to plot.")

        summary = self._summarize()
        models = self._ordered_models()
        metrics = [m for m in self._ordered_metrics() if m not in _TIME_METRICS] or (
            self._ordered_metrics()
        )

        n_metrics = len(metrics)
        n_models = len(models)
        width = 0.8 / max(n_models, 1)
        x = list(range(n_metrics))

        normalized_theme = _normalize_theme(theme)
        error_color = "white" if normalized_theme == "dark" else "black"
        with rc_context(rc_params(theme=normalized_theme)):
            fig, ax = plt.subplots(figsize=(max(6, 1.4 * n_metrics * n_models), 4))
            for i, model in enumerate(models):
                # NaN, not 0.0, for a metric this model has no score for --
                # matplotlib draws no bar for NaN, where a zero draws a real one.
                # A model without `predict_proba` would appear in the figure with
                # ROC-AUC = 0, which reads as a terrible classifier rather than
                # a metric that was never measured. `table()` already renders
                # these as a dash.
                means = [
                    summary.get(model, {}).get(m, (float("nan"), 0.0))[0] for m in metrics
                ]
                stds = [
                    summary.get(model, {}).get(m, (float("nan"), 0.0))[1] for m in metrics
                ]
                offsets = [xi + (i - (n_models - 1) / 2) * width for xi in x]
                ax.bar(
                    offsets,
                    means,
                    width,
                    yerr=stds,
                    capsize=3,
                    label=model,
                    ecolor=error_color,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([_METRIC_TITLES.get(m, m) for m in metrics])
            ax.set_ylabel("Score")
            ax.set_title("Benchmark comparison")
            ax.legend(loc="best")
            fig.tight_layout()

            if save_path is not None:
                fig.savefig(save_path)
                plt.close(fig)
            else:
                plt.show()

    def plot_history(
        self,
        save_path: str | None = None,
        theme: str = "light",
        seed: str | int = "last",
    ):
        """Plot the stored training and validation loss curves.

        Models without a loss history are left out, and a warning names them.

        Args:
            save_path: If given, the figure is saved to this path instead of being
                shown.
            theme: The plot theme, `"light"` (the default) or `"dark"`.
            seed: The run to draw: `"last"` (the default), or an integer index into the
                list of seeds. With `"last"`, a model whose last seed failed shows its
                most recent recorded curve.

        Raises:
            TypeError: If `seed` is neither `"last"` nor an `int`.
            ValueError: If no model has a training or validation loss curve.
            ImportError: If matplotlib is not installed.
        """
        if seed != "last" and not isinstance(seed, int):
            raise TypeError(f"seed must be 'last' or an int, got {seed!r}.")

        rc_context, plt = _load_matplotlib("plot_history")

        normalized_theme = _normalize_theme(theme)
        plotted = False
        skipped: list[str] = []
        with rc_context(rc_params(theme=normalized_theme)):
            fig, ax = plt.subplots(figsize=(7, 4))
            for model_idx, (model, curves) in enumerate(self.loss_curves.items()):
                if not curves:
                    skipped.append(model)
                    continue
                if seed == "last":
                    # A failed seed holds a `None` slot in the list, so "last"
                    # steps back to the newest curve that exists rather than
                    # reporting no history because the final seed happened to
                    # fail.
                    curve = next((c for c in reversed(curves) if c is not None), None)
                else:
                    curve = curves[seed]
                if curve is None:
                    skipped.append(model)
                    continue
                color = STYLE_COLORS[model_idx % len(STYLE_COLORS)]
                model_plotted = False
                for key, linestyle in (("train", "-"), ("validation", "--")):
                    values = curve.get(key, [])
                    if values:
                        ax.plot(
                            range(1, len(values) + 1),
                            values,
                            color=color,
                            linestyle=linestyle,
                            label=f"{model} {key}",
                        )
                        plotted = True
                        model_plotted = True
                if not model_plotted:
                    skipped.append(model)
            if skipped:
                warnings.warn(
                    f"Learning history is unavailable for: {', '.join(skipped)}.",
                    UserWarning,
                    stacklevel=2,
                )
            if not plotted:
                plt.close(fig)
                raise ValueError("No train or validation loss curves available to plot.")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training and validation loss")
            ax.legend(loc="best")
            fig.tight_layout()
            if save_path is not None:
                fig.savefig(save_path)
                plt.close(fig)
            else:
                plt.show()

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the raw results of every seed as a pandas DataFrame.

        Returns:
            A DataFrame with the columns `model`, `seed`, `metric` and `value`, and
                one row for each entry of `raw`.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                f"pandas is required for to_dataframe(). {_EXTRAS_HINT}"
            ) from exc
        return pd.DataFrame(self.raw)

    def save(self, path: str | Path) -> Path:
        """Save the numeric results of the benchmark to a JSON file.

        The fitted estimators are not saved, by design.

        Args:
            path: The file to write. If it is an existing directory, a file named
                `bench_<timestamp>.json` is created inside it. If it has no extension,
                `.json` is added.

        Returns:
            The path of the file that was written.
        """
        path = Path(path)
        if path.exists() and path.is_dir():
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            path = path / f"bench_{stamp}.json"
        elif path.suffix == "":
            path = path.with_suffix(".json")

        payload = {
            "raw": _json_safe(self.raw),
            "summary": _json_safe(self._summarize()),
            "predictions": _json_safe(self.predictions),
            "model_artifacts": _json_safe(self.model_artifacts),
            "loss_curves": _json_safe(self.loss_curves),
            "metric_directions": _json_safe(self.metric_directions),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkResult":
        """Load numeric benchmark results saved by `BenchmarkResult.save()`.

        Args:
            path: The JSON file to read.

        Returns:
            A `BenchmarkResult` with the stored values. It has no fitted models,
                because `save()` does not store them.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            raw=data.get("raw", []),
            predictions=_restore_prediction_arrays(data.get("predictions", {})),
            model_artifacts=data.get("model_artifacts", {}),
            loss_curves=data.get("loss_curves", {}),
            metric_directions=data.get("metric_directions", {}),
        )

    def significance_test(self, metric: str, method: str = "wilcoxon") -> "pd.DataFrame":
        """Return a matrix of p-values for each pair of models.

        The p-values compare the values of `metric` on the seeds that both models
        share.

        Args:
            metric: The metric to compare.
            method: `"wilcoxon"` (the default) for a Wilcoxon signed-rank test, or
                `"ttest"` for a paired t-test.

        Returns:
            A pandas DataFrame with one row and one column per model. The diagonal is
                `1`, and a pair with fewer than two shared seeds has the value `NaN`.

        Raises:
            ImportError: If pandas or scipy is not installed.
            ValueError: If `method` is not supported, or if fewer than two models have
                values for `metric`.
        """
        try:
            import pandas as pd
            from scipy import stats
        except ImportError as exc:
            raise ImportError(
                f"pandas and scipy are required for significance_test(). {_EXTRAS_HINT}"
            ) from exc

        if method not in {"wilcoxon", "ttest"}:
            raise ValueError("method must be 'wilcoxon' or 'ttest'.")

        models = [
            m for m in self._ordered_models() if self._metric_values_by_seed(m, metric)
        ]
        if len(models) < 2:
            raise ValueError(f"Need at least two models with metric {metric!r}.")

        matrix = pd.DataFrame(
            np.ones((len(models), len(models))), index=models, columns=models
        )
        for i, left in enumerate(models):
            left_values = self._metric_values_by_seed(left, metric)
            for right in models[i + 1 :]:
                right_values = self._metric_values_by_seed(right, metric)
                common = sorted(set(left_values) & set(right_values))
                if len(common) < 2:
                    pvalue = math.nan
                else:
                    a = np.asarray([left_values[s] for s in common], dtype=float)
                    b = np.asarray([right_values[s] for s in common], dtype=float)
                    if np.allclose(a, b):
                        pvalue = 1.0
                    elif method == "wilcoxon":
                        pvalue = float(stats.wilcoxon(a, b).pvalue)
                    else:
                        pvalue = float(stats.ttest_rel(a, b).pvalue)
                matrix.loc[left, right] = pvalue
                matrix.loc[right, left] = pvalue
        return matrix

    def confusion_matrix(self, model_name: str | None = None, **kwargs) -> np.ndarray:
        """Return a scikit-learn confusion matrix computed from the stored predictions.

        Args:
            model_name: The model whose predictions are used. `None`, the default,
                uses the first model with stored predictions.
            **kwargs (Any): Passed on to `sklearn.metrics.confusion_matrix`.

        Returns:
            The confusion matrix, as a NumPy array.

        Raises:
            ValueError: If no predictions are stored (run
                `compare(..., store_predictions=True)`), or if there are none for
                `model_name`.
            ImportError: If scikit-learn is not installed.
        """
        try:
            from sklearn.metrics import confusion_matrix
        except ImportError as exc:
            raise ImportError(
                f"scikit-learn is required for confusion_matrix(). {_EXTRAS_HINT}"
            ) from exc

        if not self.predictions:
            raise ValueError(
                "No predictions stored. Run compare(..., store_predictions=True)."
            )
        if model_name is None:
            model_name = next(iter(self.predictions))
        if model_name not in self.predictions:
            raise ValueError(f"No predictions stored for model {model_name!r}.")

        pred = self.predictions[model_name]
        return confusion_matrix(pred["y_true"], pred["y_pred"], **kwargs)
