# Benchmark

`eigenkit.benchmark` compares quantum and classical models. A single call to
`compare()` trains several estimators on one dataset with several random seeds,
and returns a `BenchmarkResult`. The result can show the scores as a formatted
table, plot them as grouped bar charts, and export them as a pandas DataFrame.

The benchmark needs the `benchmark` extra, which installs matplotlib, pandas and
scikit-learn: `pip install eigenkit[benchmark]`. The quantum models that you
compare need their own extras: `qml` for `QSVC` and `QSVR`, and `torch` for
`HQNC`, `HQNR` and the other differentiable estimators.

By default, named datasets are split into `70 %` for training, `15 %` for
validation and `15 %` for testing. The metrics are computed on the test split.
The validation split is passed only to estimators whose `fit()` method accepts
`X_val` and `y_val`.

`compare()` repeats the split into training, validation and test sets for each
seed. It does not use k-fold cross-validation. For k-fold cross-validation, call
scikit-learn directly, for example with `cross_val_score` or
`GridSearchCV(cv=...)`.

```python
from eigenkit import QSVC, HQNC
from eigenkit.benchmark import compare
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

result = compare(
    models=[
        QSVC(feature_map="Angle", reps=2),
        HQNC(
            in_features=4,
            depth=1,
            epochs=3,
            show_progress=False,
        ),
        SVC(C=0.1),
        RandomForestClassifier(n_estimators=20),
    ],
    dataset="iris",
    metrics=["accuracy", "f1"],
    n_seeds=3,
    validation_size=0.15,
    test_size=0.15,
)
print(result.table())
result.plot(save_path="comparison.png")
result.plot_history(save_path="losses.png")
```

The run used for this guide produced the following table:

```text
┌────────────────────────┬───────────────┬───────────────┐
│ Model                  │ Accuracy      │ F1            │
├────────────────────────┼───────────────┼───────────────┤
│ QSVC                   │ 0.989 ± 0.019 │ 0.989 ± 0.019 │
│ HQNC                   │ 0.389 ± 0.096 │ 0.242 ± 0.131 │
│ SVC                    │ 0.911 ± 0.051 │ 0.909 ± 0.053 │
│ RandomForestClassifier │ 0.967 ± 0.033 │ 0.966 ± 0.034 │
└────────────────────────┴───────────────┴───────────────┘
```

Each cell shows the mean and the standard deviation across seeds, as
`mean ± std`.

## Supported models

Each entry in `models` must be one of the following:

1. An **eigenkit estimator**, that is, a subclass of `BaseQuantumEstimator`,
   `BaseQuantumClassifier` or `BaseQuantumRegressor`, such as `HQNC`, `QSVC` or
   `CustomQuantumClassifier`.
2. A **built-in scikit-learn estimator**, that is, a class from the `sklearn`
   package, such as `SVC`, `RandomForestClassifier` or `LinearRegression`.

Models from other libraries are not accepted yet. This includes XGBoost,
LightGBM and custom `BaseEstimator` subclasses defined outside `sklearn`.

Every model must implement `fit(X, y)` and `predict(X)`. Internally, `compare()`
uses `sklearn.base.clone` to make a fresh copy of each model for every seed.

## `compare()`

`models` is the only required argument. Every other argument has a default
value.

| Argument | Default | Description |
|---|---|---|
| `models` | required | A list of estimator instances, or a dict that maps display names to estimators. |
| `dataset` | `"breast_cancer_wisconsin"` | A dataset name from the registry, or a dict with `X_train`, `y_train`, `X_test` and `y_test`. |
| `metrics` | `None` | The names of the metrics to compute, or `custom_metric(...)` specifications. `None` selects the default metrics for the task, listed below. |
| `n_seeds` | `5` | The number of seeds to average over. |
| `test_size` | `0.15` | The fraction of the data kept for the final test metrics. |
| `show_progress` | `False` | Whether to print progress: the current seed, the current model and the estimated time remaining. |
| `save_dir` | `None` | If set, the directory where each fitted model is saved after every seed (see below). |
| `store_predictions` | `False` | If `True`, the result keeps `y_true`, `y_pred` and, when available, `y_prob` for the last seed. |
| `validation_size` | `0.15` | The fraction of the data kept for validation. |

If you call `compare(models)` with no other argument, the benchmark uses the
`breast_cancer_wisconsin` dataset. The default `test_size` here is `0.15`, which
is different from the `0.2` that [`load_dataset`](datasets.md) uses.

Set `show_progress=True` to see the progress of the run: which model is running
and how many runs are left. By default nothing is printed, which keeps notebooks
and continuous integration (CI) logs quiet.

Pass a dict to choose the names that appear in tables, plots, progress messages
and saved file names. This is useful when you compare several instances of the
same estimator class:

```python
from sklearn.svm import SVC
from eigenkit import QSVC
from eigenkit.benchmark import compare

result = compare(
    models={
        "Quantum SVC": QSVC(feature_map="Angle", reps=1, probability=True),
        "Regularized SVC": SVC(C=0.3),
        "Flexible SVC": SVC(C=3.0),
    },
    dataset="iris",
    metrics=["accuracy"],
    n_seeds=1,
)
print(list(result.fitted_models))
```

```text
['Quantum SVC', 'Regularized SVC', 'Flexible SVC']
```

Dict keys must not have whitespace at the start or the end, and must not
contain control characters. When `save_dir` is set, the keys are also used as
prefixes of file names. In that case, they must not end with a full stop and
must not contain any of these characters: `<>:"/\|?*`. Use a different key for
each model. If the same key appears twice in a Python dict literal, Python keeps
only the last model before `compare()` receives the dict.

If you pass a dataset dict that is already split and you have a validation set,
include `X_val` and `y_val`. If they are left out, `compare()` uses only the
training and test sets that you provided.

### Saving fitted models

When `save_dir` is set, each model is written to a file after `fit()`:

| Model type | Output file |
|---|---|
| eigenkit | `{ModelName}_seed{N}.eigenkit`, written with `save()` |
| built-in scikit-learn | `{ModelName}_seed{N}.joblib`, written with `joblib.dump` |

```python
compare(
    models=[HQNC(in_features=4, depth=2), SVC()],
    dataset="iris",
    n_seeds=2,
    save_dir="runs/iris_benchmark/",
)
# runs/iris_benchmark/HQNC_seed0.eigenkit, HQNC_seed1.eigenkit, SVC_seed0.joblib, ...
```

`ModelName` is the name of the model in the benchmark. When the same class
appears more than once in a list, the names are numbered (`SVC_0`, `SVC_1`).
When `models` is a dict, `ModelName` is the dict key.

To load an eigenkit checkpoint, use `eigenkit.benchmark.load_checkpoint()` or the
estimator's `load()` method, for example `HQNC.load(...)`. To load a
scikit-learn checkpoint, use `eigenkit.benchmark.load_checkpoint()` or
`joblib.load(...)`. `.joblib` files use pickle internally, so only load them
from sources that you trust.

See [Model persistence](model-persistence.md) for more details.

### Classification metrics

The available metrics are `accuracy`, `f1` (the F1 score), `precision`,
`recall`, `roc_auc` (the area under the receiver operating characteristic
curve), `fit_time` and `predict_time`.

When `metrics` is `None`, the benchmark computes `accuracy`, `f1`, `fit_time`
and `predict_time`.

`roc_auc` needs `predict_proba`. `QSVC` with `probability=True` and `HQNC` both
provide it. For a model without `predict_proba`, requesting `roc_auc` issues a
`UserWarning`, and the metric is skipped.

### Regression metrics

The available metrics are `r2` (the coefficient of determination), `mae` (the
mean absolute error), `rmse` (the root mean squared error), `mse` (the mean
squared error), `fit_time` and `predict_time`. The type of task is taken from
the first model: a model that inherits from `RegressorMixin` makes it a
regression task.

When `metrics` is `None`, the benchmark computes `r2`, `mae`, `mse`, `fit_time`
and `predict_time`. `rmse` is not computed by default, so request it by name if
you need it.

### Custom metrics

A custom metric must be wrapped with `custom_metric(fn, *, direction,
name=None)`. The function receives `(y_true, y_pred)`. `direction` states
whether higher or lower values are better. `name` sets the column label, which
otherwise comes from the name of the function:

```python
from eigenkit.benchmark import custom_metric


def error_rate(y_true, y_pred):
    return (y_true != y_pred).mean()


def match_rate(y_true, y_pred):
    return (y_true == y_pred).mean()


metrics = [
    custom_metric(error_rate, direction="lower"),
    custom_metric(match_rate, direction="higher"),
]
```

## `BenchmarkResult`

`compare()` returns a `BenchmarkResult`. It holds the raw values for every seed,
the fitted copies of the models, and methods to display, analyse and use the
results.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `raw` | `list[dict]` | One row per value, with the keys `"model"`, `"seed"`, `"metric"` and `"value"`. |
| `fitted_models` | `dict[str, list]` | The fitted estimators for each model name, one per seed. |
| `predictions` | `dict[str, dict]` | The `y_true`, `y_pred` and `y_prob` of the last seed for each model, when `store_predictions=True`. |
| `model_artifacts` | `dict[str, list]` | The paths of the checkpoint files written to `save_dir`, grouped by model name. |
| `loss_curves` | `dict[str, list]` | The training and validation loss curves for each seed, when the fitted estimators record them. |
| `metric_directions` | `dict[str, str]` | The direction, `"higher"` or `"lower"`, of every metric in the run. |

### Methods

| Method | Description |
|---|---|
| `table(show_significance=False)` | A text table with `mean ± std` for each model and metric. It needs no optional dependency. With `show_significance=True`, the best model is marked with `*` or `**` when a test against another model gives `p < 0.05` or `p < 0.01`. |
| `plot(save_path=None, theme="light")` | A grouped bar chart with error bars. `theme` is `"light"` or `"dark"`. It needs `matplotlib`. |
| `plot_history(save_path=None, theme="light", seed="last")` | The training and validation loss curves of the models that record them. `seed` chooses the run to draw: `"last"` or an integer index. It needs `matplotlib`. |
| `to_dataframe()` | A DataFrame with one row per seed and metric. It needs `pandas`. |
| `best_model(metric=None, seed="last")` | The fitted estimator with the best mean on `metric`, taking the direction of the metric into account. |
| `predict(X, metric=None)` | A shortcut for `best_model(metric=metric).predict(X)`. |
| `inference(X, models=None, seed="last")` | The predictions and probabilities of all fitted models, or of the selected ones. |
| `significance_test(metric, method="wilcoxon")` | A matrix of p-values for each pair of models, computed from the scores of each seed. `method="ttest"` is also supported. It needs `pandas` and `scipy`. |
| `confusion_matrix(model_name=None, **kwargs)` | A scikit-learn confusion matrix computed from the stored predictions. `**kwargs` are passed to `sklearn.metrics.confusion_matrix`. It needs `store_predictions=True`. |
| `save(path)` | Writes the numeric results to a JSON file. The fitted estimators are not saved. |
| `BenchmarkResult.load(path)` | Restores a result from a JSON file written by `save()`. |

The significance markers come from a Wilcoxon signed-rank test, which needs at
least `6` seeds shared by the two models. With the default `n_seeds=5`,
`table(show_significance=True)` cannot mark any model, and it issues a warning
instead.

To use plotting, DataFrame export and significance tests, install the optional
extra:

```bash
pip install eigenkit[benchmark]
```

`plot_history()` draws the models that record a loss history, and warns about
the ones that do not. For example, `HQNC` produces training and validation
curves, while `SVC` appears only in the metric table and the bar chart.

### Accessing fitted models

Every fitted copy produced by `compare()` is stored in `result.fitted_models`, a
`dict[str, list[estimator]]` keyed by the name of the model in the benchmark.
With a list of models, the names come from the class (`SVC`, or `SVC_0` and
`SVC_1` when the same class appears more than once). With a dict, the keys are
exactly the names that you passed. Each value is a list of fitted copies, one
per seed.

```python
print(list(result.fitted_models))
# ['Quantum SVC', 'Regularized SVC', 'Flexible SVC']

quantum_fits = result.fitted_models["Quantum SVC"]
model_seed0 = quantum_fits[0]
```

#### `best_model(metric, seed)`

Returns the fitted estimator with the best mean on `metric`. When `metric` is
not given, the primary metric is chosen automatically. It is the first metric in
the results that is not a timing metric, usually `"accuracy"` for classification
and `"r2"` for regression with the default metrics.

```python
winner = result.best_model()  # auto-selects metric
winner = result.best_model(metric="f1", seed=0)  # explicit metric, seed 0
winner = result.best_model(metric="r2")  # explicit for regression
```

| Parameter | Default | Description |
|---|---|---|
| `metric` | `None` | The metric used for ranking. `None` selects the primary metric of the task. Built-in metrics and custom metrics use their registered direction. |
| `seed` | `"last"` | The seed whose copy is returned: `"last"`, or a 0-based `int`. |

#### `predict(X, metric)`

This is a shortcut for `best_model(metric=metric).predict(X)`. It returns only
the predictions of the selected model. Use `inference()` when you also need
probabilities, or the outputs of several models.

```python
preds = result.predict(X_new)  # auto-selects metric
preds = result.predict(X_new, metric="r2")  # explicit for regression
```

`X` is passed directly to the selected estimator. The return value is what that
estimator's `predict(X)` returns, usually an array of shape `(n_samples,)`.

### Inference on new data

`result.inference(X)` runs all fitted models on new data and returns their
predictions together with the class probabilities, in a single call.

```python
out = result.inference(X_new)
# {
#   "HQNC": {"predictions": array([...]), "probabilities": array([[...]])},
#   "SVC":  {"predictions": array([...]), "probabilities": None},  # no predict_proba
# }

preds = out["HQNC"]["predictions"]  # shape (n_samples,)
probs = out["HQNC"]["probabilities"]  # shape (n_samples, n_classes), or None
```

| Output key | Type | Description |
|---|---|---|
| `predictions` | `ndarray` | The output of `estimator.predict(X)`. |
| `probabilities` | `ndarray` or `None` | The output of `predict_proba(X)` for classifiers that support it, and `None` for regressors and for classifiers without `predict_proba`. |

#### Selecting models and seed

By default, `inference()` runs all models, each with the copy fitted on the last
seed. You can restrict both:

```python
# Only a subset of models
out = result.inference(X_new, models=["SVC"])

# Specific seed (0-based index into the per-seed list)
out = result.inference(X_new, seed=0)
out = result.inference(X_new, models=["HQNC", "SVC"], seed=1)
```

Passing an unknown model name raises a `ValueError` that lists the available
names.

#### Probabilities

`probabilities` is an array of shape `(n_samples, n_classes)` when the
estimator provides `predict_proba`, and `None` otherwise:

- It is filled for `HQNC`, `QSVC(probability=True)`, `CustomQuantumClassifier`
  and `SVC(probability=True)`.
- It is `None` for all regressors, for classifiers without `predict_proba` (for
  example, a plain `SVC()` without `probability=True`), and for a `QSVC` without
  `probability=True`, which also issues a warning.

## `quick_baselines()`

```python
from eigenkit import QSVC
from eigenkit.benchmark import compare, quick_baselines

results = compare(
    models=[QSVC(feature_map="Angle"), *quick_baselines()],
    dataset="iris",
    n_seeds=2,
)
```

`quick_baselines()` returns built-in scikit-learn classifiers with sensible
defaults: a `SVC` with a radial basis function (RBF) kernel and probabilities, a
`RandomForestClassifier`, a `LogisticRegression` and a `KNeighborsClassifier`.
scikit-learn 1.9 deprecates `SVC(probability=True)`, so from that version the
support vector classifier is wrapped as
`CalibratedClassifierCV(SVC(), ensemble=False)`.

## Timing a block of code

`timer()` measures the wall-clock time of the code inside a `with` block. For
example, it can measure the time needed to compute one kernel matrix on two
providers:

```python
import numpy as np

from eigenkit import QuantumKernel
from eigenkit.benchmark import timer

X = np.random.default_rng(0).uniform(0, np.pi, size=(150, 4))
for provider in ("numpy", "pennylane"):  # "pennylane" needs the pennylane extra
    kernel = QuantumKernel(n_features=4, feature_map="AngleCNOT", provider=provider)
    with timer() as t:
        kernel.compute_kernel_matrix(X)
    print(provider, f"{t['elapsed']:.2f} s")
```

## API reference

::: eigenkit.benchmark
