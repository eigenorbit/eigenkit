# Model persistence (`.eigenkit`)

The supported quantum estimators can save a **fitted** model to disk, and load it
again later without training it again.

A saved model is a **`.eigenkit` bundle** of JSON, NumPy and PyTorch files. Models
that hold scikit-learn objects (`QSVC`, `QSVR`, the reservoir estimators and the
time-series estimators) add them as joblib pickles, and loading a pickle runs code.
As with `torch.load` or `joblib.load`, **load only bundles that you trust.**

```python
from eigenkit import HQNC, load_model
from eigenkit.datasets import load_dataset

data = load_dataset("iris")
X_train, y_train, X_test = data["X_train"], data["y_train"], data["X_test"]

model = HQNC(in_features=4, n_classes=3, depth=2, epochs=30)
model.fit(X_train, y_train)

# Save (adds .eigenkit if you omit the suffix)
model.save("artifacts/my_hqnc")
loaded = HQNC.load("artifacts/my_hqnc")

# Or load without naming the class
loaded = load_model("artifacts/my_hqnc")
assert (loaded.predict(X_test) == model.predict(X_test)).all()
```

---

## The layout of a `.eigenkit` bundle

A typical bundle contains these files:

| Path | Contents |
|------|----------|
| `metadata.json` | The keys `class_name`, `library_version`, `format_version`, `saved_at` and `provider`. |
| `params.json` | The hyperparameters of the constructor, in a form that JSON can store. |
| `circuit_spec.json` | The list of blocks and the observables, for `CustomQuantum*` only. |
| `state/*` | The fitted state: weights, labels, the training data of kernels and so on. |

The state files use typed extensions:

| Extension | Meaning |
|-----------|---------|
| `.pt` | A `state_dict` saved with `torch.save`, which holds the weights of the model. |
| `.npy` | A NumPy array, for example `X_train`, or the joblib bytes of `QSVC` and `QSVR`. |
| `.json` | Scalars or lists that JSON can store, for example `train_losses` or `label_classes`. |

PyTorch is imported only when a `.pt` entry is read or written. The `QSVC` and `QSVR`
bundles, which do not use PyTorch, have no `.pt` entry, so they can be saved and
loaded with the `qml` extra alone, without PyTorch.

Example listing for `HQNC`:

```
metadata.json
params.json
state/label_classes.json
state/n_classes.json
state/train_losses.json
state/validation_losses.json
state/weights.pt
```

Example for `QSVC`:

```
metadata.json
params.json
state/X_train.npy
state/svc_joblib.npy    # sklearn SVC bytes (joblib), not a .joblib file in the zip
state/calibrated_joblib.npy    # only with probability=True: the fitted calibration
```

!!! warning "`QSVC` and `QSVR` bundles include joblib bytes"
    For `QSVC` and `QSVR`, the fitted `sklearn.svm` object is serialised with joblib
    in memory, and then stored as a `uint8` array in `state/*_joblib.npy`. Joblib uses
    pickle internally, so load these bundles only from sources that you trust.

`metadata.json` includes `"format_version": 1`. `load()` refuses a bundle whose
`format_version` is missing or different, with an error that names the version.

---

## `save()` and `load()`

### Saving

```python
model.save("path/to/model")  # writes path/to/model.eigenkit
model.save("path/to/model.eigenkit")  # same
```

Call `save()` only **after** `fit()`. The method collects:

- the hyperparameters, from `get_params()`;
- the `circuit_spec`, for custom lists of blocks;
- the state specific to the algorithm, from `_collect_state()`: the weights, the
  encoders, `train_losses_`, the training matrices of kernels and so on.

### Loading

```python
# Explicit class
loaded = HQNC.load("path/to/model.eigenkit")

# Override backend at load time (does not change saved weights)
loaded = HQNC.load("path/to/model.eigenkit", provider="qiskit")
```

```python
# Class inferred from metadata.json
from eigenkit import load_model

loaded = load_model("path/to/model.eigenkit")
```

After `load()`, the attributes that `fit()` sets are restored when the bundle holds
them. This includes `train_losses_` on `HQNC`, `HQNR`, `RecurrentHQNC`,
`RecurrentHQNR` and the `Custom*` estimators.

### Supported estimators

| Class | Notes |
|-------|--------|
| `HQNC`, `HQNR` | The bundle holds the weights and the training metadata. |
| `RecurrentHQNC`, `RecurrentHQNR` | The same, and the loaded model expects 3-D input. |
| `QSVC`, `QSVR` | The bundle holds the quantum kernel, the state of the scikit-learn SVM and `X_train`. |
| `CustomQuantumClassifier`, `CustomQuantumRegressor` | The blocks in `circuit_spec` must be serialisable. |
| `QuantumTimeSeriesClassifier`, `QuantumTimeSeriesRegressor` | The bundle holds the wrapper and the state of the inner estimator. |
| `GenericClassifier`, `GenericRegressor` | You must pass the architecture yourself when you load. See below. |

A `QSVC` or `QSVR` built with `kernel=QuantumKernel(...)` stores that kernel in
`params.json`. It stores the name of a built-in map, or a custom map as its
`CircuitSpec`, together with its weights, `scale` and `psd_projection`, so a trained
kernel is loaded trained. `load(provider=...)` also builds the kernel again on that
provider.

#### GenericClassifier and GenericRegressor

The bundle stores the weights, the training metadata and, for `GenericClassifier`,
the label encoder. It does **not** store the architecture of the `nn.Module`. You
must pass the same architecture when you load:

```python
from eigenkit import GenericClassifier, load_model

# Direct class method
clf_loaded = GenericClassifier.load("clf_hybrid.eigenkit", model=clf_net_fresh)

# Or via load_model, pass model= explicitly
clf_loaded = load_model("clf_hybrid.eigenkit", model=clf_net_fresh)
```

Calling `load_model()` or `.load()` without `model=` raises a `SerializationError`.

**Not estimators:** `optimization.QAOA` and `chemistry.VQE` are solvers that return
result objects. They are not fitted scikit-learn estimators, so the `.eigenkit`
bundles do not apply to them.

**Custom circuits:** a `CustomBlock` built with `CircuitBuilder` can be saved when it
uses the supported gates and parameter expressions of the intermediate
representation. Other block objects still need `to_dict()` and `from_dict()`, or must
be built-in block types.

---

## Base classes for custom algorithms

> **Coming in a future release:** full `.eigenkit` support for saving and loading
> custom model classes (subclasses of `BaseQuantumClassifier`, `BaseQuantumRegressor`
> or `BaseQuantumEstimator`) is not available yet. This covers both `save()` and
> `load_model()` or `.load()` for estimators that you define.

---

## Credentials are never saved

Configuration fields marked as secret are left out of `.eigenkit` bundles, and of
pickles and joblib checkpoints (including the `.joblib` files that
`benchmark.compare()` writes). Today, these fields are `QiskitRuntimeConfig.token`
and `IQMConfig.token`. They are also hidden from `repr()`, so the `provider` entry in
`metadata.json` does not show them.

After a model is loaded, a secret field takes its default value (`None`), and the
backend falls back to its own way of finding credentials. Set the credential in the
environment of the process that runs the model: `QISKIT_IBM_TOKEN` for IBM Quantum,
and `IQM_TOKEN` for IQM. The worker processes
of joblib and loky receive their models as pickles, so they need the variable too.
Copies inside one process (`copy.copy`, `copy.deepcopy`, `sklearn.base.clone`, and
so `GridSearchCV`) keep the credential.

```python
from eigenkit import QSVC, QiskitRuntimeConfig

model = QSVC(
    feature_map="Angle",
    reps=1,
    provider=QiskitRuntimeConfig(
        backend_name="ibm_fez", primitive="sampler", token="..."
    ),
)
model.fit(X_train, y_train)
model.save("artifacts/qsvc")  # the token is not written

# Later, possibly on another machine: export QISKIT_IBM_TOKEN first.
loaded = QSVC.load("artifacts/qsvc")
assert loaded.provider.token is None  # resolved from the environment instead
```

With `QSVC(kernel=QuantumKernel(..., provider=config))`, the configuration belongs to
the kernel, so the same check reads `loaded.kernel.provider.token`.

The fields that decide where a provider connects (`url`, `service_kwargs`,
`provider_kwargs` and `backend_instance`) are not saved either. If a bundle holds
them anyway, they are ignored, with a warning. Otherwise, a shared bundle could send
the token from your environment to a host of its choice. To load a model onto a
custom endpoint, pass the full configuration:
`QSVC.load(path, provider=QiskitRuntimeConfig(..., service_kwargs=...))`.

---

## Benchmark integration

`benchmark.compare()` can save the fitted models when `save_dir` is set:

```python
from eigenkit import HQNC, QSVC
from eigenkit.benchmark import compare
from sklearn.svm import SVC

compare(
    models=[HQNC(in_features=4, depth=2), SVC()],
    dataset="iris",
    n_seeds=2,
    save_dir="benchmark_runs/",
)
```

The output files are named as follows:

| Model type | Filename pattern |
|------------|------------------|
| An eigenkit estimator (`BaseQuantumEstimator`) | `{ModelName}_seed{N}.eigenkit` |
| A built-in scikit-learn estimator (`sklearn.*`) | `{ModelName}_seed{N}.joblib` |

`ModelName` is the benchmark key. When two models have the same class name, they get
an index (`SVC_0`, `SVC_1`). Pass `models` as a dictionary to choose the benchmark
key:

```python
compare(
    models={
        "Quantum SVC": QSVC(feature_map="Angle", reps=1, probability=True),
        "Regularized SVC": SVC(C=0.3),
        "Flexible SVC": SVC(C=3.0),
    },
    dataset="iris",
    metrics=["accuracy"],
    n_seeds=1,
    save_dir="benchmark_runs/",
)
# benchmark_runs/Quantum SVC_seed0.eigenkit
# benchmark_runs/Regularized SVC_seed0.joblib
# benchmark_runs/Flexible SVC_seed0.joblib
```

Custom benchmark keys follow the naming rules in [Benchmark](benchmark.md).

To load a quantum model saved by a benchmark:

```python
from eigenkit import load_model

model = load_model("benchmark_runs/HQNC_seed0.eigenkit")
```

Load the scikit-learn baselines with `joblib.load("benchmark_runs/SVC_seed0.joblib")`.
`eigenkit.benchmark.load_checkpoint(...)` can load either type of file from a
benchmark run.

`.joblib` checkpoints use pickle internally, so load them only from sources that you
trust.

See [Benchmark](benchmark.md) for the metrics, the seeds and `BenchmarkResult`.

---

## API reference

::: eigenkit.qml.load_model

::: eigenkit.qml.BaseQuantumEstimator

::: eigenkit.qml._serialization.SerializationError
