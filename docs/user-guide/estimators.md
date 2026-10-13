# Estimators

eigenkit includes quantum estimators that are compatible with sklearn and built on
PyTorch. They are hybrid classifiers and regressors, their recurrent versions,
wrappers around your own networks, and adapters for time series. Every estimator
follows the standard sklearn interface, `fit(X, y)`, `predict(X)` and `score(X, y)`,
and works with `GridSearchCV`, `Pipeline`, `clone` and the rest of sklearn. Import
them from the top level: `from eigenkit import HQNC`.

**Install:** these estimators need the `torch` extra
(`pip install eigenkit[torch]`). They run on the built-in NumPy simulator. Another
backend needs its own extra, for example `eigenkit[pennylane]` for
`provider="pennylane"`.

You can save a fitted model to a **`.eigenkit`** bundle, which is a zip file, and
load it again with `save()` and `load()`, or with `load_model()`. See
[Model persistence](model-persistence.md) for the format of the bundle, the
supported classes, and how the benchmark uses `save_dir`.

!!! note "How to get the trained network: `model_`"

    `fit()` trains a deep copy of the network, so it never changes the module that
    you gave to the estimator. After training, the object that you passed to
    `GenericClassifier(model=...)` therefore still has its initial weights, whatever
    the network did. This comes from the copy, and says nothing about the model.
    `model_` returns the network that was actually trained. It exists on every
    estimator that trains a PyTorch network (all the estimators of this page except
    the time-series wrappers), and it raises `NotFittedError` before `fit()`.

    ```python
    from torch import nn
    from eigenkit import Basic, GenericClassifier
    from eigenkit.datasets import load_dataset

    data = load_dataset("iris")
    X_train, y_train = data["X_train"], data["y_train"]

    layer = Basic(n_qubits=4, depth=2)
    clf = GenericClassifier(model=nn.Sequential(layer, nn.Linear(4, 3)), in_features=4)
    clf.fit(X_train, y_train)

    layer.state_dict()  # unchanged, this object never trained
    clf.model_[0].state_dict()  # the trained weights
    ```

## Classifiers

`HQNC` uses the PyTorch loss `CrossEntropyLoss`, and `train_losses_` stores the
cross-entropy of each epoch. When you pass validation data with
`fit(..., X_val=..., y_val=...)`, `validation_losses_` stores the matching
validation curve. `n_classes` is optional on every classifier of this section. If
you leave it out, the set of labels is inferred from `y` when you call `fit()`.

The examples of this section use the built-in `iris` dataset. It has `4` features
and `3` classes, which is what the constructors below are sized for:

```python
from eigenkit.datasets import load_dataset

data = load_dataset("iris")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
```

### HQNC

`HQNC` is the hybrid quantum neural network classifier. It places a quantum layer
in a small `nn.Sequential` pipeline, and trains the whole pipeline together with
the cross-entropy loss.

```python
from eigenkit import HQNC

model = HQNC(
    in_features=4,
    depth=2,
    n_classes=3,
    epochs=30,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.score(X_test, y_test))  # accuracy
print(model.train_losses_)  # per-epoch loss
```

You choose the quantum layer with `layer_type`:

| `layer_type` | Quantum layer | Notes |
|---|---|---|
| `"Basic"` (default) | One encoding block and one variational block. | It needs `n_qubits == in_quantum`. |
| `"Reupload"` | Standard data reuploading. | It needs `n_qubits == in_quantum`. |
| `"Ensemble"` | `N` independent circuits. | It needs `n_qubits` to divide `in_quantum`. |

Note: if `n_qubits` and `in_quantum` differ for `Basic` or `Reupload`, a linear
adapter `Linear(in_features, n_qubits)` is added in front automatically, and a
`UserWarning` is raised.

`enc_rotation`, `var_rotation`, `variational_type` and `basis` are passed to that
layer unchanged. Their default is `None`, which means "the value that the layer you
named uses". So the architecture that an estimator builds is the one that the
[layer page](layers.md) describes, and if you build it yourself from that page, you
get the same model. Pass any of these four arguments to replace the default of the
layer.

```python
# Ensemble of 4-qubit circuits over 16 features.
model = HQNC(in_features=16, n_qubits=4, layer_type="Ensemble", depth=2)
```

### RecurrentHQNC

`RecurrentHQNC` is a hybrid classifier for sequences. An encoder that is a
recurrent neural network (RNN), of type `"rnn"`, `"gru"` or `"lstm"`, reads the
sequence and passes the encoded vector to a quantum layer like the one of `HQNC`.
The whole model is trained together, with one optimiser and one loss, so the RNN
and the quantum block are updated at the same time.

`X` must be 3-D, with shape `(n_samples, seq_len, seq_feature_dim)`. By default,
only the last time step of the RNN goes to the quantum block. To flatten the whole
sequence instead, set `rnn_return_last_timestep=False` and also give
`sequence_length`.

```python
import numpy as np
from eigenkit import RecurrentHQNC

X = np.random.randn(32, 10, 4).astype(np.float32)
y = np.where(X[:, -1, 0] > 0, "up", "down")

model = RecurrentHQNC(
    seq_feature_dim=4,
    rnn_type="gru",
    rnn_hidden_dim=16,
    depth=2,
    epochs=30,
)
model.fit(X, y)
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

The quantum layer takes the same arguments as in `HQNC`: `layer_type`,
`n_qubits`, `enc_rotation`, `var_rotation`, `variational_type` and `basis`. See the [table of `layer_type` values of `HQNC`](#hqnc) above. The
other options of the RNN are `rnn_num_layers`, `rnn_dropout` and
`rnn_bidirectional`.

### CustomQuantumClassifier

`CustomQuantumClassifier` builds its architecture from any list of blocks. Use it
when none of the `layer_type` presets fits:

```python
from eigenkit import (
    Encoding,
    Variational,
    StrongVariational,
    all_z,
)
from eigenkit import CustomQuantumClassifier

blocks = [
    Encoding(in_features=4, rotation="RY"),
    StrongVariational(depth=2),
    Variational(depth=1, rotation="RX"),
]
model = CustomQuantumClassifier(
    blocks=blocks,
    observables=all_z(4),
    n_classes=2,
    epochs=30,
)

# This estimator is configured for two classes, so keep iris labels 0 and 1.
binary = y_train < 2
model.fit(X_train[binary], y_train[binary])
predictions = model.predict(X_test)
```

`observables=` accepts the same forms as `Circuit(observables=...)` (see
[Observables](observables.md)). When the number of features of the data differs
from the input size of the circuit, an `nn.Linear` adapter is added in front
automatically. The input size of the circuit is inferred from the first `Encoding`
or `Reuploading` block.

## Regressors

The examples of this section use the built-in `concrete` dataset. It has `8`
features and a continuous target. The examples use only part of the data, so that
they run quickly:

```python
from eigenkit.datasets import load_dataset

data = load_dataset("concrete")
X_train, y_train = data["X_train"][:200], data["y_train"][:200]
X_test, y_test = data["X_test"][:50], data["y_test"][:50]
```

### HQNR

`HQNR` is the regression version of `HQNC`. Its output is a single neuron, trained
with the regression loss that you choose.

```python
from eigenkit import HQNR

model = HQNR(in_features=8, depth=2, epochs=50)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.score(X_test, y_test))  # R^2
```

It takes the same `layer_type` and `n_qubits` arguments as `HQNC`.
You choose the loss with `loss_func`: `"mse"`, `"mae"` or `"huber"`.

### RecurrentHQNR

`RecurrentHQNR` is a hybrid regressor for sequences. An RNN encoder, of type
`"rnn"`, `"gru"` or `"lstm"`, reads the sequence and passes the encoded vector to a
quantum layer like the one of `HQNR`. The whole model is trained together, with one
optimiser and one loss, so the RNN and the quantum block are updated at the same
time.

`X` must be 3-D, with shape `(n_samples, seq_len, seq_feature_dim)`. By default,
only the last time step of the RNN goes to the quantum block. To flatten the whole
sequence instead, set `rnn_return_last_timestep=False` and also give
`sequence_length`.

```python
import numpy as np
from eigenkit import RecurrentHQNR

X = np.random.randn(32, 10, 4).astype(np.float32)
last = X[:, -1, :]
y = (
    0.5 * last[:, 0]
    + 0.3 * np.sin(last[:, 1])
    + 0.2 * np.cos(2 * np.pi * last[:, 2])
    + 0.1 * last[:, 3] * last[:, 0]
).astype(np.float32)

model = RecurrentHQNR(
    seq_feature_dim=4,
    rnn_type="gru",
    rnn_hidden_dim=16,
    depth=2,
    epochs=30,
)
model.fit(X, y)
predictions = model.predict(X)
```

### CustomQuantumRegressor

`CustomQuantumRegressor` is the regression version of `CustomQuantumClassifier`. It
builds the circuit from a list of blocks, and adds a `Linear` head with a single
neuron, trained with the regression loss that you choose.

## Generic estimators

`GenericClassifier` and `GenericRegressor` are wrappers, compatible with sklearn, that train **any** `torch.nn.Module` through the standard `fit` and `predict` interface. `HQNC` and `HQNR` build the architecture of the model for you. The generic estimators instead expect a complete network, output head included, so you have full control over the architecture.

!!! warning "Pre-trained weights and the behaviour of `fit()`"
    `fit()` does not start from pre-trained weights. It copies the `nn.Module`, and then calls `reset_parameters()` on every submodule that has this method (such as `nn.Linear` and the quantum layers of eigenkit), seeded with `random_state`. So the weights of these layers are drawn again before training, even if the module that you pass already holds trained weights. Each later call to `fit()` (for example inside cross-validation, or after `clone()`) also starts again from new weights, not from the weights of the previous `fit()`. To keep the trained weights, use the fitted estimator, or save it with `save()` and load it again with `load()`, and do not call `fit()` on it again.

!!! note "`random_state` does not reseed your process"

    Every estimator applies `random_state` inside `fit()` and `load()` on a private
    copy of the state of the PyTorch random number generators, that is the generator
    of the central processing unit (CPU) and the generator of every CUDA device. Two
    runs with the same `random_state` give the same model, on the CPU and on the
    graphics processing unit (GPU), and neither run changes a seed that you set
    yourself.

Only `model` is required. If you leave out `in_features` (default `None`), the
number of features is taken from the data. `sequence_input` (default `False`) tells
the estimator that `X` has shape `(batch, timesteps, features)` instead of
`(batch, features)`. The training arguments are `lr` (`0.01`), `optimizer`
(`"adam"`), `optimizer_kwargs` (`None`), `epochs` (`50`), `batch_size` (`32`),
`show_progress` (`True`) and `random_state` (`42`). `GenericClassifier` adds
`n_classes`, and `GenericRegressor` adds `loss_func`, `huber_delta` and
`huber_reduction`. None of these arguments is about the circuit, because the network
is yours.

The examples below use `moons` for the classifier and `diabetes` for the regressor,
which match the input sizes of their networks:

```python
from eigenkit.datasets import load_dataset

moons = load_dataset("moons")
X_train, y_train = moons["X_train"], moons["y_train"]
X_test, y_test = moons["X_test"], moons["y_test"]

diabetes = load_dataset("diabetes")
X_reg, y_reg = diabetes["X_train"][:200], diabetes["y_train"][:200]
X_reg_test, y_reg_test = diabetes["X_test"][:50], diabetes["y_test"][:50]
```

### GenericClassifier

```python
from torch import nn
from eigenkit import GenericClassifier
from eigenkit import Basic

N_QUBITS, N_CLASSES = 4, 2

clf_net = nn.Sequential(
    nn.Linear(2, N_QUBITS),
    nn.Tanh(),
    Basic(n_qubits=N_QUBITS, depth=2),
    nn.Linear(N_QUBITS, N_CLASSES),
)

clf = GenericClassifier(
    model=clf_net,
    in_features=2,
    epochs=30,
    lr=0.01,
    batch_size=16,
    random_state=42,
)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))  # accuracy
labels = clf.predict(X_test)
probs = clf.predict_proba(X_test)  # shape (n_samples, n_classes)
```

The `forward()` method of the network must return logits of shape `(batch, n_classes)`. The wrapper encodes the labels automatically, so `y` can hold integers, strings or any other hashable labels.

`in_features` is optional but recommended. When it is set, `fit` raises an error at once if the shape of the data does not match.

### GenericRegressor

```python
from torch import nn
from eigenkit import GenericRegressor
from eigenkit import Basic

N_QUBITS_REG, IN_FEATURES = 6, 10

reg_net = nn.Sequential(
    nn.Linear(IN_FEATURES, N_QUBITS_REG),
    nn.Tanh(),
    Basic(n_qubits=N_QUBITS_REG, depth=2),
    nn.Linear(N_QUBITS_REG, 1),
)

reg = GenericRegressor(
    model=reg_net,
    in_features=IN_FEATURES,
    loss_func="mse",
    epochs=50,
    lr=0.01,
    batch_size=32,
    random_state=42,
)
reg.fit(X_reg, y_reg)
print(reg.score(X_reg_test, y_reg_test))  # R²
preds = reg.predict(X_reg_test)  # float32 array, shape (n_samples,)
```

The `forward()` method of the network must return shape `(batch,)` or `(batch, 1)`. The losses available by name are `"mse"`, `"mae"` and `"huber"`. Any callable `(output, target) -> scalar_tensor` is also accepted, but a model trained with a callable loss cannot be saved.

### Save and load

Generic estimators use the same `.eigenkit` bundle format as the other estimators, with one difference: the bundle does not store the architecture. You must pass a module with the **same `nn.Module` structure** to `.load()`:

```python
clf.save("clf_hybrid.eigenkit")

clf_net_fresh = nn.Sequential(
    nn.Linear(2, N_QUBITS),
    nn.Tanh(),
    Basic(n_qubits=N_QUBITS, depth=2),
    nn.Linear(N_QUBITS, N_CLASSES),
)

clf_loaded = GenericClassifier.load("clf_hybrid.eigenkit", model=clf_net_fresh)
```

`load_model()` also works for generic estimators, but the `model=` argument is required, because the bundle does not store the architecture and it cannot be inferred. If you call either method without `model=`, it raises `SerializationError`.

### sklearn compatibility

`GenericClassifier` inherits from `ClassifierMixin` and `GenericRegressor` from `RegressorMixin`, so both work with `Pipeline`, `cross_val_score`, `GridSearchCV` and `clone`:

```python
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
cloned = clone(clf)  # unfitted copy with the same hyperparameters
```

See the [Generic Estimators example](../examples/generic-estimators.md) for a complete example.

## Time series

`QuantumTimeSeriesRegressor` and `QuantumTimeSeriesClassifier` wrap an inner
estimator and apply a sliding window to data in time order. Each row of `X` is one
time step. With a window of length `in_size`, the inner model receives a 3-D batch
of shape `(n_windows, in_size, n_features)`, and the window that starts at row `i`
predicts `y[i + in_size - 1 + shift]`. With `autoregressive=True`, an extra channel
is added to each time step, and it holds the target value at that step.

For tabular estimators (`HQNR`, `CustomQuantumRegressor`, and so on), the wrapper
flattens each window before it calls `fit` or `predict`. For `RecurrentHQNR` and
`RecurrentHQNC`, and for the generic estimators with `sequence_input=True`, the
windows stay 3-D, so that the RNN sees the time axis.

```python
import numpy as np
from eigenkit import (
    QuantumTimeSeriesRegressor,
    RecurrentHQNR,
)

n_feat = 5
window = 10
X = np.random.randn(200, n_feat).astype(np.float32)
y = X[:, 0] + 0.1 * np.random.randn(200).astype(np.float32)

inner = RecurrentHQNR(
    seq_feature_dim=n_feat,
    rnn_type="gru",
    rnn_hidden_dim=16,
    depth=2,
)
model = QuantumTimeSeriesRegressor(estimator=inner, in_size=window, shift=1)
model.fit(X, y)
predictions = model.predict(X)  # length n_timesteps - in_size - shift + 1
```

Classification uses the same alignment of the windows:

```python
import numpy as np
from eigenkit import (
    QuantumTimeSeriesClassifier,
    RecurrentHQNC,
)

n_feat = 5
window = 10
X = np.random.randn(200, n_feat).astype(np.float32)
y = np.where(X[:, 0] > 0, "up", "down")

inner = RecurrentHQNC(
    seq_feature_dim=n_feat,
    rnn_type="gru",
    rnn_hidden_dim=16,
    depth=2,
)
model = QuantumTimeSeriesClassifier(estimator=inner, in_size=window, shift=1)
model.fit(X, y)
predictions = model.predict(X)  # length n_timesteps - in_size - shift + 1
probabilities = model.predict_proba(X)
```


**Autoregressive mode.** Set `autoregressive=True` to add the past values of the
target to each window, as an extra channel of features. This extra channel counts
as a feature: the inner estimator must be built with
`seq_feature_dim = n_feat + 1`, and if you reuse the one above, you get
`ValueError: Last dimension of X must equal seq_feature_dim=5, got 6`. When you
predict, you must pass the known history as `y_history`:

```python
# The appended target channel makes it n_feat + 1 inputs, so this needs its
# own inner estimator.
inner_ar = RecurrentHQNC(
    seq_feature_dim=n_feat + 1,
    rnn_type="gru",
    rnn_hidden_dim=16,
    depth=2,
)

# Fit on the head of the series, predict on the tail.
X_hist, y_hist = X[:150], y[:150]
X_future, y_future = X[150:], y[150:]

model_ar = QuantumTimeSeriesClassifier(
    estimator=inner_ar,
    in_size=window,
    shift=1,
    autoregressive=True,
)
model_ar.fit(X_hist, y_hist)
predictions = model_ar.predict(X_future, y_history=y_future)
probabilities = model_ar.predict_proba(X_future, y_history=y_future)
```

## sklearn compatibility

Every algorithm class is a subclass of `sklearn.base.BaseEstimator`, and stores
each argument of its constructor as an attribute of the same name on `self`, as
sklearn expects. This is what makes `clone`, `Pipeline`, `cross_val_score` and
`GridSearchCV` work without extra code.

```python
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

cloned = clone(model)
scores = cross_val_score(model, X, y, cv=5)
```

```python
from sklearn.model_selection import GridSearchCV
from eigenkit import HQNC
from eigenkit.datasets import load_dataset

iris = load_dataset("iris")
binary = iris["y_train"] < 2  # the estimator below declares two classes
X_iris, y_iris = iris["X_train"][binary], iris["y_train"][binary]

grid = GridSearchCV(
    HQNC(in_features=4, n_classes=2),
    param_grid={"depth": [1, 2], "epochs": [20, 40]},
    cv=3,
)
grid.fit(X_iris, y_iris)
best = grid.best_estimator_
```

`BaseQuantumClassifier` inherits from `ClassifierMixin`, so its `score()` returns
the accuracy. `BaseQuantumRegressor` inherits from `RegressorMixin`, so its
`score()` returns `R²`.

Import the base classes when you build your own estimators:

```python
from eigenkit import (
    BaseQuantumClassifier,
    BaseQuantumEstimator,
    BaseQuantumRegressor,
)
```

## Choosing a backend

Every estimator accepts the same `provider=` argument as `Circuit`: the name of a
registered backend as a string, a configuration dataclass, or a `Backend` instance.

```python
from eigenkit import PennyLaneConfig

# String. Requires the matching extra, here `qiskit`.
HQNC(in_features=4, depth=2, provider="qiskit")

# Config: forwards device-level options. Requires the `pennylane` extra.
HQNC(
    in_features=4,
    depth=2,
    provider=PennyLaneConfig(device_name="lightning.qubit", shots=None),
)
```

## API reference

::: eigenkit.qml.BaseQuantumEstimator

::: eigenkit.qml.BaseQuantumClassifier

::: eigenkit.qml.BaseQuantumRegressor

::: eigenkit.torch.estimators
