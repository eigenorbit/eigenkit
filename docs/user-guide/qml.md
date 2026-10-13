# QML

The `qml` package is the scikit-learn side of the library for quantum machine
learning (QML). It holds kernel support vector machines (SVMs) and quantum reservoir
estimators. Everything here follows the scikit-learn interface, with `fit(X, y)`,
`predict(X)` and `score(X, y)`, and runs without PyTorch.

**Install:** the `qml` extra (`pip install eigenkit[qml]`) installs scikit-learn, and
not PyTorch. The estimators run on the built-in NumPy simulator. To run them on
another backend, install its extra and pass `provider=`, for example
`eigenkit[pennylane]` and `provider="pennylane"`.

## QSVC

`QSVC` is the quantum support vector classifier. It passes the quantum kernel matrix
to scikit-learn's `SVC(kernel="precomputed")`.

```python
from eigenkit import QSVC
from eigenkit.datasets import load_dataset

data = load_dataset("iris")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

model = QSVC(feature_map="AngleCNOT", reps=2, C=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The accepted `feature_map` values, what `reps` does, and how the kernel value is
computed are explained in [Quantum Kernels](kernels.md). The defaults of `QSVC`,
`feature_map="Angle"` and `reps=1`, differ from those of `QuantumKernel`
(`"AngleCNOT"` and `reps=2`).

Any other kernel, such as a custom map, a trained map, or a kernel on hardware, is
passed whole with `kernel=`. `feature_map`, `reps` and `provider` are then ignored,
and the fitted copy of the kernel is `model.kernel_`. `scale` multiplies the inputs
before they are encoded, so this bandwidth can be tuned like `C`:

```python
from eigenkit import QuantumKernel

kernel = QuantumKernel(n_features=X_train.shape[1], feature_map="AngleCNOT", reps=2)
model = QSVC(kernel=kernel, scale=0.5).fit(X_train, y_train)
margins = model.decision_function(X_test)
```

`decision_function(X)` returns the margins of the SVM. `probability=True` enables
`predict_proba` through a sigmoid calibration, as in
`CalibratedClassifierCV(SVC(), ensemble=False)`. The predictions still come from the
plain `SVC`.

## QSVR

`QSVR` is the quantum support vector regressor. It works like `QSVC`, but it passes
the kernel matrix to scikit-learn's `SVR(kernel="precomputed")`.

```python
from eigenkit import QSVR
from eigenkit.datasets import load_dataset

data = load_dataset("concrete")
X_reg, y_reg = data["X_train"][:100], data["y_train"][:100]

model = QSVR(feature_map="Angle", reps=1, C=1.0, epsilon=0.1)
model.fit(X_reg, y_reg)
```

## Use inside a scikit-learn Pipeline

The kernel estimators can also be the last step of a scikit-learn `Pipeline`. When
you tune them with `GridSearchCV`, prefix each parameter with the name of the step,
including the bandwidth `scale`:

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from eigenkit import QSVC

X, y = make_moons(n_samples=40, noise=0.1, random_state=0)

search = GridSearchCV(
    Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "qsvc",
                QSVC(
                    feature_map="Angle",
                    reps=1,
                    C=1.0,
                    probability=False,
                ),
            ),
        ]
    ),
    param_grid={
        "qsvc__C": [0.1, 1.0, 10.0],
        "qsvc__feature_map": ["Angle", "AngleCNOT"],
        "qsvc__scale": [0.5, 1.0],
    },
    scoring="accuracy",
    cv=3,
)

search.fit(X, y)
predictions = search.predict(X[:5])
```

## Quantum reservoir computing

`QuantumReservoirClassifier` and `QuantumReservoirRegressor` use a quantum circuit as
a fixed dynamical system that is *not trained*, called the **reservoir**. They train
only a cheap linear **readout** on the features that the reservoir produces. The
weights of the circuit are drawn once from `random_state` and then frozen, so there is
no gradient descent and no PyTorch. Fitting solves a ridge or logistic regression in
closed form.

The data is a batch of sequences. `X` has shape `(n_sequences, n_timesteps,
n_features)`, and `y` has shape `(n_sequences, n_timesteps)`, with one target per
time step.

```python
import numpy as np
from eigenkit import Encoding, Variational
from eigenkit import QuantumReservoirRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(40, 50, 3))
y = np.zeros((40, 50))
y[:, 1:] = X[:, :-1, 0]  # predict the previous step's first channel

reservoir = QuantumReservoirRegressor(
    blocks=[Encoding(in_features=3, rotation="RY"), Variational(depth=3, rotation="RY")],
    state_policy="leaky",
    leak=0.5,  # fading memory
    features="probabilities",
    washout=5,
)
reservoir.fit(X, y)
print(reservoir.predict(X).shape)  # (40, 50)
```

`QuantumReservoirClassifier` takes the same arguments, with one exception. It accepts
integer or label targets, reads out through a logistic model, and adds
`predict_proba`. The exception is the regularisation of the readout: the `ridge` of
the regressor (`float`, default `1e-06`) becomes `C` on the classifier (`float`,
default `1.0`). `C` is the inverse of the regularisation strength, so a larger `C`
means *less* regularisation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `blocks` | `list[Block]` | Required | The frozen reservoir circuit. |
| `state_policy` | `str` | `"carry"` | One of `"carry"`, `"leaky"` or `"stateless"`. |
| `window` | `int` | `1` | The number of time steps encoded per step with `"stateless"`. The circuit must then encode `window * n_features` inputs. |
| `leak` | `float` | `1.0` | The mixing rate for `"leaky"`. At `1.0`, it is the same as `"carry"`. |
| `features` | `str \| list[str]` | `"z"` | One of `"z"`, `"probabilities"` or `"amplitudes"`, a Hamiltonian string such as `"0.5 Z0 X1 - 0.3 Y0"`, or a list of observables. |
| `ridge` | `float` | `1e-06` | The regularisation of the linear readout. |
| `washout` | `int` | `0` | The number of first time steps dropped when the readout is fitted. Predictions are still produced for every time step. |
| `weight_scale` | `float` | `1.0` | The scale of the frozen weights that are drawn. Larger values give more energetic dynamics. |
| `provider` | `ProviderSpec` | `"numpy"` | The backend, with the same options as `Circuit(provider=...)`. |
| `random_state` | `int \| None` | `42` | The seed of the frozen weights. The reservoir is random, but fixed. |

!!! note "Fading memory matters"
    A purely unitary carry (`state_policy="carry"`, the default) preserves the norm,
    but it tends to scramble the history into a form that a linear readout cannot
    recover. So it often has little memory that can be used. Some dissipation is what
    makes a reservoir work in practice: `"leaky"` with a `leak` of about `0.3` to
    `0.7` is usually the right starting point for temporal tasks. The
    [memory capacity example](../examples/reservoir-memory.md) shows this directly.

The layers under these wrappers, which are the driver, the policy classes, the
feature extractors and the readout, are documented in
[Reservoir Computing](reservoir.md).

## API reference

::: eigenkit.qml.QSVC

::: eigenkit.qml.QSVR

::: eigenkit.qml.QuantumReservoirClassifier

::: eigenkit.qml.QuantumReservoirRegressor
