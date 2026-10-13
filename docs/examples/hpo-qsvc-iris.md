# Hyperparameter search with QSVC on Iris

This example runs a small hyperparameter optimisation (HPO) for an `eigenkit` estimator. It loads a dataset, builds a sklearn pipeline, chooses a search range, runs `GridSearchCV`, and evaluates the selected model.

The grid is small on purpose, so that the example finishes quickly. Make it larger once this first quick test passes.

## Imports

The example uses tools from sklearn, and `QSVC` (the quantum support vector classifier) and the dataset loader from `eigenkit`.


```python
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eigenkit import QSVC
from eigenkit.datasets import load_dataset
```

## Load data

`load_dataset("iris")` returns a stratified split into a training set and a test set. Iris is small enough for a quick HPO run, and it is still a real classification task with 3 classes.


```python
data = load_dataset("iris", split=True, test_size=0.25, random_state=42)

X_train = np.asarray(data["X_train"], dtype=np.float32)
y_train = np.asarray(data["y_train"], dtype=np.int64)
X_test = np.asarray(data["X_test"], dtype=np.float32)
y_test = np.asarray(data["y_test"], dtype=np.int64)
metadata = data["metadata"]

labels, counts = np.unique(y_train, return_counts=True)
class_counts = {int(label): int(count) for label, count in zip(labels, counts)}

print(f"Dataset: {metadata.name}")
print(f"Train shape: {X_train.shape}, test shape: {X_test.shape}")
print("Train class counts:", class_counts)
```

```text
Dataset: iris
Train shape: (112, 4), test shape: (38, 4)
Train class counts: {0: 38, 1: 37, 2: 37}
```


## Build the pipeline and search range

`StandardScaler` gives the features comparable ranges. `GridSearchCV` tunes `QSVC` through the name of its step in the pipeline, so `C` becomes `qsvc__C`. `feature_map="Angle"` is the angle encoding without entanglement.

The first grid changes only `C`. Very small values usually underfit, and larger values should give a useful decision boundary. Once this quick run is stable, add another dimension, such as `qsvc__feature_map`.


```python
model = Pipeline(
    [
        ("scale", StandardScaler()),
        (
            "qsvc",
            QSVC(
                feature_map="Angle",
                reps=1,
                random_state=42,
            ),
        ),
    ]
)

param_grid = {
    "qsvc__C": [0.001, 0.01, 0.1, 1.0, 10.0],
}

print("Search space:", param_grid)
```

```text
Search space: {'qsvc__C': [0.001, 0.01, 0.1, 1.0, 10.0]}
```


## Run HPO

Here, `GridSearchCV` uses 3-fold cross-validation (`cv=3`) on the training split. In the output, "CV" stands for cross-validation. With `refit=True`, it trains the selected pipeline again on the whole training split.


```python
search = GridSearchCV(
    model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=1,
    refit=True,
)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print(f"Best CV accuracy: {search.best_score_:.3f}")

results = search.cv_results_
for mean, std, params in zip(
    results["mean_test_score"],
    results["std_test_score"],
    results["params"],
):
    print(f"{mean:.3f} +/- {std:.3f}  {params}")
```

```text
Best parameters: {'qsvc__C': 1.0}
Best CV accuracy: 0.946
0.616 +/- 0.223  {'qsvc__C': 0.001}
0.616 +/- 0.223  {'qsvc__C': 0.01}
0.866 +/- 0.037  {'qsvc__C': 0.1}
0.946 +/- 0.022  {'qsvc__C': 1.0}
0.946 +/- 0.022  {'qsvc__C': 10.0}
```


## Evaluate the selected model

The search does not use the held-out test split. Use it once, at the end, for a fairer check of the selected pipeline.


```python
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.3f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Predictions for the first five test samples:", y_pred[:5].tolist())
```

```text
Test accuracy: 0.947
Confusion matrix:
[[12  0  0]
 [ 0 12  1]
 [ 0  1 12]]
Predictions for the first five test samples: [0, 1, 1, 1, 0]
```
