# Datasets

eigenkit includes a registry of standard datasets for classification, regression
and time series, which you can use directly in quantum machine learning pipelines.

The datasets need the `datasets` extra, which installs scikit-learn:
`pip install eigenkit[datasets]`. This is enough to load a dataset and to call
`DatasetReport.summarize()`. The plots of the exploratory data analysis (EDA),
`DatasetReport.plot*`, also need matplotlib, which the `viz` extra installs. The
`benchmark` extra installs it too.

## Load a dataset

```python
from eigenkit.datasets import load_dataset

data = load_dataset("iris")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
```

To also get a validation split, pass `validation_size`:

```python
data = load_dataset("iris", test_size=0.15, validation_size=0.15)
X_val, y_val = data["X_val"], data["y_val"]
```

To get the whole dataset without a train/test split, pass `split=False`:

```python
data = load_dataset("moons", split=False)
X, y = data["X"], data["y"]
```

### `load_dataset` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | Required | The name of the dataset in the registry. It is one of the names in the table below. |
| `split` | `bool` | `True` | With `True`, the data is split into `X_train` and `X_test`. With `False`, the whole dataset is returned as `X` and `y`. |
| `test_size` | `float` | `0.2` | The fraction of the samples held out for the test split. |
| `validation_size` | `float` | `0.0` | The fraction of the samples held out for validation. Any value above `0` adds `X_val` and `y_val`. |
| `random_state` | `int \| None` | `42` | The seed of the split, and of the generator for the synthetic datasets. With `None`, the split is not seeded. It does not affect the split of the `timeseries` datasets, which always follows the time order without shuffling. |
| `cache_dir` | `str \| None` | `None` | The directory where the downloaded datasets are stored. With `None`, it is the `EIGENKIT_DATA_DIR` environment variable when it is set, and `~/.eigenkit/datasets` otherwise. |

## Available datasets

The registry holds twelve datasets. The Reference column names the loader or the
generator that produces the data.

| Dataset | Task | Description | Reference |
|---|---|---|---|
| `iris` | classification | Fisher's iris dataset. The task is to classify iris flowers into three species from `4` measurements of the flowers. | `sklearn.datasets.load_iris` |
| `wine` | classification | The wine recognition dataset. The task is to classify wines into three cultivars from `13` features of a chemical analysis. | `sklearn.datasets.load_wine` |
| `breast_cancer_wisconsin` | classification | The Wisconsin diagnostic breast cancer dataset. The task is to classify tumours as malignant or benign from `30` real-valued features. | `sklearn.datasets.load_breast_cancer` |
| `moons` | classification | Two half-circles that interleave. It is a binary benchmark whose classes cannot be separated by a straight line. | `generate_moons` |
| `circles` | classification | Two concentric circles, a large one and a small one. It is a binary benchmark. | `generate_circles` |
| `checkerboard` | classification | A checkerboard pattern. It is a binary benchmark with a very non-linear decision boundary. | `generate_checkerboard` |
| `diabetes` | regression | The task is to predict a score of disease progression from `10` features. | `sklearn.datasets.load_diabetes` |
| `california_housing` | regression | The task is to predict the median house value from `8` features. | `sklearn.datasets.fetch_california_housing` |
| `concrete` | regression | The compressive strength of concrete, in megapascals. The task is to predict it from `8` features: the amounts of cement, slag, fly ash, water, superplasticiser, coarse aggregate and fine aggregate, and the age of the concrete. | [UC Irvine Machine Learning Repository, dataset 165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), by I-Cheng Yeh, under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence, downloaded from a [hosted CSV file](https://kit.eigenorbit.com/datasets/concrete.csv) |
| `sine_wave` | timeseries | A sine wave with noise. The task is to predict the next value from a window of past values. | `generate_timeseries(kind="sine")` |
| `ar_process` | timeseries | A synthetic autoregressive process of order `1`, written AR(1). Its dependence over time is more realistic than that of a sine wave. | `generate_timeseries(kind="ar")` |
| `beijing_pm25` | timeseries | The hourly air quality in Beijing. The task is to predict the concentration of fine particulate matter (PM2.5) from the weather and the wind direction. | [UC Irvine Machine Learning Repository, dataset 381](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data), by Song Xi Chen, under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence, downloaded from a [hosted CSV file](https://kit.eigenorbit.com/datasets/beijing_pm25.csv) |

`list_datasets()` returns the same list when your code runs, with all the metadata
of each dataset:

```python
from eigenkit.datasets import list_datasets

for ds in list_datasets():
    print(ds.name, ds.task, ds.n_features, ds.n_classes)
```

## Synthetic generators

To choose the size and the seed of a synthetic dataset, call its generator function
directly:

```python
from eigenkit.datasets import (
    generate_ad_hoc,
    generate_moons,
    generate_circles,
    generate_checkerboard,
    generate_regression,
    generate_timeseries,
)

X, y = generate_moons(random_state=0)
X, y = generate_circles(random_state=42)
X, y = generate_regression(random_state=7)
X, y = generate_ad_hoc(n_samples=200, n_qubits=2, gap=0.3, random_state=0)
```

`generate_ad_hoc` builds the "ad hoc" dataset of
[Havlíček et al.](https://arxiv.org/abs/1804.11326), which is designed so that a
quantum kernel separates it. Each point in `[0, 2π)^n` is encoded with the ZZ
feature map, and its label is the sign of a `Z` parity measured after a fixed
random unitary. The points closer than `gap` to the decision boundary are dropped,
and the two classes have the same size. A kernel built from the same ZZ feature map
classifies the data almost perfectly, and classical kernels usually do not. It is a
test case for quantum kernels. It says nothing about an advantage on real data.

## Exploratory data analysis report and plots

Use `DatasetReport` to run an exploratory data analysis in one call, on a built-in
dataset or on your own data.

```python
from eigenkit.datasets import DatasetReport, load_dataset

data = load_dataset("iris", split=False)
report = DatasetReport.from_data(data, theme="light")

figures = report.plot_overview()
summary = report.summarize()
print(summary["text"])
```

For your own data:

```python
import numpy as np
from eigenkit.datasets import DatasetReport

X = np.random.randn(300, 6)
y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)

report = DatasetReport.from_data(
    X=X,
    y=y,
    feature_names=[f"f{i}" for i in range(X.shape[1])],
    task="classification",
    theme="light",
)
figures = report.plot_overview(max_features=4)
```

For a real hourly time series, try the
[Beijing PM2.5 dataset](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data)
(dataset 381 of the UC Irvine Machine Learning Repository). It is split in time
order automatically, with the first `80%` of the samples for training and the last
`20%` for testing:

```python
data = load_dataset("beijing_pm25")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
```

For a time-series target, pass a time index to get the seasonal analysis:

```python
data = load_dataset("sine_wave", split=False)
report = DatasetReport.from_data(
    data,
    task="timeseries",
    time_index=data["X"],  # datetime-like index also supported
    theme="light",
)

ts_fig = report.plot(kind="timeseries_target")
ts_summary = report.summarize()["timeseries"]
print(ts_summary["recommended_lags"])
```

With `theme="dark"`, the backgrounds of the plots and the legends are transparent,
and the text is white.

## The same style for your own figures

`DatasetReport` draws inside a matplotlib context that gives every figure of the
library the same palette, font sizes and spines. This context is public, so your
own plots can use the same style. Examples are a training curve next to an EDA
report, a figure for a paper, or the plots in the
[worked examples](../examples/circles-hqnc.md):

```python
import matplotlib
import matplotlib.pyplot as plt
from eigenkit.datasets import STYLE_COLORS, rc_params

with matplotlib.rc_context(rc_params(theme="light")):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(model.train_losses_, color=STYLE_COLORS[0], label="train loss")
    ax.plot(model.validation_losses_, color=STYLE_COLORS[1], label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
```

| Name | What it is |
|---|---|
| `rc_params(theme="light")` | The settings for one figure, as a dictionary for `matplotlib.rc_context()`. The theme is `"light"` or `"dark"`, and any other value raises a `ValueError`. |
| `STYLE_COLORS` | The palette, as a tuple of eight hexadecimal colour strings in the order of the colour cycle. `rc_params()` sets it as `axes.prop_cycle`, so index it only when a series needs a fixed colour. |
| `FONTSIZE_AXES` | `20`, the font size of the axis labels and the titles. |
| `FONTSIZE_LEGEND` | `18`, the font size of the legend entries. |
| `FONTSIZE_TICKS` | `16`, the font size of the tick labels, which is also the base `font.size`. |

`rc_params()` returns a new dictionary and never changes the global
`matplotlib.rcParams`, so the settings apply only inside the `rc_context` block.
Two calls with the same theme return equal dictionaries. This is why a figure can
be drawn again with the other theme and differ only in its background and in the
colour of its text and lines.

## API reference

::: eigenkit.datasets
