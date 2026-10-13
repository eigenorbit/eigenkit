"""A failed run must not win, and a benchmark must reproduce.

Two properties without which published numbers are corrupt.

`roc_auc_score` returns NaN on a degenerate split rather than raising, and
`min`/`max` over a NaN is order-dependent -- so a NaN score left in the ranking
would make a model "best" whenever it happened to come first in `raw`. One NaN
seed would also make the whole mean NaN, so a model scoring 0.99 on four seeds
of five would lose to one scoring 0.60. NaN scores are kept out of both.

Separately, seeding has to reach inside a `Pipeline` or meta-estimator, which
has no `random_state` of its own: `set_params(random_state=seed)` raises on
it, and an inner estimator left unseeded makes three identical `compare()`
calls return three different scores.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eigenkit.benchmark import compare
from eigenkit.benchmark._results import BenchmarkResult
from eigenkit.benchmark._runner import _clone_model


NAN = float("nan")


def _results(**per_model):
    raw = [
        {"model": name, "metric": "accuracy", "value": value, "seed": seed}
        for name, values in per_model.items()
        for seed, value in enumerate(values)
    ]
    fitted = {name: [name] * len(values) for name, values in per_model.items()}
    return BenchmarkResult(raw=raw, fitted_models=fitted)


class TestNanScoresDoNotWin:
    def test_a_model_that_is_nan_everywhere_is_not_ranked(self):
        results = _results(broken=[NAN] * 3, good=[0.9] * 3)
        assert results.best_model("accuracy") == "good"

    def test_ranking_does_not_depend_on_order_in_raw(self):
        first = _results(broken=[NAN] * 3, good=[0.9] * 3).best_model("accuracy")
        second = _results(good=[0.9] * 3, broken=[NAN] * 3).best_model("accuracy")
        assert first == second == "good"

    def test_one_nan_seed_does_not_poison_the_mean(self):
        results = _results(good=[0.99, 0.99, 0.99, 0.99, NAN], mediocre=[0.6] * 5)
        mean = results._summarize()["good"]["accuracy"][0]
        assert mean == pytest.approx(0.99)
        assert results.best_model("accuracy") == "good"

    def test_a_metric_that_is_nan_for_everyone_says_so(self):
        results = _results(a=[NAN] * 3, b=[NAN] * 3)
        with pytest.raises(ValueError, match="usable"):
            results.best_model("accuracy")


class TestSeedingReachesNestedEstimators:
    def test_a_pipeline_gets_its_inner_estimator_seeded(self):
        pipeline = Pipeline(
            [("s", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=5))]
        )
        assert _clone_model(pipeline, 123).get_params()["rf__random_state"] == 123

    def test_a_bare_estimator_is_unaffected(self):
        model = RandomForestClassifier(n_estimators=5)
        assert _clone_model(model, 123).get_params()["random_state"] == 123

    def test_compare_reproduces_with_a_pipeline(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 4))
        y = (X[:, 0] > 0).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        dataset = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "task": "classification",
        }

        def run():
            pipeline = Pipeline(
                [
                    ("s", StandardScaler()),
                    ("rf", RandomForestClassifier(n_estimators=8)),
                ]
            )
            result = compare(models={"pipe": pipeline}, dataset=dataset, n_seeds=3)
            return [e["value"] for e in result.raw if e["metric"] == "accuracy"]

        assert run() == run() == run()


class TestOneFailureDoesNotSinkTheRun:
    """A comparison is a set of independent runs.

    One model raising in `_evaluate_model` must not propagate out of
    `compare()`: that would discard every result already computed -- including
    the other models' -- so a long sweep would lose everything to a single bad
    fit.
    """

    @staticmethod
    def _dataset():
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 3))
        y = (X[:, 0] > 0).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "task": "classification",
        }

    @staticmethod
    def _models():
        from sklearn.svm import SVC

        return {
            "bad": SVC(kernel="not-a-kernel"),
            "good": RandomForestClassifier(n_estimators=5),
        }

    def test_the_working_model_still_produces_results(self):
        with pytest.warns(UserWarning, match="failed on seed"):
            result = compare(models=self._models(), dataset=self._dataset(), n_seeds=2)
        assert {e["model"] for e in result.raw} == {"good"}
        assert isinstance(result.best_model(), RandomForestClassifier)

    def test_the_failure_is_reported_not_swallowed(self):
        with pytest.warns(UserWarning, match="bad failed on seed"):
            compare(models=self._models(), dataset=self._dataset(), n_seeds=1)


class TestAFailedSeedKeepsItsSlot:
    """`best_model(seed=n)` and `inference(seed=n)` document `n` as the n-th
    seed, and index straight into the per-seed lists.

    A failed seed therefore keeps its slot. Dropping it would slide every later
    seed down one, so a model that failed on seed 0 and fitted on seeds 1 and 2
    would answer `seed=0` with seed 1's estimator -- in silence -- and mislabel
    its artifact and loss curve the same way. `SVC(kernel="not-a-kernel")` fails
    on every seed, so it cannot exercise this mixed case.

    The failure is injected at `_evaluate_model` rather than with a flaky
    estimator class, because `compare()` accepts only built-in sklearn
    estimators and the kit's own bases.
    """

    @staticmethod
    def _run(monkeypatch, n_seeds=3):
        import inspect

        from eigenkit import benchmark as bench

        real = bench._evaluate_model
        signature = inspect.signature(real)

        def fail_first_seed(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            if bound.arguments["save_name"] == "flaky" and bound.arguments["seed"] == 0:
                raise RuntimeError("no fit on seed 0")
            return real(*args, **kwargs)

        monkeypatch.setattr(bench, "_evaluate_model", fail_first_seed)
        dataset = TestOneFailureDoesNotSinkTheRun._dataset()
        with pytest.warns(UserWarning, match="flaky failed on seed 0"):
            result = compare(
                models={
                    "flaky": RandomForestClassifier(n_estimators=5),
                    "good": RandomForestClassifier(n_estimators=5),
                },
                dataset=dataset,
                n_seeds=n_seeds,
            )
        return result, dataset

    def test_an_index_still_means_that_seed(self, monkeypatch):
        result, _ = self._run(monkeypatch)
        stored = result.fitted_models["flaky"]
        assert len(stored) == 3
        assert stored[0] is None
        assert [m.random_state for m in stored[1:]] == [1, 2]

    def test_the_artifact_and_curve_lists_stay_aligned(self, monkeypatch):
        result, _ = self._run(monkeypatch)
        assert len(result.model_artifacts["flaky"]) == 3
        assert len(result.loss_curves["flaky"]) == 3
        assert result.model_artifacts["flaky"][0] is None
        assert result.loss_curves["flaky"][0] is None

    def test_asking_for_the_failed_seed_says_it_failed(self, monkeypatch):
        result, dataset = self._run(monkeypatch)
        with pytest.raises(ValueError, match="flaky.*failed on seed 0"):
            result.inference(dataset["X_test"], models=["flaky"], seed=0)

    def test_a_surviving_seed_returns_its_own_fit(self, monkeypatch):
        result, dataset = self._run(monkeypatch)
        out = result.inference(dataset["X_test"], models=["flaky"], seed=1)
        assert len(out["flaky"]["predictions"]) == len(dataset["X_test"])
        assert result._get_fitted("flaky", 1).random_state == 1

    def test_last_steps_back_over_the_failure(self, monkeypatch):
        result, _ = self._run(monkeypatch)
        assert result._get_fitted("flaky", "last").random_state == 2

    def test_a_model_that_failed_everywhere_says_so(self):
        dataset = TestOneFailureDoesNotSinkTheRun._dataset()
        with pytest.warns(UserWarning, match="failed on seed"):
            result = compare(
                models=TestOneFailureDoesNotSinkTheRun._models(),
                dataset=dataset,
                n_seeds=2,
            )
        with pytest.raises(ValueError, match="failed on every seed"):
            result.inference(dataset["X_test"], models=["bad"])
