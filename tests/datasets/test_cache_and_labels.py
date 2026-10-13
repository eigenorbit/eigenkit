"""Cache integrity, and class statistics that survive real label encodings.

`cached_download` must leave nothing at the final cache path when a download
fails: a truncated file there would be returned as valid by every later call
-- forever, since no registry dataset carries a hash to check against. The
filename must be a bare name, since a name containing `../` joined onto the
cache directory would write outside it.

`summarize()` counts the labels that occur, not label values: counting with
`np.bincount`, which indexes by value, would invent zero-count classes for
every unused integer below the maximum and reject negative labels. A balanced
set labelled {0, 2} would report an imbalance ratio of 50, and {-1, 1} -- the
standard SVM convention -- would crash.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest

from eigenkit.datasets._cache import cached_download
from eigenkit.datasets.eda import DatasetReport


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EIGENKIT_DATA_DIR", str(tmp_path))
    return tmp_path


class TestCacheIntegrity:
    @pytest.mark.parametrize("name", ["../escaped.csv", "sub/dir.csv", "..", "."])
    def test_a_filename_with_path_structure_is_rejected(self, cache, name):
        with pytest.raises(ValueError, match="bare file name"):
            cached_download("http://127.0.0.1:1/unused", name)

    def test_a_failed_download_caches_nothing(self, cache):
        with pytest.raises(URLError):
            cached_download("http://127.0.0.1:1/unreachable", "data.csv")
        assert list(cache.iterdir()) == [], "a partial download was left cached"

    def test_a_completed_download_is_cached_and_reused(self, cache, monkeypatch):
        calls = []

        def fake_urlretrieve(url, path):
            calls.append(url)
            Path(path).write_text("a,b\n1,2\n")

        monkeypatch.setattr(
            "eigenkit.datasets._cache.urllib.request.urlretrieve",
            fake_urlretrieve,
        )
        first = cached_download("http://example/x.csv", "x.csv")
        second = cached_download("http://example/x.csv", "x.csv")
        assert first == second
        assert first.read_text() == "a,b\n1,2\n"
        assert len(calls) == 1, "the cached file was re-downloaded"


class TestClassStatistics:
    @pytest.mark.parametrize("labels", [(0, 1), (0, 2), (-1, 1), (5, 7), (100, 200)])
    def test_a_balanced_set_reports_no_imbalance(self, labels):
        X = np.random.default_rng(0).normal(size=(100, 3))
        low, high = labels
        y = np.array([low] * 50 + [high] * 50)
        summary = DatasetReport.from_data(X=X, y=y, task="classification").summarize()
        assert summary["class_imbalance_ratio"] == pytest.approx(1.0)
        assert summary["class_counts"] == [50, 50]
        assert summary["class_labels"] == [low, high]

    def test_a_genuinely_imbalanced_set_is_still_detected(self):
        X = np.random.default_rng(0).normal(size=(100, 3))
        y = np.array([0] * 90 + [1] * 10)
        summary = DatasetReport.from_data(X=X, y=y, task="classification").summarize()
        assert summary["class_imbalance_ratio"] == pytest.approx(9.0)
        assert any("imbalance" in r.lower() for r in summary["recommendations"])

    def test_a_balanced_set_gets_no_imbalance_recommendation(self):
        X = np.random.default_rng(0).normal(size=(100, 3))
        y = np.array([-1] * 50 + [1] * 50)
        summary = DatasetReport.from_data(X=X, y=y, task="classification").summarize()
        assert not any("imbalance" in r.lower() for r in summary["recommendations"])
