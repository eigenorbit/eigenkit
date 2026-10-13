from eigenkit.benchmark import compare, quick_baselines


def test_quick_baselines_returns_list():
    baselines = quick_baselines()
    assert len(baselines) >= 4
    for b in baselines:
        assert hasattr(b, "fit") and hasattr(b, "predict")


def test_quick_baselines_in_compare():
    results = compare(
        models=quick_baselines()[:2],
        dataset="iris",
        metrics=["accuracy"],
        n_seeds=1,
    )
    assert len(results.raw) > 0
