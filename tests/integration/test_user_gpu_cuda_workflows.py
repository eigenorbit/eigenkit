from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available (torch.cuda.is_available() is False)",
)


# --- all backends we surface in the matrix (string or config) ---
PROVIDER_IDS: tuple[str, ...] = (
    "pennylane",
    "pennylane_default_qubit",
    "pennylane_lightning_qubit",
    "pennylane_lightning_gpu",
    "qiskit",
    "braket",
    "cudaq_qpp",
    "cudaq_nvidia",
)

ALGO_IDS: tuple[str, ...] = (
    "hqnr",
    "hqnc",
    "qsvc",
    "qsvr",
    "custom_clf",
    "custom_reg",
    "recurrent_hqnr",
    "qts",
)

# ``QSVC`` / ``QSVR`` use ``QuantumKernel(n_qubits = n_features)`` — no mismatch case.
MISMATCH_ALGO_IDS: tuple[str, ...] = (
    "hqnr",
    "hqnc",
    "custom_clf",
    "custom_reg",
    "recurrent_hqnr",
    "qts",
)


def _log(msg: str, **fields: Any) -> None:
    tail = " ".join(f"{k}={v!r}" for k, v in fields.items())
    line = f"[gpu-matrix] {msg}" + (f" | {tail}" if tail else "")
    print(line, flush=True)


@contextmanager
def _wall(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
        _log("timing", label=label, seconds=f"{dt:.4f}", cuda_device=name)


def _pennylane_lightning_qubit_config():
    """CPU Lightning statevector; matrix still runs under ``@requires_cuda`` (torch CUDA session)."""
    import pennylane as qml

    from eigenkit import PennyLaneConfig

    try:
        dev = qml.device("lightning.qubit", wires=2)
        _ = dev.name
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"lightning.qubit unavailable: {exc}")
    return PennyLaneConfig(device_name="lightning.qubit")


def _pennylane_lightning_gpu_config():
    import pennylane as qml

    from eigenkit import PennyLaneConfig

    try:
        dev = qml.device("lightning.gpu", wires=2)
        _ = dev.name
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"lightning.gpu unavailable: {exc}")
    return PennyLaneConfig(device_name="lightning.gpu")


def resolve_provider(provider_id: str) -> Any:
    """Return a ``provider`` argument suitable for ``Circuit(..., provider=)``."""
    from eigenkit import (
        BraketConfig,
        CudaQuantumConfig,
        PennyLaneConfig,
        QiskitConfig,
    )

    if provider_id.startswith("pennylane"):
        pytest.importorskip("pennylane")
    if provider_id == "pennylane":
        return "pennylane"
    if provider_id == "pennylane_default_qubit":
        # Explicit default.qubit in a CUDA session (module requires torch.cuda).
        return PennyLaneConfig(device_name="default.qubit")
    if provider_id == "pennylane_lightning_qubit":
        return _pennylane_lightning_qubit_config()
    if provider_id == "pennylane_lightning_gpu":
        return _pennylane_lightning_gpu_config()
    if provider_id == "qiskit":
        pytest.importorskip("qiskit")
        return QiskitConfig()
    if provider_id == "braket":
        pytest.importorskip("braket")
        return BraketConfig()
    if provider_id == "cudaq_qpp":
        pytest.importorskip("cudaq")
        return CudaQuantumConfig(target="qpp-cpu", shots=None)
    if provider_id == "cudaq_nvidia":
        pytest.importorskip("cudaq")
        return CudaQuantumConfig(target="nvidia", shots=None)
    raise AssertionError(f"unknown provider_id: {provider_id}")


def _tensor_brief(t: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "mean": float(torch.mean(torch.abs(t)).detach().cpu()),
    }


def _snapshot_trainable_params(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def _assert_every_trainable_param_changed(
    module: torch.nn.Module,
    before: dict[str, torch.Tensor],
    *,
    provider_id: str,
    eps: float = 1e-10,
) -> None:
    params = list(module.named_parameters())
    assert params, f"expected trainable params, got none (provider_id={provider_id!r})"
    for name, p in params:
        delta = (p.detach() - before[name]).abs().max().item()
        assert delta > eps, (
            f"trainable {name!r} unchanged after 1 training epoch "
            f"(max_abs_delta={delta}, provider_id={provider_id!r})"
        )


@requires_cuda
@pytest.mark.parametrize("provider_id", PROVIDER_IDS)
def test_simple_encoding_variational_circuit_each_provider(provider_id: str):
    """Per provider: ``Encoding(2)`` + ``Variational`` + ``all_z``, forward + backward.

    Asserts finite ``layer.weights.grad`` and logs it; then logs ``x.grad``
    (parameter-shift backends may leave ``x.grad`` unset — failure is intentional).
    """
    from eigenkit import Circuit, Encoding, Variational, all_z

    prov = resolve_provider(provider_id)
    _log("simple_circuit_start", provider_id=provider_id)
    layer = Circuit(
        blocks=[
            Encoding(in_features=2, rotation="RY"),
            Variational(depth=1, rotation="RZ"),
        ],
        observables=all_z(2),
        provider=prov,
        # Without this the dispatcher returns the numpy circuit, which owns no
        # weights and takes them per call: `layer(x)` then reads as
        # `circuit(weights)` with no inputs, and `layer.weights` does not exist.
        interface="torch",
    )
    x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    with _wall(f"simple_circuit_forward[{provider_id}]"):
        y = layer(x)
    _log("simple_circuit_out", provider_id=provider_id, **_tensor_brief(y))
    assert y.shape == (4, 2)
    with _wall(f"simple_circuit_backward[{provider_id}]"):
        y.sum().backward()
    assert layer.weights.grad is not None
    assert bool(torch.all(torch.isfinite(layer.weights.grad)).item())
    _log(
        "simple_circuit_grad_weights",
        provider_id=provider_id,
        **_tensor_brief(layer.weights.grad),
    )
    _log("simple_circuit_grad", provider_id=provider_id, **_tensor_brief(x.grad))


@requires_cuda
@pytest.mark.parametrize("provider_id", PROVIDER_IDS)
def test_hqnr_all_trainable_weights_change_with_adapter(provider_id: str):
    """Every ``nn.Parameter`` updates after one epoch (HQNR with Linear 7→3 before quantum).

    Mirrors ``HQNR.fit`` training core so we can snapshot weights before ``train_loop``.
    Fails if any submodule (adapter, ``Circuit.weights``, readout head) receives no update.
    """
    from eigenkit import HQNR
    from eigenkit.torch.estimators._training import (
        make_dataloader,
        make_optimizer,
        train_loop,
    )

    prov = resolve_provider(provider_id)
    X, _, y = _tabular_xy(seed=101, feat=7)
    m = HQNR(
        in_features=7,
        n_qubits=3,
        depth=1,
        layer_type="Basic",
        epochs=1,
        batch_size=12,
        show_progress=False,
        provider=prov,
        lr=0.05,
    )
    model = m._build_model()
    before = _snapshot_trainable_params(model)
    _log(
        "hqnr_all_weights_change_start",
        provider_id=provider_id,
        n_params=len(before),
        param_names=list(before.keys()),
    )
    optimizer = make_optimizer(
        model.parameters(),
        optimizer=m.optimizer,
        lr=m.lr,
        optimizer_kwargs=m.optimizer_kwargs,
    )
    dl = make_dataloader(X, y, batch_size=m.batch_size, target_dtype=torch.float32)
    loss_fn = m._make_loss_fn()
    with _wall(f"hqnr_all_weights_train[{provider_id}]"):
        train_loop(
            model,
            dl,
            optimizer,
            loss_fn,
            m.epochs,
            show_progress=False,
            progress_desc="HQNR all-weights",
        )
    _assert_every_trainable_param_changed(model, before, provider_id=provider_id)
    _log("hqnr_all_weights_change_ok", provider_id=provider_id)


def _tabular_xy(*, n: int = 24, feat: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, feat)).astype(np.float32)
    y_cls = (X[:, 0] + 0.1 * X[:, 1] > 0).astype(np.int64)
    y_reg = (0.4 * X[:, 0] + 0.2 * X[:, 1]).astype(np.float32)
    return X, y_cls, y_reg


def _run_hqnr(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import HQNR

    X, _, y = _tabular_xy(seed=1)
    m = HQNR(
        in_features=4,
        depth=1,
        layer_type="Basic",
        epochs=1,
        batch_size=12,
        show_progress=False,
        provider=prov,
    )
    _log("hqnr_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y)
    _log("hqnr_fit_done", provider_id=pid, algo_id=aid, losses=m.train_losses_)
    pred = m.predict(X[:6])
    _log("hqnr_predict", provider_id=pid, algo_id=aid, pred_shape=pred.shape)


def _run_hqnc(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import HQNC

    X, y_cls, _ = _tabular_xy(seed=2)
    m = HQNC(
        in_features=4,
        depth=1,
        n_classes=2,
        epochs=1,
        batch_size=12,
        show_progress=False,
        provider=prov,
    )
    _log("hqnc_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y_cls)
    _log("hqnc_fit_done", provider_id=pid, algo_id=aid, losses=m.train_losses_)
    acc = float(m.score(X, y_cls))
    _log("hqnc_score", provider_id=pid, algo_id=aid, accuracy=acc)


def _run_qsvc(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import QSVC

    rng = np.random.default_rng(3)
    X = rng.standard_normal((10, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    m = QSVC(feature_map="Angle", reps=1, provider=prov)
    _log("qsvc_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y)
    _log("qsvc_fit_done", provider_id=pid, algo_id=aid)
    p = m.predict(X)
    _log("qsvc_predict", provider_id=pid, algo_id=aid, preds=p.tolist())


def _run_qsvr(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import QSVR

    rng = np.random.default_rng(4)
    X = rng.standard_normal((10, 3)).astype(np.float32)
    y = X[:, 0].astype(np.float32)
    m = QSVR(feature_map="Angle", reps=1, provider=prov)
    _log("qsvr_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y)
    _log("qsvr_fit_done", provider_id=pid, algo_id=aid)
    p = m.predict(X)
    _log("qsvr_predict", provider_id=pid, algo_id=aid, preds=p.tolist())


def _run_custom_clf(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import CustomQuantumClassifier, Encoding, Variational, all_z

    X, y_cls, _ = _tabular_xy(seed=5)
    blocks = [Encoding(in_features=4, rotation="RY"), Variational(depth=1, rotation="RZ")]
    m = CustomQuantumClassifier(
        blocks=blocks,
        observables=all_z(4),
        provider=prov,
        n_classes=2,
        epochs=1,
        batch_size=12,
        show_progress=False,
    )
    _log("custom_clf_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y_cls)
    _log("custom_clf_fit_done", provider_id=pid, algo_id=aid, losses=m.train_losses_)
    m.predict(X[:4])


def _run_custom_reg(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import CustomQuantumRegressor, Encoding, Variational, all_z

    X, _, y = _tabular_xy(seed=6)
    blocks = [Encoding(in_features=4, rotation="RY"), Variational(depth=1, rotation="RZ")]
    m = CustomQuantumRegressor(
        blocks=blocks,
        observables=all_z(4),
        provider=prov,
        epochs=1,
        batch_size=12,
        show_progress=False,
    )
    _log("custom_reg_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y)
    _log("custom_reg_fit_done", provider_id=pid, algo_id=aid, losses=m.train_losses_)
    m.predict(X[:4])


def _run_recurrent_hqnr(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import RecurrentHQNR

    rng = np.random.default_rng(7)
    X = rng.standard_normal((16, 5, 2)).astype(np.float32)
    y = X[:, -1, 0].astype(np.float32)
    m = RecurrentHQNR(
        seq_feature_dim=2,
        rnn_type="gru",
        rnn_hidden_dim=4,
        n_qubits=4,
        depth=1,
        layer_type="Basic",
        epochs=1,
        batch_size=8,
        show_progress=False,
        provider=prov,
    )
    _log("recurrent_hqnr_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y)
    _log("recurrent_hqnr_fit_done", provider_id=pid, algo_id=aid, losses=m.train_losses_)
    m.predict(X[:3])


def _run_qts(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import HQNR, QuantumTimeSeriesRegressor

    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = (X[:, 0] * 0.3 + 0.1 * X[:, 1]).astype(np.float32)
    inner = HQNR(
        in_features=4,
        depth=1,
        epochs=1,
        batch_size=8,
        show_progress=False,
        provider=prov,
    )
    m = QuantumTimeSeriesRegressor(estimator=inner, in_size=2, shift=1)
    _log("qts_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y)
    _log("qts_fit_done", provider_id=pid, algo_id=aid)
    m.predict(X)


def _run_hqnr_mismatch_feats_qubits(prov: Any, pid: str, aid: str) -> None:
    """Tabular dim 7, quantum wires 3 (library prepends ``Linear(7, 3)``)."""
    from eigenkit import HQNR

    X, _, y = _tabular_xy(seed=21, feat=7)
    m = HQNR(
        in_features=7,
        n_qubits=3,
        depth=1,
        layer_type="Basic",
        epochs=1,
        batch_size=12,
        show_progress=False,
        provider=prov,
    )
    _log("hqnr_mismatch_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y)
    _log("hqnr_mismatch_fit_done", provider_id=pid, algo_id=aid, losses=m.train_losses_)
    m.predict(X[:6])


def _run_hqnc_mismatch_feats_qubits(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import HQNC

    X, y_cls, _ = _tabular_xy(seed=22, feat=7)
    m = HQNC(
        in_features=7,
        n_qubits=3,
        depth=1,
        n_classes=2,
        epochs=1,
        batch_size=12,
        show_progress=False,
        provider=prov,
    )
    _log("hqnc_mismatch_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y_cls)
    _log("hqnc_mismatch_fit_done", provider_id=pid, algo_id=aid, losses=m.train_losses_)
    float(m.score(X, y_cls))


def _run_custom_clf_mismatch_feats_qubits(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import CustomQuantumClassifier, Encoding, Variational, all_z

    X, y_cls, _ = _tabular_xy(seed=23, feat=7)
    blocks = [Encoding(in_features=3, rotation="RY"), Variational(depth=1, rotation="RZ")]
    m = CustomQuantumClassifier(
        blocks=blocks,
        observables=all_z(3),
        in_features=7,
        provider=prov,
        n_classes=2,
        epochs=1,
        batch_size=12,
        show_progress=False,
    )
    _log("custom_clf_mismatch_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y_cls)
    _log(
        "custom_clf_mismatch_fit_done",
        provider_id=pid,
        algo_id=aid,
        losses=m.train_losses_,
    )
    m.predict(X[:4])


def _run_custom_reg_mismatch_feats_qubits(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import CustomQuantumRegressor, Encoding, Variational, all_z

    X, _, y = _tabular_xy(seed=24, feat=7)
    blocks = [Encoding(in_features=3, rotation="RY"), Variational(depth=1, rotation="RZ")]
    m = CustomQuantumRegressor(
        blocks=blocks,
        observables=all_z(3),
        in_features=7,
        provider=prov,
        epochs=1,
        batch_size=12,
        show_progress=False,
    )
    _log("custom_reg_mismatch_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y)
    _log(
        "custom_reg_mismatch_fit_done",
        provider_id=pid,
        algo_id=aid,
        losses=m.train_losses_,
    )
    m.predict(X[:4])


def _run_recurrent_hqnr_mismatch_feats_qubits(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import RecurrentHQNR

    rng = np.random.default_rng(25)
    X = rng.standard_normal((16, 5, 2)).astype(np.float32)
    y = X[:, -1, 0].astype(np.float32)
    m = RecurrentHQNR(
        seq_feature_dim=2,
        rnn_type="gru",
        rnn_hidden_dim=8,
        n_qubits=3,
        depth=1,
        layer_type="Basic",
        epochs=1,
        batch_size=8,
        show_progress=False,
        provider=prov,
    )
    _log("recurrent_hqnr_mismatch_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y)
    _log(
        "recurrent_hqnr_mismatch_fit_done",
        provider_id=pid,
        algo_id=aid,
        losses=m.train_losses_,
    )
    m.predict(X[:3])


def _run_qts_mismatch_feats_qubits(prov: Any, pid: str, aid: str) -> None:
    from eigenkit import HQNR, QuantumTimeSeriesRegressor

    rng = np.random.default_rng(26)
    X = rng.standard_normal((32, 4)).astype(np.float32)
    y = (0.3 * X[:, 0] + 0.1 * X[:, 1]).astype(np.float32)
    win, n_feat = 3, 4
    flat_dim = win * n_feat  # 12
    n_q = 3
    inner = HQNR(
        in_features=flat_dim,
        n_qubits=n_q,
        depth=1,
        epochs=1,
        batch_size=8,
        show_progress=False,
        provider=prov,
    )
    m = QuantumTimeSeriesRegressor(estimator=inner, in_size=win, shift=1)
    _log("qts_mismatch_fit_start", provider_id=pid, algo_id=aid)
    m.fit(X, y)
    _log("qts_mismatch_fit_done", provider_id=pid, algo_id=aid)
    m.predict(X)


_ALGO_RUNNERS: dict[str, Callable[[Any, str, str], None]] = {
    "hqnr": _run_hqnr,
    "hqnc": _run_hqnc,
    "qsvc": _run_qsvc,
    "qsvr": _run_qsvr,
    "custom_clf": _run_custom_clf,
    "custom_reg": _run_custom_reg,
    "recurrent_hqnr": _run_recurrent_hqnr,
    "qts": _run_qts,
}

_MISMATCH_ALGO_RUNNERS: dict[str, Callable[[Any, str, str], None]] = {
    "hqnr": _run_hqnr_mismatch_feats_qubits,
    "hqnc": _run_hqnc_mismatch_feats_qubits,
    "custom_clf": _run_custom_clf_mismatch_feats_qubits,
    "custom_reg": _run_custom_reg_mismatch_feats_qubits,
    "recurrent_hqnr": _run_recurrent_hqnr_mismatch_feats_qubits,
    "qts": _run_qts_mismatch_feats_qubits,
}


@requires_cuda
@pytest.mark.parametrize("algo_id", ALGO_IDS)
@pytest.mark.parametrize("provider_id", PROVIDER_IDS)
def test_each_algorithm_on_each_provider_gpu_log(algo_id: str, provider_id: str):
    """Full Cartesian product: same code path users hit; failures stay visible."""
    prov = resolve_provider(provider_id)
    _log("algo_provider_combo_start", algo_id=algo_id, provider_id=provider_id)
    runner = _ALGO_RUNNERS[algo_id]
    with _wall(f"algo={algo_id}|provider={provider_id}"):
        runner(prov, provider_id, algo_id)
    _log("algo_provider_combo_done", algo_id=algo_id, provider_id=provider_id)


@requires_cuda
@pytest.mark.parametrize("algo_id", MISMATCH_ALGO_IDS)
@pytest.mark.parametrize("provider_id", PROVIDER_IDS)
def test_each_algorithm_mismatch_features_vs_qubits_gpu_log(
    algo_id: str, provider_id: str
):
    """``in_features`` (or tabular width / RNN width) differs from ``n_qubits``.

    ``QSVC`` / ``QSVR`` are omitted: kernel circuits tie qubit count to feature
    count. Same provider matrix as the main GPU workflow test.
    """
    prov = resolve_provider(provider_id)
    _log(
        "algo_mismatch_combo_start",
        algo_id=algo_id,
        provider_id=provider_id,
    )
    runner = _MISMATCH_ALGO_RUNNERS[algo_id]
    with _wall(f"algo_mismatch={algo_id}|provider={provider_id}"):
        runner(prov, provider_id, algo_id)
    _log("algo_mismatch_combo_done", algo_id=algo_id, provider_id=provider_id)
