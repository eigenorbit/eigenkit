"""Read and write `.eigenkit` bundles, which are zip archives.

PyTorch is imported only when a bundle holds PyTorch state. So importing this
module, or the scikit-learn estimators that import it, does not import PyTorch.

In the future, the `.pt` files will be replaced by `.safetensors` files, when
support for TensorFlow models is added. safetensors is not a dependency yet.
"""

from __future__ import annotations

import dataclasses
import io
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np

from eigenkit.core.blocks import (
    S0,
    AmplitudeEncoding,
    AngleEncoding,
    BasicVariational,
    ControlledUnitaryBlock,
    CustomBlock,
    Grover,
    HamiltonianQPEBlock,
    QFTBlock,
    QPEBlock,
    QubitUnitary,
    Reuploading,
    StrongVariational,
    TrotterBlock,
    _data_item_from_json,
    _data_item_to_json,
)
from eigenkit.core.blocks import (
    _param_from_json as _param_from_dict,
)
from eigenkit.core.blocks import (
    _param_to_json as _param_to_dict,
)
from eigenkit.core.ir import Gate
from eigenkit.core.secret_fields import secret_field_names
from eigenkit.core.wire import observables_from_json, observables_to_json


class SerializationError(Exception):
    """Raised when a model cannot be written to a bundle, or a bundle cannot be read."""

    pass


_SUPPORTED_FORMAT_VERSION = 1


def check_format_version(metadata: dict) -> None:
    """Raise a `SerializationError` if the bundle's `format_version` is unsupported.

    Args:
        metadata: The metadata of the bundle, read from `metadata.json`.
    """
    version = metadata.get("format_version")
    if version != _SUPPORTED_FORMAT_VERSION:
        raise SerializationError(
            f"Bundle format_version={version} is not supported by this version of "
            f"eigenkit (supports: format_version={_SUPPORTED_FORMAT_VERSION}). "
            "Upgrade the library to load this bundle."
        )


def _append_suffix(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    if path.name.endswith(suffix):
        return path
    return path.with_name(f"{path.name}{suffix}")


def resolve_bundle_path(path: str | Path) -> Path:
    """Add `.eigenkit` to a bundle path, unless the name already ends with it.

    Args:
        path: The path of the bundle.

    Returns:
        The path, with the `.eigenkit` suffix.
    """
    path = Path(path)
    return _append_suffix(path, ".eigenkit")


_BLOCK_TYPES = {
    "AngleEncoding": AngleEncoding,
    "Encoding": AngleEncoding,  # public alias (configs.Encoding)
    "BasicVariational": BasicVariational,
    "Variational": BasicVariational,  # public alias (configs.Variational)
    "StrongVariational": StrongVariational,
    "Reuploading": Reuploading,
    # QAE / QPE blocks
    "AmplitudeEncoding": AmplitudeEncoding,
    "S0": S0,
    "QubitUnitary": QubitUnitary,
    "Grover": Grover,
    "QFTBlock": QFTBlock,
    "QPEBlock": QPEBlock,
    "ControlledUnitaryBlock": ControlledUnitaryBlock,
    "TrotterBlock": TrotterBlock,
    "HamiltonianQPEBlock": HamiltonianQPEBlock,
}


# ---- IR gate serialization (used by CustomBlock) ----
# _param_to_dict / _param_from_dict are imported from blocks._param_to_json /
# _param_from_json (same JSON format, single implementation).


def _gate_to_dict(g: Gate) -> dict:
    result = {
        "n": g.name,
        "q": list(g.qubits),
        "p": [_param_to_dict(p) for p in g.params],
    }
    # Serialize the data payload (AmplitudeEncoding amplitudes, QubitUnitary
    # matrix): without gate.data, a CustomBlock containing data-carrying gates
    # would not survive a round-trip.
    if g.data:
        result["d"] = [_data_item_to_json(item) for item in g.data]
    if g.adjoint:
        result["adj"] = True
    return result


def _gate_from_dict(d: dict) -> Gate:
    data = tuple(_data_item_from_json(item) for item in d.get("d", []))
    return Gate(
        name=d["n"],
        qubits=tuple(d["q"]),
        params=tuple(_param_from_dict(p) for p in d["p"]),
        data=data,
        adjoint=d.get("adj", False),
    )


def _custom_block_to_dict(block: CustomBlock) -> dict:
    return {
        "type": "CustomBlock",
        "n_qubits": block.n_qubits,
        "gates": [_gate_to_dict(g) for g in block.gates],
        "n_weights": block.n_weights,
        "n_inputs": block.n_inputs,
    }


def _custom_block_from_dict(d: dict) -> CustomBlock:
    return CustomBlock(
        n_qubits=d["n_qubits"],
        gates=tuple(_gate_from_dict(g) for g in d["gates"]),
        n_weights=d["n_weights"],
        n_inputs=d["n_inputs"],
    )


def _block_to_dict(block) -> dict:
    if isinstance(block, CustomBlock):
        return _custom_block_to_dict(block)
    if not hasattr(block, "to_dict"):
        raise SerializationError(
            f"{type(block).__name__} cannot be serialized. "
            "Implement to_dict() / from_dict() or use built-in block types."
        )
    return block.to_dict()


def _block_from_dict(d: dict):
    t = d.get("type")
    if t == "CustomBlock":
        return _custom_block_from_dict(d)
    if t not in _BLOCK_TYPES:
        raise SerializationError(
            f"Unknown block type: {t!r}. Known: {sorted({*_BLOCK_TYPES, 'CustomBlock'})}"
        )
    return _BLOCK_TYPES[t].from_dict(d)


# ---- provider config serialization ----


def serialize_provider_param(provider) -> object:
    """Convert a provider value to a form that JSON can store.

    Strings and `None` are returned unchanged. A configuration dataclass is stored
    as `{"__provider_config__": ClassName, ...fields}`. The fields marked as secret
    (`metadata={"secret": True}`, such as an access token) are never written, so
    they take their default value when the bundle is loaded. `enable_gradients` is
    not written either: the person who loads a bundle decides whether to pay for
    gradients on hardware, and a shared bundle cannot decide it for them. The
    fields that say where a provider connects (`_ENDPOINT_FIELDS`) are not written
    either.

    Raises:
        SerializationError: If `provider` is not a string, `None` or a
            configuration dataclass.
    """
    if provider is None or isinstance(provider, str):
        return provider
    if dataclasses.is_dataclass(provider) and not isinstance(provider, type):
        fields = dataclasses.asdict(provider)
        for name in (
            *secret_field_names(provider),
            "enable_gradients",
            *_ENDPOINT_FIELDS,
        ):
            fields.pop(name, None)
        return {"__provider_config__": type(provider).__name__, **fields}
    raise SerializationError(
        f"Parameter 'provider' has type {type(provider).__name__} which is not "
        "JSON-serializable. Use a registered name string or a config dataclass."
    )


# Fields that decide where a provider connects. They are never written to a
# bundle and never read from one: a bundle is data from whoever shared it, and
# loading it builds the backend with the loader's credentials (IQM_TOKEN,
# QISKIT_IBM_TOKEN), which must not go to a host the bundle names.
_ENDPOINT_FIELDS = ("url", "service_kwargs", "provider_kwargs", "backend_instance")


def deserialize_provider_param(value) -> object:
    """Restore a provider value from its JSON form.

    The endpoint fields (`_ENDPOINT_FIELDS`) in the bundle are ignored, with a
    warning. To connect elsewhere, pass `provider=` to `load()`.

    Raises:
        SerializationError: If the configuration class is unknown.
    """
    if not isinstance(value, dict) or "__provider_config__" not in value:
        return value
    cls_name = value["__provider_config__"]
    kwargs = {k: v for k, v in value.items() if k != "__provider_config__"}
    ignored = [name for name in _ENDPOINT_FIELDS if kwargs.pop(name, None)]
    if ignored:
        warnings.warn(
            f"ignored {', '.join(ignored)} in the bundle's {cls_name}: a bundle "
            "cannot choose where a provider connects. Pass provider= to load() "
            "to set them.",
            UserWarning,
            stacklevel=2,
        )
    from eigenkit import configs

    # Every config `eigenkit.configs` exports, so a new one cannot save fine and
    # then fail to load. Only the class named is imported: the others may need
    # SDKs this install lacks.
    known = sorted({"NumpyConfig", *configs._LAZY_CONFIGS})
    if cls_name not in known:
        raise SerializationError(
            f"Unknown provider config class {cls_name!r}. Known: {known}"
        )
    return getattr(configs, cls_name)(**kwargs)


def _observables_to_dict(observables) -> list | dict:
    return observables_to_json(observables)


def _observables_from_dict(data) -> object:
    return observables_from_json(data)


def serialize_circuit_spec(blocks, observables) -> dict:
    """Convert blocks and observables to a dictionary that JSON can store."""
    return {
        "blocks": [_block_to_dict(b) for b in blocks],
        "observables": _observables_to_dict(observables)
        if observables is not None
        else None,
    }


def deserialize_circuit_spec(spec: dict) -> tuple[list, object | None]:
    """Restore blocks and observables from a JSON dictionary.

    Returns:
        blocks: The list of blocks.
        observables: The observables, or `None` when none were stored.
    """
    blocks = [_block_from_dict(b) for b in spec["blocks"]]
    observables = (
        _observables_from_dict(spec["observables"]) if spec.get("observables") else None
    )
    return blocks, observables


def _is_tensor_state_dict(value: dict) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        import torch
    except ImportError:
        return False  # no torch installed -> nothing can be a torch.Tensor
    return all(isinstance(v, torch.Tensor) for v in value.values())


def to_json_safe(value):
    """Convert NumPy values to native Python types that JSON accepts.

    NumPy scalars and arrays are converted, also inside lists, tuples and
    dictionaries. Other values are returned unchanged.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    return value


def _serialize_state_value(key: str, value) -> tuple[str, bytes]:
    if isinstance(value, dict) and _is_tensor_state_dict(value):
        import torch

        buf = io.BytesIO()
        torch.save(value, buf)
        return f"state/{key}.pt", buf.getvalue()

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise SerializationError(
                f"Cannot serialize object-dtype array for key {key!r}. "
                "Use a numeric dtype or a JSON-serializable list/dict."
            )
        buf = io.BytesIO()
        np.save(buf, value)
        return f"state/{key}.npy", buf.getvalue()

    if isinstance(value, (int, float, bool, str, np.integer, np.floating, np.bool_)):
        safe = to_json_safe(value)
        return f"state/{key}.json", json.dumps({"value": safe}).encode()

    if isinstance(value, (list, dict, tuple)):
        try:
            payload = json.dumps(to_json_safe(value)).encode()
        except (TypeError, ValueError) as exc:
            raise SerializationError(
                f"Cannot serialize value for key {key!r}: not JSON-serializable."
            ) from exc
        return f"state/{key}.json", payload

    raise SerializationError(
        f"Cannot serialize value of type {type(value).__name__} for key {key!r}. "
        "Override _collect_state() to return only dict[str, Tensor] (state_dict), "
        "numpy.ndarray, int/float/bool/str, or JSON-serializable list/dict."
    )


def save_bundle(
    path: str | Path,
    *,
    metadata: dict,
    params: dict,
    circuit_spec: dict | None,
    state: dict,
) -> None:
    """Write a `.eigenkit` zip archive."""
    # Serialize everything before opening the file: a value that fails must not
    # leave a truncated archive in place of an earlier good one.
    entries = [
        ("metadata.json", json.dumps(metadata)),
        ("params.json", json.dumps(params)),
    ]
    if circuit_spec is not None:
        entries.append(("circuit_spec.json", json.dumps(circuit_spec)))
    entries += [_serialize_state_value(key, value) for key, value in state.items()]
    path = resolve_bundle_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, data in entries:
            zf.writestr(arcname, data)


def load_bundle(path: str | Path) -> tuple[dict, dict, dict | None, dict]:
    """Read a `.eigenkit` archive.

    Returns:
        metadata: The contents of `metadata.json`.
        params: The contents of `params.json`.
        circuit_spec: The contents of `circuit_spec.json`, or `None` when the
            bundle has none.
        state: The saved state, with one entry for each file in `state/`.
    """
    path = resolve_bundle_path(path)
    state: dict = {}
    circuit_spec: dict | None = None

    with zipfile.ZipFile(path, "r") as zf:
        metadata = json.loads(zf.read("metadata.json"))
        params = json.loads(zf.read("params.json"))
        if "circuit_spec.json" in zf.namelist():
            circuit_spec = json.loads(zf.read("circuit_spec.json"))

        for name in zf.namelist():
            if not name.startswith("state/") or name.endswith("/"):
                continue
            key = Path(name).stem
            data = zf.read(name)
            if name.endswith(".pt"):
                import torch

                state[key] = torch.load(io.BytesIO(data), weights_only=True)
            elif name.endswith(".npy"):
                state[key] = np.load(io.BytesIO(data), allow_pickle=False)
            elif name.endswith(".json"):
                loaded = json.loads(data)
                if isinstance(loaded, dict) and set(loaded.keys()) == {"value"}:
                    state[key] = loaded["value"]
                else:
                    state[key] = loaded
            else:
                raise SerializationError(
                    f"Unknown state file extension in bundle: {name!r}"
                )

    return metadata, params, circuit_spec, state
