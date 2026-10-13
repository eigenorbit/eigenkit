# Tests

Tests for `eigenkit`, grouped by the ring of the library they cover.
The layout mirrors `src/eigenkit/`, so a test's directory tells you
which layer it is allowed to depend on.

## Layout

| Directory | Covers |
|---|---|
| `core/` | The framework-free compute layer: IR, `CircuitSpec`, builder, blocks, observables, the numpy reference simulator, and every backend adapter (PennyLane, Qiskit, Aer, Braket, Qiskit Runtime, CUDA-Q, TensorCircuit, Qulacs, qsim). Cross-backend parity lives here. |
| `layers/` | The torch ring's `nn.Module` layers — `Basic`, `Reupload`, `Ensemble`, `MinMaxNorm`. |
| `algorithms/` | The scikit-learn-style estimators (`HQNC`/`HQNR`, the recurrent and generic variants, `QSVC`/`QSVR`, `QRC`), plus `HHL`, `IQAE` and portfolio risk. |
| `chemistry/` | Fermionic operators, Jordan-Wigner / Bravyi-Kitaev mappings, UCCSD, molecules, VQE. |
| `optimization/` | QUBO/Ising problems and QAOA. |
| `reservoir/` | Quantum reservoir computing and its backend parity. |
| `datasets/` | Bundled loaders, the synthetic generators, the cache, and EDA. |
| `benchmark/` | The model-comparison harness: baselines, metrics, results, seeding. |
| `integration/` | End-to-end workflows that need real hardware — currently the GPU/CUDA sweep. |
| `test_public_api_contract.py` | What the top-level `__init__` promises: exported names, error messages, and that importing stays cheap. |
| `test_top_level_circuit.py` | The user-facing `Circuit(...)` entry point: blocks-list construction and the polymorphic `provider=` argument. |

## Running

```bash
# Everything (optional backends auto-skip when their package is absent).
uv run pytest

# One ring, or one file, or one test.
uv run pytest tests/core
uv run pytest tests/core/test_backends.py
uv run pytest tests/core/test_backends.py::TestForwardParityExtended

# Skip the slow cases.
uv run pytest -m "not slow"
```

Coverage:

```bash
uv run --with pytest-cov pytest tests/core \
    --cov=src/eigenkit/core \
    --cov-report=term-missing
```

## Markers

| Marker | Meaning |
|---|---|
| `slow` | Deselect with `-m "not slow"`. |
| `qpu` | Needs real QPU/cloud access. Skipped by default. |
| `gpu` | Needs CUDA (`torch.cuda.is_available()`). |
| `pennylane` | Needs the `pennylane` extra. Skipped when it is not installed; set automatically on a case parametrized with `"pennylane"`. |

## Conventions

- **Optional backends skip, they do not fail.** Guard on the package with
  `pytest.importorskip("<package>")` so a bare install still runs the suite
  green. PennyLane is the exception: mark the test `@pytest.mark.pennylane`,
  and only where it really uses PennyLane. A case parametrized with the
  provider `"pennylane"` is marked automatically. A provider-agnostic test runs
  on `provider="numpy"`, so a library path that starts needing PennyLane fails
  instead of skipping.
- **The numpy simulator is the oracle.** A backend adapter is correct when it
  matches `provider="numpy"` (or PennyLane, for the gradient sweeps) to machine
  precision — that is what the parity tests in `core/` assert.
- **Test names say the behaviour**, not the function: prefer
  `test_an_unknown_provider_lists_the_known_ones` over `test_get_raises`.
