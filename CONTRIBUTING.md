# Contributing to eigenkit

Thanks for your interest in contributing. This document covers two
things:

1. **How to contribute** — local setup, running tests, opening PRs.
2. **How to extend the library** — adding backends, blocks, gates,
   observable types, layers, algorithms, and datasets.

Read [ARCHITECTURE.md](ARCHITECTURE.md) first for a tour of the
codebase. The library is a framework-free core with optional rings
around it; the ring a change belongs to decides where its code goes and
what it may import (see [Core and rings](#core-and-rings)).

---

## How to contribute

### Prerequisites

- Python 3.12, 3.13 or 3.14 (`requires-python` in `pyproject.toml`).
- [uv](https://docs.astral.sh/uv/) for dependency management.
- A `python3.13` on your `PATH`: the pre-commit hooks run under it
  (`default_language_version` in `.pre-commit-config.yaml`).

### Local setup

After cloning, install every extra, the way CI does:

```bash
uv sync --extra all --extra perceval
```

`all` covers every extra except `perceval`, which stays out of it
because its simulator dependency pins protobuf exactly; the photonics
tests need it. A plain `uv sync` installs only the core (numpy, scipy,
pydantic) and the `dev` dependency group (pytest, ruff, mkdocs,
pre-commit, import-linter, semantic-release); the docs build and most
of the test suite need the extras.

`cudaquantum` (part of `all`) installs NVIDIA CUDA-Q on Linux (CPU +
GPU) and macOS Apple Silicon (CPU only); it is a no-op on Windows and
macOS x86_64. The extras table in [README.md](README.md) lists the rest.

Install the pre-commit hooks once:

```bash
uv run pre-commit install
```

#### torch

torch is an *extra*, so a bare `pip install eigenkit` pulls none of
it. Ask for it with the `torch` extra, or with `all`, which includes it.

| Environment               | Command                         | Result                                                       |
| ------------------------- | ------------------------------- | ------------------------------------------------------------ |
| macOS dev (Apple Silicon) | `uv sync --extra torch`         | `torch==2.12.0` from PyPI                                    |
| Linux/Windows dev         | `uv sync --extra torch`         | `torch==2.12.0+cu126` from the `pytorch-cu126` index (~3 GB) |
| PyPI user (any OS)        | `pip install "eigenkit[torch]"` | whichever torch release satisfies `torch>=2.5,<3`            |

`uv.lock` pins torch 2.12.0, with a macOS wheel for Apple Silicon only
(`macosx_14_0_arm64`). The `[tool.uv.sources]` block in
`pyproject.toml` routes torch to PyTorch's CUDA 12.6 index on every
other platform. That routing is uv-only and is not part of the
published package metadata, so PyPI users never see it.

If you want CPU torch on Linux or Windows, replace the CUDA wheel after
syncing:

```bash
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cpu torch==2.12.0
```

A plain `uv run` syncs the environment back to `uv.lock`, which puts
the CUDA wheel back, so from then on pass `--no-sync`:

```bash
uv run --no-sync pytest
```

### Running tests

```bash
# The whole suite (after the full setup above).
uv run pytest

# A single file.
uv run pytest tests/core/test_circuit.py

# A single test.
uv run pytest tests/core/test_circuit.py::TestScalarVectorOutput::test_pauli_expr_accepted

# QPU / cloud integration tests (real billable infrastructure).
uv run pytest -m qpu
```

The markers are declared in `pyproject.toml`:

- `qpu` tests need real QPU or cloud access; `conftest.py` skips them
  unless you select them with `-m qpu`.
- `gpu` tests need CUDA and skip when `torch.cuda.is_available()` is
  false.
- `pennylane` tests need the `pennylane` extra; `conftest.py` skips them
  when PennyLane is not installed.
- `slow` marks long-running tests; CI deselects them with
  `-m "not slow"`.

[tests/README.md](tests/README.md) describes the layout of `tests/` and
the conventions for new tests.

Coverage:

```bash
uv run --with pytest-cov pytest tests/core \
    --cov=src/eigenkit/core \
    --cov-report=term-missing
```

For an HTML report, swap `term-missing` for `html` and open
`htmlcov/index.html`.

### Building the docs

```bash
uv run mkdocs serve         # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict
```

mkdocstrings imports every module it documents, so the build needs the
`all` extra. The strict build is what CI runs; please make sure it
passes before opening a PR.

Every page in [docs/examples/](docs/examples/) has a runnable notebook
generated from it in `examples/notebooks/`, without outputs. The page is
the source of truth and the notebook is never edited by hand. The
`example-notebooks` pre-commit hook rebuilds it when you commit a page
change; stage the rewritten notebook and commit again. The `lint` workflow
runs the same hooks, so it fails when a notebook no longer matches its page
or was committed with outputs.

### Code style

The pre-commit hooks run `ruff check --fix`, `ruff format`, a few
file-hygiene checks, and the `example-notebooks` hook. CI rejects PRs
that fail pre-commit, so it's easiest to install the hooks locally. One
command runs every hook over the whole tree, as the `lint` workflow
does:

```bash
uv run pre-commit run --all-files
```

You can also run Ruff on its own. It skips `examples/notebooks/`
(`extend-exclude` in `pyproject.toml`), because those notebooks are
generated from the pages:

```bash
uv run ruff format .
uv run ruff check --fix .
```

A few code-level conventions worth knowing:

- **Default to no comments.** Add one only when the *why* is
  non-obvious — a hidden constraint, a workaround, behavior that would
  surprise a reader. Don't restate what the code does.
- **No marketing prose.** Docstrings should be concise. No emoji
  and no heavy unicode math.
- **Prefer extending existing files** to creating new ones.
- **Frozen dataclasses** in the data layer (`core/ir.py`,
  `core/blocks.py` fragments). They're hashable, picklable, and safe to
  share between threads.
- **Type hints everywhere** at module boundaries. Use `from __future__
  import annotations`.

### Continuous integration

Every pull request to `main` runs three workflows:

- **`lint`** (`.github/workflows/lint.yml`) — the pre-commit hooks over
  every file.
- **`docs-checks`** (`.github/workflows/docs-checks.yml`) — the strict
  docs build and the checker scripts in `tools/docs-checks/`.
- **`ci`** (`.github/workflows/ci.yml`) — pytest on Python 3.12, 3.13
  and 3.14 with the `all` and `perceval` extras, without `slow` tests; a
  core-only job that installs no extra and runs `uv run lint-imports`
  and the framework-free tests; and a job with only the `torch` extra
  that runs the whole suite, so a test that imports an optional SDK
  without `pytest.importorskip` errors there. `ci` skips a pull request
  that changes only Markdown files and `docs/`.

Any finding of a docs checker fails the PR. The ones extension work
most often trips:

- `check_func_params.py --strict` — every public parameter in a
  rendered module is documented in an `Args:` (or `Attributes:`)
  section.
- `check_rosters.py` — the extras tables in [README.md](README.md) and
  [docs/getting-started.md](docs/getting-started.md) match
  `pyproject.toml`;
  [docs/user-guide/providers.md](docs/user-guide/providers.md) names
  every registered provider alias; the **sklearn estimators** list in
  [docs/index.md](docs/index.md) matches `eigenkit.__all__`; and the
  sentence in [docs/user-guide/layers.md](docs/user-guide/layers.md)
  that names the layers taking `provider=` matches their signatures.
- `check_doc_imports.py` — a page imports a name in the top-level
  `__all__` from `eigenkit`, and a domain package's own names from that
  package.
- `check_conventions.py` — page filenames under `docs/` use hyphens,
  prose uses British `-ise`, and every chapter under `docs/algorithms/`
  ends on `## Next`.
- `check_readme_cmds.py` — the commands in README.md and this file run,
  or the tool accepts their subcommand and flags. It works from lists
  of commands inside the script, so a command you add to either file
  goes into those lists too.

[tools/docs-checks/README.md](tools/docs-checks/README.md) describes
every script. Run one from the repository root:

```bash
uv run python tools/docs-checks/check_rosters.py
```

### Commit messages and releases

The project uses [python-semantic-release](https://python-semantic-release.readthedocs.io/),
so commit message style determines the next version number (when a release
is cut is up to a maintainer, see [Releasing](#releasing)). `main` is
squash-merged, so the commit semantic-release reads is the squash
commit, whose subject is the PR title: title the PR as a conventional
commit (the PR template asks for this too).

| Prefix | Bump |
|---|---|
| `feat: ...` | minor |
| `fix: ...`, `perf: ...` | patch |
| `feat!: ...` (or `BREAKING CHANGE:` in body) | major |
| `docs: ...`, `refactor: ...`, `chore: ...` | none |

Keep the subject line under 70 characters. Use the body for details.

### Releasing

No workflow runs on a push to `main`: CI runs on pull requests, and
merging never publishes a version. A maintainer cuts a release on
purpose:

1. Preview it: `uv run semantic-release --noop version` on an up-to-date
   `main`.
2. Actions → **release** → **Run workflow** on `main`. The workflow
   first reruns the tests (the suite without `slow` tests, on Python
   3.13 with the `all` and `perceval` extras, plus the core-only job),
   then semantic-release bumps `pyproject.toml` and `uv.lock`, updates
   `CHANGELOG.md`, tags `vX.Y.Z` and creates the GitHub release. The
   wheel and sdist go to PyPI through Trusted Publishing (the `pypi`
   environment), and the docs deploy to GitHub Pages at
   [docs.kit.eigenorbit.com](https://docs.kit.eigenorbit.com).

With no `feat`, `fix` or `perf` since the last tag the run cuts no
version and only redeploys the docs, which is also how a docs-only fix
goes live.

#### Release candidates

Ship a risky change (a widened dependency range, a new backend path,
anything hardware runs depend on) as a release candidate
first: tick **prerelease** in the run dialog. That cuts `X.Y.Z-rc.N`
(`X.Y.ZrcN` once installed), publishes it to PyPI and skips the docs deploy. pip and uv skip
pre-releases unless asked to, so a downstream range such as `>=1.2,<2`
stays on the last final release. Each further rc run bumps `N`; a normal
run turns `X.Y.Z-rc.N` into `X.Y.Z`.

### Pull request checklist

Before opening a PR, please confirm:

- [ ] `uv run pytest` passes.
- [ ] `uv run lint-imports` passes.
- [ ] `uv run mkdocs build --strict` passes.
- [ ] `uv run pre-commit run --all-files` is clean.
- [ ] New code has tests.
- [ ] User-facing behavior changes are reflected in `docs/`.
- [ ] If you added a backend or backend feature, you've updated the
  backend status table in [ARCHITECTURE.md](ARCHITECTURE.md) and
  [docs/user-guide/providers.md](docs/user-guide/providers.md).
- [ ] The PR title is a conventional commit (see above).

---

## How to extend the library

### Core and rings

[ARCHITECTURE.md](ARCHITECTURE.md) describes the library as a core with
rings around it, and that tells you where new code belongs:

- **The core**, `src/eigenkit/core/`: the IR, the provider protocol
  and registry, construction (`CircuitBuilder`, blocks, observables),
  the numpy `Circuit`, the variational spine, and the numpy reference
  simulator. It needs numpy and scipy and nothing else.
- **The rings**: the SDK backends (`core/backends/<name>_backend.py`,
  one extra each), the numpy domains (`optimization/`, `chemistry/`,
  `reservoir/`), `algorithms/`, `qml/` (scikit-learn), `benchmark/` and
  `datasets/`, `photonics/` (Perceval) and `torch/` (the `nn.Module`
  circuit, layers and hybrid estimators).

A ring depends on core; core never imports a ring at import time.
Rings register into core instead: backends into the provider registry,
the torch interface into `core/interfaces.py`. import-linter enforces
the boundaries: the contracts in `[tool.importlinter]` in
`pyproject.toml` forbid torch, the backend SDKs and scikit-learn in the
core modules and the numpy domains, torch and the SDKs in the qml tier,
and Perceval and every framework in the photonics compute layer. Run
them after changing imports:

```bash
uv run lint-imports
```

The core-only CI job runs the same check on an install with no extras.

### Add a new backend

A backend's code lives in one module,
`src/eigenkit/core/backends/<name>_backend.py`. The rest of the package
reaches it through lazy tables, so that `import eigenkit` never imports
its SDK; adding one touches that module, those tables, `pyproject.toml`
and the docs.

Steps:

1. **Create the file.** Three things go in it: `<Name>Config`,
   `<Name>Compiled`, `<Name>Backend`.
2. **`<Name>Config`** is a frozen dataclass holding device-level
   options (device name, shots, region, ...). It must expose
   `make_backend()` returning a `<Name>Backend` so users can pass the
   config directly to `Circuit(provider=<config>)`.
3. **`<Name>Compiled`** implements the `CompiledCircuit` protocol
   (`core/provider.py`):
   - `__init__(circuit: CircuitSpec, observables: Sequence[Observable], config)`
     does any one-time work (build a parametric circuit object,
     materialize Pauli operators, etc.).
   - `run(weights: np.ndarray, inputs: np.ndarray) -> np.ndarray` is
     required. Returns one expectation value per observable.
   - `jacobian(weights, inputs, *, need_input_grad=False)` is optional
     and returns `(weight_jacobian, input_jacobian_or_None)`. When it
     exists, `Circuit.jacobian` uses it instead of the core
     parameter-shift fallback. Set `supports_batched_jacobian = True`
     on the class if it also takes a 2-D `inputs` batch in one call.
   - `run_torch(weights, inputs) -> torch.Tensor` is optional; only
     implement it if your backend has native autograd. The presence of
     `run_torch` plus `gradient_method = "native"` on the backend
     activates the fast path.
   - `zero_state_probability(weights, inputs) -> float` is optional but
     strongly recommended; needed for `QuantumKernel` and any
     fidelity-based observable. `zero_state_probabilities(weights,
     inputs)` is its batched form; without it the core loops over the
     rows.
   - `statevector(weights, inputs)` and `probabilities(weights, inputs)`
     are optional. `Circuit.probabilities` derives the distribution from
     `statevector` when there is no `probabilities`.
   - `submit(weights, inputs) -> JobHandle` and `sample(weights, inputs)`
     are optional, for QPU/cloud backends: async dispatch and the raw
     shot histogram.
4. **`<Name>Backend`** has:
   - `name: str` — human-readable identifier.
   - `gradient_method: str` — `"adjoint"`, `"native"`,
     `"parameter_shift"`, or `"none"`. `"adjoint"` (numpy, qulacs,
     tensorcircuit) means the compiled handle ships its own `jacobian`;
     `"native"` means torch autograd through `run_torch`;
     `"parameter_shift"` leaves gradients to the core; `"none"` is for
     hardware that returns values only.
   - `__init__(config=None, name="<default>")` — accept `None` to use
     the default config.
   - `capabilities -> Capabilities` (property) — which gates, max
     qubits, batching support, shots support.
   - `compile(circuit, observables) -> <Name>Compiled` — calls into
     `<Name>Compiled.__init__`.
5. **Register it lazily** in `core/backends/__init__.py`, with
   `register_lazy("<name>", _loader("<name>_backend", "<Name>Backend"))`
   guarded on `importlib.util.find_spec("<sdk>")`: the name then exists
   once the SDK is installed, and the SDK is imported on first use.
   (Only the numpy simulator is registered eagerly.) Add an install
   hint for the name to `_EXTRA_HINT` in `core/provider.py`. A backend
   that needs a deliberate choice from the user stays opt-in instead:
   importable, but never registered by the package, so users call
   `register()` themselves. Two are opt-in today: CUDA-Q, because
   `cudaq.set_target` is process-global, and Qiskit Runtime, because it
   submits billable jobs to IBM hardware.
6. **Re-export lazily.** Add the backend class, its config and its job
   handle, if it has one, to:
   - `_LAZY_EXPORTS` in `src/eigenkit/core/backends/__init__.py`;
   - `__all__` and `_LAZY_BACKEND_EXPORTS` in
     `src/eigenkit/core/__init__.py`;
   - `_LAZY_CONFIGS` in `src/eigenkit/configs.py` (the config);
   - `_LAZY_NAMES` and `__all__` in `src/eigenkit/__init__.py`.

   Never import the module eagerly from any of these: that pulls the
   SDK into `import eigenkit`, and `tests/core/test_torch_free_import.py`
   and `test_importing_pulls_no_heavy_framework` in
   `tests/test_public_api_contract.py` fail.
7. **Add an extra** for the SDK to `[project.optional-dependencies]` in
   `pyproject.toml` and to `all`, then add a row for it to the extras
   tables in [README.md](README.md) and
   [docs/getting-started.md](docs/getting-started.md);
   `check_rosters.py` compares both tables and `all` with
   `pyproject.toml`. Commit the `uv.lock` that `uv sync` updates.
8. **Tests** in `tests/core/test_<name>_backend.py`:
   - Start the module with `pytest.importorskip("<sdk>")`, so a run
     without the extra skips it. CI's `test-missing-importorskip` job
     installs only the `torch` extra and errors on a test that imports
     an SDK without that guard.
   - Forward-pass parity against the numpy reference simulator
     (`provider="numpy"`) on a single observable, an all-Z list, and a
     Hamiltonian.
   - Gradient parity if `gradient_method` isn't `"none"`.
   - The auto-registration / opt-in registration behavior matches what
     was decided in step 5.
   - `zero_state_probability` parity (if implemented) — see
     `test_zero_state_probability.py`.
   - For QPU/cloud backends: `dry_run_info()`, `CostWarning` emission,
     `shots > 0` enforcement, and a `JobHandle` round-trip with a
     fake/local backend.
   - Add the backend to `BACKENDS` in
     `tests/core/test_declared_gates_are_lowerable.py` (registered name
     to SDK module), which runs every gate the backend declares.
9. **Docs**: add a row to the backend status table in
   [ARCHITECTURE.md](ARCHITECTURE.md) and a section in
   [docs/user-guide/providers.md](docs/user-guide/providers.md), which
   has to name the provider alias (`check_rosters.py`). Document every
   public parameter of the config and the backend in `Args:`
   (`check_func_params.py --strict`).

#### Optional dependencies

If your backend depends on a package that isn't a default dependency,
import it lazily inside a private `_require_<name>()` helper that
raises a clear `ImportError` with install instructions. Pattern:

```python
def _require_mybackend():
    try:
        import mybackend

        return mybackend
    except ImportError as e:
        raise ImportError(
            "mybackend is not installed. Install it via `uv sync --extra mybackend`."
        ) from e
```

Call `_require_mybackend()` at the top of `<Name>Compiled.__init__`
and at any other entry point that touches the SDK. The rest of the
library still imports cleanly when the package is missing.

#### Parameter expressions

Your backend's gate-application code has to handle every node in the
parameter-expression tree: `Constant`, `WeightRef`, `InputRef`,
`Scale`, `Negate`, `Sum`, and `Multiply`. If your target SDK can't
express one of them (CUDA-Q's `QuakeValue` doesn't support
`Sum`/`Multiply` over symbolic refs), raise `NotImplementedError` with
a message that points at a workaround.

### Add a new block

Blocks compile down to gate sequences, so they live in the core
(`core/blocks.py`), under the import-linter contract. They're
provider-agnostic and never import torch or a backend SDK; when one
needs numpy or scipy, it imports them inside the function that uses
them, as the existing blocks do.

Steps:

1. **Define a `@dataclass`** with the block's configuration fields.
   Use `__post_init__` for validation.
2. **Implement `required_qubits() -> int`** — return the minimum number
   of qubits the block needs. Variational-style blocks that fill
   whatever circuit they get can return `1`.
3. **Implement `build(n_qubits: int) -> IRFragment`** — use
   `CircuitBuilder` for readability, then return its gates and
   weight/input counts as an `IRFragment`. Local `WeightRef` and
   `InputRef` indices start at 0; the assembler relocates them into
   the global frame.
4. **Re-export** from `src/eigenkit/core/__init__.py` and
   `src/eigenkit/configs.py`. The `Encoding` /
   `Variational` / `StrongVariational` / `Reuploading` aliases in
   `configs.py` are how users typically import blocks.
5. **Tests** in `tests/core/test_blocks.py`:
   - Shape: gates, n_weights, n_inputs match expectation.
   - Local refs are correctly relocated by `assemble`.
   - End-to-end `Circuit(...)(x)` round-trip on at least one backend.

### Add a new gate

Gates touch every backend, so adding one is a multi-file change.

Steps:

1. **`src/eigenkit/core/ir.py`** — add the name to the `GateName`
   `Literal` and to one of the classification sets:
   `SINGLE_QUBIT_ROTATIONS`, `TWO_QUBIT_ROTATIONS`,
   `NON_PARAMETRIC_GATES` or `DATA_GATES`. `ALL_GATES` is their union,
   so it picks the gate up. Two more tables apply to some gates:
   `_ADJOINT_NAME` maps a non-parametric gate that is not its own
   inverse to its inverse (`S` and `Sdg`), and `MULTI_FREQUENCY_GATES`
   lists the gates whose angle carries more than one frequency
   (`CRY`), for which parameter-shift uses the four-term rule.
2. **`src/eigenkit/core/builder.py`** — add a method on
   `CircuitBuilder` that appends the gate via `_append`.
3. **Backends.** `validate_circuit()` in `core/provider.py` only checks
   gate names against `Capabilities.supported_gates`, and raises
   `ValueError` when the `Circuit` is constructed. Most backends derive
   that set from `ALL_GATES`, so they claim the new gate as soon as
   step 1 lands: numpy, pennylane, qiskit, qiskit-aer and qulacs
   declare `ALL_GATES` itself, and braket, iqm, qiskit-runtime and
   cudaquantum declare `ALL_GATES` minus what they cannot run. Each of
   these has to lower the gate in its gate-application code, or
   subtract it from `supported_gates` the way `cudaquantum_backend.py`
   does. qsim and tensorcircuit list their gates explicitly and change
   only if they should run it. The qiskit lowering,
   `_apply_gate_symbolic` in `qiskit_backend.py`, also serves
   qiskit-aer, qiskit-runtime and iqm.
4. **`adjoint()`** in `core/ir.py` — rotations negate their angles and
   non-parametric gates go through `_ADJOINT_NAME`, so a gate that fits
   one of the sets needs nothing here; anything else needs its own
   branch.
5. **Drawers** — `core/draw.py` (text) and `core/draw_mpl.py`
   (matplotlib). `draw_mpl` raises `NotImplementedError` for a gate it
   has no rule for, and `draw.py` gives every qubit of an unknown gate
   the same label, which is wrong for a controlled gate.
6. **Tests**:
   - Add the gate to `GATE_FIXTURES` in
     `tests/core/test_declared_gates_are_lowerable.py` and
     `tests/core/test_drawers_cover_every_gate.py`. Both assert
     `set(GATE_FIXTURES) == set(ALL_GATES)`, so they fail until you do;
     the first then runs the gate on every installed backend that
     declares it.
   - `tests/core/test_builder.py`: `CircuitBuilder` produces a gate
     with the right shape.
   - `tests/core/test_backends.py`: each backend that runs the gate
     matches the numpy simulator on a small reference circuit.
   - A backend that subtracts the gate rejects a circuit using it with
     a `ValueError` naming the gate.

### Add a new observable type

The current observable model is real-coefficient sums of Pauli
products. To add other observables (Hermitian matrices, fermionic
operators, ...):

1. **Extend `Observable`** in `core/ir.py`. The current shape is
   `Observable.terms: tuple[PauliTerm, ...]`. Consider a sum type or
   a `Generic` parameterization rather than overloading `terms`.
2. **Each backend's observable translation** must handle the new
   shape:
   - numpy: `NumpyCompiled._accumulate_observable`;
   - pennylane: `_observable_to_pennylane`;
   - qiskit: `_observable_to_qiskit`, which qiskit-aer and
     qiskit-runtime (under `primitive="estimator"`) reuse;
   - braket: `_pauli_term_to_braket`, called per term from
     `BraketCompiled.__init__`;
   - cudaquantum: `_observable_to_cudaq`;
   - qulacs: inline in `QulacsCompiled.__init__`;
   - qsim: `QsimCompiled._obs_to_paulisum`;
   - tensorcircuit: inline in `TensorCircuitCompiled.__init__`.

   The counts-based backends (`iqm_backend.py`, and
   `qiskit_runtime_backend.py` under `primitive="sampler"`) translate
   through `_sampling.plan_measurements` instead, so that is the one to
   extend for them.
3. **`PauliExpr`** in `observables.py` is independent — only adapt it
   if you want construction sugar (e.g. `Hermitian(...)` factory
   functions) for the new type.
4. **Tests** in `tests/core/test_observables.py`: construction,
   normalization (if applicable), parity across backends.

### Add a new layer

Layers are user-facing `nn.Module`s in the torch ring, at
`src/eigenkit/torch/layers/`. They're thin wrappers around
`Circuit`, so the work is mostly API design.

Steps:

1. **Pick the pattern.** Each existing layer answers a different
   question — `Basic` for the canonical "one Encoding + one
   Variational + measurement" shape, `Reupload` for standard data
   reuploading, `Ensemble` for N independent circuits, etc. Make sure
   your new layer covers a pattern that none of these expresses.
2. **Create `src/eigenkit/torch/layers/<name>.py`** with an
   `nn.Module` subclass.
3. **Accept the same `provider=` argument as `Circuit`** — string,
   config, or Backend instance. Forward it directly to the inner
   `Circuit(...)`.
4. **Expose `in_features` and `out_features`** as attributes so the
   layer composes cleanly with `nn.Linear` and friends.
5. **Re-export** from `src/eigenkit/torch/layers/__init__.py`, and
   lazily from `src/eigenkit/__init__.py`: add the name to
   `_LAZY_NAMES` and `__all__` there. An eager import would pull torch
   into `import eigenkit`.
6. **Tests** in `tests/layers/test_<name>.py`:
   - Forward-pass shape on 1-D and 2-D inputs.
   - `nn.Sequential` integration.
   - Provider polymorphism (string / config / backend instance all
     work).
   - Gradient flows end-to-end.
7. **Docs**: section in
   [docs/user-guide/layers.md](docs/user-guide/layers.md) and a
   matching example in [docs/examples/](docs/examples/) if the layer
   is non-obvious. Near its top, that page names in one sentence the
   layers that take `provider=`; add yours there (`check_rosters.py`
   compares the sentence with the signatures).

### Add a new algorithm

The sklearn-compatible estimators are split across two rings: the shared
base and the torch-free kernel estimators live in
`src/eigenkit/qml/`, and the ones that train a torch model live
in `src/eigenkit/torch/estimators/`. Both follow the standard
`fit(X, y)` / `predict(X)` / `score(X, y)` contract.
`src/eigenkit/algorithms/` holds the single-call solvers `HHL` and
`IQAE`, which inherit `BaseQuantumSolver` from `core/solver.py` and
have no `fit`, and `PortfolioRiskEstimator`, a registered
`BaseQuantumEstimator` with `fit()` that is built on `IQAE`.

Steps:

1. **Subclass `BaseQuantumClassifier` or `BaseQuantumRegressor`** from
   `qml/_base.py` and decorate the class with `@register_estimator`
   from `qml/_registry.py`. `load_model()` and
   `BaseQuantumEstimator.load()` find the class of a saved model in
   that registry, and `tests/algorithms/test_estimator_family_conformance.py`
   runs the shared contract against every registered torch-ring
   estimator. The base class accepts a `provider=` argument and a
   `random_state=`; reuse them rather than rolling your own.
2. **`__init__`**: store hyperparameters as instance attributes
   exactly as received. sklearn's `clone()` and `get_params()` rely on
   `__init__` arguments matching attribute names.
3. **`fit(X, y)`**: build the model, train it, return `self`. Public
   fitted attributes (`classes_`, `n_features_in_`, `train_losses_`)
   end in `_`, the sklearn convention; fitted internals that users do
   not touch (`_model`, `_svc`) start with `_`.
4. **`predict(X)`**: call `check_is_fitted(self, "<internal attribute>")`
   first (the torch estimators check `"_model"`), then dispatch.
5. **Save and load**: override `_collect_state()` to return the
   trained state and `_restore_state(state)` to rebuild from it. The
   base versions save nothing, so without them a saved model loads
   unfitted.
6. **For training-loop estimators** (HQNC, HQNR, ...), reuse
   `torch/estimators/_training.py`: `make_dataloader`, `make_optimizer`,
   `train_loop`, `predict_loop`, `validate_optimizer_name`, and
   `validate_regression_loss_config` cover the common cases.
7. **For kernel-based estimators**, build a `QuantumKernel` once at
   `fit` time and delegate to a sklearn precomputed-kernel estimator
   (`SVC(kernel="precomputed")`, `SVR(kernel="precomputed")`).
8. **Re-export** from the ring's own `__init__.py`
   (`src/eigenkit/qml/` or
   `src/eigenkit/torch/estimators/`) and lazily from
   `src/eigenkit/__init__.py` (`_LAZY_NAMES` and `__all__`; an eager
   import would pull torch or scikit-learn into `import eigenkit`).
   Add the estimator to the **sklearn estimators** list in
   [docs/index.md](docs/index.md) too; `check_rosters.py` compares it
   with `__all__`.
9. **Tests** in `tests/algorithms/test_<name>.py`:
   - End-to-end `fit` / `predict` / `score` on a small built-in
     dataset.
   - sklearn `clone()` round-trip.
   - `cross_val_score` runs without errors.
   - Provider polymorphism works.
10. **Docs**: section in
    [docs/user-guide/estimators.md](docs/user-guide/estimators.md) and,
    for non-trivial new estimators, a worked example in
    [docs/examples/](docs/examples/).

---

## Adding to the dataset registry

The dataset registry in `src/eigenkit/datasets/` exposes
loaders, synthetic generators, and a small EDA report system. Every
dataset is a `DatasetInfo` entry in `REGISTRY` in
`datasets/_registry.py`, and `load_dataset("name")` loads it according
to its `source`:

1. **`source="sklearn"`** — name the `sklearn.datasets` function in
   `sklearn_loader`. The registry entry is all it needs.
2. **`source="url"`** — give the `url` of a CSV or NPZ file and its
   `sha256`; the download is checked against the hash, and
   `tests/datasets/test_registry.py` requires one on every URL entry.
3. **`source="synthetic"`** — write the generator in
   `datasets/_synthetic.py` (it takes `random_state`) and add it to
   `GENERATORS` there, under the name the entry gives in
   `synthetic_generator`.

Tests go in `tests/datasets/`. The existing `test_loaders.py`,
`test_synthetic.py`, and `test_registry.py` files have the patterns.

---

## Asking questions and reporting bugs

Open an issue on GitHub. For bug reports, please include:

- The exact `provider=` you used (string, config, or Backend).
- A minimal reproducing snippet (5-20 lines is plenty).
- Expected vs. actual behavior.
- Python version and `pip list` output for the relevant SDKs.

Do not report a security vulnerability in a public issue; follow
[SECURITY.md](SECURITY.md) instead.

For design discussions or feature proposals, open an issue first
before sending a PR — it saves rework if the change doesn't fit the
core + rings model.
