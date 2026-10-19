# Architecture

This document explains how the codebase is organized and how to read it.
It has two parts:

1. **A guided reading order** — start with the foundation, work up to the
   user-facing API. Each step is a self-contained 10-30 minute read.
2. **The architecture** — how the pieces fit, why we chose this shape, how
   to extend it.

If you are new to the project, read in order. If you are looking for a
specific thing, jump to the *File Map* at the bottom.

---

## Part 1 — How to Read the Code

The library is a **lightweight numpy/scipy core** surrounded by **optional
rings**, each ignorant of the ones outside it:

```
   core/ — compute layer (pure numpy/scipy, no framework)
     • IR (pure data) → providers (executors) → construction → numpy Circuit
     • variational spine (minimize_expectation)
     • backends/numpy_backend.py (the built-in reference simulator)
   │
   ├── interfaces (core/interfaces.py) — interface registry; the torch adapter
   │      registers here lazily, so core never imports it (JAX is planned)
   ├── backends   — pennylane / qiskit / braket / iqm / cudaq / … adapters (extras)
   ├── domains    — optimization/ (QAOA), chemistry/ (VQE, mappings),
   │                reservoir/ (quantum reservoir computing)         [numpy]
   ├── algorithms — algorithms/ (HHL, IQAE, PortfolioRiskEstimator)  [sklearn]
   ├── qml        — kernels, SVMs, the sklearn estimator base        [sklearn]
   ├── tooling    — benchmark/ (comparison harness), datasets/       [sklearn]
   ├── photonics  — photonics/ (linear optics, Perceval executors)   [perceval]
   └── torch      — the nn.Module circuit, layers, hybrid estimators [torch]
```

`import eigenkit` pulls only the core, which needs numpy and scipy (pydantic,
the package's third dependency, is imported only by `datasets/`);
everything else is imported lazily when first used, and ring boundaries are
enforced in CI by import-linter. The top-level `Circuit(...)` defaults
to the **numpy** interface on the **numpy** provider, so a bare install builds
*and runs* circuits with no extra; torch and the backend SDKs are opt-in.

Read bottom-up: each layer makes sense once you understand the one below.

### Step 1 — IR: the data foundation

📄 [src/eigenkit/core/ir.py](src/eigenkit/core/ir.py)
✅ [tests/core/test_ir.py](tests/core/test_ir.py)

Pure data classes. No torch, no qiskit, no pennylane. A circuit is described
by `CircuitSpec(n_qubits, n_weights, n_inputs, gates)` where each `Gate` has a
name, target qubits, and a tuple of `Parameter` expressions.

Read in this order:
1. `Constant`, `WeightRef`, `InputRef` — three kinds of leaf parameter.
2. `Scale`, `Negate`, `Sum`, `Multiply` — expression nodes that compose with leaves. `Multiply(a, b)` is the only one that admits two symbolic operands; it powers patterns like Schuld 2021 trainable feature scaling (`Reuploading(trainable_scale=True)`). Every node also supports Python arithmetic (`_ParamOps`), so `2.0 * (pi - x[0])` builds the same nodes as writing them out by hand.
3. `resolve(parameter, weights, inputs)` — recursive evaluation.
4. Gradient analysis helpers — `find_parameter_shift_unsafe_gates`, `collect_weight_dependencies`, `chain_factor`, `shift_gate_angle`. Together they implement chain-rule-aware parameter-shift: when a gate angle wraps a weight in `Scale`/`Multiply`/`Sum`/`Negate`, naive shifting of the weight by π/2 doesn't shift the angle by π/2 (chain-rule effect). Instead, `core.Circuit._chain_rule_jacobians` shifts each parametric gate's angle in a fresh IR — by ±π/2, and also by ±3π/2 for `CRY` (the four-term rule; `MULTI_FREQUENCY_GATES`) — caches each compile per `(gate_index, shift_value)`, and accumulates `(∂θ/∂w_i) · (∂<O>/∂θ)` into the per-weight gradient. Dispatch is automatic: safe-only circuits use the fast naive path; circuits with unsafe gates use the chain-rule path. All four backend categories (PennyLane native, numpy adjoint, Qiskit parameter-shift, Braket parameter-shift) agree on `Reuploading(trainable_scale=True)` gradients to machine precision.
5. `Gate`, `CircuitSpec` — the circuit representation.
6. `PauliTerm`, `Observable`, `Observable.from_string` — what to measure.
7. `adjoint(circuit)` — example provider-agnostic transform.

> **Why a Parameter expression tree?** Lets blocks emit symbolic angles like
> `2π · x_0` (Scale) or `w_0 + x_0` (Sum) without forcing users to pre-scale
> their data, and lets `adjoint` work on trainable parameters not just
> constants. See [Parameter expressions](#parameter-expressions) under Key
> design decisions below.

After reading, you should be able to construct an IR by hand:

```python
from eigenkit.core.ir import CircuitSpec, Gate, WeightRef

ir = CircuitSpec(
    n_qubits=2,
    n_weights=1,
    n_inputs=0,
    gates=(Gate("RX", (0,), (WeightRef(0),)), Gate("CNOT", (0, 1))),
)
```

### Step 2 — Provider protocol

📄 [src/eigenkit/core/provider.py](src/eigenkit/core/provider.py)
✅ [tests/core/test_provider.py](tests/core/test_provider.py)

The contract every backend implements. Three things to read:
1. `Capabilities` — what a provider declares it can do.
2. `CompiledCircuit` protocol — the handle returned by `provider.compile()`.
3. `Provider` protocol — the backend interface.

Then the registry + dispatch:
4. `register(name, config_or_backend)` — name a backend or config for reuse.
5. `resolve_provider(spec)` — turn a string / config / backend instance into
   a `Provider`. This is the polymorphic dispatch the user-facing
   `Circuit(provider=...)` argument relies on.

Finally, the gradient fallback:
6. `parameter_shift_jacobian(...)` — the exact parameter-shift rule,
   `(f(θ+π/2) - f(θ-π/2)) / 2`, and the four-term rule (shifts ±π/2 and ±3π/2)
   for weights feeding a multi-frequency gate (`CRY`, listed in
   `ir.MULTI_FREQUENCY_GATES`). Used by backends that don't ship their own
   `jacobian`.

A compiled handle may also implement an optional `zero_state_probability(weights,
inputs) -> float` method; it is not declared on the `CompiledCircuit` Protocol,
and `Circuit` finds it by duck typing. It returns `|<0…0|ψ>|²` — the all-zeros amplitude
squared — without needing a Pauli observable. Backends implement it via
state-vector access (`qml.probs[0]`, `Statevector.data[0]`,
`bk_circuit.probability(...)` or `cudaq.get_state(...)[0]`). This is the primitive that powers fidelity-based
observables — most importantly, the quantum kernel (see `qml/_kernels.py`).

> **Why split compile from run?** Building a parametric `QuantumCircuit` or
> a PennyLane device costs orders of magnitude more than evaluating it. The
> handle returned by `compile()` is reused for every forward pass. On Qiskit
> Runtime, and on IQM in its default `transpile_mode="parametric"`, compile is
> also where the circuit is transpiled for the device, so that happens once
> too.

### Step 3 — Backends: numpy (default), PennyLane, Qiskit, Qiskit Aer, Braket, CudaQuantum, TensorCircuit, qsim, Qulacs, Qiskit Runtime, IQM

📄 [src/eigenkit/core/backends/numpy_backend.py](src/eigenkit/core/backends/numpy_backend.py)
📄 [src/eigenkit/core/backends/pennylane_backend.py](src/eigenkit/core/backends/pennylane_backend.py)
📄 [src/eigenkit/core/backends/qiskit_backend.py](src/eigenkit/core/backends/qiskit_backend.py)
📄 [src/eigenkit/core/backends/qiskit_aer_backend.py](src/eigenkit/core/backends/qiskit_aer_backend.py)
📄 [src/eigenkit/core/backends/braket_backend.py](src/eigenkit/core/backends/braket_backend.py)
📄 [src/eigenkit/core/backends/qiskit_runtime_backend.py](src/eigenkit/core/backends/qiskit_runtime_backend.py)
📄 [src/eigenkit/core/backends/iqm_backend.py](src/eigenkit/core/backends/iqm_backend.py)
📄 [src/eigenkit/core/backends/_sampling.py](src/eigenkit/core/backends/_sampling.py)
📄 [src/eigenkit/core/backends/cudaquantum_backend.py](src/eigenkit/core/backends/cudaquantum_backend.py)
📄 [src/eigenkit/core/backends/tensorcircuit_backend.py](src/eigenkit/core/backends/tensorcircuit_backend.py)
📄 [src/eigenkit/core/backends/qsim_backend.py](src/eigenkit/core/backends/qsim_backend.py)
📄 [src/eigenkit/core/backends/qulacs_backend.py](src/eigenkit/core/backends/qulacs_backend.py)
✅ [tests/core/test_backends.py](tests/core/test_backends.py), [tests/core/test_braket_backend.py](tests/core/test_braket_backend.py), [tests/core/test_braket_qpu.py](tests/core/test_braket_qpu.py), [tests/core/test_qiskit_runtime_backend.py](tests/core/test_qiskit_runtime_backend.py), [tests/core/test_cudaquantum_backend.py](tests/core/test_cudaquantum_backend.py), [tests/core/test_qulacs_backend.py](tests/core/test_qulacs_backend.py), [tests/core/test_qiskit_aer_backend.py](tests/core/test_qiskit_aer_backend.py), [tests/core/test_tensorcircuit_backend.py](tests/core/test_tensorcircuit_backend.py), [tests/core/test_qsim_backend.py](tests/core/test_qsim_backend.py), [tests/core/test_iqm_backend.py](tests/core/test_iqm_backend.py), [tests/core/test_sampling.py](tests/core/test_sampling.py)

Start with **`numpy_backend.py`** — it's the default (`provider="numpy"`) and
the easiest to read: a pure-numpy statevector simulator with no SDK. It builds
the gate matrices, applies them with `np.tensordot` (CNOT is an `np.flip`
amplitude swap; `np.einsum` only for per-sample matrices on the batched
path), reads expectation values
directly, and implements `jacobian` via the **analytic adjoint** method (a
reverse pass that is exact and `O(1)` in the parameter count). It's the
reference every other backend is checked against, and it advertises
`gradient_method="adjoint"`. The remaining backends translate IR to native SDK
objects; they all implement the same Provider protocol, with these differences:

| | PennyLane | Qiskit | Braket (local/cloud) | Braket (QPU) | Qiskit Runtime | IQM | CudaQuantum | Qulacs | EigenOrbit Router² |
|---|---|---|---|---|---|---|---|---|---|
| `gradient_method` | `"native"` | `"parameter_shift"` | `"parameter_shift"` | `"none"`¹ | `"none"`¹ | `"none"`¹ | `"parameter_shift"` | `"adjoint"` | `"none"` on a QPU, `"parameter_shift"` on a simulator |
| Symbolic params | resolved per-call | `qiskit.circuit.Parameter` | `FreeParameter` | `FreeParameter` | `qiskit.circuit.Parameter` (transpiled) | `qiskit.circuit.Parameter` (transpiled) | `make_kernel(list)` | `set_parameter` per-call on a `ParametricQuantumCircuit` | – (the circuit goes out as wire JSON; weights per call) |
| Compile output | device + qnodes + Pauli ops | parametric QuantumCircuit + SparsePauliOp | parametric Circuit + deduplicated result_types | same + lazy `AwsDevice` | transpiled circuit + layout-mapped SparsePauliOp (estimator) or a qubit-wise-commuting measurement plan (sampler) | parametric QuantumCircuit + one transpiled measured circuit per qubit-wise-commuting group (default `transpile_mode="parametric"`) | `make_kernel` + cudaq spin-op hamiltonians | `ParametricQuantumCircuit` + qulacs `Observable`s | wire JSON of the circuit and observables |
| Sync `run()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Async `submit()` | – | – | ✓ (returns `BraketJobHandle`) | ✓ | ✓ (returns `QiskitRuntimeJobHandle`) | ✓ (returns `IQMJobHandle`) | – | – | ✓ (returns `RouterJobHandle`) |
| `dry_run_info()` | – | – | ✓ | ✓ | ✓ | ✓ | – | – | ✓ |
| Cost guardrails | – | – | `CostWarning` on cloud compile | + `shots > 0` enforced + `gradient_method="none"`¹ | `CostWarning` on real-hardware compile + `shots > 0` enforced | `CostWarning` on real-hardware compile + `shots > 0` enforced | – | – | `CostWarning` with the router's price estimate on compile |
| Auto-registered | ✓ "pennylane" | ✓ "qiskit" | ✓ "braket" | ✓ "braket" (config flips behaviour) | ✗ opt-in | ✓ "iqm" (when extra installed) | ✗ opt-in (extra dep) | ✓ "qulacs" (when extra installed) | ✓ "router" (when extra installed) |

¹ `"parameter_shift"` when the device config sets `enable_gradients=True`; compiling against a paid device then raises a second `CostWarning` quoting the job count of one backward pass. Every gradient path (torch backward, `jacobian()`, the variational spine) goes through `Circuit._gradient_jacobians`, which refuses a `"none"` backend; torch's `_NoGradFn` and `variational()` refuse earlier, with messages specific to their path.

² The `router` extra installs eigenorbit-router-client, which ships this backend and `RouterConfig`; eigenkit registers `"router"` lazily when that package is importable. It runs circuits on the EigenOrbit Router, a hosted service that reaches real QPUs with one EigenOrbit API key.

Async `submit()` covers Braket, Qiskit Runtime, IQM and EigenOrbit Router.

Read PennyLane and Qiskit first (the simpler pair); Braket and CudaQuantum
follow the same shape with backend-specific quirks.

For each backend, in order:
1. `<Name>Config` — what the user can tune. Note the `make_backend()`
   factory; this is what enables `Circuit(provider=Config(...))`.
2. `_observable_to_<backend>` (or `_pauli_term_to_<backend>` for Braket) —
   translate IR observables to native Pauli ops.
3. `_apply_gate*` and `_resolve_symbolic` — translate IR gates and parameter
   expressions to native primitives.
4. `<Backend>Compiled.__init__` — the cached compile step.
5. `<Backend>Compiled.run` (and `run_torch` on PennyLane) — the per-call
   evaluation.
6. `<Backend>Backend` — capabilities + the `compile()` factory.

Backend-specific details to notice:
- **numpy** has an exact analytic-adjoint `jacobian` (no torch, no SDK) — the
  gradient path the numpy interface and the variational spine use by default.
- **PennyLane** is the only one with `run_torch` (native autograd for the torch
  interface) and its own diff-engine `jacobian`. Qulacs and TensorCircuit also
  ship their own `jacobian` (adjoint via `backprop`, and `jax.jacrev`); Qiskit,
  Qiskit Aer, Braket, CudaQuantum and qsim rely on the IR-layer
  parameter-shift fallback.
- **Qiskit** uses `qiskit.circuit.Parameter` and `assign_parameters` — the
  cleanest parametric path.
- **Qiskit Aer** reuses the plain Qiskit backend's IR translation verbatim
  (gate lowering, symbolic `Parameter` binding, `SparsePauliOp` observables)
  and swaps only the execution primitive: expectation values come from Aer's
  `save_expectation_value` rather than `Statevector.from_instruction`. Aer
  already ships in the `qiskit` extra, so it adds no new dependency. Two cores
  are exposed as two provider ids — `qiskit-aer` (dense statevector, ~1e-12
  against numpy) and `qiskit-aer-mps` (matrix product state, exact at
  unbounded bond dimension, ~1e-9). The MPS core rejects `cry` rather than decomposing
  it, so MPS-bound circuits get a basis translation first.
- **Braket** uses `FreeParameter`, but Braket *deduplicates* identical
  result-types on the SDK side, so we mirror that with a per-Pauli-term
  index dictionary in our aggregation plan. See the comment in
  `BraketCompiled.__init__`.
- **CudaQuantum** uses `cudaq.make_kernel` and supports a restricted
  parameter expression set (no `Sum` or `Multiply` over symbolic refs —
  cudaq's QuakeValue arithmetic is more limited than Qiskit/Braket). The module
  imports cudaq lazily so the rest of the library still works without it.
- **Braket QPU vs simulator** is selected via the `device_arn` field in
  `BraketConfig`. The `is_qpu` predicate (substring `/qpu/` in the ARN)
  drives three behavioural switches: `gradient_method` flips to `"none"`
  (unless `enable_gradients=True`),
  `shots > 0` is enforced, and a single `CostWarning` fires at compile
  time. The actual `AwsDevice` is constructed lazily on first `run()` /
  `submit()` so `dry_run_info()` works offline and tests don't hit AWS.
  `submit()` also accepts `method=` (`run`, `sample`, `probabilities` or
  `zero_state_probability`, listed in `BraketCompiled.submit_methods`) and AWS `tags=`, and
  `attach(task_id)` rebuilds a `BraketJobHandle` from a task ARN, so a job
  service can recover after a crash.
- **Wire codecs** in `core/wire.py` are the public JSON form of a
  `CircuitSpec` and its observables. They reuse the `.eigenkit` encoding.
- **Qiskit Runtime** targets real IBM hardware (Eagle / Heron) and noise-
  aware FakeBackends from `qiskit_ibm_runtime.fake_provider`.
  Authentication chain: explicit `token` > `QISKIT_IBM_TOKEN` env var >
  saved-account file. For tests, pass a
  `FakeBackendV2` instance via `backend_instance` and skip auth entirely.
  Compile-time `qiskit.transpile` lowers IR gates to the target backend's
  native basis (e.g. `rz, sx, ecr`); Pauli operators are layout-mapped via
  `apply_layout`. With `primitive="estimator"` (the default), per-call
  dispatch uses `EstimatorV2` PUBs, one per observable; `primitive="sampler"`
  runs `SamplerV2` over qubit-wise-commuting measurement groups and reduces
  the expectation values from counts. `submit()` returns a `QiskitRuntimeJobHandle` wrapping a
  `RuntimeJob`. Like Braket's, it accepts `method=` (`run`, `sample` or
  `probabilities`, listed in `QiskitRuntimeCompiled.submit_methods`) and IBM
  job `tags=`, and `attach(job_id)` rebuilds the handle through the backend's
  `QiskitRuntimeService`, the same recovery hooks for IBM.
- **TensorCircuit** is tensor-network simulation via tensorcircuit-ng on the
  JAX backend. Expectation values are exact contractions; the parameter batch
  is vectorised with `jax.vmap` and the forward/gradient pair is `jax.jit`-
  compiled, with gradients from native reverse-mode autodiff (`jax.jacrev`) —
  so the numpy variational spine and the torch interface both get them with no
  parameter-shift. Exactness means it buys a constant factor, not
  sub-exponential reach, and the one-time jit compile makes it a fit for
  reused / amortised workloads rather than one-shot runs.
- **qsim** is Google's Schrödinger statevector simulator, reached through the
  `qsimcirq` Cirq bridge: the IR is lowered once to a parametrised
  `cirq.Circuit` with sympy symbols, and each call resolves and runs. It is
  **single-precision (fp32)** — the pip wheel ships no fp64 build — so values
  carry ~1e-6 accumulation error and it does not pass a 1e-9 parity gate; the
  parity tests check it at 1e-4. The data-carrying primitives
  (`AmplitudeEncoding`, `QubitUnitary`, `ControlledQubitUnitary`) are rejected
  up front.
- **Qulacs** is a fast C++ statevector simulator with a *native adjoint
  gradient* (`ParametricQuantumCircuit.backprop`), so it advertises
  `gradient_method="adjoint"` (like the numpy sim) rather than parameter-shift.
  Compile builds one `ParametricQuantumCircuit`; trainable rotations become
  parametric slots bound per call with `set_parameter`, and `jacobian` reads
  every slot gradient from one `backprop` sweep, scattering onto weights/inputs
  with the same IR chain factors the numpy backend uses (so `Scale`/`Multiply`
  angles differentiate correctly too). Two conventions are reconciled: qulacs
  rotations are `exp(+iθP/2)` (angles are negated on the way in), and qulacs is
  qubit-0-LSB (statevector I/O and multi-qubit matrices are bit-reversed, exactly
  as the Qiskit backend does). Gates without a qulacs parametric primitive are
  decomposed onto parametric rotations to stay differentiable: `PhaseShift`→`RZ`,
  `CRY`→`RY·CX·RY·CX`, `CPHASE`→three `RZ`s + `CX`s. `adjoint(AmplitudeEncoding)`
  is unsupported (state prep is not invertible as a gate), matching Qiskit. The
  module imports lazily and lazy-registers as `"qulacs"` when the extra is
  installed.

After reading, you should understand exactly what work is amortized at
compile time and what work runs per call.

### Step 4 — Observables

📄 [src/eigenkit/core/observables.py](src/eigenkit/core/observables.py)
✅ [tests/core/test_observables.py](tests/core/test_observables.py)

Three entry points to building observables, each suited to a different
user persona:

1. **`Observable.z(0)`** — typed constructor. Newbies use this.
2. **`Observable.from_string("0.5 Z0 X1 + 0.3 Y2")`** — string DSL.
   Intermediate users.
3. **`Z(0) * X(1) + 0.5 * Y(2)`** — Pauli algebra via `PauliExpr`.
   Power users writing Hamiltonians.

The non-trivial part is `_normalize_paulis` — when you multiply `Z * Z` we
need to cancel them (P² = I). Different Paulis on the same qubit (X * Y)
require complex coefficients and we raise (out of scope).

### Step 5 — CircuitBuilder: imperative construction

📄 [src/eigenkit/core/builder.py](src/eigenkit/core/builder.py)
✅ [tests/core/test_builder.py](tests/core/test_builder.py)

Mutable, Qiskit-style API. `qc.rx(angle, qubit=0)`, `qc.cnot(0, 1)`, etc.
The two unique pieces:

- `add_weights(n)` and `add_inputs(n)` — declare the parameter frame, get
  back lists of `WeightRef` / `InputRef` to feed into gate methods.
- `to_block()` — wrap the current builder state as a `CustomBlock` so it
  can be composed into a larger block-based circuit.

The class is ~400 lines, mostly one short method per gate. Read end-to-end.

### Step 6 — Blocks: composable patterns

📄 [src/eigenkit/core/blocks.py](src/eigenkit/core/blocks.py)
✅ [tests/core/test_blocks.py](tests/core/test_blocks.py), [tests/core/test_reuploading_block.py](tests/core/test_reuploading_block.py)

Reusable circuit patterns that emit IR fragments. Read in this order:

1. `IRFragment` and the `Block` Protocol — the contract.
2. `AngleEncoding` — encodes data as rotations. Note how it uses
   `CircuitBuilder` internally for readability.
3. `BasicVariational` — RX/RY/RZ rotation per qubit + CNOT ladder.
4. `StrongVariational` — RZ-RY-RZ per qubit + CNOT ladder.
5. `Reuploading` — two strategies via one flag:
   - default: standard data reuploading (Pérez-Salinas 2020).
   - `trainable_scale=True`: per-(layer,feature) learnable scaling
     (Schuld 2021). Emits `Multiply(WeightRef, InputRef)` angles. The
     backends with their own `jacobian` (numpy, PennyLane, Qulacs,
     TensorCircuit) handle these directly; the parameter-shift backends
     (Qiskit, Braket, ...) automatically dispatch to the chain-rule jacobian
     on `core.Circuit`. All of them agree on gradients to machine precision.
6. `CustomBlock` — wraps a `CircuitBuilder` for use in block-style
   composition.
7. `AmplitudeEncoding`, `S0`, `QubitUnitary`, `Grover`,
   `ControlledUnitaryBlock`, `QFTBlock`, `QPEBlock`, `TrotterBlock`,
   `HamiltonianQPEBlock` — the state-preparation, oracle, QFT/QPE and
   Hamiltonian-simulation blocks that HHL and IQAE build on.
8. `_relocate_parameter` and `assemble` — the assembler. **This is the
   most important function in this file.** It walks blocks in order,
   collects gates, and offsets local `WeightRef`/`InputRef` indices into
   the global parameter frame so each block can use 0-indexed refs locally.

### Step 7 — `core/circuit.py`: the numpy circuit + interface dispatch

📄 [src/eigenkit/core/circuit.py](src/eigenkit/core/circuit.py)
✅ [tests/core/test_circuit.py](tests/core/test_circuit.py)

The framework-free class that ties the compute layer together. Read in order:

1. `_normalize_observables` — turns `None | Observable | list | PauliExpr`
   into `(list[Observable], scalar_flag)`. Handles the newbie default
   (None → all-Z).
2. `Circuit.__init__` — validates, compiles, stores the handle. Takes
   `weights` at call time; it does **not** own parameters.
3. `Circuit.__call__` / `_run_numpy` — the numpy forward pass.
4. `Circuit.jacobian(weights, inputs, *, input_grad=False)` — the **public
   gradient API**. Returns `d<O>/d(weights)` (shape `(n_weights,)` for a scalar
   observable, `(n_observables, n_weights)` for a vector; mirrors `__call__`),
   plus the input jacobian when `input_grad=True`. This is what makes the numpy
   interface differentiable without torch — `minimize_expectation` and any
   custom numpy training loop call it.
5. `_gradient_jacobians` (the engine behind `jacobian`) — prefers a backend's
   own `CompiledCircuit.jacobian` (the numpy sim's analytic adjoint, PennyLane's
   autodiff) and falls back to core parameter-shift only when absent;
   `_chain_rule_jacobians` handles weights wrapped in `Scale`/`Multiply`/etc.
6. The other measurement and inspection methods — `probabilities`, `sample`,
   `statevector(initial_state=)`, `zero_state_probability` /
   `zero_state_probabilities`, `submit`, `dry_run_info`, `resources`,
   `transpiled_circuit` — forward to the compiled handle's method of the same
   name and raise `NotImplementedError` (via `_require_backend`) when the
   backend has none; `draw` and `draw_mpl` render the `CircuitSpec` itself.
7. Interface dispatch: `interface="numpy"` is handled inline in `__call__`
   (numpy is not in the registry); `interface="torch"` delegates to the
   registered torch interface (`core/interfaces.py`), which lives in the
   torch ring. The
   torch `autograd.Function` wrappers and the `TorchCircuit` nn.Module are
   **not** in core — they sit in `eigenkit/torch/`, imported lazily
   so a numpy-only environment never touches torch.

### Step 8 — Top-level `Circuit`

📄 [src/eigenkit/circuit.py](src/eigenkit/circuit.py)
✅ [tests/test_top_level_circuit.py](tests/test_top_level_circuit.py)

The function users actually import. ~120 lines, most of them docstring. Read
end-to-end.

It dispatches on `interface` (PennyLane-style, explicit). The default
is `interface="numpy"`, `provider="numpy"`: it accepts a list of `Block`s or a
pre-built `CircuitSpec`, calls `assemble()` if needed, and returns the
framework-free `core.circuit.Circuit` (weights passed per call). Pass
`interface="torch"` for a `TorchCircuit` nn.Module that owns `weights` as a
`nn.Parameter` (from the torch ring), and `provider="pennylane"` / `"qiskit"` /
… for a faster or hardware backend.

### Step 9 — `configs.py`

📄 [src/eigenkit/configs.py](src/eigenkit/configs.py)

Just re-exports. `Encoding` is `AngleEncoding`. `Variational` is
`BasicVariational`. `StrongVariational` and `Reuploading` are themselves.
`NumpyConfig` is imported eagerly; every SDK config (`PennyLaneConfig`,
`QiskitConfig`, `BraketConfig`, `IQMConfig`, …) is resolved lazily from its
backend module on first access.

Friendly alias module so users can import everything from one place.

### Step 10 — Layers

📄 [src/eigenkit/torch/layers/basic.py](src/eigenkit/torch/layers/basic.py) — `Basic`, one Encoding + one Variational
📄 [src/eigenkit/torch/layers/reupload.py](src/eigenkit/torch/layers/reupload.py) — `Reupload`, data reuploading
📄 [src/eigenkit/torch/layers/ensemble.py](src/eigenkit/torch/layers/ensemble.py) — N independent circuits over input chunks
📄 [src/eigenkit/torch/layers/minmax.py](src/eigenkit/torch/layers/minmax.py) — `MinMaxNorm`, BatchNorm-style feature-wise [a, b] normalization
✅ [tests/layers/](tests/layers/)

The layers live in the **torch ring** (`eigenkit/torch/layers/`):
each is a thin nn.Module on top of a `TorchCircuit` (or, for `MinMaxNorm`,
purely classical). Recommended reading order: `basic.py` first (the
thinnest wrapper, one Encoding + one Variational), then `reupload.py`,
`ensemble.py`, and `minmax.py`.

---

## Part 2 — Architecture

### The core + rings model

```
┌───────────────────────────────────────────────────────────────────────┐
│ core/ — COMPUTE LAYER  (numpy + scipy only, no framework)             │
│                                                                       │
│   Data        • Gate, CircuitSpec                                     │
│               • Parameter (Constant/WeightRef/InputRef/Scale/         │
│                 Negate/Sum/Multiply); PauliTerm, Observable           │
│               • adjoint, resolve  (provider-free transforms)          │
│   Providers   • Provider protocol + Capabilities; registry            │
│               • CompiledCircuit handle (compile/run split)            │
│               • parameter_shift_jacobian (gradient floor)             │
│               • backends/numpy_backend.py (built-in reference sim)    │
│   Construct   • CircuitBuilder; Block, assemble; observable algebra   │
│   Run         • numpy Circuit; minimize_expectation (variational)     │
│   Interfaces  • core/interfaces.py — interface registry (torch        │
│                 registers from its ring, lazily; numpy is inline)     │
└───────────────────────────────────────────────────────────────────────┘
      ▲ optional rings, each depends on core, never the reverse:
      │
  backends/      pennylane · qiskit · braket · iqm · cudaquantum · …  (SDK extras)
  optimization/  QAOA, QUBO/Ising              ┐ numpy domains, framework-free
  chemistry/     VQE, JW/BK, molecules         │ (chemistry's PySCF path is
  reservoir/     quantum reservoir computing   ┘ the `chemistry` extra)
  algorithms/    HHL, IQAE, PortfolioRiskEstimator        (imports scikit-learn)
  qml/           QuantumKernel, QSVC/QSVR, sklearn base   (`qml` extra)
  benchmark/     comparison harness                       (`benchmark` extra)
  datasets/      dataset registry and loaders             (`datasets` extra)
  photonics/     linear-optics IR, Perceval executors     (`perceval` extra)
  torch/         TorchCircuit, layers, hybrid estimators  (`torch` extra)
```

Everything in `core/` except the SDK adapters in `core/backends/` is pure
numpy/scipy with no third-party framework import; import-linter enforces this
in CI for the core modules, `numpy_backend` and the numpy domain rings. The
photonics ring has its own contract: its IR, executor protocol and gradients
import neither perceval nor any framework. The rings depend on core, never the
reverse: the torch interface registers itself into `core/interfaces.py` lazily,
and the SDK backends register into the provider registry — so `import
eigenkit` stays lightweight and a numpy-only environment never loads
torch, pennylane, or sklearn.

### Data flow: what happens when a user calls a circuit

The default path is numpy: `Circuit(...)` returns the framework-free
`core.circuit.Circuit`, which takes weights per call.

```python
import numpy as np

from eigenkit import Circuit, Encoding, Variational, all_z

circuit = Circuit(  # interface="numpy", provider="numpy"
    blocks=[Encoding(in_features=4), Variational(depth=2)],
    observables=all_z(4),
)
weights = np.random.uniform(0, 2 * np.pi, circuit.spec.n_weights)
x = np.random.uniform(0, np.pi, (8, 4))  # batch of 8 samples
y = circuit(weights, x)
g = circuit.jacobian(weights, x[0])  # d<O>/d(weights), one 1-D sample
```

**At construction time:**

1. `Circuit` (top-level dispatcher) sees `interface="numpy"` and builds the
   inner `core.circuit.Circuit`.
2. `core.assemble(blocks)` walks the blocks, calls `block.build(n_qubits)` on
   each, collects `IRFragment`s, relocates local `WeightRef` / `InputRef`
   indices into the global frame, and returns one `CircuitSpec`.
3. The inner Circuit validates the spec against the provider's `Capabilities`
   (supported gates, max_qubits).
4. `provider.compile(spec, observables)` runs once. The numpy backend builds
   the gate matrices and observable terms; PennyLane builds a `qml.device` plus
   QNodes; Qiskit builds a parametric `QuantumCircuit` + `SparsePauliOp`. The
   compiled handle is stored. The numpy circuit owns **no** weights.

**Per forward call** — `circuit(weights, inputs)`:

1. `Circuit.__call__` → `_run_numpy`, which checks `inputs.ndim` (1-D single
   sample, 2-D batch).
2. `compiled.run(weights, inputs)` resolves the Parameter expressions (`resolve`
   for the numpy sim / PennyLane, `_resolve_symbolic` for Qiskit) and returns an
   expectation-value array. For 2-D inputs, backends with native batching run
   the whole batch at once; the rest loop over samples inside the backend
   (`batch_call`). `Circuit` forwards the batch unchanged either way.

**Per gradient call** — `circuit.jacobian(weights, inputs)`:

1. `_gradient_jacobians` prefers the backend's own `CompiledCircuit.jacobian`
   (the numpy sim's analytic **adjoint**, PennyLane's diff engine) and falls
   back to core **parameter-shift** when the backend supplies none.
2. Returns `d<O>/d(weights)` (and optionally `d<O>/d(inputs)`) — no torch
   involved, so chemistry / optimization / custom numpy training all
   differentiate through the same path.

**The torch interface** (`interface="torch"`, the `torch` extra) returns a
`TorchCircuit` nn.Module that owns `weights` as a `nn.Parameter`. `layer(x)`
delegates to the registered torch interface adapter: for native-autograd
backends (PennyLane) torch autograd flows through `run_torch`; for
parameter-shift backends the numpy `compiled.run` is wrapped in a
`torch.autograd.Function` whose backward calls the same jacobian ladder. Either
way it's the *same* `CircuitSpec` and provider underneath.

### Key design decisions

#### CircuitSpec-centered

Every backend speaks the same language: `CircuitSpec + Observable`.
Adding a feature (a gate transform, an optimization pass, a layered-gate
annotation for a layer-aware backend) is a single change in the data layer that every
backend picks up automatically.

#### Compile / run split

`provider.compile(spec, observables) -> CompiledCircuit` is paid once
per circuit. `compiled.run(weights, inputs) -> np.ndarray` is paid per
call. For Qiskit the saving is that binding parameters is much cheaper than
rebuilding `QuantumCircuit`s. On Qiskit Runtime, and on IQM in its default
`transpile_mode="parametric"`, compile also transpiles the circuit for the
device, once.

#### Polymorphic `provider` argument

Three shapes; each is obvious at the call site:

```python
Circuit(blocks=..., provider="pennylane")  # registry lookup
Circuit(blocks=..., provider=PennyLaneConfig(...))  # config -> backend
Circuit(blocks=..., provider=PennyLaneBackend(...))  # explicit instance
```

Defaults are pre-registered. Registration is optional and only needed
for named reuse of custom configs.

#### Observables-first measurement

Measurement is a list of `Observable`s passed via
`Circuit(observables=...)`, not a special block. Pauli sums and
Hamiltonians are first-class:

```python
Circuit(blocks=[...], observables=0.7 * Z(0) + 0.3 * X(1) * Z(2))
```

#### Discriminated configs, not one device monolith

Each backend has its own config: `PennyLaneConfig`, `QiskitConfig`,
`BraketConfig`, and so on. Each holds only the fields that apply, and
typecheckers/IDEs autocomplete the right options.

#### Parameter expressions

Gate angles are not raw values; they're a small expression tree:

| Node | Meaning | Use case |
|---|---|---|
| `Constant(0.5)` | literal | `RX(0.5)` |
| `WeightRef(3)` | `weights[3]` | trainable rotations |
| `InputRef(0)` | `inputs[0]` | data encoding |
| `Scale(expr, factor)` | `factor * expr` | `RY(2*pi * x)` |
| `Negate(expr)` | `-expr` | adjoint of a trainable |
| `Sum((e1, e2))` | `e1 + e2` | data reuploading |
| `Multiply(a, b)` | `a * b` (bilinear) | trainable feature scaling |

This keeps the data layer provider-neutral while still expressing the
patterns QML circuits actually need.

For instructions on how to extend the library (new backends, blocks,
gates, observable types) and how to contribute changes, see
[CONTRIBUTING.md](CONTRIBUTING.md).

---

## Reference

### File map

```
src/eigenkit/
├── __init__.py            ← public API surface (lazy re-exports; core only at import)
├── circuit.py             ← top-level Circuit dispatcher (numpy default; interface=)
├── configs.py             ← Encoding/Variational/StrongVariational/Reuploading + *Config
├── core/                  ← COMPUTE LAYER (numpy + scipy, framework-free)
│   ├── ir.py              ← pure data (CircuitSpec, Parameter, Observable, ...)
│   ├── provider.py        ← Provider protocol + registry (eager + lazy)
│   ├── interfaces.py      ← interface registry (torch, lazily; numpy is inline)
│   ├── observables.py     ← PauliExpr, X/Y/Z, from_string alias
│   ├── builder.py         ← CircuitBuilder (imperative)
│   ├── blocks.py          ← Block protocol + concrete blocks + assemble
│   ├── circuit.py         ← numpy Circuit (weights per call) + interface dispatch
│   ├── variational.py     ← minimize_expectation (scipy variational spine)
│   ├── optimizers.py      ← spsa, gradient-free (exported top-level)
│   ├── solver.py          ← BaseQuantumSolver (HHL/IQAE/QAOA/VQE inherit it)
│   ├── draw.py            ← ASCII circuit drawing
│   ├── draw_mpl.py        ← matplotlib drawing (the `viz` extra)
│   ├── wire.py            ← JSON codecs for CircuitSpec + observables (.eigenkit)
│   ├── secret_fields.py   ← SecretFieldsMixin: credentials never pickled or saved
│   ├── _results.py        ← array-safe equality for the result dataclasses
│   └── backends/
│       ├── numpy_backend.py         ← built-in reference simulator (adjoint grads)
│       ├── pennylane_backend.py     ← lazy-registered
│       ├── qiskit_backend.py
│       ├── qiskit_aer_backend.py    ← Aer dense + MPS cores (in the `qiskit` extra)
│       ├── braket_backend.py        ← local + cloud + QPU via config switch
│       ├── qiskit_runtime_backend.py← IBM Runtime (Eagle/Heron + FakeBackends)
│       ├── iqm_backend.py           ← IQM Resonance via iqm-client (counts-only)
│       ├── cudaquantum_backend.py   ← lazy import, opt-in registration
│       ├── tensorcircuit_backend.py ← tensor-network sim on JAX (vmap + jit)
│       ├── qsim_backend.py          ← Google qsim via Cirq (single-precision)
│       ├── qulacs_backend.py        ← fast C++ sim, native adjoint (backprop)
│       ├── _sampling.py             ← counts -> expectations/probabilities (shared)
│       ├── _kernel_support.py       ← kernel job handle + transpiled cost (Runtime, IQM)
│       └── _state_prep.py           ← the shared AmplitudeEncoding unitary
├── algorithms/            ← RING: HHL, IQAE (numpy), PortfolioRiskEstimator (an
│                            sklearn estimator, so the package imports scikit-learn)
├── optimization/          ← RING (numpy): QAOA, QUBO/Ising, max_cut
├── chemistry/             ← RING (numpy; PySCF path = extra): FermionOperator,
│   │                        Jordan-Wigner / Bravyi-Kitaev, VQE, UCCSD, molecules
│   └── interop.py         ← OpenFermion adapters
├── reservoir/             ← RING (numpy): quantum reservoir computing —
│                            driver, policies, feature maps, ridge readout
├── qml/                   ← RING (sklearn): QuantumKernel, QSVC/QSVR, QRC
│   ├── _base.py           ← BaseQuantumEstimator/Classifier/Regressor (sklearn)
│   ├── _kernels.py        ← QuantumKernel (numpy interface, zero_state_probability)
│   ├── _alignment.py      ← align_kernel (kernel-alignment training)
│   ├── _registry.py       ← estimator registry for .eigenkit bundle loading
│   ├── _serialization.py  ← .eigenkit bundles (torch imported lazily)
│   ├── qsvc.py / qsvr.py  ← quantum-kernel SVMs
│   └── qrc.py             ← the sklearn-facing reservoir estimators
├── photonics/             ← RING (`perceval` extra): linear optics, its own paradigm
│   │                        (not a qubit Provider); own import-linter contract
│   ├── ir.py              ← PhotonicCircuitSpec + components (core's Parameter tree)
│   ├── executor.py        ← Executor / Compiled protocol + lazy executor registry
│   ├── gradients.py       ← photonic parameter-shift + finite-difference jacobians
│   ├── reservoir.py       ← photonic reservoir (fixed interferometer)
│   ├── torch.py           ← PhotonicCircuit nn.Module (loaded lazily)
│   └── backends/
│       ├── perceval_backend.py      ← Perceval SLOS executor (perceval imported lazily)
│       └── perceval_remote.py       ← Quandela Cloud executor (submit, dry_run_info)
├── torch/                 ← RING (torch): differentiable interface + estimators
│   ├── circuit.py         ← TorchCircuit (nn.Module owning weights)
│   ├── interface.py       ← torch interface adapter (autograd.Functions)
│   ├── layers/            ← Basic/Reupload/Ensemble/MinMaxNorm
│   └── estimators/        ← HQNC/HQNR/Recurrent*/Custom*/Generic*/Timeseries + _nn.py
├── benchmark/             ← comparison harness (`benchmark` extra)
└── datasets/              ← supervised dataset utilities (`datasets` extra)

tests/
├── test_public_api_contract.py ← what the top-level `__init__` promises
├── test_top_level_circuit.py  ← dispatcher (incl. test_default_is_numpy)
├── core/                  ← IR, provider, variational, torch-free import, every
│                            backend adapter, and the cross-backend parity suite
├── layers/                ← torch-ring layer tests
├── algorithms/            ← estimator tests (QSVC/QSVR from qml, HQNC from torch)
│                            plus the HHL / IQAE / portfolio-risk solvers
├── optimization/          ← QAOA / QUBO tests
├── chemistry/             ← Jordan-Wigner, VQE, UCCSD, molecules (some need extra)
├── reservoir/             ← reservoir computing and its backend parity
├── photonics/             ← photonic IR, executors, gradients, torch, import surface
├── benchmark/             ← comparison-harness tests
├── datasets/              ← dataset utility tests
└── integration/           ← end-to-end runs needing real hardware (GPU sweep)

ARCHITECTURE.md
conftest.py
pyproject.toml
```

### Glossary

- **CircuitSpec** — the provider-neutral, frozen-dataclass description
  of a quantum circuit (gates, qubits, weight/input counts). The foundation
  of the core compute layer.
- **Provider** — a class implementing the `Provider` protocol that
  knows how to execute a `CircuitSpec` on a particular backend
  (PennyLane, Qiskit, etc.).
- **Backend** — informal synonym for Provider.
- **Compiled circuit** — opaque handle returned by
  `provider.compile()`. Cheap to call repeatedly. Internally caches
  devices, qnodes, parametric circuits, and Pauli operators.
- **Capabilities** — a backend's self-description (supported gates, max
  qubits, batching, shots). Drives pre-compile validation;
  `supports_batching` is informational only, and nothing dispatches on it.
- **Block** — a reusable circuit pattern that emits an `IRFragment`.
  Implements `required_qubits()` and `build(n_qubits)`. Examples:
  `Encoding`, `Variational`, `StrongVariational`, `Reuploading`,
  `CustomBlock`.
- **IRFragment** — a block's contribution to a circuit: gates with
  local parameter indices plus counts. Combined into a `CircuitSpec`
  by `assemble()`.
- **Parameter expression** — symbolic angle on a gate. Seven kinds:
  `Constant`, `WeightRef`, `InputRef`, `Scale`, `Negate`, `Sum`,
  `Multiply`. The first three are leaves; the rest compose. `Multiply`
  is bilinear (two symbolic operands) and powers patterns like
  trainable feature scaling.
- **PauliExpr** — fluent builder for observables: `Z(0) * X(1) + 0.5 *
  Y(2)`, then `.to_observable()`.
- **Interface** — `"torch"` or `"numpy"`. Determines whether the inner
  Circuit returns a torch tensor (with autograd) or a numpy array.
- **Gradient method** — how a backend produces gradients: `"native"`
  (torch interface carries grads through the device), `"adjoint"` (the
  backend ships its own analytic `jacobian`: numpy, Qulacs, TensorCircuit),
  `"parameter_shift"` (core parameter-shift on the numpy `run`), or `"none"`
  (no gradients; `.backward()`, `jacobian()` and gradient-based optimizers
  raise a clear error). `"none"` is the hardware default, and a device config
  with `enable_gradients=True` switches it to `"parameter_shift"`. Under torch,
  `"adjoint"` and `"parameter_shift"` share one `autograd.Function`, whose
  backward calls the backend's `jacobian` when it has one.
- **JobHandle** — opaque async-job wrapper returned by
  `circuit.submit(weights, inputs)` on the numpy `Circuit` (or
  `TorchCircuit.submit(x)`) for the Braket, Qiskit Runtime and IQM
  backends. Uniform interface
  (`.result()`, `.status()`); each backend has its own concrete class.
- **CostWarning** — `UserWarning` subclass emitted once at compile
  time when a backend points at billable infrastructure (cloud
  simulator or QPU). Catch or filter with `warnings.catch_warnings()`.
- **dry_run_info()** — method on cloud `CompiledCircuit` instances
  that returns a dict describing the prepared circuit (qubit count,
  gate count, transpiled depth, PUBs per call, ...) without
  submitting anything. Use it to sanity-check before paying.
- **Chain-rule-aware parameter-shift** —
  `core.Circuit._chain_rule_jacobians`. When a circuit has gate angles
  that wrap weights in `Scale`/`Multiply`/`Sum`/`Negate` (e.g.
  `Reuploading(trainable_scale=True)`), naive parameter-shift on
  weights misses the chain-rule factor. The chain-rule path shifts
  each parametric gate's angle by +/- pi/2, and also by +/- 3pi/2 for
  `CRY` (the four-term rule), in a freshly compiled CircuitSpec cached
  per `(gate_index, shift_value)`, and sums
  `(dtheta_g/dw_i) * (d<O>/dtheta_g)` across gates. Cost: two perturbed
  compiles per parametric gate (four for `CRY`) on the first backward,
  reused thereafter. Dispatched automatically based on
  `find_parameter_shift_unsafe_gates`.

### Backend status

| Backend | Auto-registered | Gradients | Notes |
|---|---|---|---|
| numpy | yes (`"numpy"`) | adjoint | **Default.** Built-in reference statevector sim, zero extra deps. |
| PennyLane | lazy (`"pennylane"`) | native | Any PennyLane device via `PennyLaneConfig`. The `pennylane` extra. |
| Qiskit | lazy (`"qiskit"`) | parameter-shift | Statevector simulator; finite shots not wired yet. |
| Qiskit Aer | lazy (`"qiskit-aer"`) | parameter-shift | Aer's dense statevector core, read through `save_expectation_value`. Ships in the `qiskit` extra — no new dependency. Matches numpy to ~1e-12. |
| Qiskit Aer MPS | lazy (`"qiskit-aer-mps"`) | parameter-shift | Same backend, `method="matrix_product_state"`. Exact at unbounded bond dimension (~1e-9, the gap being SVD truncation). |
| Braket (local) | lazy (`"braket"`) | parameter-shift | Default `LocalSimulator`. The `braket` extra. |
| Braket (cloud / QPU) | (same backend, config switch) | parameter-shift / none | `BraketConfig(device_arn=...)`. QPU disables gradients (unless `enable_gradients=True`) and enforces `shots > 0`. |
| Qiskit Runtime | opt-in | none / parameter-shift (`enable_gradients=True`) | `QiskitRuntimeConfig` targets IBM Eagle / Heron and FakeBackends. `primitive="sampler"` swaps `EstimatorV2` for `SamplerV2`, reducing expectation values from raw counts. |
| IQM | lazy (`"iqm"`) | none / parameter-shift (`enable_gradients=True`) | `IQMConfig` targets IQM Resonance through iqm-client, plus IQM's noise-modelled fake backends. Counts-only hardware, so every value is reduced from bitstring histograms. The `iqm` extra. |
| CUDA-Q | opt-in | parameter-shift | Optional `cudaq` dependency. `Sum` and `Multiply` over symbolic refs are not supported (cudaq `QuakeValue` arithmetic is restricted). |
| Qulacs | lazy (`"qulacs"`) | adjoint | Fast C++ statevector sim. Native reverse-mode gradient via `backprop`; full gate set; the `qulacs` extra. |
| TensorCircuit | lazy (`"tensorcircuit"`) | adjoint (`jax.jacrev`) | Tensor-network sim via tensorcircuit-ng on JAX, vectorised with `jax.vmap` and `jax.jit`-compiled. Exact contraction, so a constant factor rather than sub-exponential reach; best on reused/amortised workloads. The `tensorcircuit` extra. |
| qsim | lazy (`"qsim"`) | parameter-shift | Google's qsim via the `qsimcirq` Cirq bridge. **Single-precision (fp32)** — fast, ~1e-6 residual, so it does not pass a 1e-9 parity gate. The `qsim` extra. |

Braket, Qiskit Runtime and IQM support async dispatch via
`circuit.submit(weights, inputs) -> JobHandle` on the numpy `Circuit`
(`TorchCircuit.submit(x)` on the torch side). `handle.result()` blocks;
`handle.status()` polls.

### Planned

- A JAX interface (one adapter registering into `core/interfaces.py`).
- Finite-shot sampling for the local Qiskit backend.
