# QPU and Cloud Workflows

Three backends support real hardware: Braket (IonQ, Rigetti, IQM, AQT and others),
Qiskit Runtime (IBM Eagle and Heron), and IQM through the software of IQM itself. All
three support calls that wait for the result and calls that do not, inspection
before a run, and a warning before anything runs on paid hardware. The hardware devices are
called quantum processing units (QPUs).

| | Braket | Qiskit Runtime | IQM |
|---|---|---|---|
| Registration | Lazy, `provider="braket"`. | Pass a `QiskitRuntimeConfig`. `register("qiskit-runtime", ...)` is needed only for the string name. | Lazy, `provider="iqm"`. |
| Hardware reached | IonQ, Rigetti, IQM and AQT, through `device_arn`. | IBM Eagle and Heron, through `backend_name`. | IQM Crystal and Star devices, through `url` and `quantum_computer`. |
| Simulator without an account | Yes: `device_arn=None` runs `LocalSimulator`. | No. Use a `FakeBackendV2` through `backend_instance`. | No. Use an `IQMFake*` backend through `backend_instance`. |
| Selects the device by | `device_arn`. | `backend_name` (or `"least_busy"`), or a backend object. | `url` and `quantum_computer`, or a backend object. |
| Analytic evaluation | With `shots=0`, on simulators only. | Never: `shots` defaults to `1024`. | Never: the hardware only samples. |
| Measurement | Expectation result types, and counts and probability result types for `sample()` and `probabilities()`. | `EstimatorV2` or `SamplerV2`, chosen with `primitive=`. | Counts only. The expectation values are computed from them. |
| The job handle has | `result()`, `status()`, `cancel()` and `task_id()`. | `result()`, `status()`, `cancel()` and `job_id()`. | `result()`, `status()`, `cancel()`, `job_id()` and `counts()`. |
| Extra | `braket` | `qiskit` | `iqm` |

There is a second way to reach IQM hardware: a Braket `device_arn` that points at an
IQM QPU, which is billed through Amazon Web Services (AWS). The `iqm` provider talks
to IQM Resonance directly instead. It is also the only way to use the fake backends
of IQM and its Star architectures, which use MOVE gates.

The examples below use the **PyTorch** interface (`interface="torch"`). The supported
pattern is to train a differentiable `nn.Module` circuit on a simulator, and then to
use its weights on the QPU, because QPU backends refuse gradients unless you
[turn them on](#gradients-on-hardware). The NumPy interface also works on QPU
backends: pass the weights explicitly, with `circuit(weights, x)` or
`circuit.submit(weights, x)`.

## Cost warnings

The first compilation for a paid backend (a cloud simulator or a QPU) issues a single
`CostWarning`. You can catch it or filter it with the standard `warnings` module:

```python
import warnings
from eigenkit import CostWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", CostWarning)
    circuit = Circuit(blocks=..., provider=qpu_config, interface="torch")
```

QPU configurations also do the following:

- They require `shots > 0`, because analytic evaluation is not possible on hardware.
- They set `gradient_method="none"`. Calling `.backward()` on a QPU result,
  `circuit.jacobian()` or a gradient-based optimiser raises a clear error, which
  points to the workflow of training on a simulator and then running on the device.
  This does not happen when the configuration sets `enable_gradients=True` (see
  [Gradients on hardware](#gradients-on-hardware)).

## Calls that wait for the result

`Circuit(...)(x)` works on QPU backends in the same way as on simulators, but it
waits until the job is finished:

```python
from eigenkit import Circuit, BraketConfig

config = BraketConfig(
    device_arn="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    shots=1000,
)
circuit = Circuit(blocks=..., provider=config, interface="torch")
y = circuit(x)  # blocks until the QPU returns
```

## Submitting without waiting

For long jobs, use `circuit.submit(x)`, which returns a `JobHandle` immediately:

```python
handle = circuit.submit(x)
print(handle.status())  # "QUEUED", "RUNNING", "COMPLETED", ...

# Do other work...

result = handle.result()  # blocks until the job is done
```

`JobHandle.result()` returns the expectation value of each observable, as a NumPy
array. The protocol itself declares `result()` and `status()`. The three handles that
exist also implement `cancel()`. The Braket handle has `task_id()`, while the Qiskit
Runtime and IQM handles have `job_id()`. The IQM handle also has `counts()`, which
returns the raw histograms behind the expectation values, one for each measurement
group. A custom handle can implement only the two methods that the protocol names.

`submit()` accepts only a single 1-D input, and no batches. For a batch, loop in
Python.

## Dry runs

The backends that can run in the cloud have `Circuit.dry_run_info()`. It returns a
dictionary that describes the prepared circuit, without submitting anything. Use it to
check the circuit before you submit it:

```python
info = circuit.dry_run_info()
print(info["n_circuit_runs_per_call"])
```

The backends return **different keys**. They do not share a common set with
additions, and only three keys appear in all of them.

| Key | Braket | Qiskit Runtime | IQM | Meaning |
|---|:--:|:--:|:--:|---|
| `shots` | ✓ | ✓ | ✓ | The number of shots per circuit run. |
| `n_observables` | ✓ | ✓ | ✓ | The number of observables measured. |
| `n_circuit_runs_per_call` | ✓ | ✓ | ✓ | **The key number**: how many circuits the device runs for one forward call. |
| `device_arn` | ✓ | | | The chosen Braket device, or `None` for the local simulator. |
| `local_backend` | ✓ | | | The `LocalSimulator` that runs when `device_arn` is `None`. |
| `n_qubits` | ✓ | | | The number of qubits of the circuit. |
| `n_gates` | ✓ | | | The number of gates, as built. |
| `n_unique_pauli_terms` | ✓ | | | The number of distinct Pauli terms. With shots, they are measured in one task for each group of terms that commute qubit-wise. Exact runs measure them all in one task. |
| `is_cloud`, `is_qpu` | ✓ | | | Whether the target is billed, and whether it is hardware. |
| `backend_name` | | ✓ | ✓ | The device that the job would go to. |
| `is_fake` | | ✓ | | `True` for a `FakeBackendV2`, so nothing is billed. |
| `is_local` | | | ✓ | `True` for an `IQMFake*` backend, so nothing is charged. |
| `primitive` | | ✓ | | `"estimator"` or `"sampler"`. |
| `transpile_mode` | | | ✓ | `"parametric"` or `"per_call"`. |
| `n_measurement_groups` | | ✓ | ✓ | The number of groups of observables that commute qubit-wise. This sets the number of circuit runs per call when sampling. |
| `optimization_level` | | ✓ | ✓ | The effort of the transpiler that produced the numbers below. |
| `n_logical_qubits` | | ✓ | ✓ | The number of qubits before transpilation. |
| `n_physical_qubits` | | ✓ | ✓ | The number of qubits of the target device. |
| `transpiled_gate_counts` | | ✓ | ✓ | The gates after transpilation, counted by gate name. IQM reports one entry for each measurement basis. |
| `transpiled_depth` | | ✓ | ✓ | The depth of the circuit after transpilation. IQM reports one entry for each measurement basis, and leaves out both keys in `"per_call"` mode, which transpiles when the circuit is called. |

`n_circuit_runs_per_call` is the key to read before you submit. It is the number of
circuits that the device runs for a single forward call.

## Resources after transpilation

On Qiskit Runtime and IQM, `Circuit.resources()` reports the resources that the circuit
uses on its target once it is transpiled. These are the numbers that decide whether the circuit
survives the noise of the device. Nothing is submitted, and a fake `backend_instance`
gives the same report locally:

```python
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from qiskit_ibm_runtime.fake_provider import FakeFez

from eigenkit import Circuit, IQMConfig, QiskitRuntimeConfig

for config in (
    QiskitRuntimeConfig(
        backend_instance=FakeFez(),
        primitive="sampler",
        optimization_level=3,
        seed_transpiler=0,
    ),
    IQMConfig(backend_instance=IQMFakeGarnet(), optimization_level=3, seed_transpiler=0),
):
    report = Circuit(blocks=..., provider=config).resources()
    print(
        report["n_2q_gates"],
        report["depth_2q"],
        report["routing_overhead_2q"],
        report["estimated_success_probability"],
    )
```

| Key | Meaning |
|---|---|
| `gate_counts` | The native gates after transpilation, including the measurements. |
| `n_2q_gates`, `two_qubit_gate_counts` | The two-qubit operations, in total and by name (`cz`, `ecr`, or `move` on IQM). |
| `depth`, `depth_2q` | The depth of the circuit, and the depth that counts only the layers of two-qubit gates. |
| `n_2q_gates_unrouted` | The same count for the circuit on the native gates of the device, with every qubit connected to every other. |
| `routing_overhead_2q` | `n_2q_gates - n_2q_gates_unrouted`, which is the number of two-qubit gates that the coupling map adds. |
| `physical_qubits` | Where the layout placed each logical qubit. |
| `estimated_success_probability` | The product of `1 - error` over every operation and readout, from the error rates of the target (for the IQM fake backends, from their error profile). Idle time between operations is not counted. On the fake IBM devices, it matched the measured loss of contrast of a kernel within `0.02`. On the IQM fake backends, it overestimates deep circuits: `0.85` against a measured `0.42` for a 4-qubit ZZ kernel with `64` two-qubit gates on Garnet. |

`Circuit.transpiled_circuit()` returns the Qiskit circuit itself, for
`.draw("mpl")`. Set `seed_transpiler` to compare two configurations fairly, because
routing without a seed varies from run to run.

Quantum kernels use the same machinery. See
[Quantum Kernels, On hardware](kernels.md#on-hardware). A kernel matrix is one job,
and `QuantumKernel.submit_kernel_matrix()` and `retrieve_kernel_matrix()` let a long
queue finish in another session.

## Sampling: counts, probabilities, and QAOA on hardware

Hardware does not return expectation values. It returns a histogram of measured
bitstrings. Braket, Qiskit Runtime and IQM give access to this histogram directly:

```python
counts = circuit.sample(weights, x)
# {"01": 4913, "00": 3020, "11": 1584, "10": 483}
```

In the keys, qubit 0 is the most significant bit, so `"100"` means that qubit 0 is
excited and the others are in the ground state. This is the same convention as
everywhere else in the library. The counts add up to the `shots` configured for the
backend.

`circuit.probabilities(weights, x)` turns the same histogram into the full
distribution over `2**n_qubits` states. On hardware, it is an *estimate from the
frequencies of the shots*, not the exact distribution: basis states that were not
sampled come back as exact zeros, without smoothing. It is limited to `24` qubits.
Above that, a budget of shots cannot fill a dense distribution anyway, so use
`sample()` instead.

This is what lets the Quantum Approximate Optimization Algorithm (QAOA) read a
solution from real hardware, because it reads its answer from `probabilities()`. The
optimisation loop still needs a method without gradients, because hardware backends
declare `gradient_method="none"` unless the configuration sets
`enable_gradients=True`:

```python
from eigenkit.optimization import QAOA
from eigenkit import IQMConfig

solver = QAOA(
    layers=1,
    provider=IQMConfig(url="https://resonance.iqm.tech/", shots=4000),
    optimizer="COBYLA",
)
```

`optimizer="spsa"` is also accepted. Simultaneous perturbation stochastic
approximation (SPSA) estimates the gradient from two evaluations per iteration,
whatever the number of parameters, and makes a third evaluation to record the loss.
This keeps the number of jobs low on hardware.

### How an expectation value is computed from a histogram

When a backend samples, the Pauli terms of the observables are sorted into
**measurement groups**. Two terms share a group when no qubit needs two different
bases for them, which means that they commute qubit-wise. Each group is one circuit:
it rotates the qubits (`H` for `X`, and `Sdg` then `H` for `Y`), measures them, and
then computes the value from the parity of the counts. `all_z(4)` is one group, not
four circuits, and the `Z` and `ZZ` terms of a QAOA cost Hamiltonian also make one
group.

`n_measurement_groups` in `dry_run_info()` is this number, and on a sampling backend,
`n_circuit_runs_per_call` is equal to it. Set `group_commuting=False` to run one
circuit per term instead. This runs more circuits, but each term gets the full budget
of shots for itself, which is the point of comparison when you want to attribute the
shot noise to each term.

## Recommended workflow: train on a simulator, run on the QPU

QPU calls are too expensive for parameter-shift gradients, because one backward pass
costs at least `2 * n_weights` jobs. The supported workflow is:

1. Train on a simulator (`provider="numpy"` or `provider="pennylane"`).
2. Save the weights with `torch.save(circuit.state_dict(), ...)`.
3. Build the same circuit with a QPU provider, and load the weights.
4. Run the inference with `circuit(x)` or `circuit.submit(x)`.

```python
# Train.
sim_circuit = Circuit(blocks=blocks, interface="torch")
# ... training loop ...
torch.save(sim_circuit.state_dict(), "weights.pt")

# Deploy.
qpu_circuit = Circuit(
    blocks=blocks,
    provider=BraketConfig(device_arn=..., shots=1000),
    interface="torch",
)
qpu_circuit.load_state_dict(torch.load("weights.pt"))
y = qpu_circuit(x_test)
```

## Gradients on hardware

To train or fine-tune on the device itself, set `enable_gradients=True` in the device
configuration (`BraketConfig`, `QiskitRuntimeConfig` or `IQMConfig`). The backend then
reports `gradient_method="parameter_shift"`, and `.backward()`, `circuit.jacobian()`
and gradient-based optimisers run the parameter-shift rule on the hardware:

```python
qpu_circuit = Circuit(
    blocks=blocks,
    provider=IQMConfig(
        url="https://resonance.iqm.tech/",
        shots=2000,
        enable_gradients=True,
    ),
    interface="torch",
)
loss = qpu_circuit(x).sum()
loss.backward()  # 2 * n_weights forward calls on the device
```

Compiling for a paid device with this flag set issues a second `CostWarning`, which
states how many evaluations one backward pass needs there, for example *"one backward
pass costs at least 24 circuits per sample (12 parameter-shift evaluations x 2 circuits
each) at shots=2000"*. The unit is the one that the cost warning of the device counts: circuits
on IQM, Primitive Unified Blocs (PUBs) on Qiskit Runtime, and tasks on Braket. This figure is a minimum. A batch
multiplies it by the batch size, and input gradients, or weights wrapped in `Scale`,
`Multiply` or `Sum` angles, add more. Local fake backends issue neither warning. On
Braket, the flag matters only for QPU ARNs, because the local and cloud simulators
compute gradients in either case.

Gradients estimated from shots are noisy: the standard deviation of each entry scales
as `1/sqrt(shots)`. If the cost is the concern, SPSA (`optimizer="spsa"`) needs two
evaluations per step to estimate the gradient, whatever the number of weights, and it
needs no flag.

## Notes for each backend

### Braket

- `device_arn=None` runs on the `LocalSimulator`, in the same process.
- An Amazon Resource Name (ARN) that contains `/qpu/` (Rigetti, IonQ, IQM, AQT and
  others) turns on the QPU mode automatically.
- Amazon Web Services bills the tasks on its cloud simulators and QPUs. eigenkit does
  not estimate the price.
- The AWS credentials are read from the environment, through the standard boto3
  chain.
- The `submit()` of the compiled circuit takes two keyword arguments. `method=`
  chooses what the task computes: `"run"` (the default, which measures the
  observables), `"sample"`, `"probabilities"` or `"zero_state_probability"`
  (`BraketCompiled.submit_methods` lists them). Each returns what the waiting method
  of the same name returns. `tags={...}` puts AWS tags on a cloud task. The tags
  follow the AWS rules (no key that starts with `aws:`, and only letters, digits,
  spaces and `_ . : / = + - @`), and they are checked before anything is submitted.
  The local simulator ignores tags.
- `circuit.compiled.attach(task_arn)` builds a `BraketJobHandle` again for a cloud task
  submitted earlier, for example after the process that submitted it has restarted.
  Pass the same `method` as the one the task was submitted with. Values with the shape
  of another method or circuit raise a `ValueError`.
- `handle.result()` raises a `RuntimeError`, which names the task and its state, when
  the task ended as `FAILED` or `CANCELLED`, or when the polling timed out.

#### Braket shot rules and result types

`sample()` reads the measurement counts of the task, so it needs `shots > 0`
everywhere, and a QPU needs a number of shots within the range of the device. In the
keys, qubit 0 is the most significant bit, over the whole circuit, and a qubit that
no gate acts on reads `0`.

`probabilities()` and `zero_state_probability()` read the first result type that the
device publishes for the configured shots, in the order `Probability`,
`StateVector`, `Amplitude`, `DensityMatrix`. The capabilities of the device
(GetDevice) decide, so the choice is fixed for each device and number of shots:

| Device | `sample` | `probabilities`, `shots=0` | `probabilities`, `shots > 0` | `zero_state_probability`, `shots=0` | `zero_state_probability`, `shots > 0` |
|---|---|---|---|---|---|
| Local simulator | Counts. | `Probability`. | `Probability`. | `Probability`. | `Probability`. |
| SV1 | Counts. | The `Amplitude` of every basis state, squared, up to `17` qubits. | `Probability`, up to `15` qubits. | The `Amplitude` of the all-zeros state, squared, for any number of qubits. | `Probability`, up to `15` qubits. |
| DM1 | Counts. | The diagonal of the `DensityMatrix`, up to `8` qubits. | `Probability`, up to `15` qubits. | The `[0, 0]` entry of the `DensityMatrix`, up to `8` qubits. | `Probability`, up to `15` qubits. |
| QPU | Counts. | Refused, because QPUs need `shots > 0`. | `Probability`, up to `15` qubits. | Refused, because QPUs need `shots > 0`. | `Probability`, up to `15` qubits. |

"Counts" means no result type: the task returns its measurement counts, which needs
`shots > 0` on every device. SV1 and DM1 publish `Probability` only for `1` to
`100,000` shots, so their answers with `shots=0` come from the exact result types
instead, and they match the local simulator. With `shots > 0`, `probabilities()` is
the estimate from the frequencies of the shots. A device that publishes none of the
four result types for the requested shots raises a `ValueError` before anything is
submitted.

The limits on the number of qubits in the table are the limits of Braket itself. They
are checked before anything is submitted, so a wider circuit fails with a
`ValueError` and not at AWS:

- The managed simulators and QPUs of Braket return `Probability` for at most `15`
  qubits
  ([List of result types](https://docs.aws.amazon.com/braket/latest/developerguide/braket-result-types.html)),
  and both methods read it over every qubit. So every cloud device stops at `15`
  qubits with `shots > 0`, which is below the limit of `24` qubits of the dense
  `probabilities()`. On the local simulator, `Probability` is limited only by memory,
  and `sample()` has no limit anywhere.
- The exact `probabilities()` of SV1 names all the `2**n_qubits` basis states inside
  the program, and Braket limits the program of a quantum task to 5 megabytes
  ([Amazon Braket quotas](https://docs.aws.amazon.com/braket/latest/developerguide/braket-quotas.html)).
  The list alone takes 3.0 megabytes at `17` qubits, and 6.3 megabytes at `18`.
  `zero_state_probability()` names one state, so it has no limit.
- DM1 returns a density matrix of at most `8` qubits
  ([DM1 results](https://docs.aws.amazon.com/braket/latest/developerguide/braket-submit-tasks-simulators.html#braket-simulator-dm1)),
  and both exact methods need the density matrix over every qubit, which is `4**8`,
  or `65,536`, entries at the limit.

Above the limit of `Probability`, use `sample()` for the histogram, which has no
limit on the number of qubits. `zero_state_probability()` is then the count of the
all-zeros state divided by the number of shots. Above an exact limit, use
`shots > 0` for the estimate from the frequencies of the shots (up to `15` qubits), or
`sample()`.

### Qiskit Runtime

- The token is looked up in this order: an explicit `token`, then the
  `QISKIT_IBM_TOKEN` environment variable, then the saved account file.
- `channel` defaults to `"ibm_quantum_platform"`. The older `"ibm_quantum"` channel
  was retired with the old IBM Quantum Platform, and `QiskitRuntimeService` no longer
  accepts that string.
- Pass a `FakeBackendV2` instance through `backend_instance` for tests, in continuous
  integration or locally. No authentication is needed.
- There are two primitives, chosen with `QiskitRuntimeConfig(primitive=...)`. Each
  call submits one job, whichever primitive is used.

    - `"estimator"` (the default) uses `EstimatorV2`. Each observable becomes one
      Primitive Unified Bloc (PUB) in the job, so `n_circuit_runs_per_call` equals
      `n_observables`.
    - `"sampler"` uses `SamplerV2`, and computes the expectation values from the raw
      counts locally. The job holds one PUB for each measurement group, in place of
      one for each observable, and the budget of shots is explicit. This is what you
      want when the shot cost of a result is itself what you are measuring.

- `sample()` and `probabilities()` go through `SamplerV2`, whichever primitive is
  configured, because `EstimatorV2` has no bitstrings to return. `primitive` only
  chooses how the expectation values are obtained.
- Both primitives process a batch at once. The PUBs of `EstimatorV2` and `SamplerV2`
  both accept a `(B, n_params)` array, so a batch of `B` inputs stays one job, with
  one PUB for each observable (or for each group), in place of one job for each
  sample.
- The `submit()` of the compiled circuit takes two keyword arguments. `method=`
  chooses what the job computes: `"run"` (the default, which measures the
  observables), `"sample"` or `"probabilities"`
  (`QiskitRuntimeCompiled.submit_methods` lists them). Each returns what the waiting
  method of the same name returns. `"sample"` and `"probabilities"` are one
  `SamplerV2` job in the `Z` basis, whichever primitive is used.
  `zero_state_probability` is not a method of `submit()`. Quantum kernels call
  `submit_zero_state_probabilities()`, which sends a whole batch as one job, and
  `retrieve_zero_state_probabilities(job_id)` attaches to it again (see
  [Resources after transpilation](#resources-after-transpilation) and
  [Quantum Kernels, On hardware](kernels.md#on-hardware)). `tags={...}` puts IBM job
  tags on the job, each as `"key=value"`, so that you can find it again with
  `QiskitRuntimeService.jobs(job_tags=[...])`. IBM allows at most `8` tags of at most
  `86` characters. Keys and values must not be empty, and a key must not contain `=`.
  The tags are checked before anything is submitted. `submit()` takes one sample:
  2-D `weights` or `inputs` raise a `ValueError`.
- `circuit.compiled.attach(job_id)` builds a `QiskitRuntimeJobHandle` again for a job
  submitted earlier, looked up through the `QiskitRuntimeService` of the backend, for
  example after the process that submitted it has restarted. Pass the same `method` as
  the one the job was submitted with. Results with the shape of another method or
  circuit raise a `ValueError`. A fake backend has no service, so `attach()` needs a
  real IBM backend.

### IQM

```python
from eigenkit import IQMConfig

config = IQMConfig(
    url="https://resonance.iqm.tech/",
    quantum_computer="garnet",  # omit to take the server's default
    shots=2048,
)
```

- The token is looked up in this order: an explicit `token`, then the `IQM_TOKEN`
  environment variable, then whatever iqm-client finds by itself. The address of the
  server can come from `IQM_SERVER_URL` in place of `url`.
- Pass an `IQMFakeAdonis()`, `IQMFakeApollo()` or `IQMFakeDeneb()` instance through
  `backend_instance` for tests, in continuous integration or locally. They run on Aer
  with a model of the noise of IQM, with no token and no network, and the tests of
  this backend use them.
- The service of IQM returns **only counts**. There is no estimator primitive, so
  every expectation value is computed from a histogram of bitstrings. `shots > 0` is
  required, and `sample()` and `probabilities()` are fully supported, not added as an
  afterthought.
- The native gates are `r` (an `RX` rotation with a phase) and `cz`. Devices with the
  Star topology route two-qubit gates through a resonator with `move` gates, which the
  transpiler adds automatically.
- `transpile_mode` balances the cost of compilation against the depth of the circuit:

    - `"parametric"` (the default) transpiles once for each measurement group, and
      binds the values at each call. The transpiler cannot merge rotations whose
      angles it cannot see, so the circuit is longer than it needs to be. A 2-qubit
      block of an encoding and a variational layer gave `6` `r` gates in parametric
      mode, against `3` when the values were bound first.
    - `"per_call"` binds the values first and transpiles at every call. This gives
      shallower circuits and a better fidelity, at the cost of one transpilation per
      evaluation. This cost is small compared with the time in the queue, but not
      compared with a local simulator.

- A batch is one job per sample. The measurement groups of a single sample do share
  a job. Quantum kernels are the exception: `zero_state_probabilities()` binds every
  row into the one transpiled circuit, and submits them as a single job.
  `submit_zero_state_probabilities()` returns without waiting for it, and
  `retrieve_zero_state_probabilities(job_id)` attaches to it again.

## API reference

::: eigenkit.core.provider.JobHandle
::: eigenkit.core.provider.CostWarning
