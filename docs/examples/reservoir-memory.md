# Short-Term Memory with a Quantum Reservoir

The **short-term memory capacity** task is the standard test of a reservoir. You
feed it a random scalar signal, and you ask the readout to rebuild the signal
delayed by `τ` steps. A reservoir with memory rebuilds recent values well, and it
gets slowly worse as `τ` grows. A model without memory can only reproduce the
present (`τ = 0`).

This example compares the three state policies, `"stateless"`, `"carry"` (the
default) and `"leaky"`, on that task. It shows *why* a memory that fades over time,
and not just any state carried from one step to the next, is what makes a quantum
reservoir work. See the [user guide](../user-guide/reservoir.md) for the application
programming interface (API).

## The task

A scalar signal `s_t ~ Uniform(-1, 1)` drives every qubit of the reservoir (we
repeat it across the input slots). The target is the same signal delayed by `τ`
steps: `y_t = s_{t-τ}`.

```python
import numpy as np
from eigenkit import Encoding, Variational
from eigenkit import QuantumReservoirRegressor

N_QUBITS, MAX_LAG = 5, 5


def delay_data(seed, B, T, tau):
    rng = np.random.default_rng(seed)
    signal = rng.uniform(-1.0, 1.0, size=(B, T, 1))
    inputs = np.repeat(signal, N_QUBITS, axis=2)  # drive every qubit with the signal
    target = np.zeros((B, T))
    target[:, tau:] = signal[:, : T - tau, 0] if tau else signal[:, :, 0]
    return inputs, target
```

## The reservoir

The reservoir is a circuit of `5` qubits. It encodes the signal as rotation
angles, and then applies a fixed, random, entangling variational block. The weights
are sampled once (with a seed) and never trained. We read out the full measurement
distribution (`features="probabilities"`).

```python
blocks = [
    Encoding(in_features=N_QUBITS, rotation="RY"),
    Variational(depth=4, rotation="RY"),
]


def memory_curve(**kw):
    r2 = []
    for tau in range(MAX_LAG + 1):
        Xtr, ytr = delay_data(0, 60, 80, tau)
        Xte, yte = delay_data(1, 30, 80, tau)
        model = QuantumReservoirRegressor(
            blocks, features="probabilities", washout=10, random_state=2, **kw
        ).fit(Xtr, ytr)
        r2.append(max(model.score(Xte, yte), 0.0))
    return r2
```

## Compare the policies

```python
configs = {
    "memoryless (stateless)": dict(state_policy="stateless", window=1),
    "carry (unitary)": dict(state_policy="carry"),
    "leaky (alpha=0.5)": dict(state_policy="leaky", leak=0.5),
}
print(
    "lag tau:           " + "  ".join(f"{t:>4d}" for t in range(MAX_LAG + 1)) + "    MC"
)
for name, kw in configs.items():
    curve = memory_curve(**kw)
    print(
        f"{name:22s} " + "  ".join(f"{v:4.2f}" for v in curve) + f"   {sum(curve):4.2f}"
    )
```

```text
lag tau:              0     1     2     3     4     5    MC
memoryless (stateless) 1.00  0.00  0.00  0.00  0.00  0.00   1.00
carry (unitary)        0.00  0.00  0.00  0.00  0.00  0.00   0.00
leaky (alpha=0.5)      0.96  0.91  0.74  0.53  0.31  0.18   3.63
```

The last column is the **memory capacity** (MC): the sum of `R²` (the coefficient
of determination) over the lags. This single number sums up how much of the history
of the input the reservoir keeps.

## What the numbers say

- **Memoryless** rebuilds the present perfectly (`R² = 1.00` at `τ = 0`), and
  nothing else. With only the current input, it has no information about the past,
  as expected. `MC = 1.0`.
- **Pure unitary carry** scores about `0` *everywhere*, even at `τ = 0`. The carry
  is correct (the state really is `ψ_t = U(x_t)ψ_{t-1}`), but without dissipation,
  the recurrence mixes the whole history into the state in a way that a *linear*
  readout cannot separate. There is no clean fading memory to read. `MC ≈ 0`.
- **Leaky carry** shows the standard curve of short-term memory: it recalls the
  recent steps almost perfectly, and the recall fades smoothly with the lag
  (`0.96 → 0.91 → 0.74 → …`). The leak of weight `(1 - α)` toward the reset state
  lets old inputs decay, so the recent ones can still be recovered linearly.
  `MC = 3.63`, which is more than three times the baseline without memory.

## Forecasting the next step

Memory is only useful if it helps to *predict*. Here, a fast quasiperiodic signal
drives the reservoir, and the readout forecasts the **next** value,
`y_t = s_{t+1}`. A fast oscillation moves a lot between steps, so the current sample
alone cannot tell where the signal goes next. The forecast needs the context of the
previous steps.

There are two ways to give the reservoir that context. The classical one is a
sliding **window**, which embeds the last `window` values as a time delay
(`state_policy="stateless"`, with one qubit for each input in the window). The
quantum one is a memory in the state that fades over time (`state_policy="leaky"`,
which drives all `5` qubits with the scalar and carries the state). We compare both
on the prediction of one step ahead.

```python
def signal(seed, B, T):
    rng = np.random.default_rng(seed)
    t = np.arange(T + 1)
    phase = rng.uniform(0, 2 * np.pi, size=(B, 1))
    return (np.sin(0.9 * t + phase) + 0.5 * np.sin(1.4 * t + 2 * phase)) / 1.5


T = 120
s_tr, s_te = signal(0, 60, T), signal(1, 30, T)
ytr, yte = s_tr[:, 1:], s_te[:, 1:]  # predict s_{t+1}
raw_tr, raw_te = s_tr[:, :T, None], s_te[:, :T, None]

# Classical memory: a sliding window of the last W values (W qubits).
for W in (1, 3, 5):
    bw = [Encoding(in_features=W, rotation="RY"), Variational(depth=4, rotation="RY")]
    model = QuantumReservoirRegressor(
        bw,
        state_policy="stateless",
        window=W,
        features="probabilities",
        washout=20,
        random_state=2,
    ).fit(raw_tr, ytr)
    print(f"stateless window={W} ({W}q)   R^2 = {model.score(raw_te, yte):.2f}")

# Quantum fading memory: carry state, no explicit window.
tiled_tr, tiled_te = np.repeat(raw_tr, N_QUBITS, 2), np.repeat(raw_te, N_QUBITS, 2)
leaky = QuantumReservoirRegressor(
    blocks,
    state_policy="leaky",
    leak=0.5,
    features="probabilities",
    washout=20,
    random_state=2,
).fit(tiled_tr, ytr)
print(f"leaky carry (5q)          R^2 = {leaky.score(tiled_te, yte):.2f}")
```

```text
stateless window=1 (1q)   R^2 = 0.01
stateless window=3 (3q)   R^2 = 0.87
stateless window=5 (5q)   R^2 = 0.98
leaky carry (5q)          R^2 = 0.91
```

With no window (`window=1`), the readout sees only the current value, and it
cannot forecast a fast oscillation at all. A short window recovers the local
dynamics: three past values already reach `R² = 0.87`, and five reach `0.98`. The
leaky reservoir gets close too (`0.91`), by carrying a fading state instead of an
explicit window.

## Tuning notes

- **`leak`** balances the length of the memory against its clarity. A smaller
  `leak` (more decay) gives a clearer recall of the recent steps, but a shorter
  memory. A larger `leak` remembers longer, but with more noise.
- **`features`** controls how much information the readout gets.
  `"probabilities"` (used here) is quadratic in the amplitudes, and it captures more
  than `"z"`, which gives one value per qubit. For structure specific to a task,
  pass a Hamiltonian string or a list of observables.
- **The size of the reservoir and `weight_scale`** set how rich the dynamics are.
  More qubits and larger weights generally raise the memory capacity, up to a
  point.
