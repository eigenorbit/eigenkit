# Quantum reservoir computing: learning on a circuit nobody trains

## The problem

Training a quantum circuit is expensive. Every parameter needs a gradient, every
gradient needs extra evaluations of the circuit, and deep circuits run into barren
plateaus, where the gradients vanish altogether.

Reservoir computing avoids this problem entirely. You drive a **fixed, random**
circuit with your input sequence, observe how its state evolves, and fit an ordinary
linear model to what comes out. The circuit is never trained. Only the readout is
trained, and the readout has a closed-form solution. The idea comes from classical
echo state networks and liquid state machines, and
[Fujii and Nakajima](https://arxiv.org/abs/1602.08159) carried it over to quantum
dynamics.

The assumption is that a dynamical system that is rich enough does the hard part
without any training. It mixes the history of the input into a state with many
dimensions, and a linear readout is then enough to extract whatever the task needs.

## Input and output

Two ready-made estimators wrap the whole pipeline:

```python
import numpy as np
from eigenkit import Encoding, Variational, QuantumReservoirRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(20, 40, 5))
y = np.zeros((20, 40))
y[:, 1:] = X[:, :-1, 0]  # predict the first feature of the previous step

blocks = [Encoding(in_features=5, rotation="RY"), Variational(depth=4, rotation="RY")]

model = QuantumReservoirRegressor(
    blocks,
    state_policy="leaky",
    leak=0.5,
    features="probabilities",
    washout=10,
)
model.fit(X, y)  # X is (n_sequences, n_timesteps, n_features)
```

`X` has three dimensions: the sequences, the timesteps and the features. A 2-D array
raises an error. `QuantumReservoirClassifier` is the same estimator with a logistic
readout.

Underneath, `eigenkit.reservoir` holds the parts separately. `run_reservoir` drives
the circuit, the policies decide what carries over between timesteps, the feature
functions turn statevectors into numbers, and `LinearReadout` fits the ridge
regression. All of it is pure NumPy: it needs no PyTorch, and no software
development kit (SDK) of a backend.

## How it works

### Step 1: drive the reservoir

At each timestep, the input is encoded as angles of the gates of the frozen circuit,
and the state advances. The **policy** defines what "advances" means, and it is the
one choice that decides whether the reservoir remembers anything:

- **`Carry`** is a purely unitary carry-over, `ψ_t = U(x_t) ψ_{t-1}`. The norm is
  preserved, and nothing is forgotten.
- **`LeakyCarry`** mixes the carried state with a reset state, and then divides by
  the norm: `ψ̃_t = α U(x_t) ψ_{t-1} + (1 - α) ψ_reset` and `ψ_t = ψ̃_t / ‖ψ̃_t‖`.
  At `α = 1`, it is exactly `Carry`. It stands in for a dissipative reservoir with
  pure states, and it is not a true quantum channel. A true channel would need
  density matrices, which the pure-state backends do not provide.
- **`Stateless`** starts every step from `|0…0⟩`, and encodes a sliding window of the
  last `window` inputs. The memory is entirely classical. Because no state crosses
  from one timestep to the next, the driver evaluates all the timesteps in one
  batched run.

### Step 2: read out features

The trajectory of statevectors becomes a classical matrix of features, with one row
per timestep and one column per number that the extractor produces.

| `features` | What it takes from the state |
|---|---|
| `"z"` | It takes `⟨Z_q⟩` on each qubit, which gives `n` values. |
| `"probabilities"` | It takes `|ψ|²` over all the basis states, which gives `2ⁿ` values. |
| `"amplitudes"` | It takes the real and imaginary parts of `ψ`, which gives `2 · 2ⁿ` values. |
| a Hamiltonian string | It takes `⟨ψ|H_k|ψ⟩`, which gives one value per term. |

More numbers give the linear readout more to fit on, and this is usually what makes
a reservoir work. However, the middle two grow as `2ⁿ`, so a wide reservoir becomes
expensive on the classical side rather than on the quantum side.

### Step 3: fit the readout

This is the part that learns. The readout is a ridge regression with a bias term. It
is fitted on the matrix of features in one step, by solving

```text
W = (XᵀX + λI)⁻¹ XᵀY
```

where the bias column is not penalised, and `λ` is the `ridge` argument. It is a
linear solve rather than an iterative loop, so there is no optimiser and no number
of epochs to set. However, the weights are fitted to the data exactly as in any other
supervised model.

What stays fixed is the quantum side. The weights of the circuit are drawn once from
`random_state`, and they never change. The training only changes the classical
readout.

`washout` drops the first timesteps, before the reservoir has settled, from the fit.
The model still produces predictions for every timestep.

## Guarantees and limits

Nothing here converges, so there is no guarantee of convergence. A reservoir is
judged instead on its **memory capacity**: how far back it can still reproduce its
own input.

Dissipation gives the reservoir this memory, and unitarity does not. The `leaky`
policy gives the curve of fading memory that a reservoir is expected to have.
`stateless` remembers only the current step. The purely unitary `carry` scores
nothing at all on a delay task, even though it is the default. Use
`state_policy="leaky"` with a `leak` of about `0.3` to `0.7` on any temporal task,
and treat the default as the first thing to change. The
[memory-capacity example](../examples/reservoir-memory.md) measures all three.

The other limits are the usual ones. The wider sets of features grow as `2ⁿ`, so the
cost falls on the classical side. The readout is linear, so whatever it cannot
express has to come from the dynamics of the reservoir itself. The frozen weights
are one random draw, so a poor result may come from the particular `random_state`
rather than from the configuration. Vary the seed before you draw any conclusion
about the setup.

## Next

The estimators are covered in the [quantum machine learning (QML) guide](../user-guide/qml.md),
and the parts in
[Reservoir Computing](../user-guide/reservoir.md). The
[memory-capacity example](../examples/reservoir-memory.md) is where the results
above come from. The papers are listed in
[References](references.md#quantum-reservoir-computing).
