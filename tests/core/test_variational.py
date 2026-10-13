"""Tests for the variational spine (`minimize_expectation`).

Validates that the framework-neutral optimizer reaches known ground states,
works identically across backends (numpy adjoint vs PennyLane parameter-shift,
both torch-free), supports gradient-free and custom optimizers, and rejects
misuse.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from eigenkit.core.backends.numpy_backend import NumpySimulator
from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import CircuitSpec, Gate, InputRef, Observable, WeightRef
from eigenkit.core.observables import from_string
from eigenkit.core.variational import VariationalResult, minimize_expectation


def _ry_ansatz(n_qubits: int) -> CircuitSpec:
    """One RY(w_q) per qubit — enough to reach the all-|1> state."""
    return CircuitSpec(
        n_qubits=n_qubits,
        n_weights=n_qubits,
        n_inputs=0,
        gates=tuple(Gate("RY", (q,), (WeightRef(q),)) for q in range(n_qubits)),
    )


@pytest.fixture(params=["numpy", "pennylane"])
def provider(request):
    """Run the spine on each available simulator backend. numpy uses adjoint
    gradients; pennylane uses parameter-shift — both via the numpy spine."""
    if request.param == "pennylane":
        pytest.importorskip("pennylane")
    return request.param


class TestMinimizeExpectation:
    def test_single_qubit_ground_state(self, provider):
        # RY(theta)|0>, <Z> = cos(theta); minimum -1 at theta = pi.
        circuit = Circuit(_ry_ansatz(1), Observable.z(0), provider, "numpy")
        result = minimize_expectation(circuit, initial_parameters=np.array([0.3]))
        assert result.success
        assert result.optimal_value < -0.999
        assert np.isclose(
            abs(result.optimal_parameters[0]) % (2 * np.pi), np.pi, atol=1e-2
        )

    def test_two_qubit_hamiltonian(self, provider):
        # <0.5 Z0 + 0.5 Z1>; separable minimum -1 at theta0 = theta1 = pi.
        circuit = Circuit(
            _ry_ansatz(2), from_string("0.5 Z0 + 0.5 Z1"), provider, "numpy"
        )
        result = minimize_expectation(circuit, initial_parameters=np.array([0.2, -0.1]))
        assert result.optimal_value < -0.999

    def test_backends_agree(self):
        # The whole point: numpy (adjoint) and pennylane (parameter-shift)
        # reach the same optimum from the same start, both torch-free.
        pytest.importorskip("pennylane")
        spec, obs, x0 = (
            _ry_ansatz(2),
            from_string("0.7 Z0 + 0.3 Z1"),
            np.array([0.4, 0.9]),
        )
        numpy_result = minimize_expectation(
            Circuit(spec, obs, "numpy", "numpy"), initial_parameters=x0
        )
        pl_result = minimize_expectation(
            Circuit(spec, obs, "pennylane", "numpy"), initial_parameters=x0
        )
        assert np.isclose(numpy_result.optimal_value, pl_result.optimal_value, atol=1e-6)


class TestOptimizers:
    def test_gradient_free_cobyla(self):
        circuit = Circuit(_ry_ansatz(1), Observable.z(0), "numpy", "numpy")
        result = minimize_expectation(
            circuit, initial_parameters=np.array([0.3]), optimizer="COBYLA"
        )
        assert result.optimal_value < -0.99

    def test_custom_optimizer_callable(self):
        # Plain numpy gradient descent — exercises the pluggable callable path
        # (the same hook a torch/optax optimizer would use).
        circuit = Circuit(_ry_ansatz(1), Observable.z(0), "numpy", "numpy")

        def gradient_descent(fun, x0, jac, steps=300, lr=0.1):
            x = np.array(x0, dtype=np.float64)
            for _ in range(steps):
                x = x - lr * jac(x)
            return x

        result = minimize_expectation(
            circuit, initial_parameters=np.array([0.3]), optimizer=gradient_descent
        )
        assert result.optimal_value < -0.999

    def test_records_history(self):
        circuit = Circuit(_ry_ansatz(1), Observable.z(0), "numpy", "numpy")
        result = minimize_expectation(circuit, initial_parameters=np.array([0.5]))
        assert isinstance(result, VariationalResult)
        assert len(result.history) >= 1
        assert result.history[-1] <= result.history[0]
        assert result.n_evaluations >= 1


class TestValidation:
    def test_rejects_circuit_with_inputs(self):
        spec = CircuitSpec(
            n_qubits=1, n_weights=0, n_inputs=1, gates=(Gate("RY", (0,), (InputRef(0),)),)
        )
        circuit = Circuit(spec, Observable.z(0), "numpy", "numpy")
        with pytest.raises(ValueError, match="n_inputs"):
            minimize_expectation(circuit)

    def test_rejects_multiple_observables(self):
        circuit = Circuit(
            _ry_ansatz(1), [Observable.z(0), Observable.x(0)], "numpy", "numpy"
        )
        with pytest.raises(ValueError, match="one observable"):
            minimize_expectation(circuit)

    def test_rejects_bad_initial_shape(self):
        circuit = Circuit(_ry_ansatz(2), from_string("Z0 + Z1"), "numpy", "numpy")
        with pytest.raises(ValueError, match="shape"):
            minimize_expectation(circuit, initial_parameters=np.array([0.1]))

    def test_nondiff_backend_requires_gradient_free(self):
        # A fresh sim instance flipped to gradient_method="none" (no registry
        # leak): a gradient-based method must be rejected, COBYLA must work.
        sim = NumpySimulator()
        sim.gradient_method = "none"
        circuit = Circuit(_ry_ansatz(1), Observable.z(0), sim, "numpy")
        with pytest.raises(ValueError, match="gradient-free"):
            minimize_expectation(circuit, optimizer="L-BFGS-B")
        result = minimize_expectation(
            circuit, initial_parameters=np.array([0.3]), optimizer="COBYLA"
        )
        assert result.optimal_value < -0.99


class TestCustomOptimizerConvergence:
    """`success` must not be an assumption on the custom-optimizer branch.

    A callable returning only a parameter vector has no channel to report
    convergence, so the verdict is inferred from the cost: an optimizer that
    does nothing, or one that makes the cost worse, must not come back
    reporting success. `Converged` promises that the field answers "did this
    converge", and a hardcoded `success=True` would break that promise.
    """

    @staticmethod
    def _circuit():
        return Circuit(_ry_ansatz(2), Observable.z(0), "numpy", "numpy")

    def test_an_optimizer_that_does_nothing_does_not_report_success(self):
        result = minimize_expectation(self._circuit(), optimizer=lambda cost, t0, g: t0)
        assert result.success is False

    def test_an_optimizer_that_makes_it_worse_does_not_report_success(self):
        # Maximises instead of minimising. A fixed offset will not do: the
        # ansatz is periodic in its angles, so `theta0 + 10` can land on a
        # better point, and then reporting success would be correct.
        def maximises(cost, theta0, gradient):
            candidates = [theta0 + step for step in (0.0, 0.5, 1.0, 1.5, 2.0)]
            return max(candidates, key=cost)

        result = minimize_expectation(self._circuit(), optimizer=maximises)
        assert result.success is False

    def test_an_optimizer_that_improves_reports_success(self):
        def real(cost, theta0, gradient):
            return scipy_minimize(cost, theta0, method="COBYLA").x

        result = minimize_expectation(self._circuit(), optimizer=real)
        assert result.success is True
        assert result.optimal_value < 0.0

    def test_an_optimizer_may_report_convergence_itself(self):
        def claims(flag):
            def optimizer(cost, theta0, gradient):
                return scipy_minimize(cost, theta0, method="COBYLA").x, flag

            return optimizer

        assert minimize_expectation(self._circuit(), optimizer=claims(True)).success
        # Its own word wins over the inference, in both directions.
        assert not minimize_expectation(self._circuit(), optimizer=claims(False)).success

    def test_a_reported_flag_is_named_in_the_message(self):
        def optimizer(cost, theta0, gradient):
            return theta0, True

        assert (
            "reported convergence"
            in minimize_expectation(self._circuit(), optimizer=optimizer).message
        )

    def test_an_inferred_verdict_shows_the_cost_either_side(self):
        result = minimize_expectation(self._circuit(), optimizer=lambda cost, t0, g: t0)
        assert "->" in result.message

    def test_bookkeeping_evaluations_are_not_counted_as_the_optimizers(self):
        # The starting and final costs are ours. An optimizer that never
        # called the cost function must not look as though it had.
        result = minimize_expectation(self._circuit(), optimizer=lambda cost, t0, g: t0)
        assert result.n_evaluations == 0
        assert result.history == ()

    def test_a_two_element_parameter_vector_is_not_mistaken_for_a_flag(self):
        # Only a tuple whose second element is a bool is read as a report.
        circuit = Circuit(_ry_ansatz(2), Observable.z(0), "numpy", "numpy")
        n = circuit.spec.n_weights
        result = minimize_expectation(circuit, optimizer=lambda cost, t0, g: np.zeros(n))
        assert result.optimal_parameters.shape == (n,)
