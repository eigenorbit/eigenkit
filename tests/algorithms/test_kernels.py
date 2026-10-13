import numpy as np
import pytest

from eigenkit.qml import QuantumKernel


def test_kernel_matrix_shape():
    from eigenkit.qml import QuantumKernel

    X = np.random.default_rng(0).standard_normal((5, 4))
    K = QuantumKernel(n_features=4, feature_map="Angle", reps=1).compute_kernel_matrix(X)
    assert K.shape == (5, 5)


def test_kernel_matrix_rectangular():
    from eigenkit.qml import QuantumKernel

    X = np.random.default_rng(0).standard_normal((4, 4))
    Y = np.random.default_rng(1).standard_normal((3, 4))
    K = QuantumKernel(n_features=4, feature_map="Angle", reps=1).compute_kernel_matrix(
        X, Y
    )
    assert K.shape == (4, 3)


def test_kernel_matrix_symmetric():
    from eigenkit.qml import QuantumKernel

    X = np.random.default_rng(0).standard_normal((4, 4))
    K = QuantumKernel(n_features=4, feature_map="Angle", reps=1).compute_kernel_matrix(X)
    assert np.allclose(K, K.T, atol=1e-6)


def test_kernel_diagonal_ones():
    from eigenkit.qml import QuantumKernel

    X = np.random.default_rng(0).standard_normal((4, 4))
    K = QuantumKernel(n_features=4, feature_map="Angle", reps=1).compute_kernel_matrix(X)
    assert np.allclose(np.diag(K), 1.0, atol=1e-6)


def test_angle_cnot_feature_map():
    from eigenkit.qml import QuantumKernel

    X = np.random.default_rng(0).standard_normal((3, 4))
    K = QuantumKernel(
        n_features=4, feature_map="AngleCNOT", reps=2
    ).compute_kernel_matrix(X)
    assert K.shape == (3, 3)


def test_vectorized_matrix_matches_per_pair():
    # The vectorized compute_kernel_matrix must equal the per-pair definition.
    from eigenkit.qml import QuantumKernel

    qk = QuantumKernel(n_features=4, feature_map="AngleCNOT", reps=2, provider="numpy")
    X = np.random.default_rng(0).standard_normal((6, 4))
    Y = np.random.default_rng(1).standard_normal((4, 4))
    matrix = qk.compute_kernel_matrix(X, Y)
    per_pair = np.array(
        [[qk.kernel_value(X[i], Y[j]) for j in range(len(Y))] for i in range(len(X))]
    )
    assert np.allclose(matrix, per_pair, atol=1e-12)


def test_gram_fast_path_matches_inversion_definition():
    # On a statevector backend the Gram fast path (one batched evolution + BLAS
    # contraction) must reproduce the inversion-circuit fidelity, bit-identical.
    from eigenkit.qml import QuantumKernel

    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 4))
    Y = rng.standard_normal((4, 4))
    for feature_map in ("Angle", "AngleCNOT"):
        qk = QuantumKernel(
            n_features=4, feature_map=feature_map, reps=2, provider="numpy"
        )
        assert qk._statevector_feature_map() is not None  # fast path on numpy

        gram_sq = qk.compute_kernel_matrix(X)
        gram_rect = qk.compute_kernel_matrix(X, Y)
        inv_sq = np.array(
            [[qk.kernel_value(X[i], X[j]) for j in range(len(X))] for i in range(len(X))]
        )
        inv_rect = np.array(
            [[qk.kernel_value(X[i], Y[j]) for j in range(len(Y))] for i in range(len(X))]
        )
        assert np.allclose(gram_sq, inv_sq, atol=1e-11)
        assert np.allclose(gram_rect, inv_rect, atol=1e-11)


def test_batched_zero_state_probabilities_matches_loop():
    # The numpy circuit's batched primitive equals looping the single one.
    from eigenkit import Circuit, Encoding, Variational

    circuit = Circuit(
        [Encoding(in_features=3, rotation="RX"), Variational(depth=1)],
        provider="numpy",
        interface="numpy",
    )
    weights = np.zeros(circuit.spec.n_weights)
    batch = np.random.default_rng(0).uniform(0, np.pi, size=(8, 3))
    batched = circuit.zero_state_probabilities(weights, batch)
    looped = np.array([circuit.zero_state_probability(weights, row) for row in batch])
    assert batched.shape == (8,)
    assert np.allclose(batched, looped, atol=1e-12)


class TestProviderDefault:
    """`QuantumKernel` defaults to the one backend a bare install always has.

    The qml tier is contractually importable without PennyLane, so a kernel
    that defaulted to it could be constructed and not run. A kernel entry is a
    fidelity, so the choice costs no accuracy.
    """

    def test_the_default_is_numpy(self):
        assert QuantumKernel(n_features=2).provider == "numpy"

    @pytest.mark.pennylane
    def test_the_instance_reports_the_provider_it_was_given(self):
        assert QuantumKernel(n_features=2, provider="pennylane").provider == "pennylane"

    def test_none_is_not_a_spelling_of_the_default(self):
        with pytest.raises(TypeError, match="provider spec None"):
            QuantumKernel(n_features=2, provider=None)

    @pytest.mark.pennylane
    def test_numpy_and_pennylane_agree(self):
        X = np.random.default_rng(0).uniform(0.0, 1.0, size=(5, 3))
        on_numpy = QuantumKernel(
            n_features=3, feature_map="Angle", reps=1, provider="numpy"
        ).compute_kernel_matrix(X)
        on_pennylane = QuantumKernel(
            n_features=3, feature_map="Angle", reps=1, provider="pennylane"
        ).compute_kernel_matrix(X)
        np.testing.assert_allclose(on_numpy, on_pennylane, atol=1e-12)


class TestTheFastPathProbeCostsNothingToConstruct:
    """Deciding the fast path by trying `statevector()` is right -- a
    shots-configured backend has the method and raises when called -- but the
    probe *runs a circuit*, so it is deferred to first use: done in `__init__`,
    building a `QSVC` against a metered backend would pay for a submission
    before `fit()`."""

    def test_construction_does_not_probe(self):
        qk = QuantumKernel(n_features=3, feature_map="Angle", reps=1, provider="numpy")
        assert qk._fast_path_probed is False

    def test_the_probe_runs_once_however_often_it_is_asked(self):
        qk = QuantumKernel(n_features=3, feature_map="Angle", reps=1, provider="numpy")
        calls = []
        real = qk.feature_map_circuit.statevector

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        qk.feature_map_circuit.statevector = counting
        assert qk._statevector_feature_map() is not None
        assert qk._statevector_feature_map() is not None
        assert len(calls) == 1, f"probed {len(calls)} times"

    def test_first_use_is_what_probes(self):
        qk = QuantumKernel(n_features=3, feature_map="Angle", reps=1, provider="numpy")
        X = np.random.default_rng(0).standard_normal((3, 3))
        qk.compute_kernel_matrix(X)
        assert qk._fast_path_probed is True

    def test_an_unexpected_backend_error_falls_back_instead_of_raising(self):
        """A backend raising `QiskitError`, `RuntimeError` or its own SDK type
        must fall back to the inversion circuit, not break the call -- the
        exact failure the probe exists to prevent. Catching only
        `(NotImplementedError, ValueError)` would let those through."""

        class SdkSpecificError(Exception):
            pass

        def raising(*args, **kwargs):
            raise SdkSpecificError("this backend has no analytic state")

        X = np.random.default_rng(0).standard_normal((4, 3))
        reference = QuantumKernel(
            n_features=3, feature_map="Angle", reps=1, provider="numpy"
        ).compute_kernel_matrix(X)

        qk = QuantumKernel(n_features=3, feature_map="Angle", reps=1, provider="numpy")
        qk.feature_map_circuit.statevector = raising

        matrix = qk.compute_kernel_matrix(X)

        assert qk._statevector_feature_map() is None
        np.testing.assert_allclose(matrix, reference, atol=1e-12)


# ---- custom and trainable feature maps ----


def _zz_map(reps=2, trainable=False):
    """Qiskit's ZZFeatureMap written with the builder, optionally preceded by
    one shared trainable RY angle."""
    from eigenkit.core.ir import Constant, Multiply, Negate, Scale, Sum

    pi = float(np.pi)

    def emit(qc, x, w):
        if trainable:
            for q in range(len(x)):
                qc.ry(w[0], qubit=q)
        for _ in range(reps):
            for q in range(len(x)):
                qc.h(q)
            for q in range(len(x)):
                qc.phase_shift(Scale(x[q], 2.0), qubit=q)
            for a in range(len(x)):
                for b in range(a + 1, len(x)):
                    qc.cnot(a, b)
                    angle = Scale(
                        Multiply(
                            Sum((Constant(pi), Negate(x[a]))),
                            Sum((Constant(pi), Negate(x[b]))),
                        ),
                        2.0,
                    )
                    qc.phase_shift(angle, qubit=b)
                    qc.cnot(a, b)

    return emit


def _force_inversion(kernel):
    """Disable the statevector fast path so the inversion circuit is used."""

    def raising(*args, **kwargs):
        raise RuntimeError("no analytic state")

    kernel.feature_map_circuit.statevector = raising
    return kernel


class TestCustomFeatureMaps:
    X = np.random.default_rng(3).uniform(0, np.pi, size=(6, 3))
    Y = np.random.default_rng(4).uniform(0, np.pi, size=(4, 3))

    def test_a_spec_reproduces_the_builtin_map(self):
        from eigenkit.qml._kernels import _build_feature_map_spec

        builtin = QuantumKernel(n_features=3, feature_map="AngleCNOT", reps=2)
        custom = QuantumKernel(
            n_features=3, feature_map=_build_feature_map_spec(3, "AngleCNOT", 2)
        )
        np.testing.assert_allclose(
            custom.compute_kernel_matrix(self.X, self.Y),
            builtin.compute_kernel_matrix(self.X, self.Y),
            atol=1e-12,
        )

    def test_a_callable_reproduces_the_builtin_map(self):
        def z_map(qc, x, w):
            for q in range(len(x)):
                qc.rx(x[q], qubit=q)

        builtin = QuantumKernel(n_features=3, feature_map="Angle", reps=1)
        custom = QuantumKernel(n_features=3, feature_map=z_map)
        np.testing.assert_allclose(
            custom.compute_kernel_matrix(self.X),
            builtin.compute_kernel_matrix(self.X),
            atol=1e-12,
        )

    def test_blocks_are_accepted(self):
        from eigenkit import Encoding

        builtin = QuantumKernel(n_features=3, feature_map="Angle", reps=1)
        from_block = QuantumKernel(n_features=3, feature_map=Encoding(3, rotation="RX"))
        from_list = QuantumKernel(n_features=3, feature_map=[Encoding(3, rotation="RX")])
        for kernel in (from_block, from_list):
            np.testing.assert_allclose(
                kernel.compute_kernel_matrix(self.X),
                builtin.compute_kernel_matrix(self.X),
                atol=1e-12,
            )

    def test_the_inversion_circuit_equals_the_statevector_fast_path(self):
        fast = QuantumKernel(
            n_features=3, feature_map=_zz_map(trainable=True), n_weights=1, weights=[0.4]
        )
        slow = _force_inversion(
            QuantumKernel(
                n_features=3,
                feature_map=_zz_map(trainable=True),
                n_weights=1,
                weights=[0.4],
            )
        )
        np.testing.assert_allclose(
            slow.compute_kernel_matrix(self.X, self.Y),
            fast.compute_kernel_matrix(self.X, self.Y),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            slow.compute_kernel_matrix(self.X),
            fast.compute_kernel_matrix(self.X),
            atol=1e-12,
        )

    def test_zz_map_matches_qiskit(self):
        pytest.importorskip("qiskit")
        from qiskit.circuit.library import zz_feature_map
        from qiskit.quantum_info import Statevector

        X = self.X[:, :2]
        kernel = QuantumKernel(n_features=2, feature_map=_zz_map(reps=2))
        fmap = zz_feature_map(2, reps=2)
        states = np.array([Statevector(fmap.assign_parameters(x)).data for x in X])
        expected = np.abs(states.conj() @ states.T) ** 2
        np.testing.assert_allclose(kernel.compute_kernel_matrix(X), expected, atol=1e-12)

    def test_feature_count_must_match_the_map(self):
        with pytest.raises(ValueError, match="reads 3 input"):
            QuantumKernel(n_features=2, feature_map=_build_three_input_spec())

    def test_unknown_feature_map_type_is_rejected(self):
        with pytest.raises(TypeError, match="not supported"):
            QuantumKernel(n_features=2, feature_map=42)

    def test_unknown_feature_map_name_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown feature_map"):
            QuantumKernel(n_features=2, feature_map="ZZ")

    def test_kernel_spec_has_twice_the_inputs_and_shares_weights(self):
        kernel = QuantumKernel(
            n_features=3, feature_map=_zz_map(trainable=True), n_weights=1
        )
        assert kernel.kernel_spec.n_inputs == 6
        assert kernel.kernel_spec.n_weights == 1
        assert kernel.n_weights == 1
        assert kernel.n_qubits == 3

    def test_drawing_needs_no_compile(self):
        kernel = QuantumKernel(n_features=2, feature_map=_zz_map(reps=1))
        assert isinstance(kernel.draw(), str)
        assert isinstance(kernel.draw(inversion=True), str)
        assert kernel._inversion is None and kernel._forward is None


def _build_three_input_spec():
    from eigenkit import CircuitBuilder

    qc = CircuitBuilder(3)
    x = qc.add_inputs(3)
    for q in range(3):
        qc.rx(x[q], q)
    return qc.build()


class TestTrainableKernels:
    X = np.random.default_rng(5).uniform(0, np.pi, size=(5, 2))

    def _kernel(self, **kwargs):
        return QuantumKernel(
            n_features=2,
            feature_map=_zz_map(reps=1, trainable=True),
            n_weights=1,
            **kwargs,
        )

    def test_weights_change_the_kernel(self):
        kernel = self._kernel()
        K0 = kernel.compute_kernel_matrix(self.X, weights=[0.0])
        K1 = kernel.compute_kernel_matrix(self.X, weights=[1.0])
        assert not np.allclose(K0, K1)

    def test_default_weights_are_the_instance_weights_then_zeros(self):
        kernel = self._kernel()
        np.testing.assert_allclose(
            kernel.compute_kernel_matrix(self.X),
            kernel.compute_kernel_matrix(self.X, weights=[0.0]),
        )
        trained = kernel.with_weights([0.7])
        np.testing.assert_allclose(
            trained.compute_kernel_matrix(self.X),
            kernel.compute_kernel_matrix(self.X, weights=[0.7]),
        )
        np.testing.assert_allclose(trained.weights, [0.7])

    def test_wrong_weight_count_is_rejected(self):
        with pytest.raises(ValueError, match="1 weight"):
            self._kernel(weights=[0.1, 0.2])

    def test_n_weights_is_checked_for_non_callable_maps(self):
        with pytest.raises(ValueError, match="declares 0"):
            QuantumKernel(n_features=2, feature_map="Angle", n_weights=2)


class TestScaleAndClone:
    X = np.random.default_rng(6).uniform(0, 1, size=(5, 3))

    def test_scale_multiplies_the_inputs(self):
        scaled = QuantumKernel(n_features=3, feature_map="AngleCNOT", scale=2.5)
        plain = QuantumKernel(n_features=3, feature_map="AngleCNOT")
        np.testing.assert_allclose(
            scaled.compute_kernel_matrix(self.X),
            plain.compute_kernel_matrix(2.5 * self.X),
            atol=1e-12,
        )
        assert scaled.kernel_value(self.X[0], self.X[1]) == pytest.approx(
            plain.kernel_value(2.5 * self.X[0], 2.5 * self.X[1]), abs=1e-12
        )

    def test_clone_is_fresh_and_keeps_the_definition(self):
        from sklearn.base import clone

        kernel = QuantumKernel(
            n_features=2,
            feature_map=_zz_map(reps=1, trainable=True),
            n_weights=1,
            weights=[0.3],
            scale=0.5,
        )
        kernel.compute_kernel_matrix(self.X[:, :2])
        copy = clone(kernel)
        assert copy is not kernel
        assert copy._inversion is None and copy._forward is None
        assert copy.scale == 0.5
        np.testing.assert_allclose(copy.weights, [0.3])
        np.testing.assert_allclose(
            copy.compute_kernel_matrix(self.X[:, :2]),
            kernel.compute_kernel_matrix(self.X[:, :2]),
        )

    def test_a_compiled_kernel_pickles(self):
        import pickle

        kernel = QuantumKernel(n_features=3, feature_map="AngleCNOT")
        K = kernel.compute_kernel_matrix(self.X)
        restored = pickle.loads(pickle.dumps(kernel))
        np.testing.assert_allclose(restored.compute_kernel_matrix(self.X), K)

    def test_json_round_trip(self):
        kernel = QuantumKernel(
            n_features=2,
            feature_map=_zz_map(reps=1, trainable=True),
            n_weights=1,
            weights=[0.3],
            scale=0.5,
        )
        restored = QuantumKernel.from_json_dict(kernel.to_json_dict())
        np.testing.assert_allclose(
            restored.compute_kernel_matrix(self.X[:, :2]),
            kernel.compute_kernel_matrix(self.X[:, :2]),
        )


class TestInversionFallback:
    X = np.random.default_rng(7).uniform(0, np.pi, size=(6, 3))

    def test_a_gram_matrix_measures_only_the_upper_triangle(self):
        kernel = _force_inversion(QuantumKernel(n_features=3, feature_map="AngleCNOT"))
        sizes = []
        real = kernel.circuit.zero_state_probabilities

        def recording(weights, inputs):
            sizes.append(len(inputs))
            return real(weights, inputs)

        kernel.circuit.zero_state_probabilities = recording
        K = kernel.compute_kernel_matrix(self.X)
        assert sizes == [6, 6, 3], (
            "the 15 pairs above the diagonal, a row's worth per call"
        )
        np.testing.assert_allclose(np.diag(K), 1.0)
        np.testing.assert_allclose(K, K.T)

    def test_a_rectangular_matrix_is_one_call_per_row(self):
        kernel = _force_inversion(QuantumKernel(n_features=3, feature_map="AngleCNOT"))
        sizes = []
        real = kernel.circuit.zero_state_probabilities

        def recording(weights, inputs):
            sizes.append(len(inputs))
            return real(weights, inputs)

        kernel.circuit.zero_state_probabilities = recording
        kernel.compute_kernel_matrix(self.X, self.X[:4])
        assert sizes == [4] * 6, "memory stays O(n_y) on a local simulator"

    def test_psd_projection(self):
        from eigenkit.qml._kernels import project_psd

        rng = np.random.default_rng(0)
        K = QuantumKernel(n_features=3).compute_kernel_matrix(self.X)
        noisy = K + rng.normal(0, 0.05, size=K.shape)
        noisy = 0.5 * (noisy + noisy.T)
        projected = project_psd(noisy)
        assert np.linalg.eigvalsh(projected).min() > -1e-10
        np.testing.assert_allclose(projected, projected.T)

    def test_estimate_cost_on_a_statevector_backend(self):
        info = QuantumKernel(n_features=3).estimate_cost(self.X)
        assert info["method"] == "statevector"
        assert info["n_circuits"] == 0


def test_kernel_job_rejects_a_result_of_the_wrong_size():
    from eigenkit.qml._kernels import KernelJob

    class Handle:
        def result(self):
            return [0.42]

    with pytest.raises(ValueError, match="expected 6 kernel entries, got 1"):
        KernelJob(Handle(), n_x=4, n_y=None).result()
