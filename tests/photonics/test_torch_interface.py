"""Torch-interface tests for the photonic ring.

Confirms `loss.backward()` flows the exact photonic parameter-shift gradient
(vs. the executor jacobian and vs. torch.autograd.gradcheck), and that a stock
torch optimizer trains a photonic circuit end to end.
"""

import numpy as np
import pytest


pytest.importorskip("torch")
pytest.importorskip("perceval")

import torch  # noqa: E402

from eigenkit.core.ir import Constant, InputRef, Sum, WeightRef  # noqa: E402
from eigenkit.photonics import (  # noqa: E402
    BeamSplitter,
    FockProbability,
    PercevalBackend,
    PhaseShifter,
    PhotonicCircuitSpec,
    PhotonNumber,
)
from eigenkit.photonics.torch import (  # noqa: E402
    PhotonicCircuit,
    _PhotonicExpectation,
)


BALANCED = Constant(np.pi / 2)


def _mzi_spec(phase):
    return PhotonicCircuitSpec(
        n_modes=2,
        n_weights=1 if isinstance(phase, WeightRef) else 0,
        n_inputs=1 if isinstance(phase, InputRef) else 0,
        components=(
            BeamSplitter(modes=(0, 1), theta=BALANCED),
            PhaseShifter(mode=0, phi=phase),
            BeamSplitter(modes=(0, 1), theta=BALANCED),
        ),
        input_state=(1, 0),
    )


def test_backward_matches_parameter_shift_jacobian():
    spec = _mzi_spec(WeightRef(0))
    obs = PhotonNumber(mode=1)
    circ = PhotonicCircuit(spec, obs, init_weights=[0.7])
    circ().backward()
    expected, _ = (
        PercevalBackend().compile(spec).jacobian(np.array([0.7]), np.array([]), obs)
    )
    assert torch.allclose(circ.weights.grad, torch.tensor(expected), atol=1e-9)


def test_gradcheck_weights():
    spec = _mzi_spec(WeightRef(0))
    obs = FockProbability(target=(0, 1))
    compiled = PercevalBackend().compile(spec)
    w = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    x = torch.zeros(0, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda ww: _PhotonicExpectation.apply(ww, x, compiled, obs),
        (w,),
        eps=1e-6,
        atol=1e-4,
        check_batched_grad=False,
    )


def test_gradcheck_inputs():
    spec = _mzi_spec(InputRef(0))
    obs = PhotonNumber(mode=1)
    compiled = PercevalBackend().compile(spec)
    w = torch.zeros(0, dtype=torch.float64)
    x = torch.tensor([0.6], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda xx: _PhotonicExpectation.apply(w, xx, compiled, obs),
        (x,),
        eps=1e-6,
        atol=1e-4,
        check_batched_grad=False,
    )


def test_torch_optimizer_trains_mzi():
    # A stock Adam optimizer drives the photon into mode 1 through backward().
    circ = PhotonicCircuit(
        _mzi_spec(WeightRef(0)), FockProbability(target=(0, 1)), init_weights=[1.5]
    )
    opt = torch.optim.Adam(circ.parameters(), lr=0.1)
    losses = []
    for _ in range(120):
        opt.zero_grad()
        loss = (1.0 - circ()) ** 2
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.1
    assert circ().item() > 0.95


def _encoded_mzi_spec():
    """MZI whose phase is a trainable weight plus an encoded input."""
    return PhotonicCircuitSpec(
        n_modes=2,
        n_weights=1,
        n_inputs=1,
        components=(
            BeamSplitter(modes=(0, 1), theta=BALANCED),
            PhaseShifter(mode=0, phi=Sum((WeightRef(0), InputRef(0)))),
            BeamSplitter(modes=(0, 1), theta=BALANCED),
        ),
        input_state=(1, 0),
    )


def test_backward_skips_input_grad_when_not_needed():
    circ = PhotonicCircuit(_encoded_mzi_spec(), PhotonNumber(mode=1), init_weights=[0.7])
    calls = []
    jacobian = circ._compiled.jacobian

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return jacobian(*args, **kwargs)

    circ._compiled.jacobian = spy
    circ(torch.tensor([0.3], dtype=torch.float64)).backward()

    assert calls[-1]["need_input_grad"] is False
    expected, _ = (
        PercevalBackend()
        .compile(_encoded_mzi_spec())
        .jacobian(np.array([0.7]), np.array([0.3]), PhotonNumber(mode=1))
    )
    assert torch.allclose(circ.weights.grad, torch.tensor(expected), atol=1e-9)


def test_backward_returns_input_grad_when_needed():
    circ = PhotonicCircuit(_encoded_mzi_spec(), PhotonNumber(mode=1), init_weights=[0.7])
    x = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    circ(x).backward()
    # phase = w + x, so d/dx equals d/dw
    assert x.grad is not None
    assert torch.allclose(x.grad, circ.weights.grad, atol=1e-9)


def test_init_weights_length_checked():
    with pytest.raises(ValueError, match="init_weights"):
        PhotonicCircuit(
            _encoded_mzi_spec(), PhotonNumber(mode=1), init_weights=[0.1, 0.2]
        )


def test_forward_rejects_flat_batch():
    circ = PhotonicCircuit(_encoded_mzi_spec(), PhotonNumber(mode=1))
    with pytest.raises(ValueError, match="shape"):
        circ(torch.tensor([0.5, 0.7], dtype=torch.float64))


def test_observable_checked_before_training():
    with pytest.raises(ValueError, match="FockProbability"):
        PhotonicCircuit(_encoded_mzi_spec(), FockProbability(target=(2, 0)))


def test_photonic_circuit_is_exported_lazily():
    import eigenkit.photonics as photonics

    assert "PhotonicCircuit" in photonics.__all__
    assert photonics.PhotonicCircuit is PhotonicCircuit
    namespace = {}
    exec("from eigenkit.photonics import *", namespace)
    assert namespace["PhotonicCircuit"] is PhotonicCircuit
    with pytest.raises(AttributeError, match="no attribute"):
        photonics.NoSuchName  # noqa: B018
