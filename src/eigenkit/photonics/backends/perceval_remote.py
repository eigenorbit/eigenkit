"""The Quandela Cloud executor, which runs photonic circuits on remote hardware or
cloud simulators.

It sends circuits to a Quandela Cloud platform through Perceval's
`RemoteProcessor`. The platform is a real photonic quantum processing unit (QPU),
such as `"qpu:ascella"`, or a cloud simulator (`"sim:..."`). The results are
*sampled* (counts of shots), so `run` returns an **estimated** Fock distribution,
and gradients are not available, as with the qubit QPU backends. Perceval 1.3
keeps `RemoteProcessor` and `RemoteJob` in `perceval.runtime.legacy` (they are
still available as `pcvl.*`). Moving to Perceval's newer interface is planned.
The executor follows the conventions of the library for QPUs:

- It issues a `CostWarning` at the first submission, because the backend may
  charge for use.
- `submit(...)` returns a `RemoteJobHandle` without waiting for the job, and the
  handle has `.result()` and `.status()`.
- `dry_run_info()` describes the prepared job **offline**, before anything is
  paid for.

The access token comes from the `token` argument, then from `$QUANDELA_TOKEN` or
`$PERCEVAL_TOKEN`, then from the token that Perceval itself would use
(`pcvl.RemoteConfig`, which reads its environment variable, then its saved
configuration). It is never written in the code and never logged. The
`RemoteProcessor`, which connects to the cloud, is built at the first `run` or
`submit` and reused after that, so `compile()` and `dry_run_info()` need neither
a network connection nor a token.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import warnings

import numpy as np

from ...core.ir import resolve
from ...core.provider import CostWarning
from ..ir import BeamSplitter, PhaseShifter
from .perceval_backend import build_perceval_circuit, fock_tuple


_TOKEN_ENV = ("QUANDELA_TOKEN", "PERCEVAL_TOKEN")


def _saved_token():
    """The token Perceval itself would use (its env var, then its saved config), or None."""
    try:
        import perceval as pcvl
    except ImportError:
        return None
    return pcvl.RemoteConfig().get_token() or None


def _resolve_token(token):
    """Explicit ``token`` wins, then ``$QUANDELA_TOKEN`` / ``$PERCEVAL_TOKEN``, then
    Perceval's own saved token; raise if none."""
    if token:
        return token
    for var in _TOKEN_ENV:
        value = os.environ.get(var)
        if value:
            return value
    saved = _saved_token()
    if saved:
        return saved
    raise RuntimeError(
        "no Quandela Cloud token found: pass token=..., set $QUANDELA_TOKEN, or save one "
        "with pcvl.RemoteConfig"
    )


def _counts_to_distribution(counts, n_modes):
    """Normalize a perceval ``BSCount`` (Fock state -> shot count) to a probability dict."""
    total = sum(counts.values())
    if total == 0:
        # An empty dict would read as a valid measurement (every expectation 0.0).
        raise RuntimeError(
            "no shots survived post-selection on the detected photon count; raise "
            "shots or lower min_detected_photons"
        )
    return {fock_tuple(state, n_modes): count / total for state, count in counts.items()}


# Perceval dispatches some methods (for example `with_input`, and the serializer)
# through state that all instances share, so two threads that build a job at the same
# time can swap input states or corrupt an experiment. One build takes milliseconds.
_BUILD_LOCK = threading.Lock()


class _CaptureHandler:
    """A stand-in for Perceval's cloud handler that opens no connection.

    It answers the platform details that the caller already holds, and keeps the
    body of the job instead of sending it.
    """

    def __init__(self, platform: str, platform_details: dict) -> None:
        self.name = platform
        self.url = ""
        self.token = ""
        self.proxies: dict = {}
        self._platform_details = platform_details
        self.body: dict | None = None

    def fetch_platform_details(self) -> dict:
        return self._platform_details

    def create_job(self, cloud_data: dict) -> str:
        self.body = cloud_data
        return "not-submitted"


def _parametric_circuit(spec):
    """Build the circuit with one named Perceval parameter per angle.

    Returns the circuit and the angle expressions, in the order of the parameters
    `a0`, `a1`, and so on. A batch then needs one number per angle and per row,
    whatever the expressions are.
    """
    import perceval as pcvl

    circ = pcvl.Circuit(spec.n_modes)
    angles = []

    def placeholder(expression):
        angles.append(expression)
        return pcvl.P(f"a{len(angles) - 1}")

    for comp in spec.components:
        if isinstance(comp, PhaseShifter):
            circ.add(comp.mode, pcvl.PS(placeholder(comp.phi)))
        elif isinstance(comp, BeamSplitter):
            theta = placeholder(comp.theta)
            circ.add(
                tuple(comp.modes), pcvl.BS(theta=theta, phi_tr=placeholder(comp.phi))
            )
        else:
            circ.add(0, pcvl.PERM(list(comp.perm)))
    return circ, angles


def cloud_job_body(
    spec,
    weights,
    inputs,
    *,
    platform: str,
    platform_details: dict,
    shots: int,
    min_detected_photons: int | None = None,
    job_name: str,
) -> dict:
    """Build the body of a Quandela Cloud job, without a connection or a token.

    The body is the JSON that Perceval itself sends to create a job (`POST
    /api/jobs`). A service that holds the token can send it later. The job samples
    the circuit once for each row of `inputs`, so a batch is one job.

    Several threads can call this function at the same time: the calls into Perceval
    run one after another.

    Args:
        spec (PhotonicCircuitSpec): The circuit to run.
        weights (np.ndarray): The 1-D weight vector, used for every row.
        inputs (np.ndarray): One 1-D input vector, or a 2-D array with one input
            vector in each row. It needs at least one row.
        platform (str): The name of the Quandela Cloud platform, for example
            `"sim:slos"`.
        platform_details (dict): The description of the platform, as the service
            returns it (`GET /api/platforms/{name}/processor`). Perceval needs it to
            prepare a job.
        shots (int): The number of samples asked for in each row. It is also the
            limit on the shots of each row.
        min_detected_photons (int | None): Keep only the shots that detect at
            least this many photons. The default is the number of photons in the
            input state.
        job_name (str): The name of the job on Quandela Cloud.

    Returns:
        The body of the request, as a dictionary that `json.dumps` accepts.

    Raises:
        ValueError: If `weights` is not 1-D with the length that the spec declares,
            if `inputs` is not one or more rows of that length, or if the platform
            does not offer the `sample_count` command.
    """
    import perceval as pcvl
    from perceval.algorithm import Sampler

    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (spec.n_weights,):
        raise ValueError(
            f"weights has shape {weights.shape} but the spec needs ({spec.n_weights},)"
        )
    rows = np.asarray(inputs, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[np.newaxis, :]
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != spec.n_inputs:
        raise ValueError(
            f"inputs has shape {np.shape(inputs)} but the spec needs ({spec.n_inputs},) "
            f"or (rows, {spec.n_inputs}) with at least one row"
        )

    shots = int(shots)
    handler = _CaptureHandler(platform, platform_details)
    with _BUILD_LOCK:
        processor = pcvl.RemoteProcessor(rpc_handler=handler)
        circuit, angles = _parametric_circuit(spec)
        processor.set_circuit(circuit)
        processor.min_detected_photons_filter(
            spec.n_photons if min_detected_photons is None else min_detected_photons
        )
        processor.with_input(pcvl.BasicState(list(spec.input_state)))
        sampler = Sampler(processor, max_shots_per_call=shots)
        sampler.add_iteration_list(
            [
                {
                    "circuit_params": {
                        f"a{index}": float(resolve(angle, weights, row))
                        for index, angle in enumerate(angles)
                    },
                    "max_samples": shots,
                    "max_shots": shots,
                }
                for row in rows
            ]
        )
        job = sampler.sample_count
        job.name = job_name
        job.execute_async(shots)
    body = handler.body
    # Without a native `sample_count`, Perceval asks for another command and converts
    # the result itself; `cloud_job_counts` reads the service's own counts only.
    if body["payload"]["command"] != "sample_count":
        raise ValueError(
            f"the platform {platform!r} does not offer the sample_count command"
        )
    return body


def cloud_job_counts(result: dict, n_modes: int) -> list[dict]:
    """Read the counts of a finished Quandela Cloud job.

    Args:
        result (dict): The result of the job, as the service returns it (`GET
            /api/jobs/{id}/result`).
        n_modes (int): The number of modes of the circuit.

    Returns:
        One dictionary `{photon_counts: count}` for each row of the job, in the
            order of the rows. A job with a single run gives a list of one. A row
            in which no shot passed the photon filter is an empty dictionary.

    Raises:
        ValueError: If the result holds no counts, for example because the job
            failed or was cancelled.
    """
    from perceval.serialization import deserialize

    inner = result.get("results") if isinstance(result, dict) else None
    if isinstance(inner, str):
        with contextlib.suppress(ValueError):
            inner = json.loads(inner)
    if isinstance(inner, dict) and "results_list" in inner:
        entries = inner["results_list"]
    elif isinstance(inner, dict) and "results" in inner:
        entries = [inner]
    else:
        raise ValueError(f"the job result has no counts: {str(inner)[:200]!r}")
    return [
        {
            fock_tuple(state, n_modes): int(count)
            for state, count in deserialize(entry["results"]).items()
        }
        for entry in entries
    ]


class RemoteJobHandle:
    """A handle to a Quandela Cloud job that is running.

    It is the linear-optics counterpart of the qubit `JobHandle`.
    `RemoteCompiled.submit` returns it, and it is not meant to be built directly.

    Args:
        job (RemoteJob): The Perceval job that was submitted.
        n_modes (int): The number of modes of the circuit, used to read the sampled
            Fock states.
        poll_interval (float): The time, in seconds, between two checks of the
            status in `result()`. Default: `2.0`.
        poll_timeout_seconds (float): How long, in seconds, `result()` waits before
            it stops. Default: `3600`.
    """

    def __init__(
        self,
        job,
        n_modes: int,
        *,
        poll_interval: float = 2.0,
        poll_timeout_seconds: float = 3600,
    ) -> None:
        self._job = job
        self._n_modes = n_modes
        self._poll_interval = poll_interval
        self._poll_timeout_seconds = poll_timeout_seconds

    def status(self) -> str:
        """Return the state of the job as Perceval reports it, without waiting.

        For example, `"RUNNING"` or `"SUCCESS"`.
        """
        return str(self._job.status())

    def result(self) -> dict:
        """Wait until the job finishes, then return the estimated Fock distribution.

        Raises:
            TimeoutError: If the job has not finished after `poll_timeout_seconds`.
                The job keeps running. Call `result()` again or `cancel()`, or find
                the job in the Quandela console with the id given in the message.
            RuntimeError: If the job ended in error or was cancelled, or if
                post-selection discarded every shot.
        """
        deadline = time.monotonic() + self._poll_timeout_seconds
        while not self._job.is_complete:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Quandela Cloud job {self._job.id} still {self.status()} after "
                    f"{self._poll_timeout_seconds}s"
                )
            time.sleep(self._poll_interval)
        if not self._job.is_success:
            raise RuntimeError(
                f"Quandela Cloud job ended {self.status()}: {self._job.status.stop_message}"
            )
        return _counts_to_distribution(self._job.get_results()["results"], self._n_modes)

    def cancel(self) -> None:
        """Ask Quandela Cloud to cancel the job, without waiting."""
        self._job.cancel()


class RemoteCompiled:
    """A compiled handle tied to a Quandela Cloud platform.

    It implements the `Compiled` protocol (`run`, `expectation` and `jacobian`),
    and adds `submit` and `dry_run_info`. `QuandelaCloudBackend.compile` builds it.
    Nothing connects to the cloud until the first `run` or `submit`.

    Args:
        spec (PhotonicCircuitSpec): The circuit that this handle submits.
        backend (QuandelaCloudBackend): The platform, the credentials and the
            sampling settings to submit with.
    """

    def __init__(self, spec, backend) -> None:
        self._spec = spec
        self._backend = backend
        self._warned = False
        self._processor = None

    def _remote_processor(self):
        import perceval as pcvl

        token = _resolve_token(self._backend._token)
        kwargs = {"url": self._backend.url} if self._backend.url else {}
        return pcvl.RemoteProcessor(self._backend.platform, token, **kwargs)

    def _prepare(self, weights, inputs):
        import perceval as pcvl

        # Built first: its checks are offline, so a bad call fails before connecting.
        circuit = build_perceval_circuit(self._spec, weights, inputs)
        if self._processor is None:
            # Constructing a RemoteProcessor fetches the platform specs over the network,
            # so it is done once per handle. Reusing it is safe: each job serializes the
            # circuit into its payload when it is created.
            self._processor = self._remote_processor()
        processor = self._processor
        processor.set_circuit(circuit)
        mdp = self._backend.min_detected_photons
        processor.min_detected_photons_filter(
            self._spec.n_photons if mdp is None else mdp
        )
        processor.with_input(pcvl.BasicState(list(self._spec.input_state)))
        return processor

    def _submit(self, weights, inputs) -> RemoteJobHandle:
        from perceval.algorithm import Sampler

        processor = self._prepare(weights, inputs)
        if not self._warned:
            warnings.warn(
                f"submitting to Quandela Cloud platform {self._backend.platform!r} "
                f"({self._backend.shots} shots) — this may incur cost or quota usage",
                CostWarning,
                stacklevel=3,
            )
        sampler = Sampler(processor, max_shots_per_call=self._backend.shots)
        job = sampler.sample_count.execute_async(self._backend.shots)
        # Only a submission that went through spends the one-time warning; a failed
        # attempt (no token, network, bad circuit) warns again on the retry.
        self._warned = True
        return RemoteJobHandle(
            job,
            self._spec.n_modes,
            poll_timeout_seconds=self._backend.poll_timeout_seconds,
        )

    def run(self, weights, inputs) -> dict:
        """Sample the circuit on the cloud, and return the estimated Fock distribution.

        The call waits for the job to finish. The distribution is a dictionary
        `{photon_counts: probability}`.

        Args:
            weights (np.ndarray): The 1-D weight vector.
            inputs (np.ndarray): The 1-D input vector.

        Raises:
            TimeoutError: If the job has not finished within `poll_timeout_seconds`.
                The job is cancelled first, because the caller never gets its
                handle. The same happens on Ctrl-C. Use `submit` to keep a job
                running after the wait.
            RuntimeError: If the job fails, or if post-selection discards every
                shot.
        """
        handle = self._submit(weights, inputs)
        try:
            return handle.result()
        except BaseException:
            # Timeout or Ctrl-C: the caller never sees this handle, so cancel the
            # paid job here. A failed cancel must not hide the original error.
            with contextlib.suppress(Exception):
                handle.cancel()
            raise

    def submit(self, weights, inputs) -> RemoteJobHandle:
        """Submit the job and return a handle at once, without waiting for the job.

        Args:
            weights (np.ndarray): The 1-D weight vector.
            inputs (np.ndarray): The 1-D input vector.

        Returns:
            The `RemoteJobHandle` of the job, with `.result()` and `.status()`.
        """
        return self._submit(weights, inputs)

    def expectation(self, weights, inputs, observable) -> float:
        """Estimate an observable from one sampled run, and wait for the result.

        The observable is diagonal in the Fock basis.

        Args:
            weights (np.ndarray): The 1-D weight vector.
            inputs (np.ndarray): The 1-D input vector.
            observable (PhotonicObservable): A `PhotonNumber` or a `FockProbability`.

        Raises:
            ValueError: If `observable` can never match an output that is kept by
                post-selection. This is checked before anything is submitted.
        """
        from ..gradients import _check_observable, _expectation

        # Post-selection keeps outputs with at least min_detected_photons photons, so
        # a lossy target is valid here; checked before paying for a job.
        _check_observable(
            self._spec, observable, min_photons=self._backend.min_detected_photons
        )
        return _expectation(observable, self.run(weights, inputs))

    def jacobian(
        self,
        weights,
        inputs,
        observable,
        *,
        method="parameter_shift",
        need_weight_grad=True,
        need_input_grad=True,
    ):
        """Always raise, because gradients are not available on sampled hardware.

        Args:
            weights (np.ndarray): The 1-D weight vector.
            inputs (np.ndarray): The 1-D input vector.
            observable (PhotonicObservable): A `PhotonNumber` or a `FockProbability`.
            method (str): This argument is ignored.
            need_weight_grad (bool): This argument is ignored.
            need_input_grad (bool): This argument is ignored.

        Raises:
            NotImplementedError: Always. Use the local `PercevalBackend` to compute
                gradients.
        """
        raise NotImplementedError(
            "gradients are unavailable on Quandela Cloud hardware (finite-shot sampling); "
            "use the local PercevalBackend for parameter-shift / finite-difference gradients"
        )

    def dry_run_info(self) -> dict:
        """Describe the job that *would* be submitted, offline and without cost.

        Returns:
            A dictionary with the keys `"platform"`, `"n_modes"`, `"n_photons"`,
                `"input_state"`, `"n_components"` and `"shots"`.
        """
        return {
            "platform": self._backend.platform,
            "n_modes": self._spec.n_modes,
            "n_photons": self._spec.n_photons,
            "input_state": self._spec.input_state,
            "n_components": len(self._spec.components),
            "shots": self._backend.shots,
        }


class QuandelaCloudBackend:
    """An executor that runs on a Quandela Cloud platform.

    The platform is a real quantum processing unit (QPU) or a cloud simulator.

    Args:
        platform (str): The name of a Quandela Cloud platform, for example
            `"qpu:ascella"` for hardware or `"sim:slos"` for a cloud simulator.
            Default: `"sim:slos"`.
        token (str | None): The access token. Without it, the token comes from
            `$QUANDELA_TOKEN` or `$PERCEVAL_TOKEN`, then from the token that
            Perceval has saved. Default: `None`.
        shots (int): The number of shots sampled in each run. Default: `10000`.
        url (str | None): The web address of the Quandela Cloud service, if it is
            not Perceval's default. Default: `None`.
        min_detected_photons (int | None): Keep only the shots that detect at
            least this many photons (post-selection). The default is the number of
            photons in the input state.
        poll_timeout_seconds (float): How long, in seconds, `run` and `result()`
            wait for a job before they raise `TimeoutError`. Default: `3600`.
    """

    def __init__(
        self,
        platform: str = "sim:slos",
        *,
        token: str | None = None,
        shots: int = 10000,
        url: str | None = None,
        min_detected_photons: int | None = None,
        poll_timeout_seconds: float = 3600,
    ) -> None:
        self.platform = platform
        self._token = token
        self.shots = shots
        self.url = url
        self.min_detected_photons = min_detected_photons
        self.poll_timeout_seconds = poll_timeout_seconds

    def compile(self, spec) -> RemoteCompiled:
        """Tie `spec` to this platform, offline, with no network connection or token.

        Args:
            spec (PhotonicCircuitSpec): The circuit to compile.

        Returns:
            The `RemoteCompiled` handle for `spec`.
        """
        return RemoteCompiled(spec, self)
