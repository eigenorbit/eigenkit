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
import os
import time
import warnings

from ...core.provider import CostWarning
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
