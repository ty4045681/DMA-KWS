"""Training-run logging helpers (loss/metrics records on disk).

Uses the Stage II PyTorch Lightning
``TensorBoardLogger(save_dir, name=run_name)`` and additionally attaches a
dependency-free ``CSVLogger`` so loss/metrics are always recorded to a plain
``metrics.csv`` even when the ``tensorboard`` package is absent.

Logs land under ``{log_dir}/{run_name}/version_N/`` (Lightning auto-increments
the version), matching the paper's ``lightning_logs/{name}/version_0`` layout.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, MutableMapping


_EXPLICIT_JOB_KEY_ENV = "DMA_KWS_RUN_JOB_KEY"
_LOCAL_JOB_KEY_ENV = "DMA_KWS_LOCAL_RUN_JOB_KEY"
_CLAIMS_DIRNAME = ".dma_kws_run_claims"
_DEFAULT_CLAIM_TIMEOUT_SECONDS = 120.0
_DEFAULT_CLAIM_POLL_SECONDS = 0.05


def resolve_logger_version(
    log_dir: str | Path,
    run_name: str,
) -> int:
    """Return the next local Lightning ``version_N`` visible on disk.

    This scan is informational and is also used to choose the first reservation
    candidate.  Call :func:`reserve_logger_version` when the result must be
    unique across concurrently starting jobs.
    """
    run_root = Path(log_dir) / run_name
    versions: list[int] = []
    if run_root.is_dir():
        for child in run_root.iterdir():
            if not child.is_dir() or not child.name.startswith("version_"):
                continue
            suffix = child.name.removeprefix("version_")
            if suffix.isdigit():
                versions.append(int(suffix))
    return max(versions, default=-1) + 1


def _process_rank(environ: Mapping[str, str]) -> int:
    for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        raw = environ.get(name)
        if raw is None:
            continue
        try:
            rank = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
        if rank < 0:
            raise RuntimeError(f"{name} must be non-negative, got {rank}")
        return rank
    return 0


def resolve_logger_job_key(
    explicit: str | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    rank: int | None = None,
) -> str:
    """Return a launcher-stable key shared by every rank in one job.

    Precedence is an explicit argument/environment value, torchrun, then SLURM.
    For Lightning's local subprocess launcher, rank zero creates a random key in
    the environment before spawning; child ranks inherit it.  A non-zero rank
    with no shared launcher key fails closed instead of guessing a version.
    """
    env = os.environ if environ is None else environ
    process_rank = _process_rank(env) if rank is None else int(rank)
    if process_rank < 0:
        raise ValueError(f"rank must be non-negative, got {process_rank}")

    explicit_value = str(explicit or "").strip()
    if explicit_value:
        return f"explicit:{explicit_value}"

    environment_value = str(env.get(_EXPLICIT_JOB_KEY_ENV, "")).strip()
    if environment_value:
        return f"explicit:{environment_value}"

    torchrun_id = str(env.get("TORCHELASTIC_RUN_ID", "")).strip()
    # torchrun's parser defaults ``--rdzv-id`` to the literal string ``none``.
    # It is therefore a placeholder, not a job-unique identifier: treating it
    # as a claim key would collapse unrelated torchrun/SLURM jobs onto one log
    # version.  Ignore it and continue to the launcher-specific fallbacks.
    if torchrun_id and torchrun_id.lower() != "none":
        # A restarted process reconstructs CSVLogger, which would truncate an
        # existing metrics.csv. Give each elastic attempt an immutable child
        # version while keeping all ranks of that attempt on one claim.
        restart = str(env.get("TORCHELASTIC_RESTART_COUNT", "0")).strip() or "0"
        return f"torchrun:{torchrun_id}:restart={restart}"

    slurm_job_id = str(env.get("SLURM_JOB_ID", "")).strip()
    if slurm_job_id:
        parts = [f"slurm:{slurm_job_id}"]
        for name in (
            "SLURM_ARRAY_TASK_ID",
            "SLURM_STEP_ID",
            "SLURM_RESTART_COUNT",
        ):
            value = str(env.get(name, "")).strip()
            if value:
                parts.append(f"{name.lower()}={value}")
        return ":".join(parts)

    if process_rank > 0:
        inherited = str(env.get(_LOCAL_JOB_KEY_ENV, "")).strip()
        if inherited:
            return f"local:{inherited}"
        raise RuntimeError(
            "Cannot coordinate the training-log version for non-zero rank "
            f"{process_rank}: no explicit {_EXPLICIT_JOB_KEY_ENV}, "
            "TORCHELASTIC_RUN_ID, SLURM_JOB_ID, or inherited local job key is "
            "available. Set DMA_KWS_RUN_JOB_KEY to one value shared by all ranks."
        )

    local_key = uuid.uuid4().hex
    env[_LOCAL_JOB_KEY_ENV] = local_key
    return f"local:{local_key}"


def _claim_path(log_dir: str | Path, run_name: str, job_key: str) -> Path:
    digest = hashlib.sha256(job_key.encode("utf-8")).hexdigest()
    return Path(log_dir) / run_name / _CLAIMS_DIRNAME / f"{digest}.json"


def _reserve_version_directory(
    log_dir: str | Path,
    run_name: str,
    *,
    minimum_version: int = 0,
) -> int:
    """Atomically create and return one previously unused ``version_N`` dir."""
    if minimum_version < 0:
        raise ValueError("minimum_version must be non-negative")
    run_root = Path(log_dir) / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    candidate = max(minimum_version, resolve_logger_version(log_dir, run_name))
    while True:
        try:
            (run_root / f"version_{candidate}").mkdir()
        except FileExistsError:
            candidate += 1
            continue
        return candidate


def _publish_version_claim(path: Path, *, job_key: str, version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_key_sha256": hashlib.sha256(job_key.encode("utf-8")).hexdigest(),
        "version": int(version),
        "created_at_unix": time.time(),
        "publisher_pid": os.getpid(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_version_claim(path: Path, *, job_key: str, run_root: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid logger-version claim at {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"Invalid logger-version claim at {path}: expected a JSON object"
        )
    expected_digest = hashlib.sha256(job_key.encode("utf-8")).hexdigest()
    if payload.get("job_key_sha256") != expected_digest:
        raise RuntimeError(f"Logger-version claim key mismatch at {path}")
    version = payload.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise RuntimeError(
            f"Logger-version claim at {path} has invalid version {version!r}"
        )
    version_dir = run_root / f"version_{version}"
    if not version_dir.is_dir():
        raise RuntimeError(
            f"Logger-version claim at {path} points to missing directory {version_dir}"
        )
    return version


def reserve_logger_version(
    log_dir: str | Path,
    run_name: str,
    *,
    job_key: str | None = None,
    rank: int | None = None,
    minimum_version: int = 0,
    timeout_seconds: float = _DEFAULT_CLAIM_TIMEOUT_SECONDS,
    poll_seconds: float = _DEFAULT_CLAIM_POLL_SECONDS,
    environ: MutableMapping[str, str] | None = None,
) -> int:
    """Reserve one version on rank zero and return its claim on every rank.

    Creating ``version_N`` is the reservation primitive, so independent jobs
    racing on the same log root cannot receive the same number. Non-zero ranks
    never scan for their own version: they wait for rank zero's atomic claim and
    time out with an error rather than silently splitting a DDP job.
    """
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    env = os.environ if environ is None else environ
    process_rank = _process_rank(env) if rank is None else int(rank)
    if process_rank < 0:
        raise ValueError(f"rank must be non-negative, got {process_rank}")
    resolved_job_key = resolve_logger_job_key(
        job_key,
        environ=env,
        rank=process_rank,
    )
    # Resume checkpoints require a child version even when a scheduler/user
    # intentionally keeps the same job id. Scope the claim by the minimum
    # acceptable version so an old attempt cannot force reuse of its log dir.
    claim_job_key = f"{resolved_job_key}:minimum_version={minimum_version}"
    claim_path = _claim_path(log_dir, run_name, claim_job_key)
    run_root = Path(log_dir) / run_name

    claim_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    if process_rank == 0:
        lock_path = claim_path.with_suffix(f"{claim_path.suffix}.lock")
        while True:
            if claim_path.is_file():
                return _read_version_claim(
                    claim_path,
                    job_key=claim_job_key,
                    run_root=run_root,
                )
            try:
                lock_path.mkdir()
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for another rank-0 process to publish "
                        f"the logger-version claim at {claim_path}."
                    )
                time.sleep(
                    min(poll_seconds, max(0.0, deadline - time.monotonic()))
                )
                continue
            try:
                # A competing leader may have published immediately before this
                # lock was acquired; recheck under the job-scoped lock.
                if claim_path.is_file():
                    return _read_version_claim(
                        claim_path,
                        job_key=claim_job_key,
                        run_root=run_root,
                    )
                version = _reserve_version_directory(
                    log_dir,
                    run_name,
                    minimum_version=minimum_version,
                )
                _publish_version_claim(
                    claim_path,
                    job_key=claim_job_key,
                    version=version,
                )
                return version
            finally:
                try:
                    lock_path.rmdir()
                except FileNotFoundError:
                    pass

    while True:
        if claim_path.is_file():
            return _read_version_claim(
                claim_path,
                job_key=claim_job_key,
                run_root=run_root,
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for rank 0 to publish a logger-version claim "
                f"for run {run_name!r} at {claim_path}. Refusing to choose an "
                "independent version for this rank."
            )
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _logger_warning(message: str) -> None:
    """Print optional-backend warnings once under external DDP launchers."""
    if _process_rank(os.environ) == 0:
        print(message)


def logger_backend_names(loggers: list[Any]) -> list[str]:
    """Return stable names for the loggers that were actually constructed."""
    names: list[str] = []
    for logger in loggers:
        class_name = type(logger).__name__.lower()
        if "tensorboard" in class_name:
            name = "tensorboard"
        elif "wandb" in class_name:
            name = "wandb"
        elif "trackio" in class_name:
            name = "trackio"
        elif "csv" in class_name:
            name = "csv"
        else:
            name = type(logger).__name__
        if name not in names:
            names.append(name)
    return names


def _logging_config(
    config: dict[str, Any] | None,
    section: str,
) -> dict[str, Any]:
    selected = (config or {}).get(section, {}) or {}
    logging_cfg = selected.get("logging", {}) or {}
    if not isinstance(logging_cfg, dict):
        raise ValueError(f"{section}.logging must be a mapping")
    return logging_cfg


def build_loggers(
    log_dir: str | Path,
    run_name: str,
    *,
    config: dict[str, Any] | None = None,
    section: str = "stage2",
    version: int | str | None = None,
) -> list:
    """Return Lightning loggers for a training run.

    When ``config`` is provided, reads ``{section}.logging.backends`` to choose
    CSV, TensorBoard, W&B, and/or Trackio loggers. Defaults to
    ``[csv, tensorboard]`` (backward compatible with Stage I call sites that
    omit ``config``).

    pytorch_lightning is imported lazily so this module stays importable in
    environments without torch installed.
    """
    from pytorch_lightning.loggers import CSVLogger

    log_dir = str(log_dir)
    logging_cfg = _logging_config(config, section)
    raw_backends = logging_cfg.get("backends", ["csv", "tensorboard"])
    if not isinstance(raw_backends, (list, tuple)):
        raise ValueError(f"{section}.logging.backends must be a list")
    # De-duplicate without changing the user's requested order.
    backends = list(dict.fromkeys(str(backend).strip().lower() for backend in raw_backends))
    supported = {"csv", "tensorboard", "wandb", "trackio"}
    unknown = [backend for backend in backends if backend not in supported]
    if unknown:
        raise ValueError(
            f"Unsupported {section}.logging backend(s): {', '.join(unknown)}; "
            f"expected one or more of: {', '.join(sorted(supported))}"
        )

    shared_version = (
        reserve_logger_version(log_dir, run_name) if version is None else version
    )

    loggers: list = []

    if "csv" in backends:
        loggers.append(
            CSVLogger(save_dir=log_dir, name=run_name, version=shared_version)
        )

    if "tensorboard" in backends:
        try:
            from pytorch_lightning.loggers import TensorBoardLogger

            loggers.append(
                TensorBoardLogger(
                    save_dir=log_dir,
                    name=run_name,
                    version=shared_version,
                    default_hp_metric=False,
                )
            )
        except (ImportError, ModuleNotFoundError):
            _logger_warning(
                "tensorboard not installed; skipping TensorBoard logger. "
                "Install tensorboard for TensorBoard event logs."
            )

    if "wandb" in backends:
        try:
            from pytorch_lightning.loggers import WandbLogger

            wandb_cfg = logging_cfg.get("wandb", {}) or {}
            loggers.append(
                WandbLogger(
                    save_dir=log_dir,
                    name=run_name,
                    version=f"{run_name}-version_{shared_version}",
                    project=str(wandb_cfg.get("project", "dma-kws")),
                    mode=str(wandb_cfg.get("mode", "online")),
                    resume="allow",
                )
            )
        except (ImportError, ModuleNotFoundError):
            _logger_warning("wandb not installed; skipping W&B logger. Install wandb to enable.")

    if "trackio" in backends:
        trackio_logger = _build_trackio_logger(
            log_dir,
            run_name,
            shared_version,
            logging_cfg.get("trackio", {}) or {},
        )
        if trackio_logger is not None:
            loggers.append(trackio_logger)

    if not loggers:
        loggers.append(
            CSVLogger(save_dir=log_dir, name=run_name, version=shared_version)
        )

    return loggers


def _build_trackio_logger(
    log_dir: str | Path,
    run_name: str,
    version: int | str,
    trackio_cfg: dict[str, Any],
):
    """Return a thin Lightning logger backed by Trackio, or None if unavailable."""
    try:
        import trackio
        from pytorch_lightning.loggers import Logger
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except (ImportError, ModuleNotFoundError):
        _logger_warning("trackio not installed; skipping Trackio logger. Install trackio to enable.")
        return None

    project = str(trackio_cfg.get("project", "dma-kws"))
    remote_run_name = f"{run_name}-version_{version}"

    class TrackioLogger(Logger):
        def __init__(self) -> None:
            super().__init__()
            # Remote runs must be created lazily: this logger is constructed in
            # every DDP process, while only global rank zero may initialize and
            # write the shared Trackio run.
            self._run = None

        def _ensure_run(self):
            if self._run is None:
                self._run = trackio.init(
                    project=project,
                    name=remote_run_name,
                    dir=str(log_dir),
                )
            return self._run

        @property
        def name(self) -> str:
            return run_name

        @property
        def version(self) -> str:
            return str(version)

        @rank_zero_only
        def log_hyperparams(self, params: dict[str, Any]) -> None:
            self._ensure_run()
            trackio.config.update(params)

        @rank_zero_only
        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            self._ensure_run()
            payload = dict(metrics)
            if step is not None:
                payload["step"] = step
            trackio.log(payload)

        @rank_zero_only
        def finalize(self, status: str) -> None:
            if self._run is None:
                return
            finish = getattr(trackio, "finish", None)
            if callable(finish):
                finish()

        def save(self) -> None:
            return None

        def __reduce__(self):
            # The implementation class is local so runlog stays importable
            # without Trackio/Lightning. Reconstruct through the module-level
            # factory and deliberately drop the live remote handle under spawn.
            return (
                _build_trackio_logger,
                (str(log_dir), run_name, version, dict(trackio_cfg)),
            )

    return TrackioLogger()
