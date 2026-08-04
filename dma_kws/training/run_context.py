"""Canonical identity and effective runtime settings for one training run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from dma_kws.runlog import reserve_logger_version


RUN_CONTEXT_KEY = "dma_kws_run_context"


def _version_dir(version: int | str) -> str:
    return version if isinstance(version, str) else f"version_{version}"


@dataclass(frozen=True)
class RunContext:
    """One source of truth shared by Trainer, console and all log backends."""

    section: str
    run_name: str
    version: int | str
    log_root: str
    run_dir: str
    effective_max_steps: int
    log_interval: int
    resume_from: str = ""
    parent_run_id: str = ""

    @property
    def run_id(self) -> str:
        return f"{self.run_name}/{_version_dir(self.version)}"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_id"] = self.run_id
        return payload

    def identity(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_version": self.version,
            "run_section": self.section,
            "parent_run_id": self.parent_run_id,
            "resume_from": self.resume_from,
        }


def checkpoint_run_context(checkpoint: Any) -> dict[str, Any] | None:
    if not isinstance(checkpoint, Mapping):
        return None
    value = checkpoint.get(RUN_CONTEXT_KEY)
    return dict(value) if isinstance(value, Mapping) else None


def stamp_run_context(payload: dict[str, Any], context: RunContext) -> dict[str, Any]:
    payload[RUN_CONTEXT_KEY] = context.as_dict()
    return payload


def build_run_context(
    config: dict[str, Any],
    *,
    section: str,
    log_dir: str | Path,
    run_name: str,
    limit_steps: int | None = None,
    resume_from: str | Path | None = None,
    resume_checkpoint: Any = None,
    version: int | str | None = None,
    job_key: str | None = None,
    process_rank: int | None = None,
    claim_timeout_seconds: float = 120.0,
    claim_poll_seconds: float = 0.05,
    environ: MutableMapping[str, str] | None = None,
) -> RunContext:
    """Resolve run identity and runtime values before constructing loggers.

    Every invocation gets a new immutable Lightning version, including a
    full-state resume.  Lightning's CSV logger rewrites ``metrics.csv`` while
    initializing an existing version, so reusing the checkpoint's directory
    would silently destroy the pre-resume history.  A stamped checkpoint keeps
    the original run id as explicit parent lineage instead.
    """
    stage = config.get(section, {}) or {}
    trainer_stage = (
        config.get("stage2", {}) or {}
        if section == "adapt"
        else stage
    )
    max_steps_key = "max_train_steps" if section == "stage1" else "max_steps"
    configured_max_steps = int(stage.get(max_steps_key, -1))
    if section == "stage1" and configured_max_steps == 0:
        configured_max_steps = -1
    effective_max_steps = (
        int(limit_steps)
        if limit_steps is not None and int(limit_steps) > 0
        else configured_max_steps
    )
    log_interval = int(trainer_stage.get("log_interval", 10))
    normalized_root = str(Path(log_dir))
    resume_path = str(resume_from or "")

    saved = checkpoint_run_context(resume_checkpoint)
    saved_run_id = str((saved or {}).get("run_id", ""))
    minimum_version = 0
    if resume_path:
        saved_version = (saved or {}).get("version")
        if (
            isinstance(saved_version, int)
            and not isinstance(saved_version, bool)
            and saved_version >= 0
            and str((saved or {}).get("run_name", run_name)) == run_name
        ):
            minimum_version = saved_version + 1
    if version is None:
        version = reserve_logger_version(
            normalized_root,
            run_name,
            job_key=job_key,
            rank=process_rank,
            minimum_version=minimum_version,
            timeout_seconds=claim_timeout_seconds,
            poll_seconds=claim_poll_seconds,
            environ=environ,
        )
    run_dir = str(Path(normalized_root) / run_name / _version_dir(version))

    run_id = f"{run_name}/{_version_dir(version)}"
    if resume_path and saved_run_id and run_id == saved_run_id:
        raise ValueError(
            "A resumed training run must use a child run version; "
            f"{run_id!r} is the checkpoint's existing run id."
        )

    return RunContext(
        section=section,
        run_name=run_name,
        version=version,
        log_root=normalized_root,
        run_dir=run_dir,
        effective_max_steps=effective_max_steps,
        log_interval=log_interval,
        resume_from=resume_path,
        parent_run_id=saved_run_id if resume_path else "",
    )
