from pathlib import Path

import pytest

from dma_kws.runlog import build_loggers
from dma_kws.training.run_context import (
    RUN_CONTEXT_KEY,
    build_run_context,
    stamp_run_context,
)


def _config() -> dict:
    return {
        "stage1": {"max_train_steps": 100, "log_interval": 5},
        "stage2": {"max_steps": 50000, "log_interval": 10},
        "phoneme_adapter": {"max_steps": 60000, "log_interval": 50},
        "adapt": {"max_steps": 3000},
    }


def test_run_context_uses_effective_limit_and_explicit_version(tmp_path: Path):
    context = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path,
        run_name="qbyt",
        limit_steps=12,
        version=3,
    )

    assert context.run_id == "qbyt/version_3"
    assert context.effective_max_steps == 12
    assert context.log_interval == 10
    assert context.run_dir == str(tmp_path / "qbyt" / "version_3")


def test_adapt_uses_its_max_steps_and_stage2_trainer_log_interval(tmp_path: Path):
    context = build_run_context(
        _config(),
        section="adapt",
        log_dir=tmp_path,
        run_name="adapt-keyword",
        version=0,
    )

    assert context.effective_max_steps == 3000
    assert context.log_interval == 10


def test_matching_full_resume_starts_child_version(tmp_path: Path):
    original = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path,
        run_name="qbyt",
        version=7,
    )
    checkpoint: dict = {}
    stamp_run_context(checkpoint, original)
    (tmp_path / "qbyt" / "version_7").mkdir(parents=True)

    resumed = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path,
        run_name="qbyt",
        resume_from="last.ckpt",
        resume_checkpoint=checkpoint,
    )

    assert resumed.version == 8
    assert resumed.run_id == "qbyt/version_8"
    assert resumed.resume_from == "last.ckpt"
    assert resumed.parent_run_id == original.run_id


def test_resume_stays_a_child_when_parent_directory_is_missing(tmp_path: Path):
    original = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path,
        run_name="qbyt",
        version=7,
    )
    checkpoint: dict = {}
    stamp_run_context(checkpoint, original)

    resumed = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path,
        run_name="qbyt",
        resume_from="moved.ckpt",
        resume_checkpoint=checkpoint,
        job_key="missing-parent-directory",
    )

    assert resumed.version == 8
    assert resumed.run_id != original.run_id
    assert resumed.parent_run_id == original.run_id


def test_resume_rejects_explicit_parent_version(tmp_path: Path):
    original = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path,
        run_name="qbyt",
        version=2,
    )
    checkpoint: dict = {}
    stamp_run_context(checkpoint, original)

    with pytest.raises(ValueError, match="must use a child run version"):
        build_run_context(
            _config(),
            section="stage2",
            log_dir=tmp_path,
            run_name="qbyt",
            resume_from="last.ckpt",
            resume_checkpoint=checkpoint,
            version=2,
        )


def test_resume_with_same_job_key_gets_a_child_claim(tmp_path: Path):
    original = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path,
        run_name="qbyt",
        job_key="scheduler-job",
    )
    checkpoint: dict = {}
    stamp_run_context(checkpoint, original)

    resumed = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path,
        run_name="qbyt",
        resume_from="last.ckpt",
        resume_checkpoint=checkpoint,
        job_key="scheduler-job",
    )

    assert original.version == 0
    assert resumed.version == 1
    assert resumed.parent_run_id == original.run_id


def test_moved_resume_starts_child_run_with_parent_identity(tmp_path: Path):
    original = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path / "old",
        run_name="qbyt",
        version=2,
    )
    checkpoint: dict = {}
    stamp_run_context(checkpoint, original)

    resumed = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path / "new",
        run_name="qbyt",
        resume_from="moved.ckpt",
        resume_checkpoint=checkpoint,
        version=0,
    )

    assert resumed.run_id == "qbyt/version_0"
    assert resumed.parent_run_id == original.run_id


def test_stamp_run_context_is_serializable(tmp_path: Path):
    context = build_run_context(
        _config(),
        section="stage1",
        log_dir=tmp_path,
        run_name="ctc",
        version="trial-a",
    )
    payload: dict = {}

    assert stamp_run_context(payload, context) is payload
    assert payload[RUN_CONTEXT_KEY]["run_id"] == "ctc/trial-a"
    assert payload[RUN_CONTEXT_KEY]["effective_max_steps"] == 100


def test_resume_csv_logger_preserves_parent_metrics_file(tmp_path: Path):
    original = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path,
        run_name="qbyt",
        version=0,
    )
    parent_logger = build_loggers(
        tmp_path,
        "qbyt",
        config={"stage2": {"logging": {"backends": ["csv"]}}},
        version=original.version,
    )[0]
    parent_logger.log_metrics({"old_metric": 0.9}, step=1)
    parent_logger.save()
    parent_metrics = Path(original.run_dir) / "metrics.csv"
    before = parent_metrics.read_bytes()
    checkpoint: dict = {}
    stamp_run_context(checkpoint, original)

    resumed = build_run_context(
        _config(),
        section="stage2",
        log_dir=tmp_path,
        run_name="qbyt",
        resume_from="last.ckpt",
        resume_checkpoint=checkpoint,
        job_key="csv-child-resume",
    )
    child_logger = build_loggers(
        tmp_path,
        "qbyt",
        config={"stage2": {"logging": {"backends": ["csv"]}}},
        version=resumed.version,
    )[0]
    child_logger.log_hyperparams({"parent_run_id": resumed.parent_run_id})
    child_logger.log_metrics({"new_metric": 0.8}, step=2)
    child_logger.save()

    assert resumed.run_id == "qbyt/version_1"
    assert parent_metrics.read_bytes() == before
    assert (Path(resumed.run_dir) / "metrics.csv").is_file()
