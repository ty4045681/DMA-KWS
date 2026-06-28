from dma_kws.training.ddp import build_trainer_kwargs, resolve_precision


def _stage2_config(**overrides):
    base = {
        "strategy": "ddp",
        "accumulate_grad_batches": 2,
        "gradient_clip_val": 1.5,
        "val_check_interval": 500,
        "max_steps": 50000,
        "log_interval": 25,
    }
    base.update(overrides)
    return {"stage2": base}


def test_build_trainer_kwargs_reads_stage2_config():
    kwargs = build_trainer_kwargs(_stage2_config(), devices=4)

    assert kwargs["devices"] == 4
    assert kwargs["strategy"] == "ddp"
    assert kwargs["accumulate_grad_batches"] == 2
    assert kwargs["gradient_clip_val"] == 1.5
    assert kwargs["val_check_interval"] == 500
    assert kwargs["max_steps"] == 50000
    assert kwargs["log_every_n_steps"] == 25


def test_build_trainer_kwargs_uses_validation_val_check_interval():
    config = {
        "stage2": {
            "val_check_interval": 1000,
            "validation": {"val_check_interval": 250},
        }
    }

    kwargs = build_trainer_kwargs(config, devices=1)

    assert kwargs["val_check_interval"] == 250


def test_build_trainer_kwargs_ddp_only_with_multiple_devices():
    kwargs_multi = build_trainer_kwargs(_stage2_config(), devices=4)
    kwargs_single = build_trainer_kwargs(_stage2_config(), devices=1)

    assert kwargs_multi["strategy"] == "ddp"
    assert kwargs_single["strategy"] == "auto"


def test_build_trainer_kwargs_limit_steps_overrides_max_steps():
    kwargs = build_trainer_kwargs(_stage2_config(max_steps=50000), devices=1, limit_steps=20)

    assert kwargs["max_steps"] == 20


def test_build_trainer_kwargs_includes_precision_for_gpu():
    kwargs = build_trainer_kwargs(_stage2_config(), devices=4, accelerator="gpu")

    assert kwargs["precision"] == "bf16-mixed"


def test_build_trainer_kwargs_includes_precision_for_cpu():
    kwargs = build_trainer_kwargs(_stage2_config(), devices=1, accelerator="cpu")

    assert kwargs["precision"] == "32-true"


def test_build_trainer_kwargs_precision_override():
    kwargs = build_trainer_kwargs(
        _stage2_config(precision="16-mixed"),
        devices=1,
        accelerator="gpu",
    )

    assert kwargs["precision"] == "16-mixed"
    assert resolve_precision({"precision": "16-mixed"}, "gpu") == "16-mixed"
