#!/usr/bin/env python3
"""One-command keyword continual adaptation: prepare → sweep → train → eval."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.pathing import PROJECT_ROOT
from dma_kws.stage2 import adapt_console
from dma_kws.stage2.adapt_paths import (
    adapt_data_root,
    adapt_exp_root,
    clips_eval_manifest_from_adapt,
    phase_manifest,
    slugify,
)
from dma_kws.stage2.prep_console import Stage2PrepReporter


def _forward_overrides() -> list[str]:
    """Forward user-supplied Hydra config overrides to sub-scripts.

    The orchestrator re-assembles overrides for each sub-script, but keys like
    ``+experiment=...`` or ``stage2.eval.test_dir=...`` must still reach them.
    Excluded keys are already rebuilt by the orchestrator, so forwarding them
    would only create duplicates.
    """
    excluded = {
        "adapt.stage",
        "adapt.keyword",
        "adapt.phase",
        "prep.stage2_ckpt",
        "run.device",
    }
    return [
        arg for arg in sys.argv[1:]
        if "=" in arg
        and not arg.startswith("--")
        and arg.split("=", 1)[0] not in excluded
    ]


def _run_script(
    script: str,
    overrides: list[str],
    reporter: Stage2PrepReporter | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / script),
        *_forward_overrides(),
        *overrides,
    ]
    message = "Running: " + " ".join(cmd)
    if reporter is not None:
        reporter.info(f"[dim]{message}[/dim]" if reporter.use_rich else message)
    else:
        print(message)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _clip_eval_metrics(out_dir: Path) -> dict | None:
    """Read labeled metrics from ``eval_stage2_clips.py`` output (``summary.json``)."""
    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _build_eval_report(
    config: dict,
    keyword: str,
    base_ckpt: str,
    adapted_ckpt: str,
    data_root: Path,
    reporter: Stage2PrepReporter | None = None,
) -> dict:
    slug = slugify(keyword)
    exp_root = adapt_exp_root(config, keyword)
    reports_dir = exp_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    eval_manifest = phase_manifest(data_root, "real", split="eval")
    clips_manifest = reports_dir / "real_eval_clips.csv"
    if eval_manifest.is_file():
        clips_eval_manifest_from_adapt(eval_manifest, keyword, clips_manifest)

    report: dict = {"keyword": keyword, "slug": slug, "base_ckpt": base_ckpt, "adapted_ckpt": adapted_ckpt}

    if clips_manifest.is_file():
        for label, ckpt in ("base", base_ckpt), ("adapted", adapted_ckpt):
            out_dir = reports_dir / f"eval_clips_{label}"
            _run_script(
                "eval_stage2_clips.py",
                [
                    f"prep.manifest={clips_manifest}",
                    f"prep.stage2_ckpt={ckpt}",
                    f"prep.stage2_clip_output_dir={out_dir}",
                    "run.device=cpu",
                ],
                reporter,
            )
            metrics = _clip_eval_metrics(out_dir)
            if metrics is not None:
                report[f"target_{label}"] = metrics
    elif reporter is not None:
        reporter.warn(f"No real-phase eval manifest at {eval_manifest}; skipping target keyword eval")

    for label, ckpt in ("base", base_ckpt), ("adapted", adapted_ckpt):
        out_path = reports_dir / f"lph_{label}.json"
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval_stage2_libriphrase.py"),
                f"prep.checkpoint={ckpt}",
                "prep.split=hard",
                "run.device=cpu",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        metrics = json.loads(result.stdout.strip().splitlines()[-1])
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report[f"lph_{label}"] = metrics

    report_path = reports_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if reporter is not None:
        columns, rows = adapt_console.eval_comparison_table(report)
        if rows:
            reporter.print_table(columns, rows, title="Base vs Adapted")
        reporter.done(f"Eval report written to {report_path}")
    print(yaml.safe_dump({"report_path": str(report_path), "report": report}, sort_keys=False))
    return report


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    config = resolved_config(cfg)
    require_sections(config, ["paths", "adapt"])
    adapt = OmegaConf.to_container(cfg.adapt, resolve=True)
    if not isinstance(adapt, dict):
        raise SystemExit("adapt config section must be a mapping")
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    run = cfg.run

    keyword = str(adapt.get("keyword", "")).strip()
    if not keyword:
        raise SystemExit("adapt.keyword is required")

    stage = str(adapt.get("stage", "all"))
    base_ckpt = str(run.init_checkpoint or prep.get("stage2_ckpt", ""))
    if stage != "prepare" and not base_ckpt:
        raise SystemExit("prep.stage2_ckpt or run.init_checkpoint is required")

    data_root = Path(adapt["data_root"]) if adapt.get("data_root") else adapt_data_root(config, keyword)
    exp_root = adapt_exp_root(config, keyword)
    sweep_cfg = adapt.get("sweep", {}) or {}
    common = [f"adapt.keyword={keyword!r}"]

    reporter = adapt_console.adapt_reporter(config, prep)
    reporter.section(f"Keyword adaptation · {keyword}")
    reporter.print_plan(
        [
            ("keyword", keyword),
            ("slug", slugify(keyword)),
            ("stage", stage),
            ("base_checkpoint", base_ckpt or "(not required for prepare)"),
            ("data_root", str(data_root)),
            ("exp_root", str(exp_root)),
            ("device", f"{run.device} x{run.devices}"),
            ("sweep", "enabled" if sweep_cfg.get("enabled", False) else "disabled"),
        ],
        title="Adaptation Run",
    )

    if stage in {"prepare", "all"}:
        stats = adapt_console.prepare_with_console(config, adapt, prep)
        print(json.dumps(stats, indent=2))

    if stage in {"sweep", "all"} and bool(sweep_cfg.get("enabled", False)):
        reporter.section("Hyperparameter sweep")
        _run_script(
            "sweep_adapt_lora.py",
            common + [f"run.init_checkpoint={base_ckpt}", f"run.device={run.device}"],
            reporter,
        )

    params_file = adapt.get("params_file", "")
    best_params = exp_root / "sweep" / "best_params.yaml"
    if not params_file and best_params.is_file():
        params_file = str(best_params)

    if stage in {"train", "all"}:
        train_overrides = common + [f"prep.stage2_ckpt={base_ckpt}", f"run.device={run.device}"]
        if params_file:
            train_overrides.append(f"adapt.params_file={params_file}")
            reporter.info(f"Using swept hyperparameters from {params_file}")
        for phase in ("tts", "real"):
            reporter.section(f"LoRA training · phase={phase}")
            _run_script(
                "adapt_stage2_keyword.py",
                train_overrides + [f"adapt.phase={phase}"],
                reporter,
            )

    if stage in {"eval", "all"}:
        reporter.section("Evaluation · base vs adapted")
        adapted_ckpt = exp_root / "stage2_adapted.pt"
        if not adapted_ckpt.is_file():
            raise SystemExit(f"Adapted checkpoint not found: {adapted_ckpt}")
        _build_eval_report(config, keyword, base_ckpt, str(adapted_ckpt), data_root, reporter)


if __name__ == "__main__":
    main()
