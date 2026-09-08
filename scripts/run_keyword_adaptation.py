#!/usr/bin/env python3
"""One-command keyword continual adaptation: prepare → sweep → train → eval."""

from __future__ import annotations

import csv
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
from dma_kws.stage2.adapt_config import (
    adapt_method_label,
    is_full_adapt_method,
    resolve_adapt_method,
)
from dma_kws.stage2.adapt_paths import (
    adapt_data_root,
    adapt_exp_root,
    clips_eval_manifest_from_adapt,
    phase_manifest,
    resolve_adapt_train_phases,
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
        "adapt.method",
        "adapt.phase",
        "prep.stage2_ckpt",
        "run.device",
        "run.resume_from",
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


def _musan_keyword_phonemes(real_eval_manifest: Path, prep: dict) -> str:
    """Keep continuous-background queries aligned with the real clip queries."""
    if not real_eval_manifest.is_file():
        raise SystemExit(f"Joint evaluation requires the real manifest: {real_eval_manifest}")

    def normalize(value: str | None) -> str:
        text = " ".join(str(value or "").split())
        return "" if text.casefold() in {"nan", "none", "null", "<na>", "n/a", "na"} else text

    with real_eval_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        pronunciations = {
            normalized
            for row in csv.DictReader(handle)
            if (normalized := normalize(row.get("keyword_phonemes")))
        }
    if len(pronunciations) > 1:
        raise ValueError(
            f"{real_eval_manifest} has inconsistent keyword_phonemes: {sorted(pronunciations)}"
        )
    manifest_phonemes = next(iter(pronunciations), "")
    configured = normalize(prep.get("keyword_phonemes"))
    if manifest_phonemes and configured and manifest_phonemes != configured:
        raise ValueError(
            "prep.keyword_phonemes differs from real_eval.csv keyword_phonemes: "
            f"{configured!r} != {manifest_phonemes!r}. MUSAN and real-clip evaluation "
            "must use the same keyword pronunciation."
        )
    return manifest_phonemes or configured


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
    adapt = config.get("adapt", {}) or {}
    joint = resolve_adapt_train_phases(adapt) == ("joint",)
    joint_cfg = adapt.get("joint", {}) or {}
    background_eval_list = (
        str(joint_cfg.get("background_eval_list", "") or "").strip() if joint else ""
    )
    if background_eval_list:
        from dma_kws.stage2.joint_manifest import validate_background_eval_split

        validate_background_eval_split(
            (config.get("stage2", {}) or {}).get("background_negative", {}) or {},
            background_eval_list,
        )
        if not Path(background_eval_list).is_file():
            raise SystemExit(f"MUSAN evaluation list not found: {background_eval_list}")
        musan_root = str((config.get("prep", {}) or {}).get("musan_root", "") or "").strip()
        if not musan_root or not Path(musan_root).is_dir():
            raise SystemExit(
                "prep.musan_root must be an existing MUSAN root directory when "
                "adapt.joint.background_eval_list is set for FA/h evaluation"
            )
        musan_phonemes = _musan_keyword_phonemes(
            phase_manifest(data_root, "real", split="eval"), config.get("prep", {}) or {},
        )

    report: dict = {"keyword": keyword, "slug": slug, "base_ckpt": base_ckpt, "adapted_ckpt": adapted_ckpt}

    for source in (("real", "tts") if joint else ("real",)):
        eval_manifest = phase_manifest(data_root, source, split="eval")
        clips_manifest = reports_dir / f"{source}_eval_clips.csv"
        if not eval_manifest.is_file():
            if joint:
                raise SystemExit(f"Joint evaluation requires the {source} manifest: {eval_manifest}")
            if reporter is not None:
                reporter.warn(f"No {source}-phase eval manifest at {eval_manifest}; skipping {source} eval")
            continue
        clips_eval_manifest_from_adapt(
            eval_manifest, keyword, clips_manifest, manifest_root=data_root,
        )
        for label, ckpt in ("base", base_ckpt), ("adapted", adapted_ckpt):
            out_dir = reports_dir / (
                f"eval_clips_{label}" if source == "real" else f"eval_tts_clips_{label}"
            )
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
                prefix = "target" if source == "real" else "tts"
                report[f"{prefix}_{label}"] = metrics

    for label, ckpt in ("base", base_ckpt), ("adapted", adapted_ckpt):
        out_path = reports_dir / f"lph_{label}.json"
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval_stage2_libriphrase.py"),
                *_forward_overrides(),
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

    if background_eval_list:
        for label, ckpt in ("base", base_ckpt), ("adapted", adapted_ckpt):
            out_dir = reports_dir / f"eval_musan_{label}"
            _run_script(
                "eval_musan_fa.py",
                [
                    f"prep.keyword={keyword!r}",
                    f"prep.keyword_phonemes={musan_phonemes!r}",
                    f"prep.musan_root={musan_root}",
                    f"prep.musan_audio_list_path={background_eval_list}",
                    f"prep.stage2_ckpt={ckpt}",
                    f"prep.output_dir={out_dir}",
                    "run.device=cpu",
                ],
                reporter,
            )
            summary_path = out_dir / "summary.json"
            if not summary_path.is_file():
                raise RuntimeError(f"MUSAN evaluation produced no summary: {summary_path}")
            report[f"musan_{label}"] = json.loads(summary_path.read_text(encoding="utf-8"))

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
    method = resolve_adapt_method(adapt)
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    run = cfg.run

    keyword = str(adapt.get("keyword", "")).strip()
    if not keyword:
        raise SystemExit("adapt.keyword is required")

    stage = str(adapt.get("stage", "all"))
    base_ckpt = str(
        run.init_checkpoint
        or prep.get("stage2_ckpt", "")
        or adapt.get("init_checkpoint", "")
        or config.get("stage2", {}).get("init_checkpoint", "")
        or ""
    )
    train_phases = resolve_adapt_train_phases(adapt)
    resume_from = str(run.resume_from or "")
    full_resume = is_full_adapt_method(method) and bool(resume_from)
    if full_resume and stage in {"train", "all"} and len(train_phases) != 1:
        raise SystemExit(
            f"{method} run.resume_from restores one phase; set adapt.train_phases=[joint], "
            "[tts], or [real] to match the saved checkpoint."
        )
    if stage != "prepare" and not base_ckpt and not (stage == "train" and full_resume):
        raise SystemExit(
            "prep.stage2_ckpt, run.init_checkpoint, or adapt.init_checkpoint is required. "
            "A full-weight training-only resume can instead use run.resume_from."
        )

    data_root = Path(adapt["data_root"]) if adapt.get("data_root") else adapt_data_root(config, keyword)
    exp_root = adapt_exp_root(config, keyword)
    sweep_cfg = adapt.get("sweep", {}) or {}
    common = [f"adapt.keyword={keyword!r}", f"adapt.method={method}"]

    reporter = adapt_console.adapt_reporter(config, prep)
    reporter.section(f"Keyword adaptation · {keyword}")
    reporter.print_plan(
        [
            ("keyword", keyword),
            ("slug", slugify(keyword)),
            ("stage", stage),
            ("method", method),
            ("base_checkpoint", base_ckpt or ("(restored from resume)" if full_resume else "(not required for prepare)")),
            ("data_root", str(data_root)),
            ("exp_root", str(exp_root)),
            ("device", f"{run.device} x{run.devices}"),
            ("train_phases", " -> ".join(train_phases)),
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
        train_overrides = common + [f"run.device={run.device}"]
        if base_ckpt:
            train_overrides.append(f"prep.stage2_ckpt={base_ckpt}")
        if resume_from:
            train_overrides.append(f"run.resume_from={resume_from}")
        if params_file:
            train_overrides.append(f"adapt.params_file={params_file}")
            reporter.info(f"Using swept hyperparameters from {params_file}")
        for index, phase in enumerate(train_phases):
            training_label = adapt_method_label(method)
            reporter.section(f"{training_label} training · phase={phase}")
            phase_overrides = train_overrides + [f"adapt.phase={phase}"]
            if is_full_adapt_method(method) and index:
                previous_model = exp_root / train_phases[index - 1] / "stage2_adapted.pt"
                phase_overrides.extend(
                    [f"run.init_checkpoint={previous_model}", "run.resume_checkpoint="]
                )
            _run_script(
                "adapt_stage2_keyword.py",
                phase_overrides,
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
