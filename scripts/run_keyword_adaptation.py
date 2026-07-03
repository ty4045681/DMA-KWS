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

from dma_kws.config import fbank_kwargs, get_fbank_config, require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.pathing import PROJECT_ROOT
from dma_kws.stage2.adapt_paths import adapt_data_root, adapt_exp_root, clips_eval_manifest_from_adapt, phase_manifest, slugify
from dma_kws.stage2.prepare_adapt import prepare_keyword_adaptation


def _run_script(script: str, overrides: list[str]) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *overrides]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _build_eval_report(config: dict, keyword: str, base_ckpt: str, adapted_ckpt: str) -> dict:
    slug = slugify(keyword)
    data_root = adapt_data_root(config, keyword)
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
            )
            metrics_path = out_dir / "metrics.json"
            if metrics_path.is_file():
                report[f"target_{label}"] = json.loads(metrics_path.read_text(encoding="utf-8"))

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

    slug = slugify(keyword)
    common = [f"adapt.keyword={keyword!r}"]

    if stage in {"prepare", "all"}:
        data_root = adapt_data_root(config, keyword)
        stats = prepare_keyword_adaptation(
            keyword=keyword,
            data_root=data_root,
            fbank_params=fbank_kwargs(get_fbank_config(config)),
            eval_fraction=float(adapt.get("eval_fraction", 0.2)),
            eval_seed=int(adapt.get("eval_seed", 2025)),
            manifest_csv=prep.get("manifest_csv") or None,
            skip_existing=not bool(prep.get("no_skip_existing", False)),
        )
        print(json.dumps(stats, indent=2))

    sweep_cfg = adapt.get("sweep", {}) or {}
    if stage in {"sweep", "all"} and bool(sweep_cfg.get("enabled", False)):
        _run_script(
            "sweep_adapt_lora.py",
            common + [f"run.init_checkpoint={base_ckpt}", f"run.device={run.device}"],
        )

    params_file = adapt.get("params_file", "")
    best_params = adapt_exp_root(config, keyword) / "sweep" / "best_params.yaml"
    if not params_file and best_params.is_file():
        params_file = str(best_params)

    if stage in {"train", "all"}:
        train_overrides = common + [f"prep.stage2_ckpt={base_ckpt}", f"run.device={run.device}"]
        if params_file:
            train_overrides.append(f"adapt.params_file={params_file}")
        for phase in ("tts", "real"):
            _run_script(
                "adapt_stage2_keyword.py",
                train_overrides + [f"adapt.phase={phase}"],
            )

    if stage in {"eval", "all"}:
        adapted_ckpt = adapt_exp_root(config, keyword) / "stage2_adapted.pt"
        if not adapted_ckpt.is_file():
            raise SystemExit(f"Adapted checkpoint not found: {adapted_ckpt}")
        _build_eval_report(config, keyword, base_ckpt, str(adapted_ckpt))


if __name__ == "__main__":
    main()
