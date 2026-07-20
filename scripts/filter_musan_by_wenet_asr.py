#!/usr/bin/env python3
"""Copy MUSAN while excluding audio whose WeNet transcript matches wake words."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dma_kws.inference.musan_asr_filter import (
    filter_musan,
    resolve_model_files,
    validate_keywords,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musan-root", required=True, help="Source MUSAN root")
    parser.add_argument("--output-root", required=True, help="New filtered MUSAN root; must not exist")
    parser.add_argument("--report-dir", help="Report directory; defaults beside output-root")
    parser.add_argument("--model-dir", help="Directory containing the WeNet model files")
    parser.add_argument("--checkpoint", help="Override WeNet .pt checkpoint")
    parser.add_argument("--config", help="Override WeNet train.yaml")
    parser.add_argument("--cmvn", help="Override global_cmvn/global_cvmn")
    parser.add_argument("--bpe-model", help="Override SentencePiece .model")
    parser.add_argument("--units", help="Override units.txt/words.txt")
    parser.add_argument(
        "--wenet-root",
        help="WeNet source checkout containing wenet/bin/recognize.py; or set WENET_ROOT",
    )
    parser.add_argument("--keyword", action="append", default=[], help="Wake word; repeat for multiple")
    parser.add_argument("--keywords-file", help="UTF-8 file with one wake word per line")
    parser.add_argument("--threshold", type=float, default=85.0, help="RapidFuzz partial ratio threshold")
    parser.add_argument("--window-sec", type=float, default=30.0, help="ASR window length")
    parser.add_argument("--overlap-sec", type=float, default=1.0, help="Overlap between ASR windows")
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:N (default: auto)"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="WeNet decode batch size")
    parser.add_argument("--beam-size", type=int, default=10, help="WeNet search beam size")
    parser.add_argument(
        "--mode", default="attention_rescoring", help="WeNet decoding mode"
    )
    return parser


def load_keywords(values: Sequence[str], keywords_file: str | None) -> list[str]:
    keywords = list(values)
    if keywords_file:
        path = Path(keywords_file).expanduser()
        with path.open("r", encoding="utf-8") as handle:
            keywords.extend(
                line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")
            )
    return validate_keywords(keywords)


def resolve_device(value: str) -> str:
    if value != "auto":
        if value in {"cpu", "cuda"}:
            return value
        if value.startswith("cuda:") and value.removeprefix("cuda:").isdigit():
            return value
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    files = resolve_model_files(
        args.model_dir,
        checkpoint=args.checkpoint,
        config=args.config,
        cmvn=args.cmvn,
        bpe_model=args.bpe_model,
        units=args.units,
    )
    keywords = load_keywords(args.keyword, args.keywords_file)
    summary = filter_musan(
        musan_root=args.musan_root,
        output_root=args.output_root,
        report_dir=args.report_dir,
        files=files,
        keywords=keywords,
        wenet_root=args.wenet_root,
        threshold=args.threshold,
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        device=resolve_device(args.device),
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        mode=args.mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
