#!/usr/bin/env python3
"""Rebuild filtered and review MUSAN trees from cached WeNet ASR results.jsonl."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from dma_kws.inference.musan_asr_rescore import rebuild_from_results
from scripts.filter_musan_by_wenet_asr import load_keywords


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musan-root", required=True)
    parser.add_argument("--results", required=True, help="Existing WeNet filter results.jsonl")
    parser.add_argument("--output-root", required=True, help="Keep-only MUSAN output; must not exist")
    parser.add_argument("--review-root", required=True, help="Review-only MUSAN output; must not exist")
    parser.add_argument("--report-dir", required=True, help="New rescore report output; must not exist")
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--keywords-file")
    parser.add_argument("--remove-threshold", type=float, default=92.0)
    parser.add_argument("--review-threshold", type=float, default=78.0)
    args = parser.parse_args(argv)
    summary = rebuild_from_results(musan_root=args.musan_root, results_path=args.results, output_root=args.output_root, review_root=args.review_root, report_dir=args.report_dir, keywords=load_keywords(args.keyword, args.keywords_file), remove_threshold=args.remove_threshold, review_threshold=args.review_threshold)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
