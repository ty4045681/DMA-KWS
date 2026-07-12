#!/usr/bin/env bash
# Batch-evaluate exported .pt checkpoints (across one or more dirs) against a manifest.
#
# Usage:
#   bash scripts/batch_eval_stage2_clips.sh [OPTIONS]
#
# Options:
#   --dir DIR[:OUT]    Checkpoint directory, with optional per-dir output path after ':'
#                      If OUT is omitted, falls back to <base-out>/<dir_basename>
#                      (repeatable)
#   --dirs-file FILE   Text file with one CKPT_DIR[:OUT_DIR] per line (alternative to --dir)
#   --manifest PATH    Path to merged.csv manifest
#   --base-out DIR     Fallback root output dir when no per-dir OUT is given
#   --experiment NAME  Hydra experiment override (default: icefall_zipformer_stage2)
#   -h, --help         Show this help
#
# Examples:
#   # Per-dir output paths:
#   bash scripts/batch_eval_stage2_clips.sh \
#     --dir /path/to/export1:/path/to/out1 \
#     --dir /path/to/export2:/path/to/out2 \
#     --manifest /path/to/merged.csv
#
#   # Mix: one with explicit out, one falling back to --base-out:
#   bash scripts/batch_eval_stage2_clips.sh \
#     --dir /path/to/export1:/path/to/out1 \
#     --dir /path/to/export2 \
#     --manifest /path/to/merged.csv \
#     --base-out /path/to/outputs
#
#   # Directories from a file (one CKPT_DIR[:OUT_DIR] per line):
#   bash scripts/batch_eval_stage2_clips.sh \
#     --dirs-file ckpt_dirs.txt \
#     --manifest /path/to/merged.csv \
#     --base-out /path/to/outputs

set -euo pipefail

CKPT_DIRS=()
DIRS_FILE=""
MANIFEST="/home/q00931063/DMA-KWS/data/dma-kws/test/hey_eva_and_its_variants/merged.csv"
BASE_OUT="/home/q00931063/DMA-KWS/data/dma-kws/test/outputs/zipformer_stage2_only_hey_eva_and_its_variants_new"
EXPERIMENT="icefall_zipformer_stage2"

usage() {
  sed -n '3,35p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)        CKPT_DIRS+=("$2"); shift 2 ;;
    --dirs-file)  DIRS_FILE="$2";    shift 2 ;;
    --manifest)   MANIFEST="$2";     shift 2 ;;
    --base-out)   BASE_OUT="$2";     shift 2 ;;
    --experiment) EXPERIMENT="$2";   shift 2 ;;
    -h|--help)    usage ;;
    *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Load dirs from file if provided
if [[ -n "${DIRS_FILE}" ]]; then
  if [[ ! -f "${DIRS_FILE}" ]]; then
    echo "ERROR: --dirs-file not found: ${DIRS_FILE}" >&2; exit 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    CKPT_DIRS+=("${line}")
  done < "${DIRS_FILE}"
fi

if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: No checkpoint directories specified. Use --dir or --dirs-file." >&2
  exit 1
fi

shopt -s nullglob

echo "Manifest  : ${MANIFEST}"
echo "Base out  : ${BASE_OUT}"
echo "Experiment: ${EXPERIMENT}"
echo "Dirs (${#CKPT_DIRS[@]}): ${CKPT_DIRS[*]}"
echo ""

total_ckpts=0
for entry in "${CKPT_DIRS[@]}"; do
  # Split on first ':' to allow DIR:OUT_DIR syntax
  ckpt_dir="${entry%%:*}"
  per_out="${entry#*:}"
  [[ "${per_out}" == "${ckpt_dir}" ]] && per_out=""  # no colon present

  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "WARNING: directory not found, skipping: ${ckpt_dir}" >&2
    continue
  fi

  pts=("${ckpt_dir}"/*.pt)
  if [[ ${#pts[@]} -eq 0 ]]; then
    echo "WARNING: no *.pt files in ${ckpt_dir}, skipping." >&2
    continue
  fi

  dir_name="$(basename "${ckpt_dir}")"
  echo "=== ${dir_name} (${#pts[@]} checkpoint(s)) ==="

  for ckpt in "${pts[@]}"; do
    name="$(basename "${ckpt}" .pt)"
    out_dir="${per_out:-${BASE_OUT}/${dir_name}}/${name}"
    echo "--- [${dir_name}/${name}] ---"
    python3 scripts/eval_stage2_clips.py \
      +experiment="${EXPERIMENT}" \
      prep.manifest="${MANIFEST}" \
      prep.stage2_ckpt="${ckpt}" \
      prep.output_dir="${out_dir}"
    echo "Done -> ${out_dir}"
    echo ""
    (( total_ckpts++ )) || true
  done
done

echo "All done. Evaluated ${total_ckpts} checkpoint(s) total."
