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
#   --pt PT:OUT_DIR    Explicit .pt checkpoint and its required output directory (repeatable)
#   --pts-file FILE    Text file with one PT:OUT_DIR per non-comment line
#   --manifest PATH    Path to merged.csv manifest
#   --base-out DIR     Fallback root output dir when no per-dir OUT is given
#   --experiment NAME  Hydra experiment override (default: icefall_zipformer_stage2)
#   -h, --help         Show this help
#
# Examples:
#   # Scan checkpoint directories:
#   bash scripts/batch_eval_stage2_clips.sh \
#     --dir /path/to/export1:/path/to/out1 \
#     --dir /path/to/export2 \
#     --manifest /path/to/merged.csv \
#     --base-out /path/to/outputs
#
#   # Evaluate explicit checkpoints at exact output directories:
#   bash scripts/batch_eval_stage2_clips.sh \
#     --pt /path/to/stage2_step010000.pt:/path/to/out/step10000 \
#     --pt /path/to/stage2_step020000.pt:/path/to/out/step20000 \
#     --manifest /path/to/merged.csv
#
#   # Mix directory scanning with explicit checkpoints:
#   bash scripts/batch_eval_stage2_clips.sh \
#     --dirs-file ckpt_dirs.txt \
#     --pts-file checkpoints.txt \
#     --manifest /path/to/merged.csv \
#     --base-out /path/to/outputs

set -euo pipefail

CKPT_DIRS=()
EXPLICIT_PTS=()
DIRS_FILE=""
PTS_FILE=""
MANIFEST="/home/q00931063/DMA-KWS/data/dma-kws/test/hey_eva_and_its_variants/merged.csv"
BASE_OUT="/home/q00931063/DMA-KWS/data/dma-kws/test/outputs/zipformer_stage2_only_hey_eva_and_its_variants_new"
EXPERIMENT="icefall_zipformer_stage2"

usage() {
  sed -n '3,35p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)        CKPT_DIRS+=("$2");   shift 2 ;;
    --dirs-file)  DIRS_FILE="$2";      shift 2 ;;
    --pt)         EXPLICIT_PTS+=("$2"); shift 2 ;;
    --pts-file)   PTS_FILE="$2";        shift 2 ;;
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

if [[ -n "${PTS_FILE}" ]]; then
  if [[ ! -f "${PTS_FILE}" ]]; then
    echo "ERROR: --pts-file not found: ${PTS_FILE}" >&2; exit 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    EXPLICIT_PTS+=("${line}")
  done < "${PTS_FILE}"
fi

if [[ ${#CKPT_DIRS[@]} -eq 0 && ${#EXPLICIT_PTS[@]} -eq 0 ]]; then
  echo "ERROR: No checkpoints specified. Use --dir, --dirs-file, --pt, or --pts-file." >&2
  exit 1
fi

shopt -s nullglob

total_ckpts=0
run_checkpoint() {
  local ckpt="$1"
  local out_dir="$2"
  local display_label="$3"

  echo "--- [${display_label}] ---"
  python3 scripts/eval_stage2_clips.py \
    +experiment="${EXPERIMENT}" \
    prep.manifest="${MANIFEST}" \
    prep.stage2_ckpt="${ckpt}" \
    prep.output_dir="${out_dir}"
  echo "Done -> ${out_dir}"
  echo ""
  (( total_ckpts++ )) || true
}

echo "Manifest  : ${MANIFEST}"
echo "Base out  : ${BASE_OUT}"
echo "Experiment: ${EXPERIMENT}"
echo "Dirs (${#CKPT_DIRS[@]}): ${CKPT_DIRS[*]:-}"
echo "Explicit pts (${#EXPLICIT_PTS[@]}): ${EXPLICIT_PTS[*]:-}"
echo ""
if [[ ${#CKPT_DIRS[@]} -gt 0 ]]; then
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
    run_checkpoint "${ckpt}" "${out_dir}" "${dir_name}/${name}"
  done
done
fi

if [[ ${#EXPLICIT_PTS[@]} -gt 0 ]]; then
for entry in "${EXPLICIT_PTS[@]}"; do
  ckpt="${entry%%:*}"
  out_dir="${entry#*:}"
  if [[ "${out_dir}" == "${ckpt}" || -z "${ckpt}" || -z "${out_dir}" ]]; then
    echo "ERROR: Invalid explicit checkpoint entry (expected PT:OUT_DIR): ${entry}" >&2
    exit 1
  fi
  if [[ "${ckpt}" != *.pt ]]; then
    echo "WARNING: not a .pt checkpoint, skipping: ${ckpt}" >&2
    continue
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "WARNING: checkpoint not found, skipping: ${ckpt}" >&2
    continue
  fi

  name="$(basename "${ckpt}" .pt)"
  run_checkpoint "${ckpt}" "${out_dir}" "explicit/${name}"
done
fi

echo "All done. Evaluated ${total_ckpts} checkpoint(s) total."
