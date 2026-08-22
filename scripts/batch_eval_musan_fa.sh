#!/usr/bin/env bash
# Batch-evaluate exported .pt Stage-II checkpoints for MUSAN false accepts.
#
# Usage:
#   bash scripts/batch_eval_musan_fa.sh [OPTIONS]
#
# Options:
#   --keyword KEYWORD       Keyword to evaluate (repeatable)
#   --keyword-phonemes PHONES
#                           ARPAbet override for the preceding --keyword
#   --keywords-file FILE    One keyword, or keyword<TAB>phonemes, per line
#   --musan-root DIR        Root of the MUSAN corpus (must contain music/noise/speech dirs)
#   --pt PT:OUT             Explicit .pt checkpoint and its required output directory (repeatable)
#   --pts-file FILE         Text file with one PT:OUT_DIR per non-comment line
#   --base-out DIR          Fallback root output dir when no per-pt OUT is given
#   --window-sec SEC        Sliding window length in seconds (default: 3.0)
#   --hop-sec SEC           Sliding window hop in seconds (default: 3.0)
#   --experiment NAME       Hydra experiment override (default: icefall_zipformer_stage2)
#   -h, --help              Show this help
#
# --keywords-file format:
#   keyword
#   keyword<TAB>HH EY1 IY1 V AH0
#   Lines starting with '#' and blank lines are ignored. One-column rows use G2P.
#
# Examples:
#   # Single keyword, explicit checkpoints:
#   bash scripts/batch_eval_musan_fa.sh \
#     --keyword "hey eva" \
#     --keyword-phonemes "HH EY1 IY1 V AH0" \
#     --musan-root /path/to/musan \
#     --pt /path/to/stage2_step010000.pt:/path/to/out/step10000 \
#     --pt /path/to/stage2_step020000.pt:/path/to/out/step20000 \
#     --window-sec 3.0 \
#     --hop-sec 3.0
#
#   # Multiple keywords from file, directory-style output:
#   bash scripts/batch_eval_musan_fa.sh \
#     --keywords-file keywords.txt \
#     --musan-root /path/to/musan \
#     --pt /path/to/stage2_step010000.pt \
#     --base-out /path/to/outputs
#
# Output layout:
#   With --pt PT:OUT and one keyword, output lands in OUT.
#   With --pt PT:OUT and multiple keywords, output lands in OUT/<keyword_slug>.
#   With --base-out and --pt, output lands in BASE_OUT/<ckpt_name>/<keyword_slug>.
#   Each run writes only results.jsonl and summary.json.

set -euo pipefail

KEYWORDS=()
KEYWORD_PHONEMES=()
KEYWORD_PHONEMES_SET=()
KEYWORDS_FILE=""
MUSAN_ROOT=""
EXPLICIT_PTS=()
PTS_FILE=""
BASE_OUT=""
WINDOW_SEC="3.0"
HOP_SEC="3.0"
EXPERIMENT="icefall_zipformer_stage2"

usage() {
  sed -n '3,48p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keyword)
      KEYWORDS+=("$2")
      KEYWORD_PHONEMES+=("")
      KEYWORD_PHONEMES_SET+=("0")
      shift 2
      ;;
    --keyword-phonemes)
      if [[ ${#KEYWORDS[@]} -eq 0 ]]; then
        echo "ERROR: --keyword-phonemes must follow --keyword." >&2
        exit 1
      fi
      keyword_index=$((${#KEYWORDS[@]} - 1))
      if [[ "${KEYWORD_PHONEMES_SET[${keyword_index}]}" == "1" ]]; then
        echo "ERROR: duplicate --keyword-phonemes for ${KEYWORDS[${keyword_index}]}" >&2
        exit 1
      fi
      if [[ -z "${2//[[:space:]]/}" ]]; then
        echo "ERROR: --keyword-phonemes must not be empty." >&2
        exit 1
      fi
      KEYWORD_PHONEMES[${keyword_index}]="$2"
      KEYWORD_PHONEMES_SET[${keyword_index}]="1"
      shift 2
      ;;
    --keywords-file) KEYWORDS_FILE="$2"; shift 2 ;;
    --musan-root)    MUSAN_ROOT="$2";   shift 2 ;;
    --pt)            EXPLICIT_PTS+=("$2"); shift 2 ;;
    --pts-file)      PTS_FILE="$2";      shift 2 ;;
    --base-out)      BASE_OUT="$2";     shift 2 ;;
    --window-sec)    WINDOW_SEC="$2";   shift 2 ;;
    --hop-sec)       HOP_SEC="$2";      shift 2 ;;
    --experiment)    EXPERIMENT="$2";   shift 2 ;;
    -h|--help)       usage ;;
    *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Load keywords from file if provided
if [[ -n "${KEYWORDS_FILE}" ]]; then
  if [[ ! -f "${KEYWORDS_FILE}" ]]; then
    echo "ERROR: --keywords-file not found: ${KEYWORDS_FILE}" >&2; exit 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    keyword="${line%%$'\t'*}"
    phonemes=""
    if [[ "${line}" == *$'\t'* ]]; then
      phonemes="${line#*$'\t'}"
      if [[ "${phonemes}" == *$'\t'* ]]; then
        echo "ERROR: --keywords-file accepts at most two tab-separated columns: ${line}" >&2
        exit 1
      fi
    fi
    keyword="${keyword#"${keyword%%[![:space:]]*}"}"
    keyword="${keyword%"${keyword##*[![:space:]]}"}"
    phonemes="${phonemes#"${phonemes%%[![:space:]]*}"}"
    phonemes="${phonemes%"${phonemes##*[![:space:]]}"}"
    if [[ -z "${keyword}" ]]; then
      echo "ERROR: Empty keyword in --keywords-file row: ${line}" >&2
      exit 1
    fi
    KEYWORDS+=("${keyword}")
    KEYWORD_PHONEMES+=("${phonemes}")
    [[ -n "${phonemes}" ]] && KEYWORD_PHONEMES_SET+=("1") || KEYWORD_PHONEMES_SET+=("0")
  done < "${KEYWORDS_FILE}"
fi

if [[ ${#KEYWORDS[@]} -eq 0 ]]; then
  echo "ERROR: No keywords specified. Use --keyword or --keywords-file." >&2
  exit 1
fi
if [[ -z "${MUSAN_ROOT}" ]]; then
  echo "ERROR: --musan-root is required." >&2
  exit 1
fi
if [[ ! -d "${MUSAN_ROOT}" ]]; then
  echo "ERROR: MUSAN root not found: ${MUSAN_ROOT}" >&2
  exit 1
fi

# Load explicit checkpoints from file if provided
if [[ -n "${PTS_FILE}" ]]; then
  if [[ ! -f "${PTS_FILE}" ]]; then
    echo "ERROR: --pts-file not found: ${PTS_FILE}" >&2; exit 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    EXPLICIT_PTS+=("${line}")
  done < "${PTS_FILE}"
fi

if [[ ${#EXPLICIT_PTS[@]} -eq 0 ]]; then
  echo "ERROR: No checkpoints specified. Use --pt or --pts-file." >&2
  exit 1
fi

keyword_slug() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | tr ' _-' '_' \
    | tr -s '_' \
    | sed 's/^_//;s/_$//'
}

shopt -s nullglob

total_runs=0
run_checkpoint_keyword() {
  local ckpt="$1"
  local out_dir="$2"
  local keyword="$3"
  local keyword_phonemes="$4"
  local display_label="$5"
  local -a eval_command=(
    python3 scripts/eval_musan_fa.py
    "+experiment=${EXPERIMENT}"
    "prep.keyword=${keyword}"
    "prep.musan_root=${MUSAN_ROOT}"
    "prep.stage2_ckpt=${ckpt}"
    "prep.window_sec=${WINDOW_SEC}"
    "prep.hop_sec=${HOP_SEC}"
    "prep.output_dir=${out_dir}"
  )
  if [[ -n "${keyword_phonemes}" ]]; then
    eval_command+=("prep.keyword_phonemes=${keyword_phonemes}")
  fi

  echo "--- [${display_label}] ---"
  "${eval_command[@]}"
  echo "Done -> ${out_dir}"
  echo ""
  (( total_runs++ )) || true
}

if [[ -n "${BASE_OUT}" ]]; then
  mkdir -p "${BASE_OUT}"
fi

echo "MUSAN root : ${MUSAN_ROOT}"
echo "Keywords   : ${#KEYWORDS[@]}"
echo "Checkpoints: ${#EXPLICIT_PTS[@]}"
echo "Window     : ${WINDOW_SEC}s / hop ${HOP_SEC}s"
echo "Experiment : ${EXPERIMENT}"
echo "Base out   : ${BASE_OUT:-<none>}"
echo ""

for entry in "${EXPLICIT_PTS[@]}"; do
  ckpt="${entry%%:*}"
  per_out="${entry#*:}"
  [[ "${per_out}" == "${ckpt}" ]] && per_out=""

  if [[ "${ckpt}" != *.pt ]]; then
    echo "WARNING: not a .pt checkpoint, skipping: ${ckpt}" >&2
    continue
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "WARNING: checkpoint not found, skipping: ${ckpt}" >&2
    continue
  fi

  ckpt_name="$(basename "${ckpt}" .pt)"

  if [[ ${#KEYWORDS[@]} -eq 1 && -n "${per_out}" ]]; then
    # Single keyword + explicit output -> use output as-is
    run_checkpoint_keyword \
      "${ckpt}" "${per_out}" "${KEYWORDS[0]}" "${KEYWORD_PHONEMES[0]}" \
      "${ckpt_name}/${KEYWORDS[0]}"
  else
    for keyword_index in "${!KEYWORDS[@]}"; do
      keyword="${KEYWORDS[${keyword_index}]}"
      keyword_phonemes="${KEYWORD_PHONEMES[${keyword_index}]}"
      kw_slug="$(keyword_slug "${keyword}")"
      if [[ -n "${per_out}" ]]; then
        out_dir="${per_out}/${kw_slug}"
      elif [[ -n "${BASE_OUT}" ]]; then
        out_dir="${BASE_OUT}/${ckpt_name}/${kw_slug}"
      else
        echo "ERROR: No output directory for ${ckpt} / ${keyword}. Use --base-out or PT:OUT_DIR." >&2
        exit 1
      fi
      run_checkpoint_keyword \
        "${ckpt}" "${out_dir}" "${keyword}" "${keyword_phonemes}" \
        "${ckpt_name}/${keyword}"
    done
  fi
done

echo "All done. Evaluated ${total_runs} checkpoint×keyword combination(s) total."
