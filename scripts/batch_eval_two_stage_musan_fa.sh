#!/usr/bin/env bash
# Batch-evaluate two-stage MUSAN false accepts (Stage I locate + Stage II QbyT).
#
# Usage:
#   bash scripts/batch_eval_two_stage_musan_fa.sh [OPTIONS] [HYDRA OVERRIDES...]
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
#   --experiment NAME       Hydra experiment override (default: icefall_zipformer_stage2)
#   -h, --help              Show this help
#
# Remaining arguments are forwarded to eval_two_stage_musan_fa.py, e.g.
#   +locator=sherpa_zipformer_kws locator.encoder=... stage2.qbyt_alignment.max_keyword_span_frames=30
#
# FA/hour = two-stage wake-ups / total audio hours. Multiple Stage I spans that
# pass QbyT in one file all count. This is not comparable to the official
# 3s Stage-II-only grid.
#
# --keywords-file format:
#   keyword
#   keyword<TAB>HH EY1 IY1 V AH0
#   Lines starting with '#' and blank lines are ignored. One-column rows use G2P.
#
# Examples:
#   bash scripts/batch_eval_two_stage_musan_fa.sh \
#     --keyword "hey eva" \
#     --keyword-phonemes "HH EY1 IY1 V AH0" \
#     --musan-root /path/to/musan \
#     --pt /path/to/stage2.pt:/path/to/out \
#     +locator=sherpa_zipformer_kws \
#     locator.tokens=/path/tokens.txt \
#     locator.encoder=/path/encoder.onnx \
#     locator.decoder=/path/decoder.onnx \
#     locator.joiner=/path/joiner.onnx \
#     stage2.qbyt_alignment.max_keyword_span_frames=30
#
# Output layout:
#   With --pt PT:OUT and one keyword, output lands in OUT.
#   With --pt PT:OUT and multiple keywords, output lands in OUT/<keyword_key>.
#   With --base-out and --pt, output lands in
#   BASE_OUT/<ckpt_name>__<ckpt_hash>/<keyword_key>.
#   Duplicate output directories in one invocation are rejected.

set -euo pipefail

KEYWORDS=()
KEYWORD_PHONEMES=()
KEYWORD_PHONEMES_SET=()
KEYWORDS_FILE=""
MUSAN_ROOT=""
EXPLICIT_PTS=()
PTS_FILE=""
BASE_OUT=""
EXPERIMENT="icefall_zipformer_stage2"
HYDRA_OVERRIDES=()

usage() {
  sed -n '3,46p' "$0" | sed 's/^# \{0,1\}//'
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
    --experiment)    EXPERIMENT="$2";   shift 2 ;;
    --window-sec|--hop-sec)
      echo "ERROR: $1 does not apply to two-stage MUSAN FA (file-level locate)." >&2
      exit 1
      ;;
    -h|--help)       usage ;;
    --*)
      echo "ERROR: Unknown option: $1" >&2
      exit 1
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
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

keyword_key() {
  # Keep spaces, hyphens, and underscores distinct so "hey eva" and "hey-eva"
  # cannot share an output directory.
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | tr ' ' '_' \
    | tr -c 'a-z0-9_-' '_' \
    | tr -s '_' \
    | sed 's/^_//;s/_$//'
}

checkpoint_key() {
  local ckpt="$1"
  local resolved
  resolved="$(cd "$(dirname "${ckpt}")" && pwd -P)/$(basename "${ckpt}")"
  local digest
  digest="$(printf '%s' "${resolved}" | shasum -a 256 | awk '{print substr($1,1,12)}')"
  printf '%s__%s' "$(basename "${ckpt}" .pt)" "${digest}"
}

total_runs=0
declare -A SEEN_OUTPUT_DIRS=()

claim_output_dir() {
  local out_dir="$1"
  local label="$2"
  local normalized
  # python's Path.resolve() does not require the parent to exist.
  normalized="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${out_dir}")"
  if [[ -n "${SEEN_OUTPUT_DIRS[${normalized}]+x}" ]]; then
    echo "ERROR: output directory collision for ${label}: ${out_dir}" >&2
    echo "       already claimed by ${SEEN_OUTPUT_DIRS[${normalized}]}" >&2
    exit 1
  fi
  SEEN_OUTPUT_DIRS["${normalized}"]="${label}"
}

run_checkpoint_keyword() {
  local ckpt="$1"
  local out_dir="$2"
  local keyword="$3"
  local keyword_phonemes="$4"
  local display_label="$5"
  local -a eval_command=(
    python3 scripts/eval_two_stage_musan_fa.py
    "+experiment=${EXPERIMENT}"
    "prep.keyword=${keyword}"
    "prep.musan_root=${MUSAN_ROOT}"
    "prep.stage2_ckpt=${ckpt}"
    "prep.output_dir=${out_dir}"
  )
  if [[ -n "${keyword_phonemes}" ]]; then
    eval_command+=("prep.keyword_phonemes=${keyword_phonemes}")
  fi
  eval_command+=("${HYDRA_OVERRIDES[@]}")

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

  ckpt_key="$(checkpoint_key "${ckpt}")"

  if [[ ${#KEYWORDS[@]} -eq 1 && -n "${per_out}" ]]; then
    claim_output_dir "${per_out}" "${ckpt_key}/${KEYWORDS[0]}"
    run_checkpoint_keyword \
      "${ckpt}" "${per_out}" "${KEYWORDS[0]}" "${KEYWORD_PHONEMES[0]}" \
      "${ckpt_key}/${KEYWORDS[0]}"
  else
    for keyword_index in "${!KEYWORDS[@]}"; do
      keyword="${KEYWORDS[${keyword_index}]}"
      keyword_phonemes="${KEYWORD_PHONEMES[${keyword_index}]}"
      kw_key="$(keyword_key "${keyword}")"
      if [[ -n "${per_out}" ]]; then
        out_dir="${per_out}/${kw_key}"
      elif [[ -n "${BASE_OUT}" ]]; then
        out_dir="${BASE_OUT}/${ckpt_key}/${kw_key}"
      else
        echo "ERROR: No output directory for ${ckpt} / ${keyword}. Use --base-out or PT:OUT_DIR." >&2
        exit 1
      fi
      claim_output_dir "${out_dir}" "${ckpt_key}/${keyword}"
      run_checkpoint_keyword \
        "${ckpt}" "${out_dir}" "${keyword}" "${keyword_phonemes}" \
        "${ckpt_key}/${keyword}"
    done
  fi
done

echo "All done. Evaluated ${total_runs} checkpoint×keyword combination(s) total."
