#!/usr/bin/env bash
# Launch eval_musan_fa.py once per GPU, then merge shard outputs.
#
# Usage:
#   bash scripts/eval_musan_fa_shards.sh [HYDRA OVERRIDES...]
#
# Required overrides:
#   prep.output_dir=...
#
# Optional:
#   --num-shards N     Number of GPU processes (default: 2)
#   --gpus LIST        Comma-separated CUDA devices (default: 0,1,...)
#   --python BIN       Python executable (default: python3)
#   --no-plot          Skip the merged FA/hour PNG
#
# Remaining arguments are passed to every eval_musan_fa.py process.
# prep.num_shards / prep.shard_index / prep.output_dir are owned by this
# launcher and must not also be set in the override list.
#
# Example (2x V100, official no-overlap grid, fp32 scores):
#   bash scripts/eval_musan_fa_shards.sh \
#     +experiment=icefall_zipformer_stage2 \
#     'prep.keyword=hey eva' \
#     'prep.keyword_phonemes=HH EY1 IY1 V AH0' \
#     prep.musan_root=/path/to/musan \
#     prep.stage2_ckpt=/path/to/stage2.pt \
#     prep.output_dir=/path/to/out \
#     prep.batch_size=64 \
#     prep.num_workers=8

set -euo pipefail

NUM_SHARDS=2
GPUS=""
PYTHON_BIN="python3"
PLOT_FLAG=""
OVERRIDES=()

usage() {
  sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-plot)
      PLOT_FLAG="--no-plot"
      shift
      ;;
    -h|--help)
      usage
      ;;
    prep.output_dir=*)
      OUTPUT_DIR="${1#prep.output_dir=}"
      OVERRIDES+=("$1")
      shift
      ;;
    prep.num_shards=*|prep.shard_index=*)
      echo "ERROR: $1 is set by this launcher; omit it from the override list." >&2
      exit 1
      ;;
    *)
      OVERRIDES+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "ERROR: prep.output_dir=... is required." >&2
  exit 1
fi
if ! [[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --num-shards must be a positive integer." >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ -z "${GPUS}" ]]; then
  GPU_LIST=()
  for ((index=0; index<NUM_SHARDS; index++)); do
    GPU_LIST+=("${index}")
  done
fi
if [[ ${#GPU_LIST[@]} -lt ${NUM_SHARDS} ]]; then
  echo "ERROR: --gpus has ${#GPU_LIST[@]} devices but --num-shards=${NUM_SHARDS}." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
PIDS=()
for ((index=0; index<NUM_SHARDS; index++)); do
  shard_dir="${OUTPUT_DIR}/shard_${index}"
  mkdir -p "${shard_dir}"
  echo "Launching shard ${index} on GPU ${GPU_LIST[${index}]} -> ${shard_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST[${index}]}" \
    "${PYTHON_BIN}" scripts/eval_musan_fa.py \
      "${OVERRIDES[@]}" \
      "prep.output_dir=${shard_dir}" \
      "prep.num_shards=${NUM_SHARDS}" \
      "prep.shard_index=${index}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "ERROR: one or more shard processes failed." >&2
  exit 1
fi

merged_dir="${OUTPUT_DIR}/merged"
echo "Merging shards -> ${merged_dir}"
if [[ -n "${PLOT_FLAG}" ]]; then
  "${PYTHON_BIN}" scripts/merge_musan_fa.py "${OUTPUT_DIR}" "${merged_dir}" --no-plot
else
  "${PYTHON_BIN}" scripts/merge_musan_fa.py "${OUTPUT_DIR}" "${merged_dir}"
fi
echo "Done -> ${merged_dir}"
