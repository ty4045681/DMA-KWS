#!/usr/bin/env bash
# Self-contained Stage I prep: LibriSpeech manifests + optional precomputed fbank.
#
# Chain:
#   1. prepare_stage1_librispeech.py  -> train.jsonl / dev.jsonl (phonemes_g2p)
#   2. prepare_stage1_fbank.py        -> features/stage1_fbank/*.npy + fbank_path in manifest
#   3. train_stage1_ctc.py            -> checkpoints + avg_10.ckpt
#   4. Stage II --init-checkpoint     -> exp/stage1_phoneme_ctc/checkpoints/avg_10.ckpt
#
# Example:
#   bash scripts/prepare_stage1_wenet.sh --config configs/paper_ls460.yaml
#   bash scripts/prepare_stage1_wenet.sh --config configs/demo_librispeech100.yaml --limit 32
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG=""
EXTRA_PREP_ARGS=()
EXTRA_FBANK_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --limit)
      EXTRA_PREP_ARGS+=(--limit "$2")
      EXTRA_FBANK_ARGS+=(--limit "$2")
      shift 2
      ;;
    *)
      EXTRA_PREP_ARGS+=("$1")
      EXTRA_FBANK_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "Usage: $0 --config <yaml> [--limit N]" >&2
  exit 1
fi

DICT_PATH="${ROOT}/data/dict/lang_char.txt"
if [[ ! -f "${DICT_PATH}" ]]; then
  echo "ERROR: CharTokenizer dict not found at ${DICT_PATH}" >&2
  exit 1
fi

echo "==> Stage I manifest prep (LibriSpeech + G2P)"
"${PYTHON}" scripts/prepare_stage1_librispeech.py \
  --config "${CONFIG}" \
  "${EXTRA_PREP_ARGS[@]}"

echo "==> Stage I fbank prep (optional offline shards)"
"${PYTHON}" scripts/prepare_stage1_fbank.py \
  --config "${CONFIG}" \
  "${EXTRA_FBANK_ARGS[@]}"

echo "Stage I prep complete. Train with:"
echo "  ${PYTHON} scripts/train_stage1_ctc.py --config ${CONFIG} --devices <N>"
