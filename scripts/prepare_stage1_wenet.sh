#!/usr/bin/env bash
# Self-contained Stage I prep: LibriSpeech manifests + optional precomputed fbank.
#
# Chain:
#   1. prepare_stage1_librispeech.py  -> train.jsonl / dev.jsonl (phonemes_g2p)
#   2. prepare_stage1_fbank.py        -> features/stage1_fbank/*.npy + fbank_path in manifest
#   3. train_stage1_ctc.py            -> checkpoints + avg_10.ckpt
#   4. Stage II run.init_checkpoint   -> exp/stage1_phoneme_ctc/checkpoints/avg_10.ckpt
#
# Example:
#   bash scripts/prepare_stage1_wenet.sh +experiment=paper_ls460
#   bash scripts/prepare_stage1_wenet.sh +experiment=demo_librispeech100 prep.limit=32
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 +experiment=<name> [hydra overrides...]" >&2
  exit 1
fi

HYDRA_ARGS=("$@")

DICT_PATH="${ROOT}/data/dict/lang_char.txt"
if [[ ! -f "${DICT_PATH}" ]]; then
  echo "ERROR: CharTokenizer dict not found at ${DICT_PATH}" >&2
  exit 1
fi

echo "==> Stage I manifest prep (LibriSpeech + G2P)"
"${PYTHON}" scripts/prepare_stage1_librispeech.py "${HYDRA_ARGS[@]}"

echo "==> Stage I fbank prep (optional offline shards)"
"${PYTHON}" scripts/prepare_stage1_fbank.py "${HYDRA_ARGS[@]}"

echo "Stage I prep complete. Train with:"
echo "  ${PYTHON} scripts/train_stage1_ctc.py ${HYDRA_ARGS[*]} run.devices=<N>"
