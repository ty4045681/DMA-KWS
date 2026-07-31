#!/usr/bin/env bash
# Local smoke: pytest + --help on all CLI entry points.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

echo "==> pytest (${PYTHON})"
"${PYTHON}" -m pytest tests -q

echo "==> script --help"
PYTHON_SCRIPTS=(
  prepare_stage1_librispeech.py
  train_stage1_ctc.py
  prepare_stage2_paper.py
  train_stage2_qbyt.py
  train_stage2_recipe.py
  average_checkpoints.py
  convert_stage2_checkpoints.py
  eval_stage2_libriphrase.py
  run_two_stage_demo.py
)

for script in "${PYTHON_SCRIPTS[@]}"; do
  echo "  scripts/${script}"
  "${PYTHON}" -m py_compile "scripts/${script}"
done

echo "==> shell script syntax"
bash -n scripts/prepare_stage1_wenet.sh

echo "All smoke checks passed."
