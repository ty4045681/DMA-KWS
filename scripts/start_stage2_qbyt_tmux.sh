#!/usr/bin/env bash
# Start a tuned Stage II QbyT training run inside tmux.
#
#   bash scripts/start_stage2_qbyt_tmux.sh <experiment> [session-name]
#
# Examples:
#   bash scripts/start_stage2_qbyt_tmux.sh icefall_zipformer_stage2_zhen3m_50k
#   bash scripts/start_stage2_qbyt_tmux.sh \
#     icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k qbyt-v41-musan-zhen3m
#   EXTRA_OVERRIDES="run.resume_from=.../last.ckpt" bash scripts/start_stage2_qbyt_tmux.sh ...
#
# Attach / watch / stop:
#   tmux attach -t <session>
#   tmux capture-pane -pt <session> -S -40
#   tmux kill-session -t <session>
#
# Overridable: SESSION_NAME, WINDOW_NAME, CUDA_VISIBLE_DEVICES, PARQUET,
#              EXTRA_OVERRIDES (extra Hydra overrides appended verbatim).
set -euo pipefail

EXPERIMENT="${1:?usage: $0 <experiment> [session-name]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${2:-${SESSION_NAME:-qbyt-${EXPERIMENT}}}"
WINDOW="${WINDOW_NAME:-train}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
PARQUET="${PARQUET:-data/dma-kws/processed/stage2_qbyt/ls-gs-1460/aggregated_segments_with_g2p_distance.parquet}"
PANE_LOG="$ROOT/data/dma-kws/exp/stage2_qbyt/${SESSION}.pane.log"

CMD="cd $ROOT"
CMD="$CMD && export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
CMD="$CMD && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CMD="$CMD && exec .venv/bin/python scripts/train_stage2_qbyt.py"
CMD="$CMD +experiment=$EXPERIMENT"
CMD="$CMD stage2.parquet_file=$PARQUET"
CMD="$CMD ${EXTRA_OVERRIDES:-}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists - attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n "$WINDOW" -c "$ROOT"
tmux set-option -t "$SESSION" history-limit 100000
mkdir -p "$(dirname "$PANE_LOG")"
# Mirror the pane to a file without stealing the TTY from Rich/tqdm.
tmux pipe-pane -t "$SESSION:$WINDOW" -o "cat >> $PANE_LOG"

# bash -lc picks up ~/.profile -> ~/.dma-kws-env.sh (ICEFALL_ROOT, PYTHONPATH, cuDNN).
tmux send-keys -t "$SESSION:$WINDOW" "bash -lc '$CMD'" C-m

echo "started tmux session '$SESSION' (window '$WINDOW')"
echo "  experiment: $EXPERIMENT"
echo "  attach: tmux attach -t $SESSION"
echo "  pane log: $PANE_LOG"
