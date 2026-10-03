#!/usr/bin/env bash
# Chain watcher: when the gsbase full-context runs (phase A) are no longer running,
# start the gsbase streaming runs (phase B). Runs inside tmux.
#
# Process-based condition: tmux sessions linger after their pane exits, so poll the
# actual training processes instead.
set -u
cd /home/ubuntu/dma-kws
LOG=data/dma-kws/exp/stage2_qbyt/chain_watcher.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "watcher (v2, process-based) started; waiting for training processes to disappear"
while pgrep -f "train_stage2_qbyt.p[y]" >/dev/null 2>&1; do
  sleep 30
done

log "no training processes left; phase A finished. Results:"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import pandas as pd
df = pd.read_csv("data/dma-kws/exp/stage2_qbyt/runs.csv")
cols = [c for c in ("run_id", "global_step", "duration_s", "best/val/auc", "best/val/eer", "best/val/tpr_at_fpr_1e_2") if c in df.columns]
print(df[df["run_id"].str.contains("gsbase-fullctx", na=False)][cols].to_string(index=False))
PY

log "launching phase B (streaming 16/64, base checkpoint)"
for pair in "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gsbase_stream_50k qbyt-v41-musan-gsbase-stream" \
            "icefall_zipformer_stage2_v4_musan_cached_gsbase_stream_50k qbyt-v4-musan-gsbase-stream" \
            "icefall_zipformer_stage2_v3_musan_cached_gsbase_stream_50k qbyt-v3-musan-gsbase-stream"; do
  set -- $pair
  bash scripts/start_stage2_qbyt_tmux.sh "$1" "$2" 2>&1 | tee -a "$LOG"
done
log "phase B launched; phase-A->B watcher exiting"
