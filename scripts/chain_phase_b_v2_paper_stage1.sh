#!/usr/bin/env bash
# Chain: phase B (gsbase streaming) -> v2 (gru_last) trio -> phase C (paper Stage I).
# The Wenet-profile fbank extraction runs in PARALLEL (see
# scripts/prepare_wenet_fbank_parallel.sh, tmux session qbyt-wenetfbank), so this chain
# only waits for it instead of running it inline.
#
#   start: bash scripts/chain_phase_b_v2_paper_stage1.sh
#   watch: tmux attach -t qbyt-chain-paper   /  tail -f .../chain_v2_paper.log
set -u
cd /home/ubuntu/dma-kws
LOG=data/dma-kws/exp/stage2_qbyt/chain_v2_paper.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_idle()   { while pgrep -f "train_stage2_qbyt.p[y]" >/dev/null 2>&1; do sleep 60; done; }
wait_running() {
  local pat="$1"
  for _ in $(seq 1 30); do
    pgrep -f "train_stage2_qbyt.p[y].*${pat}" >/dev/null 2>&1 && return 0
    sleep 10
  done
  return 1
}
show_runs() {
  .venv/bin/python - "$1" <<'PY' 2>&1 | tee -a "$LOG"
import sys, pandas as pd
pat = sys.argv[1]
df = pd.read_csv("data/dma-kws/exp/stage2_qbyt/runs.csv")
cols = [c for c in ("run_id","global_step","duration_s","best/val/auc","best/val/eer","best/val/tpr_at_fpr_1e_2") if c in df.columns]
print(df[df["run_id"].str.contains(pat, na=False)][cols].to_string(index=False))
PY
}

log "chain started: phase B -> v2 trio (parallel with the Wenet fbank extraction) -> phase C"

wait_idle
log "phase B finished:"
show_runs "gsbase-stream"

log "STEP 1/3: launching the v2 (gru_last) trio"
for pair in "icefall_zipformer_stage2_v2_musan_cached_zhen3m_stream_50k qbyt-v2-musan-zhen3m" \
            "icefall_zipformer_stage2_v2_musan_cached_gs_stream_50k qbyt-v2-musan-gs" \
            "icefall_zipformer_stage2_v2_musan_cached_gsbase_stream_50k qbyt-v2-musan-gsbase"; do
  set -- $pair
  bash scripts/start_stage2_qbyt_tmux.sh "$1" "$2" 2>&1 | tee -a "$LOG"
done
wait_running "v2_musan" || log "WARNING: v2 processes did not appear"
wait_idle
log "v2 trio finished:"
show_runs "v2-musan"

log "STEP 2/3: waiting for the parallel Wenet fbank extraction"
while pgrep -f "prepare_stage2_pape[r]" >/dev/null 2>&1; do sleep 60; done
log "extraction finished: train=$(find data/dma-kws/features/fbank -name '*.npy' 2>/dev/null | wc -l) eval=$(find data/dma-kws/features/fbank_eval -name '*.npy' 2>/dev/null | wc -l)"

log "STEP 3/3: launching phase C (paper Stage I conformer, non-streaming per the paper config)"
for pair in "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_paperstage1_50k qbyt-v41-musan-paperstage1" \
            "icefall_zipformer_stage2_v4_musan_cached_paperstage1_50k qbyt-v4-musan-paperstage1" \
            "icefall_zipformer_stage2_v3_musan_cached_paperstage1_50k qbyt-v3-musan-paperstage1"; do
  set -- $pair
  bash scripts/start_stage2_qbyt_tmux.sh "$1" "$2" 2>&1 | tee -a "$LOG"
done
log "phase C launched; chain exiting"
