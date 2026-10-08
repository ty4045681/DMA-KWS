#!/usr/bin/env bash
# Launch the GS-KWS-base music+noise pair once the paper Stage I conformer runs finish, so
# the box never hosts more concurrent runs than it can feed (CPU is the bottleneck here).
set -u
cd /home/ubuntu/dma-kws
LOG=data/dma-kws/exp/stage2_qbyt/chain_mn_gsbase.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "waiting for the paperstage1 runs to finish"
while pgrep -f 'train_stage2_qbyt.p[y].*paperstage1' >/dev/null 2>&1; do
  sleep 60
done
log "paperstage1 finished; launching the GS-base music+noise pair"
for pair in "icefall_zipformer_stage2_v2_musan_cached_gsbase_mn_50k qbyt-v2-musan-gsbase-mn" \
            "icefall_zipformer_stage2_v3_musan_cached_gsbase_mn_50k qbyt-v3-musan-gsbase-mn"; do
  set -- $pair
  bash scripts/start_stage2_qbyt_tmux.sh "$1" "$2" 2>&1 | tee -a "$LOG"
done
log "GS-base pair launched; watcher exiting"
