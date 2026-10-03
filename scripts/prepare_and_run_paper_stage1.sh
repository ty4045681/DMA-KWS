#!/usr/bin/env bash
# Prepare everything the paper Stage I runs need, then launch them:
#   1) Wenet-profile training fbank (5,837,186 clips)    [7 workers]
#   2) Wenet-profile MUSAN background cache (320,739 crops) [3 workers]
#   3) verify counts, then launch the three paperstage1 runs
#
# Runs alone (no other training), with swap, because the previous attempt was
# OOM-killed at ~25 GB RSS while the GPU runs were also resident.
set -u
cd /home/ubuntu/dma-kws
LOG=data/dma-kws/exp/stage2_qbyt/paper_stage1_prep.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "starting parallel prep: training fbank (7w) + Wenet MUSAN cache (3w)"

.venv/bin/python scripts/prepare_stage2_paper.py \
  +experiment=paper_ls_gs1460 \
  prep.input_parquet=data/dma-kws/raw/GigaPhrase-1000/aggregated_segments_with_g2p_distance.parquet \
  prep.decoded_parquet_root=data/dma-kws/raw/LibriPhrase-GS-1460/decoded \
  prep.output_subdir=stage2_qbyt/ls-gs-1460-wenetprep \
  +prep.fbank_dir=data/dma-kws/features/fbank \
  fbank.dither=0.0 prep.num_workers=7 >> "$LOG" 2>&1 &
FB_PID=$!

.venv/bin/python scripts/prepare_stage2_background.py \
  --source-manifest data/dma-kws/processed/background/musan/recordings.jsonl \
  --output-dir data/dma-kws/processed/background/musan/cache_wenet \
  --experiment icefall_zipformer_stage2_v3_musan_cached_paperstage1_50k \
  --crops-per-recording 323 --seed 2025 --workers 3 >> "$LOG" 2>&1 &
BG_PID=$!

wait $BG_PID; log "musan wenet cache rc=$? manifest=$(ls data/dma-kws/processed/background/musan/cache_wenet/manifest.json 2>/dev/null | wc -l)"
wait $FB_PID; log "train fbank rc=$? files=$(find data/dma-kws/features/fbank -name '*.npy' 2>/dev/null | wc -l)"

FB_N=$(find data/dma-kws/features/fbank -name '*.npy' 2>/dev/null | wc -l)
EV_N=$(find data/dma-kws/features/fbank_eval -name '*.npy' 2>/dev/null | wc -l)
CACHE=$(ls data/dma-kws/processed/background/musan/cache_wenet/manifest.json 2>/dev/null | wc -l)
log "verification: train_fbank=$FB_N (need 5837186) eval_fbank=$EV_N (need 136461) wenet_musan_manifest=$CACHE"

if [ "$FB_N" -lt 5837186 ] || [ "$CACHE" -lt 1 ]; then
  log "NOT launching phase C: prerequisites incomplete"
  exit 1
fi

log "launching phase C (paper Stage I conformer, Wenet fbank + Wenet MUSAN cache)"
for pair in "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_paperstage1_50k qbyt-v41-musan-paperstage1" \
            "icefall_zipformer_stage2_v4_musan_cached_paperstage1_50k qbyt-v4-musan-paperstage1" \
            "icefall_zipformer_stage2_v3_musan_cached_paperstage1_50k qbyt-v3-musan-paperstage1"; do
  set -- $pair
  bash scripts/start_stage2_qbyt_tmux.sh "$1" "$2" 2>&1 | tee -a "$LOG"
done
log "phase C launched; prep script exiting"
