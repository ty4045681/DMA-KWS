#!/usr/bin/env bash
# Rebuild the MUSAN fbank crop caches with the training set restricted to music + noise.
#   * cache        (icefall/lhotse profile)  -> used by the icefall-encoder configs
#   * cache_wenet  (Wenet/torchaudio profile) -> used by the paper Stage I conformer configs
# Both are built from the new catalog data/dma-kws/processed/background_mn/musan.
set -u
cd /home/ubuntu/dma-kws
LOG=data/dma-kws/processed/background_mn/cache_build.log
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

MANIFEST=data/dma-kws/processed/background_mn/musan/recordings.jsonl
K="${K:-414}"   # tool suggestion for 774 train recordings at the 10k-step reference demand

log "building icefall-profile cache (K=$K, 4 workers)"
.venv/bin/python scripts/prepare_stage2_background.py \
  --source-manifest "$MANIFEST" \
  --output-dir data/dma-kws/processed/background_mn/musan/cache \
  --experiment icefall_zipformer_stage2_v3_musan_cached_zhen3m_50k \
  --crops-per-recording "$K" --seed 2025 --workers 4 >> "$LOG" 2>&1 &
P1=$!

log "building Wenet-profile cache (K=$K, 4 workers)"
.venv/bin/python scripts/prepare_stage2_background.py \
  --source-manifest "$MANIFEST" \
  --output-dir data/dma-kws/processed/background_mn/musan/cache_wenet \
  --experiment icefall_zipformer_stage2_v3_musan_cached_paperstage1_50k \
  --crops-per-recording "$K" --seed 2025 --workers 4 >> "$LOG" 2>&1 &
P2=$!

wait $P1; log "icefall cache rc=$?"
wait $P2; log "wenet cache rc=$?"

log "verify:"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, subprocess
for name in ("cache", "cache_wenet"):
    p = f"data/dma-kws/processed/background_mn/musan/{name}/manifest.json"
    try:
        m = json.load(open(p))
    except Exception as exc:
        print(f"  {name}: MISSING ({exc})"); continue
    fb = m.get("fbank", {})
    print(f"  {name}: K={m.get('K')} sources={m.get('num_sources')} crops={m.get('num_crops')} "
          f"shards={len(m.get('shards', []))} backend={fb.get('backend')} snip_edges={fb.get('snip_edges')}")
PY
du -sh data/dma-kws/processed/background_mn/musan/cache data/dma-kws/processed/background_mn/musan/cache_wenet 2>/dev/null
log "done"
