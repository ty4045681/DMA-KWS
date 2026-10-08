#!/usr/bin/env bash
# False-alarm campaign for the five v4.2 models (sinkfit checkpoints).
# Same protocol as the v4.1 campaign: keyword "hey eva", threshold 0.5,
# no stage2 calibration, fp16, batch 64, MUSAN held-out 902 files / 43.7 h,
# LibriSpeech train-other-500 stride-6 subset, 3 s and 1 s window grids.
set -u
cd /home/ubuntu/dma-kws
. /home/ubuntu/.dma-kws-env.sh
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FA=data/dma-kws/exp/stage2_qbyt/fa
MUSAN_LIST=data/dma-kws/processed/musan_split/eval_musan.list
LS_LIST=data/dma-kws/processed/ls-other-500-stride6.list
LOG="$FA/fa_campaign_v42.log"
WORKERS=6
mkdir -p "$FA"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_one() {
  local tag="$1" exp="$2" ckpt="$3" corpus="$4" window="$5" hop="$6" outname="$7"
  local root list
  if [ -f "$FA/$outname/summary.json" ]; then log "SKIP $outname: summary.json already present"; return 0; fi
  if [ "$corpus" = "musan" ]; then root=data/dma-kws/raw/musan; list="$MUSAN_LIST"; else root=data/dma-kws/raw/LibriSpeech; list="$LS_LIST"; fi
  log "START $outname window=$window hop=$hop"
  .venv/bin/python scripts/eval_musan_fa.py     +experiment="$exp"     +stage2.qbyt_readout.sink_readout=additive     +stage2.qbyt_readout.sink_zero_init=true     prep.keyword="hey eva"     'prep.keyword_phonemes=HH EY1 IY1 V AH0'     prep.musan_root="$root"     prep.musan_audio_list_path="$list"     prep.stage2_ckpt="$ckpt"     prep.window_sec="$window" prep.hop_sec="$hop"     prep.batch_size=64 prep.num_workers="$WORKERS" prep.amp=fp16     prep.output_dir="$FA/$outname" > "/tmp/fa_$outname.log" 2>&1
  log "END   $outname rc=$?"
}

# tag experiment checkpoint
JOBS="
v42-zhen icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-zh-en/checkpoints/SS-zh-en/version_0/stage2_step003000_sinkfit.pt
v42-gs icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gs_50k data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-gsfinetune/checkpoints/SS-gsfinetune/version_0/stage2_step001500_sinkfit.pt
v42-gsbase-stream icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gsbase_stream_50k data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-gsbase-stream/checkpoints/SS-gsbase-stream/version_0/stage2_step003000_sinkfit.pt
v42-gsbase-full icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gsbase_fullctx_50k data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-gsbase-fullctx/checkpoints/SS-gsbase-fullctx/version_0/stage2_step001500_sinkfit.pt
v42-paper icefall_zipformer_stage2_eps_softmin_v41_musan_cached_paperstage1_50k data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-paperstage1/checkpoints/SS-paperstage1/version_0/stage2_step003000.pt.sinkfit.pt
"

log "v4.2 FA campaign start (workers=$WORKERS)"
echo "$JOBS" | while read -r tag exp ckpt; do
  [ -z "$tag" ] && continue
  run_one "$tag" "$exp" "$ckpt" musan 3.0 3.0 "$tag-musan"
done
echo "$JOBS" | while read -r tag exp ckpt; do
  [ -z "$tag" ] && continue
  run_one "$tag" "$exp" "$ckpt" musan 1.0 1.0 "$tag-musan-1s0"
done
echo "$JOBS" | while read -r tag exp ckpt; do
  [ -z "$tag" ] && continue
  run_one "$tag" "$exp" "$ckpt" ls 3.0 3.0 "$tag-ls"
done

log "campaign finished; v4.2 vs v4.1 FA comparison:"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, os
base = {
 "v42-zhen": "v41", "v42-gs": "gs-v41", "v42-gsbase-stream": "gsbase-stream-v41",
 "v42-gsbase-full": "gsbase-full-v41", "v42-paper": None,
}
def fa(path):
    if not os.path.exists(path): return None
    m = json.load(open(path)).get("metrics", {}) or {}
    return m.get("fa_per_hour")
print("| tag | v4.2 3s FA/h | v4.2 1s FA/h | v4.2 LS FA/h | v4.1 3s FA/h | v4.1 1s FA/h | v4.1 LS FA/h |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for tag, b in base.items():
    row = [tag, fa("data/dma-kws/exp/stage2_qbyt/fa/" + tag + "-musan/summary.json"),
           fa("data/dma-kws/exp/stage2_qbyt/fa/" + tag + "-musan-1s0/summary.json"),
           fa("data/dma-kws/exp/stage2_qbyt/fa/" + tag + "-ls/summary.json")]
    if b:
        row += [fa("data/dma-kws/exp/stage2_qbyt/fa/" + b + "-musan/summary.json"),
                fa("data/dma-kws/exp/stage2_qbyt/fa/" + b + "-musan-1s0/summary.json"),
                fa("data/dma-kws/exp/stage2_qbyt/fa/" + b + "-ls/summary.json")]
    else:
        row += [None, None, None]
    cells = [("" if v is None else ("%.4f" % v)) for v in row]
    print("| %s | %s | %s | %s | %s | %s | %s |" % tuple(cells))
PY
log "v4.2 FA campaign done"
