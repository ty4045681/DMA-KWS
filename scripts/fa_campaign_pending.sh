#!/usr/bin/env bash
# False-alarm campaign for the 12 checkpoints that were never evaluated:
#   zh-en v2, GS-finetune (v2/v3/v4/v4.1), GS-base full-context (v3/v4/v4.1),
#   GS-base streaming (v2/v3/v4/v4.1).
# Corpora: MUSAN held-out (902 files / 43.7 h) and the fixed LibriSpeech
# train-other-500 stride-6 subset. Same protocol as the zh-en trio campaign:
# keyword "hey eva", 3 s window / 3 s hop, threshold 0.5, fp16, batch 64.
# Priority order: the three v2 models first, then the rest.
set -u
cd /home/ubuntu/dma-kws
. /home/ubuntu/.dma-kws-env.sh
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FA=data/dma-kws/exp/stage2_qbyt/fa
MUSAN_LIST=data/dma-kws/processed/musan_split/eval_musan.list
LS_LIST=data/dma-kws/processed/ls-other-500-stride6.list
LOG="$FA/fa_campaign_pending.log"
WORKERS="${FA_WORKERS:-4}"
mkdir -p "$FA"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_one() {  # tag experiment run_name corpus
  local tag="$1" exp="$2" run="$3" corpus="$4"
  local ckpt out root list
  ckpt=$(ls -t data/dma-kws/exp/stage2_qbyt/checkpoints/"$run"/*/version_*/stage2_step*.pt 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then log "SKIP $tag-$corpus: no checkpoint under $run"; return 0; fi
  out="$FA/${tag}-${corpus}"
  if [ -f "$out/summary.json" ]; then log "SKIP $tag-$corpus: summary.json already present"; return 0; fi
  if [ "$corpus" = "musan" ]; then root=data/dma-kws/raw/musan; list="$MUSAN_LIST"; else root=data/dma-kws/raw/LibriSpeech; list="$LS_LIST"; fi
  log "START $tag-$corpus  ckpt=$(basename $ckpt)"
  .venv/bin/python scripts/eval_musan_fa.py \
    +experiment="$exp" \
    prep.keyword="hey eva" \
    'prep.keyword_phonemes=HH EY1 IY1 V AH0' \
    prep.musan_root="$root" \
    prep.musan_audio_list_path="$list" \
    prep.stage2_ckpt="$ckpt" \
    prep.window_sec=3.0 prep.hop_sec=3.0 \
    prep.batch_size=64 prep.num_workers="$WORKERS" prep.amp=fp16 \
    prep.output_dir="$out" > "/tmp/fa_${tag}_${corpus}.log" 2>&1
  log "END   $tag-$corpus rc=$?"
}

# (tag, experiment, run_name)
JOBS="
zhen-v2 icefall_zipformer_stage2_v2_musan_cached_zhen3m_stream_50k v2-musan-zhen3m-stream-50k
gsbase-stream-v2 icefall_zipformer_stage2_v2_musan_cached_gsbase_stream_50k v2-musan-gsbase-stream-50k
gs-v2 icefall_zipformer_stage2_v2_musan_cached_gs_stream_50k v2-musan-gs-stream-50k
gsbase-stream-v3 icefall_zipformer_stage2_v3_musan_cached_gsbase_stream_50k v3-musan-gsbase-stream-50k
gsbase-stream-v4 icefall_zipformer_stage2_v4_musan_cached_gsbase_stream_50k v4-musan-gsbase-stream-50k
gsbase-stream-v41 icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gsbase_stream_50k v41-musan-gsbase-stream-50k
gsbase-full-v3 icefall_zipformer_stage2_v3_musan_cached_gsbase_fullctx_50k v3-musan-gsbase-fullctx-50k
gsbase-full-v4 icefall_zipformer_stage2_v4_musan_cached_gsbase_fullctx_50k v4-musan-gsbase-fullctx-50k
gsbase-full-v41 icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gsbase_fullctx_50k v41-musan-gsbase-fullctx-50k
gs-v3 icefall_zipformer_stage2_v3_musan_cached_gs_50k v3-musan-gs-50k
gs-v4 icefall_zipformer_stage2_v4_musan_cached_gs_50k v4-musan-gs-50k
gs-v41 icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gs_50k v41-musan-gs-50k
"

log "campaign start (workers=$WORKERS): MUSAN first for every tag, then LibriSpeech"
for corpus in musan ls; do
  echo "$JOBS" | while read -r tag exp run; do
    [ -z "$tag" ] && continue
    run_one "$tag" "$exp" "$run" "$corpus"
  done
done

log "campaign finished; summary:"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import glob, json, os
rows = []
for d in sorted(glob.glob("data/dma-kws/exp/stage2_qbyt/fa/*")):
    p = os.path.join(d, "summary.json")
    if not os.path.isfile(p):
        continue
    tag = os.path.basename(d)
    if not any(k in tag for k in ("zhen-v2", "gsbase", "gs-v2", "gs-v3", "gs-v4", "gs-v41")):
        continue
    s = json.load(open(p))
    m = s.get("metrics", {}) or {}
    rows.append((tag, m.get("num_windows"), m.get("audio_hours"), m.get("accepts", m.get("num_accepts")), m.get("fa_per_hour"), m.get("fa_per_24_hours")))
print("| tag | windows | hours | accepts | FA/hour | FA/24h |")
print("| --- | --- | --- | --- | --- | --- |")
for r in rows:
    print("| %s | %s | %s | %s | %s | %s |" % r)
PY
log "campaign log written"
