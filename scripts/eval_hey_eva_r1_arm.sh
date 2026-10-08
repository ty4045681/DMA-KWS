#!/usr/bin/env bash
# Evaluate the R1-based "hey eva" adaptation end to end.
#
# Two held-out axes measure the target keyword:
#   real  -> 4 held-out speakers (wake rate / near-miss false-trigger rate)
#   tts   -> 22 held-out voices, including the confusable variants
# Two background corpora measure false alarms at a fixed threshold, in the
# deployment scoring window and in the stricter 1 s grid:
#   MUSAN  music/noise, 43.7 h held out from training backgrounds
#   LibriSpeech other-500 stride-6 subset, 82.7 h speech
# Both the R1 base and the adapted model are scored so the delta is visible.
#
# Usage: bash scripts/eval_hey_eva_r1_arm.sh [ADAPTED_CHECKPOINT]
set -euo pipefail
cd /home/ubuntu/dma-kws
source /home/ubuntu/.dma-kws-env.sh
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
EXP=adapt_hey_eva_v42
OUT=outputs/hey_eva_v42_eval
BASE=data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-R1/checkpoints/SS-R1/version_0/stage2_step003000.pt
ADAPTED=${1:-data/dma-kws/exp/stage2_adapt_v42/hey_eva_joint_r1/stage2_adapted.pt}
MUSAN_LIST=data/dma-kws/processed/musan_split/eval_musan.list
LS_LIST=data/dma-kws/processed/ls-other-500-stride6.list

clips() {  # tag ckpt manifest
  local tag="$1" ckpt="$2" manifest="$3"
  if [ -f "$OUT/$tag/summary.json" ]; then echo "SKIP clips $tag"; return 0; fi
  echo "== clips $tag"
  $PY scripts/eval_stage2_clips.py "+experiment=$EXP" \
    "prep.manifest=$manifest" "prep.stage2_ckpt=$ckpt" \
    "prep.output_dir=$OUT/$tag" run.device=cuda
}

fa() {  # tag ckpt root list window hop
  local tag="$1" ckpt="$2" root="$3" list="$4" window="$5" hop="$6"
  if [ -f "$OUT/$tag/summary.json" ]; then echo "SKIP FA $tag"; return 0; fi
  echo "== FA $tag window=$window hop=$hop"
  $PY scripts/eval_musan_fa.py "+experiment=$EXP" \
    prep.keyword='hey eva' 'prep.keyword_phonemes=HH EY1 IY1 V AH0' \
    "prep.musan_root=$root" "prep.musan_audio_list_path=$list" \
    "prep.stage2_ckpt=$ckpt" "prep.window_sec=$window" "prep.hop_sec=$hop" \
    prep.batch_size=64 prep.num_workers=4 prep.amp=fp16 \
    "prep.output_dir=$OUT/$tag"
}

for arm in base:"$BASE" adapted:"$ADAPTED"; do
  label=${arm%%:*}; ckpt=${arm#*:}
  clips "real_r1_${label}" "$ckpt" "$OUT/real_eval_clips.csv"
  clips "tts_r1_${label}"  "$ckpt" "$OUT/tts_eval_clips.csv"
  fa "fa_r1_${label}-musan"     "$ckpt" data/dma-kws/raw/musan       "$MUSAN_LIST" 3.0 3.0
  fa "fa_r1_${label}-musan-1s0" "$ckpt" data/dma-kws/raw/musan       "$MUSAN_LIST" 1.0 1.0
  fa "fa_r1_${label}-ls"        "$ckpt" data/dma-kws/raw/LibriSpeech "$LS_LIST"    3.0 3.0
done

for label in base adapted; do
  $PY scripts/analyze_hey_eva_adapt.py --eval-dir "$OUT/real_r1_${label}" \
    --adapt-manifest data/dma-kws/processed/adapt/hey_eva_v42/manifests/real_eval.csv \
    --source-manifest data/dma-kws/raw/hey_eva_real_v2/manifests/real_reviewed_abs.csv \
    --source real --out "$OUT/real_r1_${label}_slices.json" || true
  $PY scripts/analyze_hey_eva_adapt.py --eval-dir "$OUT/tts_r1_${label}" \
    --adapt-manifest data/dma-kws/processed/adapt/hey_eva_v42/manifests/tts_eval.csv \
    --source-manifest outputs/tts/hey_eva/tts_manifest.csv \
    --source tts --out "$OUT/tts_r1_${label}_slices.json" || true
done

echo "R1 ARM EVAL DONE -> $OUT"
