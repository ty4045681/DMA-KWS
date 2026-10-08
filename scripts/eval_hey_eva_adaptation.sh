#!/usr/bin/env bash
# "hey eva" adaptation evaluation: base vs adapted on the two held-out sets,
# then false-alarm grids for the adapted checkpoint.
#
# Usage: bash scripts/eval_hey_eva_adaptation.sh [ADAPTED_CHECKPOINT]
set -euo pipefail
cd /home/ubuntu/dma-kws
source /home/ubuntu/.dma-kws-env.sh
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
EXP=adapt_hey_eva_v42
DATA=data/dma-kws/processed/adapt/hey_eva_v42
OUT=outputs/hey_eva_v42_eval
BASE=data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-zh-en/checkpoints/SS-zh-en/version_0/stage2_step003000.pt
ADAPTED=${1:-data/dma-kws/exp/stage2_adapt_v42/hey_eva/stage2_adapted.pt}
MUSAN_LIST=data/dma-kws/processed/musan_split/eval_musan.list
LS_LIST=data/dma-kws/processed/ls-other-500-stride6.list
mkdir -p "$OUT"

# Clips manifests carry the keyword + phonemes, so the verifier needs no overrides.
$PY - <<'PY'
from pathlib import Path
from dma_kws.stage2.adapt_dataset import clips_eval_manifest_from_adapt

data = Path("data/dma-kws/processed/adapt/hey_eva_v42")
out = Path("outputs/hey_eva_v42_eval")
out.mkdir(parents=True, exist_ok=True)
for phase in ("real", "tts"):
    clips_eval_manifest_from_adapt(
        data / "manifests" / f"{phase}_eval.csv",
        "hey eva",
        out / f"{phase}_eval_clips.csv",
        manifest_root=data,
    )
    print("wrote", out / f"{phase}_eval_clips.csv")
PY

run_clips() {  # label ckpt phase
  local label="$1" ckpt="$2" phase="$3"
  local dir="$OUT/${phase}_${label}"
  if [ -f "$dir/summary.json" ]; then echo "SKIP $dir"; return 0; fi
  echo "== clips $phase / $label"
  $PY scripts/eval_stage2_clips.py "+experiment=$EXP" \
    "prep.manifest=$OUT/${phase}_eval_clips.csv" \
    "prep.stage2_ckpt=$ckpt" \
    "prep.output_dir=$dir" \
    run.device=cuda
}

run_fa() {  # tag root list window hop
  local tag="$1" root="$2" list="$3" window="$4" hop="$5"
  local dir="$OUT/fa_$tag"
  if [ -f "$dir/summary.json" ]; then echo "SKIP $dir"; return 0; fi
  echo "== FA $tag window=$window"
  $PY scripts/eval_musan_fa.py "+experiment=$EXP" \
    prep.keyword='hey eva' 'prep.keyword_phonemes=HH EY1 IY1 V AH0' \
    "prep.musan_root=$root" "prep.musan_audio_list_path=$list" \
    "prep.stage2_ckpt=$ADAPTED" \
    "prep.window_sec=$window" "prep.hop_sec=$hop" \
    prep.batch_size=64 prep.num_workers=4 prep.amp=fp16 \
    "prep.output_dir=$dir"
}

for phase in real tts; do
  run_clips base "$BASE" "$phase"
  run_clips adapted "$ADAPTED" "$phase"
done

for phase in real tts; do
  for label in base adapted; do
    ckpt=$BASE; [ "$label" = adapted ] && ckpt=$ADAPTED
    if [ "$phase" = real ]; then src=real_reviewed_abs; else src=tts_manifest; fi
    case "$src" in
      real_reviewed_abs) SRC=data/dma-kws/raw/hey_eva_real_v2/manifests/real_reviewed_abs.csv ;;
      tts_manifest) SRC=outputs/tts/hey_eva/tts_manifest.csv ;;
    esac
    $PY scripts/analyze_hey_eva_adapt.py \
      --eval-dir "$OUT/${phase}_${label}" \
      --adapt-manifest "$DATA/manifests/${phase}_eval.csv" \
      --source-manifest "$SRC" \
      --source "$phase" \
      --out "$OUT/${phase}_${label}_slices.json" || true
  done
done

run_fa adapted-musan data/dma-kws/raw/musan "$MUSAN_LIST" 3.0 3.0
run_fa adapted-musan-1s0 data/dma-kws/raw/musan "$MUSAN_LIST" 1.0 1.0
run_fa adapted-ls data/dma-kws/raw/LibriSpeech "$LS_LIST" 3.0 3.0

echo "EVAL DONE -> $OUT"
