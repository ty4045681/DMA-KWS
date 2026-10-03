#!/usr/bin/env bash
# Paper-stage-I chain: after the gsbase **streaming** runs (phase B) finish:
#   1) extract the Wenet-profile training fbank -> features/fbank
#   2) extract the Wenet-profile eval fbank     -> features/fbank_eval
#   3) launch the paper Stage I conformer runs  (phase C: v4.1 / v4 / v3, 50k steps)
#
# The paper's Stage I (train/stage1_wenet/examples/librispeech-g2p/s0/conf/
# train_conformer_1460.yaml + wenet/dataset/processor.py) computes
#   waveform * (1<<15) -> torchaudio kaldi.fbank(..., snip_edges=True, low_freq=20,
#   high_freq=0, povey, no cmvn)
# which is exactly this repo's default fbank profile (configs/fbank/default.yaml) with
# dither disabled for a precomputed tree. The existing icefall-profile tree is NOT
# compatible (measured: 2 frames fewer, mean +16 vs -4.5), hence the separate extraction.
set -u
cd /home/ubuntu/dma-kws
LOG=data/dma-kws/exp/stage2_qbyt/chain_paper.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "paper chain watcher started; waiting for phase B training processes"
while pgrep -f "train_stage2_qbyt.p[y]" >/dev/null 2>&1; do
  sleep 60
done

log "phase B finished; gsbase-stream results:"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import pandas as pd
df = pd.read_csv("data/dma-kws/exp/stage2_qbyt/runs.csv")
cols = [c for c in ("run_id","global_step","duration_s","best/val/auc","best/val/eer","best/val/tpr_at_fpr_1e_2") if c in df.columns]
print(df[df["run_id"].str.contains("gsbase-stream", na=False)][cols].to_string(index=False))
PY

log "STEP 1/3: Wenet-profile training fbank -> data/dma-kws/features/fbank"
.venv/bin/python scripts/prepare_stage2_paper.py \
  +experiment=paper_ls_gs1460 \
  prep.input_parquet=data/dma-kws/raw/GigaPhrase-1000/aggregated_segments_with_g2p_distance.parquet \
  prep.decoded_parquet_root=data/dma-kws/raw/LibriPhrase-GS-1460/decoded \
  prep.output_subdir=stage2_qbyt/ls-gs-1460 \
  +prep.fbank_dir=data/dma-kws/features/fbank \
  fbank.dither=0.0 \
  prep.num_workers=8 >> "$LOG" 2>&1
log "step 1 rc=$? ; fbank files=$(find data/dma-kws/features/fbank -name '*.npy' 2>/dev/null | wc -l)"

log "STEP 2/3: Wenet-profile eval fbank -> data/dma-kws/features/fbank_eval"
.venv/bin/python scripts/prepare_stage2_eval_fbank.py \
  +experiment=icefall_zipformer_stage2_v3_musan_cached_paperstage1_50k \
  fbank.dither=0.0 prep.from_csv=true prep.num_workers=8 >> "$LOG" 2>&1
log "step 2 rc=$? ; eval fbank files=$(find data/dma-kws/features/fbank_eval -name '*.npy' 2>/dev/null | wc -l)"

log "STEP 3/3: launching phase C (paper Stage I conformer, non-streaming per the paper config)"
for pair in "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_paperstage1_50k qbyt-v41-musan-paperstage1" \
            "icefall_zipformer_stage2_v4_musan_cached_paperstage1_50k qbyt-v4-musan-paperstage1" \
            "icefall_zipformer_stage2_v3_musan_cached_paperstage1_50k qbyt-v3-musan-paperstage1"; do
  set -- $pair
  bash scripts/start_stage2_qbyt_tmux.sh "$1" "$2" 2>&1 | tee -a "$LOG"
done
log "phase C launched; paper chain watcher exiting"
