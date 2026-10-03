#!/usr/bin/env bash
# Extract the Wenet-profile fbank trees (train + eval) for the paper Stage I conformer.
#
#   * parallel-safe with a running training run: the paper parquet / clips / distances are
#     written into a scratch processed dir (stage2_qbyt/ls-gs-1460-wenetprep) so the live
#     run's data tree is never touched; only the feature trees are shared.
#   * profile: repo default (torchaudio_kaldi, snip_edges=True, low_freq 20, high_freq 0,
#     povey, 1<<15 scale) == the paper's compute_fbank, with dither disabled.
set -u
cd /home/ubuntu/dma-kws
LOG=data/dma-kws/exp/stage2_qbyt/wenet_fbank_prep.log
WORKERS="${WORKERS:-6}"
echo "[$(date -u +%H:%M:%S)] wenet fbank prep start (workers=$WORKERS)" >> "$LOG"

.venv/bin/python scripts/prepare_stage2_paper.py \
  +experiment=paper_ls_gs1460 \
  prep.input_parquet=data/dma-kws/raw/GigaPhrase-1000/aggregated_segments_with_g2p_distance.parquet \
  prep.decoded_parquet_root=data/dma-kws/raw/LibriPhrase-GS-1460/decoded \
  prep.output_subdir=stage2_qbyt/ls-gs-1460-wenetprep \
  +prep.fbank_dir=data/dma-kws/features/fbank \
  fbank.dither=0.0 \
  prep.num_workers="$WORKERS" >> "$LOG" 2>&1
echo "[$(date -u +%H:%M:%S)] train fbank rc=$? files=$(find data/dma-kws/features/fbank -name '*.npy' 2>/dev/null | wc -l)" >> "$LOG"

.venv/bin/python scripts/prepare_stage2_eval_fbank.py \
  +experiment=icefall_zipformer_stage2_v3_musan_cached_paperstage1_50k \
  fbank.dither=0.0 prep.from_csv=true prep.num_workers="$WORKERS" >> "$LOG" 2>&1
echo "[$(date -u +%H:%M:%S)] eval fbank rc=$? files=$(find data/dma-kws/features/fbank_eval -name '*.npy' 2>/dev/null | wc -l)" >> "$LOG"
echo "[$(date -u +%H:%M:%S)] wenet fbank prep done" >> "$LOG"
