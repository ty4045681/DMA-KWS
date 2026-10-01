# False-Alarm (误唤醒) Evaluation

How to measure **false accepts per 24 hours** for a trained Stage II QbyT checkpoint, plus
the recorded results for the runs in the [Experiment Log](Experiment-Log).

## 1. What is measured

`scripts/eval_musan_fa.py` slides a fixed-length window over every audio file of a list,
scores each window with the Stage II QbyT verifier for one keyword, and reports:

| field | meaning |
| --- | --- |
| `metrics.fa_per_hour` | over-threshold windows ÷ scored audio hours |
| `metrics.fa_per_24_hours` | the same × 24 — the "几次/24h" deployment number |
| `metrics.fa_per_1000_hours` | the same × 1000 (rare-event view) |
| `file_metrics.file_trigger_rate` | triggered files ÷ scored files — **not** an event rate |

Official grid: `prep.window_sec=3.0`, `prep.hop_sec=3.0` (non-overlapping windows).
FA/hour is **not** comparable across hops: the numerator counts windows, the denominator
counts source-file hours. Every window is a negative — the corpora contain no true wake
words.

Each run writes:

* `results.jsonl` — one JSON object per scored window (keyword, phonemes, score, subset, window span)
* `summary.json` — overall and per-subset (`music` / `noise` / `speech`) metrics
* `fa_per_hour_curve.png` + `fa_per_hour_curve.csv` — threshold sweep with
  `threshold, false_accepts, fa_per_hour, fa_per_24_hours, fa_per_1000_hours`
  (written when `prep.plot_curves=true`, the default)

## 2. Corpora on this host

| corpus | list | size | role |
| --- | --- | --- | --- |
| MUSAN held-out | `data/dma-kws/processed/musan_split/eval_musan.list` | 902 files / **43.7 h** | the 40 % eval side of the MUSAN split — never used as a training background |
| LibriSpeech train-other-500 | `data/dma-kws/raw/LibriSpeech/train-other-500.list` | 148,688 flac / 30 GB / **≈520 h** | clean read speech; the harder false-alarm corpus |

Lists may hold absolute paths or paths relative to the list file (`load_audio_file_list`
resolves and rejects missing/duplicate entries).

## 3. Single checkpoint on MUSAN

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_musan_fa.py \
  +experiment=<experiment matching the checkpoint readout> \
  prep.keyword="hey eva" \
  'prep.keyword_phonemes=HH EY1 IY1 V AH0' \
  prep.musan_root=data/dma-kws/raw/musan \
  prep.musan_audio_list_path=data/dma-kws/processed/musan_split/eval_musan.list \
  prep.stage2_ckpt=<trained stage2 .pt> \
  prep.window_sec=3.0 prep.hop_sec=3.0 \
  prep.batch_size=64 prep.num_workers=8 prep.amp=fp16 \
  prep.output_dir=data/dma-kws/exp/stage2_qbyt/fa/<run-name>
```

**`+experiment=` must match the checkpoint's readout** (version *and* the full
`qbyt_readout` spec): use `..._v3_musan_cached_zhen3m_50k` / `..._v4_...` /
`..._eps_softmin_v41_...` for the v3 / v4 / v4.1 checkpoints. The loader asserts the
version and spec, so a mismatch fails loudly instead of scoring silently.

## 4. LibriSpeech train-other-500

The same tool works on any window-scored corpus: point `musan_root` at the corpus root
so `detect_subset` can label the rows, and pass its list. For train-other-500 the subset
label becomes `train-other-500`:

```bash
... prep.musan_root=data/dma-kws/raw/LibriSpeech \
    prep.musan_audio_list_path=data/dma-kws/raw/LibriSpeech/train-other-500.list \
    ... prep.output_dir=data/dma-kws/exp/stage2_qbyt/fa/<run>-ls-other \
```

≈520 h is ~12× MUSAN, so either shard it (`prep.num_shards` + `prep.shard_index`, or
`scripts/eval_musan_fa_shards.sh`) or score a fixed random subset — keep the identical
subset for every checkpoint you compare.

## 5. Several checkpoints and keywords

`scripts/batch_eval_musan_fa.sh` loops over checkpoints and keywords
(`--keyword`, `--keyword-phonemes`, `--keywords-file`, `--pt PT:OUT`,
`--pts-file`, `--base-out`, `--audio-list`, `--window-sec`, `--hop-sec`,
`--experiment`). With several keywords it writes `OUT/<keyword_slug>/`. Use
`scripts/merge_musan_fa.py` / `scripts/aggregate_musan_fa.py` to merge shards or runs
into comparison tables and `scripts/plot_musan_fa_curve.py` for the threshold sweeps.

Because every readout family needs its own experiment config, invoke the batch script
once per family (one `--experiment` per call).

## 6. Practical notes

* `prep.amp=fp16` on the V100 matches validation-time precision — keep it identical
  across checkpoints, otherwise the score scale (and every threshold) shifts.
* `prep.batch_size` defaults to 64 and `prep.num_workers` to `min(8, cpu_count)`;
  window scoring is CPU-heavy (fbank extraction + keyword G2P/tokenisation per window),
  so workers matter more than batch size.
* Pass `prep.keyword_phonemes` explicitly for reproducible numbers; a blank value falls
  back to automatic G2P.
* Report the threshold with the FA number. The deployment threshold comes from
  `prep.stage2_calibration`, and the curve CSV lets you re-read FA/24 h at any
  threshold without re-scoring.
* Never score `train_background.list`: it is the training side of the MUSAN split and
  the checkpoint has seen it as background. `split.json` records the cut and the
  catalogue hashes.
* `results.jsonl` is the audit trail: window index, span, subset, keyword phonemes and
  the single deployed Stage II score.

## 7. Results — zh-en encoder, LS-GS-1460 + MUSAN backgrounds

Same data, encoder and training budget for all three; only the QbyT readout differs
(see the [Experiment Log](Experiment-Log) for the training curves). Keyword
`hey eva` (`HH EY1 IY1 V AH0`), 3 s window / 3 s hop, `prep.amp=fp16`.

## 7. Results — zh-en encoder, LS-GS-1460 + MUSAN backgrounds

Same data, encoder and training budget for all three checkpoints; only the QbyT readout
differs (training curves: [Experiment Log](Experiment-Log)). Keyword `hey eva`
(`HH EY1 IY1 V AH0`), 3 s window / 3 s hop, `prep.amp=fp16`, deployment threshold 0.5.

### 7.1 MUSAN held-out — 902 files / **43.72 h** / 51,994 windows

| metric | v3 | v4 | v4.1 |
| --- | --- | --- | --- |
| false accepts @ threshold 0.5 | **0** | **0** | **0** |
| **FA per 24 h @ 0.5** | **0** | **0** | **0** |
| per subset (music 17.0 h / noise 2.5 h / speech 24.2 h) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| highest score seen | 0.019 | 0.015 | **0.409** |
| p99.99 / p99.9 score | 0.003 / 0.001 | 0.004 / 0.001 | 0.098 / 0.032 |
| windows ≥ 0.1 / ≥ 0.05 | 0 / 0 | 0 / 0 | 4 / 26 |
| FA per 24 h @ 0.2 / 0.1 / 0.05 | 0 / 0 / 0 | 0 / 0 / 0 | 1.1 / 2.7 / 14.8 |

**Reading.** All three are clean on held-out MUSAN at the deployed threshold: **zero
accepts in 43.7 h ≈ 0 次/24h**, in every subset. The safety margins differ sharply
though — v3 and v4 never score a background window above 0.02 (≈4 % of the threshold),
whereas v4.1's worst window reaches 0.409, so its FA rate starts rising as soon as the
threshold drops below ~0.4 (1.1 per 24 h at 0.2, 2.7 at 0.1). The corpus granularity is
0.55 FAs per 24 h (one window), so "0" here means "no accept in 43.7 h", not a measured
rate below that.

### 7.2 LibriSpeech train-other-500 (stride-6 subset, 24,782 files)

_running — results are appended when the three passes finish._

### 7.3 Raw artefacts

* `data/dma-kws/exp/stage2_qbyt/fa/<tag>-musan/{results.jsonl, summary.json, fa_per_hour_curve.csv, fa_per_hour_curve.png}`
* `data/dma-kws/exp/stage2_qbyt/fa/<tag>-ls-other/…`
* campaign script/logs: `/tmp/fa_campaign.sh`, `/tmp/fa_<tag>_{musan,ls}.log`


## 8. Reproduce

```bash
# one checkpoint
bash -c '...'   # see §3

# three checkpoints, MUSAN held-out
for pair in \
  "icefall_zipformer_stage2_v3_musan_cached_zhen3m_50k:v3-musan-zhen3m-50k/v3-musan-zhen3m-50k/version_1/stage2_step048000.pt:v3" \
  "icefall_zipformer_stage2_v4_musan_cached_zhen3m_50k:v4-musan-zhen3m-50k/v4-musan-zhen3m-50k/version_1/stage2_step048000.pt:v4" \
  "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k:v41-musan-zhen3m-50k/v41-musan-zhen3m-50k/version_9/stage2_step047500.pt:v41" ; do
  exp=${pair%%:*}; rest=${pair#*:}; ckpt=${rest%%:*}; tag=${rest##*:}
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_musan_fa.py \
    +experiment="$exp" prep.keyword="hey eva" \
    'prep.keyword_phonemes=HH EY1 IY1 V AH0' \
    prep.musan_root=data/dma-kws/raw/musan \
    prep.musan_audio_list_path=data/dma-kws/processed/musan_split/eval_musan.list \
    prep.window_sec=3.0 prep.hop_sec=3.0 prep.batch_size=64 prep.num_workers=8 prep.amp=fp16 \
    prep.stage2_ckpt="data/dma-kws/exp/stage2_qbyt/checkpoints/$ckpt" \
    prep.output_dir="data/dma-kws/exp/stage2_qbyt/fa/$tag-musan"
done
```