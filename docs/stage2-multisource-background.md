# Stage II multi-source background negatives

MUSAN, DNS, and FSD50K can be mixed as weighted non-speech background negatives
for v4.1 QbyT. QbyT forward, sink count, readout version 4, loss, optimizer, and
encoder freeze are unchanged. Empty `sources` keeps the old single-list / v1
cache path.

Paths in examples are placeholders. Replace them. This document is not a claim
that full datasets or GPU training were run.

## Prepare catalogs

`configs/background_sources/example.yaml` is a plain YAML file (not a Hydra
experiment). `--config` is parsed with `yaml.safe_load`. The CLI does not
download data and does not scan directories other than each source `root` /
`metadata` path.

```yaml
output_dir: /data/background
seed: 2025
sources:
  - adapter: musan
    id: musan
    root: /data/musan
    split_dir: /data/musan_split
    eligible_ids_file: /data/curation/musan_eligible.list
    split_policy: preserve
  - adapter: dns
    id: dns
    root: /data/dns/noise
    metadata: ""
    eligible_ids_file: /data/curation/dns_eligible.list
    split_policy: group_random
  - adapter: fsd50k
    id: fsd50k
    root: /data/fsd50k
    metadata: /data/fsd50k/metadata
    eligible_ids_file: /data/curation/fsd50k_eligible.list
    split_policy: preserve
```

```bash
.venv/bin/python scripts/prepare_background_sources.py \
  --config configs/background_sources/example.yaml
```

Expected layouts (also in `scripts/prepare_background_sources.py --help`):

- **MUSAN:** `<root>/{music,noise,speech}/...`. Optional `split_dir` with
  `train_background.list`, `eval_musan.list`, `split.json`. `preserve` keeps
  those members; old eval maps to test; val is carved from remaining train
  groups.
- **DNS:** walks only the configured noise root (`.wav/.flac/.mp3/.m4a`). Does
  not scan sibling clean/speech trees. Directory presence is not eligibility.
- **FSD50K:** `<root>/FSD50K.{dev,eval}_audio/`. Ground truth
  `dev.csv` / `eval.csv` under `<root>/FSD50K.ground_truth` or `metadata`.
  Clip info must be official `FSD50K.metadata/{dev,eval}_clips_info_FSD50K.json`
  (`uploader`, `license`). Official eval is test and is never reshuffled into
  train.

`eligible_ids_file` is a one-id-per-line allowlist; its hash is stored in the
catalog. Category allow/exclude is only a prefilter. Missing eligibility
defaults to ineligible.

Output (atomic; refuses to overwrite `output_dir`):

```text
<output_dir>/
  audit.json
  musan/{recordings.jsonl,catalog.json,train.list,val.list,test.list}
  dns/...
  fsd50k/...
```

`.list` files are for inspection and old tools; they contain only
`background_eligible=true` recordings for that split. Rejected clips stay in
`recordings.jsonl` / `catalog.json`. Training identity is those two files, not
the lists.

## Cache CLI

`--split-dir` and `--source-manifest` are mutually exclusive.

| Flag | Format | Input |
| --- | --- | --- |
| `--split-dir` | v1 MUSAN cache | existing `split.json` + train/eval lists |
| `--source-manifest` | v2 | generic `recordings.jsonl`; caches eligible **train** only. Build fails if the file bytes no longer match `audio_sha256` in the catalog. |
| `--verify-only` | either | `--output-dir` only; does not need split/manifest or original WAV. v2 also checks cache `content_sha256`, catalog `audio_sha256`, and the train snapshot agree. |

```bash
.venv/bin/python scripts/prepare_stage2_background.py \
  --source-manifest /data/background/dns/recordings.jsonl \
  --output-dir /data/background/dns/cache \
  --experiment icefall_zipformer_stage2_eps_softmin_v41 \
  --crops-per-recording 8 \
  --seed 2025 \
  --workers 4

.venv/bin/python scripts/prepare_stage2_background.py \
  --verify-only \
  --output-dir /data/background/dns/cache
```

K=8 is a smoke starting point. The tool default is 128. A finite crop library
is not infinite online augmentation; the builder prints capacity estimates and
reuse range.

`--verify-only` SHA256s the index and every shard (shape/offset/finite). Training
start only checks manifest identity, recordings/index digests, shard
headers/sizes, and index ranges.

Legacy MUSAN v1:

```bash
.venv/bin/python scripts/prepare_stage2_background.py \
  --split-dir data/dma-kws/processed/musan_split \
  --output-dir data/dma-kws/processed/stage2_background_cache \
  --experiment icefall_zipformer_stage2_eps_softmin_v41
```

## `weight` vs `probability`

These are different knobs. Do not treat source weights as a replacement for
`probability`.

| Field | Where | Meaning |
| --- | --- | --- |
| `stage2.background_negative.probability` | existing negative-branch gate | Conditional on already being in the ~50% negative half. Default `0.25` ⇒ about 1/8 of all base-Stage-II draws are pure background (`label=0`, empty `query_seq`). Does not change the class prior. |
| `sources[].weight` | mix **inside** background | Finite non-negative reals. Zero-weight sources are inactive (no I/O, not in the run signature). Active weights normalize to `p_i = w_i / sum(w)`. Sorted by `id`; list order does not change sampling. |

Base Stage II expected share for source `i` is `0.5 × probability × p_i`.
Overlays use 0.4 / 0.4 / 0.2, so with `probability=0.25` that is 5% / 5% / 2.5%
of all draws. That is a sampling-gate expectation, not a substitute for logged
counts.

Joint adaptation **owns background mass** via its existing strata
(`mix_ratio`, `real_fraction`, `background_keyword_fraction`). Source weights
apply only inside the background stratum; they are not multiplied by the base
Dataset gate again. Domain IDs stay `0=libriphrase, 1=real, 2=tts, 3=background`.
`background_source_id` is a separate integer (sorted active source index; `-1`
when the item is not a background).

`adapt.joint.background_eval_list` is a config conflict when `sources` is
non-empty. Leave it empty; use `validation` plus per-source val recordings.

## Training overlays

This worktree needs `PYTHONPATH=.` for Hydra scripts.

| Overlay | Mode | Run / log / checkpoint suffix |
| --- | --- | --- |
| `icefall_zipformer_stage2_eps_softmin_v41_multisource` | `online` | `...-v41-multisource` |
| `icefall_zipformer_stage2_eps_softmin_v41_multisource_cached` | `fbank_cache` | `...-v41-multisource-cached` |

Inherited top-level MUSAN `audio_list_path` / `cache_manifest` are emptied.
Hydra replaces lists, so the cached overlay **fully restates** each source's
`id` / `weight` / `manifest` / `cache_manifest`. Run dirs are independent of
each other and of the original v4.1 experiment.

```bash
PYTHONPATH=. .venv/bin/python scripts/train_stage2_qbyt.py \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41_multisource \
  --cfg job --resolve

PYTHONPATH=. .venv/bin/python scripts/train_stage2_qbyt.py \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41_multisource_cached \
  --cfg job --resolve
```

Online mode needs original train WAVs. Cached mode needs each source
`cache_manifest`. Switching source set, weights, crop/fbank policy, or catalog
content is a **new run** (`background_data_signature` v2). Resume a matching
multi-source checkpoint only. To change data, start from complete weights with
`run.init_checkpoint`, not `resume`.

## Cache-only training

With `mode=fbank_cache` and `validation.enabled=false`, original WAVs are not
required. The Dataset reads mmap shards. Keep `validation.enabled=false` on a
cache-only machine; enabled validation always crops from original val audio.

## Risks

- **Limited crop diversity.** K pre-drawn crops per recording are a finite
  library. Same seed/K/`cache_id` is reproducible; it is not the same as
  unbounded online crops.
- **Unknown provenance.** SHA256 catches byte-identical copies. `origin_ids`
  block known derivatives across sources. Missing upstream IDs set
  `provenance_complete=false` and a capability limit. That is **not** a proof
  that transcode near-duplicates are absent.
- **Validation needs original audio.** `validation.enabled=true` requires
  eligible val waveforms on disk, even if training is cached.
- **Strict resume.** Source, weight, policy, fbank, probability, `enabled`, or
  catalog identity changes cannot resume. Disabling background with `sources`
  still set does not open manifests; the v2 signature still changes. Legacy ↔
  multi-source is also a new run.

## CPU fixture (do not duplicate)

No extra demo script. The pipeline is already covered by tests that write
short WAVs into a temp dir and commit no large audio:

1. Adapter metadata + short WAV → prepare:
   `tests/test_background_source_adapters.py`
2. Catalog → cache → delete original WAV → Dataset batch (validation off,
   no decode / no `FbankExtractor`):
   `tests/test_background_sources_integration.py`
   (`test_cache_only_training_batch_after_wavs_removed`)

## Design mapping (reuse vs new)

Plan names that already existed were reused rather than duplicated.

| Role | Reused | New |
| --- | --- | --- |
| MUSAN discovery / grouping | `dma_kws/data_prep/musan_split.py` | `MusanSourceAdapter` wraps it |
| Catalog types, hashes, isolation | — | `dma_kws/data_prep/background_manifest.py` |
| DNS / FSD50K import | — | `DnsSourceAdapter`, `Fsd50kSourceAdapter` |
| Prepare facade | — | `prepare_background_sources` |
| Online crop / fbank | `TrainingBackgroundSampler`, `background_sampling`, `FbankExtractor` | `OnlineBackgroundSource` calls the same crop contract |
| Cache reader / mmap LRU | `BackgroundFeatureCache`, `ShardStore` | v2 index + `CachedBackgroundSource` |
| Cache builder | `prepare_stage2_background` | `--source-manifest` v2 path; `--split-dir` still v1 |
| Sampler factory | empty `sources` → legacy adapter | `build_background_sampler`, `MultiSourceBackgroundSampler` |
| Dataset gate / labels | `_sample_pair` 50% negative × `probability`; `label=0`, `query_seq=[]` | `background_source_id` on the item |
| Joint mass / domain IDs | `JointAdaptationDataset` strata | source mix only inside background |
| Val crops | legacy `BackgroundValidationDataset` | `MultiSourceBackgroundValidationDataset` (online val) |
| Resume identity | legacy joint path hash when `sources` empty | `background_data_signature` v2 |
| Overlays | inherit v4.1 readout | `*_multisource`, `*_multisource_cached` |

No extra DNS/FSD training datasets, no event bus, no State/Command queue.

## Not executed here

- No full MUSAN / DNS / FSD50K / LibriPhrase download
- No CUDA GPU; no `run.limit_steps=20` real train
- No encoder checkpoint / parquet / wav_dir smoke on this machine
- No paper-scale FA/h or quality claim from the 0.4/0.4/0.2 mix
