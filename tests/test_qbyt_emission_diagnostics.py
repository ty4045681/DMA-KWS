"""Contracts of the offline QbyT emission/readout diagnostics.

The whole point of these diagnostics is that an ablation logit is comparable
with the deployed ``qbyt_raw_logit``. That only holds if the deployed ablation
re-derives the score through the identical emission and graph arithmetic, so the
first test is the regression guard for both the ``frame_class_log_probs``
extraction in ``qbyt/model.py`` and the filler reconstruction here.
"""

import json
import math

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("torch")

from dma_kws.inference.qbyt_diagnostics import (
    AblationSpec,
    _target_llr,
    clip_emission_diagnostics,
    summarize_emission_diagnostics,
)
from dma_kws.pathing import load_qbyt_class
from scripts import probe_qbyt_emissions

ENCODER_DIM = 24
EMBED_DIM = 32


class _ProbeFakeDataset:
    """Module-level so DataLoader worker processes can pickle it.

    Index 2 yields no features, standing in for a clip that subsamples away to
    zero encoder frames.
    """

    def __init__(self, *, audio_paths, **_kwargs):
        self._paths = list(audio_paths)

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        return index, (None if index == 2 else torch.zeros(20, 80)), 1.0


def _model(*, span: int = 64, gap: int = 1, num_embeds: int = 12):
    torch.manual_seed(0)
    QbyT = load_qbyt_class()
    return QbyT(
        encoder_output_size=ENCODER_DIM,
        num_embeds=num_embeds,
        embed_dim=EMBED_DIM,
        post_num_layers=2,
        local_context_kernel=5,
        min_phone_duration_frames=1,
        max_phone_duration_frames=4,
        max_inter_phone_gap_frames=gap,
        max_keyword_span_frames=span,
        weakest_phone_temperature=0.2,
        weakest_phone_weight=1.0,
        dropout=0.0,
    ).eval()


def _batch(model, phone_counts, frame_counts):
    generator = torch.Generator().manual_seed(7)
    phone_width, frame_width = max(phone_counts), max(frame_counts)
    anchors = torch.zeros(len(phone_counts), phone_width, dtype=torch.long)
    speech = torch.zeros(len(frame_counts), frame_width, ENCODER_DIM)
    for index, (phones, frames) in enumerate(zip(phone_counts, frame_counts)):
        anchors[index, :phones] = torch.randint(
            1, model.text_projection.num_embeddings, (phones,), generator=generator
        )
        speech[index, :frames] = torch.randn(frames, ENCODER_DIM, generator=generator)
    return (
        speech,
        anchors,
        torch.tensor(frame_counts, dtype=torch.long),
        torch.tensor(phone_counts, dtype=torch.long),
    )


def test_deployed_ablation_reproduces_the_forward_utterance_logit():
    model = _model()
    speech, anchors, speech_lengths, anchor_lengths = _batch(model, [5, 3], [24, 16])
    with torch.no_grad():
        expected, _ = model(
            speech,
            anchors,
            speech_lengths=speech_lengths,
            text_lengths=anchor_lengths,
        )
    records = clip_emission_diagnostics(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        ablations=(AblationSpec("deployed"),),
    )
    for index, record in enumerate(records):
        assert record["deployed"]["logit"] == pytest.approx(
            float(expected[index]), abs=1e-4
        )


def test_frame_class_log_probs_is_a_normalized_posterior():
    model = _model()
    speech, _anchors, speech_lengths, _anchor_lengths = _batch(model, [4], [20])
    frame_mask = model._length_mask(speech_lengths, speech.size(1))
    with torch.no_grad():
        log_probs = model.frame_class_log_probs(speech, frame_mask)
    assert log_probs.shape == (
        1,
        speech.size(1),
        model.text_projection.num_embeddings + 1,
    )
    totals = log_probs.logsumexp(dim=-1)
    assert torch.allclose(totals, torch.zeros_like(totals), atol=1e-5)


def test_one_vs_rest_filler_charges_a_substitution_onto_an_in_query_phone():
    """The mechanism under test, on a hand-built posterior.

    The frame is acoustically phone 2, and the query contains both phone 1 and
    phone 2. The query-relative filler removes phone 2's mass from the
    denominator, so phone 1 is charged against 0.25 instead of 0.85.
    """
    probs = torch.tensor([[[0.15, 0.60, 0.10, 0.05, 0.07, 0.03]]])
    class_log_probs = probs.log()
    anchors = torch.tensor([[1, 2]])
    phone_mask = torch.tensor([[True, True]])
    frame_mask = torch.tensor([[True]])

    query_relative, _ = _target_llr(
        class_log_probs, anchors, phone_mask, frame_mask, 4, filler="query_relative"
    )
    one_vs_rest, _ = _target_llr(
        class_log_probs, anchors, phone_mask, frame_mask, 4, filler="one_vs_rest"
    )

    # Substituted phone: 0.10 + 0.05 + 0.07 + 0.03 = 0.25 of filler mass left.
    assert float(query_relative[0, 0, 0]) == pytest.approx(math.log(0.15 / 0.25), abs=1e-5)
    assert float(one_vs_rest[0, 0, 0]) == pytest.approx(math.log(0.15 / 0.85), abs=1e-5)
    assert float(query_relative[0, 0, 0]) > float(one_vs_rest[0, 0, 0]) + 1.0

    # The correct phone also loses, which is why the fix moves the whole score
    # scale and the deployment threshold has to be re-fitted after adopting it.
    assert float(query_relative[0, 1, 0]) == pytest.approx(math.log(0.60 / 0.25), abs=1e-5)
    assert float(one_vs_rest[0, 1, 0]) == pytest.approx(math.log(0.60 / 0.40), abs=1e-5)
    assert float(query_relative[0, 1, 0]) > float(one_vs_rest[0, 1, 0])


def test_target_llr_zeroes_padded_phones_and_frames():
    probs = torch.tensor(
        [[[0.15, 0.60, 0.10, 0.05, 0.07, 0.03], [0.20, 0.20, 0.20, 0.20, 0.10, 0.10]]]
    )
    anchors = torch.tensor([[1, 0]])
    phone_mask = torch.tensor([[True, False]])
    frame_mask = torch.tensor([[True, False]])
    for filler in ("query_relative", "one_vs_rest"):
        target_llr, _ = _target_llr(
            probs.log(), anchors, phone_mask, frame_mask, 4, filler=filler
        )
        assert float(target_llr[0, 1, 0]) == 0.0
        assert float(target_llr[0, 0, 1]) == 0.0


def test_an_empty_query_is_refused_rather_than_scored():
    model = _model()
    speech, anchors, speech_lengths, anchor_lengths = _batch(model, [3], [18])
    with pytest.raises(ValueError, match="non-empty query"):
        clip_emission_diagnostics(
            model, speech, anchors, speech_lengths, torch.tensor([0])
        )
    with pytest.raises(ValueError, match="non-empty query"):
        clip_emission_diagnostics(
            model,
            speech,
            anchors[:, :0],
            speech_lengths,
            torch.tensor([0]),
        )


def test_no_legal_path_reports_none_instead_of_the_aligner_sentinel():
    """A sentinel left in a numeric field would poison every aggregate mean."""
    model = _model(span=4)
    speech, anchors, speech_lengths, anchor_lengths = _batch(model, [6], [24])
    record = clip_emission_diagnostics(
        model, speech, anchors, speech_lengths, anchor_lengths
    )[0]
    deployed = record["deployed"]
    assert deployed["has_legal_path"] is False
    # The score itself stays numeric: the sentinel *is* what deployment returns.
    assert deployed["logit"] == pytest.approx(model.aligner.invalid_score)
    assert deployed["prefix_llr"] is None
    assert deployed["weakest_phone_evidence"] is None
    # phone_evidence[i] belongs to the *prefix* 0..i, and a 4-frame span cap
    # still admits the first four prefixes; only the tail is undefined.
    assert deployed["phone_evidence"][0] is not None
    assert deployed["phone_evidence"][3] is not None
    assert deployed["phone_evidence"][4] is None
    assert deployed["phone_evidence"][5] is None

    summary = summarize_emission_diagnostics([{**record, "group": "g"}])
    assert summary["g"]["deployed"]["prefix_llr"] is None
    assert summary["g"]["deployed"]["illegal_path_rate"] == 1.0


def test_quantiles_ignore_undefined_clips_but_keep_an_honest_count():
    legal = {"group": "g", "encoder_frames": 10, "masked_query_mass_mean": 0.2,
             "deployed": {"logit": -1.0, "prefix_llr": -1.0, "one_edit_vs_filler": 2.0,
                          "weakest_phone_evidence": 0.5, "phone_top_frames_llr_mean": 0.1,
                          "has_legal_path": True, "best_frame_span": 8,
                          "best_frames_monotonic": True}}
    illegal = {**legal, "deployed": {**legal["deployed"], "prefix_llr": None,
                                     "weakest_phone_evidence": None,
                                     "has_legal_path": False}}
    summary = summarize_emission_diagnostics([legal, illegal])["g"]
    assert summary["num_clips"] == 2
    assert summary["deployed"]["prefix_llr"]["count"] == 1
    assert summary["deployed"]["logit"]["count"] == 2
    assert summary["deployed"]["illegal_path_rate"] == 0.5


def test_relaxed_bounds_ablation_recovers_a_keyword_longer_than_the_span_cap():
    # 6 phones need 6 frames at min_phone_duration_frames=1, so a 4-frame span
    # cap makes every path illegal regardless of the acoustic evidence.
    model = _model(span=4)
    speech, anchors, speech_lengths, anchor_lengths = _batch(model, [6], [24])
    records = clip_emission_diagnostics(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        ablations=(
            AblationSpec("deployed"),
            AblationSpec("relaxed_bounds", max_keyword_span_frames=50),
        ),
    )
    deployed = records[0]["deployed"]
    relaxed = records[0]["relaxed_bounds"]
    assert deployed["has_legal_path"] is False
    assert relaxed["has_legal_path"] is True
    assert relaxed["logit"] > deployed["logit"]


def test_masked_query_mass_matches_the_deployed_filler_mask():
    model = _model()
    speech, anchors, speech_lengths, anchor_lengths = _batch(model, [4], [18])
    frame_mask = model._length_mask(speech_lengths, speech.size(1))
    with torch.no_grad():
        probs = model.frame_class_log_probs(speech, frame_mask).exp()
    frames = int(speech_lengths[0])
    used = {int(anchors[0, position]) - 1 for position in range(int(anchor_lengths[0]))}
    expected = float(probs[0, :frames, sorted(used)].sum(dim=-1).mean())
    record = clip_emission_diagnostics(
        model, speech, anchors, speech_lengths, anchor_lengths
    )[0]
    assert record["masked_query_mass_mean"] == pytest.approx(expected, abs=1e-5)
    assert 0.0 <= record["masked_query_mass_mean"] <= 1.0


def test_masked_query_mass_counts_a_repeated_phone_once():
    """``filler_class_mask`` removes a phone set, not a multiset.

    A keyword like "hey google" uses ``G`` twice; summing per position would
    report a discount larger than the probability the filler actually loses, and
    could exceed 1.0.
    """
    model = _model()
    speech, anchors, speech_lengths, anchor_lengths = _batch(model, [4], [18])
    repeated = anchors.clone()
    repeated[0, 1] = repeated[0, 0]
    repeated[0, 3] = repeated[0, 0]
    records = clip_emission_diagnostics(
        model, speech, repeated, speech_lengths, anchor_lengths
    )
    distinct = clip_emission_diagnostics(
        model, speech, anchors, speech_lengths, anchor_lengths
    )
    assert records[0]["masked_query_mass_mean"] <= 1.0
    assert records[0]["masked_query_mass_mean"] < distinct[0]["masked_query_mass_mean"]


def test_diagnostics_expose_the_per_phone_span_the_bounds_are_compared_against():
    model = _model()
    speech, anchors, speech_lengths, anchor_lengths = _batch(model, [5], [30])
    record = clip_emission_diagnostics(
        model, speech, anchors, speech_lengths, anchor_lengths
    )[0]
    deployed = record["deployed"]
    assert record["encoder_frames"] == 30
    assert record["phone_count"] == 5
    assert len(deployed["phone_best_frame_index"]) == 5
    assert len(deployed["phone_best_frame_llr"]) == 5
    assert len(deployed["phone_evidence"]) == 5
    indices = deployed["phone_best_frame_index"]
    assert all(0 <= index < 30 for index in indices)
    assert deployed["best_frame_span"] == max(indices) - min(indices)
    assert isinstance(deployed["best_frames_monotonic"], bool)


def test_probe_writes_one_record_per_clip_grouped_by_manifest_field(
    tmp_path, monkeypatch
):
    """End-to-end plumbing: a wrong field name here costs a whole GPU run."""

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("audio_path,keyword\n", encoding="utf-8")
    rows = [
        {
            "audio_path": "eva.wav",
            "keyword": "hey eva",
            "keyword_phonemes": "HH EY1",
            "label": 1,
            "variant": "hey_eva",
        },
        {
            "audio_path": "ava.wav",
            "keyword": "hey eva",
            "keyword_phonemes": "HH EY1",
            "label": 0,
            "variant": "hey_ava",
        },
        {
            "audio_path": "tooshort.wav",
            "keyword": "hey eva",
            "keyword_phonemes": "HH EY1",
            "label": 0,
            "variant": "hey_ava",
        },
    ]

    class FakeVerifier:
        fbank_extractor = object()
        fbank_kwargs: dict = {}
        min_fbank_frames = 1

        class _Alignment:
            @staticmethod
            def as_dict():
                return {"topology": "keyword_filler_segmental_crf_v1"}

        class _Stream:
            @staticmethod
            def describe():
                return "full-context"

        qbyt_alignment = _Alignment()
        stream_policy = _Stream()

        def emission_diagnostics(self, feats, keyword_ids_batch):
            assert len(feats) == len(keyword_ids_batch)
            assert all(ids == [1, 2] for ids in keyword_ids_batch)
            return [
                {
                    "encoder_frames": 20,
                    "phone_count": 2,
                    "masked_query_mass_mean": 0.4,
                    "masked_query_mass_max": 0.6,
                    "deployed": {
                        "logit": -1.5,
                        "has_legal_path": True,
                        "weakest_phone_evidence": 0.2,
                        "phone_top_frames_llr_mean": -0.3,
                        "best_frame_span": 12,
                        "best_frames_monotonic": True,
                    },
                }
                for _ in feats
            ]

    class FakeRunner:
        verifier = FakeVerifier()
        sample_rate = 16000

        @staticmethod
        def resolve_keyword_phonemes(keyword, keyword_phonemes=None, *, field_name=""):
            return str(keyword_phonemes).split()

        @staticmethod
        def enroll_phonemes(phonemes):
            return [{"HH": 1, "EY1": 2}[phone] for phone in phonemes]

        @staticmethod
        def from_config(_config, _prep, _device):
            return FakeRunner()

    monkeypatch.setattr(
        probe_qbyt_emissions,
        "resolved_config",
        lambda _cfg: {
            "paths": {},
            "stage1": {},
            "stage2": {},
            "demo": {},
            "tokenizer": {},
        },
    )
    monkeypatch.setattr(probe_qbyt_emissions, "load_manifest", lambda _path: rows)
    monkeypatch.setattr(
        probe_qbyt_emissions, "resolve_accelerator", lambda _device: ("cpu", 1)
    )
    monkeypatch.setattr(probe_qbyt_emissions, "Stage2ClipRunner", FakeRunner)
    monkeypatch.setattr(
        probe_qbyt_emissions, "ClipFeatureDataset", _ProbeFakeDataset
    )

    cfg = OmegaConf.create(
        {
            "prep": {
                "manifest": str(manifest_path),
                "stage2_ckpt": str(manifest_path),
                "output_dir": str(tmp_path / "out"),
                "group_field": "variant",
                "batch_size": 2,
                "num_workers": 0,
            },
            "run": {"device": "cpu"},
        }
    )

    summary = probe_qbyt_emissions.run_probe(cfg)

    assert summary["num_clips"] == 2
    assert summary["num_skipped"] == 1
    assert summary["audio_padding_ms"] == {"left": 160, "right": 160}
    assert set(summary["by_group"]) == {"hey_eva", "hey_ava"}
    assert [spec["name"] for spec in summary["ablations"]] == [
        "deployed",
        "one_vs_rest_filler",
        "relaxed_bounds",
        "one_vs_rest_and_relaxed",
    ]

    written = [
        json.loads(line)
        for line in (tmp_path / "out" / "emission_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["group"] for record in written] == ["hey_eva", "hey_ava"]
    assert [record["label"] for record in written] == [1, 0]
    assert written[0]["deployed"]["logit"] == pytest.approx(-1.5)
    assert written[0]["masked_query_mass_mean"] == pytest.approx(0.4)


def test_probe_rejects_a_group_field_that_is_not_a_manifest_column():
    rows = [{"audio_path": "a.wav", "keyword": "hey eva", "variant": "hey_ava"}]
    assert probe_qbyt_emissions._resolve_group_field(rows, "variant") == "variant"
    assert probe_qbyt_emissions._resolve_group_field(rows, "") == ""
    with pytest.raises(SystemExit, match="not a manifest column"):
        probe_qbyt_emissions._resolve_group_field(rows, "varient")


def test_summary_groups_records_and_reports_illegal_path_rate():
    model = _model(span=4)
    speech, anchors, speech_lengths, anchor_lengths = _batch(model, [6, 2], [24, 24])
    records = clip_emission_diagnostics(
        model, speech, anchors, speech_lengths, anchor_lengths
    )
    for record, group in zip(records, ("neg", "pos")):
        record["group"] = group
    summary = summarize_emission_diagnostics(records)
    assert set(summary) == {"neg", "pos"}
    assert summary["neg"]["num_clips"] == 1
    # The 6-phone query cannot fit the 4-frame span cap; the 2-phone one can.
    assert summary["neg"]["deployed"]["illegal_path_rate"] == 1.0
    assert summary["pos"]["deployed"]["illegal_path_rate"] == 0.0
    assert summary["pos"]["encoder_frames"]["p50"] == 24.0
