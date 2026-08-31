from __future__ import annotations

import math
import pickle

import pytest

from dma_kws.inference.audio_aug import AudioAugWaveformTransform
from dma_kws.inference.musan_mix import MusanWaveformMixer
from dma_kws.inference.stage2_clip import (
    ClipFeatureDataset,
    Stage2ClipRunner,
    collate_clip_feature_batch,
)
from dma_kws.inference.waveform_augmentation import WaveformAugmentationPipeline
from dma_kws.tokenizer import load_char_tokenizer


def _fake_phonemes(text: str) -> list[str]:
    """Mimic g2p_en: upper-case ARPAbet with stress digits on vowels."""
    phones = {"hey eva": ["HH", "EY1", "IY1", "V", "AH0"]}
    return phones.get(text.lower(), text.upper().split())


class _PipelineAudioAug:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.clean_seen = None

    def apply_pre_mix(self, index, waveform, sample_rate):
        del index, sample_rate
        return waveform * 2.0

    def apply_additive_delta(self, index, clean, sample_rate):
        del index, sample_rate
        self.clean_seen = clean.clone()
        return clean * 3.0

    def apply_post_mix(self, index, waveform, sample_rate):
        del index, sample_rate
        return waveform + 1.0

    def summary(self):
        return {"enabled": self.enabled, "kind": "test"}

    def recipe_metadata(self, index):
        return {"row_index": index, "kind": "test"}


class _PipelineMusanMixer:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.clean_seen = None

    def __call__(self, index, clean, sample_rate):
        del index, sample_rate
        self.clean_seen = clean.clone()
        return clean + clean * 4.0

    def summary(self):
        return {"enabled": self.enabled, "kind": "musan"}

    def recipe_metadata(self, index):
        return {"row_index": index, "kind": "musan"}


class _PhasedPipelineMusanMixer(_PipelineMusanMixer):
    def apply_pre_mix(self, index, waveform, sample_rate):
        del index, sample_rate
        return waveform * 5.0

    def apply_additive_delta(self, index, clean, sample_rate):
        del index, sample_rate
        self.clean_seen = clean.clone()
        return clean * 4.0

    def __call__(self, index, clean, sample_rate):  # pragma: no cover
        del index, clean, sample_rate
        raise AssertionError("combined pipeline must use the MUSAN phase API")


class _VariableLengthAudioAug(_PipelineAudioAug):
    def apply_pre_mix(self, index, waveform, sample_rate):
        del index, sample_rate
        return waveform[:, ::2]

    def apply_additive_delta(self, index, clean, sample_rate):
        del index, sample_rate
        self.clean_seen = clean.clone()
        return clean * 0.0

    def apply_post_mix(self, index, waveform, sample_rate):
        del index, sample_rate
        return waveform


def test_clip_feature_loading_api_is_public():
    assert ClipFeatureDataset.__name__ == "ClipFeatureDataset"
    batch = [(0, "feature", 1.0)]
    assert collate_clip_feature_batch(batch) is batch


def test_clip_feature_dataset_padding_precedes_min_frame_guard(monkeypatch):
    torch = pytest.importorskip("torch")
    source = torch.tensor([[0.25, -0.5, 0.75]])
    captured = {}

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )

    def fake_has_min_fbank_frames(num_samples, **_kwargs):
        captured["num_samples"] = num_samples
        return True

    monkeypatch.setattr(
        "dma_kws.inference.audio_utils.has_min_fbank_frames",
        fake_has_min_fbank_frames,
    )

    def fake_waveform_to_fbank(waveform, *, sample_rate, **_kwargs):
        captured["waveform"] = waveform.clone()
        captured["sample_rate"] = sample_rate
        return torch.ones(1, 80)

    monkeypatch.setattr(
        "dma_kws.stage2.features.waveform_to_fbank",
        fake_waveform_to_fbank,
    )

    class ResamplingExtractor:
        @staticmethod
        def prepare_waveform(waveform, sample_rate):
            return waveform.repeat_interleave(2, dim=1), sample_rate * 2

    dataset = ClipFeatureDataset(
        audio_paths=["clip.wav"],
        sample_rate=1000,
        fbank_extractor=ResamplingExtractor(),
        fbank_kwargs={
            "frame_length": 25,
            "frame_shift": 10,
            "snip_edges": True,
        },
        min_fbank_frames=1,
        left_padding_ms=2,
        right_padding_ms=3,
    )

    index, feat, end_sec = dataset[0]

    assert index == 0
    assert feat.shape == (1, 80)
    assert end_sec == pytest.approx(0.003)
    assert captured["num_samples"] == 16
    assert captured["sample_rate"] == 2000
    expected = torch.tensor(
        [[0.0] * 4 + [0.25, 0.25, -0.5, -0.5, 0.75, 0.75] + [0.0] * 6]
    )
    assert torch.equal(captured["waveform"], expected)
    assert torch.equal(source, torch.tensor([[0.25, -0.5, 0.75]]))


def test_clip_feature_dataset_transforms_prepared_audio_before_padding(monkeypatch):
    torch = pytest.importorskip("torch")
    source = torch.tensor([[1.0, 2.0]])
    captured = {}

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )
    monkeypatch.setattr(
        "dma_kws.inference.audio_utils.has_min_fbank_frames",
        lambda *_args, **_kwargs: True,
    )

    def fake_waveform_to_fbank(waveform, *, sample_rate, **_kwargs):
        captured["fbank_waveform"] = waveform.clone()
        captured["fbank_sample_rate"] = sample_rate
        return torch.ones(1, 80)

    monkeypatch.setattr(
        "dma_kws.stage2.features.waveform_to_fbank",
        fake_waveform_to_fbank,
    )

    class PreparingExtractor:
        @staticmethod
        def prepare_waveform(waveform, sample_rate):
            return waveform + 1.0, sample_rate * 2

    def transform(index, waveform, sample_rate):
        captured["transform"] = (index, waveform.clone(), sample_rate)
        return waveform * 10.0

    dataset = ClipFeatureDataset(
        audio_paths=["clip.wav"],
        sample_rate=1000,
        fbank_extractor=PreparingExtractor(),
        fbank_kwargs={
            "frame_length": 25,
            "frame_shift": 10,
            "snip_edges": True,
        },
        min_fbank_frames=1,
        left_padding_ms=1,
        right_padding_ms=1,
        waveform_transform=transform,
    )

    index, feat, end_sec = dataset[0]

    assert index == 0
    assert feat.shape == (1, 80)
    assert end_sec == pytest.approx(0.002)
    transform_index, transform_waveform, transform_sample_rate = captured["transform"]
    assert transform_index == 0
    assert transform_sample_rate == 2000
    assert torch.equal(transform_waveform, torch.tensor([[2.0, 3.0]]))
    assert captured["fbank_sample_rate"] == 2000
    assert torch.equal(
        captured["fbank_waveform"],
        torch.tensor([[0.0, 0.0, 20.0, 30.0, 0.0, 0.0]]),
    )


def test_clip_feature_dataset_observes_transformed_waveform_before_padding(monkeypatch):
    torch = pytest.importorskip("torch")
    source = torch.tensor([[1.0, 2.0]])
    captured = {"events": []}

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )
    monkeypatch.setattr(
        "dma_kws.inference.audio_utils.has_min_fbank_frames",
        lambda *_args, **_kwargs: True,
    )

    def fake_waveform_to_fbank(waveform, *, sample_rate, **_kwargs):
        captured["events"].append("fbank")
        captured["fbank"] = (waveform.clone(), sample_rate)
        return torch.ones(1, 80)

    monkeypatch.setattr(
        "dma_kws.stage2.features.waveform_to_fbank",
        fake_waveform_to_fbank,
    )

    class PreparingExtractor:
        @staticmethod
        def prepare_waveform(waveform, sample_rate):
            return waveform + 1.0, sample_rate * 2

    def transform(index, waveform, sample_rate):
        assert (index, sample_rate) == (0, 2000)
        return waveform * 10.0

    def observe(index, waveform, sample_rate):
        captured["events"].append("observer")
        captured["observed"] = (index, waveform.clone(), sample_rate)

    dataset = ClipFeatureDataset(
        audio_paths=["clip.wav"],
        sample_rate=1000,
        fbank_extractor=PreparingExtractor(),
        fbank_kwargs={
            "frame_length": 25,
            "frame_shift": 10,
            "snip_edges": True,
        },
        min_fbank_frames=1,
        left_padding_ms=1,
        right_padding_ms=1,
        waveform_transform=transform,
        waveform_observer=observe,
    )

    dataset[0]

    observed_index, observed_waveform, observed_rate = captured["observed"]
    assert observed_index == 0
    assert observed_rate == 2000
    assert torch.equal(observed_waveform, torch.tensor([[20.0, 30.0]]))
    fbank_waveform, fbank_rate = captured["fbank"]
    assert fbank_rate == 2000
    assert torch.equal(
        fbank_waveform,
        torch.tensor([[0.0, 0.0, 20.0, 30.0, 0.0, 0.0]]),
    )
    assert captured["events"] == ["observer", "fbank"]


def test_clip_feature_dataset_observer_runs_without_transform(monkeypatch):
    torch = pytest.importorskip("torch")
    source = torch.tensor([[0.25, -0.5]])
    observed = []

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )
    monkeypatch.setattr(
        "dma_kws.inference.audio_utils.has_min_fbank_frames",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "dma_kws.stage2.features.waveform_to_fbank",
        lambda *_args, **_kwargs: torch.ones(1, 80),
    )

    class PassthroughExtractor:
        @staticmethod
        def prepare_waveform(waveform, sample_rate):
            return waveform, sample_rate

    dataset = ClipFeatureDataset(
        audio_paths=["clean.wav"],
        sample_rate=16000,
        fbank_extractor=PassthroughExtractor(),
        fbank_kwargs={
            "frame_length": 25,
            "frame_shift": 10,
            "snip_edges": True,
        },
        min_fbank_frames=1,
        waveform_observer=lambda index, waveform, sample_rate: observed.append(
            (index, waveform.clone(), sample_rate)
        ),
    )

    dataset[0]

    assert len(observed) == 1
    assert observed[0][0] == 0
    assert torch.equal(observed[0][1], source)
    assert observed[0][2] == 16000


def test_clip_feature_dataset_allows_variable_length_transform_and_reports_duration(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    source = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    captured = {}

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )
    monkeypatch.setattr(
        "dma_kws.inference.audio_utils.has_min_fbank_frames",
        lambda *_args, **_kwargs: True,
    )

    def fake_waveform_to_fbank(waveform, *, sample_rate, **_kwargs):
        captured["waveform"] = waveform.clone()
        captured["sample_rate"] = sample_rate
        return torch.ones(1, 80)

    monkeypatch.setattr(
        "dma_kws.stage2.features.waveform_to_fbank",
        fake_waveform_to_fbank,
    )

    class PassthroughExtractor:
        @staticmethod
        def prepare_waveform(waveform, sample_rate):
            return waveform, sample_rate

    def speed_transform(index, waveform, sample_rate):
        assert (index, sample_rate) == (0, 1000)
        return waveform[:, ::2]

    dataset = ClipFeatureDataset(
        audio_paths=["clip.wav"],
        sample_rate=1000,
        fbank_extractor=PassthroughExtractor(),
        fbank_kwargs={
            "frame_length": 25,
            "frame_shift": 10,
            "snip_edges": True,
        },
        min_fbank_frames=1,
        left_padding_ms=1,
        right_padding_ms=1,
        waveform_transform=speed_transform,
        include_augmented_duration=True,
    )

    index, feat, end_sec, augmented_duration_sec = dataset[0]

    assert index == 0
    assert feat.shape == (1, 80)
    assert end_sec == pytest.approx(0.004)
    assert augmented_duration_sec == pytest.approx(0.002)
    assert captured["sample_rate"] == 1000
    assert torch.equal(captured["waveform"], torch.tensor([[0.0, 1.0, 3.0, 0.0]]))


@pytest.mark.parametrize(
    "transform",
    [
        lambda _index, waveform, _sample_rate: waveform.squeeze(0),
        lambda _index, waveform, _sample_rate: waveform.repeat(2, 1),
        lambda _index, waveform, _sample_rate: waveform[:, :0],
    ],
)
def test_clip_feature_dataset_rejects_non_mono_or_non_2d_transform(
    monkeypatch,
    transform,
):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (torch.ones(1, 10), sample_rate),
    )

    class PassthroughExtractor:
        @staticmethod
        def prepare_waveform(waveform, sample_rate):
            return waveform, sample_rate

    dataset = ClipFeatureDataset(
        audio_paths=["clip.wav"],
        sample_rate=16000,
        fbank_extractor=PassthroughExtractor(),
        fbank_kwargs={},
        min_fbank_frames=1,
        waveform_transform=transform,
    )

    with pytest.raises(ValueError, match="mono 2-D waveform"):
        dataset[0]


def test_waveform_augmentation_pipeline_orders_phases_and_shares_clean_reference():
    torch = pytest.importorskip("torch")
    audio_aug = _PipelineAudioAug()
    musan = _PipelineMusanMixer()
    pipeline = WaveformAugmentationPipeline(
        audio_aug=audio_aug,
        musan_mixer=musan,
    )
    source = torch.tensor([[1.0, 2.0]])

    output = pipeline(3, source, 16000)

    expected_clean = source * 2.0
    assert torch.equal(audio_aug.clean_seen, expected_clean)
    assert torch.equal(musan.clean_seen, expected_clean)
    assert torch.equal(output, expected_clean * 8.0 + 1.0)
    assert torch.equal(source, torch.tensor([[1.0, 2.0]]))


def test_waveform_augmentation_pipeline_applies_musan_pre_mix_before_all_deltas():
    torch = pytest.importorskip("torch")
    audio_aug = _PipelineAudioAug()
    musan = _PhasedPipelineMusanMixer()
    pipeline = WaveformAugmentationPipeline(
        audio_aug=audio_aug,
        musan_mixer=musan,
    )
    source = torch.tensor([[1.0, 2.0]])

    output = pipeline(3, source, 16000)

    expected_clean = source * 2.0 * 5.0
    assert torch.equal(audio_aug.clean_seen, expected_clean)
    assert torch.equal(musan.clean_seen, expected_clean)
    assert torch.equal(output, expected_clean * 8.0 + 1.0)


def test_waveform_augmentation_pipeline_rejects_partial_musan_phase_api():
    torch = pytest.importorskip("torch")

    class PartialMusan(_PipelineMusanMixer):
        def apply_pre_mix(self, index, waveform, sample_rate):
            del index, sample_rate
            return waveform

    pipeline = WaveformAugmentationPipeline(
        audio_aug=_PipelineAudioAug(),
        musan_mixer=PartialMusan(),
    )

    with pytest.raises(TypeError, match="must define both"):
        pipeline(0, torch.ones(1, 4), 16000)


def test_real_pipeline_additive_branches_use_volume_varied_clean_rms():
    torch = pytest.importorskip("torch")
    audio_paths = ["clean.wav"]
    audio_aug = AudioAugWaveformTransform.from_prep(
        {
            "audio_aug": {
                "seed": 13,
                "pcm_policy": "float_unclipped",
                "transforms": {
                    "noise_mix": {
                        "enabled": True,
                        "snr_db": 14.0,
                    }
                },
            }
        },
        audio_paths=audio_paths,
    )
    musan = MusanWaveformMixer.from_prep(
        {
            "musan_mix": {
                "seed": 17,
                "stationary_noise": {
                    "enabled": True,
                    "kind": "white_gaussian",
                    "snr_db": 11.0,
                },
                "volume_variation": {
                    "enabled": True,
                    "low_gain_db": -8.0,
                    "high_gain_db": 4.0,
                    "segment_ms_min": 50.0,
                    "segment_ms_max": 50.0,
                    "transition_ms": 10.0,
                },
            }
        },
        audio_paths=audio_paths,
    )
    pipeline = WaveformAugmentationPipeline(
        audio_aug=audio_aug,
        musan_mixer=musan,
    )
    clean = torch.sin(torch.linspace(0.0, 40.0, 4096)).unsqueeze(0) * 0.4

    varied = musan.apply_pre_mix(0, clean, 16000)
    audio_delta = audio_aug.apply_additive_delta(0, varied, 16000)
    musan_delta = musan.apply_additive_delta(0, varied, 16000)
    output = pipeline(0, clean, 16000)

    clean_rms = float(varied.double().square().mean().sqrt())
    audio_rms = float(audio_delta.double().square().mean().sqrt())
    musan_rms = float(musan_delta.double().square().mean().sqrt())
    assert 20.0 * math.log10(clean_rms / audio_rms) == pytest.approx(
        14.0,
        abs=1.0e-5,
    )
    assert 20.0 * math.log10(clean_rms / musan_rms) == pytest.approx(
        11.0,
        abs=1.0e-5,
    )
    assert torch.allclose(
        output,
        varied + audio_delta + musan_delta,
        atol=1.0e-7,
    )


def test_waveform_augmentation_pipeline_preserves_direct_musan_only_result():
    torch = pytest.importorskip("torch")

    class ExactMusan:
        enabled = True

        def __init__(self):
            self.output = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)

        def __call__(self, index, waveform, sample_rate):
            del index, waveform, sample_rate
            return self.output

    musan = ExactMusan()
    pipeline = WaveformAugmentationPipeline(musan_mixer=musan)

    output = pipeline(0, torch.ones(1, 3), 16000)

    assert output is musan.output
    assert pipeline.changes_duration is False


def test_waveform_augmentation_pipeline_allows_pre_mix_length_change():
    torch = pytest.importorskip("torch")
    audio_aug = _VariableLengthAudioAug()
    musan = _PipelineMusanMixer()
    pipeline = WaveformAugmentationPipeline(
        audio_aug=audio_aug,
        musan_mixer=musan,
    )
    source = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    output = pipeline(0, source, 16000)

    expected_clean = torch.tensor([[1.0, 3.0]])
    assert torch.equal(audio_aug.clean_seen, expected_clean)
    assert torch.equal(musan.clean_seen, expected_clean)
    assert torch.equal(output, expected_clean * 5.0)


def test_waveform_augmentation_pipeline_disabled_is_identity_and_omits_metadata():
    torch = pytest.importorskip("torch")
    audio_aug = _PipelineAudioAug(enabled=False)
    musan = _PipelineMusanMixer(enabled=False)
    pipeline = WaveformAugmentationPipeline(
        audio_aug=audio_aug,
        musan_mixer=musan,
    )
    source = torch.tensor([[1.0, 2.0]])

    assert pipeline.enabled is False
    assert pipeline(0, source, 16000) is source
    assert pipeline.summary() == {
        "audio_aug": {"enabled": False, "kind": "test"},
        "musan_mix": {"enabled": False, "kind": "musan"},
    }
    assert pipeline.recipe_metadata(0) == {}


def test_waveform_augmentation_pipeline_metadata_and_pickle_roundtrip():
    torch = pytest.importorskip("torch")
    pipeline = WaveformAugmentationPipeline(
        audio_aug=_PipelineAudioAug(),
        musan_mixer=_PipelineMusanMixer(),
    )

    restored = pickle.loads(pickle.dumps(pipeline))
    output = restored(2, torch.tensor([[1.0, 2.0]]), 16000)

    assert torch.equal(output, torch.tensor([[17.0, 33.0]]))
    assert restored.summary() == {
        "audio_aug": {"enabled": True, "kind": "test"},
        "musan_mix": {"enabled": True, "kind": "musan"},
    }
    assert restored.recipe_metadata(2) == {
        "audio_aug": {"row_index": 2, "kind": "test"},
        "musan_mix": {"row_index": 2, "kind": "musan"},
    }


def test_clip_feature_dataset_padding_can_satisfy_encoder_length_guard(monkeypatch):
    torch = pytest.importorskip("torch")
    source = torch.ones(1, 10)
    captured = {}

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )

    def fake_waveform_to_fbank(waveform, *, sample_rate, **_kwargs):
        captured["num_samples"] = waveform.size(1)
        captured["sample_rate"] = sample_rate
        return torch.ones(2, 80)

    monkeypatch.setattr(
        "dma_kws.stage2.features.waveform_to_fbank",
        fake_waveform_to_fbank,
    )

    class PassthroughExtractor:
        @staticmethod
        def prepare_waveform(waveform, sample_rate):
            return waveform, sample_rate

    dataset = ClipFeatureDataset(
        audio_paths=["short.wav"],
        sample_rate=1000,
        fbank_extractor=PassthroughExtractor(),
        fbank_kwargs={
            "frame_length": 25,
            "frame_shift": 10,
            "snip_edges": True,
        },
        min_fbank_frames=2,
        left_padding_ms=20,
        right_padding_ms=20,
    )

    index, feat, end_sec = dataset[0]

    assert index == 0
    assert feat.shape == (2, 80)
    assert end_sec == pytest.approx(0.01)
    assert captured == {"num_samples": 50, "sample_rate": 1000}


@pytest.mark.parametrize("left_padding_ms,right_padding_ms", [(-1, 0), (0, -1)])
def test_clip_feature_dataset_rejects_negative_padding(
    left_padding_ms, right_padding_ms
):
    with pytest.raises(ValueError, match="must be >= 0"):
        ClipFeatureDataset(
            audio_paths=[],
            sample_rate=16000,
            fbank_extractor=None,
            fbank_kwargs={},
            min_fbank_frames=1,
            left_padding_ms=left_padding_ms,
            right_padding_ms=right_padding_ms,
        )


def _fake_g2p():
    return _fake_phonemes


class FakeWaveform:
    def __init__(self, num_samples: int) -> None:
        self._num_samples = num_samples

    def size(self, dim: int) -> int:
        if dim == 1:
            return self._num_samples
        raise ValueError(f"Unsupported dim: {dim}")


def _fake_audio_loader(path: str, *, sample_rate: int):
    del path
    waveform = FakeWaveform(sample_rate * 2)
    return waveform, sample_rate


class FakeVerifier:
    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = [0.9] if scores is None else scores
        self.calls: list[tuple] = []

    def verify_candidates(self, waveform, sample_rate: int, keyword_ids: list[int], candidates):
        self.calls.append((waveform, sample_rate, keyword_ids, candidates))
        output = []
        for candidate, score in zip(candidates, self._scores):
            output.append(
                {
                    "start_sec": candidate.start_sec,
                    "end_sec": candidate.end_sec,
                    "stage1_score": candidate.stage1_score,
                    "qbyt_score": score,
                }
            )
        return output


def _build_runner(
    *,
    threshold: float,
    verifier: FakeVerifier,
    monkeypatch=None,
) -> Stage2ClipRunner:
    if monkeypatch is not None:
        monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
        monkeypatch.setattr(
            "dma_kws.inference.stage2_clip.text_to_phonemes",
            lambda _g2p, text: _fake_phonemes(text),
        )
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    return Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": threshold},
        sample_rate=16000,
    )


def test_clip_runner_scores_full_audio_span(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", _fake_audio_loader)
    verifier = FakeVerifier(scores=[0.85])
    runner = _build_runner(threshold=0.5, verifier=verifier, monkeypatch=monkeypatch)

    result = runner.run("/tmp/clip.wav", "hello")

    assert result["audio"] == "/tmp/clip.wav"
    assert result["clip_span_sec"] == {"start_sec": 0.0, "end_sec": 2.0}
    candidates = verifier.calls[0][3]
    assert len(candidates) == 1
    assert candidates[0].start_sec == 0.0
    assert candidates[0].end_sec == 2.0
    assert candidates[0].stage1_score == 0.0


def test_clip_runner_detected_above_threshold(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", _fake_audio_loader)
    runner = _build_runner(
        threshold=0.6,
        verifier=FakeVerifier(scores=[0.85]),
        monkeypatch=monkeypatch,
    )

    result = runner.run("/tmp/clip.wav", "hello")

    assert result["detected"] is True
    assert result["qbyt_score"] == 0.85
    assert result["skipped"] is False


def test_clip_runner_skipped_when_too_short(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", _fake_audio_loader)
    runner = _build_runner(
        threshold=0.5,
        verifier=FakeVerifier(scores=[]),
        monkeypatch=monkeypatch,
    )

    result = runner.run("/tmp/clip.wav", "hello")

    assert result["detected"] is False
    assert result["qbyt_score"] == 0.0
    assert result["skipped"] is True


class FakeBatchVerifier:
    def __init__(self, scores: list[float]) -> None:
        from dma_kws.stage2.fbank import FbankExtractor

        self._scores = list(scores)
        self.batches: list[int] = []
        self.keyword_ids_batches: list[list[list[int]]] = []
        self.min_fbank_frames = 7
        self.fbank_extractor = FbankExtractor(dither=0.0)
        self.fbank_kwargs = {
            "num_mel_bins": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "dither": 0.0,
            "window_type": "povey",
            "backend": "torchaudio_kaldi",
            "target_sample_rate": None,
            "snip_edges": True,
            "low_freq": 20.0,
            "high_freq": 0.0,
        }

    def score_clip_feats(self, feats, keyword_ids_batch):
        assert len(feats) == len(keyword_ids_batch)
        self.batches.append(len(feats))
        self.keyword_ids_batches.append(
            [list(keyword_ids) for keyword_ids in keyword_ids_batch]
        )
        scores = self._scores[: len(feats)]
        self._scores = self._scores[len(feats):]
        return scores


def test_clip_runner_run_batch(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    def loader(path: str, *, sample_rate: int):
        if path.endswith("short.wav"):
            return torch.zeros(1, 10), sample_rate
        return torch.zeros(1, sample_rate * 2), sample_rate

    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", loader)
    monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
    g2p_calls: list[str] = []

    def fake_text_to_phonemes(_g2p, text):
        g2p_calls.append(text)
        return _fake_phonemes(text)

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.text_to_phonemes", fake_text_to_phonemes
    )

    verifier = FakeBatchVerifier(scores=[0.9, 0.2])
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    runner = Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": 0.5},
        sample_rate=16000,
    )

    rows = [
        {"audio_path": "/tmp/a.wav", "keyword": "hello"},
        {"audio_path": "/tmp/short.wav", "keyword": "hello"},
        {"audio_path": "/tmp/b.wav", "keyword": "hello"},
    ]
    observed_indices = []
    results = runner.run_batch(
        rows,
        batch_size=8,
        num_workers=0,
        waveform_observer=(
            lambda index, _waveform, _sample_rate: observed_indices.append(index)
        ),
    )

    assert [record["skipped"] for record in results] == [False, True, False]
    assert results[0]["qbyt_score"] == 0.9
    assert results[0]["detected"] is True
    assert results[1]["qbyt_score"] == 0.0
    assert results[1]["detected"] is False
    assert results[2]["qbyt_score"] == 0.2
    assert results[2]["detected"] is False
    assert g2p_calls == ["hello"]
    assert verifier.batches == [2]
    assert observed_indices == [0, 1, 2]
    assert set(results[0]) == {
        "audio",
        "keyword",
        "keyword_phonemes",
        "clip_span_sec",
        "qbyt_score",
        "threshold",
        "detected",
        "skipped",
    }


def test_clip_runner_run_batch_reports_augmented_duration_and_preserves_source_span(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (
            torch.zeros(1, sample_rate * 2),
            sample_rate,
        ),
    )
    monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.text_to_phonemes",
        lambda _g2p, text: _fake_phonemes(text),
    )
    verifier = FakeBatchVerifier(scores=[0.9])
    runner = Stage2ClipRunner(
        verifier=verifier,
        tokenizer=load_char_tokenizer(
            "data/dict/lang_char.txt",
            split_with_space=" ",
        ),
        demo_cfg={"qbyt_threshold": 0.5},
        sample_rate=16000,
    )

    result = runner.run_batch(
        [{"audio_path": "/tmp/a.wav", "keyword": "hello"}],
        waveform_transform=lambda _index, waveform, _sample_rate: waveform[:, ::2],
        num_workers=0,
    )[0]

    assert result["clip_span_sec"] == {"start_sec": 0.0, "end_sec": 2.0}
    assert result["augmented_duration_sec"] == pytest.approx(1.0)


def test_clip_runner_run_batch_uses_per_row_keyword_phoneme_overrides(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (
            torch.zeros(1, sample_rate * 2),
            sample_rate,
        ),
    )
    monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
    g2p_calls: list[str] = []
    ee_vah = ["HH", "EY1", "IY1", "V", "AH0"]
    ay_vah = ["HH", "EY1", "EY1", "V", "AH0"]

    def fake_text_to_phonemes(_g2p, text):
        g2p_calls.append(text)
        assert text == "hey eva"
        return ay_vah

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.text_to_phonemes", fake_text_to_phonemes
    )

    verifier = FakeBatchVerifier(scores=[0.9, 0.2, 0.8])
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    runner = Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": 0.5},
        sample_rate=16000,
    )
    rows = [
        {
            "audio_path": "/tmp/ee-vah.wav",
            "keyword": "hey eva",
            "keyword_phonemes": "HH EY1 IY1 V AH0",
        },
        {
            "audio_path": "/tmp/ay-vah.wav",
            "keyword": "hey eva",
            "keyword_phonemes": ay_vah,
        },
        {"audio_path": "/tmp/default.wav", "keyword": "hey eva"},
    ]

    results = runner.run_batch(rows, batch_size=8, num_workers=0)

    assert [result["keyword_phonemes"] for result in results] == [
        ee_vah,
        ay_vah,
        ay_vah,
    ]
    assert g2p_calls == ["hey eva"]
    assert verifier.keyword_ids_batches == [
        [
            [tokenizer.symbol_table[phone] for phone in ee_vah],
            [tokenizer.symbol_table[phone] for phone in ay_vah],
            [tokenizer.symbol_table[phone] for phone in ay_vah],
        ]
    ]


@pytest.mark.parametrize(
    "override",
    ["", None, [], "HH NOT_A_PHONE"],
)
def test_clip_runner_rejects_invalid_keyword_phoneme_override(monkeypatch, override):
    runner = _build_runner(
        threshold=0.5,
        verifier=FakeVerifier(scores=[0.7]),
        monkeypatch=monkeypatch,
    )

    with pytest.raises(ValueError, match="Manifest row 1 keyword_phonemes"):
        runner.run_batch(
            [
                {
                    "audio_path": "/tmp/clip.wav",
                    "keyword": "hey eva",
                    "keyword_phonemes": override,
                }
            ]
        )


def test_clip_runner_result_record_shape(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", _fake_audio_loader)
    runner = _build_runner(
        threshold=0.5,
        verifier=FakeVerifier(scores=[0.7]),
        monkeypatch=monkeypatch,
    )

    result = runner.run("/tmp/clip.wav", "hello")

    assert set(result) == {
        "audio",
        "keyword",
        "keyword_phonemes",
        "clip_span_sec",
        "qbyt_score",
        "threshold",
        "detected",
        "skipped",
    }
