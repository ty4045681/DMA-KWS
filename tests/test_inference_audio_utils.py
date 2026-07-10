from dma_kws.inference.audio_utils import (
    has_min_fbank_frames,
    min_samples_for_fbank_frames,
    num_fbank_frames,
)


def test_snip_edges_true_seven_frame_boundary():
    assert min_samples_for_fbank_frames(7, sample_rate=16000) == 1360
    assert num_fbank_frames(1359, sample_rate=16000) == 6
    assert num_fbank_frames(1360, sample_rate=16000) == 7
    assert not has_min_fbank_frames(1359, min_frames=7, sample_rate=16000)
    assert has_min_fbank_frames(1360, min_frames=7, sample_rate=16000)


def test_snip_edges_false_seven_frame_boundary():
    kwargs = {"sample_rate": 16000, "snip_edges": False}

    assert min_samples_for_fbank_frames(7, **kwargs) == 1040
    assert num_fbank_frames(1039, **kwargs) == 6
    assert num_fbank_frames(1040, **kwargs) == 7
    assert not has_min_fbank_frames(1039, min_frames=7, **kwargs)
    assert has_min_fbank_frames(1040, min_frames=7, **kwargs)


def test_snip_edges_false_one_second_has_100_frames():
    assert num_fbank_frames(16000, sample_rate=16000, snip_edges=False) == 100
