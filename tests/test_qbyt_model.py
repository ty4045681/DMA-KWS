"""QbyT scoring must not depend on how a batch happens to be padded.

QbyT concatenates the phoneme text and the encoder output along time, so both
blocks are padded to the batch maximum. Every invariant here exists because a
readout that is expressed in per-sample lengths, but applied to a batch-padded
layout, silently reads the wrong frame: a sample scored alone and the same
sample scored next to a longer keyword would disagree.
"""

import pytest
import torch

pytest.importorskip("torch")

from dma_kws.pathing import load_qbyt_class

EMBED_DIM = 64
ENCODER_DIM = 48
SHORT_TEXT_LEN = 8
LONG_TEXT_LEN = 45
AUDIO_LEN = 20
# The wrong readout is text_len + speech_len - 1. With a long companion anchor in
# the batch that index (27) lands before the audio block even starts (45), so the
# pooled GRU state has consumed no audio at all.
assert SHORT_TEXT_LEN + AUDIO_LEN - 1 < LONG_TEXT_LEN


def _model(
    seed: int = 0,
    *,
    readout_mode: str = "gru_last",
    readout_temperature: float = 1.0,
):
    QbyT = load_qbyt_class()
    torch.manual_seed(seed)
    model = QbyT(
        encoder_output_size=ENCODER_DIM,
        num_embeds=73,
        embed_dim=EMBED_DIM,
        post_num_layers=2,
        readout_mode=readout_mode,
        readout_temperature=readout_temperature,
    )
    return model.eval()


class _FixedPositionScorer(torch.nn.Module):
    """Ignore matcher states and emit deterministic per-position logits."""

    def __init__(self, values, *, dtype=torch.float32):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=dtype))

    def forward(self, states):
        if states.size(1) != self.values.numel():
            raise AssertionError("fixed scorer width does not match the text tensor")
        return self.values.view(1, -1, 1).expand(states.size(0), -1, -1)


def _sample(text_len: int, audio_len: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    text = torch.randint(3, 70, (text_len,), generator=generator)
    audio = torch.randn(audio_len, ENCODER_DIM, generator=generator)
    return text, audio


def _score(model, texts, audios, text_lengths, speech_lengths):
    """Pad ``texts``/``audios`` to their batch maxima and score them."""
    text_width = max(text.size(0) for text in texts)
    audio_width = max(audio.size(0) for audio in audios)
    text_batch = torch.zeros(len(texts), text_width, dtype=torch.long)
    audio_batch = torch.zeros(len(audios), audio_width, ENCODER_DIM)
    for row, (text, audio) in enumerate(zip(texts, audios)):
        text_batch[row, : text.size(0)] = text
        audio_batch[row, : audio.size(0)] = audio
    with torch.no_grad():
        logits, _ = model(
            audio_batch,
            text_batch,
            speech_lengths=torch.tensor(speech_lengths),
            text_lengths=torch.tensor(text_lengths),
        )
    return logits


def test_score_is_invariant_to_batch_companions():
    """The same pair must score the same next to a short and a long keyword.

    This is the failure that makes training and validation disagree: training
    shuffles all keyword lengths into one large batch, while the LibriPhrase eval
    CSVs are concatenated 1-word first and consumed with shuffle=False, so the
    batch text width is systematically different between the two.
    """
    model = _model()
    text, audio = _sample(SHORT_TEXT_LEN, AUDIO_LEN, seed=1)
    short_companion, short_audio = _sample(SHORT_TEXT_LEN + 1, AUDIO_LEN, seed=2)
    long_companion, long_audio = _sample(LONG_TEXT_LEN, AUDIO_LEN, seed=3)

    with_short = _score(
        model,
        [text, short_companion],
        [audio, short_audio],
        [SHORT_TEXT_LEN, SHORT_TEXT_LEN + 1],
        [AUDIO_LEN, AUDIO_LEN],
    )[0]
    with_long = _score(
        model,
        [text, long_companion],
        [audio, long_audio],
        [SHORT_TEXT_LEN, LONG_TEXT_LEN],
        [AUDIO_LEN, AUDIO_LEN],
    )[0]

    torch.testing.assert_close(with_short, with_long, atol=1e-5, rtol=1e-4)


def test_batched_score_matches_unpadded_single():
    """A batch=1 call with no padding at all is the ground truth.

    Pinning against it fixes the *meaning* of the readout -- the pooled state
    after consuming exactly [valid text][valid audio] -- rather than only
    checking that two padded batches happen to agree.
    """
    model = _model()
    text, audio = _sample(SHORT_TEXT_LEN, AUDIO_LEN, seed=1)
    long_companion, long_audio = _sample(LONG_TEXT_LEN, AUDIO_LEN * 2, seed=3)

    alone = _score(model, [text], [audio], [SHORT_TEXT_LEN], [AUDIO_LEN])[0]
    batched = _score(
        model,
        [text, long_companion],
        [audio, long_audio],
        [SHORT_TEXT_LEN, LONG_TEXT_LEN],
        [AUDIO_LEN, AUDIO_LEN * 2],
    )[0]

    torch.testing.assert_close(alone, batched, atol=1e-5, rtol=1e-4)


def test_audio_padding_does_not_leak():
    """Frames past speech_lengths must not reach the score."""
    model = _model()
    text, audio = _sample(SHORT_TEXT_LEN, AUDIO_LEN, seed=1)
    garbage = torch.randn(AUDIO_LEN * 2, ENCODER_DIM, generator=torch.Generator().manual_seed(9))
    padded_audio = torch.cat([audio, garbage], dim=0)

    clean = _score(model, [text], [audio], [SHORT_TEXT_LEN], [AUDIO_LEN])[0]
    noisy = _score(model, [text], [padded_audio], [SHORT_TEXT_LEN], [AUDIO_LEN])[0]

    torch.testing.assert_close(clean, noisy, atol=1e-5, rtol=1e-4)


def test_text_padding_does_not_leak():
    """Token ids past text_lengths must not reach the score."""
    model = _model()
    text, audio = _sample(SHORT_TEXT_LEN, AUDIO_LEN, seed=1)
    garbage = torch.randint(3, 70, (LONG_TEXT_LEN - SHORT_TEXT_LEN,), generator=torch.Generator().manual_seed(9))
    padded_text = torch.cat([text, garbage], dim=0)

    clean = _score(model, [text], [audio], [SHORT_TEXT_LEN], [AUDIO_LEN])[0]
    noisy = _score(model, [padded_text], [audio], [SHORT_TEXT_LEN], [AUDIO_LEN])[0]

    torch.testing.assert_close(clean, noisy, atol=1e-5, rtol=1e-4)


def test_seq_logits_cover_the_padded_anchor_width():
    """seq_logits must stay aligned with the collated seq_label width.

    ``build_seq_label`` returns one label per anchor token, so the Stage II
    sequence loss pads its targets to the same width as the anchor tensor and
    masks the rest. Narrowing this output would silently misalign the two.
    """
    model = _model()
    text, audio = _sample(SHORT_TEXT_LEN, AUDIO_LEN, seed=1)
    long_companion, long_audio = _sample(LONG_TEXT_LEN, AUDIO_LEN, seed=3)

    text_batch = torch.zeros(2, LONG_TEXT_LEN, dtype=torch.long)
    text_batch[0, :SHORT_TEXT_LEN] = text
    text_batch[1] = long_companion
    with torch.no_grad():
        _, seq_logits = model(
            torch.stack([audio, long_audio]),
            text_batch,
            speech_lengths=torch.tensor([AUDIO_LEN, AUDIO_LEN]),
            text_lengths=torch.tensor([SHORT_TEXT_LEN, LONG_TEXT_LEN]),
        )
    assert seq_logits.shape == (2, LONG_TEXT_LEN)


def test_valid_text_logits_are_invariant_to_batch_companions():
    """The supervised part of seq_logits must not move with the batch either."""
    model = _model()
    text, audio = _sample(SHORT_TEXT_LEN, AUDIO_LEN, seed=1)
    long_companion, long_audio = _sample(LONG_TEXT_LEN, AUDIO_LEN, seed=3)

    with torch.no_grad():
        _, alone = model(
            audio.unsqueeze(0),
            text.unsqueeze(0),
            speech_lengths=torch.tensor([AUDIO_LEN]),
            text_lengths=torch.tensor([SHORT_TEXT_LEN]),
        )
        text_batch = torch.zeros(2, LONG_TEXT_LEN, dtype=torch.long)
        text_batch[0, :SHORT_TEXT_LEN] = text
        text_batch[1] = long_companion
        _, batched = model(
            torch.stack([audio, long_audio]),
            text_batch,
            speech_lengths=torch.tensor([AUDIO_LEN, AUDIO_LEN]),
            text_lengths=torch.tensor([SHORT_TEXT_LEN, LONG_TEXT_LEN]),
        )

    torch.testing.assert_close(
        alone[0, :SHORT_TEXT_LEN],
        batched[0, :SHORT_TEXT_LEN],
        atol=1e-5,
        rtol=1e-4,
    )


def test_eps_score_is_masked_mean_of_valid_position_logits():
    model = _model(readout_mode="eps_mean")
    short_text, short_audio = _sample(SHORT_TEXT_LEN, AUDIO_LEN, seed=1)
    long_text, long_audio = _sample(LONG_TEXT_LEN, AUDIO_LEN, seed=3)
    text_batch = torch.zeros(2, LONG_TEXT_LEN, dtype=torch.long)
    text_batch[0, :SHORT_TEXT_LEN] = short_text
    text_batch[1] = long_text
    with torch.no_grad():
        logits, _, details = model.forward_with_readout_details(
            torch.stack([short_audio, long_audio]),
            text_batch,
            speech_lengths=torch.tensor([AUDIO_LEN, AUDIO_LEN]),
            text_lengths=torch.tensor([SHORT_TEXT_LEN, LONG_TEXT_LEN]),
        )

    position_logits = details.position_logits
    assert position_logits.shape == (2, LONG_TEXT_LEN)
    assert details.position_mask.dtype == torch.bool
    assert details.position_mask[0].sum() == SHORT_TEXT_LEN
    assert details.position_mask[1].sum() == LONG_TEXT_LEN
    torch.testing.assert_close(
        position_logits[0, SHORT_TEXT_LEN:],
        torch.zeros_like(position_logits[0, SHORT_TEXT_LEN:]),
    )
    expected = torch.stack(
        [
            position_logits[0][details.position_mask[0]].mean(),
            position_logits[1][details.position_mask[1]].mean(),
        ]
    )
    torch.testing.assert_close(logits, expected)


def test_eps_softmin_is_stable_normalized_log_mean_exp_of_valid_logits():
    model = _model(readout_mode="eps_softmin", readout_temperature=1.0)
    # The final value occupies text padding and is intentionally much smaller
    # than both valid values. A masking error would make it dominate soft-min.
    model.final_pos_fc = _FixedPositionScorer(
        [1.0, 3.0, -1000.0],
        dtype=torch.float16,
    )
    with torch.no_grad():
        logits, _, details = model.forward_with_readout_details(
            torch.randn(1, AUDIO_LEN, ENCODER_DIM),
            torch.tensor([[3, 4, 0]]),
            speech_lengths=torch.tensor([AUDIO_LEN]),
            text_lengths=torch.tensor([2]),
        )

    # -log((exp(-1) + exp(-3)) / 2)
    assert logits.dtype == torch.float32
    assert logits.item() == pytest.approx(1.5662191695169727, abs=1e-6)
    assert details.position_mask.tolist() == [[True, True, False]]
    torch.testing.assert_close(
        details.position_logits,
        torch.tensor([[1.0, 3.0, 0.0]], dtype=torch.float16),
    )


def test_eps_softmin_temperature_controls_mean_to_min_interpolation():
    text = torch.tensor([[3, 4]])
    audio = torch.randn(1, AUDIO_LEN, ENCODER_DIM)
    pooled = []
    for temperature in (100.0, 1.0, 0.5):
        model = _model(
            readout_mode="eps_softmin",
            readout_temperature=temperature,
        )
        model.final_pos_fc = _FixedPositionScorer([1.0, 3.0])
        with torch.no_grad():
            logits, _ = model(
                audio,
                text,
                speech_lengths=torch.tensor([AUDIO_LEN]),
                text_lengths=torch.tensor([2]),
            )
        pooled.append(logits.item())

    assert pooled[0] == pytest.approx(1.995000083331111, abs=1e-5)
    assert pooled[1] == pytest.approx(1.5662191695169727, abs=1e-6)
    assert pooled[2] == pytest.approx(1.3374986263210678, abs=1e-6)
    assert pooled[0] > pooled[1] > pooled[2] > 1.0


def test_eps_softmin_preserves_equal_and_single_valid_logits():
    model = _model(readout_mode="eps_softmin", readout_temperature=0.3)
    model.final_pos_fc = _FixedPositionScorer([5.0, 5.0])
    audio = torch.randn(2, AUDIO_LEN, ENCODER_DIM)
    with torch.no_grad():
        logits, _ = model(
            audio,
            torch.tensor([[3, 4], [3, 0]]),
            speech_lengths=torch.tensor([AUDIO_LEN, AUDIO_LEN]),
            text_lengths=torch.tensor([2, 1]),
        )

    torch.testing.assert_close(logits, torch.tensor([5.0, 5.0]))


def test_gru_readout_details_do_not_invent_eps_position_logits():
    model = _model(readout_mode="gru_last")
    text, audio = _sample(SHORT_TEXT_LEN, AUDIO_LEN, seed=1)

    with torch.no_grad():
        logits, seq_logits, details = model.forward_with_readout_details(
            audio.unsqueeze(0),
            text.unsqueeze(0),
            speech_lengths=torch.tensor([AUDIO_LEN]),
            text_lengths=torch.tensor([SHORT_TEXT_LEN]),
        )
        ordinary_logits, ordinary_seq_logits = model(
            audio.unsqueeze(0),
            text.unsqueeze(0),
            speech_lengths=torch.tensor([AUDIO_LEN]),
            text_lengths=torch.tensor([SHORT_TEXT_LEN]),
        )

    assert details.position_logits is None
    assert details.position_mask.tolist() == [[True] * SHORT_TEXT_LEN]
    torch.testing.assert_close(logits, ordinary_logits)
    torch.testing.assert_close(seq_logits, ordinary_seq_logits)


@pytest.mark.parametrize("readout_mode", ["eps_mean", "eps_softmin"])
def test_eps_score_is_invariant_to_batch_companions(readout_mode):
    model = _model(readout_mode=readout_mode)
    text, audio = _sample(SHORT_TEXT_LEN, AUDIO_LEN, seed=1)
    short_companion, short_audio = _sample(SHORT_TEXT_LEN + 1, AUDIO_LEN, seed=2)
    long_companion, long_audio = _sample(LONG_TEXT_LEN, AUDIO_LEN, seed=3)

    with_short = _score(
        model,
        [text, short_companion],
        [audio, short_audio],
        [SHORT_TEXT_LEN, SHORT_TEXT_LEN + 1],
        [AUDIO_LEN, AUDIO_LEN],
    )[0]
    with_long = _score(
        model,
        [text, long_companion],
        [audio, long_audio],
        [SHORT_TEXT_LEN, LONG_TEXT_LEN],
        [AUDIO_LEN, AUDIO_LEN],
    )[0]

    torch.testing.assert_close(with_short, with_long, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("readout_mode", ["eps_mean", "eps_softmin"])
def test_eps_readout_registers_only_its_active_final_head(readout_mode):
    model = _model(readout_mode=readout_mode)
    state_keys = set(model.state_dict())

    assert "final_pos_fc.weight" in state_keys
    assert "final_pos_fc.bias" in state_keys
    assert not any(key.startswith("gru.") for key in state_keys)
    assert not any(key.startswith("fc.") for key in state_keys)
    assert model.gru is None
    assert model.fc is None


@pytest.mark.parametrize("readout_mode", ["eps_mean", "eps_softmin"])
def test_eps_final_position_scorer_receives_utterance_gradient(readout_mode):
    model = _model(readout_mode=readout_mode).train()
    text, audio = _sample(SHORT_TEXT_LEN, AUDIO_LEN, seed=1)

    logits, _ = model(
        audio.unsqueeze(0),
        text.unsqueeze(0),
        speech_lengths=torch.tensor([AUDIO_LEN]),
        text_lengths=torch.tensor([SHORT_TEXT_LEN]),
    )
    logits.sum().backward()

    assert model.final_pos_fc.weight.grad is not None
    assert torch.isfinite(model.final_pos_fc.weight.grad).all()


@pytest.mark.parametrize("readout_mode", ["eps_mean", "eps_softmin"])
def test_eps_empty_anchor_has_finite_neutral_logit_without_device_sync(readout_mode):
    model = _model(readout_mode=readout_mode)
    with torch.no_grad():
        logits, _, details = model.forward_with_readout_details(
            torch.randn(1, AUDIO_LEN, ENCODER_DIM),
            torch.zeros(1, 1, dtype=torch.long),
            speech_lengths=torch.tensor([AUDIO_LEN]),
            text_lengths=torch.tensor([0]),
        )
    torch.testing.assert_close(logits, torch.zeros_like(logits))
    assert not bool(details.position_mask.any())
    torch.testing.assert_close(
        details.position_logits,
        torch.zeros_like(details.position_logits),
    )


def test_unknown_readout_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported QbyT readout mode"):
        _model(readout_mode="unknown")


@pytest.mark.parametrize(
    "temperature",
    [0.0, -1.0, float("nan"), float("inf"), True, "not-a-number"],
)
def test_invalid_readout_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="readout_temperature"):
        _model(readout_mode="eps_softmin", readout_temperature=temperature)
