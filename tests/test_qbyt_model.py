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


def _model(seed: int = 0):
    QbyT = load_qbyt_class()
    torch.manual_seed(seed)
    model = QbyT(
        encoder_output_size=ENCODER_DIM,
        num_embeds=73,
        embed_dim=EMBED_DIM,
        post_num_layers=2,
    )
    return model.eval()


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
