"""Stage II training batch collation aligned with main ``train_collate_fn``."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def train_collate_fn(batch: list[dict]) -> dict:
    """Collate LibriPhrase train samples into a padded batch."""
    feats = [item["feat"] for item in batch]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
    feat_lengths = torch.tensor([f.size(0) for f in feats])

    anchor = [item["anchor_seq"] for item in batch]
    anchor = pad_sequence(anchor, batch_first=True, padding_value=0)

    labels = torch.tensor([item["label"] for item in batch])

    seq_labels = [item["seq_label"] for item in batch]
    padded_seq_labels = pad_sequence(seq_labels, batch_first=True, padding_value=-1)
    seq_label_mask = (padded_seq_labels != -1).float()

    out = {
        "anchor": anchor,
        "feat": padded_feats,
        "feat_lengths": feat_lengths,
        "label": labels,
        "seq_label": padded_seq_labels,
        "seq_label_mask": seq_label_mask,
    }
    if "query_seq" in batch[0]:
        # Phoneme sequence of the clip that is actually in ``feat``. For negative
        # pairs that is a different phrase than ``anchor``, so the auxiliary CTC
        # loss must use this and not the anchor.
        query_seqs = [item["query_seq"] for item in batch]
        out["query_seq"] = pad_sequence(query_seqs, batch_first=True, padding_value=0)
        out["query_lengths"] = torch.tensor([seq.size(0) for seq in query_seqs])
    if "source" in batch[0]:
        out["source"] = torch.tensor([int(item["source"]) for item in batch])
    return out


def test_collate_fn(batch: list[dict]) -> dict:
    """Collate LibriPhrase eval samples into a padded batch (no seq labels)."""
    feats = [item["feat"] for item in batch]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
    feat_lengths = torch.tensor([f.size(0) for f in feats])

    anchor = [item["anchor_seq"] for item in batch]
    anchor = pad_sequence(anchor, batch_first=True, padding_value=0)

    labels = torch.tensor([item["label"] for item in batch])

    out = {
        "anchor": anchor,
        "feat": padded_feats,
        "feat_lengths": feat_lengths,
        "label": labels,
    }
    if "sample_id" in batch[0]:
        out["sample_id"] = torch.tensor(
            [int(item["sample_id"]) for item in batch],
            dtype=torch.long,
        )
    return out


test_collate_fn.__test__ = False  # not a pytest test; name matches main qbyt API
