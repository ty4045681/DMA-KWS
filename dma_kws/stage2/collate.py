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

    return {
        "anchor": anchor,
        "feat": padded_feats,
        "feat_lengths": feat_lengths,
        "label": labels,
    }


test_collate_fn.__test__ = False  # not a pytest test; name matches main qbyt API
