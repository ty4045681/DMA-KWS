import torch

from dma_kws.stage2.collate import train_collate_fn


def _sample(anchor_len: int, feat_len: int, label: int) -> dict:
    return {
        "anchor_seq": torch.arange(1, anchor_len + 1, dtype=torch.long),
        "feat": torch.ones(feat_len, 80),
        "label": torch.tensor(label, dtype=torch.long),
        "seq_label": torch.ones(anchor_len, dtype=torch.long),
    }


def test_train_collate_fn_pads_feats_and_anchors():
    batch = [_sample(anchor_len=3, feat_len=4, label=1), _sample(anchor_len=2, feat_len=6, label=0)]

    collated = train_collate_fn(batch)

    assert collated["feat"].shape == (2, 6, 80)
    assert torch.allclose(collated["feat"][0, 4:], torch.zeros(2, 80))
    assert torch.allclose(collated["feat"][1], torch.ones(6, 80))

    assert collated["anchor"].shape == (2, 3)
    assert torch.equal(collated["anchor"][0], torch.tensor([1, 2, 3]))
    assert torch.equal(collated["anchor"][1], torch.tensor([1, 2, 0]))

    assert torch.equal(collated["feat_lengths"], torch.tensor([4, 6]))
    assert torch.equal(collated["label"], torch.tensor([1, 0]))


def test_train_collate_fn_builds_seq_label_mask():
    batch = [
        {
            "anchor_seq": torch.tensor([1, 2, 3], dtype=torch.long),
            "feat": torch.zeros(2, 80),
            "label": torch.tensor(1, dtype=torch.long),
            "seq_label": torch.tensor([1, 0, 1], dtype=torch.long),
        },
        {
            "anchor_seq": torch.tensor([4, 5], dtype=torch.long),
            "feat": torch.zeros(3, 80),
            "label": torch.tensor(0, dtype=torch.long),
            "seq_label": torch.tensor([0, 1], dtype=torch.long),
        },
    ]

    collated = train_collate_fn(batch)

    assert collated["seq_label"].shape == (2, 3)
    assert torch.equal(collated["seq_label"][0], torch.tensor([1, 0, 1]))
    assert torch.equal(collated["seq_label"][1], torch.tensor([0, 1, -1]))
    assert torch.equal(collated["seq_label_mask"][0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(collated["seq_label_mask"][1], torch.tensor([1.0, 1.0, 0.0]))
