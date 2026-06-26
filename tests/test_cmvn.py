import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from dma_kws.cmvn import GlobalCMVN, build_global_cmvn, load_json_cmvn_stats


def test_load_json_cmvn_stats_computes_mean_and_istd(tmp_path):
    cmvn_file = tmp_path / "global_cmvn"
    frame_num = 4.0
    mean_stat = [4.0, 8.0]
    var_stat = [20.0, 68.0]
    cmvn_file.write_text(
        json.dumps(
            {
                "mean_stat": mean_stat,
                "var_stat": var_stat,
                "frame_num": frame_num,
            }
        ),
        encoding="utf-8",
    )

    mean, istd = load_json_cmvn_stats(cmvn_file)

    expected_mean = torch.tensor([1.0, 2.0], dtype=torch.float32)
    expected_var = torch.tensor([4.0, 13.0], dtype=torch.float32)
    expected_istd = 1.0 / torch.sqrt(expected_var)

    assert torch.allclose(mean, expected_mean)
    assert torch.allclose(istd, expected_istd)


def test_global_cmvn_forward_applies_normalization():
    mean = torch.tensor([1.0, 2.0], dtype=torch.float32)
    istd = torch.tensor([0.5, 0.25], dtype=torch.float32)
    module = GlobalCMVN(mean, istd)

    x = torch.tensor(
        [
            [[3.0, 6.0], [5.0, 10.0]],
            [[1.0, 2.0], [9.0, 18.0]],
        ],
        dtype=torch.float32,
    )
    y = module(x)

    expected = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0]],
            [[0.0, 0.0], [4.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(y, expected)


def test_build_global_cmvn_returns_none_when_disabled():
    assert build_global_cmvn({}) is None
    assert build_global_cmvn({"cmvn": "none"}) is None


def test_build_global_cmvn_loads_from_stage1_config(tmp_path):
    cmvn_file = tmp_path / "global_cmvn"
    cmvn_file.write_text(
        json.dumps(
            {
                "mean_stat": [2.0],
                "var_stat": [6.0],
                "frame_num": 2.0,
            }
        ),
        encoding="utf-8",
    )

    module = build_global_cmvn(
        {
            "cmvn": "global_cmvn",
            "cmvn_conf": {
                "cmvn_file": str(cmvn_file),
                "is_json_cmvn": True,
            },
        }
    )

    assert isinstance(module, GlobalCMVN)
    assert module.mean.item() == pytest.approx(1.0)
    assert module.istd.item() == pytest.approx(1.0 / (2.0**0.5))


def test_build_global_cmvn_requires_cmvn_file():
    with pytest.raises(ValueError, match="cmvn_file is required"):
        build_global_cmvn({"cmvn": "global_cmvn", "cmvn_conf": {}})


def test_build_encoder_passes_cmvn_and_wenet_options(tmp_path):
    pytest.importorskip("torch")

    cmvn_file = tmp_path / "global_cmvn"
    cmvn_file.write_text(
        json.dumps(
            {
                "mean_stat": [2.0] * 80,
                "var_stat": [6.0] * 80,
                "frame_num": 2.0,
            }
        ),
        encoding="utf-8",
    )

    stage1_cfg = {
        "input_dim": 80,
        "encoder_output_dim": 144,
        "attention_heads": 4,
        "linear_units": 576,
        "num_blocks": 2,
        "causal": True,
        "cnn_module_norm": "layer_norm",
        "use_dynamic_chunk": True,
        "use_dynamic_left_chunk": True,
        "gradient_checkpointing": True,
        "cmvn": "global_cmvn",
        "cmvn_conf": {
            "cmvn_file": str(cmvn_file),
            "is_json_cmvn": True,
        },
    }

    fake_encoder = MagicMock(return_value=MagicMock())
    qbyt_root = Path(__file__).resolve().parents[1] / "qbyt"
    sys.path.insert(0, str(qbyt_root))
    with patch("models.encoder.ConformerEncoder", fake_encoder):
        from dma_kws.nn import build_encoder

        build_encoder(stage1_cfg, output_dim=144)

    assert fake_encoder.call_count == 1
    kwargs = fake_encoder.call_args.kwargs
    assert kwargs["causal"] is True
    assert kwargs["cnn_module_norm"] == "layer_norm"
    assert kwargs["use_dynamic_chunk"] is True
    assert kwargs["use_dynamic_left_chunk"] is True
    assert kwargs["gradient_checkpointing"] is True
    assert isinstance(kwargs["global_cmvn"], GlobalCMVN)
    assert kwargs["global_cmvn"].mean.shape == (80,)
