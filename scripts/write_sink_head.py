#!/usr/bin/env python3
"""Write the post-hoc fitted sink head back into a QbyT checkpoint.

The analysis JSON (scripts/analyze_v41_offline.py) fits a logistic sink head on
half the hard split and a 2-feature combo [standardized pooled, fitted sink
logit]. This script maps those coefficients back into the model's additive
sink readout:

    sink_fc.weight = sink_coef / sink_scale          (elementwise)
    sink_fc.bias   = sink_intercept - sum(sink_coef * sink_mean / sink_scale)
    sink_alpha     = combo_coef[1] * pooled_scale / combo_coef[0]

so the deployed score (pooled + alpha * sink_logit) becomes affine-equivalent
to the fitted combo. The trunk and text head stay untouched (frozen-trunk
second stage). Provenance is recorded in the output checkpoint.

Usage:
  .venv/bin/python scripts/write_sink_head.py CHECKPOINT.pt ANALYSIS.json OUT.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> None:
    checkpoint_path = Path(sys.argv[1])
    analysis_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    probe = json.loads(analysis_path.read_text())["sink_probe"]
    state = data["model_state_dict"]

    coef = torch.tensor(probe["sink_coef"], dtype=torch.float32)
    intercept = float(probe["sink_intercept"])
    scale = torch.tensor(probe["sink_scale"], dtype=torch.float32)
    mean = torch.tensor(probe["sink_mean"], dtype=torch.float32)
    w1, w2 = probe["combo_coef"]
    if w1 <= 0:
        raise ValueError(f"combo pooled coefficient must be positive; got {w1}")

    weight = coef / scale
    bias = intercept - float((coef * mean / scale).sum())
    alpha = w2 * float(probe["pooled_scale"]) / w1

    state["qbyt.sink_fc.weight"] = weight.reshape(1, -1).contiguous()
    state["qbyt.sink_fc.bias"] = torch.tensor([bias], dtype=torch.float32)
    state["qbyt.sink_alpha"] = torch.tensor([alpha], dtype=torch.float32)

    data["sink_head_source"] = str(analysis_path)
    data["sink_head_fit"] = "posthoc logistic on half of the hard split"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print(
        f"wrote {out_path}: alpha={alpha:.4f} weight_norm={weight.norm():.4f} "
        f"bias={bias:+.4f}"
    )


if __name__ == "__main__":
    main()
