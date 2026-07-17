import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dump_zipformer_encoder_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("dump_zipformer_encoder_calibration", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_writer_downcasts_int64_and_uint64_for_mindspore_lite(tmp_path: Path) -> None:
    specs = (
        MODULE.InputSpec(0, "signed", "tensor(int64)", (2,), "000_signed"),
        MODULE.InputSpec(1, "unsigned", "tensor(uint64)", (2,), "001_unsigned"),
        MODULE.InputSpec(2, "features", "tensor(float)", (2,), "002_features"),
    )
    writer = MODULE.CalibrationBinWriter(specs, tmp_path, limit=1)

    writer.write(
        {
            "signed": np.array([-1, 7], dtype=np.int64),
            "unsigned": np.array([3, 11], dtype=np.uint64),
            "features": np.array([1.5, -2.0], dtype=np.float32),
        }
    )

    assert np.fromfile(tmp_path / "000_signed" / "000000.bin", dtype=np.int32).tolist() == [
        -1,
        7,
    ]
    assert np.fromfile(
        tmp_path / "001_unsigned" / "000000.bin", dtype=np.uint32
    ).tolist() == [3, 11]
    assert np.fromfile(
        tmp_path / "002_features" / "000000.bin", dtype=np.float32
    ).tolist() == [1.5, -2.0]
