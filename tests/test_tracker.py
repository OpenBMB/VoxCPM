from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "voxcpm" / "training" / "tracker.py"

spec = importlib.util.spec_from_file_location("voxcpm.training.tracker", MODULE_PATH)
tracker_mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tracker_mod)


class _Scalar:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


def test_format_metric_accepts_tensor_like_values():
    assert tracker_mod._format_metric(_Scalar(1.5)) == "1.500000"
    assert tracker_mod._format_metric(2) == "2.000000"
    assert tracker_mod._format_metric("n/a") == "n/a"


def test_log_interval_is_tracked_per_split(capsys):
    tracker = tracker_mod.TrainingTracker(rank=0)
    tracker.log_metrics({"loss": 1.0}, split="train")
    tracker.log_metrics({"loss": 0.5}, split="val")
    err = capsys.readouterr().err
    assert "log interval" not in err.split("[val]")[-1]

    tracker.log_metrics({"loss": 0.4}, split="val")
    err = capsys.readouterr().err
    assert "[val]" in err
    assert "log interval" in err
