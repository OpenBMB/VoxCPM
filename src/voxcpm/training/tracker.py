from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path
from typing import Dict, Optional


def _format_metric(value) -> str:
    """Format a logged metric without raising on tensors or other numeric wrappers."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            return str(value)
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


class TrainingTracker:
    """
    Lightweight tracker inspired by the minimcpm-audio training workflow.

    It keeps track of the current global step, prints rank-aware messages,
    optionally writes to TensorBoard via a provided writer, and stores progress
    in a logfile for later inspection.
    """

    def __init__(
        self,
        *,
        writer=None,
        log_file: Optional[str] = None,
        rank: int = 0,
    ):
        self.writer = writer
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.step = 0
        # Per-split timestamps so a validation log does not report the gap since the last training log.
        self._last_log_time: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    def print(self, message: str):
        if self.rank == 0:
            print(message, flush=True, file=sys.stderr)
            if self.log_file:
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(message + "\n")

    def log_metrics(self, metrics: Dict[str, float], split: str):
        if self.rank == 0:
            now = time.time()
            dt_str = ""
            previous = self._last_log_time.get(split)
            if previous is not None:
                dt = now - previous
                dt_str = f", log interval: {dt:.2f}s"
            self._last_log_time[split] = now

            formatted = ", ".join(f"{k}: {_format_metric(v)}" for k, v in metrics.items())
            self.print(f"[{split}] step {self.step}: {formatted}{dt_str}")
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{split}/{key}", value, self.step)

    def done(self, split: str, message: str):
        self.print(f"[{split}] {message}")

    # ------------------------------------------------------------------ #
    # State dict
    # ------------------------------------------------------------------ #
    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        self.step = int(state.get("step", 0))

    # ------------------------------------------------------------------ #
    # Context manager compatibility (for parity with minicpm-audio code)
    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def live(self):
        yield
