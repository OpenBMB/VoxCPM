from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path


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
        log_file: str | None = None,
        rank: int = 0,
    ):
        self.writer = writer
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.step = 0
        # Record the time of the last log to calculate the interval
        self._last_log_time: float | None = None

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    def print(self, message: str):
        if self.rank == 0:
            print(message, flush=True, file=sys.stderr)
            if self.log_file:
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(message + "\n")

    def log_metrics(self, metrics: dict[str, float], split: str):
        if self.rank == 0:
            now = time.time()
            dt_str = ""
            if self._last_log_time is not None:
                dt = now - self._last_log_time
                dt_str = f", log interval: {dt:.2f}s"
            self._last_log_time = now

            formatted = ", ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
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
