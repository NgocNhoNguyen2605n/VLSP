"""Tiến độ gọn, ETA và log có xuống dòng khi output được thu qua pipe."""

import math
import sys
import time


def duration(seconds):
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class BatchProgress:
    def __init__(self, label, total, enabled=True, remaining_label="Còn", log_every=10):
        self.label, self.total, self.enabled = label, total, enabled
        self.remaining_label = remaining_label
        self.stream = sys.stdout
        self.inline = self.stream.isatty()
        self.log_every = max(1, log_every)
        self.started_at, self.width = time.monotonic(), 0
        self.last_logged_at = self.started_at

    def update(self, completed):
        if not self.enabled:
            return
        now = time.monotonic()
        if not self.inline and completed not in (0, 1, self.total):
            if completed % self.log_every and now - self.last_logged_at < 30:
                return
        elapsed = max(0, now - self.started_at)
        eta = duration(elapsed / completed * (self.total - completed)) if completed else "--:--:--"
        line = f"{self.label} | Batch {completed}/{self.total} | {self.remaining_label} ~{eta}"
        if self.inline:
            print("\r" + line.ljust(self.width), end="", file=self.stream, flush=True)
            self.width = len(line)
        else:
            print(line, file=self.stream, flush=True)
        self.last_logged_at = now
        if completed == self.total:
            self.close()

    def close(self):
        if self.enabled and self.width:
            print(file=self.stream, flush=True)
            self.width = 0

    def __enter__(self):
        self.update(0)
        return self

    def __exit__(self, *exc):
        self.close()
