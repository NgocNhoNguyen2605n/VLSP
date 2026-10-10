"""Tiến độ gọn trên một dòng, ETA cho epoch hoặc pha hiện tại."""

import math
import time


def duration(seconds):
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class BatchProgress:
    def __init__(self, label, total, enabled=True, remaining_label="Còn"):
        self.label, self.total, self.enabled = label, total, enabled
        self.remaining_label = remaining_label
        self.started_at, self.width = time.monotonic(), 0

    def update(self, completed):
        if not self.enabled:
            return
        elapsed = max(0, time.monotonic() - self.started_at)
        eta = duration(elapsed / completed * (self.total - completed)) if completed else "--:--:--"
        line = f"{self.label} | Batch {completed}/{self.total} | {self.remaining_label} ~{eta}"
        print("\r" + line.ljust(self.width), end="", flush=True)
        self.width = len(line)
        if completed == self.total:
            self.close()

    def close(self):
        if self.enabled and self.width:
            print(flush=True)
            self.width = 0

    def __enter__(self):
        self.update(0)
        return self

    def __exit__(self, *exc):
        self.close()
