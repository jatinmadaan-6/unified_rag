"""
Append-only, flush-and-fsync-on-every-write JSONL checkpointing.

The whole point: a QA pair is written to disk THE INSTANT it's accepted —
not buffered in memory and written out in a batch at the end of a round.
If Colab disconnects mid-run, everything accepted before the disconnect
is already safely on disk, and re-running the pipeline resumes instead of
starting over from zero.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class CheckpointWriter:
    """Wraps an output JSONL file in append mode. Every write() call is
    flushed and fsync'd immediately, so a crash/disconnect the very next
    instant still leaves that record safely on disk."""

    def __init__(self, path: str):
        self.path = path
        # "a" (append) mode is what makes resuming safe: re-running the
        # pipeline against the same output_path never clobbers what a
        # previous, interrupted run already wrote.
        self._fh = open(path, "a", buffering=1)

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001 - never let cleanup crash the run
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def load_existing_records(path: str) -> List[dict]:
    """Read whatever is already on disk from a previous (possibly
    interrupted) run, so a restart can resume from where it left off
    instead of regenerating everything from scratch."""
    if not os.path.exists(path):
        return []
    records: List[dict] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A disconnect can cut off the very last line mid-write.
                # Skip it rather than crashing the whole resume.
                logger.warning("Skipping malformed checkpoint line %d in %s "
                                "(likely cut off by a previous disconnect).", line_no, path)
                continue
    return records
