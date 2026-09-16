"""One coordinator-owned record bar for local and spawned theory workers."""

from __future__ import annotations

from collections import Counter
from multiprocessing import parent_process
import sys
from threading import Event, Thread
from time import monotonic

from tqdm import tqdm


_worker_progress_queue = None


def install_progress_queue(queue):
    """Install the spawn-context queue without sending it through task pickling."""
    global _worker_progress_queue
    _worker_progress_queue = queue


def report_worker_record(original_index, status, device):
    """Workers send only small terminal-status events; they never render a bar."""
    if _worker_progress_queue is not None:
        _worker_progress_queue.put((original_index, status, str(device)))


class RecordProgress:
    """Consume all device events while the coordinator also computes its shard.

    Keep this context outside the executor so workers finish flushing their
    events before the stop marker is queued. Final worker results can be reported
    again to reconcile failures; repeated record IDs never advance the bar twice.
    """

    def __init__(self, *, total, devices, context):
        self.total = total
        self.devices = devices
        self.queue = context.Queue()
        self.statuses = {}
        self.counts = Counter()

    def __enter__(self):
        self.bar = tqdm(
            total=self.total,
            desc="[Theory] Records",
            unit="record",
            dynamic_ncols=True,
            leave=True,
            disable=False,
        )
        self.bar.set_postfix(
            devices=self.devices, reduced=0, resumed=0, failed=0, refresh=False
        )
        self._thread = Thread(target=self._consume, name="theory-progress", daemon=True)
        self._thread.start()
        return self

    def report(self, original_index, status, device):
        self.queue.put((original_index, status, str(device)))

    def _consume(self):
        while (event := self.queue.get()) is not None:
            original_index, status, device = event
            previous = self.statuses.get(original_index)
            if previous == status:
                continue
            if previous is not None:
                self.counts[previous] -= 1
            self.statuses[original_index] = status
            self.counts[status] += 1
            self.bar.set_postfix(
                devices=self.devices,
                record=original_index,
                device=device,
                status=status,
                reduced=self.counts["reduced"],
                resumed=self.counts["resumed"],
                failed=self.counts["failed"],
                refresh=False,
            )
            if previous is None:
                self.bar.update(1)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.queue.put(None)
            self._thread.join()
        finally:
            self.bar.close()
            self.queue.close()
            self.queue.join_thread()


class StageProgress:
    """Parent-only elapsed-time logs for work without a meaningful item count.

    The heartbeat keeps scalar aggregation and filesystem publication visible
    after worker record bars finish. It uses tqdm.write to preserve any active
    parent bar; workers never start a competing logger or timer thread.
    """

    def __init__(self, label, *, heartbeat_seconds=20.):
        if not 0 < heartbeat_seconds < float("inf"):
            raise ValueError("heartbeat_seconds must be finite and positive")
        self.label = str(label)
        self.heartbeat_seconds = heartbeat_seconds
        self.detail = ""
        self.enabled = parent_process() is None

    def set_detail(self, detail):
        self.detail = str(detail)

    def _write(self, message):
        tqdm.write("[Theory] " + message, file=sys.stderr)
        sys.stderr.flush()

    def __enter__(self):
        self.started = monotonic()
        if self.enabled:
            self._stop = Event()
            self._write(self.label)
            self._thread = Thread(target=self._heartbeat, name="theory-stage-progress", daemon=True)
            self._thread.start()
        return self

    def _heartbeat(self):
        while not self._stop.wait(self.heartbeat_seconds):
            detail = "; " + self.detail if self.detail else ""
            self._write(f"{self.label}: still running ({monotonic() - self.started:.1f}s{detail})")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            self._stop.set()
            self._thread.join()
            status = "Finished" if exc_type is None else "Failed"
            self._write(f"{status}: {self.label} ({monotonic() - self.started:.1f}s)")
        return False
