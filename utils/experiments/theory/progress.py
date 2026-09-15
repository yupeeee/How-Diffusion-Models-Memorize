"""One coordinator-owned record bar for local and spawned theory workers."""

from __future__ import annotations

from collections import Counter
from threading import Thread

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
