"""One coordinator-owned record bar for local and spawned theory workers."""

from __future__ import annotations

from collections import Counter
from multiprocessing import parent_process
import sys
from threading import Thread
from time import monotonic

from tqdm import tqdm


_worker_progress_queue = None


# Presentation labels live outside the source-hashed measurement modules so
# renaming a progress bar never invalidates expensive numerical caches.
_OPERATION_LABELS = {
    ("reference_mean", "estimate_reference_mean"): "Analytical reference mean",
    ("direct_probes", "run_direct_probes"): "Forward-loss and Gaussian probes",
    ("direct_reduce", "run_direct_analysis"): "Trajectory metrics and posterior integration",
    ("evidence_reduce", "run_evidence_analysis"): "Reference SNR sweep and terminal bounds",
    ("numerical_reduce", "run_precision_analysis"): "Posterior and condition interval refinement",
    ("four_stage_reduce", "prepare_four_stage_primary"): "Initial and trajectory measurements",
    ("four_stage_reduce", "run_four_stage_analysis"): "Posterior response and trajectory shape",
    ("gaussian_controls", "run_gaussian_control_analysis"): "Initial Gaussian unconditional controls",
    ("counterfactual_probes", "run_counterfactual_analysis"): "Counterfactual denoiser probes",
    ("candidate_reduce", "_dispatch"): "Candidate measurements",
    ("reduce", "_dispatch_theory_shards"): "Cached trajectory theory measurements",
}
_CANDIDATE_STAGE_LABELS = {
    "endpoints": "Candidate endpoint measurements",
    "integration": "Candidate posterior integration",
}


def _operation_label(caller):
    """Resolve existing callers without changing their scientific source hashes."""
    module = caller.f_globals.get("__name__", "").rsplit(".", 1)[-1]
    function = caller.f_code.co_name
    key = module, function
    if key == ("candidate_reduce", "_dispatch"):
        stage = caller.f_locals.get("stage")
        if stage in _CANDIDATE_STAGE_LABELS:
            return _CANDIDATE_STAGE_LABELS[stage]
    return _OPERATION_LABELS.get(key, function.strip("_").replace("_", " ").capitalize())


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

    def __init__(self, *, total, devices, context, label=None):
        self.label = str(label) if label is not None else _operation_label(sys._getframe(1))
        self.total = total
        self.devices = devices
        self.queue = context.Queue()
        self.statuses = {}
        self.counts = Counter()

    def __enter__(self):
        self.bar = tqdm(
            total=self.total,
            desc="[Theory] " + self.label,
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
    """Parent-only start/completion logs without periodic progress messages.

    Existing tqdm bars provide live progress. Stages with no item count report
    their boundaries once, without starting a competing timer or progress bar.
    """

    def __init__(self, label):
        self.label = str(label)
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
            self._write(self.label)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            status = "Finished" if exc_type is None else "Failed"
            self._write(f"{status}: {self.label} ({monotonic() - self.started:.1f}s)")
        return False
