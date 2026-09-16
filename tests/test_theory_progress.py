"""A single parent progress bar stays live while device workers are computing."""

from __future__ import annotations

import os
from multiprocessing import get_context
from threading import Event

import pytest

from utils.experiments.theory import progress as theory_progress


class RecordingBar:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n = 0
        self.closed = False
        self.render_pids = []
        self.postfix = {}
        self.updated = Event()

    def set_postfix(self, **kwargs):
        self.postfix = kwargs
        self.render_pids.append(os.getpid())

    def update(self, count):
        self.n += count
        self.render_pids.append(os.getpid())
        self.updated.set()

    def close(self):
        self.closed = True
        self.render_pids.append(os.getpid())


@pytest.fixture
def bars(monkeypatch):
    created = []

    def create(**kwargs):
        bar = RecordingBar(**kwargs)
        created.append(bar)
        return bar

    monkeypatch.setattr(theory_progress, "tqdm", create)
    return created


def _report_from_child_then_wait(queue, release):
    theory_progress.install_progress_queue(queue)
    theory_progress.report_worker_record("child-record", "reduced", "cuda:1")
    if not release.wait(20):
        raise RuntimeError("Coordinator never consumed live child progress")


def test_theory_progress_is_visible_when_stderr_is_captured(capsys):
    with theory_progress.RecordProgress(
        total=1, devices=1, context=get_context("spawn"), label="First-step reference mean"
    ) as progress:
        progress.report("record-0", "reduced", "cpu")
    rendered = capsys.readouterr().err
    assert "[Theory] First-step reference mean" in rendered
    assert "1/1" in rendered
    assert "reduced=1" in rendered
    assert not progress._thread.is_alive()


def test_final_status_reconciliation_counts_each_record_once(bars):
    with theory_progress.RecordProgress(
        total=3, devices=2, context=get_context("spawn"), label="Posterior and condition interval refinement"
    ) as progress:
        progress.report("record-a", "reduced", "cuda:0")
        progress.report("record-a", "reduced", "cuda:0")
        progress.report("record-b", "resumed", "cuda:1")
        progress.report("record-b", "resumed", "cuda:1")
        progress.report("record-c", "failed", "cuda:0")
        # A result-level failure can supersede an earlier worker event. This
        # changes the status summary, but must not advance the record count.
        progress.report("record-a", "failed", "cuda:0")
        progress.report("record-c", "failed", "cuda:0")
    assert len(bars) == 1
    bar = bars[0]
    assert bar.kwargs == {
        "total": 3,
        "desc": "[Theory] Posterior and condition interval refinement",
        "unit": "record",
        "dynamic_ncols": True,
        "leave": True,
        "disable": False,
    }
    assert bar.n == 3
    assert bar.postfix["devices"] == 2
    assert progress.statuses == {
        "record-a": "failed",
        "record-b": "resumed",
        "record-c": "failed",
    }
    assert progress.counts["reduced"] == 0
    assert progress.counts["resumed"] == 1
    assert progress.counts["failed"] == 2
    assert bar.closed
    assert not progress._thread.is_alive()
    with pytest.raises(ValueError, match="closed"):
        progress.queue.put(("late", "reduced", "cpu"))


def test_child_progress_updates_before_busy_parent_or_child_finishes(bars):
    context = get_context("spawn")
    release = context.Event()
    with theory_progress.RecordProgress(
        total=2, devices=2, context=context
    ) as progress:
        child = context.Process(
            target=_report_from_child_then_wait,
            args=(progress.queue, release),
        )
        child.start()
        try:
            # The coordinator thread is occupied here, as it is when reducing
            # shard 0. The consumer must render the child update independently,
            # while the child is still active and has not returned its result.
            assert bars[0].updated.wait(15), "Child progress stalled behind local work"
            assert child.is_alive()
            assert bars[0].n == 1
            assert progress.statuses == {"child-record": "reduced"}
            progress.report("local-record", "resumed", "cuda:0")
        finally:
            release.set()
            child.join(15)
            if child.is_alive():
                child.terminate()
                child.join(5)
        assert child.exitcode == 0
        assert child.pid != os.getpid()
    assert len(bars) == 1
    assert bars[0].n == 2
    assert set(bars[0].render_pids) == {os.getpid()}
    assert progress.counts["reduced"] == progress.counts["resumed"] == 1
    assert not progress._thread.is_alive()


def test_exception_closes_partial_progress_without_hiding_failure(bars):
    with pytest.raises(RuntimeError, match="injected dispatch failure"):
        with theory_progress.RecordProgress(
            total=4, devices=2, context=get_context("spawn")
        ) as progress:
            progress.report("completed", "reduced", "cuda:1")
            raise RuntimeError("injected dispatch failure")
    assert len(bars) == 1
    assert bars[0].closed
    assert bars[0].n == 1
    assert progress.statuses == {"completed": "reduced"}
    assert not progress._thread.is_alive()
    with pytest.raises(ValueError, match="closed"):
        progress.queue.put(("late", "reduced", "cpu"))


def test_stage_logs_start_finish_and_elapsed_without_creating_a_bar(monkeypatch):
    from types import SimpleNamespace
    messages = []
    monkeypatch.setattr(theory_progress, "tqdm", SimpleNamespace(write=lambda message, **options: messages.append(message)))
    with theory_progress.StageProgress("Saving scalar tables") as stage:
        stage.set_detail("audit_data/feedback.csv: 100 rows")
    assert messages[0] == "[Theory] Saving scalar tables"
    assert messages[-1].startswith("[Theory] Finished: Saving scalar tables (")
    assert not hasattr(stage, "_thread")


def test_post_record_stage_has_no_heartbeat_or_extra_bar(monkeypatch):
    from types import SimpleNamespace
    messages = []
    def forbidden(*args, **kwargs):
        raise AssertionError("Stage started a competing progress timer")
    monkeypatch.setattr(theory_progress, "Thread", forbidden)
    monkeypatch.setattr(theory_progress, "tqdm", SimpleNamespace(write=lambda message, **options: messages.append(message)))
    with theory_progress.StageProgress("Reducing plot inputs") as stage:
        stage.set_detail("paired bootstrap intervals")
        assert messages == ["[Theory] Reducing plot inputs"]
    assert len(messages) == 2
    assert not any("still running" in message for message in messages)
    assert messages[-1].startswith("[Theory] Finished: Reducing plot inputs")
    assert not hasattr(stage, "_thread")


def test_stage_failure_is_reported_and_propagated(monkeypatch):
    from types import SimpleNamespace
    messages = []
    monkeypatch.setattr(theory_progress, "tqdm", SimpleNamespace(write=lambda message, **options: messages.append(message)))
    with pytest.raises(ValueError, match="failed serialization"):
        with theory_progress.StageProgress("Writing metadata") as stage:
            raise ValueError("failed serialization")
    assert messages[-1].startswith("[Theory] Failed: Writing metadata (")
    assert not any("Finished" in message for message in messages)
    assert not hasattr(stage, "_thread")


def test_worker_does_not_start_stage_logger_or_extra_bar(monkeypatch):
    monkeypatch.setattr(theory_progress, "parent_process", lambda: object())
    def forbidden(*args, **kwargs):
        raise AssertionError("Worker started a competing progress display")
    monkeypatch.setattr(theory_progress, "Thread", forbidden)
    monkeypatch.setattr(theory_progress.StageProgress, "_write", forbidden)
    with theory_progress.StageProgress("Worker") as stage:
        stage.set_detail("hidden")
    assert not stage.enabled
