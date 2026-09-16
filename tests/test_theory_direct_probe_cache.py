"""Unexecuted narrow source-transition regressions for completed probe caches."""

from copy import deepcopy
from pathlib import Path

import pytest

from utils.common.io import atomic_write_json, canonical_hash, file_sha256, read_json
from utils.experiments.theory import direct_probe_cache as cache


@pytest.fixture(autouse=True)
def audited_fixed_source(monkeypatch):
    monkeypatch.setattr(cache, "AUDITED_FIXED_SOURCE_SHA256", "b" * 64)
    monkeypatch.setattr(cache, "AUDITED_ZERO_BASELINE_PROBE_SHA256", "b" * 64)
    monkeypatch.setattr(cache, "AUDITED_ZERO_BASELINE_MATH_SHA256", "e" * 64)
    monkeypatch.setattr(cache, "ZERO_BASELINE_MATH_PREDECESSOR_SHA256", "c" * 64)


def task_paths(directory, task):
    base = Path(directory) / "tasks" / task["table"] / task["task_hash"]
    return base.with_suffix(".parquet"), base.with_suffix(".json")


def make_task(*, source_hash="b" * 64, math_hash="c" * 64, **extra):
    identity = {
        "source_code": {
            cache.PROBE_SOURCE_KEY: source_hash,
            cache.MATH_SOURCE_KEY: math_hash,
            "models/sampling.py": "d" * 64,
        },
        "policy": {"schema_version": 1, "formula_version": "direct-probe-math-1"},
        "table": "forward_loss_draws",
        "count": 64,
        "pair": {"record_id": "pair", "target_id": "target"},
        "loss_seed": 0,
        "reference_law_hash": "law",
        **extra,
    }
    return {"table": identity["table"], "count": identity["count"],
            "identity": identity, "task_hash": canonical_hash(identity)}


def save_task(directory, task, *, marker_updates=None):
    data, marker = task_paths(directory, task)
    data.parent.mkdir(parents=True, exist_ok=True)
    # Resolution verifies bytes and metadata without parsing numerical payloads.
    data.write_bytes(b"unchanged-completed-numerical-observations")
    value = {
        "schema_version": 1, "complete": True, "task_hash": task["task_hash"],
        "identity": task["identity"], "rows": task["count"],
        "sha256": file_sha256(data),
    }
    atomic_write_json(marker, value | (marker_updates or {}))
    return data, marker


def test_exact_task_precedes_compatible_and_read_only_resolution_preserves_sources(tmp_path):
    current = make_task()
    prior = make_task(source_hash=cache.CACHED_BATCHING_PREDECESSOR_SHA256)
    old_data, old_marker = save_task(tmp_path, prior)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (old_data, old_marker)}
    untouched_task = deepcopy(current)
    assert cache.resolve_completed_task(tmp_path, current, path_for_task=task_paths) == (
        prior["task_hash"], old_data, old_marker
    )
    assert current == untouched_task
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    assert not any(path.exists() for path in task_paths(tmp_path, current))
    assert read_json(old_marker)["task_hash"] == prior["task_hash"]
    new_data, new_marker = save_task(tmp_path, current)
    assert cache.resolve_completed_task(tmp_path, current, path_for_task=task_paths) == (
        current["task_hash"], new_data, new_marker
    )


@pytest.mark.parametrize("marker_updates", [
    {"complete": False}, {"schema_version": 2}, {"rows": 63},
    {"rows": True}, {"sha256": "corrupt"}, {"task_hash": "other"},
    {"identity": {"different": "science"}},
])
def test_incomplete_or_changed_predecessors_are_never_resumed(tmp_path, marker_updates):
    prior = make_task(source_hash=cache.CACHED_BATCHING_PREDECESSOR_SHA256)
    save_task(tmp_path, prior, marker_updates=marker_updates)
    assert cache.resolve_completed_task(tmp_path, make_task(), path_for_task=task_paths) is None


@pytest.mark.parametrize("difference", ["draws", "loss_seed", "reference_law", "math_source", "unknown_predecessor"])
def test_compatibility_requires_every_other_scientific_field_to_match(tmp_path, difference):
    prior = make_task(source_hash=cache.CACHED_BATCHING_PREDECESSOR_SHA256)
    identity = deepcopy(prior["identity"])
    if difference == "draws":
        identity["count"] = 128
    elif difference == "loss_seed":
        identity["loss_seed"] = 5
    elif difference == "reference_law":
        identity["reference_law_hash"] = "different-law"
    elif difference == "math_source":
        identity["source_code"]["experiments/theory/direct_probe_math.py"] = "e" * 64
    else:
        identity["source_code"][cache.PROBE_SOURCE_KEY] = "f" * 64
    prior.update(identity=identity, count=identity["count"], task_hash=canonical_hash(identity))
    save_task(tmp_path, prior)
    assert cache.resolve_completed_task(tmp_path, make_task(), path_for_task=task_paths) is None


@pytest.mark.parametrize("damaged", ["data", "marker", "symlink"])
def test_missing_invalid_and_unsafe_predecessor_artifacts_are_rejected(tmp_path, damaged):
    prior = make_task(source_hash=cache.CACHED_BATCHING_PREDECESSOR_SHA256)
    data, marker = save_task(tmp_path, prior)
    if damaged == "data":
        data.write_bytes(b"incomplete-task")
    elif damaged == "marker":
        marker.write_text("{broken")
    else:
        real = data.with_suffix(".payload")
        data.rename(real)
        data.symlink_to(real.name)
    assert cache.resolve_completed_task(tmp_path, make_task(), path_for_task=task_paths) is None


def test_a_malformed_requested_task_cannot_point_to_an_unrelated_completed_shard(tmp_path):
    current = make_task()
    save_task(tmp_path, current)
    malformed = current | {"identity": current["identity"] | {"loss_seed": 8}}
    assert cache.resolve_completed_task(tmp_path, malformed, path_for_task=task_paths) is None


def test_future_orchestrator_edits_do_not_inherit_predecessor_compatibility(tmp_path):
    prior = make_task(source_hash=cache.CACHED_BATCHING_PREDECESSOR_SHA256)
    save_task(tmp_path, prior)
    future = make_task(source_hash="9" * 64)
    assert cache.resolve_completed_task(tmp_path, future, path_for_task=task_paths) is None
    data, marker = save_task(tmp_path, future)
    assert cache.resolve_completed_task(tmp_path, future, path_for_task=task_paths) == (
        future["task_hash"], data, marker
    )


@pytest.mark.parametrize("table", sorted(cache.ZERO_BASELINE_UNCHANGED_TABLES))
def test_zero_baseline_addition_reuses_only_unchanged_observations_read_only(tmp_path, table):
    prior = make_task(table=table)
    current = make_task(table=table, math_hash="e" * 64)
    data, marker = save_task(tmp_path, prior)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (data, marker)}
    untouched = deepcopy(current)
    assert cache.resolve_completed_task(tmp_path, current, path_for_task=task_paths) == (
        prior["task_hash"], data, marker
    )
    assert current == untouched
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    assert not any(path.exists() for path in task_paths(tmp_path, current))
    # The real source task hash survives into caller provenance; no scalar row
    # or completion marker is relabeled as a newly measured observation.
    assert read_json(marker)["task_hash"] == prior["task_hash"]
    new_data, new_marker = save_task(tmp_path, current)
    assert cache.resolve_completed_task(tmp_path, current, path_for_task=task_paths) == (
        current["task_hash"], new_data, new_marker
    )


@pytest.mark.parametrize("table", ["gaussian_reference", "genuine_gaussian_unconditional_controls", "unknown_table"])
def test_zero_baseline_addition_never_relabels_missing_zero_fields_or_other_tables(tmp_path, table):
    save_task(tmp_path, make_task(table=table))
    requested = make_task(table=table, math_hash="e" * 64)
    assert cache.resolve_completed_task(tmp_path, requested, path_for_task=task_paths) is None


@pytest.mark.parametrize("changed", ["count", "loss_seed", "reference_law", "formula", "checkpoint", "sampling", "math", "orchestrator"])
def test_zero_baseline_compatibility_requires_all_other_identity_fields(tmp_path, changed):
    prior = make_task()
    current = make_task(math_hash="e" * 64)
    identity = deepcopy(prior["identity"])
    if changed == "count":
        identity["count"] = 128
    elif changed == "loss_seed":
        identity["loss_seed"] = 123
    elif changed == "reference_law":
        identity["reference_law_hash"] = "different-law"
    elif changed == "formula":
        identity["policy"]["formula_version"] = "unreviewed-formula"
    elif changed == "checkpoint":
        identity["checkpoint"] = {"model_id": "different-model"}
    elif changed == "sampling":
        identity["source_code"]["models/sampling.py"] = "f" * 64
    elif changed == "math":
        identity["source_code"][cache.MATH_SOURCE_KEY] = "f" * 64
    else:
        identity["source_code"][cache.PROBE_SOURCE_KEY] = "f" * 64
    prior.update(identity=identity, count=identity["count"], task_hash=canonical_hash(identity))
    save_task(tmp_path, prior)
    assert cache.resolve_completed_task(tmp_path, current, path_for_task=task_paths) is None


@pytest.mark.parametrize("marker_updates", [{"complete": False}, {"rows": 63}, {"sha256": "corrupt"}, {"schema_version": 2}])
def test_zero_baseline_compatibility_still_requires_verified_completed_payload(tmp_path, marker_updates):
    save_task(tmp_path, make_task(), marker_updates=marker_updates)
    requested = make_task(math_hash="e" * 64)
    assert cache.resolve_completed_task(tmp_path, requested, path_for_task=task_paths) is None


@pytest.mark.parametrize("future", ["math", "orchestrator"])
def test_future_sources_do_not_inherit_zero_baseline_compatibility(tmp_path, future):
    save_task(tmp_path, make_task())
    requested = make_task(math_hash="9" * 64 if future == "math" else "e" * 64,
                          source_hash="9" * 64 if future == "orchestrator" else "b" * 64)
    assert cache.resolve_completed_task(tmp_path, requested, path_for_task=task_paths) is None
