"""Multi-device cache reductions: dispatch, genuine spawn, parity, and resume.

CUDA dispatch is checked with fake devices without GPU execution. Integration
cases require two CUDA devices and launch actual spawn workers, exercising IPC,
pickling and process-local state without downloading or loading model weights.
"""

from __future__ import annotations

from concurrent.futures import Future
from io import StringIO

import numpy as np
import pandas as pd
import pytest
import torch

from tests import test_theory_reduction as reduction_fixtures
from utils.common.io import (
    atomic_torch_save,
    atomic_write_json,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.experiments.theory import (
    cache_reader,
    progress as progress_ui,
    reduce as reducer,
)
from utils.experiments.theory.contracts import TheoryError, analysis_parent
from utils.experiments.theory.metrics import clean_estimates
from utils.experiments.theory.plotting import validate_bundle


CONFIG = reduction_fixtures.CONFIG
saved_cache = reduction_fixtures.saved_cache


@pytest.fixture
def progress_bars(monkeypatch):
    bars = []
    original = progress_ui.tqdm

    def capture(**kwargs):
        bar = original(**kwargs, file=StringIO())
        bars.append(bar)
        return bar

    monkeypatch.setattr(progress_ui, "tqdm", capture)
    return bars


requires_two_cuda = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required for actual parallel reductions")


def _two_cuda(_request):
    return (torch.device("cuda:0"), torch.device("cuda:1"))


def _table_snapshot(bundle):
    paths = [bundle / "initial_metrics.parquet", bundle / "endpoint_metrics.parquet"]
    paths += sorted((bundle / "plotdata").glob("*.parquet"))
    return {str(path.relative_to(bundle)): pd.read_parquet(path) for path in paths}


def _assert_table_parity(before, bundle):
    for relative, expected in before.items():
        observed = pd.read_parquet(bundle / relative)
        sort_columns = ["run_id", "original_index", "seed", "step_index"]
        expected = expected.sort_values(sort_columns).reset_index(drop=True)
        observed = observed.sort_values(sort_columns).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            expected, observed, check_exact=False, rtol=1e-12, atol=1e-12
        )


def test_auto_device_resolution_preserves_all_visible_cuda(monkeypatch):
    calls = []
    devices = tuple(torch.device("cuda", index) for index in range(4))

    def fake_resolve(requested):
        calls.append(requested)
        return devices

    monkeypatch.setattr(reducer, "resolve_devices", fake_resolve)
    assert reducer._resolve_theory_devices("auto") == devices
    assert calls == ["auto"]


def test_explicit_device_remains_one_device(monkeypatch):
    requested = torch.device("cuda", 2)
    monkeypatch.setattr(reducer, "resolve_devices", lambda value: (requested,))
    assert reducer._resolve_theory_devices("cuda:2") == (requested,)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_theory_rejects_non_cuda_without_fallback(monkeypatch, backend):
    monkeypatch.setattr(
        reducer, "resolve_devices", lambda value: (torch.device(backend),)
    )
    for request in ("auto", backend):
        with pytest.raises(TheoryError, match="requires CUDA; CPU/MPS fallback is disabled"):
            reducer._resolve_theory_devices(request)


@pytest.mark.parametrize("cuda_count", (1, 2, 8))
def test_fake_cuda_dispatch_is_disjoint_bounded_and_uses_spawn(
    saved_cache, monkeypatch, tmp_path, cuda_count
):
    sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    records = sources.records["experiment"]
    devices = tuple(torch.device("cuda", index) for index in range(cuda_count))
    calls, pools = [], []

    def fake_worker(**kwargs):
        indices = [record.original_index for record in kwargs["records"]]
        calls.append(
            dict(
                worker_index=kwargs["worker_index"],
                device=str(kwargs["device"]),
                indices=indices,
                worker_count=kwargs["worker_count"],
            )
        )
        return reducer._TheoryShardResult(
            worker_index=kwargs["worker_index"],
            device=str(kwargs["device"]),
            assigned_indices=indices,
            status_by_index={index: "reduced" for index in indices},
            failed_rows=[],
            pid=1000 + kwargs["worker_index"],
            cpu_threads=1,
        )

    class ImmediatePool:
        def __init__(self, **kwargs):
            pools.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, *args, **kwargs):
            future = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)
            return future

    monkeypatch.setattr(reducer, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(reducer, "_run_theory_shard", fake_worker)
    results = reducer._dispatch_theory_shards(
        sources=sources,
        records=records,
        bundle=tmp_path / "fake-bundle",
        analysis_hash="a" * 64,
        devices=devices,
        recompute=False,
        candidate_chunk_size=2,
        query_chunk_size=2,
        auxiliary_hashes={},
    )
    worker_count = min(len(records), len(devices))
    assert len(results) == worker_count
    assert len(pools) == int(worker_count > 1)
    if pools:
        assert pools[0]["max_workers"] == worker_count - 1
        assert pools[0]["mp_context"].get_start_method() == "spawn"
    assert {call["worker_count"] for call in calls} == {worker_count}
    ordered = sorted(calls, key=lambda call: call["worker_index"])
    expected = [record.original_index for record in records]
    assert [call["indices"] for call in ordered] == [
        expected[index::worker_count] for index in range(worker_count)
    ]
    assert [call["device"] for call in ordered] == [
        f"cuda:{index}" for index in range(worker_count)
    ]
    all_indices = [index for call in calls for index in call["indices"]]
    assert len(all_indices) == len(set(all_indices)) == len(records)


@requires_two_cuda
def test_two_actual_cuda_spawn_workers_match_serial_and_share_fixed_center(
    saved_cache, monkeypatch, progress_bars
):
    root = saved_cache["root"]
    serial = reducer.run_theory(
        root, **CONFIG, device="cuda:0", candidate_chunk_size=2, query_chunk_size=2
    )
    before = _table_snapshot(serial)
    center_before = safe_torch_load(serial / "center.pt").clone()
    monkeypatch.setattr(reducer, "_resolve_theory_devices", _two_cuda)
    parallel = reducer.run_theory(
        root,
        **CONFIG,
        device="auto",
        recompute=True,
        candidate_chunk_size=2,
        query_chunk_size=2,
    )
    manifest = validate_bundle(parallel)
    assert len(progress_bars) == 2, "Exactly one bar per run, regardless of devices"
    for bar in progress_bars:
        assert bar.n == bar.total == len(saved_cache["selection"].included_indices)
        assert "failed=0" in bar.postfix
    assert "devices=2" in progress_bars[-1].postfix
    assert parallel == serial, "Worker count and ordinal layout are execution metadata"
    _assert_table_parity(before, parallel)
    assert manifest["execution"]["worker_count"] == 2
    assert manifest["execution"]["resolved_devices"] == ["cuda:0", "cuda:1"]
    shards = manifest["execution"]["device_shards"]
    assert len({shard["pid"] for shard in shards}) == 2
    assert all(shard["cpu_threads"] >= 1 for shard in shards)
    assert len(
        {index for shard in shards for index in shard["assigned_indices"]}
    ) == len(saved_cache["selection"].included_indices)
    torch.testing.assert_close(
        center_before, safe_torch_load(parallel / "center.pt"), rtol=0, atol=0
    )

    initial = pd.read_parquet(parallel / "initial_metrics.parquet")
    sources = cache_reader.discover_sources(root, **CONFIG)
    schedule = cache_reader.load_schedule(sources)
    # Recompute in float64 from the saved cumulative alpha; saved float32
    # square-root convenience fields already contain extra rounding.
    cumulative = float(schedule["alphas_cumprod_t"][0])
    alpha, sigma = cumulative**0.5, (1 - cumulative) ** 0.5
    center = safe_torch_load(parallel / "center.pt")
    for record in sources.selected:
        z, u, c, target = cache_reader.load_record(sources.experiment, record)
        m_u, _, _, _ = clean_estimates(
            z[:, 0], u[:, 0], c[:, 0], alpha, sigma, CONFIG["guidance_scale"]
        )
        expected = (
            (m_u - center).double().flatten(1).square().mean(dim=1).sqrt().numpy()
        )
        actual = initial.loc[
            initial.original_index.eq(record.original_index)
        ].sort_values("seed")
        np.testing.assert_allclose(
            actual.unconditional_center_rmse, expected, rtol=1e-12, atol=1e-12
        )
    expected_keys = {
        (record.original_index, seed, step)
        for record in sources.selected
        for seed in range(CONFIG["num_seeds"])
        for step in range(CONFIG["num_inference_steps"])
    }
    trajectory = pd.concat(
        [
            pd.read_parquet(path)
            for path in (parallel / "trajectory_metrics").glob("*.parquet")
        ]
    )
    actual_keys = list(
        zip(trajectory.original_index, trajectory.seed, trajectory.step_index)
    )
    assert (
        len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys
    )


@requires_two_cuda
def test_parallel_failure_withholds_manifest_and_resumes_completed_other_worker(
    saved_cache, monkeypatch, progress_bars
):
    root = saved_cache["root"]
    sources = cache_reader.discover_sources(root, **CONFIG)
    failing_index = sources.selected[0].original_index
    original = reducer._record_metrics

    def fail_local_record(sources, record, *args, **kwargs):
        if record.original_index == failing_index:
            raise RuntimeError("injected coordinator-worker record failure")
        return original(sources, record, *args, **kwargs)

    monkeypatch.setattr(reducer, "_resolve_theory_devices", _two_cuda)
    monkeypatch.setattr(reducer, "_record_metrics", fail_local_record)
    with pytest.raises(TheoryError, match="failed|complete manifest"):
        reducer.run_theory(
            root, **CONFIG, device="auto", candidate_chunk_size=2, query_chunk_size=2
        )
    bundles = [
        path for path in analysis_parent(root, **CONFIG).iterdir() if path.is_dir()
    ]
    assert len(bundles) == 1
    bundle = bundles[0]
    assert not (bundle / "manifest.json").exists()
    completed = sorted((bundle / "trajectory_metrics").glob("*.parquet"))
    assert completed, "The independent subprocess worker must finish its valid record"
    assert all(failing_index not in path.name for path in completed)
    completed_artifacts = [
        path
        for directory in ("trajectory_metrics", "initial_shards", "endpoint_shards")
        for path in (bundle / directory).iterdir()
        if path.is_file()
    ]
    protected = {
        path.relative_to(bundle): (file_sha256(path), path.stat().st_mtime_ns)
        for path in completed_artifacts
    }
    failures = pd.read_csv(bundle / "failed.csv", dtype={"original_index": str})
    assert failing_index in set(failures.original_index)
    assert len(progress_bars) == 1
    assert progress_bars[0].n == progress_bars[0].total == len(sources.selected)
    assert "failed=1" in progress_bars[0].postfix
    monkeypatch.setattr(reducer, "_record_metrics", original)
    # Resume with a different device layout: numerical identity and completed
    # atomic record shards must remain reusable.
    monkeypatch.setattr(
        reducer, "_resolve_theory_devices", lambda request: (torch.device("cuda:0"),)
    )
    resumed = reducer.run_theory(
        root, **CONFIG, device="auto", candidate_chunk_size=2, query_chunk_size=2
    )
    assert resumed == bundle
    resumed_manifest = validate_bundle(resumed)
    assert resumed_manifest["complete"]
    assert resumed_manifest["execution"]["worker_count"] == 1
    assert len(progress_bars) == 2
    assert progress_bars[1].n == progress_bars[1].total == len(sources.selected)
    assert f"resumed={len(completed)}" in progress_bars[1].postfix
    assert "failed=0" in progress_bars[1].postfix
    for path in completed_artifacts:
        assert (file_sha256(path), path.stat().st_mtime_ns) == protected[
            path.relative_to(bundle)
        ]
    assert failing_index in set(
        pd.read_csv(bundle / "failed.csv", dtype={"original_index": str}).original_index
    )


@requires_two_cuda
def test_worker_initial_consistency_uses_global_anchor_not_its_own_first_record(
    saved_cache, monkeypatch
):
    root = saved_cache["root"]
    indices = sorted(saved_cache["selection"].included_indices)
    assert len(indices) >= 2
    changed_index = indices[1]  # Assigned to subprocess worker 1 by round-robin.
    paths = saved_cache["paths"]["experiment"]
    prediction_path = paths.noise_prediction_path(changed_index)
    u, c = safe_torch_load(prediction_path)
    u[:, 0] += 0.5
    digest = atomic_torch_save((u, c), prediction_path)
    marker = read_json(paths.record_path(changed_index))
    marker["tensor_file_sha256"]["noise_prediction"] = digest
    atomic_write_json(paths.record_path(changed_index), marker)
    monkeypatch.setattr(reducer, "_resolve_theory_devices", _two_cuda)
    with pytest.raises(TheoryError, match="failed|complete manifest"):
        reducer.run_theory(
            root, **CONFIG, device="auto", candidate_chunk_size=2, query_chunk_size=2
        )
    bundles = [
        path for path in analysis_parent(root, **CONFIG).iterdir() if path.is_dir()
    ]
    assert len(bundles) == 1
    assert not (bundles[0] / "manifest.json").exists()
    failures = pd.read_csv(bundles[0] / "failed.csv", dtype={"original_index": str})
    failed = failures.loc[failures.original_index.eq(changed_index)]
    assert len(failed) == 1
    assert "initial unconditional outputs inconsistent" in failed.iloc[0]["error"]
