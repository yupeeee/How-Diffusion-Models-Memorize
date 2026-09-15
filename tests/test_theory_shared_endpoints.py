"""One protected record read supplies both base and candidate endpoint outputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests.test_theory_reduction import CONFIG, saved_cache as _saved_cache
from utils.common.io import file_sha256, read_json
from utils.experiments.theory import candidate_reduce as candidate, reduce as base
from utils.experiments.theory.cache_reader import discover_sources, load_schedule
from utils.experiments.theory.candidate_contracts import read_tables
from utils.experiments.theory.scheduler_adapter import SchedulerAdapter

saved_cache = _saved_cache


def forbidden(*args, **kwargs):
    raise AssertionError("A second protected record load is forbidden")


def _shared(fixture, **options):
    return base.run_theory(
        fixture["root"],
        device="cpu",
        defer_path_integration=True,
        prepare_candidate_endpoints=True,
        **options,
        **CONFIG,
    )


def _candidate(fixture, bundle, **options):
    return candidate.run_candidates(
        fixture["root"],
        base_bundle=bundle,
        device="cpu",
        render=False,
        endpoint_only=True,
        **options,
        **CONFIG,
    )


def test_one_full_read_per_record_and_exact_two_pass_endpoint_parity(
    saved_cache, monkeypatch
):
    calls = []
    original = base.load_record

    def counted(paths, record, **options):
        if not options.get("initial_only"):
            calls.append(record.original_index)
        return original(paths, record, **options)

    monkeypatch.setattr(base, "load_record", counted)
    monkeypatch.setattr(candidate, "load_record", forbidden)
    bundle = _shared(saved_cache, query_chunk_size=2)
    expected = sorted(saved_cache["selection"].included_indices)
    assert sorted(calls) == expected
    manifest = read_json(bundle / "manifest.json")
    assert manifest["prepare_candidate_endpoints"]
    assert manifest["candidate_endpoint_extension"]["complete"]
    assert manifest["candidate_endpoint_extension"]["record_indices"] == expected
    extension = bundle / candidate.SHARED_ENDPOINT_DIRECTORY
    snapshot = {
        p: (file_sha256(p), p.stat().st_mtime_ns)
        for p in extension.rglob("*")
        if p.is_file()
    }
    endpoint = _candidate(saved_cache, bundle, query_chunk_size=2)
    assert sorted(calls) == expected
    assert snapshot == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in snapshot}
    loaded = read_tables(endpoint)
    assert (
        len(loaded["trajectory"])
        == len(expected) * CONFIG["num_seeds"] * CONFIG["num_inference_steps"]
    )

    # Independent legacy disk entry point reproduces every saved scalar and compact
    # segment array. Its explicit extra load occurs only in this parity test.
    monkeypatch.setattr(candidate, "load_record", original)
    sources = discover_sources(saved_cache["root"], **CONFIG)
    schedule = load_schedule(sources)
    support, support_meta, _ = base.build_support(sources, query_chunk_size=2)
    center = torch.load(bundle / "center.pt", weights_only=True)
    adapter = SchedulerAdapter(schedule)
    for record in sources.selected:
        expected_result = candidate._endpoint_record(
            sources,
            record,
            bundle,
            support,
            center,
            schedule,
            adapter,
            device=torch.device("cpu"),
            query_chunk_size=2,
            bank_hash=support_meta["tensor_sha256"],
        )
        paths = candidate._stage_paths(extension, record, "endpoints")
        for computed, path in zip(expected_result[:5], paths[:5], strict=True):
            restored = pd.read_parquet(path)
            # Round-trip list columns through the same Parquet representation.
            comparison = Path(saved_cache["root"]) / "comparison.parquet"
            computed.to_parquet(comparison, index=False)
            pd.testing.assert_frame_equal(
                pd.read_parquet(comparison), restored, check_exact=True
            )
        saved_segments = list(candidate._read_segments(paths[-1], "cpu"))
        assert len(saved_segments) == len(expected_result[-1])
        for (a_keys, a), (b_keys, b) in zip(
            saved_segments, expected_result[-1], strict=True
        ):
            assert a_keys == b_keys and a.keys() == b.keys()
            for name in a:
                if isinstance(a[name], torch.Tensor):
                    torch.testing.assert_close(
                        a[name], b[name], rtol=0, atol=0, equal_nan=True
                    )
                elif isinstance(a[name], float) and np.isnan(a[name]):
                    assert np.isnan(b[name])
                else:
                    assert a[name] == b[name]


@pytest.mark.parametrize("missing_index", [-1, 0, "identity", "corrupt_identity"])
def test_shared_resume_repairs_only_missing_extension_record(
    saved_cache, monkeypatch, missing_index
):
    bundle = _shared(saved_cache)
    sources = discover_sources(saved_cache["root"], **CONFIG)
    records = sorted(sources.selected, key=lambda record: record.original_index)
    extension = bundle / candidate.SHARED_ENDPOINT_DIRECTORY
    preserved = candidate._stage_paths(extension, records[0], "endpoints")
    before = {p: (file_sha256(p), p.stat().st_mtime_ns) for p in preserved}
    if missing_index == "corrupt_identity":
        (extension / "extension_identity.json").write_text("{")
    elif missing_index == "identity":
        (extension / "extension_identity.json").unlink()
    else:
        missing = candidate._stage_paths(extension, records[1], "endpoints")[
            missing_index
        ]
        missing.unlink()
    calls = []
    original = base.load_record

    def counted(paths, record, **options):
        if not options.get("initial_only"):
            calls.append(record.original_index)
        return original(paths, record, **options)

    monkeypatch.setattr(base, "load_record", counted)
    monkeypatch.setattr(candidate, "load_record", forbidden)
    assert _shared(saved_cache) == bundle
    assert calls == (
        [] if isinstance(missing_index, str) else [records[1].original_index]
    )
    assert before == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in preserved}
    _candidate(saved_cache, bundle)
    calls.clear()
    assert _shared(saved_cache) == bundle
    assert calls == []


def test_shared_failure_withholds_publication_and_reuses_finished_record(
    saved_cache, monkeypatch
):
    sources = discover_sources(saved_cache["root"], **CONFIG)
    chosen = sorted(sources.selected, key=lambda record: record.original_index)[
        -1
    ].original_index
    actual = candidate._endpoint_record

    def fail_selected(sources, record, *args, **kwargs):
        if record.original_index == chosen:
            raise RuntimeError("injected shared endpoint failure")
        return actual(sources, record, *args, **kwargs)

    monkeypatch.setattr(candidate, "_endpoint_record", fail_selected)
    with pytest.raises(base.TheoryError, match="failed"):
        _shared(saved_cache)
    parent = base.analysis_parent(saved_cache["root"], **CONFIG)
    assert not list(parent.glob("*/manifest.json"))
    monkeypatch.setattr(candidate, "_endpoint_record", actual)
    calls = []
    original = base.load_record

    def counted(paths, record, **options):
        if not options.get("initial_only"):
            calls.append(record.original_index)
        return original(paths, record, **options)

    monkeypatch.setattr(base, "load_record", counted)
    bundle = _shared(saved_cache)
    assert calls == [chosen]
    assert read_json(bundle / "manifest.json")["complete"]


def test_shared_two_cpu_spawn_workers_keep_whole_records_and_import_without_load(
    saved_cache, monkeypatch
):
    def two_cpu(_):
        return [torch.device("cpu"), torch.device("cpu")]

    monkeypatch.setattr(base, "_resolve_theory_devices", two_cpu)
    monkeypatch.setattr(candidate, "_resolve_theory_devices", two_cpu)
    monkeypatch.setattr(candidate, "load_record", forbidden)
    bundle = _shared(saved_cache)
    execution = read_json(bundle / "manifest.json")["execution"]
    assert execution["worker_count"] == 2
    assert all(
        len(shard["assigned_indices"]) == 1 for shard in execution["device_shards"]
    )
    assert all(not shard["failed_rows"] for shard in execution["device_shards"])
    endpoint = _candidate(saved_cache, bundle)
    assert read_json(endpoint / "analysis_manifest.json")["endpoint_complete"]


def test_shared_option_requires_explicit_deferred_base(tmp_path):
    with pytest.raises(base.TheoryError, match="defer_path_integration"):
        base.run_theory(tmp_path, prepare_candidate_endpoints=True, **CONFIG)


def test_shared_contract_rejects_changed_endpoint_source_recipe(tmp_path, monkeypatch):
    candidate.prepare_shared_endpoint_extension(
        tmp_path,
        analysis_hash="a" * 64,
        config=CONFIG,
        support_metadata={"weights": [0.25, 0.75]},
        candidate_chunk_size=256,
        query_chunk_size=16,
        backend_types=["cpu"],
    )
    saved = candidate._shared_endpoint_contract(tmp_path)
    original = candidate._source_hashes

    def changed(names):
        return original(names) | {"candidate_feedback.py": "0" * 64}

    monkeypatch.setattr(candidate, "_source_hashes", changed)
    with pytest.raises(base.TheoryError, match="recipe changed"):
        candidate._shared_endpoint_contract(tmp_path)
    assert saved["source_code"]["candidate_feedback.py"] != "0" * 64
