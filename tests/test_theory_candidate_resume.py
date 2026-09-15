"""Repairable scalar/segment caches and actual spawned candidate workers."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
import torch

from tests.test_theory_candidate_pipeline import protected_snapshot
from tests.test_theory_reduction import CONFIG, _reduce, saved_cache as saved_cache
from utils.common.io import (
    atomic_write_frame_parquet,
    atomic_write_json,
    file_sha256,
    read_json,
)
from utils.experiments.theory import candidate_reduce as discovery
from utils.experiments.theory import candidate_summaries
from utils.experiments.theory import progress as progress_ui
from utils.experiments.theory.candidate_contracts import (
    ROW_KEYS,
    SCHEMA_VERSION,
    read_tables,
    selected_suite,
    validate_candidate_bundle,
)
from utils.experiments.theory.contracts import TheoryError


def _run(fixture, base, **kwargs):
    return discovery.run_candidates(
        fixture["root"],
        base_bundle=base,
        render=False,
        device="cpu",
        candidate_chunk_size=2,
        query_chunk_size=2,
        **kwargs,
        **CONFIG,
    )


def _stub_unrelated_summary_families(monkeypatch):
    # These tests exercise real record reads, NPZ payloads, stage stamps,
    # integrations and publication. Dedicated summary and pipeline tests cover
    # the gallery estimands; avoid recomputing all of them at each repair step.
    def counts(trajectory, **kwargs):
        return {"fixture_population": pd.DataFrame({"row_count": [len(trajectory)]})}

    monkeypatch.setattr(candidate_summaries, "build_candidate_summaries", counts)


def test_missing_npz_repairs_only_affected_endpoint_then_integrates_without_raw_reread(
    saved_cache, monkeypatch
):
    _stub_unrelated_summary_families(monkeypatch)
    base = _reduce(saved_cache)
    protected_before = protected_snapshot(saved_cache)
    bundle = _run(saved_cache, base, endpoint_only=True)
    cache = Path(
        read_json(bundle / "analysis_manifest.json")["integration_payload_cache"]
    )
    selected = sorted(saved_cache["selection"].included_indices)
    missing = selected[0]
    segment = cache / "segments" / f"part-{missing}.npz"
    assert segment.is_file()
    intact = {
        index: (
            file_sha256(cache / "segments" / f"part-{index}.npz"),
            (cache / "completion/endpoints" / f"{index}.json").stat().st_mtime_ns,
        )
        for index in selected[1:]
    }
    segment.unlink()
    reads = []
    original = discovery.load_record

    def counted(paths, record, **kwargs):
        reads.append(record.original_index)
        return original(paths, record, **kwargs)

    monkeypatch.setattr(discovery, "load_record", counted)
    repaired = _run(saved_cache, base, endpoint_only=True)
    assert repaired == bundle
    assert reads == [missing]
    assert segment.is_file()
    stamp = read_json(cache / "completion/endpoints" / f"{missing}.json")
    assert stamp["files"][segment.relative_to(cache).as_posix()] == file_sha256(segment)
    for index, (digest, mtime) in intact.items():
        assert file_sha256(cache / "segments" / f"part-{index}.npz") == digest
        assert (
            cache / "completion/endpoints" / f"{index}.json"
        ).stat().st_mtime_ns == mtime

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Repaired integration must use segment payloads, not raw trajectories"
        )

    monkeypatch.setattr(discovery, "load_record", forbidden)
    complete = _run(saved_cache, base)
    assert complete == bundle
    assert validate_candidate_bundle(complete, require_complete=True)["complete"]
    assert protected_snapshot(saved_cache) == protected_before


def _scalar_bundle(tmp_path, left, right):
    """A fully hashed tiny scalar bundle; join validation is exercised unchanged."""
    bundle = tmp_path / "scalar_bundle"
    bundle.mkdir()
    for name in (
        "initial",
        "endpoint",
        "reference_initial",
        "reference_snr",
        "bank_geometry",
    ):
        atomic_write_frame_parquet(
            pd.DataFrame({"fixture": [1]}), bundle / f"{name}.parquet"
        )
    atomic_write_json(bundle / "registry.json", {"fixture": True})
    atomic_write_frame_parquet(left, bundle / "trajectory_metrics/part-record.parquet")
    atomic_write_frame_parquet(right, bundle / "integration_shards/part-record.parquet")
    files = {
        p.relative_to(bundle).as_posix(): file_sha256(p)
        for p in bundle.rglob("*")
        if p.is_file()
    }
    atomic_write_json(
        bundle / "analysis_manifest.json",
        dict(
            schema_version=SCHEMA_VERSION,
            endpoint_complete=True,
            complete=False,
            config=CONFIG,
            numerical_files=files,
        ),
    )
    return bundle


def _identity(seed=0):
    return dict(
        run_id="run-a",
        original_index="00001",
        target_id="target-a",
        seed=seed,
        step_index=0,
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ("orphan", "orphan sample identities"),
        ("duplicate", "Duplicate candidate sample identities"),
        ("overwrite", "overwrites endpoint measurements"),
    ],
)
def test_saved_scalar_join_rejects_orphans_duplicates_and_endpoint_overwrites(
    tmp_path, change, match
):
    left = pd.DataFrame([_identity() | {"candidate_log_probability_gain": -2.0}])
    right = pd.DataFrame(
        [_identity() | {"candidate_margin_original_status": "negative"}]
    )
    if change == "orphan":
        right.loc[0, "seed"] = 999
    elif change == "duplicate":
        right = pd.concat([right, right], ignore_index=True)
    else:
        right["candidate_log_probability_gain"] = 100.0
    bundle = _scalar_bundle(tmp_path, left, right)
    with pytest.raises(TheoryError, match=match):
        read_tables(bundle)


def test_partial_integration_join_retains_endpoint_rows_with_explicit_status(tmp_path):
    left = pd.DataFrame(
        [
            _identity(seed) | {"candidate_log_probability_gain": -float(seed + 1)}
            for seed in (0, 1)
        ]
    )
    right = pd.DataFrame(
        [
            _identity()
            | {
                "candidate_margin_original_l2": -3.0,
                "candidate_margin_original_status": "negative",
                "candidate_margin_original_estimated_sign": "negative",
                "candidate_integration_reason": "available",
            }
        ]
    )
    joined = read_tables(_scalar_bundle(tmp_path, left, right))[
        "trajectory"
    ].sort_values("seed")
    assert len(joined) == 2
    assert joined.candidate_log_probability_gain.tolist() == [-1.0, -2.0]
    assert joined.candidate_margin_original_status.tolist() == [
        "negative",
        "not_applicable",
    ]
    assert joined.candidate_margin_original_estimated_sign.tolist() == [
        "negative",
        "not_applicable",
    ]
    assert pd.isna(joined.candidate_margin_original_l2.iloc[1])
    assert (
        joined.candidate_integration_reason.iloc[1]
        == "endpoint_segment_not_applicable_or_unavailable"
    )


def test_two_actual_cpu_workers_use_round_robin_and_one_progress_bar_per_stage(
    saved_cache, monkeypatch
):
    _stub_unrelated_summary_families(monkeypatch)
    base = _reduce(saved_cache)
    protected_before = protected_snapshot(saved_cache)
    bars = []
    original = progress_ui.tqdm

    def capture(**kwargs):
        bar = original(**kwargs, file=StringIO())
        bars.append(bar)
        return bar

    monkeypatch.setattr(progress_ui, "tqdm", capture)
    monkeypatch.setattr(
        discovery,
        "_resolve_theory_devices",
        lambda request: (torch.device("cpu"), torch.device("cpu")),
    )
    bundle = _run(saved_cache, base)
    manifest = validate_candidate_bundle(bundle, require_complete=True)
    selected = sorted(saved_cache["selection"].included_indices)
    assert manifest["execution"]["resolved_devices"] == ["cpu", "cpu"]
    assert len(bars) == 2
    for bar in bars:
        assert bar.n == bar.total == len(selected)
        assert "devices=2" in bar.postfix
        assert "failed=0" in bar.postfix
    for stage in ("endpoint_workers", "integration_workers"):
        reports = manifest["execution"][stage]
        assert len(reports) == 2
        assert len({report["pid"] for report in reports}) == 2
        for index, report in enumerate(reports):
            assert report["assigned_indices"] == selected[index::2]
            assert report["device"] == "cpu"
            assert report["cpu_threads"] >= 1
            assert not report["failed_rows"]
    data = read_tables(bundle)
    assert len(data["initial"]) == len(selected) * CONFIG["num_seeds"]
    assert (
        len(data["trajectory"]) == len(data["initial"]) * CONFIG["num_inference_steps"]
    )
    assert not data["trajectory"].duplicated(ROW_KEYS).any()
    assert selected_suite(saved_cache["root"], CONFIG) == "candidates"
    assert protected_snapshot(saved_cache) == protected_before
