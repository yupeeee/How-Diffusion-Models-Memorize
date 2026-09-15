"""Source identity and provenance tests using real offline cache artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from tests.test_theory_reduction import CONFIG, saved_cache as _saved_cache

from utils.common.io import (
    atomic_torch_save,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
)
from utils.experiments.cache import GenerationCacheError
from utils.experiments.theory import cache_reader
from utils.experiments.theory.contracts import TheoryError, validate_source_metadata
from utils.experiments.theory.reduce import _science_only

saved_cache = _saved_cache


def _forbidden(*args, **kwargs):
    raise AssertionError(
        "Raw tensor deserialization is forbidden during metadata discovery"
    )


def _manifest_from_sources(fixture, sources):
    bundle = fixture["root"] / "metadata-only-audit"
    bundle.mkdir()
    atomic_write_json(
        bundle / "manifest.json",
        {
            "source_root": str(fixture["root"]),
            "source_metadata_files": sources.metadata_files,
            "source_marker_inventory": sources.marker_inventory,
        },
    )
    return bundle


def test_discovery_preserves_frozen_selection_without_tensor_deserialization(
    saved_cache, monkeypatch
):
    monkeypatch.setattr(torch, "load", _forbidden)
    monkeypatch.setattr(cache_reader, "safe_torch_load", _forbidden)
    sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    assert {record.original_index for record in sources.selected} == saved_cache[
        "selection"
    ].included_indices
    for record in sources.selected:
        assert record.metadata["seeds"] == [0, 1, 2]
        assert record.metadata["prompt_raw"] == "  duplicated raw prompt\n"
        frame = sources.proximity[
            sources.proximity.original_index.eq(record.original_index)
        ]
        assert sorted(frame.seed.tolist()) == [0, 1, 2]
        assert frame.sscd.min() < 0.05
    assert len({r.metadata["prompt_raw"] for r in sources.selected}) == 1
    assert len({r.original_index for r in sources.selected}) > 1


def test_experiment_outcomes_do_not_change_frozen_prompt_eligibility(saved_cache):
    path = saved_cache["proximity"] / "proximity.csv"
    frame = pd.read_csv(path, dtype={"original_index": str}, keep_default_na=False)
    # Discovery may inspect these scalar measurements, but cannot use them as
    # a new inclusion gate. Same-seed score validation is a separate later step.
    frame.loc[:, "sscd"] = -0.25
    frame.loc[:, "l2_norm"] = 1e6
    frame.to_csv(path, index=False)
    sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    assert {r.original_index for r in sources.selected} == saved_cache[
        "selection"
    ].included_indices
    assert all(r.metadata["seeds"] == [0, 1, 2] for r in sources.selected)


def test_discovery_rejects_unhashed_scientific_config_mutation(saved_cache):
    path = saved_cache["paths"]["experiment"].run_config
    config = read_json(path)
    config["scientific_config"]["model_revision"] = "d" * 40
    atomic_write_json(path, config)
    with pytest.raises(GenerationCacheError, match="scientific_config_hash differs"):
        cache_reader.discover_sources(saved_cache["root"], **CONFIG)


def test_discovery_rejects_duplicate_proximity_identity(saved_cache):
    path = saved_cache["proximity"] / "proximity.csv"
    frame = pd.read_csv(path, dtype={"original_index": str}, keep_default_na=False)
    pd.concat((frame, frame.iloc[:1]), ignore_index=True).to_csv(path, index=False)
    with pytest.raises(
        TheoryError, match="Duplicate protected proximity observation identity"
    ):
        cache_reader.discover_sources(saved_cache["root"], **CONFIG)


def test_discovery_rejects_missing_selected_generation_record(saved_cache):
    index = sorted(saved_cache["selection"].included_indices)[0]
    saved_cache["paths"]["experiment"].record_path(index).unlink()
    with pytest.raises(TheoryError, match="Selected experiment records missing"):
        cache_reader.discover_sources(saved_cache["root"], **CONFIG)


def test_discovery_rejects_target_payload_stale_against_own_marker(saved_cache):
    index = sorted(saved_cache["selection"].included_indices)[0]
    path = saved_cache["paths"]["experiment"].target_latent_path(index)
    with path.open("ab") as stream:
        stream.write(b"different target cache")
    with pytest.raises(TheoryError, match="Target-latent identity differs"):
        cache_reader.discover_sources(saved_cache["root"], **CONFIG)


def test_discovery_rejects_conflicting_paired_target_values_with_updated_hash(
    saved_cache,
):
    index = sorted(saved_cache["selection"].included_indices)[0]
    paths = saved_cache["paths"]["experiment"]
    path = paths.target_latent_path(index)
    target = torch.load(path, weights_only=True)
    updated_hash = atomic_torch_save(target + 1.0, path)
    marker = read_json(paths.record_path(index))
    marker["tensor_file_sha256"]["target_latent"] = updated_hash
    atomic_write_json(paths.record_path(index), marker)
    with pytest.raises(TheoryError, match="target.latent|target latent|Target.latent"):
        cache_reader.discover_sources(saved_cache["root"], **CONFIG)


@pytest.mark.parametrize(
    "key,value",
    [
        ("init_noise_sigma", 0.0),
        ("init_noise_sigma", float("nan")),
        ("trajectory_order", "image_to_noise"),
        ("scheduler_name", "ddpm"),
    ],
)
def test_saved_schedule_semantics_fail_before_reduction(saved_cache, key, value):
    sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    for paths in saved_cache["paths"].values():
        schedule = torch.load(paths.schedule, weights_only=True)
        schedule[key] = value
        # Intentionally bypass the valid-cache writer for malformed input.
        torch.save(schedule, paths.schedule)
    with pytest.raises(TheoryError):
        cache_reader.load_schedule(sources)


def test_root_plot_metadata_validation_never_reads_pt_payloads(
    saved_cache, monkeypatch
):
    sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    bundle = _manifest_from_sources(saved_cache, sources)
    before = file_sha256(bundle / "manifest.json")
    original_open = Path.open

    def metadata_open(path, *args, **kwargs):
        if path.suffix == ".pt":
            raise AssertionError(
                f"Plot metadata validation opened scientific tensor {path}"
            )
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_open)
    monkeypatch.setattr(torch, "load", _forbidden)
    validate_source_metadata(bundle, saved_cache["root"])
    assert file_sha256(bundle / "manifest.json") == before


def test_root_plot_detects_audit_only_metadata_change_without_tensor_reads(
    saved_cache, monkeypatch
):
    sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    bundle = _manifest_from_sources(saved_cache, sources)
    run_path = saved_cache["paths"]["experiment"].run_config
    config = read_json(run_path)
    config["created_at"] = "2030-01-01T00:00:00Z"
    atomic_write_json(run_path, config)
    monkeypatch.setattr(torch, "load", _forbidden)
    with pytest.raises(TheoryError, match="Stale/missing source metadata"):
        validate_source_metadata(bundle, saved_cache["root"])


def test_scientific_identity_excludes_incidental_timestamps_but_retains_tensor_hashes():
    first = {
        "created_at": "before",
        "completed_at": "before",
        "preview_completed_at": "before",
        "nested": {"duration_seconds": 1.2, "tensor_file_sha256": {"latent": "a" * 64}},
        "schedule": {"alpha": [0.1, 0.5], "native_prediction_type": "v_prediction"},
    }
    later = copy.deepcopy(first)
    later.update(created_at="later", completed_at="later", preview_completed_at="later")
    later["nested"]["duration_seconds"] = 55.0
    assert canonical_hash(_science_only(first)) == canonical_hash(_science_only(later))
    changed = copy.deepcopy(later)
    changed["nested"]["tensor_file_sha256"]["latent"] = "b" * 64
    assert canonical_hash(_science_only(first)) != canonical_hash(
        _science_only(changed)
    )
    changed = copy.deepcopy(later)
    changed["schedule"]["alpha"][0] = 0.2
    assert canonical_hash(_science_only(first)) != canonical_hash(
        _science_only(changed)
    )


def test_source_metadata_snapshot_retains_exact_original_file_hashes(saved_cache):
    sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    for relative, digest in sources.metadata_files.items():
        assert digest == file_sha256(saved_cache["root"] / relative)
        assert Path(relative).suffix in {".json", ".csv"}
    config_file = saved_cache["paths"]["experiment"].run_config
    original = config_file.read_bytes()
    # Same parsed scientific JSON can have a different audit file hash.
    config_file.write_text(json.dumps(read_json(config_file), separators=(",", ":")))
    assert config_file.read_bytes() != original
    assert (
        file_sha256(config_file)
        != sources.metadata_files[
            config_file.relative_to(saved_cache["root"]).as_posix()
        ]
    )


def test_paired_target_identity_accepts_identical_values_with_different_serialization(
    saved_cache,
):
    index = sorted(saved_cache["selection"].included_indices)[0]
    paths = saved_cache["paths"]["experiment"]
    path = paths.target_latent_path(index)
    target = torch.load(path, weights_only=True)
    # Legacy serialization differs byte-for-byte but preserves exact values.
    torch.save(target, path, _use_new_zipfile_serialization=False)
    marker = read_json(paths.record_path(index))
    marker["tensor_file_sha256"]["target_latent"] = file_sha256(path)
    atomic_write_json(paths.record_path(index), marker)
    sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    assert index in {record.original_index for record in sources.selected}


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "run", "initial", "final"]
)
def test_shard_validation_checks_exact_identity_and_state_prediction_grid(mutation):
    from types import SimpleNamespace
    from utils.experiments.theory.reduce import _validate_shard

    record = SimpleNamespace(
        original_index="00001", metadata={"record_id": "record-00001"}
    )
    rows = [
        dict(
            run_id="run-hash",
            original_index="00001",
            record_id="record-00001",
            seed=seed,
            step_index=step,
            is_initial=step == 0,
            is_final_update=step == 2,
            unconditional_target_error_rmse=1.0,
            conditional_target_error_rmse=2.0,
            target_id="target-hash",
            candidate_variation_l2=0.2,
            candidate_condition_margin_rmse=-0.5,
            candidate_log_probability_gain=-0.1,
            feedback_eligible=step < 2,
            sscd=0.1,
            branch_status="valid",
            paper_terminal_bound_l2=None,
        )
        for seed in range(3)
        for step in range(3)
    ]
    valid = pd.DataFrame(rows)
    _validate_shard(valid, record, 3, 3, "run-hash")
    if mutation == "missing":
        invalid = valid.iloc[1:].copy()
    elif mutation == "duplicate":
        invalid = pd.concat((valid, valid.iloc[:1]), ignore_index=True)
    else:
        invalid = valid.copy()
        if mutation == "run":
            invalid.loc[0, "run_id"] = "wrong-run"
        else:
            invalid.loc[
                0, "is_initial" if mutation == "initial" else "is_final_update"
            ] = mutation == "final"
    with pytest.raises(TheoryError):
        _validate_shard(invalid, record, 3, 3, "run-hash")


def test_audit_timestamp_refresh_reuses_numerical_bundle_and_refreshes_provenance(
    saved_cache, monkeypatch
):
    from utils.experiments.theory import reduce as reducer

    bundle = reducer.run_theory(saved_cache["root"], **CONFIG)
    manifest = read_json(bundle / "manifest.json")
    numerical_hashes = dict(manifest["numerical_files"])
    path = saved_cache["paths"]["experiment"].run_config
    config = read_json(path)
    config["created_at"] = "2030-01-01T00:00:00Z"
    atomic_write_json(path, config)
    monkeypatch.setattr(reducer, "_record_metrics", _forbidden)
    monkeypatch.setattr(reducer, "choose_center", _forbidden)
    assert reducer.run_theory(saved_cache["root"], **CONFIG) == bundle
    refreshed = read_json(bundle / "manifest.json")
    assert refreshed["source_metadata_history"][-1] == manifest["source_metadata_files"]
    assert refreshed["source_metadata_files"] != manifest["source_metadata_files"]
    assert refreshed["numerical_files"] == numerical_hashes
    assert all(
        file_sha256(bundle / relative) == digest
        for relative, digest in numerical_hashes.items()
    )
    validate_source_metadata(bundle, saved_cache["root"])
