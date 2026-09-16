"""Paper path, identity, CSV and whole-role transactional contracts."""

import json

import numpy as np
import pandas as pd
import pytest

from utils.common.io import atomic_write_json, file_sha256
from utils.common.cli import generation_cache_parent_name, generation_cache_namespace
from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory import paper_contracts as c
from tests.test_theory_paper_plotting import compact_fixture


@pytest.mark.parametrize(
    "model,scheduler,g,T,N",
    [
        ("sdv1", "ddim", 7.5, 50, 20),
        ("sdv1", "ddpm", -2.5, 3, 4),
        ("realvis", "ddim", 0, 1, 1),
        ("sdv2", "ddim", 5, 20, 3),
    ],
)
def test_paths_match_shared_generation_and_proximity_convention(
    tmp_path, model, scheduler, g, T, N
):
    identity = dict(
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=g,
        num_inference_steps=T,
        num_seeds=N,
    )
    expected = (
        tmp_path
        / "outputs"
        / generation_cache_parent_name(**identity)
        / "theory"
        / generation_cache_namespace(**identity, seed_start=0)
    )
    assert c.PaperPaths.build(tmp_path, **identity).output_directory == expected
    with pytest.raises(TheoryError):
        c.PaperPaths.build(tmp_path, seed_start=N, **identity)


def test_csv_exact_round_trip_with_json_sorted_schema_and_integer_extremes(tmp_path):
    frame = pd.DataFrame(
        {
            "z": [np.nextafter(0.1, 0), -1e250, np.nan],
            "record_id": ["NA", "000001", "null"],
            "signed_integer": pd.array([2**63 - 1, -(2**63), None], dtype="Int64"),
            "status": ["NA", "unresolved", "positive"],
        }
    )
    path = tmp_path / "input.csv"
    spec = c.write_plot_table(frame, path)
    atomic_write_json(tmp_path / "schema.json", spec)
    read = c.read_plot_table(path, json.loads((tmp_path / "schema.json").read_text()))
    assert list(read.record_id) == ["NA", "000001", "null"]
    assert list(read.status) == list(frame.status)
    np.testing.assert_array_equal(read.z, frame.z)
    pd.testing.assert_series_equal(read.signed_integer, frame.signed_integer)


def test_scientific_and_recipe_mismatches_require_analysis(tmp_path):
    compact_fixture(tmp_path / "bundle")
    bundle = tmp_path / "bundle"
    with pytest.raises(TheoryError, match="configuration differs"):
        c.load_paper_inputs(
            bundle,
            expected_config=dict(
                model_name="sdv1", scheduler_name="ddim", num_seeds=2, center="zero"
            ),
        )
    config = json.loads((bundle / "run_config.json").read_text())
    first_source = next(iter(config["scientific_identity"]["measurement_sources"]))
    config["scientific_identity"]["measurement_sources"][first_source] = "stale"
    from utils.common.io import canonical_hash

    config["scientific_hash"] = canonical_hash(config["scientific_identity"])
    atomic_write_json(bundle / "run_config.json", config)
    with pytest.raises(TheoryError, match="measurement definitions changed"):
        c.load_paper_inputs(bundle)


def test_transaction_rollback_archive_lock_and_symlink_rejection(tmp_path):
    output = tmp_path / "theory" / "experiment_S0_N2"
    compact_fixture(output)
    before = {
        p.relative_to(output): file_sha256(p) for p in output.rglob("*") if p.is_file()
    }
    with c.publication_lock(output):
        with pytest.raises(TheoryError, match="Another theory"):
            with c.publication_lock(output):
                pass
        with pytest.raises(RuntimeError):
            with c.staged_publication(output) as stage:
                (stage / "summary.json").write_text("broken")
                raise RuntimeError("injected failure before publication")
    assert before == {
        p.relative_to(output): file_sha256(p) for p in output.rglob("*") if p.is_file()
    }
    with c.publication_lock(output):
        with c.staged_publication(output, archive_previous=True) as stage:
            (stage / "user-note.txt").write_text("kept")
    assert (output / "user-note.txt").read_text() == "kept"
    archives = list((output.parent / ".archives" / output.name).iterdir())
    assert len(archives) == 1
    receipt = json.loads((archives[0] / "archive_receipt.json").read_text())
    assert receipt["original_path"] == str(output)
    (tmp_path / "unsafe").symlink_to(output, target_is_directory=True)
    with pytest.raises(TheoryError, match="symbolic-link"):
        c.contained_path(tmp_path / "unsafe", "x.png")


def test_crash_marker_cannot_remove_active_role(tmp_path):
    output = tmp_path / "experiment_S0_N2"
    output.mkdir()
    marker = tmp_path / ("." + output.name + ".transaction.json")
    atomic_write_json(marker, {"backup": output.name, "stage": output.name})
    with pytest.raises(TheoryError, match="Unsafe theory transaction"):
        with c.publication_lock(output):
            pass
    assert output.exists()


def test_legacy_retirement_preserves_numerical_and_unrecognized_files(tmp_path):
    base = tmp_path / "outputs" / "sdv1_ddim_g7.5_T50_N20"
    source = base / "theory_candidates" / ("a" * 64)
    source.mkdir(parents=True)
    (source / "IR").mkdir()
    (source / "IR" / "IR01.png").write_bytes(b"owned-png")
    (source / "IR" / "IR01.pdf").write_bytes(b"owned-pdf")
    (source / "user.png").write_bytes(b"keep")
    (source / "initial.parquet").write_bytes(b"scientific-data")
    images = {
        name: file_sha256(source / name) for name in ["IR/IR01.png", "IR/IR01.pdf"]
    }
    atomic_write_json(source / "candidate_figure_manifest.json", {"files": images})
    output = base / "theory" / "experiment_S0_N20"
    output.mkdir(parents=True)
    manifest = {"analysis_hash": source.name}
    before = file_sha256(source / "initial.parquet")
    first = c.retire_legacy_figures(tmp_path, output, source, manifest)
    assert first["archived_count"] == 2
    archive = c.contained_path(
        tmp_path, str(c.Path(first["archive"]).relative_to(tmp_path))
    )
    for name, digest in images.items():
        assert not (source / name).exists()
        assert file_sha256(archive / (source / name).relative_to(tmp_path)) == digest
    receipt = archive / "manifest.json"
    stamp = receipt.stat().st_mtime_ns
    second = c.retire_legacy_figures(tmp_path, output, source, manifest)
    assert second["archived_count"] == 2
    assert receipt.stat().st_mtime_ns == stamp
    assert file_sha256(source / "initial.parquet") == before
    assert (source / "user.png").read_bytes() == b"keep"


def test_plot_preflight_never_hashes_backing_or_audit_csvs(tmp_path, monkeypatch):
    bundle = tmp_path / "paper"
    compact_fixture(bundle)
    summary = json.loads((bundle / "summary.json").read_text())
    (bundle / "audit_data").mkdir()
    large = bundle / "audit_data" / "backing.csv"
    large.write_text("raw,analysis\n1,2\n")
    summary["numerical_files"]["audit_data/backing.csv"] = file_sha256(large)
    atomic_write_json(bundle / "summary.json", summary)
    original = c.file_sha256

    def guarded(path):
        if c.Path(path) == large:
            raise AssertionError("Plot preflight hashed a backing audit")
        return original(path)

    monkeypatch.setattr(c, "file_sha256", guarded)
    c.load_paper_inputs(bundle)
    monkeypatch.setattr(c, "file_sha256", original)
    c.audit_paper_integrity(bundle)
    large.write_text("corrupt")
    c.load_paper_inputs(bundle)
    with pytest.raises(TheoryError, match="analysis artifact"):
        c.audit_paper_integrity(bundle)


def test_stage_links_immutable_csv_and_replacement_cannot_mutate_active(tmp_path):
    output = tmp_path / "paper"
    compact_fixture(output)
    table = output / "plot_data" / "initial_loss_recovery.csv"
    before = file_sha256(table)
    with c.publication_lock(output):
        with pytest.raises(RuntimeError):
            with c.staged_publication(output) as stage:
                linked = stage / table.relative_to(output)
                assert linked.stat().st_ino == table.stat().st_ino
                c.write_plot_table(pd.DataFrame({"x": [999]}), linked)
                assert file_sha256(table) == before
                raise RuntimeError("rollback")
    assert file_sha256(table) == before


def test_interrupted_staging_is_recovered_without_touching_active(tmp_path):
    output = tmp_path / "paper"
    compact_fixture(output)
    digest = file_sha256(output / "summary.json")
    stage = tmp_path / ".paper.stage-interrupted"
    stage.mkdir()
    (stage / "partial.png.tmp").write_bytes(b"not published")
    marker = tmp_path / ".paper.transaction.json"
    atomic_write_json(
        marker,
        {
            "phase": "staging",
            "stage": stage.name,
            "backup": ".paper.backup-unused",
            "archive_previous": False,
        },
    )
    with c.publication_lock(output):
        assert not stage.exists()
        assert not marker.exists()
        assert file_sha256(output / "summary.json") == digest


def test_nullable_boolean_objects_remain_flags_in_compact_csv(tmp_path):
    frame = pd.DataFrame({
        "direct_prop5_nonnegative_condition": pd.Series([True, None, False], dtype=object),
        "direct_theorem7_condition_certified": pd.Series([False, True, None], dtype=object),
        "record_id": ["True", "False", "NA"],
    })
    path = tmp_path / "nullable.csv"
    spec = c.write_plot_table(frame, path)
    assert spec["schema"]["direct_prop5_nonnegative_condition"] == "boolean"
    assert spec["schema"]["direct_theorem7_condition_certified"] == "boolean"
    restored = c.read_plot_table(path, spec)
    assert str(restored.direct_prop5_nonnegative_condition.dtype) == "boolean"
    assert restored.direct_prop5_nonnegative_condition.iloc[0]
    assert pd.isna(restored.direct_prop5_nonnegative_condition.iloc[1])
    assert not restored.direct_prop5_nonnegative_condition.iloc[2]
    assert restored.record_id.tolist() == ["True", "False", "NA"]
