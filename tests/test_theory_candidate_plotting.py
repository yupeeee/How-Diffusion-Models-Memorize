"""Candidate gallery tests use only small saved scalar fixtures."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from utils.experiments.theory import candidate_plotting as plotting
from utils.experiments.theory.candidate_registry import (
    candidate_registry,
    feedback_snapshots,
)
from utils.experiments.theory.candidate_summaries import build_candidate_summaries
from utils.experiments.theory.contracts import TheoryError, digest_file


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture(scope="module")
def candidate_bundle(tmp_path_factory):
    bundle = tmp_path_factory.mktemp("candidate-scalars") / ("a" * 64)
    bundle.mkdir()
    records = []
    doses = []
    for prompt in range(3):
        for seed in range(3):
            sign = 1 if seed == 1 else -1
            for step in range(12):
                gain = sign * (step + 1) * 0.2
                identity = dict(
                    run_id="fixture",
                    original_index=prompt,
                    record_id=f"record-{prompt}",
                    target_id=f"target-{prompt}",
                    seed=seed,
                )
                row = identity | dict(
                    step_index=step,
                    timestep=999 - step * 80,
                    destination_timestep=919 - step * 80,
                    snr=10 ** (-2 + step / 3),
                    destination_snr=10 ** (-2 + (step + 1) / 3),
                    destination_sigma=0.5 if step < 11 else 0,
                    kappa=0.1,
                    feedback_eligible=step < 11,
                    terminal_sscd=[-0.02, 0.85, 1.03][seed],
                    latent_dimension=4,
                    unconditional_target_error_rmse=0.2
                    + prompt * 0.01
                    + seed * 0.02
                    + 1 / (step + 1),
                    conditional_target_error_rmse=0.1 + seed * 0.01 + 0.1 / (step + 1),
                    joint_target_error_rmse=0.2 + 1 / (step + 1),
                    branch_gap_rmse=0.3 + 0.03 * seed,
                    independent_replay_rmse=1e-9,
                    update_rounding_sensitivity_rmse=1e-6,
                    matched_shift_rmse=0.08,
                    candidate_displacement_rmse=0.08,
                    candidate_current_reference_error_u_l2=0.6,
                    candidate_radius_l2=2.0,
                    target_log_complement=-1.0,
                    current_unconditional_candidate_reference_error_rmse=0.3,
                    candidate_radius_tail_rmse=0.2,
                    candidate_log_probability_gain=gain,
                    candidate_log_odds_gain=gain * 2,
                    candidate_normalized_log_odds_gain=gain / 4,
                    candidate_normalized_log_probability_gain=gain / 8,
                    candidate_matched_log_odds=-4 + step,
                    candidate_guided_log_odds=-4 + step + gain,
                    candidate_gain_status=("positive" if sign > 0 else "negative")
                    if step < 11
                    else "not_applicable",
                    candidate_profile_class=("increasing" if sign > 0 else "decreasing")
                    if step < 11
                    else "not_applicable",
                    candidate_current_alignment_cosine=sign * 0.4,
                    candidate_next_reference_target_contraction_rmse=sign * 0.1,
                    candidate_specificity_contrast=sign * 0.2,
                    initial_conditional_target_rank=1 + seed,
                    initial_unconditional_target_rank=3 + seed,
                    initial_conditional_target_tie_count=1,
                    initial_unconditional_target_tie_count=1,
                    initial_retrieval_bank_size=6,
                    a_parallel=2.0 + seed,
                    off_target=0.3,
                    initial_target_coordinate_c=0.6,
                    initial_target_coordinate_g=2.5,
                    injection_relative_mismatch=0.5,
                    last_prediction_error_gap_cosine=-0.7,
                    terminal_A_rmse=0.2,
                    terminal_B_rmse=0.3,
                    terminal_rmse=0.1,
                    candidate_terminal_bound_rmse=0.7,
                    candidate_terminal_bound_applies_to_endpoint=True,
                    terminal_clean_applicable=prompt != 2,
                    terminal_clean_status="clean"
                    if prompt != 2
                    else "nonzero_additive_noise",
                )
                for index, stem in enumerate(
                    ("original", "combined", "signed_error", "projected_variation")
                ):
                    row[f"candidate_margin_{stem}_status"] = (
                        "positive" if seed == 1 and index > 0 else "negative"
                    )
                    row[f"candidate_margin_{stem}_estimated_sign"] = (
                        "positive" if seed == 1 else "negative"
                    )
                records.append(row)
                if step < 11:
                    grid = np.unique(np.r_[np.linspace(0, 1, 21), 1 / 7.5])
                    doses.append(
                        identity
                        | dict(
                            step_index=step,
                            snr=row["snr"],
                            terminal_sscd=row["terminal_sscd"],
                            **{"lambda": grid.tolist()},
                            candidate_dose_log_probability_gain=(gain * grid).tolist(),
                        )
                    )
    trajectory = pd.DataFrame(records)
    initial = trajectory[trajectory.step_index == 0].copy()
    endpoint = trajectory[trajectory.step_index == 11].copy()
    baseline = pd.DataFrame(
        [
            dict(
                run_id="fixture",
                seed=seed,
                unconditional_center_rmse=0.01 * (seed + 1),
                initial_candidate_reference_movement_rmse=0.02 * (seed + 1),
                initial_unconditional_candidate_reference_error_rmse=0.03 * (seed + 1),
            )
            for seed in range(3)
        ]
    )
    sweep = pd.DataFrame(
        [
            dict(
                seed=seed,
                snr_grid_index=i,
                analytical_snr=0.01 * 10 ** (-i / 8),
                actual_initial_snr=0.01,
                candidate_reference_movement_rmse=0.02 * (seed + 1) * 10 ** (-i / 8),
            )
            for i in range(49)
            for seed in range(3)
        ]
    )
    bank = pd.DataFrame(
        dict(
            atom_index=range(6),
            atom_id=[f"a{i}" for i in range(6)],
            candidate_atom_center_distance_rmse=np.linspace(0.2, 1, 6),
            weight=np.ones(6) / 6,
        )
    )
    summaries = build_candidate_summaries(
        trajectory,
        initial=initial,
        endpoint=endpoint,
        dose=pd.DataFrame(doses),
        baseline=baseline,
        reference_sweep=sweep,
    )
    tables = {
        "initial.parquet": initial,
        "endpoint.parquet": endpoint,
        "reference_initial.parquet": baseline,
        "reference_snr.parquet": sweep,
        "bank_geometry.parquet": bank,
        "trajectory_metrics/part-0.parquet": trajectory,
    }
    tables.update(
        {"summaries/" + name + ".parquet": frame for name, frame in summaries.items()}
    )
    for name, frame in tables.items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    _write(bundle / "registry.json", candidate_registry())
    files = {
        path.relative_to(bundle).as_posix(): digest_file(path)
        for path in bundle.rglob("*")
        if path.is_file()
    }
    manifest = dict(
        schema_version=1,
        analysis_hash=bundle.name,
        endpoint_complete=True,
        complete=True,
        integration_status="complete",
        config=dict(
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=12,
            num_seeds=3,
        ),
        numerical_files=files,
        provenance={"candidate_bank_hash": "b" * 64, "selection_hash": "c" * 64},
    )
    _write(bundle / "analysis_manifest.json", manifest)
    return bundle


def test_registry_exact_inventory_and_meaning():
    registry = candidate_registry()
    expected = {
        f"{family}{number:02d}"
        for family, count in [
            ("IR", 4),
            ("UB", 4),
            ("IA", 3),
            ("MD", 2),
            ("PF", 12),
            ("TS", 5),
            ("TR", 4),
        ]
        for number in range(1, count + 1)
    }
    assert {card["id"] for card in registry["designs"]} == expected
    for card in registry["designs"]:
        assert all(
            card[key]
            for key in (
                "question",
                "formula",
                "evidence_tag",
                "scope",
                "required_columns",
                "affirmative_pattern_meaning",
                "does_not_establish",
            )
        )
    assert (
        next(card for card in registry["designs"] if card["id"] == "MD01")["namespace"]
        == "qa"
    )
    assert len([card for card in registry["designs"] if card["family"] == "PF"]) == 12


def test_snapshots_fixed_by_indices_not_gains():
    assert feedback_snapshots(range(49)) == [0, 1, 2, 4, 9, 24, 48]
    assert feedback_snapshots([3, 7, 11]) == [3, 7, 11]
    assert feedback_snapshots([]) == []


def test_renderer_has_no_scientific_imports():
    source = Path(plotting.__file__).read_text()
    parsed = ast.parse(source)
    imports = []
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        any(
            word in name
            for word in (
                "torch",
                "diffusers",
                "candidate_metrics",
                "support",
                "feedback",
                "cache_reader",
                "reduce",
            )
        )
        for name in imports
    )


def test_scatter_preserves_extreme_negative_values_and_tail_colors():
    frame = pd.DataFrame(
        {
            "x": [-100.0, 1.0, 3.0],
            "y": [-200.0, 5.0, 8.0],
            "terminal_sscd": [-0.1, 0.8, 1.1],
        }
    )
    fig, ax = plt.subplots()
    stats = plotting._scatter(
        ax,
        frame,
        "x",
        "y",
        color=True,
        equal=True,
        equality_label="Equal candidate evidence",
    )
    assert stats["finite_observations"] == 3
    assert stats["all_data_ranges"]["y"] == [-200, 8]
    assert ax.get_ylim()[0] < -200 and ax.get_xlim()[0] < -100
    assert ax.get_legend_handles_labels()[1] == ["Equal candidate evidence"]
    plt.close(fig)


def test_curve_limits_include_all_data_extrema():
    frame = pd.DataFrame(
        {
            "snr": [0.01, 0.1, 1.0],
            "median": [-1.0, 0.0, 1.0],
            "q25": [-2.0, -1.0, 0.0],
            "q75": [0.0, 1.0, 2.0],
            "minimum": [-100.0, -4.0, -1.0],
            "maximum": [1.0, 4.0, 200.0],
        }
    )
    fig, ax = plt.subplots()
    plotting._curve(ax, frame, "snr", "Median")
    assert ax.get_ylim()[0] < -100 and ax.get_ylim()[1] > 200
    plt.close(fig)


def test_plot_only_all_designs_and_reproducible_numerical_contract(
    candidate_bundle, monkeypatch
):
    # Saved plotting does not import a raw reader; installing raising sentinels
    # also protects accidental future calls through existing cache modules.
    from utils.experiments.theory import cache_reader, support, centers

    def forbidden(*args, **kwargs):
        raise AssertionError("Protected scientific reader called by plot-only")

    for module in (cache_reader, support, centers):
        for name in (
            "load_tensor",
            "load_record",
            "read_record",
            "load_support",
            "build_support",
            "fit_center",
        ):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    before = {
        path: (digest_file(path), path.stat().st_mtime_ns)
        for path in candidate_bundle.rglob("*")
        if path.is_file()
    }
    result = plotting.render_candidates(candidate_bundle)
    assert result["base_design_count"] == 34
    assert {card["candidate_id"] for card in result["candidates"]} == {
        card["id"] for card in candidate_registry()["designs"]
    }
    assert all(
        card["numeric_status"] == "available" for card in result["candidates"]
    ), [
        (card["variant_id"], card["numeric_status"])
        for card in result["candidates"]
        if card["numeric_status"] != "available"
    ]
    for path, state in before.items():
        assert (digest_file(path), path.stat().st_mtime_ns) == state
    for card in result["candidates"]:
        assert {"png", "pdf", "json"} <= card["files"].keys()
        sidecar = json.loads((candidate_bundle / card["files"]["json"]).read_text())
        assert sidecar["data_hashes"] and sidecar["provenance"]
    gallery = (candidate_bundle / "index.html").read_text()
    assert "https://" not in gallery and "<script src=" not in gallery
    assert "last_prediction_geometry" in gallery and "qa/MD01" in gallery
    comparison = pd.read_csv(candidate_bundle / "comparison_summary.csv")
    assert comparison.candidate_id.nunique() == 34
    baseline = json.loads((candidate_bundle / "UB/UB01.json").read_text())
    assert baseline["counts"]["source_rows"] == 3
    index = plotting.render_candidate_index(
        [candidate_bundle], candidate_bundle.parent / "cross"
    )
    assert index.exists()
    assert (
        pd.read_csv(index.parent / "comparison_summary.csv").candidate_id.nunique()
        == 34
    )


def test_unavailable_terminal_kept_and_geometry_separate(candidate_bundle):
    tables = plotting.read_tables(
        candidate_bundle, include_dose=False, include_controls=False
    )
    tables["endpoint"]["terminal_clean_applicable"] = False
    cards, _ = plotting._variant_cards(candidate_registry(), tables)
    tr = next(card for card in cards if card["variant_id"] == "TR01")
    status, reason = plotting._availability(tr, tables["endpoint"], tables)
    assert status == "not_applicable" and "clean-update" in reason
    geometry = next(
        card for card in cards if card["variant_id"] == "TR01__last_prediction_geometry"
    )
    assert (
        not geometry["clean_update"]
        and geometry["artifact_stem"] == "last_prediction_geometry/TR01"
    )


def test_incomplete_integration_preserves_endpoint_figures(candidate_bundle):
    tables = plotting.read_tables(
        candidate_bundle, include_dose=False, include_controls=False
    )
    tables["manifest"]["complete"] = False
    tables["manifest"]["integration_status"] = "pending"
    cards = {card["id"]: card for card in candidate_registry()["designs"]}
    assert (
        plotting._availability(cards["PF07"], tables["summaries"]["coverage"], tables)[
            0
        ]
        == "unavailable"
    )
    assert (
        plotting._availability(
            cards["PF01"], tables["summaries"]["timeseries"], tables
        )[0]
        == "available"
    )


def test_missing_scalar_schema_requires_recomputation(candidate_bundle):
    manifest = json.loads((candidate_bundle / "analysis_manifest.json").read_text())
    del manifest["numerical_files"]["summaries/dose.parquet"]
    original = (candidate_bundle / "analysis_manifest.json").read_text()
    _write(candidate_bundle / "analysis_manifest.json", manifest)
    try:
        with pytest.raises(
            TheoryError, match="--recompute-experiments --figure-suite candidates"
        ):
            plotting.validate_candidates(candidate_bundle)
    finally:
        (candidate_bundle / "analysis_manifest.json").write_text(original)


def test_heatmap_uses_all_population_not_last_group():
    rows = pd.DataFrame(
        [
            dict(
                group=group,
                progress_bin=0,
                regime_bin=0,
                positive_fraction=value,
                n_observations=30,
            )
            for group, value in [
                ("all", 0.4),
                ("SSCD > 0.75", 0.9),
                ("SSCD <= 0.75", 0.1),
            ]
        ]
    )
    card = next(
        card for card in candidate_registry()["designs"] if card["id"] == "PF11"
    )
    fig, ax = plt.subplots()
    plotting._draw(ax, card, rows, {})
    assert float(ax.images[0].get_array()[0, 0]) == 0.4
    plt.close(fig)


def test_recovery_distinguishes_missing_from_never():
    card = next(
        card for card in candidate_registry()["designs"] if card["id"] == "TS04"
    )
    frame = pd.DataFrame(
        {
            "conditional_first_step": [np.nan, np.nan, 1.0],
            "unconditional_first_step": [2.0, np.nan, 2.0],
            "conditional_status": ["right_censored", "unavailable", "reached"],
            "unconditional_status": ["reached", "unavailable", "reached"],
        }
    )
    fig, ax = plt.subplots()
    stats = plotting._draw(
        ax, card, frame, {"trajectory": pd.DataFrame({"step_index": range(4)})}
    )
    assert stats["conditional_censored"] == 1
    assert stats["conditional_unavailable"] == 1
    assert stats["rendered_observations"] == 3
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert "Never" in labels and "Missing" in labels
    plt.close(fig)


def test_empty_outcome_group_has_no_phantom_curve():
    card = next(
        card for card in candidate_registry()["designs"] if card["id"] == "PF03"
    )
    rows = []
    for group, fraction in [("SSCD > 0.75", np.nan), ("SSCD <= 0.75", 0.2)]:
        for status in ["positive", "numerically_unresolved"]:
            rows.append(
                {
                    "group": group,
                    "metric": "candidate_gain_status",
                    "step_index": 0,
                    "snr": 0.1,
                    "status": status,
                    "fraction": fraction,
                }
            )
    fig, ax = plt.subplots()
    plotting._coverage(ax, pd.DataFrame(rows), card)
    labels = ax.get_legend_handles_labels()[1]
    assert "SSCD > 0.75" not in labels
    assert "SSCD <= 0.75" in labels
    plt.close(fig)


def test_prompt_balanced_ecdf_reweights_usable_subset():
    frame = pd.DataFrame(
        {
            "run_id": ["r"] * 4,
            "original_index": [0, 0, 0, 1],
            "record_id": ["a", "a", "a", "b"],
            "target_id": ["ta", "ta", "ta", "tb"],
            "seed": [0, 1, 2, 0],
            "rank": [1.0, 1.0, -1.0, 3.0],
        }
    )
    fig, ax = plt.subplots()
    assert plotting._ecdf(ax, frame, "rank", "Rank") == 3
    values = ax.lines[0].get_ydata()
    assert values[-1] == pytest.approx(1.0)
    assert values[1] == pytest.approx(0.5)
    plt.close(fig)


@pytest.mark.parametrize(
    "status,expected",
    [
        ("zero_displacement", "not_applicable"),
        ("arithmetic_unresolved", "numerically_unresolved"),
    ],
)
def test_nonfinite_direction_status_is_not_generic_unavailable(status, expected):
    card = next(
        card for card in candidate_registry()["designs"] if card["id"] == "PF02"
    )
    tables = {
        "trajectory": pd.DataFrame(
            {"candidate_direction_normalization_status": [status]}
        )
    }
    assert plotting._nonfinite_status(card, tables)[0] == expected


def test_posterior_axes_identify_candidate_reference():
    for axes in plotting.CANDIDATE_AXIS_LABELS.values():
        assert any("candidate" in label.lower() for label in axes.values())


def test_empty_joint_tolerance_group_has_no_blank_available_figure():
    card = next(
        card for card in candidate_registry()["designs"] if card["id"] == "TS03"
    )
    card["group"] = "SSCD > 0.75"
    frame = pd.DataFrame(
        {
            "group": ["SSCD > 0.75"] * 3,
            "tolerance": [0.0, 0.5, 1.0],
            "fraction": [np.nan] * 3,
        }
    )
    fig, ax = plt.subplots()
    result = plotting._draw(ax, card, frame, {})
    assert result["finite_summary_cells"] == 0
    assert not ax.lines
    plt.close(fig)


def test_render_summarize_render_never_pins_renderer_artifacts(
    candidate_bundle, tmp_path, monkeypatch
):
    """A staged integration publication must not turn plot reports into measurements."""
    import shutil
    from matplotlib.figure import Figure
    from utils.experiments.theory import candidate_reduce, candidate_summaries

    bundle = tmp_path / candidate_bundle.name
    bundle.mkdir()
    metadata = json.loads((candidate_bundle / "analysis_manifest.json").read_text())
    for relative in metadata["numerical_files"]:
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_bundle / relative, destination)
    # Real producer bundles always own their numerical audit before plotting.
    _write(
        bundle / "numerical_audit.json",
        {"endpoint_population": 108, "integration_status": "pending"},
    )
    metadata["numerical_files"]["numerical_audit.json"] = digest_file(
        bundle / "numerical_audit.json"
    )
    metadata["eligible_steps"] = list(range(11))
    _write(bundle / "analysis_manifest.json", metadata)
    summaries = plotting.read_tables(
        bundle, include_dose=False, include_controls=False
    )["summaries"]
    monkeypatch.setattr(
        candidate_summaries,
        "build_candidate_summaries",
        lambda *args, **kwargs: summaries,
    )
    # Full real image exports are tested above. This regression exercises all
    # renderer paths and publication ownership without repeating rasterization.
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda self, path, **kwargs: Path(path).write_bytes(
            b"%PDF fixture" if Path(path).suffix == ".pdf" else b"PNG fixture"
        ),
    )
    first = plotting.render_candidates(bundle)
    assert "candidate_render_audit.json" in first["files"]
    updated = candidate_reduce._summarize(
        bundle, metadata, complete=True, integration_status="complete"
    )
    assert not (set(first["files"]) & set(updated["numerical_files"]))
    assert "candidate_figure_manifest.json" not in updated["numerical_files"]
    assert "candidate_render_audit.json" not in updated["numerical_files"]
    assert "comparison_summary.csv" not in updated["numerical_files"]
    assert "numerical_audit.json" in updated["numerical_files"]
    second = plotting.render_candidates(bundle)
    assert second["base_design_count"] == 34
    assert len(second["candidates"]) == len(first["candidates"])
    plotting.validate_candidates(bundle)


def test_large_snapshot_log_odds_have_no_colliding_shared_multiplier():
    card = next(
        card for card in candidate_registry()["designs"] if card["id"] == "PF04"
    )
    frame = pd.DataFrame(
        {
            "candidate_matched_log_odds": [-4e6, -1e6],
            "candidate_guided_log_odds": [-4e6 + 3, -1e6 - 2],
            "terminal_sscd": [0.2, 0.8],
            "step_index": [48, 48],
        }
    )
    fig, ax = plt.subplots()
    plotting._draw(ax, card, frame, {"manifest": {"config": {"guidance_scale": 7.5}}})
    fig.canvas.draw()
    assert ax.xaxis.get_offset_text().get_text() == ""
    assert ax.yaxis.get_offset_text().get_text() == ""
    assert any("e6" in tick.get_text() for tick in ax.get_xticklabels())
    assert all("e+" not in tick.get_text() for tick in ax.get_xticklabels())
    plt.close(fig)


def test_summary_ranges_keep_plotted_group_metric_identity_and_do_not_pool():
    card = next(
        card for card in candidate_registry()["designs"] if card["id"] == "PF01"
    )
    frame = pd.DataFrame(
        {
            "group": ["all", "SSCD > 0.75", "SSCD <= 0.75"],
            "metric": ["candidate_log_probability_gain"] * 3,
            "snr": [0.1] * 3,
            "median": [999.0, 2.0, -5.0],
            "q25": [900.0, 1.0, -6.0],
            "q75": [1000.0, 3.0, -4.0],
            "minimum": [0.0, -20.0, -30.0],
            "maximum": [9999.0, 20.0, 30.0],
        }
    )
    records = plotting._displayed_summary_ranges(card, frame, {})
    assert {row["group"] for row in records} == {"SSCD > 0.75", "SSCD <= 0.75"}
    assert {row["metric"] for row in records} == {"candidate_log_probability_gain"}
    assert {row["y_min"] for row in records} == {2.0, -5.0}
    assert all("not a pooled effect" in row["scope"] for row in records)
    assert {row["all_data_minimum"] for row in records} == {-20.0, -30.0}


def test_provenance_uses_real_top_level_manifest_schema_without_mutating_it():
    manifest = {
        "analysis_hash": "a" * 64,
        "config": {"model_name": "sdv2", "scheduler_name": "ddim"},
        "base_bundle": "/saved/base",
        "base_analysis_hash": "b" * 64,
        "endpoint_hash": "c" * 64,
        "integration_hash": "d" * 64,
        "source_root": "/saved/source",
        "source_metadata_files": {"metadata.json": "e" * 64},
        "source_marker_inventory": {"done": ["record.json"]},
        "source_code": {"candidate_reduce.py": "f" * 64},
        "center_metadata": {"estimator": "reference-initial"},
        "support_metadata": {"bank_hash": "0" * 64},
        "scheduler_adapter": {"method": "saved_affine_update"},
        "manuscript_sha256": "1" * 64,
    }
    before = json.dumps(manifest, sort_keys=True)
    provenance = plotting._provenance(manifest, "2" * 64)
    for key, value in manifest.items():
        if key not in {
            "analysis_hash",
            "source_metadata_files",
            "source_marker_inventory",
            "support_metadata",
        }:
            assert provenance[key] == value
    for key in ("source_metadata_files", "source_marker_inventory"):
        assert provenance[key] == plotting._manifest_reference(manifest[key], key)
    assert provenance["support_metadata"]["bank_hash"] == "0" * 64
    assert provenance["support_metadata"][
        "full_inventory_reference"
    ] == plotting._manifest_reference(manifest["support_metadata"], "support_metadata")
    assert provenance["analysis_manifest"] == {
        "path": "../analysis_manifest.json",
        "sha256": "2" * 64,
    }
    provenance["support_metadata"]["bank_hash"] = "changed"
    assert json.dumps(manifest, sort_keys=True) == before


def test_provenance_compacts_bank_inventories_with_reproducible_hashes():
    import hashlib

    support = {
        "atom_ids": ["target-a", "target-b"],
        "aliases": {"cached-target": "target-a"},
        "weights": [0.5, 0.5],
        "support_size": 2,
        "tensor_sha256": "a" * 64,
        "weight_definition": "predeclared_uniform_unique_atoms",
    }
    actual = plotting._provenance({"support_metadata": support})["support_metadata"]
    assert actual["support_size"] == 2
    assert actual["tensor_sha256"] == "a" * 64
    assert actual["weight_definition"] == "predeclared_uniform_unique_atoms"
    for key in ("atom_ids", "aliases", "weights"):
        assert key not in actual
        reference = actual[key + "_reference"]
        payload = json.dumps(
            support[key],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        assert reference["sha256"] == hashlib.sha256(payload).hexdigest()
        assert reference["manifest_key"] == "support_metadata." + key
        assert reference["entry_count"] == len(support[key])
        assert reference["hash_algorithm"] == "SHA-256"
        assert "sorted keys" in reference["serialization"]


def test_baseline_populations_and_exact_saved_tolerance_grid():
    registry = candidate_registry()
    by_id = {card["id"]: card for card in registry["designs"]}
    assert "distinct candidate-bank atom" in by_id["UB02"]["population"]
    assert "unique evaluation run and initial-noise seed" in by_id["UB03"]["population"]
    assert "all 49 prespecified analytical SNR values" in by_id["UB04"]["population"]
    assert (
        "Equal weight per unique cached initial-noise seed"
        in by_id["UB04"]["weighting"]
    )
    tables = {
        "manifest": {
            "numerical_files": {"summaries/joint_tolerance.parquet": "f" * 64}
        },
        "summaries": {
            "joint_tolerance": pd.DataFrame(
                {"tolerance": [0.2, 0.0, 0.2, 0.07, np.nan]}
            )
        },
    }
    grid = plotting._presentation_metadata(tables)["joint_tolerance_grid"]
    assert grid["values"] == [0.0, 0.07, 0.2]
    assert grid["source_sha256"] == "f" * 64
    assert grid["source_table"] == "summaries/joint_tolerance.parquet"
