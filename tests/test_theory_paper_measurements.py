"""Exact paper estimands, missingness, and independent terminal accounting."""

from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from utils.experiments.theory.candidate_summaries import (
    within_prompt_center,
    descriptive_trend,
)
from utils.experiments.theory.paper_measurements import (
    build_nonfeedback_plot_inputs,
    exact_weighted_ecdf,
    extend_terminal_accounting,
    repair_current_reference,
    terminal_affine_accounting,
    terminal_contract_status,
)
from tests.test_theory_reduction import CONFIG, saved_cache as saved_cache


def sample(prompt=0, seed=0, step=0, **updates):
    row = dict(
        run_id="run",
        original_index=prompt,
        record_id=f"r{prompt}",
        target_id=f"t{prompt}",
        candidate_target_atom_id=f"a{prompt}",
        seed=seed,
        step_index=step,
        timestep=999 - 20 * step,
        snr=0.01 * (step + 1),
        sigma=0.2,
        latent_dimension=4,
        terminal_sscd=0.8 if seed == 0 else 0.5,
        conditional_target_error_rmse=1.0 + seed,
        unconditional_target_error_rmse=2.0 + seed,
        branch_gap_rmse=1.0,
        candidate_current_reference_error_u_l2=2.0,
        candidate_unconditional_reference_error_l2=3.0,
        candidate_radius_l2=4.0,
        target_log_complement=-1.0,
        support_status="valid",
        initial_conditional_target_rank=1.0 + seed,
        initial_unconditional_target_rank=2.0 + seed,
        initial_conditional_target_tie_count=1,
        initial_unconditional_target_tie_count=2,
        terminal_A_rmse=0.2,
        terminal_B_rmse=0.3,
        terminal_rmse=0.4,
        candidate_terminal_bound_rmse=0.6,
        candidate_terminal_bound_applies_to_endpoint=False,
        last_prediction_error_gap_cosine=-0.5,
        A=0.5,
        kappa=0.5,
        destination_alpha=0.9,
        destination_sigma=0.1,
        destination_noise_std=0.0,
        affine_applicable=True,
        scheduler_status="valid_affine",
        independent_replay_rmse=1e-8,
        update_rounding_sensitivity_rmse=1e-6,
    )
    row.update(updates)
    return row


def tables_fixture(steps=50):
    trajectory = pd.DataFrame(
        [sample(p, s, k) for p in range(2) for s in range(2) for k in range(steps)]
    )
    initial = trajectory.loc[trajectory.step_index == 0].copy()
    initial["paired_initial_improvement_rmse"] = (
        initial.unconditional_target_error_rmse - initial.conditional_target_error_rmse
    )
    within = within_prompt_center(
        initial, "paired_initial_improvement_rmse", "terminal_sscd"
    )
    return dict(
        trajectory=trajectory,
        initial=initial,
        endpoint=trajectory.loc[trajectory.step_index == steps - 1].copy(),
        reference_initial=pd.DataFrame(
            {
                "run_id": ["run", "run"],
                "seed": [0, 1],
                "unconditional_center_rmse": [0.2, 0.4],
            }
        ),
        bank_geometry=pd.DataFrame(
            {
                "candidate_atom_id": ["a0", "a1", "a2"],
                "candidate_atom_center_distance_rmse": [1.0, 2.0, 3.0],
                "weight": [1 / 3, 1 / 3, 1 / 3],
            }
        ),
        summaries={
            "within_prompt_initial": within,
            "within_prompt_initial_trend": descriptive_trend(within),
        },
        manifest={
            "center_metadata": {
                "seeds": [2, 3],
                "evaluation_seeds": [0, 1],
                "vector_sha256": "center",
                "evaluation_repeated_prediction_max_abs_difference": 0.0,
                "evaluation_repeated_prediction_tolerance": 1e-3,
            }
        },
    )


def test_current_reference_tail_uses_saved_current_state_not_next_eligibility():
    frame = pd.DataFrame(
        [
            sample(step=48),
            sample(
                step=49,
                candidate_current_reference_error_u_l2=None,
                destination_sigma=0.0,
            ),
            sample(
                seed=1,
                step=49,
                candidate_current_reference_error_u_l2=None,
                candidate_unconditional_reference_error_l2=None,
            ),
        ]
    )
    original = frame.copy(deep=True)
    result, audit = repair_current_reference(frame)
    assert result.current_unconditional_candidate_reference_error_rmse.tolist()[:2] == [
        1.0,
        1.5,
    ]
    assert result.paper_current_reference_source.tolist() == [
        "saved_candidate_endpoint_current_reference",
        "saved_base_v3_current_reference",
        "unavailable",
    ]
    assert np.isnan(result.current_unconditional_candidate_reference_error_rmse.iloc[2])
    assert (
        result.paper_current_reference_status.iloc[2]
        == "missing_current_reference_measurement"
    )
    assert audit["last_prediction"]["step_index"] == 49
    pd.testing.assert_frame_equal(frame, original)


def test_zero_or_missing_current_noise_is_not_repaired_from_stale_finite_value():
    fixed, _ = repair_current_reference(
        pd.DataFrame([sample(sigma=0.0), sample(sigma=np.nan)])
    )
    assert fixed.paper_current_reference_error_u_l2.isna().all()


def test_exact_ecdf_preserves_ties_and_structural_seed_mass_with_missingness():
    result, meta = exact_weighted_ecdf(
        pd.DataFrame(
            [
                sample(0, 0, value=0.0),
                sample(0, 1, value=np.nan),
                sample(1, 0, value=2.0),
            ]
        ),
        "value",
    )
    assert result.value.tolist() == [0.0, 2.0]
    assert result.cdf.tolist() == [1 / 3, 1.0]
    assert meta["structural_weight"] == 2 and meta["missing_weight"] == 0.5
    tied, _ = exact_weighted_ecdf(
        pd.DataFrame(
            [sample(0, 0, value=2.0), sample(0, 1, value=2.0), sample(1, 0, value=5.0)]
        ),
        "value",
    )
    assert tied.value.tolist() == [2.0, 5.0] and tied.cdf.tolist() == [0.5, 1.0]


def test_fixed_joint_snapshots_are_exact_paired_ecdfs_with_shared_full_range():
    tables = tables_fixture()
    tail = (tables["trajectory"].step_index == 48) & (tables["trajectory"].seed == 1)
    tables["trajectory"].loc[tail, "unconditional_target_error_rmse"] = 1234.0
    frames, meta = build_nonfeedback_plot_inputs(
        tables, config={"num_seeds": 2, "num_inference_steps": 50}
    )
    early, late = (
        meta["figures"]["joint_target_recovery_" + name] for name in ("early", "late")
    )
    assert (
        early["snapshot"]["step_index"] == 10 and late["snapshot"]["step_index"] == 48
    )
    assert early["x_limits"] == late["x_limits"] == [0.0, 1234.0]
    assert frames["joint_target_recovery_late"].value.max() == 1234
    assert set(frames["initial_target_retrieval_rank"].branch) == {
        "conditional",
        "unconditional",
    }
    assert len(frames["initial_unconditional_concentration"]) == 5
    assert meta["figures"]["terminal_error_terms"]["status"] == "not_applicable"
    assert not meta["audit"]["blocking"]


def test_saved_within_prompt_fit_is_not_refit_and_degenerate_rows_remain():
    tables = tables_fixture()
    tables["summaries"]["within_prompt_initial_trend"].loc[
        lambda x: x.group == "all", ["slope", "intercept"]
    ] = [12.0, -3.0]
    frames, metadata = build_nonfeedback_plot_inputs(
        tables, config={"num_seeds": 2, "num_inference_steps": 50}
    )
    assert (
        metadata["figures"]["initial_recovery_within_prompt"]["trend"]["slope"] == 12.0
    )
    assert len(frames["initial_recovery_within_prompt"]) == len(tables["initial"])
    assert (
        frames["initial_recovery_within_prompt"].within_prompt_status
        == "no_usable_seed_variation"
    ).all()


def test_short_run_joint_snapshot_alias_and_duplicate_seed_rejection():
    _, meta = build_nonfeedback_plot_inputs(
        tables_fixture(2), config={"num_seeds": 2, "num_inference_steps": 2}
    )
    assert meta["figures"]["joint_target_recovery_late"]["status"] == "alias"
    tables = tables_fixture()
    tables["reference_initial"] = pd.concat(
        [tables["reference_initial"], tables["reference_initial"].iloc[:1]]
    )
    with pytest.raises(ValueError, match="one observation per unique"):
        build_nonfeedback_plot_inputs(
            tables, config={"num_seeds": 2, "num_inference_steps": 50}
        )


def test_terminal_structural_exclusion_is_distinct_from_missing_and_failed():
    rows = [
        sample(A=0.0, kappa=1.0, destination_alpha=1.0, destination_sigma=0.0),
        sample(),
        sample(
            A=0.0,
            kappa=1.0,
            destination_alpha=1.0,
            destination_sigma=0.0,
            destination_noise_std=1e-10,
            independent_replay_rmse=np.nan,
        ),
        sample(A=None),
        sample(independent_replay_rmse=1.0),
        sample(affine_applicable=False, scheduler_status="valid_nonlinear_drift"),
    ]
    result, audit = terminal_contract_status(pd.DataFrame(rows))
    assert result.paper_terminal_status.tolist() == [
        "clean_update_verified",
        "nonclean_affine_update",
        "nonclean_affine_update",
        "missing_contract_metadata",
        "failed_numeric_reconstruction",
        "unsupported_nonaffine_update",
    ]
    assert (
        result.paper_terminal_accounting_status.iloc[2]
        == "unavailable_saved_sampling_noise"
    )
    assert audit["blocking"]


def adapter(**updates):
    values = dict(A=0.5, B=0.4, noise_std=0.0, affine=True, alpha=0.8, sigma=0.6)
    values.update(updates)
    return SimpleNamespace(coefficients=lambda step: SimpleNamespace(**values))


def test_nonclean_accounting_includes_all_cross_terms_without_fitting_endpoint():
    z = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    eu = torch.tensor([[0.5, 0.2]], dtype=torch.float64)
    ec = torch.tensor([[-0.2, 0.7]], dtype=torch.float64)
    target = torch.tensor([0.1, -0.3], dtype=torch.float64)
    a = adapter()
    g = 2.0
    mu = (z - 0.6 * eu) / 0.8
    mc = (z - 0.6 * ec) / 0.8
    mg = mu + g * (mc - mu)
    out = 0.5 * z + 0.4 * mg
    result = terminal_affine_accounting(a, z, out, eu, ec, target, g, step=0)
    assert result["terminal_accounting_numeric_pass"].all()
    assert torch.allclose(
        result["terminal_accounting_error_squared_l2_from_terms"],
        (out - target).square().sum(-1),
        atol=1e-14,
        rtol=0,
    )
    changed = terminal_affine_accounting(a, z, out + 0.1, eu, ec, target, g, step=0)
    assert not changed["terminal_accounting_numeric_pass"].any()
    assert torch.equal(
        changed["terminal_accounting_term0_l2"], result["terminal_accounting_term0_l2"]
    )


def test_stochastic_accounting_rejects_noise_fitted_from_the_endpoint():
    z = torch.ones(1, 2, dtype=torch.float64)
    target = torch.zeros(2, dtype=torch.float64)
    a = adapter(A=0.0, B=1.0, noise_std=1e-10)
    result = terminal_affine_accounting(a, z, z, z, z, target, 2.0, step=0)
    assert result["terminal_accounting_status"] == "unavailable_saved_sampling_noise"
    with pytest.raises(ValueError, match="never an endpoint-fitted"):
        terminal_affine_accounting(
            a,
            z,
            z,
            z,
            z,
            target,
            2.0,
            step=0,
            realized_noise=torch.zeros_like(z),
            noise_provenance="recovered_innovation",
        )


def test_terminal_zero_error_ratios_and_incomplete_bound_comparison_are_explicit():
    tables = tables_fixture()
    endpoint = tables["endpoint"]
    endpoint.loc[:, ["A", "kappa", "destination_alpha", "destination_sigma"]] = [
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    endpoint.loc[:, "candidate_terminal_bound_applies_to_endpoint"] = True
    endpoint.loc[endpoint.index[0], "terminal_rmse"] = 0.0
    endpoint.loc[endpoint.index[1], "candidate_terminal_bound_rmse"] = np.nan
    frames, meta = build_nonfeedback_plot_inputs(
        tables, config={"num_seeds": 2, "num_inference_steps": 50}
    )
    details = meta["figures"]["terminal_bound_tightness"]
    assert details["comparison_status"] == "incomplete_candidate_bound_comparison"
    assert (
        details["counts"]["observable_triangle"]["undefined_zero_endpoint_error"] == 1
    )
    assert set(frames["terminal_bound_tightness"].distribution) == {
        "observable_triangle",
        "candidate_reference",
    }


def test_invalid_reference_norm_is_not_hidden_by_saved_fallback():
    rows, _ = repair_current_reference(
        pd.DataFrame(
            [
                sample(candidate_current_reference_error_u_l2=-1.0),
                sample(candidate_current_reference_error_u_l2=np.inf),
            ]
        )
    )
    assert rows.paper_current_reference_error_u_l2.isna().all()
    assert (
        rows.paper_current_reference_status == "invalid_saved_current_reference_norm"
    ).all()


def test_nullable_terminal_contract_reports_missing_metadata():
    rows, audit = terminal_contract_status(pd.DataFrame([sample(A=pd.NA)]))
    assert rows.paper_terminal_status.item() == "missing_contract_metadata"
    assert audit["blocking"]


def test_saved_center_consistency_and_disjoint_reference_seeds_are_required():
    tables = tables_fixture()
    tables["manifest"]["center_metadata"][
        "evaluation_repeated_prediction_max_abs_difference"
    ] = 1.0
    with pytest.raises(ValueError, match="consistency check"):
        build_nonfeedback_plot_inputs(
            tables, config={"num_seeds": 2, "num_inference_steps": 50}
        )
    tables = tables_fixture()
    tables["manifest"]["center_metadata"]["seeds"] = [0, 1]
    with pytest.raises(ValueError, match="seed provenance"):
        build_nonfeedback_plot_inputs(
            tables, config={"num_seeds": 2, "num_inference_steps": 50}
        )


def test_diagnostics_have_saved_all_curves_and_terminal_tolerance_units():
    tables = tables_fixture()
    endpoint = tables["endpoint"]
    endpoint.loc[:, ["A", "kappa", "destination_alpha", "destination_sigma"]] = [
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    for name in (
        "a_parallel",
        "off_target",
        "initial_target_coordinate_c",
        "initial_target_coordinate_g",
        "injection_relative_mismatch",
    ):
        tables["initial"][name] = 0.0
    frames, metadata = build_nonfeedback_plot_inputs(
        tables,
        config={
            "num_seeds": 2,
            "num_inference_steps": 50,
            "target_error_tolerance": 0.5,
        },
        diagnostics=True,
    )
    assert set(frames["synchronization_bound_components"].group) == {
        "all",
        "SSCD > 0.75",
        "SSCD <= 0.75",
    }
    assert set(frames["matched_update_residual"].group) == {
        "all",
        "SSCD > 0.75",
        "SSCD <= 0.75",
    }
    assert (
        metadata["figures"]["terminal_error_terms"]["independent_tolerance_rmse"]
        == 0.25
    )
    assert metadata["figures"]["target_synchronization"]["actual_initial_snr"] == 0.01


def _terminal_source_fixture(fixture):
    from utils.common.io import file_sha256
    from utils.experiments.theory.cache_reader import discover_sources

    sources = discover_sources(fixture["root"], **CONFIG)
    record = sources.selected[0]
    rows = pd.DataFrame(
        [
            sample(
                prompt=record.original_index,
                seed=seed,
                step=CONFIG["num_inference_steps"] - 1,
                run_id=sources.runs["experiment"]["scientific_config_hash"],
                record_id=record.metadata["record_id"],
            )
            for seed in (2, 0, 1)
        ]
    )
    manifest = {
        "config": CONFIG,
        "source_metadata_files": sources.metadata_files,
        "support_metadata": {
            "source_schedule_sha256": file_sha256(sources.experiment.schedule)
        },
    }
    return rows, manifest, sources.experiment, record.original_index


def test_archived_terminal_supplement_one_read_per_record_and_saved_reuse(
    saved_cache, monkeypatch
):
    from tests.test_theory_candidate_pipeline import protected_snapshot

    rows, manifest, paths, index = _terminal_source_fixture(saved_cache)
    before = protected_snapshot(saved_cache)
    source_paths = {
        paths.latent_path(index),
        paths.noise_prediction_path(index),
        paths.target_latent_path(index),
    }
    calls = []
    original_load = torch.load

    def counted(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path) in source_paths:
            calls.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", counted)
    result, receipt = extend_terminal_accounting(
        rows,
        manifest=manifest,
        source_logs=saved_cache["root"] / "logs",
        verify_payload_hashes=True,
    )
    assert len(calls) == 3 and set(calls) == source_paths
    assert result.seed.tolist() == [2, 0, 1]
    assert receipt["records_loaded"] == 1
    assert receipt["full_payload_hashes_verified"]
    assert len(receipt["records"][0]["terminal_slice_sha256"]) == 5
    assert result.terminal_accounting_numeric_pass.all()
    assert (result.terminal_accounting_status == "available").all()
    np.testing.assert_allclose(
        result.terminal_accounting_error_squared_l2_from_terms.astype(float),
        result.terminal_accounting_predicted_error_l2.astype(float) ** 2,
        atol=1e-13,
    )
    assert protected_snapshot(saved_cache) == before
    merged = rows.merge(result, validate="one_to_one")

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved terminal accounting must not reopen tensors")

    monkeypatch.setattr(torch, "load", forbidden)
    reused, repeated = extend_terminal_accounting(
        merged, manifest=manifest, source_logs=saved_cache["root"] / "logs"
    )
    pd.testing.assert_frame_equal(reused, result)
    assert repeated["status"] == "reused_saved_accounting"
    assert repeated["records_loaded"] == 0


def test_terminal_supplement_requires_exact_separately_saved_schedule_hash(
    saved_cache,
):
    rows, manifest, paths, _ = _terminal_source_fixture(saved_cache)
    logical_schedule = paths.schedule.relative_to(saved_cache["root"]).as_posix()
    assert logical_schedule not in manifest["source_metadata_files"]
    manifest["support_metadata"]["source_schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="metadata does not match.*schedule.pt"):
        extend_terminal_accounting(
            rows, manifest=manifest, source_logs=saved_cache["root"] / "logs"
        )


def test_stochastic_terminal_supplement_records_missing_noise_without_raw_reads(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unavailable independent noise requires no raw read")

    monkeypatch.setattr(torch, "load", forbidden)
    result, receipt = extend_terminal_accounting(
        pd.DataFrame([sample(step=2, destination_noise_std=0.1)]),
        manifest={"config": CONFIG},
        source_logs=tmp_path,
    )
    assert receipt["records_loaded"] == 0
    assert (
        result.terminal_accounting_status.item() == "unavailable_saved_sampling_noise"
    )


@pytest.mark.parametrize("guidance", [-2.0, 0.0, 0.5, 1.0])
def test_clean_update_does_not_override_terminal_theorem_guidance_domain(guidance):
    tables = tables_fixture()
    endpoint = tables["endpoint"]
    endpoint.loc[:, ["A", "kappa", "destination_alpha", "destination_sigma"]] = [
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    endpoint["terminal_B_rmse"] = (guidance - 1) * 0.2
    frames, metadata = build_nonfeedback_plot_inputs(
        tables,
        config={"num_seeds": 2, "num_inference_steps": 50, "guidance_scale": guidance},
        diagnostics=True,
    )
    contract = frames["terminal_contract"]
    assert contract.paper_terminal_clean_verified.all()
    assert not contract.paper_terminal_theorem_applicable.any()
    assert not metadata["audit"]["terminal"]["blocking"]
    for stem in (
        "terminal_error_terms",
        "terminal_bound_tightness",
        "last_prediction_error_terms",
        "terminal_cancellation",
    ):
        figure = metadata["figures"][stem]
        assert figure["status"] == "not_applicable"
        assert figure["reason"] == "inapplicable_manuscript_guidance_domain_g_gt_1"
        assert frames[stem].empty


def test_every_nonfeedback_figure_has_exact_units_and_population_weighting():
    from utils.experiments.theory.paper_measurements import FIGURE_AXIS_UNITS
    from utils.experiments.theory.paper_registry import LEGACY_CORE, LEGACY_TERMINAL, LEGACY_DIAGNOSTICS

    tables = tables_fixture()
    tables["reference_initial"]["initial_candidate_reference_movement_rmse"] = [
        0.1,
        0.2,
    ]
    tables["reference_initial"][
        "initial_unconditional_candidate_reference_error_rmse"
    ] = [0.2, 0.3]
    frames, metadata = build_nonfeedback_plot_inputs(
        tables, config={"num_seeds": 2, "num_inference_steps": 50}, diagnostics=True
    )
    expected = {
        entry["stem"]
        for entry in LEGACY_CORE + LEGACY_TERMINAL + LEGACY_DIAGNOSTICS
        if not entry["legacy_design_id"].startswith("PF")
    }
    assert set(FIGURE_AXIS_UNITS) == set(metadata["figures"]) == expected
    for name, item in metadata["figures"].items():
        assert item["axis_units"] == FIGURE_AXIS_UNITS[name]
        assert all(
            units in item["normalization"] for units in item["axis_units"].values()
        )
        assert item["weighting"]
    for name in (
        "initial_target_retrieval_rank",
        "terminal_bound_tightness",
        "initial_injection_geometry",
        "initial_target_coordinates",
        "initial_injection_mismatch",
        "terminal_cancellation",
    ):
        assert all(
            value.startswith("dimensionless")
            for value in metadata["figures"][name]["axis_units"].values()
        )
    within = metadata["figures"]["initial_recovery_within_prompt"]
    assert "centered difference" in within["axis_units"]["x"]
    assert "centered SSCD, dimensionless" in within["axis_units"]["y"]
    assert "usable within-prompt variation" in within["weighting"]
    assert "unaggregated point" in metadata["figures"]["initial_recovery"]["weighting"]
    for name in (
        "initial_candidate_reference_discrepancy",
        "initial_reference_snr_sweep",
    ):
        item = metadata["figures"][name]
        assert item["population_counts"]["rows"] == 2
        assert "unique" in item["weighting"] and "run/seed" in item["weighting"]
    populations = metadata["figures"]["initial_unconditional_concentration"][
        "population_counts"
    ]
    assert populations["initial_unconditional"]["rows"] == 2
    assert populations["candidate_atoms"]["rows"] == 3
    assert len(frames["initial_recovery"]) == 4
    for name in ("synchronization_bound_components", "matched_update_residual"):
        assert (
            "displayed metric comparison uses saved all-population curves"
            in metadata["figures"][name]["weighting"]
        )


def test_explicit_atom_ecdf_preserves_source_record_multiplicity():
    from utils.experiments.theory.paper_measurements import exact_weighted_ecdf
    atoms = pd.DataFrame({"distance": [1., 2., 2.], "weight": [.5, .25, .25]})
    curve, counts = exact_weighted_ecdf(atoms, "distance", weight_column="weight", distribution="candidate_atoms")
    assert curve.value.tolist() == [1., 2.]
    assert curve.cdf.tolist() == [.5, 1.]
    assert curve.weight.tolist() == [.5, .5]
    assert counts["structural_weight"] == 1. and counts["finite_weight"] == 1.


@pytest.mark.parametrize("bad_mass", [0., -1., np.nan, np.inf])
def test_explicit_atom_ecdf_rejects_invalid_saved_mass(bad_mass):
    from utils.experiments.theory.paper_measurements import exact_weighted_ecdf
    atoms = pd.DataFrame({"distance": [1., 2.], "weight": [.75, bad_mass]})
    with pytest.raises(ValueError, match="masses.*finite and positive"):
        exact_weighted_ecdf(atoms, "distance", weight_column="weight")


def test_legacy_initial_concentration_uses_saved_atom_masses_and_equal_gaussian_seed_weights():
    tables = tables_fixture()
    tables["bank_geometry"]["weight"] = [.5, .25, .25]
    frames, metadata = build_nonfeedback_plot_inputs(tables, config={"num_seeds": 2, "num_inference_steps": 50})
    frame = frames["initial_unconditional_concentration"]
    atoms = frame.loc[frame.distribution.eq("candidate_atoms")]
    gaussian = frame.loc[frame.distribution.eq("initial_unconditional")]
    assert atoms.cdf.tolist() == [.5, .75, 1.]
    assert gaussian.cdf.tolist() == [.5, 1.]
    assert "source-record multiplicity" in metadata["figures"]["initial_unconditional_concentration"]["weighting"]


def test_legacy_initial_concentration_refuses_missing_atom_mass_receipt():
    tables = tables_fixture()
    tables["bank_geometry"] = tables["bank_geometry"].drop(columns="weight")
    with pytest.raises(ValueError, match="weight"):
        build_nonfeedback_plot_inputs(tables, config={"num_seeds": 2, "num_inference_steps": 50})
