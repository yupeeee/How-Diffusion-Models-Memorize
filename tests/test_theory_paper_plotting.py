"""Fixed paper exports use saved compact inputs and the shared publisher."""

import re

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from utils.common.io import atomic_write_json, canonical_hash, file_sha256
from utils.experiments import plotting as shared
from utils.experiments.theory import paper_plotting as plotting
from utils.experiments.theory.contracts import TheoryError, numerical_config
from utils.experiments.theory.paper_contracts import (
    BUNDLE_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION,
    load_paper_inputs,
    measurement_sources,
    write_plot_table,
)
from utils.experiments.theory.paper_registry import (
    GROUPS, REGISTRY_VERSION, PREVIOUS_REGISTRY_VERSION, paper_registry,
    previous_paper_registry, _ZERO_BASELINE_FIGURES,
)


def compact_fixture(root, *, terminal=False, diagnostics=False, previous_registry=False, saved_prompt_curve=False, saved_grouped_coverage=False, saved_zero_baselines=False, saved_prompt_reference_curves=False, saved_reference_gap_curve=False, saved_guidance_fit=False, saved_reference_variation=False):
    """Represent already-reduced measurements, including a signed extreme tail."""
    root.mkdir(parents=True, exist_ok=True)
    scientific = numerical_config(model_name="sdv1", scheduler_name="ddim", num_seeds=2, mean_source="cached-targets")
    mean_receipt = {"source": "declared_finite_bank_mean", "vector_sha256": "fixture-bank-mean",
                    "sample_count": None, "mean_norm_rmse": .4, "mean_norm_l2": .8}
    identity = {"config": scientific, "measurement_sources": measurement_sources()}
    digest = canonical_hash(identity)
    config = dict(
        schema_version=BUNDLE_SCHEMA_VERSION,
        metric_schema_version=METRIC_SCHEMA_VERSION,
        scientific_config=scientific,
        scientific_identity=identity,
        scientific_hash=digest,
        registry_version=PREVIOUS_REGISTRY_VERSION if previous_registry else REGISTRY_VERSION,
        provenance={"center_metadata": {"vector_sha256": "abc"},
                    "reference_law": {"theory_mean": mean_receipt}},
    )
    figures, specs, hashes = {}, {}, {}
    mean_stems = {entry["stem"] for entry in paper_registry() if entry.get("requires_theory_mean")}
    registry = previous_paper_registry(diagnostics) if previous_registry else paper_registry(diagnostics)
    if previous_registry and saved_prompt_curve:
        # Historical exports can coexist with newly saved compact analysis data.
        registry.append(next(entry for entry in paper_registry() if entry["stem"] == "branch_gap_per_prompt"))
    if previous_registry and saved_grouped_coverage:
        grouped = next(entry for entry in paper_registry() if entry["stem"] == "terminal_bound_coverage")
        registry = [grouped if entry["stem"] == grouped["stem"] else entry for entry in registry]
    if previous_registry and saved_zero_baselines:
        registry.extend(_ZERO_BASELINE_FIGURES)
    if previous_registry and saved_prompt_reference_curves:
        registry.extend(entry for entry in paper_registry() if entry["stem"] in {
            "conditional_reference_error_per_prompt", "unconditional_reference_error_per_prompt",
            "target_probability_per_prompt"})
    if previous_registry and saved_reference_gap_curve:
        registry.append(next(entry for entry in paper_registry() if entry["stem"] == "reference_branch_gap_per_prompt"))
    if previous_registry and saved_guidance_fit:
        registry.append(next(entry for entry in paper_registry() if entry["stem"] == "corollary3_guidance_scale_vs_loss"))
    if previous_registry and saved_reference_variation:
        registry.append(next(entry for entry in paper_registry() if entry["stem"] == "reference_variation_per_prompt"))
    for entry in registry:
        stem = entry["stem"]
        if entry.get("conditional_terminal") and not terminal:
            figures[stem] = {
                "status": "not_applicable",
                "reason": "nonclean_affine_update",
                "status_counts": {"nonclean_affine_update": 4},
            }
            continue
        metadata = dict(
            status="available",
            reason=None,
            actual_initial_snr=0.01,
            counts={"rows": 4, "prompts": 2, "seeds": 2},
            trend={"slope": 0.03, "intercept": 0.01},
            symlog_linthresh=0.02,
        )
        if stem in mean_stems:
            metadata.update(theory_mean=dict(mean_receipt), theory_mean_source=mean_receipt["source"],
                            theory_mean_sha256=mean_receipt["vector_sha256"], mean_norm_rmse=.4,
                            mean_norm_l2=.8, bank_mean_norm_rmse=.4, bank_mean_norm_l2=.8,
                            mean_offset_rmse=0., mean_offset_l2=0.)
        if entry.get("comparison_centre") == "zero":
            metadata.update(comparison_centre="zero", declared_law_mean_unchanged=True,
                            mean_norm_rmse=.4, mean_norm_l2=.8, formula_version="zero-baseline-1")
        if entry["kind"] == "pair_loss":
            frame = pd.DataFrame(dict(
                x=[.1, .2, 1., 4.], y=[.3, .4, 1.1, 3.],
                control_y=[.8, 1., 2., 4.], mean_terminal_sscd=[-.1, .5, .8, 1.1],
                x_low=[.08, .15, .8, 3.5], x_high=[.12, .25, 1.2, 4.5],
                y_low=[.2, .3, 1., 2.5], y_high=[.4, .5, 1.2, 3.5],
                record_id=["NA", "0001", "1e5", "null"],
            ))
            metadata["counts"] = {"pairs": 4, "gaussian_seeds_per_pair": 2, "forward_draws": 64}
        elif entry["kind"] == "four_guidance_fit":
            frame = pd.DataFrame(dict(
                run_id=["run"] * 4, original_index=["0", "1", "2", "3"],
                record_id=["NA", "0001", "1e5", "null"], target_id=["target" + str(k) for k in range(4)],
                x=[0., .2, 1., 4.], y=[-1., .5, 7.5, 9.], mean_terminal_sscd=[.1, .5, .8, 1.],
                seed_count=[2] * 4, seed_ids_json=['[0, 1]'] * 4, latent_dimension=[4] * 4,
                snr=[.01] * 4, guidance_scale=[scientific["guidance_scale"]] * 4,
                fit_residual_rmse=[.1, .2, .3, .4], direction_norm_rmse=[1., 1.5, .5, 2.]))
            metadata.update(
                formula_version="corollary3-guidance-fit-1", guidance_scale=scientific["guidance_scale"],
                expected_seed_count=2, expected_seed_ids=[0, 1],
                aggregation="joint_no_intercept_least_squares_shared_target_direction",
                fit_definition="argmin_a sum_seed ||(hat{x}_T(c;g)-mu)-a(x_star-mu)||^2",
                color_population="mean terminal SSCD over exactly the same complete evaluation seed cohort",
                normalization="x=forward loss mean/(d*SNR_T), no square root; y=signed fitted coefficient",
                cohort_audit_table=entry["cohort_audit_table"],
                counts={"pairs": 4, "plotted_pairs": 4, "excluded_pairs": 0, "seeds_per_pair": 2, "rows": 4})
        elif entry["kind"] == "four_response":
            grid = sorted(set([k/40 for k in range(41)] + [1/scientific["guidance_scale"]]))
            frame = pd.DataFrame([
                dict(group=group, s=s, median=sign*s, q25=sign*s-.05*s, q75=sign*s+.05*s,
                     minimum=sign*s-.1*s, maximum=sign*s+.1*s, denominator_weight=1., eligible_count=2)
                for group, sign in zip(GROUPS, (1., -2.)) for s in grid])
            metadata.update(dose_grid=grid, guidance_scale=scientific["guidance_scale"], symlog_linthresh=1e-3,
                            cfg_endpoint_label="Reconstructed matched CFG")
        elif entry["kind"] == "four_chronological":
            metrics = [("gap", .2), ("joint_error", .5)] if stem == "branch_gap_synchronization" else (
                [("conditional", .3), ("unconditional", .5)] if stem == "branch_target_errors" else
                [("gap", .2), ("joint_error", .5), ("bound", 1.)])
            frame = pd.DataFrame([
                dict(group=group, metric=metric, step_index=k, snr=snr, median=value,
                     q25=value*.8, q75=value*1.2, minimum=value*.7, maximum=value*1.3,
                     denominator_weight=1., segment_id=0)
                for group in GROUPS for metric, value in metrics
                for k, snr in [(0, .01), (1, .1), (2, 10.)]])
            metadata["prediction_steps"] = 3
        elif entry["kind"] == "four_prompt_chronological":
            value_column = entry.get("value_column", "mean_gap_rmse")
            domain = entry.get("value_domain", "nonnegative")
            transition_only = entry.get("prediction_domain") == "positive_noise_transitions"
            frame = pd.DataFrame([
                dict(run_id="run", original_index=str(pair), record_id=str(pair), target_id="target" + str(pair),
                     step_index=k, **{value_column: value}, mean_terminal_sscd=sscd,
                     seed_count=2, seed_ids_json='[0, 1]', latent_dimension=4, cohort_complete=True)
                for pair, sscd in ((0, .2), (1, .9))
                for k, value in enumerate((.1 * pair, .4 + .1 * pair, .9 + .1 * pair)
                                         if domain == "probability" else
                                         (.1 + pair, .3 + pair) if transition_only else (.1 + pair, .3 + pair, .05 + pair))])
            metadata.update(prediction_steps=3, expected_seed_count=2, expected_seed_ids=[0, 1],
                            value_column=value_column, value_domain=domain,
                            aggregation=("arithmetic_mean_of_saved_seed_probabilities" if domain == "probability"
                                         else "arithmetic_mean_of_saved_seed_norms"),
                            color_population="mean terminal SSCD over exactly the same complete evaluation seed cohort",
                            cohort_audit_table=entry["cohort_audit_table"],
                            counts={"pairs": 2, "plotted_pairs": 2, "excluded_pairs": 0, "seeds_per_pair": 2, "rows": len(frame)})
            if transition_only:
                metadata.update(
                    prediction_domain="positive_noise_transitions",
                    manuscript_domain="Equation 15: t=2,...,T; chronological steps 0,...,T-2; t=1 is excluded",
                    numerical_scope="Original saved numerical estimates; unresolved condition signs or quadrature budget do not certify exact integrals",
                    numerical_status_counts={"direct_prop5_integral_status": {"estimated_converged": 4, "numerically_unresolved": 4}},
                    measurement_audit_table="audit_data/feedback_endpoints.csv",
                    reference_definition="Current unconditional posterior clean reference bar{x}_t(empty), not selected global mu")
        elif entry["kind"] == "four_peak":
            frame = pd.DataFrame([
                dict(group=group, step_index=k, fraction=value, denominator_weight=1., sample_count=1)
                for group in GROUPS for k, value in [(0, .1), (1, .2), (2, .5)]])
            metadata.update(prediction_steps=3, shape_group_counts={group: {
                "samples": 5, "resolved_peak_count": 4, "unassigned_weight_fraction": .2,
                "status_counts": {"resolved_peak": 4, "flat": 1}} for group in GROUPS})
        elif entry["kind"] == "four_motion":
            group = entry["outcome_group"]
            frame = pd.DataFrame([
                dict(group=group, metric=metric, step_index=k, mean=value, minimum=value-.1,
                     maximum=value+.1, denominator_weight=1.)
                for metric, value in [("conditional", .3), ("unconditional", -.1), ("quadratic", .04), ("change", .24)]
                for k in (0, 1)])
            metadata.update(prediction_steps=3, group=group)
        elif entry["kind"] in {"four_terminal_scatter", "four_counterfactual"}:
            scope = "original_clean_terminal_theorem" if terminal else "finite_terminal_update_extension_deterministic"
            frame = pd.DataFrame(dict(x=[0., .2, 1., 4.], y=[0., .1, .8, 3.], terminal_sscd=[-.1, .5, .8, 1.1],
                                      applicable=[True]*4, terminal_scope=[scope]*4, step_index=[0]*4,
                                      counterfactual_I_net=[0., .1, .2, 1.]))
            metadata.update(axis_scale="linear", terminal_scope=scope, original_clean_counts={"applicable": 4 if terminal else 0, "total": 4},
                            zero_counts={"x": 1, "y": 1, "both": 1}, manuscript_extension_required=not terminal)
        elif entry["kind"] in {"reference_convergence", "reference_native_sweep", "four_reference"}:
            native_sweep = entry["kind"] in {"reference_native_sweep", "four_reference"}
            rows = []
            for metric in ("reference", "learned", "reference_error"):
                for k, snr in enumerate((.01, .1, 10.) if native_sweep else (.01,)):
                    value = {"reference": .5, "learned": .7, "reference_error": .3}[metric] + k
                    rows.append(dict(metric=metric, source_range="native" if native_sweep else "native_initial", segment_id=0, step_index=k,
                                     snr=snr, median=value, q25=value*.8, q75=value*1.2, minimum=value*.7, maximum=value*1.3))
            if not native_sweep or entry["kind"] == "four_reference":
                rows += [dict(metric="reference", source_range="analytical", segment_id=0, step_index=-1,
                              snr=snr, median=v, q25=v*.8, q75=v*1.2, minimum=v*.7, maximum=v*1.3)
                         for snr, v in ((1e-8, .0005), (1e-5, .02), (.01, .5))]
            frame = pd.DataFrame(rows)
            metadata["reference_scale_rmse"] = 1.
            metadata["native_sweep_figure"] = "appendix/lemma2_native_gaussian_sweep"
        elif entry["kind"] == "injection_geometry":
            frame = pd.DataFrame(dict(x=[-1., .5, 1., 4.], y=[.3, .8, 0., 3.],
                                      terminal_sscd=[.1, .5, .8, 1.], marker_class=["observed"]*4))
        elif entry["kind"] == "feedback_fractions":
            frame = pd.DataFrame([
                dict(group=group, metric=metric, step_index=k, snr=snr, denominator_weight=1.,
                     fraction=f if metric == "feedback" else 0.,
                     upper_fraction=min(1., f+.1) if metric == "feedback" else .1,
                     unresolved_count=1, eligible_count=10)
                for group in GROUPS for metric in ("feedback", "condition")
                for k, snr, f in [(0, .01, .2), (1, .1, .5), (2, 10., .1)]
            ])
            metadata["condition_zero_overlap"] = True
            metadata["condition_positive_counts"] = {group: 0 for group in GROUPS}
            metadata["condition_unresolved_counts"] = {group: 3 for group in GROUPS}
        elif entry["kind"] == "numerical_resolution":
            frame = pd.DataFrame([
                dict(group=group, metric=metric, step_index=k, snr=snr,
                     resolved_fraction=resolved, unknown_fraction=1-resolved,
                     denominator_weight=1., unresolved_count=1)
                for group in GROUPS
                for metric in ("feedback", "condition", "source_robust_feedback")
                for k, snr, resolved in [(0, .01, .3), (1, .1, .8), (2, 10., .6)]
            ])
            metadata["numerical_resolution_table"] = "audit_data/proposition5_numerical_causes.csv"
        elif entry["kind"] == "synchronization_curves":
            frame = pd.DataFrame([
                dict(group=group, metric=metric, step_index=k, snr=snr, median=v,
                     q25=v*.8, q75=v*1.2, minimum=v*.7, maximum=v*1.3)
                for group in GROUPS for metric, v in [("joint_error", .5), ("gap", .2), ("bound", 1.)]
                for k, snr in [(0, .01), (1, .1), (2, 10.)]
            ])
        elif entry["kind"] == "four_terminal_grouped_cdf":
            names = ("actual", "observable", "reference")
            frame = pd.DataFrame([
                dict(group=group, distribution=name, value=value*scale*(index+1), cdf=cdf,
                     denominator_weight=2., eligible_count=4, prompt_count=2)
                for group, scale in zip(GROUPS, (1., 2.))
                for index, name in enumerate(names)
                for value, cdf in [(.2, .25), (1., .75), (4., 1.)]
            ])
            metadata.update(
                formula_version="terminal-coverage-by-sscd-1",
                grouped_zero_mass={group: {name: 0. for name in names} for group in GROUPS},
                grouped_infinite_mass={group: {name: 0. for name in names} for group in GROUPS},
                group_population={group: dict(denominator_weight=2., eligible_count=4, prompt_count=2) for group in GROUPS},
                empty_groups=[], group_common_population_table="audit_data/terminal_grouped_common_population.csv",
                group_sscd_exclusion_table="audit_data/terminal_grouped_sscd_exclusions.csv",
                pooled_cdf_audit_table="audit_data/terminal_pooled_cdf.csv",
                pooled_metadata_audit_table="audit_data/terminal_pooled_cdf_metadata.csv",
                terminal_scope="original_clean_terminal_theorem" if terminal else "finite_terminal_update_extension_deterministic",
                original_clean_counts={"applicable": 8 if terminal else 0, "total": 8},
                manuscript_extension_required=not terminal,
                counts={"total": 8, "common_eligible": 8, "eligible": 8, "excluded": 0, "excluded_before_sscd_grouping": 0, "missing_or_nonfinite_sscd": 0})
        elif entry["kind"] in {"terminal_cdf", "terminal_components"}:
            components = entry["kind"] == "terminal_components"
            key = "component" if components else "distribution"
            names = ("conditional", "guidance", "scheduler") if components else ("actual", "observable", "reference")
            frame = pd.DataFrame([
                {key: name, "value": value, "cdf": cdf}
                for name in names for value, cdf in [(.2, .25), (1., .75), (4., 1.)]
            ])
            metadata.update(zero_mass={name: 0. for name in names},
                            terminal_scope="original_clean_terminal_theorem" if terminal else "finite_terminal_update_extension_deterministic",
                            original_clean_counts={"applicable": 4 if terminal else 0, "total": 4},
                            manuscript_extension_required=not terminal)
        elif entry["kind"] == "scatter":
            frame = pd.DataFrame(
                dict(
                    x=[-1.0, 0.2, 1.0, 4.0],
                    y=[0.3, 0.4, 1.1, 3.0],
                    terminal_sscd=[-0.1, 0.5, 0.8, 1.1],
                    record_id=["NA", "0001", "1e5", "null"],
                    seed=["0", "1", "0", "1"],
                    marker_class=["observed"] * 4,
                    applicable=[True] * 4,
                )
            )
            if stem == "lemma4_matched_displacement":
                frame["marker_class"] = ["independently_checked", "constructed"] * 2
                frame["direct_lemma4_verification_source"] = [
                    "independent_deterministic_affine_counterfactual",
                    "constructed_shared_innovation_from_saved_endpoint_not_independent",
                ] * 2
            if entry.get("terminal_applicability"):
                frame["applicable"] = terminal
                frame["marker_class"] = "observed" if terminal else "inapplicable"
                metadata["clean_update_prerequisite"] = {"applicable": 4 if terminal else 0, "total": 4}
            if entry.get("frequency_intervals"):
                frame["x"] = [.1, .3, .7, 1.]
                frame["y"] = [.05, .2, .5, .8]
                frame["frequency_low"] = [0., .1, .4, .7]
                frame["frequency_high"] = [.1, .3, .6, .9]
        elif entry["kind"] == "ecdf":
            records = []
            if "distribution" in entry["required_columns"]:
                names = (
                    ["observable_triangle", "candidate_reference"]
                    if entry.get("conditional_terminal")
                    else ["initial_unconditional", "candidate_atoms"]
                )
                identities = [dict(distribution=name) for name in names]
            else:
                identities = [
                    dict(
                        group=group,
                        **({"branch": branch} if entry.get("branches") else {}),
                    )
                    for group in GROUPS
                    for branch in (
                        ["conditional", "unconditional"]
                        if entry.get("branches")
                        else [None]
                    )
                ]
            for identity_row in identities:
                records += [
                    dict(value=x, cdf=y, **identity_row)
                    for x, y in [(0.2, 0.25), (1.0, 0.75), (4.0, 1.0)]
                ]
            frame = pd.DataFrame(records)
            if entry.get("shared_limits"):
                metadata.update(
                    x_limits=[0.0, 4.5],
                    y_limits=[0.0, 1.0],
                    snapshot={
                        "step_index": 10 if stem.endswith("early") else 48,
                        "snr": 0.1,
                    },
                )
        elif entry["kind"] == "fraction":
            frame = pd.DataFrame(
                [
                    dict(
                        group=group, step_index=k, snr=snr, fraction=f, upper_fraction=u
                    )
                    for group in GROUPS
                    for k, snr, f, u in [
                        (0, 0.01, 0.2, 0.4),
                        (10, 0.1, 0.5, 0.5),
                        (48, 10.0, 0.1, 0.8),
                    ]
                ]
            )
        elif entry["kind"] == "coverage":
            frame = pd.DataFrame(
                [
                    dict(
                        group="All",
                        margin=margin,
                        step_index=k,
                        snr=snr,
                        resolved_positive_fraction=0.1,
                        estimated_strict_positive_fraction=0.5,
                        estimated_nonnegative_fraction=0.6,
                        stable_gain_positive_fraction=0.4,
                    )
                    for margin in [
                        "original",
                        "combined",
                        "signed_error",
                        "projected_variation",
                    ]
                    for k, snr in [(0, 0.01), (10, 0.1), (48, 10.0)]
                ]
            )
        else:
            metric_branches = {
                "lemma2_unconditional_baseline": ["reference", "learned", "reference_error"],
                "synchronization_bound_components": [
                    "gap",
                    "conditional",
                    "unconditional_reference",
                    "candidate_radius_tail",
                ],
                "matched_update_residual": [
                    "independent_residual",
                    "source_precision_envelope",
                ],
            }
            curve_groups = ["all"] if stem in metric_branches else GROUPS
            frame = pd.DataFrame(
                [
                    dict(
                        group=group,
                        branch=branch,
                        metric=branch,
                        step_index=k,
                        snr=snr,
                        analytical_snr=snr,
                        median=v,
                        q25=v - 0.05,
                        q75=v + 0.05,
                        minimum=-100.0 if entry.get("signed") and k == 10 else v - 0.1,
                        maximum=v + 0.1,
                        **{"lambda": k / 48},
                    )
                    for group in curve_groups
                    for branch in (
                        ["conditional", "unconditional"]
                        if entry.get("branches")
                        else metric_branches.get(stem, ["gap"])
                    )
                    for k, snr, v in [
                        (0, 0.01, 0.2),
                        (10, 0.1, -0.4 if entry.get("signed") else 0.5),
                        (48, 10.0, 0.1),
                    ]
                ]
            )
        spec = write_plot_table(frame, root / entry["source_table"])
        spec["path"] = entry["source_table"]
        specs[stem] = spec
        hashes[entry["source_table"]] = spec["sha256"]
        figures[stem] = metadata
    summary = dict(
        schema_version=1,
        complete=True,
        scientific_hash=digest,
        figures=figures,
        plot_data=specs,
        numerical_files=hashes,
    )
    for filename, value in [
        ("run_config.json", config),
        ("summary.json", summary),
        ("audit.json", dict(schema_version=1, blocking=False, scientific_hash=digest)),
    ]:
        atomic_write_json(root / filename, value)
    return root


def test_fixed_registry_has_exact_paper_selection():
    entries = paper_registry()
    assert len([e for e in entries if e["category"] == "main"]) == 4
    assert (
        len(
            [
                e
                for e in entries
                if e["category"] == "appendix" and not e.get("conditional_terminal")
            ]
        )
        == 16
    )
    assert not any(e.get("conditional_terminal") for e in entries)
    assert not any(e.get("comparison_centre") == "zero" for e in entries)
    assert all(len(e["outputs"]) == 2 for e in entries)
    assert all(
        "\\frac" not in str(e["axes"]) and "Empirical" not in str(e["axes"]) for e in entries
    )
    assert all("mu_ref" not in e["formula"] for e in entries)
    assert all("\n" not in label for e in entries for label in e["axes"].values())


def test_real_exports_single_page_single_axes_and_no_numeric_mutation(
    tmp_path, monkeypatch
):
    root = compact_fixture(tmp_path / "paper", terminal=True)
    initial_files = {
        p: (file_sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }
    seen = []
    real_publish = plotting.publish_figures

    def capture(output, items, **options):
        for figure, names in items:
            assert len([a for a in figure.axes if a.get_label() != "<colorbar>"]) == 1
            assert list(figure.get_size_inches()) == [4.0, 4.0]
            seen.append(names)
        real_publish(output, items, **options)

    monkeypatch.setattr(plotting, "publish_figures", capture)

    def forbidden(*args, **kwargs):
        raise AssertionError("saved-only rendering touched a noncompact reader")

    import torch

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    manifest = plotting.render_paper(root)
    assert len(seen) == 20
    assert len(manifest["files"]) == 41
    assert not plt.get_fignums()
    for names in seen:
        assert (root / names["png"]).read_bytes().startswith(b"\x89PNG")
        pdf = (root / names["pdf"]).read_bytes()
        assert pdf.startswith(b"%PDF")
        assert len(re.findall(rb"/Type\s*/Page\b", pdf)) == 1
        assert b"/Font" in pdf
    assert {
        p: (file_sha256(p), p.stat().st_mtime_ns) for p in initial_files
    } == initial_files
    assert all(
        file_sha256(root / name) == digest for name, digest in manifest["files"].items()
    )
    config, summary, audit, frames = load_paper_inputs(root)
    assert frames["initial_loss_recovery"].record_id.tolist() == [
        "NA",
        "0001",
        "1e5",
        "null",
    ]
    assert frames["initial_loss_recovery"].mean_terminal_sscd.tolist() == [-0.1, 0.5, 0.8, 1.1]
    second = plotting.render_paper(root)
    assert second["files"].keys() == manifest["files"].keys()
    assert all(
        second["files"][name] == digest
        for name, digest in manifest["files"].items()
        if name.endswith(".png")
    )


def test_export_failure_keeps_old_pair_manifest_and_closes_figures(
    tmp_path, monkeypatch
):
    root = compact_fixture(tmp_path / "paper")
    first = plotting.render_paper(root)
    before = {
        name: (root / name).read_bytes()
        for name in [*first["files"], "figure_manifest.json"]
    }
    real_save = Figure.savefig
    calls = 0

    def fail_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second export failure")
        return real_save(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", fail_second)
    with pytest.raises(RuntimeError, match="second export"):
        plotting.render_paper(root)
    assert not plt.get_fignums()
    assert all((root / name).read_bytes() == value for name, value in before.items())
    assert not list(root.rglob("*.tmp"))


@pytest.mark.parametrize(
    "relative", ["main/initial_loss_recovery.png", "figure_captions.md"]
)
def test_unowned_collision_is_checked_before_any_export(
    tmp_path, monkeypatch, relative
):
    root = compact_fixture(tmp_path / "paper")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("user-owned")
    monkeypatch.setattr(
        plotting,
        "publish_figures",
        lambda *a: pytest.fail("export before collision check"),
    )
    with pytest.raises(TheoryError, match="unowned"):
        plotting.render_paper(root)
    assert path.read_text() == "user-owned"
    assert not plt.get_fignums()


def test_saved_signs_extrema_trend_and_ambiguity_are_drawn_without_refitting(tmp_path):
    root = compact_fixture(tmp_path / "paper", diagnostics=True)
    config, summary, _audit, frames = load_paper_inputs(root, diagnostics=True)
    entries = {e["stem"]: e for e in paper_registry(True)}

    def draw(stem):
        return plotting._draw(
            entries[stem], frames[stem], summary["figures"][stem], config
        )

    fig, stats = draw("posterior_feedback_normalized_gain")
    assert stats["axis_limits"]["y"][0] < -100
    assert fig.axes[0].get_yscale() == "symlog"
    plt.close(fig)
    fig, _ = draw("posterior_feedback_positive_fraction")
    assert np.allclose(fig.axes[0].lines[0].get_ydata(), [0.2, 0.5, 0.1])
    assert len(fig.axes[0].collections) == 2
    plt.close(fig)
    fig, _ = draw("initial_recovery_within_prompt")
    trend = next(
        line
        for line in fig.axes[0].lines
        if line.get_label() == "Saved descriptive trend"
    )
    assert np.allclose(trend.get_ydata(), 0.01 + 0.03 * trend.get_xdata())
    plt.close(fig)
    fig, _ = draw("target_synchronization")
    assert [line.get_linestyle() for line in fig.axes[0].lines[:4]] == [
        "-",
        "--",
        "-",
        "--",
    ]
    assert len(fig.axes[0].collections) == 4
    plt.close(fig)


def test_invalid_saved_limits_and_ambiguity_do_not_hide_data(tmp_path):
    root = compact_fixture(tmp_path / "paper", diagnostics=True)
    config, summary, _audit, frames = load_paper_inputs(root, diagnostics=True)
    entries = {e["stem"]: e for e in paper_registry(True)}
    stem = "joint_target_recovery_early"
    with pytest.raises(TheoryError, match="crop"):
        plotting._draw(entries[stem], frames[stem], {"x_limits": [0, 1]}, config)
    stem = "posterior_feedback_positive_fraction"
    frame = frames[stem].copy()
    frame.loc[0, "upper_fraction"] = np.nan
    with pytest.raises(TheoryError, match="ambiguity"):
        plotting._draw(entries[stem], frame, {}, config)
    assert not plt.get_fignums()


@pytest.mark.parametrize(
    "names", [{"png": "../a.png", "pdf": "../a.pdf"}, {"png": "a.png", "pdf": "b.pdf"}]
)
def test_public_shared_publisher_rejects_unsafe_or_mismatched_pair(tmp_path, names):
    with pytest.raises(shared.PlottingError):
        shared.publish_figures(tmp_path, [(None, names)])


def test_optional_diagnostics_use_saved_all_population_and_analytical_snr(tmp_path):
    root = compact_fixture(tmp_path / "paper", diagnostics=True)
    manifest = plotting.render_paper(root, diagnostics=True)
    entries = {e["stem"]: e for e in manifest["figures"]}
    assert (
        entries["synchronization_bound_components"]["display_audit"]["population"]
        == "all"
    )
    assert (
        entries["matched_update_residual"]["display_audit"]["x_column"] == "step_index"
    )
    assert (
        entries["initial_reference_snr_sweep"]["display_audit"]["x_column"]
        == "analytical_snr"
    )
    assert entries["posterior_feedback_coverage_audit"]["display_audit"][
        "estimated_and_resolved_distinct"
    ]
    assert entries["terminal_error_terms"]["outputs"] == {}
    assert not (root / "diagnostics/terminal_error_terms.png").exists()
    assert not plt.get_fignums()


def test_early_late_alias_omits_duplicate_file_with_named_reason(tmp_path):
    root = compact_fixture(tmp_path / "paper", diagnostics=True)
    import json

    path = root / "summary.json"
    summary = json.loads(path.read_text())
    summary["figures"]["joint_target_recovery_late"].update(
        status="alias",
        reason="Early and late fixed snapshots coincide",
        alias_of="joint_target_recovery_early",
    )
    atomic_write_json(path, summary)
    manifest = plotting.render_paper(root, diagnostics=True)
    entry = next(
        e for e in manifest["figures"] if e["stem"] == "joint_target_recovery_late"
    )
    assert entry["status"] == "alias" and entry["outputs"] == {}
    assert not (root / "diagnostics/joint_target_recovery_late.pdf").exists()


def test_dimensionless_and_mixed_axis_units_are_explicit_in_registry():
    entries = {e["stem"]: e for e in paper_registry(diagnostics=True)}
    for stem in [
        "initial_target_retrieval_rank",
        "terminal_bound_tightness",
        "initial_injection_geometry",
        "initial_target_coordinates",
        "initial_injection_mismatch",
        "terminal_cancellation",
    ]:
        assert "dimensionless" in entries[stem]["normalization"].lower()
    assert "y:" in entries["initial_recovery_within_prompt"]["normalization"]
    assert "unique" in entries["initial_candidate_reference_discrepancy"]["weighting"]


def test_curated_registry_preserves_scientific_sources_and_support_order():
    from utils.experiments.theory.paper_registry import (
        FIGURE_RETIREMENTS, PAPER_APPENDIX_ORDER, PAPER_MAIN_ORDER,
        PLOT_RECIPE_VERSION, RETIRED_RENDER_STEMS,
    )
    old = {entry["stem"]: entry for entry in previous_paper_registry()}
    assert len(old) == 15
    current = paper_registry()
    assert tuple(entry["stem"] for entry in current[:4]) == PAPER_MAIN_ORDER
    assert tuple(entry["stem"] for entry in current[4:]) == PAPER_APPENDIX_ORDER
    expected_support = (PAPER_APPENDIX_ORDER[:2], PAPER_APPENDIX_ORDER[2:4],
                        PAPER_APPENDIX_ORDER[4:7], PAPER_APPENDIX_ORDER[7:9])
    for entry, supporting in zip(current[:4], expected_support):
        assert tuple(entry["supporting_figure_ids"]) == supporting
    for entry in current:
        if entry["stem"] in old:
            previous = old[entry["stem"]]
            for field in ("source_table", "formula_version", "formula", "caption_key"):
                if entry["stem"] == "terminal_bound_coverage" and field in {"formula_version", "formula"}:
                    continue  # Explicitly requested SSCD-conditional aggregation.
                assert entry[field] == previous[field]
            if entry["stem"] == "terminal_bound_coverage":
                assert entry["formula_version"] == "terminal-coverage-by-sscd-1"
                assert entry["kind"] == "four_terminal_grouped_cdf"
                assert {"group", "denominator_weight", "eligible_count", "prompt_count"} <= set(entry["required_columns"])
        elif entry["stem"] == "branch_gap_per_prompt":
            assert entry["paper_slot"] == "A10"
            assert entry["formula_version"] == "branch-gap-per-prompt-1"
        elif entry["stem"] == "reference_variation_per_prompt":
            assert entry["paper_slot"] == "A16"
            assert entry["kind"] == "four_prompt_chronological"
            assert entry["formula_version"] == "reference-variation-per-prompt-1"
            assert entry["value_column"] == "mean_reference_variation_rmse"
            assert entry["source_scalar_column"] == "direct_prop5_variation_rmse"
            assert entry["prediction_domain"] == "positive_noise_transitions"
            assert "saved_matched_updates" in entry["input_source"]
        elif entry["stem"] == "corollary3_guidance_scale_vs_loss":
            assert entry["paper_slot"] == "A15"
            assert entry["kind"] == "four_guidance_fit"
            assert entry["formula_version"] == "corollary3-guidance-fit-1"
            assert entry["requires_theory_mean"] is True
            assert {"x", "y", "seed_ids_json", "fit_residual_rmse", "direction_norm_rmse"} <= set(entry["required_columns"])
        else:
            expected = {
                "conditional_reference_error_per_prompt": ("A11", "mean_conditional_error_rmse", "nonnegative"),
                "unconditional_reference_error_per_prompt": ("A12", "mean_unconditional_reference_error_rmse", "nonnegative"),
                "target_probability_per_prompt": ("A13", "mean_target_probability", "probability"),
                "reference_branch_gap_per_prompt": ("A14", "mean_reference_gap_rmse", "nonnegative"),
            }
            assert (entry["paper_slot"], entry["value_column"], entry["value_domain"]) == expected[entry["stem"]]
            assert entry["kind"] == "four_prompt_chronological"
            assert entry["formula_version"] == "prompt-trajectory-scalars-1"
            assert entry["value_column"] in entry["required_columns"]
            if entry["stem"] == "reference_branch_gap_per_prompt":
                assert entry["source_scalar_column"] == "direct_reference_target_error_rmse"
        assert entry["figure_id"] == entry["stable_stem"] == entry["plot_data_key"] == entry["stem"]
        assert entry["plot_recipe_version"] == PLOT_RECIPE_VERSION
        assert not entry["allow_unavailable"]
        assert entry["outputs"]["png"] == f"{entry['category']}/{entry['stem']}.png"
    for diagnostics in (False, True):
        for counterfactual in (False, True):
            selected = paper_registry(diagnostics, counterfactual=counterfactual)
            assert sum(entry["category"] == "main" for entry in selected) == 4
            assert sum(entry["category"] == "appendix" for entry in selected) == 16
            assert not RETIRED_RENDER_STEMS.intersection(entry["stem"] for entry in selected)
            optional = [entry for entry in selected if entry["stem"] == "counterfactual_unconditional_response"]
            assert bool(optional) == diagnostics
            if optional:
                assert optional[0]["category"] == "diagnostics"
    assert len(FIGURE_RETIREMENTS) == 14
    assert {(item["stem"], item["old_category"], item["new_category"]) for item in FIGURE_RETIREMENTS[:2]} == {
        ("terminal_bound_coverage", "appendix", "main"),
        ("final_reproduction_bound", "main", "appendix"),
    }
    assert all(item["new_category"] is None for item in FIGURE_RETIREMENTS[2:])


def test_previous_registry_fixture_retains_fifteen_saved_tables(tmp_path):
    import json
    root = compact_fixture(tmp_path / "previous", previous_registry=True)
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "run_config.json").read_text())
    assert config["registry_version"] == PREVIOUS_REGISTRY_VERSION
    assert len(summary["plot_data"]) == 15
    assert {"branch_gap_motion_high_sscd", "branch_gap_motion_lower_sscd"}.issubset(summary["plot_data"])
    assert not {"conditional_reference_error_per_prompt", "unconditional_reference_error_per_prompt",
                "target_probability_per_prompt", "reference_branch_gap_per_prompt",
                "corollary3_guidance_scale_vs_loss", "reference_variation_per_prompt"}.intersection(summary["plot_data"])


def test_default_render_retains_dormant_ownership_receipts(tmp_path):
    root = compact_fixture(tmp_path / "paper")
    first = plotting.render_paper(root)
    # Two dormant origins are merged, without granting ownership to unknown files.
    dormant = root / "diagnostics/retained_diagnostic.pdf"
    dormant.parent.mkdir()
    dormant.write_bytes(b"previous owned diagnostic")
    already_preserved = root / "diagnostics/other_diagnostic.png"
    already_preserved.write_bytes(b"previously dormant owned diagnostic")
    unknown = root / "diagnostics/user_notes.pdf"
    unknown.write_bytes(b"unowned user content")
    first["files"]["diagnostics/retained_diagnostic.pdf"] = file_sha256(dormant)
    first["preserved_files"] = {"diagnostics/other_diagnostic.png": file_sha256(already_preserved)}
    atomic_write_json(root / "figure_manifest.json", first)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (dormant, already_preserved, unknown)}
    second = plotting.render_paper(root)
    assert second["preserved_files"] == {
        "diagnostics/retained_diagnostic.pdf": file_sha256(dormant),
        "diagnostics/other_diagnostic.png": file_sha256(already_preserved),
    }
    assert "diagnostics/user_notes.pdf" not in second["preserved_files"]
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before
    assert len(second["files"]) == 41
    assert second["requested_counts"] == {"main": 4, "appendix": 16, "diagnostics": 0}
    captions = (root / "figure_captions.md").read_text()
    assert "[A5: branch_target_errors](appendix/branch_target_errors.pdf)" in captions
    assert "[A9: final_reproduction_bound](appendix/final_reproduction_bound.pdf)" in captions


def test_prompt_reference_compact_fixtures_keep_fixed_cohorts_and_probability_domain(tmp_path):
    root = compact_fixture(tmp_path / "prompt_references")
    _, summary, _, frames = load_paper_inputs(root)
    identity = ["run_id", "original_index", "record_id", "target_id", "step_index",
                "seed_count", "seed_ids_json", "cohort_complete", "mean_terminal_sscd"]
    gap = frames["branch_gap_per_prompt"][identity]
    values = {
        "conditional_reference_error_per_prompt": "mean_conditional_error_rmse",
        "unconditional_reference_error_per_prompt": "mean_unconditional_reference_error_rmse",
        "target_probability_per_prompt": "mean_target_probability",
        "reference_branch_gap_per_prompt": "mean_reference_gap_rmse",
    }
    for stem, value_column in values.items():
        frame = frames[stem]
        assert frame[identity].equals(gap)
        assert np.isfinite(frame[value_column]).all() and frame[value_column].ge(0).all()
        assert summary["figures"][stem]["value_column"] == value_column
        assert "mean_gap_rmse" not in frame
    probability = frames["target_probability_per_prompt"].mean_target_probability
    assert probability.between(0., 1.).all()
    assert probability.min() == 0. and probability.max() == 1.
