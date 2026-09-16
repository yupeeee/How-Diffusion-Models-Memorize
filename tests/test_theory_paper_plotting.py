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
from utils.experiments.theory.paper_registry import GROUPS, paper_registry


def compact_fixture(root, *, terminal=False, diagnostics=False):
    """Represent already-reduced measurements, including a signed extreme tail."""
    root.mkdir(parents=True, exist_ok=True)
    scientific = numerical_config(model_name="sdv1", scheduler_name="ddim", num_seeds=2)
    identity = {"config": scientific, "measurement_sources": measurement_sources()}
    digest = canonical_hash(identity)
    config = dict(
        schema_version=BUNDLE_SCHEMA_VERSION,
        metric_schema_version=METRIC_SCHEMA_VERSION,
        scientific_config=scientific,
        scientific_identity=identity,
        scientific_hash=digest,
        provenance={"center_metadata": {"vector_sha256": "abc"}},
    )
    figures, specs, hashes = {}, {}, {}
    for entry in paper_registry(diagnostics):
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
        if entry["kind"] == "pair_loss":
            frame = pd.DataFrame(dict(
                x=[.1, .2, 1., 4.], y=[.3, .4, 1.1, 3.],
                control_y=[.8, 1., 2., 4.], mean_terminal_sscd=[-.1, .5, .8, 1.1],
                x_low=[.08, .15, .8, 3.5], x_high=[.12, .25, 1.2, 4.5],
                y_low=[.2, .3, 1., 2.5], y_high=[.4, .5, 1.2, 3.5],
                record_id=["NA", "0001", "1e5", "null"],
            ))
            metadata["counts"] = {"pairs": 4, "gaussian_seeds_per_pair": 2, "forward_draws": 64}
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
                     unresolved_count=1)
                for group in GROUPS for metric in ("feedback", "condition")
                for k, snr, f in [(0, .01, .2), (1, .1, .5), (2, 10., .1)]
            ])
            metadata["condition_zero_overlap"] = True
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
        == 11
    )
    assert not any(e.get("conditional_terminal") for e in entries)
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
    assert len(seen) == 15
    assert len(manifest["files"]) == 31
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
