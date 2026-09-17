"""Manuscript label contracts; authored for the author to run, not executed here."""
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
import pytest

from tests.test_theory_four_stage_plotting import draw, inputs
from utils.experiments.theory import paper_notation as notation
from utils.experiments.theory.paper_plotting import _caption
from utils.experiments.theory.paper_registry import GROUPS, GROUP_LABELS, GROUP_COLORS, measurement_registry, paper_registry, previous_paper_registry


def _legend_texts(ax):
    return [text.get_text() for artist in ax.get_children() if isinstance(artist, Legend)
            for text in artist.get_texts()]


def test_comprehensive_measurement_inventory_retains_all_twenty_notation_contracts():
    previous = {entry["stem"]: entry for entry in previous_paper_registry()}
    entries = measurement_registry()
    assert len(entries) == 20
    for entry in entries:
        assert entry["plot_recipe_version"] == notation.NOTATION_VERSION
        for axis in entry["axes"].values():
            assert not any(word in axis for word in ("RMSE", "Empirical", "_K", "^K", "_{ref}", r"\frac", "<=", "\n"))
        assert entry.get("comparison_centre") != "zero"
        if entry["stem"] == "reference_variation_per_prompt":
            assert entry["axes"] == {"x": r"$T-t$", "y": r"$\mathcal{V}_t/\sqrt{d}$"}
            assert entry["formula_version"] == "reference-directional-variation-per-prompt-1"
            assert entry["prediction_domain"] == "positive_noise_transitions"
            assert "historical Equation 15" in entry["notation_details"]
            assert "positive part is taken after the signed integral" in entry["notation_details"]
            assert "no unit direction" in entry["notation_details"]
            assert "T-t ends at T-2" in entry["notation_details"]
            assert "unresolved condition signs" in entry["notation_details"]
            continue
        if entry["stem"] == "corollary3_guidance_scale_vs_loss":
            assert entry["axes"] == {"x": r"$\sqrt{L_T(c)/(d\,\mathrm{SNR}_T)}$", "y": r"$\widehat{g}$"}
            assert entry["formula_version"] == "corollary3-guidance-fit-1"
            assert entry["requires_theory_mean"] is True
            assert entry["saved_x_definition"] == "L_T(c)/(d*SNR_T)"
            assert entry["display_x_definition"] == "sqrt(L_T(c)/(d*SNR_T))"
            assert "display takes one square root of saved x" in entry["notation_details"]
            assert "outside the mean over forward draws" in entry["notation_details"]
            assert "signed joint no-intercept least-squares" in entry["notation_details"]
            assert "Equation 54" in entry["notation_details"]
            continue
        if entry["stem"] == "branch_gap_per_prompt":
            assert entry["axes"] == {"x": r"$T-t$", "y": r"$\|\boldsymbol{\Delta}_t\|/\sqrt{d}$"}
            assert "norm is computed before seed averaging" in entry["notation_details"]
            continue
        if entry["stem"] in {
            "conditional_reference_error_per_prompt", "unconditional_reference_error_per_prompt",
            "target_probability_per_prompt", "reference_branch_gap_per_prompt",
        }:
            expected_y = {
                "conditional_reference_error_per_prompt": r"$e_t(c)/\sqrt{d}$",
                "unconditional_reference_error_per_prompt": r"$e_t(\varnothing)/\sqrt{d}$",
                "target_probability_per_prompt": r"$p_t$",
                "reference_branch_gap_per_prompt": r"$\|\bar{\boldsymbol{\Delta}}_t\|/\sqrt{d}$",
            }
            assert entry["axes"] == {"x": r"$T-t$", "y": expected_y[entry["stem"]]}
            assert entry["formula_version"] == "prompt-trajectory-scalars-1"
            continue
        if entry["stem"] == "terminal_bound_coverage":
            assert entry["source_table"] == previous[entry["stem"]]["source_table"]
            assert entry["kind"] == "four_terminal_grouped_cdf"
            assert entry["formula_version"] == "projected-gap-error-1"
            continue  # Group-conditional ECDFs were explicitly requested later.
        if entry["stem"] in {"posterior_feedback_over_time", "posterior_feedback_condition_margin",
                             "synchronization_bound", "final_reproduction_bound"}:
            assert entry["formula_version"] == "projected-gap-error-1"
            assert entry["measurement_contract"] == "projected-gap-error-1"
            assert entry["direct_statement"] is False
            assert "Q<=S is not asserted" in entry["mathematical_scope"]
            assert entry["source_table"] == previous[entry["stem"]]["source_table"]
            continue
        for key in ("source_table", "formula", "formula_version", "required_columns"):
            # Eligible count was added by the prior curation, independent of notation.
            if key == "required_columns" and entry["stem"] == "posterior_feedback_over_time":
                assert set(entry[key]) == set(previous[entry["stem"]][key]) | {"eligible_count"}
            else:
                assert entry[key] == previous[entry["stem"]][key]
    by_stem = {entry["stem"]: entry for entry in entries}
    initial = by_stem["initial_loss_recovery"]["axes"]
    assert initial["x"] == r"$\sqrt{L_T(c)/(d\,\mathrm{SNR}_T)}$"
    assert initial["y"] == r"$\sqrt{\mathbb{E}_{\mathbf{x}_T}[\|\hat{\mathbf{x}}_T(b)-\mathbf{x}^{\star}\|^2]/d}$"
    margin = by_stem["posterior_feedback_condition_margin"]["axes"]
    assert all(term in margin["x"] for term in (r"\boldsymbol{\Delta}_t", r"-e_t^{\parallel}(\boldsymbol{\Delta})", r"-\mathcal{V}_t"))
    assert r"\log[p_{t-1}(\mathbf{x}_{t-1})/p_{t-1}(\mathbf{x}_{t-1}^{\mathrm{cf}})]" in margin["y"]
    assert "e_t(c)" not in margin["x"] and r"e_t(\varnothing)" not in margin["x"]
    margin_entry = by_stem["posterior_feedback_condition_margin"]
    assert "step_index" in margin_entry["required_columns"]
    assert margin_entry["display_policy"] == "finite_saved_values"
    assert margin_entry["per_timestep_exports"]["directory"] == "appendix/posterior_feedback_condition_margin"
    assert notation.TERMINAL_EXPRESSIONS["reference"] == r"e_1(c)+(g-1)[e_1^{\parallel}(\boldsymbol{\Delta})+R(1-p_1)]"
    assert by_stem["conditional_reference_error_per_prompt"]["axes"]["y"] == r"$e_t(c)/\sqrt{d}$"
    assert by_stem["unconditional_reference_error_per_prompt"]["axes"]["y"] == r"$e_t(\varnothing)/\sqrt{d}$"
    for stem in ("terminal_observable_bound", "final_reproduction_bound"):
        assert by_stem[stem]["axes"]["x"].endswith(r"]/\sqrt{d}$")
        assert "correction" not in by_stem[stem]["axes"]["x"]
        assert "saved independent terminal correction" in by_stem[stem]["notation_details"]
        assert "not the uncorrected expression alone" in by_stem[stem]["notation_details"]
        assert by_stem[stem]["axes"]["y"] == r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|/\sqrt{d}$"


def test_initial_labels_keep_rms_averaging_and_both_branch_controls(tmp_path):
    data = inputs(tmp_path)
    frame = data[2]["initial_loss_recovery"].copy(deep=True)
    fig, audit = draw(data, "initial_loss_recovery")
    try:
        assert {r"$b=c$", r"$b=\varnothing$"}.issubset(_legend_texts(fig.axes[0]))
        assert fig.axes[1].get_ylabel() == "SSCD"
        assert audit["colorbar_label"] == "SSCD"
        assert audit["paired_connector_count"] == 4 and not audit["equality_guide"]
        assert data[2]["initial_loss_recovery"].equals(frame)
    finally:
        plt.close(fig)


def test_reference_legend_does_not_conflate_gaussian_probes_with_trajectory_inputs(tmp_path):
    fig, audit = draw(inputs(tmp_path), "unconditional_reference_convergence")
    try:
        text = _legend_texts(fig.axes[0])
        assert r"$e_t(\mathbf{z},\varnothing)$" in text
        assert r"$\|\bar{\mathbf{x}}_t(\mathbf{z},\varnothing)-\boldsymbol{\mu}\|$" in text
        assert r"$\|\hat{\mathbf{x}}_t(\mathbf{z},\varnothing)-\boldsymbol{\mu}\|$" in text
        assert r"$\mathrm{SNR}_T$" in text
        assert audit["native_sweep_preserved"] and not audit["network_extrapolation"]
    finally:
        plt.close(fig)


def test_sync_legends_compare_only_branch_gap_and_lemma6_bound(tmp_path):
    data = inputs(tmp_path)
    fig, audit = draw(data, "synchronization_bound")
    try:
        text = _legend_texts(fig.axes[0])
        assert r"$\|\boldsymbol{\Delta}_t\|$" in text
        assert r"$e_t^{\parallel}(\boldsymbol{\Delta})+R(1-p_t)$" in text
        assert not any(r"\max" in label for label in text)
        assert audit["displayed_metrics"] == ["gap", "bound"]
        assert audit["quantity_styles"] == {"gap": "-", "bound": "--"}
        lines = fig.axes[0].lines
        assert len(lines) == 2 * len(GROUPS)
        assert [line.get_linestyle() for line in lines] == ["-", "--"] * len(GROUPS)
        assert not fig.axes[0].collections
        legend = fig.axes[0].get_legend()
        handles = legend.legend_handles
        assert [line.get_linestyle() for line in handles[-2:]] == ["-", "--"]
        assert [label.get_text() for label in legend.get_texts()][:2] == [GROUP_LABELS[group] for group in GROUPS]
        assert legend._loc == 8 and legend._ncols == 2
        anchor = legend.get_bbox_to_anchor().transformed(fig.axes[0].transAxes.inverted())
        assert anchor.x0 == pytest.approx(.5) and anchor.y0 == pytest.approx(1.01)
        assert audit["legend_placement"] == "outside top"
        assert audit["legend_columns"] == "SSCD groups; manuscript quantities"
        assert len([artist for artist in fig.axes[0].get_children() if isinstance(artist, Legend)]) == 1
        assert fig.axes[0].get_xlabel() == r"$T-t$"
        assert fig.axes[0].get_ylabel() == r"$\cdot/\sqrt{d}$"
    finally:
        plt.close(fig)
    fig, _ = draw(data, "branch_target_errors")
    try:
        assert {r"$b=c$", r"$b=\varnothing$"}.issubset(_legend_texts(fig.axes[0]))
        assert fig.axes[0].get_ylabel() == r"$\|\hat{\mathbf{x}}_t(b)-\mathbf{x}^{\star}\|/\sqrt{d}$"
        assert r"e_t(\varnothing)" not in fig.axes[0].get_ylabel()
    finally:
        plt.close(fig)


def test_coverage_formula_legends_match_saved_curves_and_preserve_correction_scope(tmp_path):
    data = inputs(tmp_path)
    stem = "terminal_bound_coverage"
    fig, audit = draw(data, stem)
    try:
        ax = fig.axes[0]
        assert ax.get_ylabel() == r"$\Pr(\,\cdot\leq\tau)$"
        assert len(ax.lines) == 6
        for line, (group, name) in zip(ax.lines, ((group, name) for group in GROUPS for name in ("actual", "observable", "reference"))):
            assert line.get_label() == notation.TERMINAL_LABELS[name]
            assert line.get_color() == GROUP_COLORS[group]
            assert audit["legend_quantities"][name] == line.get_label()
            saved = data[2][stem].loc[lambda rows: rows.group.eq(group) & rows.distribution.eq(name)]
            np.testing.assert_array_equal(line.get_xdata()[1:-1], saved.value)
            np.testing.assert_array_equal(line.get_ydata()[1:-1], saved.cdf)
        assert set(GROUP_LABELS.values()) <= set(_legend_texts(ax))
        assert ax.get_legend().get_title().get_text() == ""
        assert ax.get_legend()._loc == 8 and audit["legend_placement"] == "outside top"
        assert audit["terminal_scope"] == "finite_terminal_update_extension_deterministic"
        assert audit["terminal_correction_display"] == "caption_only; corrected numerical values retained"
        assert all("correction" not in label for label in _legend_texts(ax))
        entry = deepcopy(data[3][stem])
        entry.update(measurement_metadata=data[1]["figures"][stem], status="available",
                     scientific_hash="saved-science", display_audit=audit)
        caption = _caption(entry)
        assert "base expression + saved independent terminal correction" in caption
        assert "finite_terminal_update_extension_deterministic" in caption
    finally:
        plt.close(fig)
    metadata = deepcopy(data[1]["figures"]["final_reproduction_bound"])
    metadata["terminal_scope"] = "finite_terminal_update_extension_gaussian_noise_bound"
    fig, audit = draw(data, "final_reproduction_bound", metadata=metadata)
    try:
        assert fig.axes[0].get_legend().get_title().get_text() == "Probabilistic extension"
        assert audit["comparison_reference"] == "Probability-qualified bound"
        assert audit["diagonal_label"] == r"$x=y$"
    finally:
        plt.close(fig)


def test_caption_defines_manuscript_symbols_and_preserves_empirical_scope(tmp_path):
    data = inputs(tmp_path)
    for stem in ("initial_loss_recovery", "branch_gap_posterior_response", "unconditional_reference_convergence",
                 "corollary3_guidance_scale_vs_loss", "reference_variation_per_prompt", "synchronization_bound",
                 "posterior_feedback_condition_margin", "final_reproduction_bound"):
        entry = deepcopy(data[3][stem])
        entry.update(measurement_metadata=data[1]["figures"][stem], status="available",
                     scientific_hash="saved-science")
        caption = _caption(entry)
        assert "Notation details:" in caption
        assert "does not identify it with the full training distribution" in caption
        if stem == "initial_loss_recovery":
            assert "square root is outside the mean" in caption
        if stem == "branch_gap_posterior_response":
            assert "Equation 66" in caption and "Reconstructed matched CFG" in caption
        if stem == "unconditional_reference_convergence":
            assert "fixed Gaussian input bank" in caption
        if stem in {"synchronization_bound", "posterior_feedback_condition_margin", "final_reproduction_bound"}:
            assert "projected-gap-error-1" in caption
            assert "Q<=S is not asserted" in caption
            assert "derived" in caption and "refinements" in caption
            assert "may be negative" in caption
            assert "gap-projection identity" in caption
        if stem == "reference_variation_per_prompt":
            assert r"$\mathcal{V}_t/\sqrt{d}$" in caption
            assert "Equation 15" in caption and "t=1 is excluded" in caption
            assert "positive_noise_transitions" in caption
            assert "numerically_unresolved" in caption
            assert "positive part" in caption and "signed integral" in caption
            assert "audit_data/feedback_endpoints.csv" in caption
            assert "not selected global mu" in caption
        if stem == "corollary3_guidance_scale_vs_loss":
            assert "Normalization: " + entry["normalization"] in caption
            assert entry["axes"]["x"] == r"$\sqrt{L_T(c)/(d\,\mathrm{SNR}_T)}$"
            assert entry["axes"]["x"] in caption
            assert entry["axes"]["y"] == r"$\widehat{g}$"
            assert entry["axes"]["y"] in caption


def test_retired_zero_render_routes_are_blocked_and_selected_mean_comparison_remains(tmp_path):
    import pandas as pd
    from utils.experiments.theory.contracts import TheoryError
    from utils.experiments.theory.paper_registry import _ZERO_BASELINE_FIGURES, RETIRED_RENDER_STEMS
    from utils.experiments.theory.paper_plotting import _draw

    zero_stems = {entry["stem"] for entry in _ZERO_BASELINE_FIGURES}
    assert zero_stems <= RETIRED_RENDER_STEMS
    for entry in _ZERO_BASELINE_FIGURES:
        with pytest.raises(TheoryError, match="retired"):
            _draw(entry, pd.DataFrame(), {}, {})
    for diagnostics in (False, True):
        assert zero_stems.isdisjoint(entry["stem"] for entry in paper_registry(diagnostics=diagnostics))
    data = inputs(tmp_path)
    assert zero_stems.isdisjoint(data[2])
    fig, audit = draw(data, "unconditional_reference_convergence")
    try:
        ax = fig.axes[0]
        assert "Reference limit" not in _legend_texts(ax)
        assert audit["horizontal_reference_value"] == data[1]["figures"]["unconditional_reference_convergence"]["mean_offset_rmse"]
        assert any(np.array_equal(line.get_ydata(), [audit["horizontal_reference_value"]] * 2) for line in ax.lines)
        assert audit["native_sweep_preserved"] and not audit["network_extrapolation"]
    finally:
        plt.close(fig)
    fig, _ = draw(data, "initial_unconditional_mean_concentration")
    try:
        assert r"\boldsymbol{\mu}" in fig.axes[0].get_xlabel()
        assert fig.axes[0].get_ylabel() == "CDF"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("change, message", [
    ({"sample_count": 100}, "sample count"),
    ({"mean_seed": 9}, "mean seed"),
    ({"level": {"step_index": 1}}, "first-step"),
    ({"vector_sha256": "different"}, "inconsistent"),
])
def test_reference_mean_render_rejects_inconsistent_estimator_receipt(tmp_path, change, message):
    from utils.experiments.theory.contracts import TheoryError
    data = inputs(tmp_path)
    data[0]["scientific_config"].update(mean_source="reference-initial", num_mean_samples=10000, mean_seed=0)
    stem = "unconditional_reference_convergence"
    metadata = deepcopy(data[1]["figures"][stem])
    receipt = dict(source="initial_unconditional_reference_monte_carlo", vector_sha256="reference-mean",
                   estimator_hash="reference-estimator", sample_count=10000, mean_seed=0,
                   level={"step_index": 0})
    metadata.update(theory_mean=receipt, theory_mean_source=receipt["source"], theory_mean_sha256=receipt["vector_sha256"])
    receipt.update(change)
    with pytest.raises(TheoryError, match=message):
        draw(data, stem, metadata=metadata)



def _minimum_snr_mean_inputs(tmp_path):
    data = inputs(tmp_path)
    science = data[0]["scientific_config"]
    science.update(mean_source="reference-min-snr", num_mean_samples=10000,
                   mean_seed=0, reference_snr_decades=6.)
    stem = "unconditional_reference_convergence"
    metadata = deepcopy(data[1]["figures"][stem])
    initial_snr = metadata["actual_initial_snr"]
    snr = initial_snr * 10. ** -science["reference_snr_decades"]
    receipt = dict(
        source="minimum_snr_unconditional_reference_monte_carlo",
        vector_sha256="minimum-snr-mean", estimator_hash="minimum-snr-estimator",
        sample_count=10000, mean_seed=0, reference_snr_decades=6.,
        initial_snr=initial_snr, estimation_snr=snr,
        initial_level={"source_range": "native", "step_index": 0, "timestep": 981,
                       "snr": initial_snr},
        level={"source_range": "analytical", "grid_index": 0, "step_index": None,
               "timestep": None, "snr": snr, "alpha": float(np.sqrt(snr / (1 + snr))),
               "sigma": float(1 / np.sqrt(1 + snr))},
    )
    metadata.update(theory_mean=receipt, theory_mean_source=receipt["source"],
                    theory_mean_sha256=receipt["vector_sha256"], mean_offset_rmse=.0025)
    return data, stem, metadata


def test_minimum_snr_mean_plot_preserves_measured_limit_and_native_anchor(tmp_path):
    data, stem, metadata = _minimum_snr_mean_inputs(tmp_path)
    fig, audit = draw(data, stem, metadata=metadata)
    try:
        ax = fig.axes[0]
        assert audit["horizontal_reference_value"] == .0025
        assert any(np.array_equal(line.get_ydata(), [.0025, .0025]) for line in ax.lines)
        native_guides = [line for line in ax.lines if line.get_linestyle() == ":"
                         and len(line.get_xdata()) == 2
                         and line.get_xdata()[0] == line.get_xdata()[1]]
        assert len(native_guides) == 1
        assert r"$\mathrm{SNR}_T$" in _legend_texts(ax)
        assert list(native_guides[0].get_xdata()) == [metadata["actual_initial_snr"]] * 2
        assert audit["native_sweep_preserved"] and not audit["network_extrapolation"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("path, value, message", [
    (("sample_count",), 100, "sample count"),
    (("mean_seed",), 9, "mean seed"),
    (("source",), "initial_unconditional_reference_monte_carlo", "inconsistent"),
    (("estimator_hash",), None, "minimum-SNR"),
    (("reference_snr_decades",), 5., "minimum-SNR"),
    (("estimation_snr",), .01, "minimum-SNR"),
    (("initial_snr",), .1, "minimum-SNR"),
    (("level", "snr"), .01, "minimum-SNR"),
    (("level", "alpha"), .1, "minimum-SNR"),
    (("level", "source_range"), "native", "minimum-SNR"),
    (("level", "grid_index"), 1, "minimum-SNR"),
    (("level", "step_index"), 0, "minimum-SNR"),
    (("level", "timestep"), 981, "minimum-SNR"),
    (("initial_level", "snr"), .1, "minimum-SNR"),
])
def test_minimum_snr_mean_plot_rejects_wrong_estimation_level(tmp_path, path, value, message):
    from utils.experiments.theory.contracts import TheoryError
    data, stem, metadata = _minimum_snr_mean_inputs(tmp_path)
    field = metadata["theory_mean"]
    for key in path[:-1]:
        field = field[key]
    field[path[-1]] = value
    with pytest.raises(TheoryError, match=message):
        draw(data, stem, metadata=metadata)


@pytest.mark.parametrize("stem, ylabel", [
    ("conditional_reference_error_per_prompt", r"$e_t(c)/\sqrt{d}$"),
    ("unconditional_reference_error_per_prompt", r"$e_t(\varnothing)/\sqrt{d}$"),
    ("target_probability_per_prompt", r"$p_t$"),
    ("reference_branch_gap_per_prompt", r"$\|\bar{\boldsymbol{\Delta}}_t\|/\sqrt{d}$"),
])
def test_prompt_reference_labels_use_manuscript_scalars_without_extra_normalization(tmp_path, stem, ylabel):
    data = inputs(tmp_path)
    before = data[2][stem].copy(deep=True)
    fig, audit = draw(data, stem)
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == r"$T-t$" and ax.get_ylabel() == ylabel
        assert fig.axes[1].get_ylabel() == "SSCD"
        assert audit["colorbar_label"] == "SSCD"
        assert len(ax.texts) == 0 and ax.get_legend() is None
        assert audit["complete_seed_cohort_checked"]
        if stem == "target_probability_per_prompt":
            assert tuple(ax.get_ylim()) == (0., 1.)
            assert audit["value_domain"] == "probability"
            assert r"\sqrt{d}" not in ax.get_ylabel()
        else:
            assert audit["value_domain"] == "nonnegative"
        assert data[2][stem].equals(before)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("diagnostics,counterfactual", [(False, False), (True, False), (False, True), (True, True)])
def test_selected_publication_has_six_pairs_under_figures_without_scientific_aliasing(diagnostics, counterfactual):
    from utils.experiments.theory.paper_registry import PAPER_MAIN_ORDER, PAPER_APPENDIX_ORDER, REGISTRY_VERSION
    entries = paper_registry(diagnostics, counterfactual=counterfactual)
    inventory = {entry["stem"]: entry for entry in measurement_registry()}
    assert REGISTRY_VERSION == "four-stage-paper-curation-20"
    assert tuple(entry["stem"] for entry in entries) == PAPER_MAIN_ORDER
    assert PAPER_APPENDIX_ORDER == () and len(entries) == 6
    assert [entry["paper_slot"] for entry in entries] == [f"M{index}" for index in range(1, 7)]
    assert len({path for entry in entries for path in entry["outputs"].values()}) == 12
    for entry in entries:
        stem = entry["stem"]
        filename = "guidance_scale_vs_loss" if stem == "corollary3_guidance_scale_vs_loss" else stem
        assert entry["outputs"] == {ext: f"figures/{filename}.{ext}" for ext in ("png", "pdf")}
        assert entry["category"] == entry["section"] == "main"
        assert "per_timestep_exports" not in entry
        for key in ("source_table", "formula_version", "formula_id", "plot_data_key", "axes", "required_columns"):
            assert entry[key] == inventory[stem][key]
    margin = next(entry for entry in entries if entry["stem"] == "posterior_feedback_condition_margin")
    assert margin["pooled_scatter_alpha"] == .01
    assert "pooled_scatter_alpha_scale" not in margin
