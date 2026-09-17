"""Saved-only four-stage artist contracts; tests intentionally not executed here."""
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
import numpy as np
import pytest

from tests.test_theory_paper_plotting import compact_fixture, load_measurement_inputs
from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.paper_plotting import _draw
from utils.experiments.theory.paper_registry import measurement_registry


def inputs(tmp_path):
    root = compact_fixture(tmp_path / "paper", terminal=False)
    config, summary, _, frames = load_measurement_inputs(root)
    return config, summary, frames, {entry["stem"]: entry for entry in measurement_registry()}


def draw(data, stem, *, frame=None, metadata=None):
    config, summary, frames, entries = data
    return _draw(entries[stem], frames[stem] if frame is None else frame,
                 summary["figures"][stem] if metadata is None else metadata, config)


def test_dose_uses_H_fixed_symlog_signed_range_and_all_endpoints(tmp_path):
    data = inputs(tmp_path)
    fig, audit = draw(data, "branch_gap_posterior_response")
    try:
        ax = fig.axes[0]
        assert ax.get_yscale() == "symlog" and audit["symlog_linthresh"] == 1e-3
        assert ax.get_xlim() == (0., 1.) and ax.get_ylim()[0] < -2
        assert not audit["log_odds_substitution"]
        texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert not {r"$s=0$", r"$s=1/g$", r"$s=1$"}.intersection(texts)
        vertical = [line for line in ax.lines if line.get_linestyle() == ":"
                    and len(line.get_xdata()) == 2
                    and np.all(np.asarray(line.get_xdata()) == line.get_xdata()[0])]
        expected = 1 / data[1]["figures"]["branch_gap_posterior_response"]["guidance_scale"]
        assert len(vertical) == 1 and vertical[0].get_xdata()[0] == expected
        assert audit["vertical_reference_positions"] == [expected]
        tick_labels = dict(zip(ax.get_xticks(), (tick.get_text() for tick in ax.get_xticklabels())))
        assert tick_labels[expected] == r"$1/g$"
        assert audit["vertical_reference_label_placement"] == "x-axis tick"
        assert audit["vertical_reference_tick_label"] == r"$1/g$"
        assert ax.get_legend()._loc == 4 and audit["legend_placement"] == "inside lower right"
        assert audit["cfg_endpoint_label"] == "Reconstructed matched CFG"
        assert not ax.texts
    finally:
        plt.close(fig)
    frame = data[2]["branch_gap_posterior_response"].copy()
    frame.loc[0, "denominator_weight"] = 99
    with pytest.raises(TheoryError, match="changes population"):
        draw(data, "branch_gap_posterior_response", frame=frame)


def test_chronological_indices_include_initialization_and_no_terminal_prediction(tmp_path):
    fig, audit = draw(inputs(tmp_path), "branch_gap_synchronization")
    try:
        assert audit["chronological_prediction_range"] == [0, 2]
        assert not audit["terminal_output_prediction"]
        assert fig.axes[0].get_xlabel() == r"$T-t$"
        assert fig.axes[0].get_xticklabels()[0].get_text() == "0"
        assert "0 is initialization" in audit["horizontal_coordinate"]
        assert fig.axes[0].get_xlim() == (0., 2.)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_reference_preserves_both_native_network_curves_and_reference_only_extension(tmp_path):
    fig, audit = draw(inputs(tmp_path), "unconditional_reference_convergence")
    try:
        assert audit["native_sweep_preserved"] and not audit["network_extrapolation"]
        for color in ("#e67d24", "#3b9955"):
            lines = [line for line in fig.axes[0].lines if line.get_color() == color]
            assert any(np.array_equal(line.get_xdata(), [.01, .1, 10.]) for line in lines)
        notes = fig.axes[0].texts
        assert len(notes) == 1 and notes[0].get_text() == "Analytical reference only"
        assert notes[0].get_fontsize() == 10
        assert audit["analytical_only_range"][0] < notes[0].get_position()[0] < audit["analytical_only_range"][1]
    finally:
        plt.close(fig)



def test_reference_retains_selected_mean_offset_guide_without_legend_entry(tmp_path):
    data = inputs(tmp_path)
    stem = "unconditional_reference_convergence"
    metadata = deepcopy(data[1]["figures"][stem])
    metadata.update(mean_norm_rmse=.8, bank_mean_norm_rmse=.5, mean_offset_rmse=.3,
                    reference_limit_rmse=.3, y_limits=[0., .01])
    fig, audit = draw(data, stem, metadata=metadata)
    try:
        assert audit["horizontal_reference_value"] == .3
        assert audit["horizontal_reference_label"] == "Reference limit"
        assert not audit["horizontal_reference_legend_visible"]
        ax = fig.axes[0]
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "Reference limit" not in labels
        guides = [line for line in ax.lines
                  if line.get_color() == ".5" and line.get_linestyle() == "--"]
        assert len(guides) == 1
        assert np.array_equal(guides[0].get_ydata(), [.3, .3])
        assert ax.get_ylim()[1] >= .3
        assert not audit["network_extrapolation"]
    finally:
        plt.close(fig)


def test_terminal_log_zeros_use_edge_markers_without_pseudodata(tmp_path):
    data = inputs(tmp_path)
    stem = "final_reproduction_bound"
    frame = data[2][stem].copy()
    frame.loc[1, ["x", "y"]] = [1e-5, 1e-6]
    frame.loc[2, ["x", "y"]] = [np.inf, .8]
    before = frame.copy(deep=True)
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["axis_scale"] = "log"
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        assert fig.axes[0].get_xscale() == fig.axes[0].get_yscale() == "log"
        assert audit["zero_counts"] == {"x": 1, "y": 1}
        assert audit["infinite_bound_count"] == 1
        assert audit["zero_position_policy"].startswith("axes-edge")
        assert fig.axes[0].get_xlim() == fig.axes[0].get_ylim()
        assert fig.axes[1].get_ylabel() == "SSCD"
        assert frame.equals(before)
    finally:
        plt.close(fig)


def test_retained_measurement_artists_have_no_small_free_annotations_and_readable_legends(tmp_path):
    from matplotlib.legend import Legend
    data = inputs(tmp_path)
    for entry in data[3].values():
        if entry["category"] != "appendix":
            continue
        fig, _ = draw(data, entry["stem"])
        try:
            ax = fig.axes[0]
            assert not ax.get_title()
            assert all(text.get_gid() == "curation-required-label" and
                       text.get_fontsize() == (10 if text.get_text() == "Analytical reference only" else 12)
                       for text in ax.texts)
            for legend in (artist for artist in ax.get_children() if isinstance(artist, Legend)):
                assert all(label.get_fontsize() == 12 for label in legend.get_texts())
                assert legend.get_title().get_fontsize() == 12
        finally:
            plt.close(fig)


def test_grouped_terminal_coverage_preserves_exact_mass_and_factorized_legend(tmp_path):
    import pandas as pd
    from utils.experiments.theory.paper_notation import TERMINAL_LABELS
    from utils.experiments.theory.paper_registry import GROUPS, GROUP_COLORS, GROUP_LABELS
    data = inputs(tmp_path)
    stem = "terminal_bound_coverage"
    names = ("actual", "observable", "reference")
    frame = pd.DataFrame([
        dict(group=group, distribution=name, value=value, cdf=cdf,
             denominator_weight=1., eligible_count=4, prompt_count=2)
        for group in GROUPS for name in names
        for value, cdf in [(0., .25), (.2, .5), (1., .75), (np.inf, 1.)]])
    before = frame.copy(deep=True)
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["group_population"] = {group: {"denominator_weight": 1., "eligible_count": 4, "prompt_count": 2} for group in GROUPS}
    metadata["grouped_zero_mass"] = {group: {name: .25 for name in names} for group in GROUPS}
    metadata["grouped_infinite_mass"] = deepcopy(metadata["grouped_zero_mass"])
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        assert len(fig.axes) == 1 and len(ax.lines) == 6
        assert not ax.texts and not ax.get_title()
        labels = [label.get_text() for label in ax.get_legend().get_texts() if label.get_text()]
        assert labels == [GROUP_LABELS[group] for group in GROUPS] + [TERMINAL_LABELS[name] for name in names]
        assert not any("Zero:" in label or "correction" in label for label in labels)
        assert ax.get_legend().get_title().get_text() == ""
        assert ax.get_legend()._loc == 8 and audit["legend_placement"] == "outside top"
        np.testing.assert_allclose(ax.transAxes.inverted().transform(ax.get_legend().get_bbox_to_anchor().p0), [.5, 1.01])
        assert audit["terminal_scope"] == "finite_terminal_update_extension_deterministic"
        assert audit["terminal_scope_display"] == "caption_only"
        assert not audit["zero_mass_legend_visible"] and not audit["infinite_mass_legend_visible"]
        assert audit["grouped_zero_mass"] == metadata["grouped_zero_mass"]
        assert audit["grouped_infinite_mass"] == metadata["grouped_infinite_mass"]
        assert audit["common_population_checked"] and audit["plotted_curve_count"] == 6
        for line, (group, name) in zip(ax.lines, ((group, name) for group in GROUPS for name in names), strict=True):
            assert line.get_color() == GROUP_COLORS[group]
            assert line.get_linestyle() == {"actual": "-", "observable": "--", "reference": "-."}[name]
            assert line.get_label() == TERMINAL_LABELS[name]
            assert line.get_gid() == group + ":" + name
            np.testing.assert_array_equal(line.get_xdata()[1:-1], [.2, 1.])
            np.testing.assert_array_equal(line.get_ydata(), [.25, .5, .75, .75])
        assert frame.equals(before) and audit["zero_replacement"] is None
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mutation", ["denominator", "missing_group", "missing_quantity", "zero_mass", "infinite_mass", "missing_population", "scope"])
def test_grouped_terminal_coverage_rejects_inconsistent_saved_receipts(tmp_path, mutation):
    from utils.experiments.theory.paper_registry import GROUPS
    data = inputs(tmp_path)
    stem = "terminal_bound_coverage"
    frame = data[2][stem].copy(deep=True)
    metadata = deepcopy(data[1]["figures"][stem])
    if mutation == "denominator":
        frame.loc[frame.index[0], "denominator_weight"] += 1.
    elif mutation == "missing_group":
        frame = frame.loc[frame.group.ne(GROUPS[0])]
    elif mutation == "missing_quantity":
        frame = frame.loc[~(frame.group.eq(GROUPS[0]) & frame.distribution.eq("reference"))]
    elif mutation == "zero_mass":
        metadata["grouped_zero_mass"][GROUPS[0]]["actual"] = .5
    elif mutation == "infinite_mass":
        metadata["grouped_infinite_mass"][GROUPS[0]]["actual"] = .5
    elif mutation == "missing_population":
        del metadata["group_population"]
    else:
        metadata["manuscript_extension_required"] = False
    with pytest.raises(TheoryError):
        draw(data, stem, frame=frame, metadata=metadata)


def test_grouped_terminal_empty_group_remains_explicit_without_a_fake_curve(tmp_path):
    from utils.experiments.theory.paper_registry import GROUPS, GROUP_LABELS
    data = inputs(tmp_path)
    stem = "terminal_bound_coverage"
    frame = data[2][stem].loc[data[2][stem].group.eq(GROUPS[0])].copy()
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["group_population"][GROUPS[1]] = {"denominator_weight": 0., "eligible_count": 0, "prompt_count": 0}
    for field in ("grouped_zero_mass", "grouped_infinite_mass"):
        metadata[field][GROUPS[1]] = {name: None for name in ("actual", "observable", "reference")}
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        assert len(fig.axes[0].lines) == 3 and audit["empty_groups"] == [GROUPS[1]]
        assert audit["group_population"][GROUPS[1]]["eligible_count"] == 0
        assert GROUP_LABELS[GROUPS[1]] not in [item.get_text() for item in fig.axes[0].get_legend().get_texts()]
    finally:
        plt.close(fig)


def test_grouped_terminal_degenerate_support_uses_plateaus_without_epsilon(tmp_path):
    import pandas as pd
    from utils.experiments.theory.paper_registry import GROUPS
    data = inputs(tmp_path)
    stem = "terminal_bound_coverage"
    names = ("actual", "observable", "reference")
    frame = pd.DataFrame([
        dict(group=group, distribution=name, value=0. if name == "actual" else np.inf, cdf=1.,
             denominator_weight=1., eligible_count=4, prompt_count=2)
        for group in GROUPS for name in names])
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["group_population"] = {group: {"denominator_weight": 1., "eligible_count": 4, "prompt_count": 2} for group in GROUPS}
    metadata["grouped_zero_mass"] = {group: {name: float(name == "actual") for name in names} for group in GROUPS}
    metadata["grouped_infinite_mass"] = {group: {name: float(name != "actual") for name in names} for group in GROUPS}
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        assert ax.get_xlim() == (1., 10.) and ax.get_ylim() == (0., 1.)
        assert audit["empty_positive_domain"] and audit["full_positive_support"] is None
        assert audit["zero_replacement"] is None and not ax.texts
        for line in ax.lines:
            np.testing.assert_array_equal(line.get_xdata(), [1., 10.])
            np.testing.assert_array_equal(line.get_ydata(), [1., 1.] if line.get_gid().endswith(":actual") else [0., 0.])
    finally:
        plt.close(fig)


def test_grouped_terminal_full_union_support_ignores_obsolete_pooled_limits(tmp_path):
    data = inputs(tmp_path)
    stem = "terminal_bound_coverage"
    metadata = deepcopy(data[1]["figures"][stem])
    metadata.update(x_limits=[.1, .3], y_limits=[.2, .4])
    fig, audit = draw(data, stem, metadata=metadata)
    try:
        values = data[2][stem].value.to_numpy(dtype=float)
        positive = values[(values > 0) & np.isfinite(values)]
        assert fig.axes[0].get_xlim() == pytest.approx((positive.min() / 1.15, positive.max() * 1.15))
        assert fig.axes[0].get_ylim() == (0., 1.)
        assert audit["full_positive_support"] == [positive.min(), positive.max()]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("stem, expected_top", [
    ("branch_gap_synchronization", .6 * 1.06),
    ("branch_target_errors", .6 * 1.06),
    ("synchronization_bound", 1. * 1.06),
])
def test_chronological_limits_use_all_displayed_curves_and_bands_only(tmp_path, stem, expected_top):
    data = inputs(tmp_path)
    frame = data[2][stem].copy()
    frame["minimum"], frame["maximum"] = 0., 1e6
    if stem == "synchronization_bound":
        # The saved maximum-distance curve must not affect artists or limits.
        frame.loc[frame.metric.eq("joint_error"), ["median", "q25", "q75"]] = [1e5, 1e4, 1e6]
    before = frame.copy(deep=True)
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["y_limits"] = [0., 2e6]  # Old presentation limits cannot override curation.
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        assert fig.axes[0].get_ylim() == pytest.approx((0., expected_top))
        assert "drawn IQRs" in audit["display_range_policy"]
        assert frame.equals(before)
    finally:
        plt.close(fig)


def test_peak_axis_contains_full_bin_mass_with_fixed_padding(tmp_path):
    fig, audit = draw(inputs(tmp_path), "branch_gap_peak_step")
    try:
        assert fig.axes[0].get_ylim() == pytest.approx((0., .53))
        assert fig.axes[0].get_xlim() == (0., 2.)
        assert audit["shape_group_counts"]
        assert all(np.array_equal(line.get_xdata(), [0, 1, 2]) for line in fig.axes[0].lines)
    finally:
        plt.close(fig)


def test_margin_keeps_signed_points_zero_guides_and_positive_empty_space(tmp_path):
    data = inputs(tmp_path)
    stem = "posterior_feedback_condition_margin"
    frame = data[2][stem].copy()
    frame["x"] = [-100., -1., -.1, -.01]
    frame["y"] = [-2., -.2, 0., .3]
    before = frame.copy(deep=True)
    fig, audit = draw(data, stem, frame=frame)
    try:
        ax = fig.axes[0]
        assert ax.get_xlim()[0] < -100. and ax.get_xlim()[1] > 0.
        assert ax.get_ylim()[0] < -2. and ax.get_ylim()[1] > .3
        assert ax.get_yscale() == "symlog"
        assert any(np.all(np.asarray(line.get_xdata()) == 0) for line in ax.lines)
        assert any(np.all(np.asarray(line.get_ydata()) == 0) for line in ax.lines)
        assert audit["finite_pairs"] == 4 and audit["legend_placement"] == "none"
        assert audit["display_policy"] == "finite_saved_values"
        assert ax.get_legend() is None
        assert frame.equals(before)
    finally:
        plt.close(fig)


def test_feedback_shared_zero_has_explicit_common_curve_and_saved_counts(tmp_path):
    data = inputs(tmp_path)
    stem = "posterior_feedback_over_time"
    fig, audit = draw(data, stem)
    try:
        ax = fig.axes[0]
        labels = [label.get_text() for label in ax.get_legend().get_texts()]
        assert any(r"$\Pr(\cdot)=0$ (both groups)" in value for value in labels)
        assert any("SSCD $> 0.75$: 0/30; unresolved 3" in value and "SSCD $\\leq 0.75$: 0/30; unresolved 3" in value for value in labels)
        assert "Condition upper bound" in labels
        assert audit["shared_zero_baseline"]
        assert all(counts == {"positive": 0, "unresolved": 3, "total": 30} for counts in audit["shared_zero_group_counts"].values())
        assert any(np.all(np.asarray(line.get_ydata()) == 0) for line in ax.lines)
        assert not ax.texts
    finally:
        plt.close(fig)
    metadata = deepcopy(data[1]["figures"][stem])
    del metadata["condition_positive_counts"]
    with pytest.raises(TheoryError, match="saved per-group"):
        draw(data, stem, metadata=metadata)


def test_retired_motion_kind_has_no_direct_renderer_route(tmp_path):
    import pandas as pd
    from utils.experiments.theory.paper_registry import previous_paper_registry
    retired = next(entry for entry in previous_paper_registry() if entry["kind"] == "four_motion")
    with pytest.raises(TheoryError, match="retired"):
        _draw(retired, pd.DataFrame(), {}, {})


@pytest.mark.parametrize("stem, column, ylabel", [
    ("branch_gap_per_prompt", "mean_gap_rmse", r"$\|\boldsymbol{\Delta}_t\|/\sqrt{d}$"),
    ("conditional_reference_error_per_prompt", "mean_conditional_error_rmse", r"$e_t(c)/\sqrt{d}$"),
    ("unconditional_reference_error_per_prompt", "mean_unconditional_reference_error_rmse", r"$e_t(\varnothing)/\sqrt{d}$"),
    ("target_probability_per_prompt", "mean_target_probability", r"$p_t$"),
    ("reference_branch_gap_per_prompt", "mean_reference_gap_rmse", r"$\|\bar{\mathbf{x}}_t(c)-\bar{\mathbf{x}}_t(\varnothing)\|/\sqrt{d}$"),
    ("reference_variation_per_prompt", "mean_reference_variation_rmse", r"$V_t/\sqrt{d}$"),
])
def test_prompt_scalar_has_one_fixed_sscd_color_per_saved_prompt_curve(tmp_path, stem, column, ylabel):
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    data = inputs(tmp_path)
    before = data[2][stem].copy(deep=True)
    fig, audit = draw(data, stem)
    try:
        ax = fig.axes[0]
        assert len(ax.lines) == 2 and ax.get_legend() is None
        assert ax.get_xlabel() == r"$T-t$"
        assert ax.get_ylabel() == ylabel
        transitions = stem == "reference_variation_per_prompt"
        last = 1 if transitions else 2
        assert ax.get_xlim() == (0., float(last)) and ax.get_ylim()[0] == 0
        assert audit["chronological_prediction_range"] == [0, last]
        assert audit["prediction_domain"] == ("positive_noise_transitions" if transitions else "pre_update_predictions")
        assert audit["last_pre_update_sentinel_excluded"] is transitions
        assert fig.axes[1].get_ylabel() == "SSCD"
        assert audit["sscd_color_range"] == [0., 1.]
        assert audit["plotted_prompt_target_curves"] == 2
        assert audit["value_column"] == column
        if stem == "target_probability_per_prompt":
            assert ax.get_ylim() == (0., 1.) and audit["value_domain"] == "probability"
        assert audit["complete_seed_cohort_checked"] and not audit["terminal_output_prediction"]
        groups = list(before.groupby(["run_id", "original_index", "record_id", "target_id"], sort=True))
        for line, (_, saved) in zip(ax.lines, groups, strict=True):
            saved = saved.sort_values("step_index")
            np.testing.assert_array_equal(line.get_xdata(), saved.step_index)
            np.testing.assert_allclose(line.get_ydata(), saved[column])
            np.testing.assert_allclose(line.get_color(), colormaps["viridis"](Normalize(0, 1, clip=True)(saved.mean_terminal_sscd.iloc[0])))
        assert before.equals(data[2][stem])
        assert not ax.texts and not ax.get_title()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("stem, column", [
    ("branch_gap_per_prompt", "mean_gap_rmse"),
    ("conditional_reference_error_per_prompt", "mean_conditional_error_rmse"),
    ("unconditional_reference_error_per_prompt", "mean_unconditional_reference_error_rmse"),
    ("target_probability_per_prompt", "mean_target_probability"),
    ("reference_branch_gap_per_prompt", "mean_reference_gap_rmse"),
    ("reference_variation_per_prompt", "mean_reference_variation_rmse"),
])
@pytest.mark.parametrize("mutation", ["missing_step", "changed_color", "changed_seed_count", "changed_seed_ids", "negative_value"])
def test_prompt_scalar_renderer_rejects_inconsistent_saved_cohorts(tmp_path, stem, column, mutation):
    data = inputs(tmp_path)
    frame = data[2][stem].copy()
    if mutation == "missing_step":
        frame = frame.iloc[1:].copy()
    elif mutation == "changed_color":
        frame.loc[0, "mean_terminal_sscd"] += .1
    elif mutation == "changed_seed_count":
        frame.loc[0, "seed_count"] = 1
    elif mutation == "changed_seed_ids":
        frame.loc[0, "seed_ids_json"] = "[0, 2]"
    else:
        frame.loc[0, column] = -1.
    with pytest.raises(TheoryError, match="Per-prompt"):
        draw(data, stem, frame=frame)



@pytest.mark.parametrize("saved_domain", [None, "pre_update_predictions", "terminal_output"])
def test_reference_variation_requires_explicit_matching_transition_domain(tmp_path, saved_domain):
    data = inputs(tmp_path)
    stem = "reference_variation_per_prompt"
    metadata = deepcopy(data[1]["figures"][stem])
    if saved_domain is None:
        metadata.pop("prediction_domain", None)
    else:
        metadata["prediction_domain"] = saved_domain
    with pytest.raises(TheoryError, match="saved prediction_domain"):
        draw(data, stem, metadata=metadata)


@pytest.mark.parametrize("extra_index", [2, 3])
def test_reference_variation_rejects_last_preupdate_and_terminal_zero_sentinels(tmp_path, extra_index):
    import pandas as pd

    data = inputs(tmp_path)
    stem = "reference_variation_per_prompt"
    frame = data[2][stem].copy(deep=True)
    extra = frame.iloc[[0]].copy()
    extra["step_index"] = extra_index
    extra["mean_reference_variation_rmse"] = 0.
    frame = pd.concat([frame, extra], ignore_index=True)
    with pytest.raises(TheoryError, match="every positive-noise transition exactly once"):
        draw(data, stem, frame=frame)


def test_reference_variation_owns_full_transition_and_value_ranges(tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    data = inputs(tmp_path)
    stem = "reference_variation_per_prompt"
    frame = data[2][stem].copy(deep=True)
    frame["mean_reference_variation_rmse"] = [0., 40., 2., 80.]
    before = frame.copy(deep=True)
    metadata = deepcopy(data[1]["figures"][stem])
    metadata.update(x_limits=[0., 2.], y_limits=[0., .1],
                    numerical_scope="Saved finite variation estimates; condition-sign resolution is reported separately")
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        assert ax.get_xlim() == (0., 1.) and ax.get_ylim()[0] == 0.
        assert ax.get_ylim()[1] > 80.
        np.testing.assert_array_equal(np.concatenate([line.get_ydata() for line in ax.lines]), before.mean_reference_variation_rmse)
        assert audit["numerical_scope"] == metadata["numerical_scope"]
        assert audit["prediction_domain"] == "positive_noise_transitions"
        assert audit["last_pre_update_sentinel_excluded"]
        assert ax.get_legend() is None and not ax.texts and not ax.get_title()
        assert fig.axes[1].get_ylabel() == "SSCD"
        FigureCanvasAgg(fig).draw()
        assert fig.axes[1].get_position().height == pytest.approx(ax.get_position().height)
        assert frame.equals(before)
    finally:
        plt.close(fig)


def test_reference_variation_two_predictions_plot_one_positive_noise_transition(tmp_path):
    data = inputs(tmp_path)
    stem = "reference_variation_per_prompt"
    frame = data[2][stem].loc[lambda rows: rows.step_index.eq(0)].copy()
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["prediction_steps"] = 2
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        assert ax.get_xlim() == (-.2, .2)
        assert audit["chronological_prediction_range"] == [0, 0]
        assert len(ax.lines) == 2
        for line in ax.lines:
            np.testing.assert_array_equal(line.get_xdata(), [0])
            assert line.get_marker() == "."
    finally:
        plt.close(fig)


def test_prompt_probability_renderer_rejects_out_of_range_and_wrong_quantity(tmp_path):
    data = inputs(tmp_path)
    stem = "target_probability_per_prompt"
    frame = data[2][stem].copy()
    frame.loc[0, "mean_target_probability"] = 1.01
    with pytest.raises(TheoryError, match="probabilities must lie"):
        draw(data, stem, frame=frame)
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["value_column"] = "mean_unconditional_reference_error_rmse"
    with pytest.raises(TheoryError, match="saved value_column"):
        draw(data, stem, metadata=metadata)



def test_guidance_fit_sqrt_displays_saved_squared_loss_without_changing_fit_or_cache(tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from utils.experiments.plotting import SCATTER_ALPHA, SCATTER_SIZE

    data = inputs(tmp_path)
    stem = "corollary3_guidance_scale_vs_loss"
    frame = data[2][stem].copy(deep=True)
    assert len(frame) == 4
    frame["x"] = [0., 1., 4., 9.]
    frame["y"] = [-1., 0., 5., 12.]
    frame["mean_terminal_sscd"] = [-.1, .3, .8, 1.1]
    frame["guidance_scale"] = 2.25
    data[0]["scientific_config"]["guidance_scale"] = 2.25
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["guidance_scale"] = 2.25
    before = frame.copy(deep=True)
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        scatter = ax.collections[0]
        np.testing.assert_array_equal(scatter.get_offsets(), np.column_stack([np.sqrt(frame.x), frame.y]))
        np.testing.assert_array_equal(scatter.get_array(), frame.mean_terminal_sscd)
        assert np.all(scatter.get_sizes() == SCATTER_SIZE)
        assert scatter.get_alpha() == SCATTER_ALPHA
        assert scatter.cmap.name == "viridis" and (scatter.norm.vmin, scatter.norm.vmax) == (0., 1.)
        assert ax.get_xscale() == ax.get_yscale() == "linear"
        assert ax.get_xlim()[0] == 0. and 3. < ax.get_xlim()[1] < 9.
        assert ax.get_ylim()[0] < -1. and ax.get_ylim()[1] > 12.
        assert ax.get_xlabel() == data[3][stem]["axes"]["x"]
        assert ax.get_ylabel() == data[3][stem]["axes"]["y"]
        guide = [line for line in ax.lines if line.get_label() == r"$g=2.25$"]
        assert len(guide) == 1 and list(guide[0].get_ydata()) == [2.25, 2.25]
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [r"$g=2.25$"]
        assert not ax.texts and not ax.get_title()
        assert fig.axes[1].get_ylabel() == "SSCD"
        FigureCanvasAgg(fig).draw()
        assert fig.axes[1].get_position().height == pytest.approx(ax.get_position().height)
        assert audit["plotted_prompt_target_pairs"] == 4 and audit["excluded_compact_rows"] == 0
        assert audit["guidance_reference_value"] == 2.25
        assert audit["square_root_applied_to_loss"] and audit["x_transform"] == "sqrt"
        assert audit["saved_horizontal_coordinate"] == "L_T(c)/(d*SNR_T)"
        assert not audit["dimensional_normalization_applied_by_renderer"]
        assert not audit["guidance_estimate_clipping"] and not audit["fit_recomputed_by_renderer"]
        assert audit["complete_seed_cohort_checked"] and audit["pair_identity_uniqueness_checked"]
        assert audit["out_of_color_range_pairs"] == 2
        assert frame.equals(before)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("column, value", [
    ("x", -1.), ("x", np.nan), ("y", np.inf), ("mean_terminal_sscd", np.nan),
    ("seed_count", 1), ("seed_ids_json", "[0, 2]"), ("seed_ids_json", "[0, true]"),
    ("latent_dimension", 4.5), ("snr", 0.), ("guidance_scale", 9.),
    ("fit_residual_rmse", -1.), ("direction_norm_rmse", 0.),
])
def test_guidance_fit_rejects_invalid_saved_rows_instead_of_filtering(tmp_path, column, value):
    data = inputs(tmp_path)
    stem = "corollary3_guidance_scale_vs_loss"
    frame = data[2][stem].copy(deep=True)
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(TheoryError, match="Guidance-fit"):
        draw(data, stem, frame=frame)


@pytest.mark.parametrize("change", [
    {"guidance_scale": 9.}, {"expected_seed_count": 3}, {"expected_seed_ids": [0, 2]},
    {"aggregation": "mean_of_individual_fit_coefficients"}, {"fit_definition": ""},
    {"actual_initial_snr": .5}, {"status": "error"}, {"counts": {"plotted_pairs": 100}},
])
def test_guidance_fit_rejects_inconsistent_reduction_and_population_receipts(tmp_path, change):
    data = inputs(tmp_path)
    stem = "corollary3_guidance_scale_vs_loss"
    metadata = deepcopy(data[1]["figures"][stem])
    metadata.update(change)
    with pytest.raises(TheoryError, match="Guidance-fit"):
        draw(data, stem, metadata=metadata)


def test_guidance_fit_rejects_duplicate_or_missing_prompt_identity(tmp_path):
    data = inputs(tmp_path)
    stem = "corollary3_guidance_scale_vs_loss"
    frame = data[2][stem].copy(deep=True)
    frame.iloc[-1] = frame.iloc[0]
    with pytest.raises(TheoryError, match="unique complete prompt-target"):
        draw(data, stem, frame=frame)
    frame = data[2][stem].copy(deep=True)
    frame.loc[0, "record_id"] = None
    with pytest.raises(TheoryError, match="unique complete prompt-target"):
        draw(data, stem, frame=frame)


def test_guidance_fit_degenerate_axes_still_show_every_saved_zero_loss_point(tmp_path):
    data = inputs(tmp_path)
    stem = "corollary3_guidance_scale_vs_loss"
    frame = data[2][stem].copy(deep=True)
    guidance = data[0]["scientific_config"]["guidance_scale"]
    frame["x"], frame["y"] = 0., guidance
    fig, audit = draw(data, stem, frame=frame)
    try:
        ax = fig.axes[0]
        assert len(ax.collections[0].get_offsets()) == len(frame)
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), np.column_stack([np.sqrt(frame.x), frame.y]))
        assert ax.get_xlim()[0] == 0. < ax.get_xlim()[1]
        assert ax.get_ylim()[0] < guidance < ax.get_ylim()[1]
        assert audit["excluded_compact_rows"] == 0
    finally:
        plt.close(fig)



def test_guidance_fit_ignores_stale_limits_that_hide_saved_points_or_configured_guidance(tmp_path):
    data = inputs(tmp_path)
    stem = "corollary3_guidance_scale_vs_loss"
    frame = data[2][stem].copy(deep=True)
    frame["y"] = [-.5, 0., .5, 1.]
    guidance = data[0]["scientific_config"]["guidance_scale"]
    metadata = deepcopy(data[1]["figures"][stem])
    metadata.update(x_limits=[.001, .02], y_limits=[0., .01])
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), np.column_stack([np.sqrt(frame.x), frame.y]))
        assert ax.get_xlim()[0] == 0. and ax.get_xlim()[1] > np.sqrt(frame.x.max())
        assert ax.get_ylim()[0] < frame.y.min()
        assert ax.get_ylim()[1] > max(guidance, frame.y.max())
        assert audit["guidance_reference_value"] == guidance
    finally:
        plt.close(fig)


@pytest.mark.parametrize("with_classifications", [True, False])
def test_margin_draws_every_finite_saved_pair_as_circle_without_status_legend(tmp_path, with_classifications):
    data = inputs(tmp_path)
    stem = "posterior_feedback_condition_margin"
    frame = data[2][stem].reindex(range(6)).copy()
    frame["x"] = [-100., -1., -.1, -.01, np.nan, np.inf]
    frame["y"] = [-2., -.2, 0., .3, 1., -1.]
    frame["terminal_sscd"] = [.2, np.nan, np.inf, .9, .5, .5]
    frame["marker_class"] = ["observed", "numerically_unresolved", "unrecognized_audit_status", None, "observed", "observed"]
    frame["condition_sign_status"] = ["positive", "unresolved", "negative", "unavailable", "positive", "negative"]
    frame["fixed_cache_gain_sign"] = ["negative", "unresolved", "zero", "unavailable", "positive", "negative"]
    frame["applicable"] = [True, False, False, None, True, True]
    if not with_classifications:
        frame = frame.drop(columns=["marker_class", "condition_sign_status", "fixed_cache_gain_sign", "applicable"])
    before = frame.copy(deep=True)
    fig, audit = draw(data, stem, frame=frame)
    try:
        ax = fig.axes[0]
        collections = [item for item in ax.collections if isinstance(item, PathCollection)]
        offsets = np.vstack([np.asarray(item.get_offsets()) for item in collections])
        assert sorted(map(tuple, offsets)) == sorted(map(tuple, frame.loc[:3, ["x", "y"]].to_numpy()))
        circle = MarkerStyle("o")
        circle_path = circle.get_path().transformed(circle.get_transform())
        for item in collections:
            assert np.array_equal(item.get_paths()[0].codes, circle_path.codes)
            assert np.allclose(item.get_paths()[0].vertices, circle_path.vertices)
        grey = [item for item in collections if item.get_array() is None]
        assert sum(len(item.get_offsets()) for item in grey) == 2
        assert all(np.allclose(item.get_facecolors()[:, :3], to_rgba(".5")[:3]) for item in grey)
        assert ax.get_legend() is None and not ax.texts
        assert audit["display_policy"] == "finite_saved_values"
        assert audit["legend_placement"] == "none"
        assert audit["finite_pairs"] == 4
        assert fig.axes[1].get_ylabel() == "SSCD"
        assert frame.equals(before)
    finally:
        plt.close(fig)


# The historical inventory above remains an explicit renderer reference. These
# tests opt into the six selected publication routes without changing tables.
def _styled_data(data):
    from utils.experiments.theory.paper_registry import paper_registry
    entries = dict(data[3])
    entries.update({entry["stem"]: entry for entry in paper_registry()})
    return (*data[:3], entries)


def _assert_same_line_and_band_arrays(before, after):
    assert len(before.lines) == len(after.lines)
    for old, new in zip(before.lines, after.lines, strict=True):
        np.testing.assert_array_equal(old.get_xdata(), new.get_xdata())
        np.testing.assert_array_equal(old.get_ydata(), new.get_ydata())
    assert len(before.collections) == len(after.collections)
    for old, new in zip(before.collections, after.collections, strict=True):
        assert len(old.get_paths()) == len(new.get_paths())
        for old_path, new_path in zip(old.get_paths(), new.get_paths(), strict=True):
            np.testing.assert_array_equal(old_path.vertices, new_path.vertices)
    assert before.get_xscale() == after.get_xscale()
    assert before.get_yscale() == after.get_yscale()
    np.testing.assert_array_equal(before.get_xlim(), after.get_xlim())
    np.testing.assert_array_equal(before.get_ylim(), after.get_ylim())


def test_selected_reference_theme_preserves_every_curve_band_and_native_cutoff(tmp_path):
    from utils.experiments.theory import paper_style as theme
    data = inputs(tmp_path)
    stem = "unconditional_reference_convergence"
    before = data[2][stem].copy(deep=True)
    old, old_audit = draw(data, stem)
    styled, audit = draw(_styled_data(data), stem)
    try:
        ax = styled.axes[0]
        _assert_same_line_and_band_arrays(old.axes[0], ax)
        colors = [line.get_color() for line in ax.lines]
        assert theme.ANALYTICAL_COLOR in colors and theme.LEARNED_COLOR in colors and theme.ERROR_COLOR in colors
        reference_lines = [line for line in ax.lines if line.get_color() == theme.ANALYTICAL_COLOR]
        assert {line.get_linestyle() for line in reference_lines} == {"-", ":"}
        for color in (theme.LEARNED_COLOR, theme.ERROR_COLOR):
            for line in ax.lines:
                if line.get_color() == color:
                    assert np.asarray(line.get_xdata()).min() >= audit["analytical_only_range"][1]
        assert all(band.get_alpha() == theme.BAND_ALPHA for band in ax.collections)
        note = next(text for text in ax.texts if text.get_text() == "Analytical reference only")
        assert note.get_ha() == "left" and note.get_fontsize() == theme.ANNOTATION_FONT_SIZE
        low, high = audit["analytical_only_range"]
        assert low < note.get_position()[0] < np.sqrt(low * high)
        assert len(styled.axes) == 1 and audit["band_coordinates_unchanged"]
        assert data[2][stem].equals(before)
        assert audit["horizontal_reference_value"] == old_audit["horizontal_reference_value"]
    finally:
        plt.close(old)
        plt.close(styled)


def test_selected_guidance_matches_sparse_palette_and_anchors_saved_g_legend(tmp_path):
    from utils.experiments.theory import paper_style as theme
    data = inputs(tmp_path)
    stem = "corollary3_guidance_scale_vs_loss"
    frame = data[2][stem].copy(deep=True)
    frame["x"], frame["y"] = [0., 1., 4., 9.], [-8., 0., 5., 25.]
    frame["mean_terminal_sscd"] = [-.1, .3, .8, 1.1]
    frame["guidance_scale"] = 2.25
    data[0]["scientific_config"]["guidance_scale"] = 2.25
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["guidance_scale"] = 2.25
    before = frame.copy(deep=True)
    fig, audit = draw(_styled_data(data), stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        points = ax.collections[0]
        np.testing.assert_array_equal(points.get_offsets(), np.column_stack([np.sqrt(frame.x), frame.y]))
        np.testing.assert_array_equal(points.get_array(), frame.mean_terminal_sscd)
        assert points.cmap.name == theme.SSCD_CMAP.name
        assert (points.norm.vmin, points.norm.vmax) == (0., 1.)
        assert np.all(points.get_sizes() == theme.SPARSE_SIZE) and points.get_alpha() == theme.SPARSE_ALPHA
        assert not points.get_rasterized()
        legend = ax.get_legend()
        assert legend is not None and legend._ncols == 1 and legend._loc == 1
        assert not legend.get_frame_on() and not ax.texts
        assert [text.get_text() for text in legend.get_texts()] == [r"$g=2.25$"]
        guide = next(line for line in ax.lines if line.get_label() == r"$g=2.25$")
        assert len(legend.legend_handles) == 1
        handle = legend.legend_handles[0]
        assert handle.get_linestyle() == guide.get_linestyle() == "--"
        np.testing.assert_allclose(to_rgba(handle.get_color()), to_rgba(guide.get_color()))
        np.testing.assert_array_equal(guide.get_ydata(), [2.25, 2.25])
        assert audit["legend_placement"] == "right edge below reference line"
        assert audit["guidance_label_placement"] == "right edge below reference line with point offset"

        def assert_mixed_transform_offset():
            # Transform evaluation needs no canvas draw: x follows the axes'
            # right edge, y follows the saved guidance in data coordinates.
            baseline = ax.get_yaxis_transform().transform((1., 2.25))
            anchor = legend.get_bbox_to_anchor().p0
            np.testing.assert_allclose((anchor - baseline) * 72. / fig.dpi, [-4., -5.], atol=1e-10)

        assert_mixed_transform_offset()
        assert ax.get_ylim()[0] < -8. and ax.get_ylim()[1] > 25.
        assert fig.axes[1].get_ylabel() == "SSCD" and audit["guidance_reference_value"] == 2.25
        np.testing.assert_allclose(fig.axes[1].get_yticks(), np.linspace(0., 1., 6))
        assert audit["out_of_color_range_pairs"] == 2 and frame.equals(before)
        ax.set_xlim(0., 12.)
        ax.set_ylim(-20., 100.)
        assert_mixed_transform_offset()
        np.testing.assert_array_equal(points.get_offsets(), np.column_stack([np.sqrt(frame.x), frame.y]))
        assert frame.equals(before)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("stem,expected_count", [("synchronization_bound", 4), ("terminal_bound_coverage", 6)])
def test_selected_grouped_themes_preserve_all_arrays_and_two_column_semantic_legend(tmp_path, stem, expected_count):
    from matplotlib.legend import Legend
    from utils.experiments.theory import paper_style as theme
    from utils.experiments.theory.paper_registry import GROUPS, GROUP_LABELS
    from utils.experiments.theory.paper_notation import CHRONOLOGICAL_LABELS, TERMINAL_EXPRESSIONS, TERMINAL_LABELS
    data = inputs(tmp_path)
    frame = data[2][stem].copy(deep=True)
    if stem == "terminal_bound_coverage":
        # The conservative reference tail must remain in the shared display.
        frame.loc[frame.distribution.eq("reference"), "value"] *= 100.
    else:
        frame.loc[frame.metric.eq("bound"), ["median", "q25", "q75"]] *= 100.
    before = frame.copy(deep=True)
    old, _ = draw(data, stem, frame=frame)
    styled, audit = draw(_styled_data(data), stem, frame=frame)
    try:
        ax = styled.axes[0]
        if stem == "terminal_bound_coverage":
            _assert_same_line_and_band_arrays(old.axes[0], ax)
            assert not ax.collections  # Saved CDFs have no confidence-band inputs.
        else:
            # Newly displayed saved IQRs can expand y limits but never change medians.
            assert len(old.axes[0].lines) == len(ax.lines)
            for old_line, new_line in zip(old.axes[0].lines, ax.lines, strict=True):
                np.testing.assert_array_equal(old_line.get_xdata(), new_line.get_xdata())
                np.testing.assert_array_equal(old_line.get_ydata(), new_line.get_ydata())
            assert old.axes[0].get_xlim() == ax.get_xlim()
            assert old.axes[0].get_yscale() == ax.get_yscale()
            assert not old.axes[0].collections and len(ax.collections) == 4
        assert len(ax.lines) == expected_count
        assert all(line.get_linewidth() == theme.CURVE_WIDTH and line.get_alpha() == theme.CURVE_ALPHA for line in ax.lines)
        width = expected_count // 2
        for index, group in enumerate(GROUPS):
            for line in ax.lines[index * width:(index + 1) * width]:
                np.testing.assert_allclose(to_rgba(line.get_color()), to_rgba(theme.GROUP_COLORS[group]))
        legends = [artist for artist in ax.get_children() if isinstance(artist, Legend)]
        assert len(legends) == 1
        legend = legends[0]
        assert legend is ax.get_legend() and legend._ncols == 2
        group_labels = [GROUP_LABELS[group] for group in GROUPS]
        assert group_labels == [r"SSCD $> 0.75$", r"SSCD $\leq 0.75$"]
        mathematical_key = ([TERMINAL_LABELS[name] for name in ("actual", "observable", "reference")]
                            if stem == "terminal_bound_coverage" else
                            [CHRONOLOGICAL_LABELS[metric] for metric in ("gap", "bound")])
        rows_per_column = len(mathematical_key)
        first_column = group_labels + [""] * (rows_per_column - len(group_labels))
        labels = [text.get_text() for text in legend.get_texts()]
        # Matplotlib fills each column from top to bottom: the blank terminal
        # handle must remain after both groups, never amongst quantity labels.
        assert labels == first_column + mathematical_key
        assert labels[:rows_per_column] == first_column
        assert labels[rows_per_column:] == mathematical_key
        handles = legend.legend_handles
        for index, group in enumerate(GROUPS):
            np.testing.assert_allclose(to_rgba(handles[index].get_color()), to_rgba(theme.GROUP_COLORS[group]))
        if rows_per_column > len(group_labels):
            assert handles[len(group_labels)].get_alpha() == 0.
        for handle in handles[rows_per_column:]:
            np.testing.assert_allclose(to_rgba(handle.get_color()), to_rgba(theme.TEXT_COLOR))
        anchor = ax.transAxes.inverted().transform(legend.get_bbox_to_anchor().p0)
        assert anchor[1] > 1.  # The sole legend remains above the data axes.
        if stem == "terminal_bound_coverage":
            assert audit["legend_quantity_expressions"] == TERMINAL_EXPRESSIONS
            assert audit["quantity_styles"]["reference"] == theme.LINE_STYLES["reference"]
            assert ax.get_xlim()[1] >= frame.loc[np.isfinite(frame.value), "value"].max()
        else:
            assert audit["quantity_styles"] == {"gap": theme.LINE_STYLES["observed"], "bound": theme.LINE_STYLES["observable"]}
            assert ax.get_ylim()[1] > frame.loc[frame.metric.isin(["gap", "bound"]), "q75"].max()
        assert frame.equals(before) and audit["legend_placement"] == "outside top"
        assert audit["legend_columns"] == "SSCD groups; manuscript quantities"
    finally:
        plt.close(old)
        plt.close(styled)


def test_selected_theme_does_not_change_later_historical_renderer_or_rcparams(tmp_path):
    from utils.experiments.theory.paper_registry import GROUP_COLORS
    data = inputs(tmp_path)
    before = dict(plt.rcParams)
    styled, _ = draw(_styled_data(data), "synchronization_bound")
    legacy, _ = draw(data, "synchronization_bound")
    try:
        assert dict(plt.rcParams) == before
        assert all(line.get_color() in GROUP_COLORS.values() for line in legacy.axes[0].lines)
    finally:
        plt.close(styled)
        plt.close(legacy)



def test_selected_synchronization_bands_use_exact_saved_quartiles_and_segments(tmp_path):
    from utils.experiments.theory import paper_style as theme
    from utils.experiments.theory.paper_registry import GROUPS
    data = inputs(tmp_path)
    stem = "synchronization_bound"
    frame = data[2][stem].copy(deep=True)
    frame["segment_id"] = np.where(frame.step_index < 2, "early", "last")
    frame["q25"] = frame["median"] * .4
    frame["q75"] = frame["median"] + 20. + frame["step_index"]
    before = frame.copy(deep=True)
    metadata = deepcopy(data[1]["figures"][stem])
    metadata_before = deepcopy(metadata)
    fig, audit = draw(_styled_data(data), stem, frame=frame, metadata=metadata)
    legacy, _ = draw(data, stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        expected = []
        for group in GROUPS:
            for metric in ("gap", "bound"):
                rows = frame.loc[frame.group.eq(group) & frame.metric.eq(metric)].sort_values("step_index")
                expected.extend((group, part) for _, part in rows.groupby("segment_id", sort=False))
        assert len(ax.collections) == len(ax.lines) == len(expected)
        assert len(expected) == 8 and not legacy.axes[0].collections
        assert audit["band_metrics"] == ["gap", "bound"]
        assert audit["band_endpoints_recomputed"] is False
        assert "not a confidence interval" in audit["distribution_band"]
        for band, line, (group, rows) in zip(ax.collections, ax.lines, expected, strict=True):
            assert band.get_alpha() == theme.BAND_ALPHA
            assert band.get_zorder() == 1 < line.get_zorder()
            np.testing.assert_allclose(band.get_facecolors()[0, :3], to_rgba(theme.GROUP_COLORS[group])[:3])
            np.testing.assert_array_equal(line.get_xdata(), rows.step_index)
            np.testing.assert_array_equal(line.get_ydata(), rows["median"])
            assert len(band.get_paths()) == 1
            vertices = band.get_paths()[0].vertices
            np.testing.assert_array_equal(np.unique(vertices[:, 0]), np.sort(rows.step_index.to_numpy()))
            for row in rows.itertuples():
                ordinates = vertices[vertices[:, 0] == row.step_index, 1]
                assert ordinates.min() == row.q25
                assert ordinates.max() == row.q75
        displayed = frame.loc[frame.metric.isin(["gap", "bound"])]
        assert ax.get_ylim()[0] <= displayed.q25.min()
        assert ax.get_ylim()[1] > displayed.q75.max()
        assert frame.equals(before) and metadata == metadata_before
    finally:
        plt.close(fig)
        plt.close(legacy)
