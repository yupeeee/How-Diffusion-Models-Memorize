"""Unexecuted fixed-design renderer regressions using only synthetic scalars."""

from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
import pandas as pd
import pytest

from utils.experiments.plotting import PLOT_STYLE
from utils.experiments.theory import paper_plotting as plotting
from utils.experiments.theory.contracts import TheoryError
from tests.test_theory_paper_plotting import load_measurement_inputs as load_paper_inputs
from utils.experiments.theory.paper_registry import evidence_registry as paper_registry
from tests.test_theory_paper_plotting import compact_fixture


def inputs(tmp_path):
    root = compact_fixture(tmp_path / "paper", terminal=False, diagnostics=True)
    config, summary, _, frames = load_paper_inputs(root, diagnostics=True)
    entries = {entry["stem"]: entry for entry in paper_registry()}
    return config, summary, frames, entries


def draw(data, stem, *, frame=None, metadata=None):
    config, summary, frames, entries = data
    with plt.rc_context(PLOT_STYLE):
        return plotting._draw(entries[stem], frames[stem] if frame is None else frame,
                              summary["figures"][stem] if metadata is None else metadata,
                              config)


def test_pair_loss_has_distinct_mean_sscd_and_control_without_equality(tmp_path):
    data = inputs(tmp_path)
    frame = data[2]["theorem1_loss_recovery"].copy()
    frame.loc[0, "control_y"] = .1  # The connector rule also keeps unfavorable comparisons.
    fig, audit = draw(data, "theorem1_loss_recovery", frame=frame)
    try:
        ax = fig.axes[0]
        assert ax.get_xlim()[0] == 0 and ax.get_ylim()[0] == 0
        assert fig.axes[1].get_ylabel() == "Mean terminal SSCD"
        assert not audit["equality_guide"]
        assert audit["paired_connector_count"] == 4
        segments = ax.collections[0].get_segments()
        assert len(segments) == 4
        for segment, row in zip(segments, frame.itertuples()):
            np.testing.assert_array_equal(segment[:, 0], [row.x, row.x])
            np.testing.assert_allclose(segment[:, 1], sorted([row.y, row.control_y]))
        assert len(ax.collections[1].get_offsets()) == 4
        assert len(ax.collections[2].get_offsets()) == 4
        assert len(ax.collections[1].get_facecolors()) == 0
        assert ax.get_ylabel() == "Initial target error (RMS)"
        assert all(line.get_label() != "Equality" for line in ax.lines)
        assert any("forward draws/pair" in text.get_text() for text in ax.texts)
        assert "Monte Carlo bootstrap intervals" in [text.get_text() for text in ax.get_legend().get_texts()]
    finally:
        plt.close(fig)


def test_reference_primary_has_only_initial_network_markers(tmp_path):
    data = inputs(tmp_path)
    stem = "lemma2_unconditional_baseline"
    fig, audit = draw(data, stem)
    try:
        ax = fig.axes[0]
        assert ax.get_xscale() == "log" and ax.get_ylim()[0] == 0
        assert audit["analytical_only_range"] == [1e-8, .01]
        assert audit["native_initial_markers_only"]
        assert not audit["scale_reference_rendered"]
        for color in ("#e67d24", "#3b9955"):
            learned = [line for line in ax.lines if line.get_color() == color]
            assert learned
            for line in learned:
                np.testing.assert_array_equal(line.get_xdata(), [.01])
                assert line.get_linestyle() == "None"
        assert ax.get_xlim()[1] < .1
        assert data[1]["figures"][stem]["native_sweep_figure"] == "appendix/lemma2_native_gaussian_sweep"
    finally:
        plt.close(fig)
    wrong = data[2][stem].copy()
    wrong.loc[wrong.metric.eq("learned"), "source_range"] = "analytical"
    with pytest.raises(TheoryError, match="Only the analytical reference"):
        draw(data, stem, frame=wrong)
    wrong = data[2][stem].copy()
    wrong.loc[wrong.metric.eq("learned"), "snr"] = .1
    with pytest.raises(TheoryError, match="one saved summary at SNR_T"):
        draw(data, stem, frame=wrong)
    assert not plt.get_fignums()


def test_complete_native_sweep_keeps_late_errors_and_breaks_missing_labels(tmp_path):
    data = inputs(tmp_path)
    stem = "lemma2_native_gaussian_sweep"
    frame = data[2][stem].copy()
    frame = frame.loc[~(frame.metric.eq("learned") & frame.step_index.eq(1))]
    frame.loc[frame.metric.eq("learned") & frame.step_index.eq(2), "segment_id"] = 1
    fig, audit = draw(data, stem, frame=frame)
    try:
        ax = fig.axes[0]
        learned = [line for line in ax.lines if line.get_color() == "#e67d24"]
        assert len(learned) == 2
        assert all(np.min(line.get_xdata()) >= .01 for line in learned)
        assert max(np.max(line.get_xdata()) for line in learned) == 10
        assert ax.get_ylim()[1] > 2.7
        assert audit["complete_saved_native_range"] and not audit["native_initial_markers_only"]
    finally:
        plt.close(fig)


def test_injection_geometry_keeps_off_axis_and_negative_alignment(tmp_path):
    data = inputs(tmp_path)
    stem = "corollary3_initial_cfg_amplification"
    fig, audit = draw(data, stem)
    try:
        ax = fig.axes[0]
        assert ax.get_aspect() == 1
        assert ax.get_xlim()[0] < -1 and ax.get_xlim()[1] > 4
        assert ax.get_ylim()[1] > 3
        assert audit["ideal_location"] == [1, 0]
        assert audit["relative_error_contours"] == [.25, .5, 1.]
        assert not audit["fitted_line"]
        np.testing.assert_array_equal(ax.lines[-1].get_xdata(), [1])
        np.testing.assert_array_equal(ax.lines[-1].get_ydata(), [0])
        assert fig.axes[1].get_ylabel() == "SSCD"
        assert ax.get_ylim()[0] < 0 < ax.get_ylim()[1]
        assert {"0.25", "0.5", "1"} <= {text.get_text() for text in ax.texts}
        assert len(ax.get_legend().get_texts()) == 1
    finally:
        plt.close(fig)


def test_feedback_retains_shared_zero_condition_reversal_and_unknown_mass(tmp_path):
    data = inputs(tmp_path)
    fig, audit = draw(data, "proposition5_posterior_feedback")
    try:
        ax = fig.axes[0]
        assert ax.get_ylim() == (0, 1) and ax.get_xscale() == "log"
        solid = [line for line in ax.lines if line.get_linestyle() == "-"]
        assert len(solid) == 2
        for line in solid:
            np.testing.assert_allclose(line.get_ydata(), [.2, .5, .1])
        dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(dashed) == 1
        np.testing.assert_array_equal(dashed[0].get_ydata(), [0, 0])
        assert not dashed[0].get_clip_on()
        boundaries = [line for line in ax.lines if line.get_linestyle() == ":" and line.get_color() != ".45"]
        assert len(boundaries) == 2
        for line in boundaries:
            np.testing.assert_allclose(line.get_ydata(), [.1, .1, .1])
        assert len(ax.collections) == 2
        assert all(collection.get_hatch() in {None, ""} for collection in ax.collections)
        assert audit["shared_zero_baseline"] and audit["condition_unresolved_observation_count"] == 6
        assert "not statistical uncertainty" in audit["band_definition"]
        assert any("Unresolved condition sample-transitions: 6" in text.get_text() for text in ax.texts)
        assert len([artist for artist in ax.get_children() if isinstance(artist, Legend)]) == 2
    finally:
        plt.close(fig)


def test_feedback_never_uses_shared_baseline_for_nonzero_condition(tmp_path):
    data = inputs(tmp_path)
    stem = "proposition5_posterior_feedback"
    frame = data[2][stem].copy()
    chosen = frame.index[frame.metric.eq("condition")][0]
    frame.loc[chosen, ["fraction", "upper_fraction"]] = [.2, .9]
    fig, audit = draw(data, stem, frame=frame)
    try:
        assert not audit["shared_zero_baseline"]
        assert not any("Both original-condition" in text.get_text() for text in fig.axes[0].texts)
        dashed = [line for line in fig.axes[0].lines if line.get_linestyle() == "--"]
        assert len(dashed) == 2
        assert any(np.max(line.get_ydata()) == .2 for line in dashed)
        assert len(fig.axes[0].collections) == 2
    finally:
        plt.close(fig)


def test_feedback_large_unresolved_mass_remains_visible(tmp_path):
    data = inputs(tmp_path)
    stem = "proposition5_posterior_feedback"
    frame = data[2][stem].copy()
    frame.loc[frame.metric.eq("feedback"), "upper_fraction"] = 1.
    fig, _ = draw(data, stem, frame=frame)
    try:
        assert len(fig.axes[0].collections) == 2
        for band in fig.axes[0].collections:
            assert max(path.vertices[:, 1].max() for path in band.get_paths()) == 1.
        assert fig.axes[0].get_ylim() == (0, 1)
    finally:
        plt.close(fig)


def test_numerical_resolution_keeps_three_layers_and_all_steps(tmp_path):
    data = inputs(tmp_path)
    fig, audit = draw(data, "proposition5_numerical_resolution")
    try:
        ax = fig.axes[0]
        assert ax.get_ylim() == (0, 1) and ax.get_xscale() == "log"
        assert len(ax.lines) == 7
        assert [line.get_linestyle() for line in ax.lines[:6]] == ["-", "--", "-.", "-", "--", "-."]
        for line in ax.lines[:6]:
            np.testing.assert_allclose(line.get_xdata(), [.01, .1, 10.])
            np.testing.assert_allclose(line.get_ydata(), [.3, .8, .6])
        assert audit["numerical_resolution_table"] == "audit_data/proposition5_numerical_causes.csv"
    finally:
        plt.close(fig)


def test_matched_displacement_is_neutral_and_labels_actual_construction(tmp_path):
    data = inputs(tmp_path)
    fig, audit = draw(data, "lemma4_matched_displacement")
    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_aspect() == 1
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "Independent affine update" in labels
        assert "Constructed innovation" in labels
        for collection in ax.collections:
            np.testing.assert_allclose(collection.get_facecolors()[0, :3], [.35] * 3)
        assert "constructed_shared_innovation_from_saved_endpoint_not_independent" in audit["marker_counts"]
    finally:
        plt.close(fig)


def test_synchronization_draws_all_three_quantities_and_only_joint_error_iqr(tmp_path):
    data = inputs(tmp_path)
    fig, audit = draw(data, "lemma6_target_specific_synchronization")
    try:
        ax = fig.axes[0]
        assert [line.get_linestyle() for line in ax.lines[:6]] == ["-", "--", "-.", "-", "--", "-."]
        assert len(ax.collections) == 2
        assert ax.get_ylim()[0] == 0
        assert ax.get_ylabel() == "Target error, gap, or bound (RMSE)"
        assert audit["shaded_quantities"] == ["joint_error"]
        assert len([artist for artist in ax.get_children() if isinstance(artist, Legend)]) == 2
    finally:
        plt.close(fig)


def test_synchronization_renderer_preserves_Q_above_tightened_D_bound(tmp_path):
    data = inputs(tmp_path)
    stem = "lemma6_target_specific_synchronization"
    frame = data[2][stem].copy(deep=True)
    values = np.where(frame.metric.eq("joint_error"), 4., 0.)
    for column in ("median", "q25", "q75", "minimum", "maximum"):
        frame[column] = values
    metadata = deepcopy(data[1]["figures"][stem])
    metadata.pop("y_limits", None)
    before = frame.copy(deep=True)
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        for line, expected in zip(ax.lines[:6], (4., 0., 0., 4., 0., 0.), strict=True):
            assert np.asarray(line.get_ydata()).size > 0
            np.testing.assert_array_equal(line.get_ydata(), np.full_like(line.get_ydata(), expected))
        assert ax.get_ylim()[0] == 0 and ax.get_ylim()[1] > 4.
        assert audit["shaded_quantities"] == ["joint_error"]
        assert frame.equals(before)
    finally:
        plt.close(fig)


def terminal_support():
    return pd.DataFrame([
        {"distribution": name, "value": value, "cdf": cdf}
        for name, points in {
            "actual": [(0., .25), (1e-7, .5), (1., 1.)],
            "observable": [(0., .25), (1e-5, .5), (10., 1.)],
            "reference": [(1e-3, .25), (1., .5), (100., 1.)],
        }.items() for value, cdf in points
    ])


def test_terminal_log_ecdf_preserves_zeros_full_tails_and_extension_scope(tmp_path):
    data = inputs(tmp_path)
    stem = "theorem7_final_reproduction"
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["zero_mass"] = {"actual": .25, "observable": .25, "reference": 0.}
    frame = terminal_support()
    saved = frame.copy(deep=True)
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_xlim()[0] < 1e-7 and ax.get_xlim()[1] > 100
        assert audit["full_positive_support"] == [1e-7, 100]
        assert audit["zero_replacement"] is None
        assert "extension" in ax.get_title()
        assert all(np.min(line.get_xdata()) > 0 for line in ax.lines)
        assert ax.lines[0].get_ydata()[0] == .25
        assert any("Mass at zero" in text.get_text() for text in ax.texts)
        assert any("0/4 original clean updates" in text.get_text() for text in ax.texts)
        pd.testing.assert_frame_equal(frame, saved)
    finally:
        plt.close(fig)
    wrong = metadata | {"manuscript_extension_required": False}
    with pytest.raises(TheoryError, match="manuscript requirement"):
        draw(data, stem, frame=frame, metadata=wrong)


def test_terminal_zero_mass_annotation_is_omitted_only_when_all_masses_are_zero(tmp_path):
    data = inputs(tmp_path)
    fig, audit = draw(data, "theorem7_final_reproduction")
    try:
        assert not audit["zero_mass_annotation_visible"]
        assert audit["zero_mass"] == {"actual": 0., "observable": 0., "reference": 0.}
        assert not any("Mass at zero" in text.get_text() for text in fig.axes[0].texts)
        assert any("original clean updates" in text.get_text() for text in fig.axes[0].texts)
        assert fig.axes[0].get_ylabel() == "Coverage of latent-error tolerance"
    finally:
        plt.close(fig)


def test_all_zero_terminal_population_never_gets_fake_positive_observations(tmp_path):
    data = inputs(tmp_path)
    stem = "theorem7_final_reproduction"
    frame = pd.DataFrame([dict(distribution=name, value=0., cdf=1.) for name in ("actual", "observable", "reference")])
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["zero_mass"] = {name: 1. for name in frame.distribution}
    metadata["predeclared_tolerance_rmse"] = 0.
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        assert audit["empty_positive_domain"]
        assert audit["full_positive_support"] is None and audit["zero_replacement"] is None
        assert any("All mass is at exactly zero" in text.get_text() for text in fig.axes[0].texts)
        assert any("Predeclared tolerance: 0" in text.get_text() for text in fig.axes[0].texts)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("tolerance", [1e-9, 1e4])
def test_terminal_cdf_plateaus_cover_declared_tolerance_outside_change_points(tmp_path, tolerance):
    data = inputs(tmp_path)
    stem = "theorem7_final_reproduction"
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["zero_mass"] = {"actual": .25, "observable": .25, "reference": 0.}
    metadata["predeclared_tolerance_rmse"] = tolerance
    fig, audit = draw(data, stem, frame=terminal_support(), metadata=metadata)
    try:
        ax = fig.axes[0]
        assert ax.get_xlim()[0] < tolerance < ax.get_xlim()[1]
        for line in ax.lines[:3]:
            np.testing.assert_allclose(line.get_xdata()[[0, -1]], ax.get_xlim())
        assert audit["full_positive_support"] == [1e-7, 100.]
        assert audit["zero_replacement"] is None
    finally:
        plt.close(fig)


def test_plot_bundle_cannot_call_measurements_or_refit_scientific_statistics(tmp_path, monkeypatch):
    root = compact_fixture(tmp_path / "portable")
    from utils.experiments.theory import direct_figures, direct_probes, reference_law
    from utils.experiments.theory import direct_reduce
    from utils.models import loading
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("Portable rendering attempted a measurement or scientific refit")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(np, "quantile", forbidden)
    monkeypatch.setattr(direct_figures, "build_direct_plot_inputs", forbidden)
    monkeypatch.setattr(direct_probes, "run_direct_probes", forbidden)
    monkeypatch.setattr(reference_law, "build_reference_law", forbidden)
    monkeypatch.setattr(direct_reduce, "load_record", forbidden)
    monkeypatch.setattr(loading, "load_model_components", forbidden)
    monkeypatch.setattr(plotting, "publish_figures", lambda root, items, **options: None)
    # Draw compact inputs directly, so no filesystem publication is needed.
    config, summary, _, frames = load_paper_inputs(root, diagnostics=True)
    for entry in paper_registry():
        if summary["figures"][entry["stem"]]["status"] != "available":
            continue
        with plt.rc_context(PLOT_STYLE):
            fig, _ = plotting._draw(entry, frames[entry["stem"]], summary["figures"][entry["stem"]], config)
        plt.close(fig)


def test_terminal_infinite_bound_mass_keeps_finite_cdf_plateau(tmp_path):
    data = inputs(tmp_path)
    stem = "theorem7_final_reproduction"
    frame = terminal_support()
    frame.loc[frame.distribution.eq("reference") & frame.value.eq(100.), "value"] = np.inf
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["zero_mass"] = {"actual": .25, "observable": .25, "reference": 0.}
    metadata["infinite_mass"] = {"actual": 0., "observable": 0., "reference": .5}
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        reference = fig.axes[0].lines[2]
        assert reference.get_ydata()[-1] == .5
        assert np.isfinite(reference.get_xdata()).all()
        assert audit["infinite_mass"]["reference"] == .5
        assert any("Mass at +infinity" in text.get_text() for text in fig.axes[0].texts)
    finally:
        plt.close(fig)


def test_caption_preserves_scope_links_resolution_and_looseness_metadata():
    entries = {entry["stem"]: entry for entry in paper_registry()}
    examples = {
        "lemma2_unconditional_baseline": {
            "reference_scale_rmse": 2.,
            "initial_measurement_scope": "initial network markers only",
        },
        "corollary3_initial_cfg_amplification": {
            "sample_relative_error_quantiles": {"median": 1.2},
            "pair_relative_error_quantiles": {"median": 1.1},
            "geometry_contour_coverage": {"0.25": .1},
            "injection_premise_table": "audit_data/corollary3_premise.csv",
        },
        "proposition5_posterior_feedback": {
            "condition_unresolved_counts": {"group": 12},
            "condition_unresolved_fractions": {"group": [0., .9]},
            "observed_condition_coverage": {"group": [0., 0.]},
            "numerical_layers": {"fixed_cache": "separate from source sensitivity"},
            "numerical_cause_counts_table": "audit_data/proposition5_numerical_causes.csv",
        },
        "theorem7_final_reproduction": {
            "zero_mass": {"actual": 0., "observable": 0., "reference": 0.},
            "original_clean_counts": {"applicable": 0, "total": 4},
            "terminal_scope": "finite_terminal_update_extension_deterministic",
            "looseness_audit_table": "audit_data/terminal_looseness.csv",
            "looseness_identity": "(g-1)*(slack_radius+slack_triangle)",
        },
    }
    for stem, metadata in examples.items():
        entry = deepcopy(entries[stem])
        entry.update(measurement_metadata=metadata, status="available", scientific_hash="synthetic")
        caption = plotting._caption(entry)
        for key in metadata:
            assert key.replace("_", " ").capitalize() + ":" in caption
        if stem == "lemma2_unconditional_baseline":
            assert "appendix/lemma2_native_gaussian_sweep.png" in caption
        if stem == "proposition5_posterior_feedback":
            assert "appendix/proposition5_numerical_resolution.png" in caption
