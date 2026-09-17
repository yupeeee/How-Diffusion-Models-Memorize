"""Author-run appearance regressions; saved proximity scalars are never recomputed."""
from collections import Counter

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.legend import Legend
import numpy as np
import pandas as pd
import pytest

from utils.experiments import plotting


def _observations():
    # Two retained seeds per prompt include negative/low SSCD values; category
    # and inclusion are properties of the original prompt, never of its score.
    rows = []
    for index, (kind, keep, rho) in enumerate((("MV", False, -.6), ("RV", False, .2),
                                            ("TV", True, -.4), ("N", True, None),
                                            ("unknown", False, None))):
        for seed in (0, 1):
            rows.append(dict(model_name="sdv2", scheduler_name="ddim", seed_role="experiment",
                             original_index=f"prompt-{index}", record_id=f"record-{index}", seed=seed,
                             kind=kind, include_prompt=keep, observation_status="complete",
                             l2_norm=float(index * 10 + seed + 1), sscd=(-.31 if seed == 0 else .82),
                             experiment_prompt_spearman=rho))
    return pd.DataFrame(rows)


def _draw(frame):
    return plotting._scatter_figure(frame, plotting.AnalysisStatistics(
        frame["original_index"].nunique(), 0, 0, None, None))


def _assert_summary_box(summary, axis):
    from matplotlib.patches import BoxStyle
    assert summary.get_position() == pytest.approx((.02, .02))
    assert summary.get_transform() is axis.transAxes
    assert summary.get_horizontalalignment() == "left"
    assert summary.get_verticalalignment() == "bottom"
    assert summary.get_fontsize() == 12
    box = summary.get_bbox_patch()
    assert box is not None
    assert isinstance(box.get_boxstyle(), BoxStyle.Round)
    assert box.get_boxstyle().pad == .3
    assert box.get_alpha() == .85
    assert box.get_facecolor() == pytest.approx((1., 1., 1., .85))
    assert box.get_edgecolor() == pytest.approx((.75, .75, .75, .85))


def _drawn_xy(figure):
    points = figure.axes[0].collections
    assert len(points) == 1
    return [tuple(row) for row in points[0].get_offsets().tolist()]


def _palette():
    cmap = matplotlib.colormaps["magma"]
    return {"MV": cmap(.70), "RV": cmap(.20), "TV": "#168C91", "N": "#536B8A",
            "Other / unlabeled": "#858B93"}


def test_display_order_is_repeatable_and_retained_view_is_subsequence_without_rng_changes():
    frame = _observations()
    before = frame.copy(deep=True)
    rng = np.random.get_state()
    figures = []
    try:
        figures.append(_draw(frame))
        figures.append(_draw(frame.iloc[::-1]))
        selected = frame.loc[frame["include_prompt"]]
        figures.append(_draw(selected))
        order = _drawn_xy(figures[0])
        assert order == _drawn_xy(figures[1])
        selected_xy = set(zip(selected["l2_norm"], selected["sscd"]))
        assert _drawn_xy(figures[2]) == [pair for pair in order if pair in selected_xy]
        assert Counter(order) == Counter(zip(frame["l2_norm"], frame["sscd"]))
        assert min(y for _x, y in order) == -.31
        assert figures[0].axes[0].get_ylim()[0] < -.31
        assert all(fig.axes[0].get_xscale() == fig.axes[0].get_yscale() == "linear" for fig in figures)
        assert all(len(fig.axes) == 1 for fig in figures)  # Categorical colors need no SSCD colorbar.
        pd.testing.assert_frame_equal(frame, before)
        after = np.random.get_state()
        assert after[0] == rng[0]
        np.testing.assert_array_equal(after[1], rng[1])
        assert after[2:] == rng[2:]
    finally:
        for fig in figures:
            plt.close(fig)


def test_order_does_not_depend_on_categories_outcomes_or_selection():
    frame = _observations()
    changed = frame.copy(deep=True)
    changed["kind"] = list(reversed(changed["kind"].tolist()))
    changed["sscd"] = np.arange(len(changed), dtype=float) - 3.5
    changed["include_prompt"] = ~changed["include_prompt"]
    changed["experiment_prompt_spearman"] = 0.999
    changed["view"] = "retained"
    figures = [_draw(frame), _draw(changed)]
    try:
        assert [x for x, _y in _drawn_xy(figures[0])] == [x for x, _y in _drawn_xy(figures[1])]
    finally:
        for fig in figures:
            plt.close(fig)


def test_duplicate_coordinates_and_duplicate_records_keep_every_observation():
    frame = _observations()
    # Separate prompt identities share the same coordinates, and one exact row
    # is deliberately repeated. Neither condition authorizes deduplication.
    frame.loc[2, ["l2_norm", "sscd"]] = frame.loc[0, ["l2_norm", "sscd"]].to_numpy()
    repeated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    before = repeated.copy(deep=True)
    fig = _draw(repeated)
    try:
        assert Counter(_drawn_xy(fig)) == Counter(zip(repeated["l2_norm"], repeated["sscd"]))
        assert len(_drawn_xy(fig)) == len(frame) + 1
        pd.testing.assert_frame_equal(repeated, before)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("model", ("sdv1", "sdv2", "realvis"))
@pytest.mark.parametrize("role", ("reference", "experiment"))
def test_tv_n_subset_keeps_original_colors_equal_markers_and_mathtt_legend(model, role):
    frame = _observations()
    frame["model_name"] = model
    frame["seed_role"] = role
    selected = frame.loc[frame["include_prompt"]]
    figures = [_draw(frame), _draw(selected)]
    palette = _palette()
    try:
        for figure, source, categories in ((figures[0], frame, ("MV", "RV", "TV", "N", "Other / unlabeled")),
                                           (figures[1], selected, ("TV", "N"))):
            axis = figure.axes[0]
            points = axis.collections[0]
            assert points.get_alpha() == .35
            assert points.get_sizes().tolist() == [10.]
            assert points.get_rasterized() is True
            assert len(points.get_edgecolors()) == 0
            source_kind = {float(row.l2_norm): plotting._plot_kind(row.kind) for row in source.itertuples()}
            for (x, _y), color in zip(points.get_offsets(), points.get_facecolors(), strict=True):
                assert color == pytest.approx(to_rgba(palette[source_kind[float(x)]], alpha=.35))
            legends = [item for item in axis.get_children() if isinstance(item, Legend)]
            assert len(legends) == 1
            assert legends[0]._loc == 1 and legends[0]._ncols == 1
            assert legends[0].get_title().get_text() == ""
            assert all(not legend.get_frame_on() for legend in legends)
            texts = [text for legend in legends for text in legend.get_texts()]
            handles = [handle for legend in legends for handle in legend.legend_handles]
            assert [text.get_text() for text in texts] == [
                plotting.category_legend_label(kind) for kind in categories]
            assert all(r"\mathtt" in text.get_text() for text in texts)
            for handle, kind in zip(handles, categories, strict=True):
                assert handle.get_alpha() == 1.
                assert to_rgba(handle.get_markerfacecolor()) == pytest.approx(to_rgba(palette[kind]))
            assert len({handle.get_markersize() for handle in handles}) == 1
    finally:
        for figure in figures:
            plt.close(figure)


def test_paired_views_share_data_rectangle_limits_ticks_and_summary_without_changing_rows(tmp_path, monkeypatch):
    frame = _observations()
    selected = frame.loc[frame["include_prompt"]].copy()
    before, selected_before = frame.copy(deep=True), selected.copy(deep=True)
    calls = []
    def capture(output, figures, **options):
        calls.append((list(figures), options))
    monkeypatch.setattr(plotting, "_publish_figures", capture)
    plotting._write_scatter_views(tmp_path, all_prompts=frame, selected=selected,
                                 spearman_column="experiment_prompt_spearman")
    assert len(calls) == 1
    figures, options = calls[0]
    by_view = {names["pdf"]: fig for fig, names in figures}
    all_figure = by_view[plotting.PROXIMITY_PDF_FIGURES["all_prompts"]["pdf"]]
    selected_figure = by_view[plotting.PROXIMITY_PDF_FIGURES["selected"]["pdf"]]
    full, retained = all_figure.axes[0], selected_figure.axes[0]
    assert full.get_xlim() == retained.get_xlim()
    assert full.get_ylim() == retained.get_ylim()
    np.testing.assert_array_equal(full.get_xticks(), retained.get_xticks())
    np.testing.assert_array_equal(full.get_yticks(), retained.get_yticks())
    assert full.get_ylim()[0] < -.31
    for fig, axis in ((all_figure, full), (selected_figure, retained)):
        width, height = fig.get_size_inches()
        box = axis.get_position()
        assert (box.width * width, box.height * height) == pytest.approx((3.1, 3.1))
        assert tuple(fig.get_size_inches()) == pytest.approx((4.45, 4.2))
        assert getattr(fig, "_proximity_canvas_extent_artist", None) is None
        summary = next(text for text in axis.texts if text.get_text().startswith("#Prompts:"))
        assert summary.get_text().count("\n") == 2
        _assert_summary_box(summary, axis)
        assert axis.get_xlabel() == r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|$"
        assert axis.get_ylabel() == "SSCD"
        assert all(spine.get_visible() for spine in axis.spines.values())
    summary = next(text for text in [*selected_figure.texts, *retained.texts]
                  if text.get_text().startswith("#Prompts:"))
    assert "#Prompts: 2" in summary.get_text()
    assert r"Median $\rho$: -0.400" in summary.get_text()
    assert r"$\rho < 0$: 1/1 (100.0%)" in summary.get_text()
    assert options["export_options"]
    for _figure, names in figures:
        assert options["export_options"][names["pdf"]]["dpi"] == 150
        assert set(names) == {"pdf"}
        assert options["formats"] == ("pdf",)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(selected, selected_before)
    assert not any(plt.fignum_exists(fig.number) for fig, _names in figures)


def test_zero_evaluable_prompts_stay_undefined_and_summary_box_is_preserved():
    frame = _observations().iloc[:2]
    figure = plotting._scatter_figure(frame, plotting.AnalysisStatistics(1, 0, 0, None, None))
    try:
        summary = next(text for text in [*figure.texts, *figure.axes[0].texts]
                      if text.get_text().startswith("#Prompts:"))
        assert "#Prompts: 1" in summary.get_text()
        assert r"Median $\rho$: undefined" in summary.get_text()
        assert r"$\rho < 0$: 0/0 (undefined)" in summary.get_text()
        _assert_summary_box(summary, figure.axes[0])
    finally:
        plt.close(figure)


def test_scatter_theme_is_scoped_and_keeps_global_gmm_tokens():
    from utils.experiments import proximity_style
    rc_before = dict(matplotlib.rcParams)
    plotting_tokens = (dict(plotting.PLOT_STYLE), dict(plotting._KIND_COLORS),
                       plotting.SCATTER_SIZE, plotting.SCATTER_ALPHA, plotting.FIGURE_DPI)
    figure = _draw(_observations())
    plt.close(figure)
    assert dict(matplotlib.rcParams) == rc_before
    assert (dict(plotting.PLOT_STYLE), dict(plotting._KIND_COLORS),
            plotting.SCATTER_SIZE, plotting.SCATTER_ALPHA, plotting.FIGURE_DPI) == plotting_tokens
    assert proximity_style.STYLE_VERSION


def test_identity_order_excludes_proximity_outcomes_view_and_dataframe_index():
    from utils.experiments import proximity_style
    frame = _observations()
    order = proximity_style.display_order(frame)
    changed = frame.copy(deep=True)
    changed["l2_norm"] = -changed["l2_norm"]
    changed["sscd"] = changed["sscd"] * 100
    changed["kind"] = "MV"
    changed["include_prompt"] = ~changed["include_prompt"]
    changed["experiment_prompt_spearman"] = None
    changed["view"] = "selected"
    changed.index = list(reversed(range(100, 100 + len(changed))))
    np.testing.assert_array_equal(proximity_style.display_order(changed), order)
    shuffled = frame.iloc[[4, 1, 8, 3, 6, 0, 9, 5, 2, 7]]
    identities = ["model_name", "scheduler_name", "seed_role", "original_index", "record_id", "seed"]
    assert list(frame.iloc[order][identities].itertuples(index=False, name=None)) == list(
        shuffled.iloc[proximity_style.display_order(shuffled)][identities].itertuples(index=False, name=None))


def test_dense_low_sscd_cloud_retains_every_point_without_special_overlays():
    count = 256
    frame = pd.DataFrame({"original_index": [f"dense-{index}" for index in range(count)],
                          "seed": np.arange(count) % 20, "kind": "N",
                          "l2_norm": np.linspace(10., 80., count),
                          "sscd": np.linspace(-.2, .05, count)})
    before = frame.copy(deep=True)
    figure = _draw(frame)
    try:
        points = figure.axes[0].collections
        assert len(points) == 1
        assert len(points[0].get_offsets()) == count
        assert points[0].get_alpha() == .35
        assert points[0].get_sizes().tolist() == [10.]
        assert Counter(_drawn_xy(figure)) == Counter(zip(frame["l2_norm"], frame["sscd"]))
        pd.testing.assert_frame_equal(frame, before)
    finally:
        plt.close(figure)


def test_only_points_are_rasterized_and_inside_annotations_use_the_compact_canvas():
    figure = _draw(_observations())
    try:
        axis = figure.axes[0]
        assert axis.collections[0].get_rasterized()
        summary = next(text for text in axis.texts if text.get_text().startswith("#Prompts:"))
        legends = [item for item in axis.get_children() if isinstance(item, Legend)]
        for artist in (figure, axis, axis.xaxis.label, axis.yaxis.label, summary, *legends,
                       *axis.spines.values()):
            assert not artist.get_rasterized()
        # This check belongs to the author's future test run, never to the
        # source-only editing task. It also checks the compact annotation layout.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for artist in (axis.xaxis.label, axis.yaxis.label, summary, *legends):
            bounds = artist.get_window_extent(renderer)
            assert bounds.x0 >= figure.bbox.x0
            assert bounds.y0 >= figure.bbox.y0
            assert bounds.x1 <= figure.bbox.x1
            assert bounds.y1 <= figure.bbox.y1
    finally:
        plt.close(figure)


def test_scatter_exception_restores_rc_and_closes_partial_figure(monkeypatch):
    rc_before = dict(matplotlib.rcParams)
    figures_before = set(plt.get_fignums())
    def fail_summary(*args, **kwargs):
        raise RuntimeError("injected summary failure")
    monkeypatch.setattr(plotting.proximity_style, "add_summary_box", fail_summary)
    with pytest.raises(RuntimeError, match="summary failure"):
        _draw(_observations())
    assert dict(matplotlib.rcParams) == rc_before
    assert set(plt.get_fignums()) == figures_before


def test_pair_export_exception_restores_rc_and_closes_both_figures(tmp_path, monkeypatch):
    frame = _observations()
    before = frame.copy(deep=True)
    rc_before = dict(matplotlib.rcParams)
    figures_before = set(plt.get_fignums())
    captured = []
    def fail_export(output, figures, **options):
        captured.extend(figure for figure, _names in figures)
        raise RuntimeError("injected pair export failure")
    monkeypatch.setattr(plotting, "_publish_figures", fail_export)
    with pytest.raises(RuntimeError, match="pair export failure"):
        plotting._write_scatter_views(tmp_path, all_prompts=frame,
                                     selected=frame.loc[frame["include_prompt"]],
                                     spearman_column="experiment_prompt_spearman")
    assert len(captured) == 2
    assert dict(matplotlib.rcParams) == rc_before
    assert set(plt.get_fignums()) == figures_before
    pd.testing.assert_frame_equal(frame, before)


def _gmm_style_inputs():
    from tests.test_proximity import _gmm_figure_inputs

    frame, configuration = _gmm_figure_inputs()
    frame["model_name"] = "sdv1"
    frame["scheduler_name"] = "ddim"
    frame["seed_role"] = "reference"
    frame["seed"] = np.arange(len(frame))
    frame.loc[0, "sscd"] = -.31
    frame.loc[4, "l2_norm"] = 800.  # Finite outlier stays visible, despite its incomplete prompt.
    return frame, configuration


def test_gmm_publication_theme_preserves_fitted_rows_geometry_and_component_mean_colors():
    import copy
    from matplotlib.collections import PathCollection
    from matplotlib.patches import Ellipse
    from utils.experiments import proximity_style, publication_style

    frame, configuration = _gmm_style_inputs()
    before, config_before = frame.copy(deep=True), copy.deepcopy(configuration)
    rc_before = dict(matplotlib.rcParams)
    original_tokens = dict(plotting._GMM_COMPONENT_COLORS)
    figure = plotting._gmm_fit_figure(frame, configuration)
    try:
        axis = figure.axes[0]
        assert len(figure.axes) == 1  # Component colors do not imply an SSCD colorbar.
        assert figure._proximity_style_version == "proximity-gmm-publication-1"
        assert proximity_style.GMM_STYLE_VERSION == "proximity-gmm-publication-1"
        assert tuple(figure.get_size_inches()) == pytest.approx((4.45, 4.2))
        width, height = figure.get_size_inches()
        box = axis.get_position()
        assert (box.width * width, box.height * height) == pytest.approx((3.1, 3.1))
        assert axis.get_xlabel() == r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|$"
        assert axis.get_ylabel() == "SSCD"
        assert axis.get_xscale() == axis.get_yscale() == "linear"
        assert axis.xaxis.label.get_fontsize() == axis.yaxis.label.get_fontsize() == 18
        assert all(label.get_fontsize() == 15 for label in (*axis.get_xticklabels(), *axis.get_yticklabels()))
        assert all(spine.get_visible() for spine in axis.spines.values())
        assert axis.get_ylim()[0] < -.31 and axis.get_xlim()[1] > 800.

        dots = [item for item in axis.collections if isinstance(item, PathCollection)]
        assert len(dots) == 1
        points = dots[0]
        assert points.get_sizes().tolist() == [10.]
        assert points.get_alpha() == .35 and points.get_rasterized() is True
        assert len(points.get_edgecolors()) == 0
        fitted = frame.loc[frame.observation_status.eq("complete")]
        observed = [tuple(pair) for pair in points.get_offsets().tolist()]
        assert Counter(observed) == Counter(zip(fitted.l2_norm, fitted.sscd))
        palette = {"low_sscd_mode": publication_style.SSCD_CMAP(0.),
                   "high_sscd_mode": publication_style.SSCD_CMAP(1.)}
        assert proximity_style.GMM_COMPONENT_COLORS == palette
        by_xy = {(row.l2_norm, row.sscd): row.gmm_component for row in fitted.itertuples()}
        for xy, rgba in zip(observed, points.get_facecolors(), strict=True):
            assert rgba == pytest.approx(to_rgba(palette[by_xy[xy]], alpha=.35))

        fit = configuration["gmm_fit"]
        scale = np.asarray(fit["feature_scale"])
        means = np.asarray(fit["feature_mean"]) + np.asarray(fit["means_standardized"]) * scale
        covariances = np.asarray(fit["covariances_standardized"]) * scale[None, :, None] * scale[None, None, :]
        ellipses = [patch for patch in axis.patches if isinstance(patch, Ellipse)]
        assert len(ellipses) == 4 and len(axis.lines) == 2
        for index, component in enumerate(("low_sscd_mode", "high_sscd_mode")):
            mean_artist = axis.lines[index]
            np.testing.assert_allclose(mean_artist.get_xdata(), [means[index, 0]])
            np.testing.assert_allclose(mean_artist.get_ydata(), [means[index, 1]])
            assert to_rgba(mean_artist.get_color()) == pytest.approx(to_rgba(palette[component]))
            assert mean_artist.get_marker() == "x"
            for sigma, ellipse in zip((1., 2.), ellipses[2 * index:2 * index + 2], strict=True):
                np.testing.assert_allclose(ellipse.center, means[index])
                angle = np.deg2rad(ellipse.angle)
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                eigenvalues = np.array([ellipse.width, ellipse.height]) ** 2 / (4 * sigma ** 2)
                np.testing.assert_allclose(rotation @ np.diag(eigenvalues) @ rotation.T,
                                           covariances[index], rtol=1e-12, atol=1e-12)
                assert ellipse.get_edgecolor() == pytest.approx(to_rgba(palette[component], alpha=.9))
                assert not ellipse.get_fill() and not ellipse.get_rasterized()

        legend = axis.get_legend()
        assert legend is not None and legend._loc == 1 and legend._ncols == 1
        assert legend.get_title().get_text() == "" and not legend.get_frame_on()
        assert [label.get_text() for label in legend.get_texts()] == [
            "Low SSCD", "High SSCD"]
        assert all(label.get_fontsize() == 12 for label in legend.get_texts())
        assert all(handle.get_alpha() == 1. for handle in legend.legend_handles)
        for artist in (figure, axis, axis.xaxis.label, axis.yaxis.label, legend,
                       *axis.lines, *axis.spines.values()):
            assert not artist.get_rasterized()
        assert not axis.texts  # No new statistics/claims are inferred from the fit.
        pd.testing.assert_frame_equal(frame, before)
        assert configuration == config_before
        assert dict(matplotlib.rcParams) == rc_before
        assert plotting._GMM_COMPONENT_COLORS == original_tokens
    finally:
        plt.close(figure)


def test_gmm_point_order_depends_only_on_saved_identity_and_does_not_use_global_rng():
    frame, configuration = _gmm_style_inputs()
    changed = frame.copy(deep=True)
    complete = changed.observation_status.eq("complete")
    changed.loc[complete, "gmm_component"] = list(reversed(changed.loc[complete, "gmm_component"].tolist()))
    changed.loc[complete, "sscd"] = np.arange(int(complete.sum())) - 3.5
    changed["include_prompt"] = False
    changed["view"] = "different-view"
    changed.index = list(reversed(range(100, 100 + len(changed))))
    rng = np.random.get_state()
    figures = []
    try:
        figures.append(plotting._gmm_fit_figure(frame, configuration))
        figures.append(plotting._gmm_fit_figure(frame.iloc[::-1], configuration))
        figures.append(plotting._gmm_fit_figure(changed, configuration))
        assert _drawn_xy(figures[0]) == _drawn_xy(figures[1])
        assert [x for x, _y in _drawn_xy(figures[0])] == [x for x, _y in _drawn_xy(figures[2])]
        after = np.random.get_state()
        assert after[0] == rng[0]
        np.testing.assert_array_equal(after[1], rng[1])
        assert after[2:] == rng[2:]
    finally:
        for figure in figures:
            plt.close(figure)


def test_gmm_duplicate_coordinates_keep_each_fitted_observation():
    frame, configuration = _gmm_style_inputs()
    repeated = frame.iloc[[0]].copy()
    repeated["original_index"] = "distinct-prompt-same-coordinates"
    repeated["seed"] = 27
    frame = pd.concat([frame, repeated], ignore_index=True)
    configuration["gmm_fit"]["usable_observation_count"] += 1
    before = frame.copy(deep=True)
    figure = plotting._gmm_fit_figure(frame, configuration)
    try:
        fitted = frame.loc[frame.observation_status.eq("complete")]
        assert Counter(_drawn_xy(figure)) == Counter(zip(fitted.l2_norm, fitted.sscd))
        assert Counter(_drawn_xy(figure))[(1., -.31)] == 2
        pd.testing.assert_frame_equal(frame, before)
    finally:
        plt.close(figure)


def test_gmm_export_uses_shared_pdf_150_dpi_and_preserves_other_publisher_defaults(tmp_path, monkeypatch):
    from pathlib import Path

    frame, configuration = _gmm_style_inputs()
    rc_before = dict(matplotlib.rcParams)
    before_figures = set(plt.get_fignums())
    calls = []

    def capture(output, figures, **options):
        calls.append((Path(output), tuple(figures), options))

    monkeypatch.setattr(plotting, "_publish_figures", capture)
    plotting.write_gmm_fit_figure(tmp_path / "cached", frame=frame, configuration=configuration,
                                  figure_directory=tmp_path / "published")
    assert len(calls) == 1
    output, items, options = calls[0]
    assert output == tmp_path / "published"
    assert len(items) == 1 and items[0][1] == {"pdf": "proximity_vs_sscd_gmm_fit.pdf"}
    assert options["formats"] == ("pdf",)
    assert options["export_options"]["proximity_vs_sscd_gmm_fit.pdf"]["dpi"] == 150
    assert plotting.FIGURE_DPI == 150 and plotting.FIGURE_PAD_INCHES == .05
    assert dict(matplotlib.rcParams) == rc_before
    assert set(plt.get_fignums()) == before_figures
    assert not tmp_path.joinpath("cached").exists() and not tmp_path.joinpath("published").exists()


@pytest.mark.parametrize("failure", ["layout", "export"])
def test_gmm_exception_restores_rc_and_closes_partial_figure(tmp_path, monkeypatch, failure):
    frame, configuration = _gmm_style_inputs()
    before = frame.copy(deep=True)
    rc_before = dict(matplotlib.rcParams)
    before_figures = set(plt.get_fignums())

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected GMM " + failure + " failure")

    if failure == "layout":
        monkeypatch.setattr(plotting.proximity_style, "apply_layout", fail)
    else:
        monkeypatch.setattr(plotting, "_publish_figures", fail)
    with pytest.raises(RuntimeError, match="injected GMM " + failure):
        plotting.write_gmm_fit_figure(tmp_path, frame=frame, configuration=configuration)
    assert dict(matplotlib.rcParams) == rc_before
    assert set(plt.get_fignums()) == before_figures
    pd.testing.assert_frame_equal(frame, before)
    assert not list(tmp_path.iterdir())
