"""Author-run presentation invariants; these tests are not executed by the edit task."""
from collections import Counter
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd
import pytest

from tests.test_theory_paper_plotting import compact_fixture
from utils.experiments import plotting as shared
from utils.experiments.theory import paper_plotting as plotting, paper_style as style
from utils.experiments.theory.paper_contracts import load_paper_inputs
from utils.experiments.theory.paper_registry import paper_registry


def _inputs(tmp_path):
    root = compact_fixture(tmp_path / "paper", terminal=True)
    config, summary, _, frames = load_paper_inputs(root)
    return config, summary, frames, {entry["stem"]: entry for entry in paper_registry()}


def _points(ax):
    records = []
    for artist in ax.collections:
        if not isinstance(artist, PathCollection):
            continue
        xy = np.asarray(artist.get_offsets())
        color = artist.get_array()
        records.extend((float(x), float(y), None if color is None else float(color[index]))
                       for index, (x, y) in enumerate(xy))
    return records


def _rng_equal(before, after):
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_dense_scatter_preserves_rows_statuses_masks_and_symlog_without_global_rng(tmp_path):
    config, summary, frames, entries = _inputs(tmp_path)
    stem = "posterior_feedback_condition_margin"
    frame = frames[stem].copy(deep=True)
    frame["x"] = [-1000., -.001, 0., 1000.]
    frame["y"] = [-1e6, np.nan, 0., 1e5]
    frame["terminal_sscd"] = [-.2, .3, np.nan, 1.2]
    frame["marker_class"] = ["numerically_unresolved", "observed", "observed", "numerically_unresolved"]
    before, metadata = frame.copy(deep=True), deepcopy(summary["figures"][stem])
    unstyled = deepcopy(entries[stem])
    unstyled["outputs"] = {ext: f"appendix/{stem}.{ext}" for ext in ("png", "pdf")}
    figures = []
    random_before = np.random.get_state()
    try:
        old, old_stats = plotting._draw(unstyled, frame, metadata, config)
        figures.append(old)
        new, stats = plotting._draw(entries[stem], frame, metadata, config)
        figures.append(new)
        repeated, repeat_stats = plotting._draw(entries[stem], frame, metadata, config)
        figures.append(repeated)
        assert Counter(_points(new.axes[0])) == Counter(_points(old.axes[0]))
        assert _points(new.axes[0]) == _points(repeated.axes[0])
        assert stats["saved_marker_counts"] == old_stats["saved_marker_counts"]
        assert stats["finite_pairs"] == 3 and stats["missing_coordinate_pairs"] == 1
        assert stats["all_data_ranges"] == old_stats["all_data_ranges"]
        assert stats["axis_limits"] == old_stats["axis_limits"]
        assert stats["display_order_seed"] == style.DISPLAY_ORDER_SEED
        assert repeat_stats["display_order_seed"] == stats["display_order_seed"]
        for key in ("base", "linthresh", "linscale"):
            assert getattr(new.axes[0].yaxis.get_transform(), key) == getattr(old.axes[0].yaxis.get_transform(), key)
        assert new.axes[0].get_yscale() == "symlog"
        for artist in new.axes[0].collections:
            if isinstance(artist, PathCollection):
                assert artist.get_alpha() == style.DENSE_ALPHA
                assert np.all(artist.get_sizes() == style.DENSE_MARKER_SIZE)
                assert artist.get_rasterized()
                if artist.get_array() is not None:
                    assert artist.get_cmap().name == style.SSCD_CMAP.name
                    assert artist.norm.vmin == 0 and artist.norm.vmax == 1 and artist.norm.clip
        assert all(not line.get_rasterized() for line in new.axes[0].lines)
        assert all(not label.get_rasterized() for label in (new.axes[0].xaxis.label, new.axes[0].yaxis.label))
    finally:
        for figure in figures:
            plt.close(figure)
    pd.testing.assert_frame_equal(frame, before)
    assert metadata == summary["figures"][stem]
    _rng_equal(random_before, np.random.get_state())


def test_all_selected_scatter_colors_match_opaque_fixed_colorbars(tmp_path):
    config, summary, frames, entries = _inputs(tmp_path)
    for stem in ("initial_loss_recovery", "corollary3_guidance_scale_vs_loss", "posterior_feedback_condition_margin"):
        fig, _ = plotting._draw(entries[stem], frames[stem], summary["figures"][stem], config)
        try:
            colored = [artist for artist in fig.axes[0].collections
                       if isinstance(artist, PathCollection) and artist.get_array() is not None]
            assert colored
            for artist in colored:
                np.testing.assert_allclose(artist.cmap(np.linspace(0, 1, 11)), style.SSCD_CMAP(np.linspace(0, 1, 11)))
                assert (artist.norm.vmin, artist.norm.vmax) == (0., 1.)
            bar = next(axis for axis in fig.axes if axis.get_label() == "<colorbar>")
            np.testing.assert_allclose(bar.get_yticks(), np.linspace(0, 1, 6))
            mesh = next(artist for artist in bar.collections if artist.get_array() is not None)
            assert mesh.get_alpha() == 1.
            assert mesh.get_rasterized()
            assert mesh.cmap.name == style.SSCD_CMAP.name
        finally:
            plt.close(fig)


def test_shared_loss_axes_and_export_settings_do_not_change_artist_data(tmp_path):
    config, summary, frames, entries = _inputs(tmp_path)
    # Retain fitted outliers above and below the configured guidance.
    frames["corollary3_guidance_scale_vs_loss"] = frames["corollary3_guidance_scale_vs_loss"].copy()
    frames["corollary3_guidance_scale_vs_loss"]["y"] = [-15., .5, 7.5, 50.]
    pending, rendered = [], []
    try:
        for stem, entry in entries.items():
            figure, stats = plotting._draw(entry, frames[stem], summary["figures"][stem], config)
            pending.append((figure, entry["outputs"]))
            rendered.append({**entry, "display_audit": stats})
        before = [_points(figure.axes[0]) for figure, _ in pending]
        options = plotting._finalize_presentations(pending, rendered)
        assert [_points(figure.axes[0]) for figure, _ in pending] == before
        first, third = pending[0][0].axes[0], pending[2][0].axes[0]
        assert first.get_xlim() == third.get_xlim()
        np.testing.assert_array_equal(first.get_xticks(), third.get_xticks())
        assert third.get_ylim()[0] < -15 and third.get_ylim()[1] > 50
        boxes = []
        for figure, names in pending:
            bbox = figure.axes[0].get_window_extent().transformed(figure.dpi_scale_trans.inverted())
            boxes.append((bbox.width, bbox.height))
            assert options[names["png"]].get("dpi", shared.FIGURE_DPI) == shared.FIGURE_DPI
            assert options[names["pdf"]].get("dpi") == (style.DENSE_PDF_DPI if "condition_margin" in names["pdf"] else None)
        np.testing.assert_allclose(boxes, [boxes[0]] * 6)
        assert boxes[0][0] == pytest.approx(boxes[0][1])
        assert all(value["bbox_inches"] == "tight" for value in options.values())
        assert shared.FIGURE_PAD_INCHES == .05
        assert all(entry["display_audit"]["export"] == {"bbox_inches": "tight", "pad_inches": .05}
                   for entry in rendered)
    finally:
        for figure, _ in pending:
            plt.close(figure)


def test_theme_context_does_not_change_proximity_or_later_matplotlib_defaults(tmp_path):
    config, summary, frames, entries = _inputs(tmp_path)
    before = dict(mpl.rcParams)
    old_palette = shared.add_sscd_colorbar
    shared_style = dict(shared.PLOT_STYLE)
    stem = "corollary3_guidance_scale_vs_loss"
    figure, _ = plotting._draw(entries[stem], frames[stem], summary["figures"][stem], config)
    plt.close(figure)
    assert dict(mpl.rcParams) == before
    assert shared.PLOT_STYLE == shared_style and shared.add_sscd_colorbar is old_palette
    later = plt.Figure()
    try:
        assert later.get_facecolor() == mpl.colors.to_rgba(before["figure.facecolor"])
    finally:
        plt.close(later)
