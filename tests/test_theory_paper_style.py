"""Scoped visual theme and exporter regressions; the author runs these tests."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pytest

from utils.experiments import plotting as shared
from utils.experiments.theory import paper_style as style
from utils.experiments.theory.paper_registry import GROUPS, measurement_registry, paper_registry


def test_fixed_palette_is_shared_magma_slice_not_per_figure_normalization():
    expected = matplotlib.colormaps["magma"](np.linspace(.2, .7, 256))
    np.testing.assert_array_equal(style.SSCD_CMAP.colors, expected)
    assert (style.SSCD_NORM.vmin, style.SSCD_NORM.vmax) == (0., 1.)
    assert style.SSCD_NORM.clip
    np.testing.assert_array_equal(style.GROUP_COLORS[GROUPS[0]], expected[-1])
    np.testing.assert_array_equal(style.GROUP_COLORS[GROUPS[1]], expected[0])
    assert style.SSCD_TICKS == (0., .2, .4, .6, .8, 1.)


def test_theme_is_scoped_to_selected_publication_routes_and_restores_defaults():
    baseline = dict(matplotlib.rcParams)
    legacy = dict(shared.PLOT_STYLE)
    assert len(paper_registry()) == 6
    assert all(style.is_selected(entry) for entry in paper_registry())
    assert not any(style.is_selected(entry) for entry in measurement_registry(True))
    assert not style.is_selected({"stem": "proximity", "outputs": {"png": "figures/proximity.png"}})
    assert not style.is_selected({"stem": "initial_loss_recovery", "outputs": {"png": "main/loss.png"}})
    assert not style.is_selected({"stem": "initial_loss_recovery", "outputs": {"png": "/figures/loss.png"}})
    assert not style.is_selected({"stem": "initial_loss_recovery", "outputs": {}})
    with matplotlib.rc_context(style.THEORY_STYLE):
        assert matplotlib.rcParams["font.family"] == ["STIXGeneral"]
        assert matplotlib.rcParams["mathtext.fontset"] == "stix"
        assert matplotlib.rcParams["axes.labelsize"] == style.AXIS_LABEL_SIZE
        assert matplotlib.rcParams["axes.grid.which"] == "major"
    assert dict(matplotlib.rcParams) == baseline
    assert shared.PLOT_STYLE == legacy
    assert shared.FIGURE_SIZE == (4., 4.) and shared.FIGURE_DPI == 150
    scaled = style.theory_rc(1.25)
    for key, size in (("axes.labelsize", 18), ("xtick.labelsize", 15), ("legend.fontsize", 12)):
        assert scaled[key] == size * 1.25
    with pytest.raises(ValueError, match="positive"):
        style.theory_rc(0)


def test_layout_preserves_scientific_scales_limits_ticks_and_data_aspect():
    fig = plt.Figure()
    ax = fig.subplots()
    ax.plot([-2., 0., 8.], [-4., 0., 11.])
    ax.set_xscale("symlog", base=2, linthresh=.03, linscale=.4)
    ax.set_yscale("symlog", base=10, linthresh=.001, linscale=.75)
    ax.set_xlim(-3., 9.)
    ax.set_ylim(-6., 12.)
    ax.set_xticks([-2., 0., 8.])
    ax.set_yticks([-4., 0., 11.])
    ax.minorticks_on()
    before = (ax.get_xlim(), ax.get_ylim(), ax.get_aspect(), ax.xaxis._scale, ax.yaxis._scale)
    original = [(line.get_xdata().copy(), line.get_ydata().copy()) for line in ax.lines]
    receipt = style.apply_layout(fig, ax)
    assert (ax.get_xlim(), ax.get_ylim(), ax.get_aspect(), ax.xaxis._scale, ax.yaxis._scale) == before
    np.testing.assert_array_equal(ax.get_xticks(), [-2., 0., 8.])
    np.testing.assert_array_equal(ax.get_yticks(), [-4., 0., 11.])
    for line, (x, y) in zip(ax.lines, original):
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
    box = ax.get_position()
    np.testing.assert_allclose([box.width * fig.get_figwidth(), box.height * fig.get_figheight()],
                               [style.PLOT_BOX_INCHES] * 2)
    assert all(spine.get_visible() and spine.get_linewidth() == .7 for spine in ax.spines.values())
    assert all(tick.gridline.get_visible() for tick in ax.xaxis.get_major_ticks())
    assert not any(tick.gridline.get_visible() for tick in ax.xaxis.get_minor_ticks())
    assert receipt["data_aspect"] == "preserved"
    assert receipt["manuscript_size_check"] == "pending_author_visual_inspection"
    plt.close(fig)


def test_fixed_opaque_raster_colorbars_do_not_shrink_the_panel():
    figures = []
    try:
        for values in ([.2, .3], [.8, .9], [0., 1.]):
            fig = plt.Figure()
            figures.append(fig)
            ax = fig.subplots()
            style.apply_layout(fig, ax)
            before = ax.get_position().bounds
            points = ax.scatter([0., 1.], [1., 0.], c=values, cmap=style.SSCD_CMAP,
                                norm=style.SSCD_NORM, alpha=.1)
            colorbar = style.add_theory_sscd_colorbar(fig, ax, values)
            style.common_export_bounds([fig])
            np.testing.assert_allclose(ax.get_position().bounds, before)
            assert colorbar.mappable is not points
            assert colorbar.cmap is style.SSCD_CMAP and colorbar.norm is style.SSCD_NORM
            assert colorbar.solids.get_alpha() == 1.
            assert colorbar.solids.get_rasterized() is True
            assert np.all(colorbar.solids.get_linewidths() == 0.)
            assert not np.any(colorbar.solids.get_antialiased())
            np.testing.assert_allclose(colorbar.get_ticks(), style.SSCD_TICKS)
            assert colorbar.ax.get_ylabel() == "SSCD"
            plot_box, color_box = ax.get_position(), colorbar.ax.get_position()
            assert color_box.y0 == pytest.approx(plot_box.y0)
            assert color_box.height == pytest.approx(plot_box.height)
            assert color_box.width * fig.get_figwidth() == pytest.approx(style.COLORBAR_WIDTH_INCHES)
            assert (color_box.x0 - plot_box.x1) * fig.get_figwidth() == pytest.approx(style.COLORBAR_PAD_INCHES)
    finally:
        for fig in figures:
            plt.close(fig)


def test_legacy_colorbar_defaults_remain_viridis_and_existing_font_sizes():
    fig, ax = plt.subplots()
    try:
        colorbar = shared.add_sscd_colorbar(fig, ax, np.array([.1, .2]))
        assert colorbar.cmap.name == "viridis"
        assert (colorbar.norm.vmin, colorbar.norm.vmax) == (0., 1.)
        assert colorbar.solids.get_alpha() == shared.COLORBAR_ALPHA
        assert colorbar.ax.yaxis.label.get_fontsize() == shared.AXIS_LABEL_FONT_SIZE
        assert all(text.get_fontsize() == shared.AXIS_NUMBER_FONT_SIZE for text in colorbar.ax.get_yticklabels())
    finally:
        plt.close(fig)


def test_common_bounds_attach_renderer_to_direct_figures_and_include_external_keys():
    figures = []
    try:
        for index in range(6):
            fig = plt.Figure()
            figures.append(fig)
            ax = fig.subplots()
            style.apply_layout(fig, ax)
            ax.plot([0., 1.], [0., 1.])
            ax.set_xlabel(r"$L_T/(d\,\mathrm{SNR}_T)$")
            ax.set_ylabel("Long mathematical quantity " + str(index))
            if index % 2:
                group = ax.legend([Line2D([], [], color="red")], ["SSCD > 0.75"],
                                  loc="lower center", bbox_to_anchor=(.5, 1.14))
                ax.add_artist(group)
                ax.legend([Line2D([], [], linestyle="--")], ["Observable bound"],
                          loc="lower center", bbox_to_anchor=(.5, 1.02))
            else:
                style.add_theory_sscd_colorbar(fig, ax, [0., 1.])
        common = style.common_export_bounds(figures)
        for fig in figures:
            assert callable(fig.canvas.get_renderer)
            tight = fig.get_tightbbox(fig.canvas.get_renderer())
            full = Bbox.from_bounds(0., 0., *fig.get_size_inches())
            for box in (tight, full):
                assert common.x0 <= box.x0 - shared.FIGURE_PAD_INCHES + 1e-12
                assert common.y0 <= box.y0 - shared.FIGURE_PAD_INCHES + 1e-12
                assert common.x1 >= box.x1 + shared.FIGURE_PAD_INCHES - 1e-12
                assert common.y1 >= box.y1 + shared.FIGURE_PAD_INCHES - 1e-12
    finally:
        for fig in figures:
            plt.close(fig)
    with pytest.raises(ValueError, match="at least one"):
        style.common_export_bounds([])


class SavedFigure:
    def __init__(self, *, fail_pdf=False):
        self.calls = []
        self.fail_pdf = fail_pdf

    def savefig(self, path, **options):
        self.calls.append(dict(options))
        Path(path).write_bytes(b"replacement")
        if self.fail_pdf and options["format"] == "pdf":
            raise RuntimeError("failed PDF export")


def test_export_options_only_change_requested_artifacts_and_keep_pairs(tmp_path):
    dense, sparse = SavedFigure(), SavedFigure()
    dense_names = {ext: f"figures/dense.{ext}" for ext in ("png", "pdf")}
    sparse_names = {ext: f"figures/sparse.{ext}" for ext in ("png", "pdf")}
    bounds = Bbox.from_bounds(-.1, -.1, 5.5, 5.6)
    options = {relative: {"bbox_inches": bounds} for relative in (*dense_names.values(), *sparse_names.values())}
    options[dense_names["pdf"]]["dpi"] = style.DENSE_PDF_DPI
    shared.publish_figures(tmp_path, [(dense, dense_names), (sparse, sparse_names)], export_options=options)
    assert [call["dpi"] for call in dense.calls] == [150, 150]
    assert [call["dpi"] for call in sparse.calls] == [150, 150]
    for call in (*dense.calls, *sparse.calls):
        assert call["bbox_inches"] is bounds
        assert call["pad_inches"] == shared.FIGURE_PAD_INCHES
    assert {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()} == set(options)


def test_default_export_options_retain_existing_shared_publisher_behavior(tmp_path):
    figure = SavedFigure()
    names = {ext: f"default.{ext}" for ext in ("png", "pdf")}
    shared.publish_figures(tmp_path, [(figure, names)])
    assert figure.calls == [dict(format=ext, dpi=150, bbox_inches="tight", pad_inches=.05)
                            for ext in ("png", "pdf")]


def test_pdf_override_failure_restores_previous_pair_and_cleans_staging(tmp_path):
    names = {ext: f"figures/dense.{ext}" for ext in ("png", "pdf")}
    for relative in names.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"previous")
    events = []
    with pytest.raises(RuntimeError, match="failed PDF"):
        shared.publish_figures(tmp_path, [(SavedFigure(fail_pdf=True), names)],
                               export_options={names["pdf"]: {"dpi": 600}},
                               progress=lambda name, status: events.append((name, status)))
    assert events == [(names["png"], "saving"), (names["png"], "saved"), (names["pdf"], "saving")]
    assert all((tmp_path / name).read_bytes() == b"previous" for name in names.values())
    assert {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()} == set(names.values())


@pytest.mark.parametrize("overrides", [
    {"unknown.pdf": {"dpi": 600}}, {"pair.pdf": {"dpi": 0}},
    {"pair.pdf": {"dpi": float("nan")}}, {"pair.pdf": {"dpi": True}},
    {"pair.pdf": {"rasterized": True}}, {"pair.pdf": {"bbox_inches": "invalid"}},
    {"pair.pdf": {"bbox_inches": Bbox.from_bounds(0., 0., 0., 1.)}},
])
def test_invalid_export_overrides_fail_before_output_creation(tmp_path, overrides):
    output = tmp_path / "not-created"
    figure = SavedFigure()
    with pytest.raises(shared.PlottingError):
        shared.publish_figures(output, [(figure, {"png": "pair.png", "pdf": "pair.pdf"})],
                               export_options=overrides)
    assert not output.exists() and figure.calls == []
