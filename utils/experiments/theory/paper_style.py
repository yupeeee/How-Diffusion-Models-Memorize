"""Presentation-only theme for the six selected theory publications.

The current 18/15/12 point typography is the starting export size; manuscript
inclusion width is not specified, so final-size visual inspection is pending.
Use THEORY_STYLE inside rc_context. Importing this module changes no rcParams.
Neither this theme nor its version belongs to a scientific cache identity.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.transforms import Bbox
import numpy as np

from utils.experiments.plotting import FIGURE_DPI, FIGURE_PAD_INCHES, PLOT_STYLE, add_sscd_colorbar
from utils.experiments.publication_style import (
    COLORS, PLOT_BOX_INCHES, SSCD_CMAP, SSCD_NORM, SSCD_TICKS,
)
from .paper_registry import GROUPS

STYLE_VERSION = "six-theory-visual-6"
TYPOGRAPHY_SCALE = 1.0
AXIS_LABEL_SIZE = 18 * TYPOGRAPHY_SCALE
TICK_LABEL_SIZE = 15 * TYPOGRAPHY_SCALE
LEGEND_SIZE = LEGEND_FONT_SIZE = 12 * TYPOGRAPHY_SCALE
ANNOTATION_SIZE = ANNOTATION_FONT_SIZE = 10 * TYPOGRAPHY_SCALE
SPARSE_SIZE = SPARSE_MARKER_SIZE = 24
SPARSE_ALPHA = .8
DENSE_MARKER_SIZE = 8
DENSE_ALPHA = .1
DENSE_PDF_DPI = FIGURE_DPI
DISPLAY_ORDER_SEED = 0
CURVE_WIDTH = 1.6
CURVE_ALPHA = 1.0
BAND_ALPHA = .14
CONTROL_WIDTH = .8
CONNECTOR_WIDTH = .65
CONNECTOR_ALPHA = .45
BACKGROUND_COLOR = COLORS["background"]
TEXT_COLOR = COLORS["text"]
AXIS_COLOR = COLORS["axis"]
REFERENCE_COLOR = COLORS["reference"]
CONTROL_COLOR = COLORS["control"]
GRID_COLOR = COLORS["grid"]
ANALYTICAL_COLOR = COLORS["analytical"]
LEARNED_COLOR = COLORS["learned"]
ERROR_COLOR = COLORS["reference_error"]
CONNECTOR_COLOR = COLORS["connector"]
GROUP_COLORS = {GROUPS[0]: SSCD_CMAP(1.0), GROUPS[1]: SSCD_CMAP(0.0)}
LINE_STYLES = {
    "observed": "solid", "observable": (0, (4.5, 2.0)),
    "reference": (0, (6.0, 1.8, 1.2, 1.8)), "boundary": (0, (1.2, 2.0)),
}

# Preserve the approximately 3.1-inch subplot in the existing 4-inch canvas;
# enlarge only shared outer margins to reserve colorbar and external legends.
BASE_FIGURE_SIZE_INCHES = (4.0, 4.0)
LEFT_INCHES, BOTTOM_INCHES = 1.0, .85
RIGHT_INCHES, TOP_INCHES = 1.05, 1.15
CANVAS_SIZE_INCHES = (LEFT_INCHES + PLOT_BOX_INCHES + RIGHT_INCHES,
                      BOTTOM_INCHES + PLOT_BOX_INCHES + TOP_INCHES)
COLORBAR_WIDTH_INCHES, COLORBAR_PAD_INCHES = .15, .14
LABEL_PAD = 7
LEGEND_Y = 1.02
LEGEND_HANDLE_LENGTH = 3.4
SELECTED_STEMS = frozenset({
    "initial_loss_recovery", "unconditional_reference_convergence",
    "corollary3_guidance_scale_vs_loss", "posterior_feedback_condition_margin",
    "synchronization_bound", "terminal_bound_coverage",
})
LAYOUT_RECEIPT = {
    "canvas_inches": CANVAS_SIZE_INCHES, "plot_box_inches": PLOT_BOX_INCHES,
    "colorbar_width_inches": COLORBAR_WIDTH_INCHES, "colorbar_pad_inches": COLORBAR_PAD_INCHES,
    "shared_export_padding_inches": FIGURE_PAD_INCHES,
    "export_dpi": FIGURE_DPI,
    "colorbar_rendering": "opaque raster color strip; vector labels, ticks and outline",
    "export_bounds": "per_figure_tight_with_0.05_inch_padding",
    "data_aspect": "preserved", "typography_scale": TYPOGRAPHY_SCALE,
    "manuscript_size_check": "pending_author_visual_inspection",
}


def theory_rc(typography_scale=TYPOGRAPHY_SCALE):
    """Return local rc settings; one scale controls all exported text sizes."""
    scale = float(typography_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Theory typography scale must be finite and positive")
    return {**PLOT_STYLE, "figure.figsize": CANVAS_SIZE_INCHES,
            "font.size": 15 * scale, "axes.labelsize": 18 * scale,
            "axes.titlesize": 15 * scale, "figure.titlesize": 15 * scale,
            "xtick.labelsize": 15 * scale, "ytick.labelsize": 15 * scale,
            "legend.fontsize": 12 * scale,
            "figure.facecolor": BACKGROUND_COLOR, "axes.facecolor": BACKGROUND_COLOR,
            "savefig.facecolor": BACKGROUND_COLOR, "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR, "axes.edgecolor": AXIS_COLOR,
            "xtick.color": AXIS_COLOR, "ytick.color": AXIS_COLOR,
            "axes.linewidth": .7, "grid.color": GRID_COLOR, "grid.linewidth": .4,
            "grid.alpha": 1., "axes.grid": True, "axes.grid.which": "major",
            "axes.axisbelow": True, "axes.spines.top": True, "axes.spines.right": True,
            "legend.frameon": False, "legend.handlelength": LEGEND_HANDLE_LENGTH,
            "figure.autolayout": False, "figure.constrained_layout.use": False}


THEORY_STYLE = theory_rc()


def is_selected(entry):
    """Scope the refresh to selected publication routes, never legacy callers."""
    if not isinstance(entry, Mapping) or entry.get("stem") not in SELECTED_STEMS:
        return False
    outputs = entry.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        return False
    return all(isinstance(name, str) and not PurePosixPath(name).is_absolute()
               and len(PurePosixPath(name).parts) == 2
               and PurePosixPath(name).parts[0] == "figures"
               and PurePosixPath(name).suffix in {".png", ".pdf"}
               for name in outputs.values())


def apply_layout(figure, axis):
    """Reserve equal physical panels without changing data aspect or ranges."""
    figure.set_size_inches(CANVAS_SIZE_INCHES, forward=True)
    width, height = CANVAS_SIZE_INCHES
    axis.set_position((LEFT_INCHES / width, BOTTOM_INCHES / height,
                       PLOT_BOX_INCHES / width, PLOT_BOX_INCHES / height))
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axis.set_facecolor(BACKGROUND_COLOR)
    axis.set_axisbelow(True)
    axis.xaxis.labelpad = axis.yaxis.labelpad = LABEL_PAD
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    axis.tick_params(axis="both", which="both", colors=AXIS_COLOR, width=.7,
                     labelsize=TICK_LABEL_SIZE)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(.7)
    axis.grid(False, which="minor")
    axis.grid(True, which="major", color=GRID_COLOR, linewidth=.4, alpha=1.)
    figure._theory_visual_refresh = True
    return dict(LAYOUT_RECEIPT)


def add_theory_sscd_colorbar(figure, axis, values):
    """Use the reserved slot: opaque fixed colors, ticks and physical width."""
    width, height = figure.get_size_inches()
    box = axis.get_position()
    cax = figure.add_axes((box.x1 + COLORBAR_PAD_INCHES / width, box.y0,
                           COLORBAR_WIDTH_INCHES / width, box.height))
    def locator(colorbar_axis, renderer):
        active = axis.get_position()
        current_width = figure.get_size_inches()[0]
        return Bbox.from_bounds(active.x1 + COLORBAR_PAD_INCHES / current_width,
                                active.y0, COLORBAR_WIDTH_INCHES / current_width, active.height)
    cax.set_axes_locator(locator)
    colorbar = add_sscd_colorbar(figure, axis, values, cmap=SSCD_CMAP, norm=SSCD_NORM,
                                 ticks=SSCD_TICKS, cax=cax, labelsize=AXIS_LABEL_SIZE,
                                 ticksize=TICK_LABEL_SIZE, rasterized=True)
    # Rasterize only the opaque color strip: adjacent vector cells can show
    # hairline seams in PDF viewers. No cell edges or antialiasing are needed
    # for a contiguous gradient; text, ticks and outline stay vector artwork.
    if colorbar.solids is not None:
        colorbar.solids.set_alpha(1.)
        colorbar.solids.set_edgecolor("face")
        colorbar.solids.set_linewidth(0.)
        colorbar.solids.set_antialiased(False)
    colorbar.ax.grid(False, which="both")
    colorbar.outline.set_edgecolor(AXIS_COLOR)
    colorbar.outline.set_linewidth(.7)
    colorbar.ax.yaxis.label.set_color(TEXT_COLOR)
    colorbar.ax.tick_params(colors=AXIS_COLOR, width=.7)
    return colorbar


def common_export_bounds(figures):
    """Draw at render time and preserve one complete export extent for a suite.

    Tight bounds include labels and both external legend artists. Including
    every full canvas retains the unoccupied colorbar/legend slots. Matplotlib
    applies pad_inches only for the string 'tight', so explicitly pad the shared
    inch Bbox before it is supplied to savefig. This function never reads data.
    """
    bounds = []
    for figure in figures:
        # The paper renderer constructs Figure directly, whose base canvas has
        # no renderer. Attach Agg only when needed; no figure-manager side effects.
        if not callable(getattr(figure.canvas, "get_renderer", None)):
            FigureCanvasAgg(figure)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds.append(Bbox.from_bounds(0., 0., *figure.get_size_inches()))
        tight = figure.get_tightbbox(renderer)
        if tight is not None:
            if not np.isfinite(tight.extents).all():
                raise ValueError("Theory presentation has nonfinite export bounds")
            bounds.append(tight)
    if not bounds:
        raise ValueError("Common theory export bounds require at least one figure")
    return Bbox.union(bounds).padded(FIGURE_PAD_INCHES)
