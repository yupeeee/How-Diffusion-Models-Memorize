"""Scoped presentation for proximity scatters and the reference GMM diagnostic.

Scatter colors encode original categories; GMM colors encode saved components. This
helper reads only the caller's in-memory identity columns for display order;
it changes no data, scientific random state, plotting defaults, or output files.
"""
from __future__ import annotations

import hashlib
import json

import matplotlib
import numpy as np

from . import publication_style as common

STYLE_VERSION = "proximity-publication-2"
GMM_STYLE_VERSION = "proximity-gmm-publication-1"
SCATTER_SIZE = 10
SCATTER_ALPHA = .35
PDF_DPI = common.FIGURE_DPI
AXIS_LABEL_SIZE = AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_SIZE = TICK_LABEL_FONT_SIZE = 15
LEGEND_SIZE = LEGEND_FONT_SIZE = 12
SUMMARY_SIZE = SUMMARY_FONT_SIZE = 12
PLOT_BOX_INCHES = common.PLOT_BOX_INCHES
LEFT_INCHES, RIGHT_INCHES = 1., .35
BOTTOM_INCHES, TOP_INCHES = .85, .25
CANVAS_SIZE_INCHES = (LEFT_INCHES + PLOT_BOX_INCHES + RIGHT_INCHES,
                      BOTTOM_INCHES + PLOT_BOX_INCHES + TOP_INCHES)
LABEL_PAD = 7
CATEGORY_ORDER = ("MV", "RV", "TV", "N", "Other / unlabeled")
COLORS = {
    "MV": common.SSCD_CMAP(1.0),
    "RV": common.SSCD_CMAP(0.0),
    "TV": common.COLORS["learned"],
    "N": common.COLORS["analytical"],
    "Other / unlabeled": common.COLORS["reference"],
}
GMM_COMPONENT_COLORS = {
    "low_sscd_mode": common.SSCD_CMAP(0.0),
    "high_sscd_mode": common.SSCD_CMAP(1.0),
}
DISPLAY_ORDER_SALT = "proximity-publication-display-order-1"
DISPLAY_IDENTITY_FIELDS = (
    "model_name", "scheduler_name", "seed_role", "role", "original_index",
    "record_id", "source_row_number", "seed", "target_image_sha256",
)


def style_context(base_style):
    """Return a local rc context, preserving the caller's defaults on exit."""
    return matplotlib.rc_context({
        **base_style,
        "figure.figsize": CANVAS_SIZE_INCHES,
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": TICK_LABEL_SIZE, "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": TICK_LABEL_SIZE, "figure.titlesize": TICK_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE, "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.facecolor": common.COLORS["background"],
        "axes.facecolor": common.COLORS["background"],
        "savefig.facecolor": common.COLORS["background"],
        "text.color": common.COLORS["text"], "axes.labelcolor": common.COLORS["text"],
        "axes.edgecolor": common.COLORS["axis"],
        "xtick.color": common.COLORS["axis"], "ytick.color": common.COLORS["axis"],
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.linewidth": .7, "axes.spines.top": True, "axes.spines.right": True,
        "axes.axisbelow": True, "axes.grid": True, "axes.grid.which": "major",
        "grid.color": common.COLORS["grid"], "grid.linewidth": .4, "grid.alpha": 1.,
        "legend.frameon": False, "figure.autolayout": False,
        "figure.constrained_layout.use": False,
    })


def apply_layout(figure, axis):
    """Keep equal square data rectangles with compact label margins.

    Axis scales, limits, tick locations and data-unit aspect are untouched.
    Legends and statistics live inside the axes; export uses true tight bounds.
    """
    width, height = CANVAS_SIZE_INCHES
    figure.set_size_inches(CANVAS_SIZE_INCHES, forward=True)
    axis.set_position((LEFT_INCHES / width, BOTTOM_INCHES / height,
                       PLOT_BOX_INCHES / width, PLOT_BOX_INCHES / height))
    figure.patch.set_facecolor(common.COLORS["background"])
    axis.set_facecolor(common.COLORS["background"])
    axis.set_axisbelow(True)
    axis.xaxis.labelpad = axis.yaxis.labelpad = LABEL_PAD
    for label in (axis.xaxis.label, axis.yaxis.label):
        label.set_fontsize(AXIS_LABEL_SIZE)
        label.set_color(common.COLORS["text"])
    axis.tick_params(axis="both", which="both", direction="out", width=.7,
                     colors=common.COLORS["axis"], labelsize=TICK_LABEL_SIZE)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(common.COLORS["axis"])
        spine.set_linewidth(.7)
    axis.grid(False, which="minor")
    axis.grid(True, which="major", color=common.COLORS["grid"], linewidth=.4, alpha=1.)
    return {"style_version": STYLE_VERSION, "canvas_inches": CANVAS_SIZE_INCHES,
            "plot_box_inches": PLOT_BOX_INCHES, "data_aspect": "preserved",
            "legend_placement": "inside upper right; single column",
            "summary_placement": "inside lower left"}


def display_order(frame):
    """Return a stable positional permutation from observation identity alone.

    The salt and field list exclude view, category, coordinates and selection.
    A surviving observation therefore keeps its relative order in both views.
    Identical identities retain input order: no observations are deduplicated.
    Index identity is only a fallback when no supported identity column exists.
    No scientific or global random generator is consulted.
    """
    fields = tuple(name for name in DISPLAY_IDENTITY_FIELDS if name in frame.columns)
    if fields:
        rows = frame.loc[:, list(fields)].itertuples(index=False, name=None)
    else:
        fields = ("frame_index",)
        rows = ((index,) for index in frame.index)
    keys = []
    for row in rows:
        identity = [(name, str(value)) for name, value in zip(fields, row, strict=True)]
        payload = json.dumps([DISPLAY_ORDER_SALT, identity], ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        keys.append(hashlib.sha256(payload).hexdigest())
    return np.argsort(np.asarray(keys, dtype="U64"), kind="stable")


def add_legend(axis, handles, categories, *, category_colors=None):
    """Put represented original categories in one column inside the upper right."""
    palette = COLORS if category_colors is None else category_colors
    if len(handles) != len(categories):
        raise ValueError("Proximity legend categories and handles must match")
    if not handles:
        return ()
    if len(set(categories)) != len(categories) or any(name not in palette for name in categories):
        raise ValueError("Proximity legend requires unique normalized categories")
    for handle in handles:
        handle.set_alpha(1.)
    legend = axis.legend(
        handles=handles, loc="upper right", ncol=1, frameon=False,
        fontsize=LEGEND_SIZE,
        labelcolor=common.COLORS["text"], handlelength=1., handletextpad=.35,
        borderaxespad=.5, borderpad=0., labelspacing=.25,
    )
    return (legend,)


def add_summary_box(axis, text):
    """Restore the original rounded statistics box in axes-relative coordinates."""
    return axis.text(
        .02, .02, text, transform=axis.transAxes,
        ha="left", va="bottom", fontsize=SUMMARY_SIZE, color=common.COLORS["axis"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": "0.75", "alpha": .85},
    )
