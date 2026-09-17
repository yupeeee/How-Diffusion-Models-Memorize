"""Fixed evidence designs drawn only from saved scalar summaries.

Statistical intervals, reference evaluations, population weights, numerical
classifications and ECDF change points must already exist in the compact input.
The only calculations here position artists and validate their displayed ranges.
"""
from __future__ import annotations

import math

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from utils.experiments.plotting import (
    SUMMARY_FONT_SIZE, SCATTER_ALPHA, SCATTER_SIZE, SSCD_COLOR_RANGE, add_sscd_colorbar,
)
from .contracts import TheoryError
from . import paper_style
from .paper_registry import GROUPS, GROUP_COLORS, GROUP_LABELS

EVIDENCE_RENDERING_VERSION = "precise-seven-evidence-stix-2"
EVIDENCE_KINDS = frozenset({
    "pair_loss", "reference_convergence", "injection_geometry",
    "feedback_fractions", "synchronization_curves", "terminal_cdf",
    "terminal_components", "reference_native_sweep", "numerical_resolution",
})
_REFERENCE = {
    "reference": (r"$\|\bar{\mathbf{x}}^K_t-\boldsymbol{\mu}_K\|/\sqrt{d}$", "#2878b5", "-"),
    "learned": (r"$\|\hat{\mathbf{x}}_t(\varnothing)-\boldsymbol{\mu}_K\|/\sqrt{d}$", "#e67d24", "--"),
    "reference_error": (r"$e^K_t(\varnothing)/\sqrt{d}$", "#3b9955", ":"),
}
_SYNCHRONIZATION = {
    "joint_error": ("Joint target error", "-"),
    "gap": ("Branch gap", "--"),
    "bound": ("Complete bound", "-."),
}
_TERMINAL = {
    "actual": ("Observed reproduction", ".2", "-"),
    "observable": (r"$B_{\mathrm{obs}}+\delta_{\mathrm{sched}}$", "#e5ba26", "--"),
    "reference": (r"$B_{\mathrm{ref}}+\delta_{\mathrm{sched}}$", "#563580", "-."),
}
_COMPONENTS = {
    "conditional": (r"$e_1(c)/\sqrt{d}$", "#2878b5", "-"),
    "guidance": (r"$(g-1)\|\boldsymbol{\Delta}_1\|/\sqrt{d}$", "#e67d24", "--"),
    "scheduler": (r"$\delta_{\mathrm{sched}}/\sqrt{d}$", "#3b9955", "-."),
}
_SCOPES = {
    "original_clean_terminal_theorem": "Theorem 7",
    "finite_terminal_update_extension_deterministic": "Finite-terminal-update extension",
    "finite_terminal_update_extension_gaussian_noise_bound": "Finite-terminal-update extension\n(Gaussian noise bound)",
}


def _number(frame, name):
    if name not in frame:
        raise TheoryError(f"Saved evidence table lacks {name}")
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _legend(ax, handles, *, loc="best", size=8):
    if handles:
        return ax.legend(handles=handles, loc=loc, frameon=False, fontsize=size,
                         handlelength=1.8, handletextpad=.45, borderaxespad=.5,
                         labelspacing=.3, ncol=1)


def _split_legends(ax, quantities, groups, extras=()):
    # Both legends stay inside; colors and quantities never share a column.
    first = _legend(ax, groups, loc="upper left", size=8)
    if first is not None:
        ax.add_artist(first)
    _legend(ax, [*quantities, *extras], loc="upper right", size=8)


def _groups(frame):
    present = set(frame.group.dropna().astype(str))
    if present - set(GROUPS):
        raise TheoryError(f"Unknown saved outcome groups: {sorted(present - set(GROUPS))}")
    return [group for group in GROUPS if group in present]


def _nonnegative_limits(ax, arrays, *, axis="y"):
    finite = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    finite = finite[np.isfinite(finite)]
    if not len(finite) or np.any(finite < 0):
        raise TheoryError("Nonnegative evidence axis has missing or negative values")
    upper = float(finite.max())
    getattr(ax, "set_" + axis + "lim")(0, upper * 1.06 if upper else 1)
    getattr(ax, axis + "axis").set_major_locator(MaxNLocator(nbins=5))


def _snr(ax, metadata):
    value = metadata.get("actual_initial_snr")
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise TheoryError("Evidence curves require saved positive actual_initial_snr")
    value = float(value)
    ax.set_xscale("log")
    ax.axvline(value, color=".45", linestyle=":", linewidth=.8)
    return value


def _native_segments(rows):
    """Break artists at explicit missing values or omitted native indices."""
    rows = rows.sort_values("snr", kind="stable")
    if "segment_id" in rows:
        return [part for _, part in rows.groupby("segment_id", sort=False, dropna=False)]
    if "step_index" not in rows or len(rows) < 2:
        return [rows]
    steps = _number(rows, "step_index")
    breaks = np.flatnonzero(np.isfinite(steps[:-1]) & np.isfinite(steps[1:]) & (np.abs(np.diff(steps)) != 1)) + 1
    return [rows.iloc[a:b] for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(rows)])]


def _curve(ax, rows, *, color, style, band=False, native=True):
    segments = _native_segments(rows) if native else [rows.sort_values("snr", kind="stable")]
    count = 0
    for segment in segments:
        x, y = _number(segment, "snr"), _number(segment, "median")
        if np.any(np.isfinite(x) & (x <= 0)):
            raise TheoryError("Saved SNR curve contains nonpositive coordinates")
        valid = np.isfinite(x) & np.isfinite(y)
        count += int(valid.sum())
        ax.plot(x, y, color=color, linestyle=style, linewidth=1.4,
                marker="." if len(segment) == 1 else None, markersize=3)
        if band:
            lower, upper = _number(segment, "q25"), _number(segment, "q75")
            if np.any(valid & (~np.isfinite(lower) | ~np.isfinite(upper))):
                raise TheoryError("Saved descriptive interval is missing for a curve value")
            if np.any(valid & ((lower > y) | (upper < y))):
                raise TheoryError("Saved descriptive interval does not contain its median")
            ax.fill_between(x, lower, upper, color=color, alpha=.12, linewidth=0)
    return count


def _pair_loss(fig, ax, frame, metadata, *, styled=False):
    x, y, control = (_number(frame, key) for key in ("x", "y", "control_y"))
    scores = _number(frame, "mean_terminal_sscd")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if not valid.any():
        raise TheoryError("Loss recovery has no common finite pair summaries")
    size = paper_style.SPARSE_SIZE if styled else SCATTER_SIZE
    alpha = paper_style.SPARSE_ALPHA if styled else SCATTER_ALPHA
    outline = paper_style.CONTROL_COLOR if styled else ".55"
    interval_color = paper_style.REFERENCE_COLOR if styled else ".55"
    ax.vlines(x[valid], np.minimum(y[valid], control[valid]), np.maximum(y[valid], control[valid]),
              color=paper_style.CONNECTOR_COLOR if styled else ".65",
              linewidth=paper_style.CONNECTOR_WIDTH if styled else .55,
              alpha=paper_style.CONNECTOR_ALPHA if styled else .3, zorder=1)
    control_options = {"zorder": 3} if styled else {}
    conditional_options = {"zorder": 4} if styled else {}
    ax.scatter(x[valid], control[valid], s=size if styled else size * .65,
               facecolors="none", edgecolors=outline,
               linewidths=paper_style.CONTROL_WIDTH if styled else .6,
               alpha=alpha, rasterized=not styled, **control_options)
    colored = valid & np.isfinite(scores)
    ax.scatter(x[colored], y[colored], c=scores[colored],
               cmap=paper_style.SSCD_CMAP if styled else "viridis",
               norm=paper_style.SSCD_NORM if styled else Normalize(*SSCD_COLOR_RANGE, clip=True), s=size,
               edgecolors="none", alpha=alpha, rasterized=not styled, **conditional_options)
    if (valid & ~colored).any():
        ax.scatter(x[valid & ~colored], y[valid & ~colored],
                   color=paper_style.CONTROL_COLOR if styled else ".5", s=size,
                   edgecolors="none", alpha=alpha, rasterized=not styled, **conditional_options)
    if styled:
        bar = paper_style.add_theory_sscd_colorbar(fig, ax, scores[colored])
    else:
        bar = add_sscd_colorbar(fig, ax, scores[colored])
        bar.set_label("Mean terminal SSCD")
    intervals = all(key in frame for key in ("x_low", "x_high", "y_low", "y_high"))
    rendered_intervals = False
    if intervals:
        xl, xh, yl, yh = (_number(frame, key) for key in ("x_low", "x_high", "y_low", "y_high"))
        interval_rows = valid & np.isfinite(xl) & np.isfinite(xh) & np.isfinite(yl) & np.isfinite(yh)
        if np.any(interval_rows & ((xl > xh) | (yl > yh) | (xl < 0) | (yl < 0))):
            raise TheoryError("Invalid saved Monte Carlo bootstrap interval")
        if valid.sum() <= 40:
            # Percentile intervals need not contain their original point estimate.
            interval_options = {"zorder": 2} if styled else {}
            ax.hlines(y[interval_rows], xl[interval_rows], xh[interval_rows], color=interval_color, alpha=.35, linewidth=.6, **interval_options)
            ax.vlines(x[interval_rows], yl[interval_rows], yh[interval_rows], color=interval_color, alpha=.35, linewidth=.6, **interval_options)
            rendered_intervals = bool(interval_rows.any())
    _nonnegative_limits(ax, [x[valid], xl if intervals else [] , xh if intervals else []], axis="x")
    _nonnegative_limits(ax, [y[valid], control[valid], yl if intervals else [], yh if intervals else []])
    handles = [Line2D([], [], marker="o", linestyle="none", color=paper_style.TEXT_COLOR if styled else ".3", label="Conditional"),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", color=outline, label="Unconditional control")]
    if rendered_intervals:
        handles.append(Line2D([], [], color=interval_color, linewidth=.7, label="Monte Carlo bootstrap intervals"))
    _legend(ax, handles, loc="upper left")
    counts = metadata.get("counts", {})
    needed = {"pairs", "gaussian_seeds_per_pair", "forward_draws"}
    if not needed <= set(counts) or metadata.get("actual_initial_snr") is None:
        raise TheoryError("Loss recovery requires saved pair/seed/draw counts and initial SNR")
    initial = float(metadata["actual_initial_snr"])
    ax.text(.97, .03, rf"$\mathrm{{SNR}}_T={initial:.3g}$" + "\n" +
            f"{counts['pairs']} pairs; {counts['gaussian_seeds_per_pair']} Gaussian seeds/pair\n{counts['forward_draws']} forward draws/pair",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    return {"finite_pair_summaries": int(valid.sum()), "excluded_nonfinite_pairs": int((~valid).sum()),
            "colorbar_label": "SSCD" if styled else "Mean terminal SSCD", "equality_guide": False,
            "color_population": "Pair mean terminal SSCD",
            "selected_sparse_style": bool(styled), "scatter_size": float(size), "scatter_alpha": float(alpha),
            "control_outline_color": outline, "conditional_missing_sscd_count": int((valid & ~colored).sum()),
            "paired_connector_count": int(valid.sum()),
            "bootstrap_intervals_rendered": rendered_intervals,
            "bootstrap_interval_definition": "Monte Carlo bootstrap intervals",
            "interval_display_limit_pairs": 40}


def _reference(ax, frame, metadata, *, native_sweep=False):
    boundary = _snr(ax, metadata)
    if set(frame.metric.astype(str)) != set(_REFERENCE):
        raise TheoryError("Lemma 2 requires every saved reference/network quantity")
    allowed = {"native"} if native_sweep else {"analytical", "native_initial"}
    if set(frame.source_range.astype(str)) - allowed:
        raise TheoryError("Lemma-2 table has an unsupported measurement range for this figure")
    handles, total = [], 0
    if native_sweep:
        if np.any(_number(frame, "snr") < boundary * (1 - 1e-12)):
            raise TheoryError("A native network curve crosses below its saved SNR boundary")
        for metric, (label, color, style) in _REFERENCE.items():
            rows = frame.loc[frame.metric.eq(metric)]
            count = _curve(ax, rows, color=color, style=style, band=True)
            if not count:
                raise TheoryError(f"Required native curve has no finite observations: {metric}")
            total += count
            handles.append(Line2D([], [], color=color, linestyle=style, label=label))
        analytical_range = None
    else:
        analytical = frame.loc[frame.source_range.eq("analytical")]
        if analytical.empty or analytical.metric.ne("reference").any():
            raise TheoryError("Only the analytical reference may extend below initialization")
        snr = _number(analytical, "snr")
        if not np.isfinite(snr).all() or np.any(snr <= 0) or np.any(snr > boundary * (1 + 1e-12)):
            raise TheoryError("Invalid initial analytical reference range")
        lower = float(snr.min())
        if lower >= boundary or not np.isclose(snr.max(), boundary, rtol=1e-12, atol=0):
            raise TheoryError("Analytical reference grid must end at the actual initial SNR")
        total += _curve(ax, analytical, color=_REFERENCE["reference"][1], style="-", band=True, native=False)
        initial = frame.loc[frame.source_range.eq("native_initial")]
        for (metric, (label, color, _)), marker in zip(_REFERENCE.items(), ("o", "s", "D")):
            rows = initial.loc[initial.metric.eq(metric)]
            if len(rows) != 1 or not np.allclose(_number(rows, "snr"), boundary, rtol=1e-12, atol=0):
                raise TheoryError(f"Initial Lemma-2 marker must be one saved summary at SNR_T: {metric}")
            if "step_index" in rows and np.any(_number(rows, "step_index") != 0):
                raise TheoryError("Initial Lemma-2 marker must use the actual first prediction")
            y, low, high = (_number(rows, key)[0] for key in ("median", "q25", "q75"))
            if not all(math.isfinite(value) for value in (y, low, high)) or not 0 <= low <= y <= high:
                raise TheoryError("Invalid saved initial descriptive interval")
            ax.errorbar([boundary], [y], yerr=[[y-low], [high-y]], fmt=marker,
                        markersize=4.5, color=color, elinewidth=.8, capsize=2, linestyle="none", zorder=4)
            handles.append(Line2D([], [], color=color, marker=marker,
                                  linestyle="-" if metric == "reference" else "none", label=label))
            total += 1
        ax.axvspan(lower, boundary, color=".85", alpha=.22, zorder=-1)
        ax.text(.02, .02, "Analytical reference only", transform=ax.transAxes, fontsize=10, va="bottom")
        ax.set_xlim(lower / 1.1, boundary * 1.6)
        analytical_range = [lower, boundary]
    handles.append(Line2D([], [], color=".45", linestyle=":", label=r"$\mathrm{SNR}_T$"))
    _legend(ax, handles, loc="upper left")
    _nonnegative_limits(ax, [_number(frame, name) for name in ("median", "q25", "q75")])
    return {"finite_summary_cells": total, "initial_snr_reference": boundary,
            "analytical_only_range": analytical_range, "network_extension": False,
            "native_initial_markers_only": not native_sweep,
            "complete_saved_native_range": native_sweep,
            "scale_reference_rendered": False,
            "band_definition": "Descriptive IQR over unique Gaussian seeds"}


def _geometry(fig, ax, frame, metadata):
    x, y, scores = (_number(frame, key) for key in ("x", "y", "terminal_sscd"))
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any() or np.any(y[valid] < 0):
        raise TheoryError("Injection geometry requires finite coordinates and nonnegative perpendicular norms")
    colored = valid & np.isfinite(scores)
    ax.scatter(x[colored], y[colored], c=scores[colored], cmap="viridis", norm=Normalize(*SSCD_COLOR_RANGE, clip=True),
               s=SCATTER_SIZE, edgecolors="none", alpha=SCATTER_ALPHA, rasterized=True)
    if (valid & ~colored).any():
        ax.scatter(x[valid & ~colored], y[valid & ~colored], color=".5", s=SCATTER_SIZE, edgecolors="none", alpha=SCATTER_ALPHA, rasterized=True)
    add_sscd_colorbar(fig, ax, scores[colored])
    xmin, xmax = min(0., float(x[valid].min())), max(1., float(x[valid].max()))
    padding = .045 * max(xmax - xmin, 1.)
    ylim = max(float(y[valid].max()) * 1.06, .1)
    lower_padding = .035 * max(ylim, xmax - xmin, 1.)
    theta = np.linspace(0, math.pi, 257)
    for level in (.25, .5, 1.):
        ax.plot(1 + level * np.cos(theta), level * np.sin(theta), color=".65", linewidth=.6, linestyle=":")
        # All three contours intersect the left half-axis [0,1], even when
        # the observations cluster tightly around the limiting vector.
        ax.annotate(f"{level:g}", (1 - level, 0), xytext=(0, 3), textcoords="offset points", fontsize=7, color=".45", ha="center")
    ax.plot([1], [0], marker="*", color=".15", markersize=8, linestyle="none", clip_on=False)
    _legend(ax, [Line2D([], [], marker="*", linestyle="none", color=".15", label="Corollary 3 limiting vector")], loc="upper right")
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(-lower_padding, ylim)
    ax.set_aspect("equal", adjustable="box")
    return {"finite_observations": int(valid.sum()), "excluded_nonfinite_rows": int((~valid).sum()),
            "ideal_location": [1., 0.], "relative_error_contours": [.25, .5, 1.],
            "equal_geometry": True, "fitted_line": False}


def _feedback(ax, frame, metadata):
    groups = _groups(frame)
    if set(frame.metric.astype(str)) != {"feedback", "condition"}:
        raise TheoryError("Feedback requires both observed strict gain and original strict condition")
    condition = frame.loc[frame.metric.eq("condition")]
    condition_values = _number(condition, "fraction")
    shared_zero = (len(groups) == 2 and len(condition_values) > 0
                   and np.isfinite(condition_values).all() and np.all(condition_values == 0))
    unknown_counts = _number(condition, "unresolved_count")
    if not np.isfinite(unknown_counts).all() or np.any(unknown_counts < 0) or np.any(unknown_counts != np.floor(unknown_counts)):
        raise TheoryError("Condition disclosure requires saved unresolved observation counts")
    total = 0
    for group in groups:
        paired = frame.loc[frame.group.eq(group)]
        if paired.duplicated(["metric", "step_index"]).any():
            raise TheoryError("Duplicated feedback curve cells")
        left = paired.loc[paired.metric.eq("feedback")].sort_values("step_index")
        right = paired.loc[paired.metric.eq("condition")].sort_values("step_index")
        if len(left) != len(right) or not np.array_equal(_number(left, "step_index"), _number(right, "step_index")):
            raise TheoryError("Feedback and condition use different native populations")
        for column in ("snr", "denominator_weight"):
            if not np.array_equal(_number(left, column), _number(right, column), equal_nan=True):
                raise TheoryError("Feedback and condition use different saved denominators or timesteps")
        for metric in ("feedback", "condition"):
            rows = paired.loc[paired.metric.eq(metric)]
            if rows.empty:
                raise TheoryError("Feedback curve missing on a saved outcome population")
            for segment in _native_segments(rows):
                x, y, upper, denominator = (_number(segment, key) for key in ("snr", "fraction", "upper_fraction", "denominator_weight"))
                finite = np.isfinite(x) & np.isfinite(y)
                if np.any(finite & (~np.isfinite(denominator) | (denominator <= 0))):
                    raise TheoryError("A finite feedback fraction requires positive saved population weight")
                if np.any(finite & ((x <= 0) | (y < 0) | (y > 1) | ~np.isfinite(upper) | (upper < y) | (upper > 1))):
                    raise TheoryError("Invalid saved numerical-ambiguity envelope")
                if metric == "feedback":
                    ax.plot(x, y, color=GROUP_COLORS[group], linestyle="-", linewidth=1.5,
                            marker="." if len(segment) == 1 else None, markersize=3)
                    ax.fill_between(x, y, upper, color=GROUP_COLORS[group], alpha=.13, linewidth=0)
                else:
                    if not shared_zero:
                        ax.plot(x, y, color=GROUP_COLORS[group], linestyle="--", linewidth=1,
                                clip_on=False, marker="." if len(segment) == 1 else None, markersize=3)
                    # The original condition's partial-identification region is
                    # shown by boundaries, never a second opaque/hatching band.
                    if np.any(finite & (upper > y)):
                        ax.plot(x, upper, color=GROUP_COLORS[group], linestyle=":", linewidth=.8,
                                marker="." if len(segment) == 1 else None, markersize=3)
                total += int(finite.sum())
    if not total:
        raise TheoryError("Available feedback figure has no finite saved fractions")
    if shared_zero:
        ax.axhline(0, color=".3", linestyle="--", linewidth=1, clip_on=False)
        ax.text(.98, .04, "Both original-condition curves: 0\n"
                + f"Unresolved condition sample-transitions: {int(unknown_counts.sum())}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    boundary = _snr(ax, metadata)
    ax.set_ylim(0, 1)
    _split_legends(ax,
        [Line2D([], [], color=".25", linestyle="-", label="Strict feedback"),
         Line2D([], [], color=".25", linestyle="--", label="Original strict condition")],
        [Line2D([], [], color=GROUP_COLORS[group], label=GROUP_LABELS[group]) for group in groups],
        [Patch(facecolor=".85", alpha=.4, label="Unknown feedback mass"),
         Line2D([], [], color=".45", linestyle=":", label="Condition upper boundary"),
         Line2D([], [], color=".45", linestyle=":", label=r"$\mathrm{SNR}_T$")])
    return {"finite_summary_cells": total, "initial_snr_reference": boundary,
            "band_definition": "Numerical ambiguity, not statistical uncertainty",
            "condition_zero_overlap": shared_zero, "shared_zero_baseline": shared_zero,
            "condition_unresolved_observation_count": int(unknown_counts.sum()),
            "condition_display": "thin lower/upper boundaries", "hatching": False,
            "complete_saved_native_range": True}


def _numerical_resolution(ax, frame, metadata):
    definitions = {"feedback": ("Fixed-cache feedback", "-"),
                   "condition": ("Original condition", "--"),
                   "source_robust_feedback": ("Source-robust feedback", "-.")}
    groups = _groups(frame)
    if set(frame.metric.astype(str)) != set(definitions):
        raise TheoryError("Resolution appendix requires fixed-cache, condition and source-robust signs")
    total = 0
    for group in groups:
        population = None
        for metric, (_, style) in definitions.items():
            rows = frame.loc[frame.group.eq(group) & frame.metric.eq(metric)].sort_values("step_index")
            if rows.empty or rows.step_index.duplicated().any():
                raise TheoryError("Resolution appendix has missing or repeated native cells")
            current = np.column_stack([_number(rows, key) for key in ("step_index", "snr", "denominator_weight")])
            if population is not None and not np.array_equal(population, current, equal_nan=True):
                raise TheoryError("Resolution layers use different saved populations")
            population = current
            for segment in _native_segments(rows):
                x, y, unknown, denominator = (_number(segment, key) for key in ("snr", "resolved_fraction", "unknown_fraction", "denominator_weight"))
                finite = np.isfinite(x) & np.isfinite(y)
                if np.any(finite & ((x <= 0) | (y < 0) | (y > 1) | ~np.isfinite(unknown)
                                   | (unknown < 0) | (unknown > 1) | (np.abs(y + unknown - 1) > 1e-12)
                                   | ~np.isfinite(denominator) | (denominator <= 0))):
                    raise TheoryError("Invalid saved numerical-resolution fractions")
                ax.plot(x, y, color=GROUP_COLORS[group], linestyle=style, linewidth=1.3,
                        marker="." if len(segment) == 1 else None, markersize=3)
                total += int(finite.sum())
    if not total:
        raise TheoryError("Resolution appendix has no finite saved rates")
    boundary = _snr(ax, metadata)
    ax.set_ylim(0, 1)
    _split_legends(ax,
        [Line2D([], [], color=".25", linestyle=style, label=label) for label, style in definitions.values()],
        [Line2D([], [], color=GROUP_COLORS[group], label=GROUP_LABELS[group]) for group in groups],
        [Line2D([], [], color=".45", linestyle=":", label=r"$\mathrm{SNR}_T$")])
    return {"finite_summary_cells": total, "initial_snr_reference": boundary,
            "complete_saved_native_range": True, "unknown_mass_definition": "1 minus resolved fraction",
            "numerical_resolution_table": metadata.get("numerical_resolution_table")}


def _synchronization(ax, frame, metadata):
    groups = _groups(frame)
    if set(frame.metric.astype(str)) != set(_SYNCHRONIZATION):
        raise TheoryError("Synchronization requires joint target error, branch gap and full bound")
    total = 0
    for group in groups:
        for metric, (_, style) in _SYNCHRONIZATION.items():
            rows = frame.loc[frame.group.eq(group) & frame.metric.eq(metric)]
            if rows.empty:
                raise TheoryError("Missing synchronization quantity on a saved outcome population")
            total += _curve(ax, rows, color=GROUP_COLORS[group], style=style, band=metric == "joint_error")
    boundary = _snr(ax, metadata)
    quantities = [Line2D([], [], color=".25", linestyle=style, label=label) for label, style in _SYNCHRONIZATION.values()]
    _split_legends(ax, quantities,
        [Line2D([], [], color=GROUP_COLORS[group], label=GROUP_LABELS[group]) for group in groups],
        [Line2D([], [], color=".45", linestyle=":", label=r"$\mathrm{SNR}_T$")])
    _nonnegative_limits(ax, [_number(frame, key) for key in ("median", "q25", "q75")])
    return {"finite_summary_cells": total, "initial_snr_reference": boundary,
            "shaded_quantities": ["joint_error"], "band_definition": "Pointwise descriptive IQR",
            "bound_aggregation": "Saved samplewise bound; no sum of aggregate medians"}


def _ecdf_support(rows):
    rows = rows.sort_values("value", kind="stable")
    values, cdf = _number(rows, "value"), _number(rows, "cdf")
    if not len(values) or np.isnan(values).any() or np.isneginf(values).any() or not np.isfinite(cdf).all():
        raise TheoryError("Exact saved ECDF support is missing or invalid")
    if np.any(values < 0) or np.any(values[1:] <= values[:-1]) or np.any(np.diff(cdf) < -1e-12) or np.any((cdf < 0) | (cdf > 1)):
        raise TheoryError("Invalid exact saved weighted ECDF support")
    if abs(float(cdf[-1]) - 1) > 1e-10:
        raise TheoryError("Saved weighted ECDF does not retain its full support")
    return values, cdf


def _terminal(ax, frame, metadata, *, components=False):
    key, definitions = ("component", _COMPONENTS) if components else ("distribution", _TERMINAL)
    if set(frame[key].astype(str)) != set(definitions):
        raise TheoryError("Terminal coverage requires all three curves on the common saved population")
    zero_mass = metadata.get("zero_mass")
    if not isinstance(zero_mass, dict) or not set(definitions) <= set(zero_mass):
        raise TheoryError("Terminal ECDF requires the exact saved zero mass for every curve")
    infinite_mass = metadata.get("infinite_mass", {name: 0. for name in definitions})
    if not isinstance(infinite_mass, dict) or not set(definitions) <= set(infinite_mass):
        raise TheoryError("Terminal ECDF requires the saved infinite mass for every curve")
    support = {}
    for name in definitions:
        values, cdf = _ecdf_support(frame.loc[frame[key].eq(name)])
        zero = float(zero_mass[name])
        actual_zero = float(cdf[values == 0][0]) if np.any(values == 0) else 0.
        if not math.isfinite(zero) or not 0 <= zero <= 1 or abs(zero - actual_zero) > 1e-12:
            raise TheoryError("Saved terminal zero mass differs from exact ECDF support")
        infinite = float(infinite_mass[name])
        finite = np.isfinite(values)
        actual_infinite = 1 - float(cdf[finite][-1]) if finite.any() else 1.
        if not math.isfinite(infinite) or not 0 <= infinite <= 1 or abs(infinite - actual_infinite) > 1e-10:
            raise TheoryError("Saved terminal infinite mass differs from exact ECDF support")
        support[name] = (values, cdf)
    positives = np.concatenate([values[(values > 0) & np.isfinite(values)] for values, _ in support.values()])
    if len(positives):
        lower, upper = float(positives.min()), float(positives.max())
        limits = (lower / 1.15, upper * 1.15)
    else:
        # No fabricated epsilon or positive observation: this is an explicit
        # empty-domain display window, not an invented ECDF change point.
        limits = (1., 10.)
        message = "All mass is at exactly zero" if all(float(zero_mass[name]) == 1. for name in definitions) else "No finite positive change points"
        ax.text(.5, .45, message, transform=ax.transAxes, ha="center", fontsize=9)
    tolerance = metadata.get("predeclared_tolerance_rmse")
    if tolerance is not None:
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise TheoryError("A predeclared tolerance must be nonnegative and finite")
        if tolerance > 0:
            # Extend the exact zero-mass/final-mass plateaus along with the
            # displayed axis, including when the declared tolerance is outside
            # the observed change points. This introduces no new observations.
            limits = (min(limits[0], tolerance / 1.15), max(limits[1], tolerance * 1.15))
    for name, (label, color, style) in definitions.items():
        values, cdf = support[name]
        positive = (values > 0) & np.isfinite(values)
        x = np.r_[limits[0], values[positive], limits[1]]
        y = np.r_[float(zero_mass[name]), cdf[positive], 1. - float(infinite_mass[name])]
        ax.step(x, y, where="post", color=color, linestyle=style, linewidth=1.4, label=label)
    ax.set_xscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(0, 1)
    scope = metadata.get("terminal_scope")
    if scope not in _SCOPES:
        raise TheoryError(f"Unsupported terminal display scope: {scope}")
    if scope != "original_clean_terminal_theorem" and metadata.get("manuscript_extension_required") is not True:
        raise TheoryError("A finite terminal extension must declare its manuscript requirement")
    if not components:
        ax.set_title(_SCOPES[scope], fontsize=SUMMARY_FONT_SIZE)
    handles = [Line2D([], [], color=color, linestyle=style, label=label) for label, color, style in definitions.values()]
    if tolerance is not None:
        if tolerance == 0:
            ax.text(.02, .73, "Predeclared tolerance: 0", transform=ax.transAxes, fontsize=8)
        else:
            ax.axvline(tolerance, color=".5", linestyle=":", linewidth=.8)
            handles.append(Line2D([], [], color=".5", linestyle=":", label="Predeclared tolerance"))
    _legend(ax, handles, loc="lower right")
    short = {"actual": "Actual", "observable": "Obs. bound", "reference": "Ref. bound",
             "conditional": "Conditional", "guidance": "Guidance", "scheduler": "Scheduler"}
    zero_mass_visible = any(float(zero_mass[name]) > 0 for name in definitions)
    if zero_mass_visible:
        text = "Mass at zero\n" + "; ".join(f"{short[name]} {float(zero_mass[name]):.3g}" for name in definitions)
        ax.text(.02, .98, text, transform=ax.transAxes, fontsize=7, va="top", wrap=True)
    if any(float(infinite_mass[name]) > 0 for name in definitions):
        infinity_text = "Mass at +infinity\n" + "; ".join(f"{short[name]} {float(infinite_mass[name]):.3g}" for name in definitions)
        ax.text(.02, .78, infinity_text, transform=ax.transAxes, fontsize=7, va="top", wrap=True)
    clean = metadata.get("original_clean_counts")
    if not isinstance(clean, dict) or not {"applicable", "total"} <= set(clean):
        raise TheoryError("Terminal coverage requires original clean-update applicability counts")
    ax.text(.02, .86, f"{clean['applicable']}/{clean['total']} original clean updates", transform=ax.transAxes, fontsize=8, va="top")
    return {"terminal_scope": scope, "zero_mass": zero_mass, "infinite_mass": infinite_mass,
            "full_positive_support": [float(positives.min()), float(positives.max())] if len(positives) else None,
            "empty_positive_domain": not len(positives), "zero_replacement": None,
            "original_clean_counts": clean, "empirical_cdf_confidence_band": False,
            "zero_mass_annotation_visible": zero_mass_visible}


def draw_evidence(fig, ax, entry, frame, metadata, config):
    """Draw one designated evidence design without changing its saved inputs."""
    kind = entry["kind"]
    if kind == "pair_loss":
        return _pair_loss(fig, ax, frame, metadata, styled=paper_style.is_selected(entry))
    if kind in {"reference_convergence", "reference_native_sweep"}:
        return _reference(ax, frame, metadata, native_sweep=kind == "reference_native_sweep")
    if kind == "injection_geometry":
        return _geometry(fig, ax, frame, metadata)
    if kind == "feedback_fractions":
        return _feedback(ax, frame, metadata)
    if kind == "numerical_resolution":
        return _numerical_resolution(ax, frame, metadata)
    if kind == "synchronization_curves":
        return _synchronization(ax, frame, metadata)
    if kind in {"terminal_cdf", "terminal_components"}:
        return _terminal(ax, frame, metadata, components=kind == "terminal_components")
    raise TheoryError(f"Unknown fixed evidence design: {kind}")
