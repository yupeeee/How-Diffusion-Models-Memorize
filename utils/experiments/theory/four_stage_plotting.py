"""Four fixed mechanism designs, using immutable compact scalar inputs only.

All cohorts, quantiles, weights, shape classifications and endpoint scopes come
from the analysis stage. This module only validates and positions artists.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

from utils.experiments.plotting import SCATTER_SIZE, SSCD_COLOR_RANGE, add_sscd_colorbar
from .contracts import TheoryError
from .evidence_plotting import (
    _REFERENCE, _curve, _groups, _legend, _nonnegative_limits,
    _number, _snr,
)
from .paper_registry import GROUP_COLORS, GROUP_LABELS

FOUR_STAGE_RENDERING_VERSION = "four-stage-stix-1"
FOUR_STAGE_KINDS = frozenset({
    "four_response", "four_chronological", "four_reference", "four_peak",
    "four_motion", "four_terminal_scatter", "four_counterfactual",
})
_QUANTITIES = {
    "gap": ("Branch gap", "--"), "joint_error": ("Joint target error", "-"),
    "bound": ("Complete bound", "-."), "conditional": ("Conditional", "-"),
    "unconditional": ("Unconditional", "--"),
}
_MOTION = {
    "conditional": ("Conditional motion", "#2878b5", "-"),
    "unconditional": ("Unconditional motion", "#e67d24", "--"),
    "quadratic": ("Quadratic term", "#3b9955", ":"),
    "change": ("Observed change", ".2", "-."),
}


def _index_axis(ax, metadata, *, motion=False):
    count = metadata.get("prediction_steps")
    if count is None or int(count) < 1:
        raise TheoryError("Chronological plot lacks its saved prediction count")
    last = int(count) - (2 if motion else 1)
    if last < 0:
        raise TheoryError("No consecutive prediction pair is available")
    ax.set_xlim(-.2 if last == 0 else 0, .2 if last == 0 else last)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ticks = sorted(set([0, last] + [int(value) for value in ax.get_xticks() if 0 <= value <= last]))
    ax.set_xticks(ticks, ["0\nInitialization" if value == 0 else str(value) for value in ticks])
    return last


def _common_curves(frame, metrics):
    """Compare saved cohort receipts, not the underlying statistical samples."""
    for group in _groups(frame):
        chosen = frame.loc[frame.group.eq(group)]
        if set(chosen.metric.astype(str)) != set(metrics):
            raise TheoryError("Compared curves must preserve every declared quantity")
        reference = None
        for metric in metrics:
            rows = chosen.loc[chosen.metric.eq(metric)].sort_values("step_index")
            keys = ["step_index", "denominator_weight"]
            if "snr" in rows:
                keys.append("snr")
            receipt = rows[keys].to_numpy(dtype=float)
            if rows.step_index.duplicated().any():
                raise TheoryError("Duplicate curve summary at one prediction index")
            if reference is not None and not np.array_equal(receipt, reference, equal_nan=True):
                raise TheoryError("Compared curves have different saved populations or schedules")
            reference = receipt


def _chronological(ax, entry, frame, metadata):
    metrics = ("gap", "joint_error") if entry["stem"] == "branch_gap_synchronization" else (
        ("conditional", "unconditional") if entry["stem"] == "branch_target_errors" else ("gap", "joint_error", "bound"))
    _common_curves(frame, metrics)
    last = _index_axis(ax, metadata)
    quantities, groups, total = [], [], 0
    for group in _groups(frame):
        color = GROUP_COLORS[group]
        groups.append(Line2D([], [], color=color, label=GROUP_LABELS[group]))
        for metric in metrics:
            rows = frame.loc[frame.group.eq(group) & frame.metric.eq(metric)].sort_values("step_index")
            x = _number(rows, "step_index")
            if np.any((x < 0) | (x > last) | (x != np.floor(x))):
                raise TheoryError("Curve includes an invalid or terminal-output prediction index")
            parts = [part for _, part in rows.groupby("segment_id", sort=False)] if "segment_id" in rows else [rows]
            for part in parts:
                x, y = _number(part, "step_index"), _number(part, "median")
                total += int(np.isfinite(y).sum())
                ax.plot(x, y, color=color, linestyle=_QUANTITIES[metric][1], linewidth=1.5,
                        marker="." if len(part) == 1 else None, markersize=3)
                if metric == "joint_error" or entry["stem"] == "branch_target_errors":
                    lower, upper = _number(part, "q25"), _number(part, "q75")
                    if np.any(np.isfinite(y) & ((lower > y) | (upper < y))):
                        raise TheoryError("Saved IQR does not bracket the median")
                    ax.fill_between(x, lower, upper, color=color, alpha=.10, linewidth=0)
    for metric in metrics:
        label, style = _QUANTITIES[metric]
        quantities.append(Line2D([], [], color=".3", linestyle=style, label=label))
    _nonnegative_limits(ax, [_number(frame, name) for name in ("median", "q25", "q75", "minimum", "maximum")])
    first = _legend(ax, groups, loc="upper left", size=10)
    if first is not None:
        ax.add_artist(first)
    _legend(ax, quantities, loc="upper right", size=10)
    return {"finite_summary_cells": total, "chronological_prediction_range": [0, last],
            "common_population_checked": True, "terminal_output_prediction": False}


def _response(ax, frame, metadata):
    grid = np.asarray(metadata.get("dose_grid", []), dtype=float)
    guidance = float(metadata.get("guidance_scale", np.nan))
    if not len(grid) or not math.isfinite(guidance) or guidance < 1:
        raise TheoryError("Applicable dose response requires saved g>=1 and its exact grid")
    if not all(np.any(grid == value) for value in (0., 1 / guidance, 1.)):
        raise TheoryError("Fixed dose grid omits a required matched endpoint")
    threshold = float(metadata.get("symlog_linthresh", np.nan))
    if threshold != 1e-3:
        raise TheoryError("Dose response requires its fixed 1e-3 natural-log linear threshold")
    handles = []
    for group in _groups(frame):
        rows = frame.loc[frame.group.eq(group)].sort_values("s")
        x, y = _number(rows, "s"), _number(rows, "median")
        if not np.array_equal(x, grid):
            raise TheoryError("Dose summaries do not span the exact complete saved grid")
        for name in ("denominator_weight", "eligible_count"):
            if len(set(_number(rows, name))) != 1:
                raise TheoryError("Dose response silently changes population across s")
        lower, upper = _number(rows, "q25"), _number(rows, "q75")
        if not np.isfinite(np.r_[y, lower, upper]).all() or np.any((lower > y) | (upper < y)):
            raise TheoryError("Invalid finite complete-cohort dose summaries")
        color = GROUP_COLORS[group]
        ax.plot(x, y, color=color, linewidth=1.5)
        ax.fill_between(x, lower, upper, color=color, alpha=.16, linewidth=0)
        handles.append(Line2D([], [], color=color, label=GROUP_LABELS[group]))
    ax.set_yscale("symlog", linthresh=threshold, base=10)
    ax.axhline(0, color=".55", linewidth=.8)
    for position in (0., 1 / guidance, 1.):
        ax.axvline(position, color=".55", linestyle=":", linewidth=.8)
    ax.set_xticks(sorted(set([0., 1 / guidance, .5, 1.])))
    handles.extend([
        Line2D([], [], color=".55", linestyle=":", label="s=0: unconditional"),
        Line2D([], [], color=".55", linestyle=":", label="s=1/g: conditional-only"),
        Line2D([], [], color=".55", linestyle=":", label="s=1: " + metadata.get("cfg_endpoint_label", "Reconstructed matched CFG")),
    ])
    values = np.concatenate([_number(frame, name) for name in ("minimum", "maximum", "q25", "q75")])
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise TheoryError("Dose response has no finite saved range")
    lo, hi = min(0., float(finite.min())), max(0., float(finite.max()))
    pad = max((hi - lo) * .05, threshold)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, 1)
    _legend(ax, handles, loc="upper left", size=10)
    return {"common_cohort_across_doses": True, "dose_count": len(grid),
            "fixed_snapshot": 0, "log_odds_substitution": False,
            "symlog_linthresh": threshold, "cfg_endpoint_label": metadata.get("cfg_endpoint_label")}


def _reference(ax, frame, metadata):
    boundary = _snr(ax, metadata)
    if set(frame.source_range.astype(str)) - {"analytical", "native"}:
        raise TheoryError("Reference comparison mixes unsupported input-law ranges")
    analytical = frame.loc[frame.source_range.eq("analytical")]
    native = frame.loc[frame.source_range.eq("native")]
    if analytical.empty or analytical.metric.ne("reference").any():
        raise TheoryError("Below-native extension must remain reference-only")
    x = _number(analytical, "snr")
    if np.any(x <= 0) or np.any(x > boundary * (1 + 1e-12)):
        raise TheoryError("Analytical reference extension has invalid saved SNR")
    lower = float(x.min())
    ax.axvspan(lower, boundary, color=".85", alpha=.22, zorder=-1)
    _curve(ax, analytical, color=_REFERENCE["reference"][1], style=":", band=True, native=False)
    handles = []
    for metric, (label, color, style) in _REFERENCE.items():
        rows = native.loc[native.metric.eq(metric)]
        if rows.empty:
            raise TheoryError("Full native comparison is missing a required saved quantity")
        if np.any(_number(rows, "snr") < boundary * (1 - 1e-12)):
            raise TheoryError("Network/reference native curve extends into analytical-only range")
        _curve(ax, rows, color=color, style=style, band=True)
        handles.append(Line2D([], [], color=color, linestyle=style, label=label))
    handles.append(Line2D([], [], color=".45", linestyle=":", label=r"$\mathrm{SNR}_T$"))
    _legend(ax, handles, loc="upper left", size=10)
    _nonnegative_limits(ax, [_number(frame, name) for name in ("minimum", "maximum", "q25", "q75")])
    return {"analytical_only_range": [lower, boundary], "native_sweep_preserved": True,
            "network_extrapolation": False}


def _peak(ax, frame, metadata):
    _index_axis(ax, metadata)
    handles = []
    for group in _groups(frame):
        rows = frame.loc[frame.group.eq(group)].sort_values("step_index")
        values = _number(rows, "fraction")
        if np.any((values < 0) | (values > 1)) or values.sum() > 1 + 1e-12:
            raise TheoryError("Invalid saved individual-peak probability mass")
        color = GROUP_COLORS[group]
        ax.step(_number(rows, "step_index"), values, where="mid", color=color)
        handles.append(Line2D([], [], color=color, label=GROUP_LABELS[group]))
    ax.set_ylim(0, 1)
    _legend(ax, handles, loc="upper right", size=10)
    return {"individual_peak_distribution": True, "flat_cases_not_assigned": True,
            "shape_group_counts": metadata.get("shape_group_counts", {})}


def _motion(ax, frame, metadata):
    _common_curves(frame, tuple(_MOTION))
    last = _index_axis(ax, metadata, motion=True)
    handles = []
    for metric, (label, color, style) in _MOTION.items():
        rows = frame.loc[frame.metric.eq(metric)].sort_values("step_index")
        x = _number(rows, "step_index")
        if np.any((x < 0) | (x > last)):
            raise TheoryError("Motion is not a consecutive-prediction comparison")
        breaks = np.flatnonzero(np.diff(x) != 1) + 1
        for start, stop in zip(np.r_[0, breaks], np.r_[breaks, len(rows)]):
            segment = rows.iloc[start:stop]
            ax.plot(_number(segment, "step_index"), _number(segment, "mean"), color=color,
                    linestyle=style, linewidth=1.4, marker="." if len(segment) == 1 else None)
        handles.append(Line2D([], [], color=color, linestyle=style, label=label))
    ax.axhline(0, color=".5", linewidth=.8)
    values = np.concatenate([_number(frame, name) for name in ("mean", "minimum", "maximum")])
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise TheoryError("Motion summaries have no finite values")
    lo, hi = min(0., float(finite.min())), max(0., float(finite.max()))
    pad = max(.05 * (hi - lo), .01 if hi == lo else 0.)
    ax.set_ylim(lo - pad, hi + pad)
    _legend(ax, handles, loc="upper right", size=10)
    return {"additive_common_population": True, "summary_statistic": "weighted_mean"}


def _terminal(fig, ax, frame, metadata, *, counterfactual=False):
    x, y, score = (_number(frame, name) for name in ("x", "y", "terminal_sscd"))
    if np.any(np.isfinite(x) & (x < 0)) or np.any(np.isfinite(y) & (y < 0)):
        raise TheoryError("Nonnegative endpoint comparison contains negative values")
    finite = np.isfinite(x) & np.isfinite(y)
    log = not counterfactual and metadata.get("axis_scale") == "log"
    positive = np.r_[x[np.isfinite(x) & (x > 0)], y[np.isfinite(y) & (y > 0)]]
    if log and not len(positive):
        raise TheoryError("Saved logarithmic endpoint policy has no positive domain")
    if log:
        lo, hi = float(positive.min()) / 1.2, float(positive.max()) * 1.2
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        values = np.r_[x[np.isfinite(x)], y[np.isfinite(y)]]
        if not len(values):
            raise TheoryError("No finite endpoint coordinate is available")
        lo, hi = 0., float(values.max()) * 1.06 if values.max() > 0 else 1.
    tolerance = metadata.get("tolerance_rmse")
    if tolerance is not None and math.isfinite(float(tolerance)) and float(tolerance) > 0:
        hi = max(hi, float(tolerance) * 1.06)
        if log:
            lo = min(lo, float(tolerance) / 1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    interior = finite & ((x > 0) & (y > 0) if log else True)
    colors = dict(cmap="viridis", norm=Normalize(*SSCD_COLOR_RANGE, clip=True),
                  s=SCATTER_SIZE, edgecolors="none", alpha=.8, rasterized=True)
    def points(mask, px, py, *, transform=None, marker="o"):
        if not mask.any():
            return
        selected = np.flatnonzero(mask)
        observed = np.isfinite(score[selected])
        extra = {"transform": transform} if transform is not None else {}
        ax.scatter(np.asarray(px)[observed], np.asarray(py)[observed], c=score[selected][observed], marker=marker, **colors, **extra)
        if (~observed).any():
            ax.scatter(np.asarray(px)[~observed], np.asarray(py)[~observed], color=".5", marker=marker, s=SCATTER_SIZE, **extra)
    points(interior, x[interior], y[interior])
    if log:
        zero_x = finite & (x == 0) & (y > 0)
        zero_y = finite & (y == 0) & (x > 0)
        both = finite & (x == 0) & (y == 0)
        points(zero_x, np.full(zero_x.sum(), .018), y[zero_x], transform=blended_transform_factory(ax.transAxes, ax.transData), marker="<")
        points(zero_y, x[zero_y], np.full(zero_y.sum(), .018), transform=blended_transform_factory(ax.transData, ax.transAxes), marker="v")
        points(both, np.full(both.sum(), .018), np.full(both.sum(), .018), transform=ax.transAxes, marker="s")
    infinite = np.isposinf(x) & np.isfinite(y)
    visible_infinite = infinite & ((y > 0) if log else True)
    points(visible_infinite, np.full(visible_infinite.sum(), .98), y[visible_infinite],
           transform=blended_transform_factory(ax.transAxes, ax.transData), marker=">")
    if log:
        zero_infinite = infinite & (y == 0)
        points(zero_infinite, np.full(zero_infinite.sum(), .98), np.full(zero_infinite.sum(), .018), transform=ax.transAxes, marker=">")
    add_sscd_colorbar(fig, ax, score[np.isfinite(score)])
    scope = metadata.get("terminal_scope")
    if counterfactual:
        label = "Zero measured improvement"
    elif scope == "finite_terminal_update_extension_gaussian_noise_bound":
        label = "Probability-qualified bound"
    elif scope == "original_clean_terminal_theorem":
        label = "Clean-update theorem bound"
    elif scope == "finite_terminal_update_extension_deterministic":
        certainty = metadata.get("terminal_certainty", [])
        label = "Pathwise extension bound" if any("pathwise" in str(value) for value in certainty) else "Deterministic extension bound"
    else:
        raise TheoryError("Terminal scatter lacks a supported recorded correction scope")
    ax.plot([lo, hi], [lo, hi], color=".4", linestyle="--", linewidth=1, label=label)
    handles = [Line2D([], [], color=".4", linestyle="--", label=label)]
    if log and ((x == 0).any() or (y == 0).any()):
        handles.append(Line2D([], [], color=".5", marker="v", linestyle="none", label=f"Zero edges: x={int((x == 0).sum())}, y={int((y == 0).sum())}"))
    if infinite.any():
        handles.append(Line2D([], [], color=".5", marker=">", linestyle="none", label=f"Infinite bound: {int(infinite.sum())}"))
    if tolerance is not None and math.isfinite(float(tolerance)) and float(tolerance) >= 0 and (not log or float(tolerance) > 0):
        ax.axvline(float(tolerance), color=".6", linestyle=":", linewidth=.8)
        ax.axhline(float(tolerance), color=".6", linestyle=":", linewidth=.8)
        handles.append(Line2D([], [], color=".6", linestyle=":", label="Supplied latent tolerance"))
    _legend(ax, handles, loc="upper left", size=10)
    return {"finite_sample_pairs": int(finite.sum()), "infinite_bound_count": int(infinite.sum()),
            "zero_counts": {"x": int((x == 0).sum()), "y": int((y == 0).sum())},
            "zero_position_policy": "axes-edge markers, not substituted numeric values" if log else "exact data coordinates",
            "axis_scale": "log" if log else "linear", "comparison_reference": label}


def draw_four_stage(fig, ax, entry, frame, metadata, config):
    kind = entry["kind"]
    if kind == "four_response":
        return _response(ax, frame, metadata)
    if kind == "four_chronological":
        return _chronological(ax, entry, frame, metadata)
    if kind == "four_reference":
        return _reference(ax, frame, metadata)
    if kind == "four_peak":
        return _peak(ax, frame, metadata)
    if kind == "four_motion":
        return _motion(ax, frame, metadata)
    return _terminal(fig, ax, frame, metadata, counterfactual=kind == "four_counterfactual")
