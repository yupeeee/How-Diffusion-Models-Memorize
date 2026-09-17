"""Four fixed mechanism designs, using immutable compact scalar inputs only.

All cohorts, quantiles, weights, shape classifications and endpoint scopes come
from the analysis stage. This module only validates and positions artists.
"""
from __future__ import annotations

import json
import math

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

from utils.experiments.plotting import SCATTER_ALPHA, SCATTER_SIZE, SSCD_COLOR_RANGE, add_sscd_colorbar
from .contracts import TheoryError
from .evidence_plotting import (
    _REFERENCE, _SCOPES, _curve, _ecdf_support, _groups, _legend, _nonnegative_limits,
    _number, _snr,
)
from .paper_registry import GROUPS, GROUP_COLORS, GROUP_LABELS, RETIRED_RENDER_STEMS
from .paper_notation import (
    CHRONOLOGICAL_LABELS, DOSE_REFERENCE_TICK_LABEL, NOTATION_VERSION, REFERENCE_LABELS, TERMINAL_LABELS, ZERO_REFERENCE_LABELS,
)

FOUR_STAGE_RENDERING_VERSION = NOTATION_VERSION
# Saved compact-table identities; rendering never imports the analysis reducer.
_PROMPT_CURVE_KEYS = ("run_id", "original_index", "record_id", "target_id")
FOUR_STAGE_KINDS = frozenset({
    "four_response", "four_chronological", "four_reference", "four_peak",
    "four_terminal_scatter", "four_counterfactual", "four_prompt_chronological", "four_terminal_grouped_cdf",
    "four_guidance_fit",
})
_QUANTITIES = {
    metric: (CHRONOLOGICAL_LABELS[metric], style)
    for metric, style in (("gap", "--"), ("joint_error", "-"),
                          ("bound", "-."), ("conditional", "-"), ("unconditional", "--"))
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
    ax.set_xticks(ticks, [str(value) for value in ticks])
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
    styles = {metric: _QUANTITIES[metric][1] for metric in metrics}
    if entry["stem"] == "synchronization_bound":
        metrics = ("gap", "bound")
        # Historical compact tables also retain the paired maximum target
        # distance. Omit only its artwork; keep the saved values unchanged.
        frame = frame.loc[~frame.metric.eq("joint_error")]
        styles = {"gap": "-", "bound": "--"}
    _common_curves(frame, metrics)
    last = _index_axis(ax, metadata)
    quantities, groups, total = [], [], 0
    displayed = []
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
                displayed.append(y)
                ax.plot(x, y, color=color, linestyle=styles[metric], linewidth=1.5,
                        marker="." if len(part) == 1 else None, markersize=3)
                if metric == "joint_error" or entry["stem"] == "branch_target_errors":
                    lower, upper = _number(part, "q25"), _number(part, "q75")
                    if np.any(np.isfinite(y) & ((lower > y) | (upper < y))):
                        raise TheoryError("Saved IQR does not bracket the median")
                    ax.fill_between(x, lower, upper, color=color, alpha=.10, linewidth=0)
                    displayed.extend([lower, upper])
    for metric in metrics:
        label = _QUANTITIES[metric][0]
        quantities.append(Line2D([], [], color=".3", linestyle=styles[metric], label=label))
    _nonnegative_limits(ax, displayed)
    legend_audit = {}
    if entry["stem"] == "synchronization_bound":
        # Match terminal-bound coverage: one column for SSCD groups and one
        # for quantities, with the complete legend above the data axes.
        groups.extend(Line2D([], [], linestyle="none", alpha=0., label="")
                      for _ in range(len(quantities) - len(groups)))
        ax.legend(handles=[*groups, *quantities], loc="lower center", bbox_to_anchor=(.5, 1.01),
                  bbox_transform=ax.transAxes, ncol=2, frameon=False, borderaxespad=0,
                  fontsize=10, handlelength=1.65,
                  handletextpad=.4, columnspacing=.7, labelspacing=.3)
        legend_audit = {"legend_placement": "outside top", "legend_columns": "SSCD groups; manuscript quantities"}
    else:
        first = _legend(ax, groups, loc="upper left", size=10)
        if first is not None:
            ax.add_artist(first)
        _legend(ax, quantities, loc="upper right", size=10)
    return {"finite_summary_cells": total, "chronological_prediction_range": [0, last],
            "displayed_metrics": list(metrics), "quantity_styles": styles, **legend_audit,
            "common_population_checked": True, "terminal_output_prediction": False,
            "horizontal_coordinate": "T-t equals saved chronological index k; 0 is initialization, not native timestep zero",
            "display_range_policy": "Complete displayed medians and drawn IQRs plus fixed 6% upper padding; raw extrema remain saved"}


def _prompt_chronological(fig, ax, frame, metadata, entry):
    """Draw saved prompt means; all seed aggregation belongs to analysis."""
    value_column = entry.get("value_column", "mean_gap_rmse")
    domain = entry.get("value_domain", "nonnegative")
    name = "Per-prompt branch-gap" if value_column == "mean_gap_rmse" else "Per-prompt trajectory"
    if domain not in {"nonnegative", "probability"}:
        raise TheoryError(name + " plot has an unknown value domain")
    for field, expected in (("value_column", value_column), ("value_domain", domain)):
        if metadata.get(field, expected) != expected:
            raise TheoryError(name + " plot disagrees with its saved " + field)
    prediction_domain = entry.get("prediction_domain", "pre_update_predictions")
    if prediction_domain not in {"pre_update_predictions", "positive_noise_transitions"}:
        raise TheoryError(name + " plot has an unknown prediction domain")
    transitions = prediction_domain == "positive_noise_transitions"
    # Transition-only quantities require an explicit saved receipt. A full
    # prediction table's final zero sentinel cannot stand in for measured V_t.
    saved_domain = metadata.get("prediction_domain", None if transitions else prediction_domain)
    if saved_domain != prediction_domain:
        raise TheoryError(name + " plot disagrees with its saved prediction_domain")
    required = list(_PROMPT_CURVE_KEYS) + ["step_index", value_column, "mean_terminal_sscd", "seed_count", "seed_ids_json", "cohort_complete"]
    if not set(required).issubset(frame) or frame.empty or frame[list(_PROMPT_CURVE_KEYS)].isna().any().any():
        raise TheoryError(name + " plot lacks complete saved curve identities")
    last = _index_axis(ax, metadata, motion=transitions)
    seeds = metadata.get("expected_seed_ids")
    count = metadata.get("expected_seed_count")
    if not isinstance(count, int) or count < 1 or seeds != list(range(count)):
        raise TheoryError(name + " plot lacks its declared seed cohort")
    norm, cmap = Normalize(*SSCD_COLOR_RANGE, clip=True), colormaps["viridis"]
    displayed, colors, curves = [], [], 0
    for _, rows in frame.groupby(list(_PROMPT_CURVE_KEYS), sort=True, dropna=False):
        rows = rows.sort_values("step_index")
        x, y = _number(rows, "step_index"), _number(rows, value_column)
        color = _number(rows, "mean_terminal_sscd")
        if not np.array_equal(x, np.arange(last + 1)):
            expected_domain = "positive-noise transition" if transitions else "pre-update prediction"
            raise TheoryError(name + " curve must contain every " + expected_domain + " exactly once")
        if not (np.isfinite(y).all() and (y >= 0).all() and np.isfinite(color).all() and (color == color[0]).all()):
            raise TheoryError(name + " curve requires finite nonnegative values and one fixed SSCD color")
        if domain == "probability" and (y > 1).any():
            raise TheoryError(name + " probabilities must lie in [0, 1]")
        if not (rows.cohort_complete.eq(True).all() and (_number(rows, "seed_count") == count).all()):
            raise TheoryError(name + " curve changed its complete seed cohort")
        try:
            same_seeds = all(json.loads(value) == seeds for value in rows.seed_ids_json)
        except (TypeError, ValueError):
            same_seeds = False
        if not same_seeds:
            raise TheoryError(name + " curve changed its saved seed identities")
        ax.plot(x, y, color=cmap(norm(color[0])), linewidth=1.1, alpha=.7,
                marker="." if len(rows) == 1 else None, markersize=3)
        displayed.append(y)
        colors.append(color[0])
        curves += 1
    if domain == "probability":
        ax.set_ylim(0., 1.)
    else:
        _nonnegative_limits(ax, displayed)
    add_sscd_colorbar(fig, ax, np.asarray(colors))
    return {"plotted_prompt_target_curves": curves, "chronological_prediction_range": [0, last],
            "terminal_output_prediction": False, "complete_seed_cohort_checked": True,
            "horizontal_coordinate": "T-t equals saved chronological index k; 0 is initialization, not native timestep zero",
            "aggregation": ("Saved arithmetic mean of per-seed probabilities; not the exponential of a mean log probability"
                            if domain == "probability" else "Saved arithmetic mean of seed norms; no plot-time aggregation or normalization"),
            "value_column": value_column, "value_domain": domain,
            "prediction_domain": prediction_domain,
            "last_pre_update_sentinel_excluded": transitions,
            "numerical_scope": metadata.get("numerical_scope"),
            "color_population": "Saved mean terminal SSCD across the same seed cohort, fixed along each curve",
            "sscd_color_range": list(SSCD_COLOR_RANGE), "colorbar_label": "SSCD",
            "out_of_color_range_curves": int(np.count_nonzero((np.asarray(colors) < SSCD_COLOR_RANGE[0]) | (np.asarray(colors) > SSCD_COLOR_RANGE[1]))),
            "display_range_policy": ("Full probability interval [0, 1]"
                                     if domain == "probability" else "All saved prompt mean curves, zero origin and fixed 6% upper padding")}


def _guidance_fit(fig, ax, frame, metadata, config):
    """Display sqrt(saved loss ratio) against already-fitted prompt coefficients."""
    name = "Guidance-fit"
    required = set(_PROMPT_CURVE_KEYS) | {
        "x", "y", "mean_terminal_sscd", "seed_count", "seed_ids_json", "latent_dimension",
        "snr", "guidance_scale", "fit_residual_rmse", "direction_norm_rmse",
    }
    if not required.issubset(frame) or frame.empty:
        raise TheoryError(name + " plot lacks its saved pair quantities")
    if (frame[list(_PROMPT_CURVE_KEYS)].isna().any().any()
            or frame.duplicated(list(_PROMPT_CURVE_KEYS)).any()
            or any(frame[key].astype(str).str.strip().eq("").any() for key in _PROMPT_CURVE_KEYS)):
        raise TheoryError(name + " plot requires unique complete prompt-target identities")
    science = config.get("scientific_config", config)
    try:
        guidance = float(science["guidance_scale"])
        recorded_guidance = float(metadata["guidance_scale"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise TheoryError(name + " plot lacks its configured guidance reference") from error
    if (not math.isfinite(guidance) or not math.isfinite(recorded_guidance)
            or not math.isclose(guidance, recorded_guidance, rel_tol=1e-12, abs_tol=0.)):
        raise TheoryError(name + " guidance reference differs from its configuration")
    expected_count, expected_seeds = metadata.get("expected_seed_count"), metadata.get("expected_seed_ids")
    if (type(expected_count) is not int or expected_count < 1
            or type(science.get("num_seeds")) is not int or science["num_seeds"] != expected_count
            or not isinstance(expected_seeds, list)
            or any(type(seed) is not int for seed in expected_seeds)
            or expected_seeds != list(range(expected_count))):
        raise TheoryError(name + " plot lacks its configured complete seed cohort")
    if metadata.get("aggregation") != "joint_no_intercept_least_squares_shared_target_direction":
        raise TheoryError(name + " plot lacks its pooled least-squares aggregation receipt")
    if not isinstance(metadata.get("fit_definition"), str) or not metadata["fit_definition"].strip():
        raise TheoryError(name + " plot lacks its saved fit definition")
    if metadata.get("status") not in {"available", "complete"}:
        raise TheoryError(name + " plot cannot display an unavailable or failed reduction")
    values = {field: _number(frame, field) for field in required - set(_PROMPT_CURVE_KEYS) - {"seed_ids_json"}}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise TheoryError(name + " plot requires finite saved quantities for every pair")
    x, y, score = values["x"], values["y"], values["mean_terminal_sscd"]
    if ((x < 0).any() or (values["fit_residual_rmse"] < 0).any()
            or (values["direction_norm_rmse"] <= 0).any()):
        raise TheoryError(name + " plot requires nonnegative loss/residual and positive fit direction")
    dimension = values["latent_dimension"]
    if ((dimension < 1).any() or (dimension != np.floor(dimension)).any()
            or not np.all(dimension == dimension[0])):
        raise TheoryError(name + " plot has inconsistent latent dimensions")
    if not np.allclose(values["guidance_scale"], guidance, rtol=1e-12, atol=0.):
        raise TheoryError(name + " table guidance differs from its configured reference")
    snr = values["snr"]
    if (snr <= 0).any() or not np.allclose(snr, snr[0], rtol=1e-12, atol=0.):
        raise TheoryError(name + " plot requires one positive finite initial SNR")
    if metadata.get("actual_initial_snr") is not None:
        try:
            initial_snr = float(metadata["actual_initial_snr"])
        except (TypeError, ValueError, OverflowError) as error:
            raise TheoryError(name + " plot has an invalid saved initial SNR") from error
        if not math.isfinite(initial_snr) or not math.isclose(initial_snr, float(snr[0]), rel_tol=1e-12, abs_tol=0.):
            raise TheoryError(name + " initial SNR differs from its measurement receipt")
    if not (values["seed_count"] == expected_count).all():
        raise TheoryError(name + " pair does not preserve its complete seed cohort")
    if "cohort_complete" in frame and not frame.cohort_complete.eq(True).all():
        raise TheoryError(name + " plot contains an incomplete seed cohort")
    try:
        seed_lists = [json.loads(value) for value in frame.seed_ids_json]
        complete = all(isinstance(seeds, list) and all(type(seed) is int for seed in seeds)
                       and seeds == expected_seeds for seeds in seed_lists)
    except (TypeError, ValueError):
        complete = False
    if not complete:
        raise TheoryError(name + " pair has inconsistent saved seed identities")
    plotted_count = metadata.get("counts", {}).get("plotted_pairs")
    if plotted_count is not None and plotted_count != len(frame):
        raise TheoryError(name + " pair count differs from its saved reduction receipt")
    # Compact recipe 1 stores L_T(c)/(d*SNR_T). Apply its display square
    # root once, after validating nonnegativity, without rewriting saved inputs.
    plotted_x = np.sqrt(x)
    ax.scatter(plotted_x, y, c=score, cmap="viridis", norm=Normalize(*SSCD_COLOR_RANGE, clip=True),
               s=SCATTER_SIZE, edgecolors="none", alpha=SCATTER_ALPHA, rasterized=True)
    add_sscd_colorbar(fig, ax, score)
    label = rf"$g={guidance:g}$"
    ax.axhline(guidance, color=".4", linestyle="--", linewidth=1, label=label)
    _legend(ax, [Line2D([], [], color=".4", linestyle="--", label=label)], loc="center right", size=10)
    xmax = float(plotted_x.max())
    xupper = xmax * 1.06 if xmax > 0 else 1.
    lower, upper = min(float(y.min()), guidance), max(float(y.max()), guidance)
    span = upper - lower
    padding = .06 * (span if span > 0 else max(abs(lower), 1.))
    if not all(math.isfinite(value) for value in (xupper, lower - padding, upper + padding)):
        raise TheoryError(name + " complete display range is not representable")
    ax.set_xlim(0., xupper)
    ax.set_ylim(lower - padding, upper + padding)
    return {
        "plotted_prompt_target_pairs": len(frame), "excluded_compact_rows": 0,
        "complete_seed_cohort_checked": True, "pair_identity_uniqueness_checked": True,
        "aggregation": metadata["aggregation"], "fit_definition": metadata["fit_definition"],
        "fit_recomputed_by_renderer": False, "dimensional_normalization_applied_by_renderer": False,
        "square_root_applied_to_loss": True, "x_transform": "sqrt", "guidance_estimate_clipping": False,
        "saved_horizontal_coordinate": "L_T(c)/(d*SNR_T)",
        "horizontal_coordinate": "sqrt(L_T(c)/(d*SNR_T)); square root of the saved forward-draw mean, not mean of draw-wise roots",
        "vertical_coordinate": "Saved signed pooled no-intercept least-squares guidance estimate",
        "guidance_reference_value": guidance, "guidance_reference_label": label,
        "legend_placement": "inside upper left", "axis_scale": "linear",
        "sscd_color_range": list(SSCD_COLOR_RANGE), "colorbar_label": "SSCD",
        "out_of_color_range_pairs": int(np.count_nonzero((score < SSCD_COLOR_RANGE[0]) | (score > SSCD_COLOR_RANGE[1]))),
        "display_range_policy": "All finite saved pairs and configured guidance; nonnegative loss axis; signed fit axis with 6 percent padding",
    }


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
    # The boundary endpoints remain measured and ticked, but only the matched
    # conditional-only state needs an interior reference line.
    ax.axvline(1 / guidance, color=".55", linestyle=":", linewidth=.8)
    ticks = sorted(set([0., 1 / guidance, .5, 1.]))
    ax.set_xticks(ticks, [DOSE_REFERENCE_TICK_LABEL if value == 1 / guidance else f"{value:g}"
                         for value in ticks])
    values = np.concatenate([_number(frame, name) for name in ("minimum", "maximum", "q25", "q75")])
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise TheoryError("Dose response has no finite saved range")
    lo, hi = min(0., float(finite.min())), max(0., float(finite.max()))
    pad = max((hi - lo) * .05, threshold)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, 1)
    _legend(ax, handles, loc="lower right", size=10)
    return {"common_cohort_across_doses": True, "dose_count": len(grid),
            "fixed_snapshot": 0, "log_odds_substitution": False,
            "legend_placement": "inside lower right",
            "vertical_reference_positions": [1 / guidance],
            "vertical_reference_meaning": "Matched conditional-only update at s=1/g",
            "vertical_reference_label_placement": "x-axis tick",
            "vertical_reference_tick_label": DOSE_REFERENCE_TICK_LABEL,
            "symlog_linthresh": threshold, "cfg_endpoint_label": metadata.get("cfg_endpoint_label")}


def _reference(ax, frame, metadata, *, zero_baseline=False):
    labels = ZERO_REFERENCE_LABELS if zero_baseline else REFERENCE_LABELS
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
    note = ax.text(math.sqrt(lower * boundary), .02, "Analytical reference only",
                   transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
    note.set_gid("curation-required-label")
    _curve(ax, analytical, color=_REFERENCE["reference"][1], style=":", band=True, native=False)
    handles = []
    for metric, (label, color, style) in _REFERENCE.items():
        rows = native.loc[native.metric.eq(metric)]
        if rows.empty:
            raise TheoryError("Full native comparison is missing a required saved quantity")
        if np.any(_number(rows, "snr") < boundary * (1 - 1e-12)):
            raise TheoryError("Network/reference native curve extends into analytical-only range")
        _curve(ax, rows, color=color, style=style, band=True)
        handles.append(Line2D([], [], color=color, linestyle=style, label=labels[metric]))
    displayed = [_number(frame, name) for name in ("minimum", "maximum", "q25", "q75")]
    limit_field = "bank_mean_norm_rmse" if zero_baseline else "mean_offset_rmse"
    reference_limit = float(metadata.get(limit_field, math.nan))
    if not math.isfinite(reference_limit) or reference_limit < 0:
        raise TheoryError("Reference comparison lacks its saved finite-bank limit: " + limit_field)
    saved_limit = metadata.get("reference_limit_rmse", reference_limit)
    if not math.isclose(float(saved_limit), reference_limit, rel_tol=1e-12, abs_tol=1e-14):
        raise TheoryError("Reference limit metadata disagrees with the selected comparison centre")
    ax.axhline(reference_limit, color=".5", linestyle="--", linewidth=.8)
    displayed.append(np.asarray([reference_limit]))
    handles.append(Line2D([], [], color=".45", linestyle=":", label=r"$\mathrm{SNR}_T$"))
    _legend(ax, handles, loc="upper left", size=10)
    _nonnegative_limits(ax, displayed)
    audit = {"analytical_only_range": [lower, boundary], "native_sweep_preserved": True,
             "network_extrapolation": False}
    audit.update(comparison_centre="zero" if zero_baseline else "selected_mu",
                 declared_law_mean_unchanged=True,
                 horizontal_reference_value=reference_limit,
                 horizontal_reference_label="Reference limit",
                 horizontal_reference_legend_visible=False,
                 horizontal_reference_meaning=("Lower-SNR norm of the exact finite-bank mean divided by sqrt(d); distinct from selected mu"
                    if zero_baseline else "Lower-SNR distance between the exact finite-bank mean and selected mu divided by sqrt(d)"))
    return audit


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
    masses = _number(frame, "fraction")
    largest = float(masses[np.isfinite(masses)].max()) if np.isfinite(masses).any() else 0.
    ax.set_ylim(0, largest * 1.06 if largest > 0 else .05)
    _legend(ax, handles, loc="upper right", size=10)
    return {"individual_peak_distribution": True, "flat_cases_not_assigned": True,
            "shape_group_counts": metadata.get("shape_group_counts", {}),
            "y_range_policy": "Full displayed weighted bin masses plus fixed 6% padding; all-zero display window 0..0.05"}



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
                  s=SCATTER_SIZE, edgecolors="none", alpha=SCATTER_ALPHA, rasterized=True)
    def points(mask, px, py, *, transform=None, marker="o"):
        if not mask.any():
            return
        selected = np.flatnonzero(mask)
        observed = np.isfinite(score[selected])
        extra = {"transform": transform} if transform is not None else {}
        ax.scatter(np.asarray(px)[observed], np.asarray(py)[observed], c=score[selected][observed], marker=marker, **colors, **extra)
        if (~observed).any():
            ax.scatter(np.asarray(px)[~observed], np.asarray(py)[~observed], color=".5", marker=marker, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, **extra)
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
    comparison_scope = label
    equality = r"$x=y$"
    ax.plot([lo, hi], [lo, hi], color=".4", linestyle="--", linewidth=1, label=equality)
    handles = [Line2D([], [], color=".4", linestyle="--", label=equality)]
    if log and ((x == 0).any() or (y == 0).any()):
        handles.append(Line2D([], [], color=".5", marker="v", linestyle="none", label=rf"$x=0$: {int((x == 0).sum())}; $y=0$: {int((y == 0).sum())}"))
    if infinite.any():
        handles.append(Line2D([], [], color=".5", marker=">", linestyle="none", label=rf"$x=+\infty$: {int(infinite.sum())}"))
    if tolerance is not None and math.isfinite(float(tolerance)) and float(tolerance) >= 0 and (not log or float(tolerance) > 0):
        ax.axvline(float(tolerance), color=".6", linestyle=":", linewidth=.8)
        ax.axhline(float(tolerance), color=".6", linestyle=":", linewidth=.8)
        handles.append(Line2D([], [], color=".6", linestyle=":", label=r"$\tau/\sqrt{d}$"))
    legend = _legend(ax, handles, loc="upper left", size=10)
    if legend is not None:
        titles = {"Zero measured improvement": "Matched comparison",
                  "Probability-qualified bound": "Probabilistic extension",
                  "Clean-update theorem bound": "Theorem 7",
                  "Pathwise extension bound": "Pathwise extension",
                  "Deterministic extension bound": "Deterministic extension"}
        legend.set_title(titles[comparison_scope], prop={"size": 10})
    return {"finite_sample_pairs": int(finite.sum()), "infinite_bound_count": int(infinite.sum()),
            "zero_counts": {"x": int((x == 0).sum()), "y": int((y == 0).sum())},
            "zero_position_policy": "axes-edge markers, not substituted numeric values" if log else "exact data coordinates",
            "axis_scale": "log" if log else "linear", "comparison_reference": comparison_scope,
            "diagonal_label": equality, "diagonal_coordinate_scope": "Scalar plot coordinates x and y; not latent-vector notation"}



def _terminal_grouped_cdf(ax, frame, metadata):
    """Render saved exact ECDFs; group membership and weights are never rebuilt."""
    names = ("actual", "observable", "reference")
    styles = {"actual": "-", "observable": "--", "reference": "-."}
    columns = {"group", "distribution", "value", "cdf", "denominator_weight", "eligible_count", "prompt_count"}
    if not columns.issubset(frame) or frame.group.isna().any():
        raise TheoryError("Grouped terminal coverage requires saved group/population columns")
    present = _groups(frame)
    zero, infinite = metadata.get("grouped_zero_mass"), metadata.get("grouped_infinite_mass")
    populations = metadata.get("group_population")
    if any(not isinstance(value, dict) or set(value) != set(GROUPS) for value in (zero, infinite, populations)):
        raise TheoryError("Grouped terminal coverage lacks exact saved group populations or masses")
    scope = metadata.get("terminal_scope")
    if scope not in _SCOPES:
        raise TheoryError(f"Unsupported terminal display scope: {scope}")
    if scope != "original_clean_terminal_theorem" and metadata.get("manuscript_extension_required") is not True:
        raise TheoryError("A finite terminal extension must declare its manuscript requirement")
    clean = metadata.get("original_clean_counts")
    if not isinstance(clean, dict) or not {"applicable", "total"}.issubset(clean):
        raise TheoryError("Terminal coverage requires original clean-update applicability counts")
    support, empty = {}, []
    for group in GROUPS:
        population = populations[group]
        if not isinstance(population, dict) or not {"denominator_weight", "eligible_count", "prompt_count"}.issubset(population):
            raise TheoryError("Grouped terminal coverage has an incomplete population receipt")
        expected = {}
        for field in ("denominator_weight", "eligible_count", "prompt_count"):
            try:
                value = float(population[field])
            except (TypeError, ValueError) as error:
                raise TheoryError("Grouped terminal coverage has an invalid population receipt") from error
            if not math.isfinite(value) or value < 0 or (field != "denominator_weight" and value != math.floor(value)):
                raise TheoryError("Grouped terminal coverage has an invalid population receipt")
            expected[field] = value
        if any(not isinstance(mapping[group], dict) or set(mapping[group]) != set(names) for mapping in (zero, infinite)):
            raise TheoryError("Grouped terminal coverage lacks saved zero/infinite mass for a quantity")
        if expected["eligible_count"] == 0:
            if group in present or expected["denominator_weight"] != 0 or expected["prompt_count"] != 0:
                raise TheoryError("Empty terminal group contains a curve or nonzero denominator")
            if any(mapping[group][name] is not None for mapping in (zero, infinite) for name in names):
                raise TheoryError("Empty terminal group must retain undefined ECDF mass")
            empty.append(group)
            continue
        if expected["denominator_weight"] <= 0 or not 0 < expected["prompt_count"] <= expected["eligible_count"]:
            raise TheoryError("Nonempty terminal group requires positive saved population weight and counts")
        selected = frame.loc[frame.group.eq(group)]
        if set(selected.distribution.astype(str)) != set(names):
            raise TheoryError("Grouped terminal coverage requires all three quantities on each nonempty population")
        for field, value in expected.items():
            observed = _number(selected, field)
            if not np.isfinite(observed).all() or not np.all(observed == value):
                raise TheoryError("Compared terminal curves have different saved group populations")
        for name in names:
            values, cdf = _ecdf_support(selected.loc[selected.distribution.eq(name)])
            try:
                z, inf = float(zero[group][name]), float(infinite[group][name])
            except (TypeError, ValueError) as error:
                raise TheoryError("Nonempty terminal group requires finite saved zero/infinite masses") from error
            actual_zero = float(cdf[values == 0][0]) if np.any(values == 0) else 0.
            finite = np.isfinite(values)
            actual_infinite = 1. - float(cdf[finite][-1]) if finite.any() else 1.
            if (not math.isfinite(z) or not 0 <= z <= 1 or abs(z - actual_zero) > 1e-12
                    or not math.isfinite(inf) or not 0 <= inf <= 1 or abs(inf - actual_infinite) > 1e-10):
                raise TheoryError("Saved grouped terminal zero/infinite mass differs from exact ECDF support")
            support[group, name] = values, cdf
    if not support:
        raise TheoryError("Available grouped terminal coverage has no nonempty saved group")
    positives = np.concatenate([values[(values > 0) & np.isfinite(values)] for values, _ in support.values()])
    limits = ((float(positives.min()) / 1.15, float(positives.max()) * 1.15)
              if len(positives) else (1., 10.))
    tolerance = metadata.get("predeclared_tolerance_rmse")
    if tolerance is not None:
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise TheoryError("A predeclared tolerance must be nonnegative and finite")
        if tolerance > 0:
            limits = (min(limits[0], tolerance / 1.15), max(limits[1], tolerance * 1.15))
    for (group, name), (values, cdf) in support.items():
        positive = (values > 0) & np.isfinite(values)
        x = np.r_[limits[0], values[positive], limits[1]]
        y = np.r_[float(zero[group][name]), cdf[positive], 1. - float(infinite[group][name])]
        line, = ax.step(x, y, where="post", color=GROUP_COLORS[group], linestyle=styles[name],
                        linewidth=1.4, label=TERMINAL_LABELS[name])
        line.set_gid(group + ":" + name)
    ax.set_xscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(0, 1)
    quantities = [Line2D([], [], color=".25", linestyle=styles[name], label=TERMINAL_LABELS[name]) for name in names]
    if tolerance is not None and tolerance > 0:
        ax.axvline(tolerance, color=".5", linestyle=":", linewidth=.8)
        quantities.append(Line2D([], [], color=".5", linestyle=":", label=r"$\tau/\sqrt{d}$"))
    groups = [Line2D([], [], color=GROUP_COLORS[group], label=GROUP_LABELS[group]) for group in GROUPS if group not in empty]
    # Matplotlib fills columns first. Blank handles keep the outcome keys in
    # one column and the three manuscript quantities in the other.
    groups.extend(Line2D([], [], linestyle="none", alpha=0., label="") for _ in range(len(quantities) - len(groups)))
    ax.legend(handles=[*groups, *quantities], loc="lower center", bbox_to_anchor=(.5, 1.01),
              bbox_transform=ax.transAxes, ncol=2, frameon=False, borderaxespad=0,
              fontsize=10, handlelength=1.65,
              handletextpad=.4, columnspacing=.7, labelspacing=.3)
    return {"terminal_scope": scope, "terminal_scope_display": "caption_only",
            "grouped_zero_mass": zero, "grouped_infinite_mass": infinite,
            "group_population": populations, "empty_groups": empty,
            "group_common_population_table": metadata.get("group_common_population_table"),
            "full_positive_support": [float(positives.min()), float(positives.max())] if len(positives) else None,
            "empty_positive_domain": not len(positives), "zero_replacement": None,
            "original_clean_counts": clean, "empirical_cdf_confidence_band": False,
            "legend_placement": "outside top", "legend_columns": "SSCD groups; manuscript quantities",
            "legend_quantities": {name: TERMINAL_LABELS[name] for name in names},
            "group_colors": {group: GROUP_COLORS[group] for group in GROUPS}, "quantity_styles": styles,
            "plotted_curve_count": len(support), "common_population_checked": True,
            "terminal_labels_abbreviated": True,
            "terminal_correction_display": "caption_only; corrected numerical values retained",
            "mass_disclosure": "Exact per-group zero and infinite masses in caption metadata; no epsilon replacement",
            "zero_mass_annotation_visible": False, "zero_mass_legend_visible": False,
            "infinite_mass_legend_visible": False,
            "predeclared_tolerance_rmse": tolerance, "annotations_saved_in_caption": True}

def draw_four_stage(fig, ax, entry, frame, metadata, config):
    if entry["stem"] in RETIRED_RENDER_STEMS or entry["kind"] == "four_motion":
        raise TheoryError("Requested figure render is retired; its saved scientific measurements remain available")
    kind = entry["kind"]
    if kind == "four_terminal_grouped_cdf":
        return _terminal_grouped_cdf(ax, frame, metadata)
    if kind == "four_response":
        return _response(ax, frame, metadata)
    if kind == "four_prompt_chronological":
        return _prompt_chronological(fig, ax, frame, metadata, entry)
    if kind == "four_guidance_fit":
        return _guidance_fit(fig, ax, frame, metadata, config)
    if kind == "four_chronological":
        return _chronological(ax, entry, frame, metadata)
    if kind == "four_reference":
        return _reference(ax, frame, metadata, zero_baseline=entry.get("comparison_centre") == "zero")
    if kind == "four_peak":
        return _peak(ax, frame, metadata)
    return _terminal(fig, ax, frame, metadata, counterfactual=kind == "four_counterfactual")
