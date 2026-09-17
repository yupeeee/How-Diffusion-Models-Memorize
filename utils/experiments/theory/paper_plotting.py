"""Render the fixed paper suite exclusively from validated compact scalar CSVs."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, ScalarFormatter  # noqa: E402
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.common.io import atomic_write_json, file_sha256, json_value
from utils.experiments.plotting import (
    PLOT_STYLE,
    AXIS_LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    SCATTER_ALPHA,
    SCATTER_SIZE,
    SSCD_COLOR_RANGE,
    add_sscd_colorbar,
    publish_figures,
)
from .contracts import TheoryError
from .evidence_plotting import EVIDENCE_KINDS, draw_evidence
from .four_stage_plotting import FOUR_STAGE_KINDS, draw_four_stage
from .paper_contracts import contained_path, load_paper_inputs, recompute_command
from .paper_registry import (
    GROUPS, GROUP_COLORS, GROUP_LABELS, PLOT_RECIPE_VERSION, REGISTRY_VERSION,
    RENDER_RETIREMENTS, RETIRED_RENDER_STEMS, paper_registry,
)
from .progress import StageProgress
from . import paper_style
from .paper_notation import (
    FEEDBACK_CONDITION_LABEL, FEEDBACK_GAIN_LABEL, INITIAL_BRANCH_LABELS,
    INITIAL_DISTRIBUTION_LABELS, NOTATION_VERSION, TERMINAL_LABELS,
    projected_error_display_notation,
)

RENDERING_VERSION = NOTATION_VERSION
_DISTRIBUTIONS = {
    "initial_unconditional": (r"$\hat{\mathbf{x}}_T(\varnothing)$", GROUP_COLORS[GROUPS[1]]),
    "candidate_atoms": ("Candidate atoms", GROUP_COLORS[GROUPS[0]]),
    "observable_triangle": ("Observable triangle bound", GROUP_COLORS[GROUPS[1]]),
    "candidate_reference": ("Candidate-reference bound", GROUP_COLORS[GROUPS[0]]),
}
_METRICS = {
    "gap": ("Branch gap", "#2878b5"),
    "conditional": ("Conditional error", "#e67d24"),
    "unconditional_reference": ("Unconditional reference error", "#3b9955"),
    "candidate_radius_tail": ("Candidate radius term", "#bf4c58"),
    "independent_residual": ("Independent vector residual", "#2878b5"),
    "source_precision_envelope": ("Source-precision envelope", "#e67d24"),
}
_DIRECT_METRICS = {
    "reference": (r"$\|\bar{\mathbf{x}}_t^K-\boldsymbol{\mu}_K\|/\sqrt{d}$", "#2878b5", "-"),
    "learned": (r"$\|\hat{\mathbf{x}}_t-\boldsymbol{\mu}_K\|/\sqrt{d}$", "#e67d24", "--"),
    "reference_error": (r"$e_t^K(\varnothing)/\sqrt{d}$", "#3b9955", ":"),
}
_MARKERS = {
    "observed": ("o", "Observed"),
    "independently_checked": ("o", "Independent"),
    "constructed": ("^", "Constructed"),
    "numerically_unresolved": ("x", "Unresolved"),
    "inapplicable": ("+", "Inapplicable"),
}


def _number(frame, name):
    if name not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _finite_limits(*arrays, include_zero=False):
    values = np.concatenate(
        [np.asarray(value, dtype=float).ravel() for value in arrays]
    )
    values = values[np.isfinite(values)]
    if include_zero:
        values = np.r_[values, 0.0]
    if not len(values):
        raise TheoryError("No finite saved values for an available paper figure")
    low, high = float(values.min()), float(values.max())
    padding = 0.045 * (high - low) if high > low else max(abs(high) * 0.05, 0.05)
    return [low - padding, high + padding]


def _compact_ticks(axis):
    low, high = axis.get_view_interval()
    if max(abs(low), abs(high)) >= 1e4:
        axis.set_major_formatter(
            FuncFormatter(
                lambda value, _pos: (
                    f"{value:.2g}".replace("e+0", "e")
                    .replace("e+", "e")
                    .replace("e-0", "e-")
                )
            )
        )


def _legend(ax, handles=None, *, strip=False, loc=None):
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        labels = [handle.get_label() for handle in handles]
    seen, unique = set(), []
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in seen:
            unique.append((handle, label))
            seen.add(label)
    if not unique:
        return
    items, names = zip(*unique)
    options = dict(
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.0,
        handletextpad=0.5,
        borderaxespad=0.2,
        labelspacing=0.35,
        columnspacing=1.2,
    )
    if loc is not None:
        options["borderaxespad"] = 0.6
        ax.legend(items, names, loc=loc, ncol=1, **options)
    elif strip or len(unique) > 4:
        ax.legend(
            items,
            names,
            loc="lower left",
            bbox_to_anchor=(0, 1.015, 1, 0),
            ncol=2,
            mode="expand",
            **options,
        )
    else:
        ax.legend(items, names, loc="best", **options)


def _synchronization_legend(ax, outcome_handles, branch_handles):
    """Keep outcome and branch/reference keys in separate columns inside the axes."""
    options = dict(
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.0,
        handletextpad=0.5,
        borderaxespad=0.6,
        labelspacing=0.35,
    )
    if outcome_handles:
        outcome_legend = ax.legend(handles=outcome_handles, loc="upper left", **options)
        ax.add_artist(outcome_legend)
    ax.legend(handles=branch_handles, loc="upper right", **options)


def _snr_reference(ax, frame, metadata, *, analytical=False):
    value = metadata.get("actual_initial_snr", metadata.get("initial_snr"))
    if value is None and not analytical and "step_index" in frame and len(frame):
        steps = _number(frame, "step_index")
        if np.isfinite(steps).any():
            candidates = _number(frame, "snr")[steps == np.nanmin(steps)]
            candidates = candidates[np.isfinite(candidates)]
            if len(candidates):
                value = float(candidates[0])
    if value is not None and math.isfinite(float(value)) and float(value) > 0:
        ax.axvline(
            float(value), color=".45", linestyle=":", linewidth=0.9, label=r"$\mathrm{SNR}_T$"
        )
        return float(value)
    return None


def _curve_groups(frame):
    keys = [key for key in ("group", "branch", "metric", "segment_id") if key in frame]
    if not keys:
        return [((), frame)]
    return list(frame.groupby(keys, sort=False, dropna=False))


def _identity(series, frame):
    keys = [key for key in ("group", "branch", "metric", "segment_id") if key in frame]
    values = series if isinstance(series, tuple) else (series,)
    return dict(zip(keys, values))


def _draw_scatter(fig, ax, entry, frame, metadata, config):
    styled = paper_style.is_selected(entry)
    dense = styled and entry["stem"] == "posterior_feedback_condition_margin"
    if dense:
        # Local display randomness neither consumes the experiment RNG nor uses
        # SSCD/sign/status to order points. Every associated row field moves together.
        order = np.random.default_rng(paper_style.DISPLAY_ORDER_SEED).permutation(len(frame))
        frame = frame.iloc[order]
    x, y = _number(frame, "x"), _number(frame, "y")
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise TheoryError("Available scatter has no finite coordinate pairs")
    classes = frame.get("marker_class", pd.Series("observed", index=frame.index)).fillna("numerically_unresolved").astype(str).to_numpy()
    styles = _MARKERS
    value_only = entry.get("display_policy") == "finite_saved_values"
    saved_classes = classes.copy()
    if value_only:
        # Saved coordinates are the observation. Audit classifications neither
        # gate inclusion nor determine marker shape for this presentation.
        classes = np.full(len(frame), "values", dtype=object)
        styles = {"values": ("o", "_nolegend_")}
    neutral_identity = entry["stem"] == "lemma4_matched_displacement"
    if neutral_identity:
        column = "direct_lemma4_verification_source"
        if column not in frame:
            raise TheoryError("Matched displacement requires actual saved verification provenance")
        methods = {
            "independent_deterministic_affine_counterfactual": ("o", "Independent affine update"),
            "independently_saved_additive_noise": ("s", "Independent saved noise"),
            "independently_saved_rng_replay": ("D", "Independent RNG replay"),
            "constructed_shared_innovation_from_saved_endpoint_not_independent": ("^", "Constructed innovation"),
            "inapplicable_nonaffine_nonpositive_kappa_or_negative_guidance": ("+", "Inapplicable"),
        }
        provenance = frame[column].fillna("missing_provenance").astype(str).to_numpy()
        unknown = set(provenance[finite]) - set(methods)
        if unknown:
            raise TheoryError(f"Unknown matched-update provenance: {sorted(unknown)}")
        classes = np.where(finite, provenance, classes)
        styles = {**_MARKERS, **methods}
    unknown = set(classes) - set(styles)
    if unknown:
        raise TheoryError(f"Unknown saved direct marker classes: {sorted(unknown)}")
    colors = _number(frame, "terminal_sscd")
    colored = finite & np.isfinite(colors) & bool(entry.get("sscd")) & (not neutral_identity)
    visible_classes = set(classes[finite])
    # The pooled condition scatter has an explicit opacity; all other
    # scatter plots share the publication default.
    scatter_alpha = SCATTER_ALPHA
    if not entry.get("parent_figure"):
        scatter_alpha = entry.get("pooled_scatter_alpha", SCATTER_ALPHA)
    if dense:
        scatter_alpha = paper_style.DENSE_ALPHA
    for marker_class, (marker, label) in styles.items():
        selected = finite & (classes == marker_class)
        if not selected.any():
            continue
        options = dict(s=paper_style.DENSE_MARKER_SIZE if dense else SCATTER_SIZE,
                       alpha=scatter_alpha, marker=marker, rasterized=True)
        if styled:
            options["zorder"] = 3
        if marker not in {"x", "+"}:
            options["edgecolors"] = "none"
        show_label = label if len(visible_classes) > 1 or marker_class != "observed" else "_nolegend_"
        if (selected & colored).any():
            mask = selected & colored
            ax.scatter(x[mask], y[mask], c=colors[mask], cmap=paper_style.SSCD_CMAP if styled else "viridis",
                       norm=paper_style.SSCD_NORM if styled else Normalize(*SSCD_COLOR_RANGE, clip=True),
                       label=show_label, **options)
            show_label = "_nolegend_"
        if (selected & ~colored).any():
            mask = selected & ~colored
            ax.scatter(x[mask], y[mask], color=paper_style.COLORS["missing"] if styled else ".35" if neutral_identity else ".5" if entry.get("sscd") else GROUP_COLORS[GROUPS[1]], label=show_label, **options)
    if entry.get("sscd") and not neutral_identity:
        (paper_style.add_theory_sscd_colorbar if styled else add_sscd_colorbar)(fig, ax, colors[colored])
    if entry.get("equal"):
        limits = _finite_limits(
            x[finite], y[finite], include_zero=entry.get("include_zero", entry["stem"] == "initial_recovery")
        )
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        clean = metadata.get("clean_update_prerequisite", {})
        eligible_diagonal = not entry.get("direct_statement") or "applicable" not in frame or frame.applicable.fillna(False).any()
        if eligible_diagonal and (not entry.get("terminal_applicability") or clean.get("applicable", 0) > 0):
            ax.plot(limits, limits, color=".45", linestyle="--", linewidth=1, label=entry["reference"])
    else:
        ax.set_xlim(_finite_limits(x[finite], include_zero=entry.get("include_zero", False)))
        ax.set_ylim(_finite_limits(y[finite], include_zero=entry.get("include_zero", False)))
    if value_only:
        # Include the condition boundary without clipping any saved point, even
        # when every margin at an individual timestep has the same sign.
        lower, upper = min(float(x[finite].min()), 0.), max(float(x[finite].max()), 0.)
        span = upper - lower
        padding = .06 * span if span > 0 else .05
        ax.set_xlim(lower - padding, upper + padding)
        ax.set_ylim(_finite_limits(y[finite], include_zero=True))
    if entry.get("zero_guides"):
        guide = dict(color=paper_style.COLORS["reference"] if styled else ".65",
                     linewidth=.85 if styled else .7, zorder=2)
        ax.axhline(0, label="Zero gain" if entry.get("direct_statement") else "_nolegend_", **guide)
        ax.axvline(0, label="Zero condition margin" if entry.get("direct_statement") else "_nolegend_", **guide)
    if entry.get("signed"):
        threshold = float(metadata.get("symlog_linthresh", 0.001))
        if not math.isfinite(threshold) or threshold <= 0:
            raise TheoryError("Invalid saved symlog threshold")
        ax.set_yscale("symlog", linthresh=threshold)
    if entry.get("terminal_applicability"):
        clean = metadata.get("clean_update_prerequisite")
        if not isinstance(clean, dict) or not {"applicable", "total"}.issubset(clean):
            raise TheoryError("Theorem 7 requires saved clean-update applicability counts")
        ax.text(.03, .97, f"{clean['applicable']}/{clean['total']} satisfy the\nclean-update prerequisite", transform=ax.transAxes, va="top", fontsize=9)
    if entry.get("frequency_intervals"):
        lower, upper = _number(frame, "frequency_low"), _number(frame, "frequency_high")
        valid_intervals = finite & np.isfinite(lower) & np.isfinite(upper)
        if np.any(lower[valid_intervals] > y[valid_intervals]) or np.any(upper[valid_intervals] < y[valid_intervals]):
            raise TheoryError("Saved frequency intervals do not contain their estimates")
        ax.errorbar(x[valid_intervals], y[valid_intervals], yerr=[y[valid_intervals] - lower[valid_intervals], upper[valid_intervals] - y[valid_intervals]], fmt="none", ecolor=".55", alpha=.45, linewidth=.7)
    if entry.get("saved_trend"):
        trend = metadata.get("trend", {})
        slope, intercept = trend.get("slope"), trend.get("intercept")
        if (
            slope is not None
            and intercept is not None
            and math.isfinite(float(slope))
            and math.isfinite(float(intercept))
        ):
            limits = np.asarray(ax.get_xlim())
            ax.plot(
                limits,
                float(intercept) + float(slope) * limits,
                color=".2",
                linewidth=1.2,
                label="Saved descriptive trend",
            )
    if entry.get("independent_tolerance"):
        tolerance = metadata.get("independent_tolerance_rmse")
        guidance = config["scientific_config"]["guidance_scale"]
        if tolerance is not None and guidance >= 1:
            limits = np.asarray(ax.get_xlim())
            ax.plot(
                limits,
                float(tolerance) - limits,
                color=".4",
                linestyle=":",
                linewidth=1,
                label="Supplied error tolerance",
            )
    for axis in (ax.xaxis, ax.yaxis):
        if axis is ax.yaxis and entry.get("signed"):
            continue
        axis.set_major_locator(MaxNLocator(nbins=5))
        _compact_ticks(axis)
    if not value_only:
        _legend(ax, loc="lower right" if entry.get("terminal_applicability") or neutral_identity else None)
    return {
        **({"display_policy": "finite_saved_values", "legend_placement": "none",
            "marker": "o", "sign_classification_used_for_display": False,
            "zero_padding_policy": "6% of the finite zero-inclusive margin span on both sides; no fabricated observations",
            "zero_guide_labels": "caption_only",
            "saved_marker_counts": {name: int((finite & (saved_classes == name)).sum())
                                    for name in sorted(set(saved_classes[finite]))}}
           if value_only else {}),
        "finite_pairs": int(finite.sum()),
        "scatter_alpha": float(scatter_alpha),
        **({"display_order": "local fixed permutation of saved rows, independent of SSCD, sign and status",
            "display_order_seed": paper_style.DISPLAY_ORDER_SEED,
            "scatter_size": paper_style.DENSE_MARKER_SIZE,
            "rasterization": f"dense scatter artist only; PDF {paper_style.DENSE_PDF_DPI:g} DPI; text, axes and guides remain vector; the opaque colorbar is rasterized without cell seams",
            "color_value": "saved sample-level terminal SSCD"} if dense else {}),
        "missing_coordinate_pairs": int((~finite).sum()),
        "marker_counts": {name: int((finite & (classes == name)).sum()) for name in sorted(visible_classes)},
        "symlog_linthresh": float(metadata.get("symlog_linthresh", 0.001)) if entry.get("signed") else None,
        "all_data_ranges": {
            "x": [float(x[finite].min()), float(x[finite].max())],
            "y": [float(y[finite].min()), float(y[finite].max())],
        },
    }


def _draw_ecdf(ax, entry, frame, metadata):
    keys = [key for key in ("group", "branch", "distribution") if key in frame]
    grouped = (
        list(frame.groupby(keys, sort=False, dropna=False)) if keys else [((), frame)]
    )
    present_groups, branches, observed = set(), set(), 0
    for identities, rows in grouped:
        identities = identities if isinstance(identities, tuple) else (identities,)
        identity = dict(zip(keys, identities))
        values, cdf = _number(rows, "value"), _number(rows, "cdf")
        valid = np.isfinite(values) & np.isfinite(cdf)
        if not valid.any():
            continue
        values, cdf = values[valid], cdf[valid]
        order = np.argsort(values, kind="stable")
        values, cdf = values[order], cdf[order]
        if np.any(np.diff(cdf) < -1e-12) or np.any((cdf < -1e-12) | (cdf > 1 + 1e-12)):
            raise TheoryError("Invalid saved empirical-CDF support")
        group, branch = identity.get("group"), identity.get("branch")
        distribution = identity.get("distribution")
        if distribution:
            label, color = _DISTRIBUTIONS.get(
                str(distribution), (str(distribution).replace("_", " "), "#343434")
            )
            if entry["stem"] in {"initial_unconditional_mean_concentration", "initial_unconditional_mean_concentration_zero"}:
                label = INITIAL_DISTRIBUTION_LABELS.get(str(distribution), label)
        else:
            label, color = GROUP_LABELS.get(str(group), str(group or "All")), GROUP_COLORS.get(str(group), "#343434")
            if group:
                present_groups.add(str(group))
            if branch:
                branches.add(str(branch))
        style = "--" if branch == "unconditional" else "-"
        ax.step(
            np.r_[values[0], values],
            np.r_[0.0, cdf],
            where="post",
            color=color,
            linestyle=style,
            linewidth=1.4,
            label=label if not branches else "_nolegend_",
        )
        observed += len(values)
    if not observed:
        raise TheoryError("Available ECDF has no finite saved support")
    ax.set_xlim(_finite_limits(_number(frame, "value")))
    ax.set_ylim(0, 1.01)
    if entry.get("reference_value") is not None:
        ax.axvline(
            entry["reference_value"],
            color=".45",
            linestyle=":",
            linewidth=0.9,
            label=entry["reference"],
        )
    if branches:
        handles = [
            Line2D([], [], color=GROUP_COLORS[group], label=GROUP_LABELS[group], linewidth=1.5)
            for group in GROUPS
            if group in present_groups
        ]
        handles += [
            Line2D(
                [],
                [],
                color=".25",
                linestyle="--" if branch == "unconditional" else "-",
                label=branch.title(),
                linewidth=1.5,
            )
            for branch in ("conditional", "unconditional")
            if branch in branches
        ]
        if entry["stem"] == "initial_target_retrieval_rank":
            _legend(ax, handles, loc="lower right")
        else:
            _legend(ax, handles, strip=True)
    else:
        _legend(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    _compact_ticks(ax.xaxis)
    return {
        "finite_support_rows": observed,
        "all_data_ranges": {
            "x": [
                float(np.nanmin(_number(frame, "value"))),
                float(np.nanmax(_number(frame, "value"))),
            ]
        },
    }


def _draw_curves(ax, entry, frame, metadata):
    metric_diagnostic = entry["stem"] in {
        "synchronization_bound_components",
        "matched_update_residual",
    }
    if metric_diagnostic and "group" in frame:
        frame = frame.loc[frame.group.astype(str).str.lower() == "all"]
        if frame.empty:
            raise TheoryError(
                "Metric diagnostic requires its saved all-population curves"
            )
    x_column = (
        "step_index"
        if entry.get("update_index")
        else "analytical_snr"
        if entry.get("analytical_snr")
                else "s"
                if entry["kind"] == "four_response"
        else "lambda"
        if entry["kind"] == "dose"
        else "snr"
    )
    if x_column not in frame and entry.get("analytical_snr") and "snr" in frame:
        x_column = "snr"
    present_groups, branches, metric_names, handles = set(), set(), set(), []
    observed = 0
    for identities, rows in _curve_groups(frame):
        identity = _identity(identities, frame)
        group, branch, metric = (
            identity.get("group"),
            identity.get("branch"),
            identity.get("metric"),
        )
        if (
            group is not None
            and str(group) not in GROUPS
            and str(group).lower() != "all"
        ):
            raise TheoryError(f"Unrecognized saved outcome group {group}")
        if str(group).lower() == "all" and any(
            frame.get("group", pd.Series(dtype=str)).isin(GROUPS)
        ):
            continue
        x, y = _number(rows, x_column), _number(rows, "median")
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        if not np.isfinite(x).any() or not np.isfinite(y).any():
            continue
        color = GROUP_COLORS.get(str(group), "#563580")
        label = GROUP_LABELS.get(str(group), str(group or "Analytical reference"))
        if metric is not None and metric_diagnostic:
            metric = branch if str(branch) in _METRICS else metric
            label, color = _METRICS.get(
                str(metric), (str(metric).replace("_", " "), color)
            )
            metric_names.add(str(metric))
        line_style = "--" if branch == "unconditional" else "-"
        if entry.get("direct_metrics"):
            if str(metric) not in _DIRECT_METRICS:
                raise TheoryError(f"Unknown direct Gaussian metric: {metric}")
            label, color, line_style = _DIRECT_METRICS[str(metric)]
            metric_names.add(str(metric))
        if (
            branch is None
            and entry.get("branches")
            and str(metric) in {"conditional", "unconditional"}
        ):
            branch = str(metric)
        if group is not None:
            present_groups.add(str(group))
        if branch is not None and entry.get("branches"):
            branches.add(str(branch))
        (line,) = ax.plot(
            x,
            y,
            color=color,
            linestyle=line_style,
            linewidth=1.4,
            marker="." if "segment_id" in rows and len(rows) == 1 else None,
            markersize=3,
            label=label,
        )
        handles.append(line)
        if "q25" in rows and "q75" in rows:
            low, high = _number(rows, "q25")[order], _number(rows, "q75")[order]
            ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        observed += int((np.isfinite(x) & np.isfinite(y)).sum())
    if not observed:
        raise TheoryError("Available curve has no finite saved summaries")
    extrema = [
        _number(frame, name)
        for name in ("median", "q25", "q75", "minimum", "maximum")
        if name in frame
    ]
    ax.set_ylim(_finite_limits(*extrema, include_zero=entry.get("signed", False) or entry.get("include_zero", False)))
    if entry.get("include_zero") and not entry.get("signed"):
        ax.set_ylim(bottom=0)
    if entry.get("signed"):
        threshold = float(
            metadata.get("symlog_linthresh", metadata.get("linthresh", 0.001))
        )
        ax.set_yscale("symlog", linthresh=threshold)
        ax.axhline(0, color=".5", linewidth=0.8)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    initial = None
    if x_column in {"snr", "analytical_snr"}:
        valid = _number(frame, x_column)
        valid = valid[np.isfinite(valid)]
        if np.any(valid <= 0):
            raise TheoryError(
                "A saved logarithmic SNR axis contains a nonpositive value"
            )
        ax.set_xscale("log")
        initial = _snr_reference(
            ax, frame, metadata, analytical=entry.get("analytical_snr", False)
        )
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if entry["kind"] == "dose":
        guidance = float(metadata.get("guidance_scale", metadata.get("g", 1)))
        for value, label, style in [
            (0, r"Unconditional ($s=0$)", ":"),
            (1 / guidance, r"Conditional ($s=1/g$)", "--"),
            (1, r"Guided ($s=1$)", ":"),
        ]:
            ax.axvline(value, color=".45", linestyle=style, linewidth=0.8, label=label)
    if branches:
        handles = [
            Line2D([], [], color=GROUP_COLORS[group], label=GROUP_LABELS[group], linewidth=1.5)
            for group in GROUPS
            if group in present_groups
        ]
        outcome_count = len(handles)
        handles += [
            Line2D(
                [],
                [],
                color=".25",
                linestyle="--" if branch == "unconditional" else "-",
                label=branch.title(),
                linewidth=1.5,
            )
            for branch in ("conditional", "unconditional")
            if branch in branches
        ]
        if initial is not None:
            handles.append(
                Line2D([], [], color=".45", linestyle=":", label=r"$\mathrm{SNR}_T$")
            )
        if entry["stem"] in {"target_synchronization", "lemma6_branch_target_errors"}:
            _synchronization_legend(ax, handles[:outcome_count], handles[outcome_count:])
        else:
            _legend(ax, handles, strip=True)
    else:
        _legend(ax, strip=(len(metric_names) > 2 and not entry.get("direct_metrics")) or entry["kind"] == "dose")
    return {
        "finite_summary_cells": observed,
        "population": "unique Gaussian seeds" if entry.get("direct_metrics") else "all" if metric_diagnostic else "saved outcome cohorts",
        "initial_snr_reference": initial,
        "x_column": x_column,
        "symlog_linthresh": float(
            metadata.get("symlog_linthresh", metadata.get("linthresh", 0.001))
        )
        if entry.get("signed")
        else None,
    }


def _draw_fraction(ax, frame, metadata):
    observed = 0
    for group in GROUPS:
        rows = frame.loc[frame.group.astype(str) == group].sort_values(
            "snr", kind="stable"
        )
        x, y, upper = (
            _number(rows, "snr"),
            _number(rows, "fraction"),
            _number(rows, "upper_fraction"),
        )
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        if (
            np.any((y[finite] < -1e-12) | (y[finite] > 1 + 1e-12))
            or not np.isfinite(upper[finite]).all()
            or np.any(x[finite] <= 0)
            or np.any(upper[finite] < y[finite] - 1e-12)
            or np.any(upper[finite] > 1 + 1e-12)
        ):
            raise TheoryError(
                "Invalid saved same-denominator numerical-ambiguity envelope"
            )
        ax.plot(x, y, color=GROUP_COLORS[group], linewidth=1.5, label=GROUP_LABELS[group])
        ax.fill_between(x, y, upper, color=GROUP_COLORS[group], alpha=0.16, linewidth=0)
        observed += int(finite.sum())
    if not observed:
        raise TheoryError("Available fraction plot has no finite saved summaries")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.025)
    initial = _snr_reference(ax, frame, metadata)
    _legend(ax)
    return {
        "finite_summary_cells": observed,
        "initial_snr_reference": initial,
        "band": "Saved unresolved numerical-sign/classification mass on the same eligible denominator.",
    }


def _draw_coverage(ax, frame, metadata):
    if "group" in frame:
        frame = frame.loc[frame.group.astype(str).str.lower() == "all"]
    if frame.empty:
        raise TheoryError("Coverage diagnostic requires its saved All population")
    margin_column = "margin" if "margin" in frame else "metric"
    colors = ("#2878b5", "#e67d24", "#3b9955", "#bf4c58")
    handles = []
    observed = 0
    for index, (margin, rows) in enumerate(frame.groupby(margin_column, sort=False)):
        rows = rows.sort_values("snr", kind="stable")
        x = _number(rows, "snr")
        color = colors[index % len(colors)]
        label = {
            "original": "Original margin",
            "combined": "Combined error",
            "signed_error": "Signed error",
            "projected_variation": "Projected variation",
        }.get(str(margin), str(margin))
        resolved = _number(rows, "resolved_positive_fraction")
        estimated = _number(rows, "estimated_strict_positive_fraction")
        ax.plot(x, resolved, color=color, linewidth=1.3)
        ax.plot(x, estimated, color=color, linewidth=1, linestyle=":")
        handles.append(Line2D([], [], color=color, label=label))
        observed += int(np.isfinite(resolved).sum() + np.isfinite(estimated).sum())
    rows = (
        frame.drop_duplicates("step_index")
        if "step_index" in frame
        else frame.drop_duplicates("snr")
    )
    rows = rows.sort_values("snr", kind="stable")
    ax.plot(
        _number(rows, "snr"),
        _number(rows, "stable_gain_positive_fraction"),
        color=".25",
        linestyle="--",
        label="Observed positive gain",
    )
    handles += [
        Line2D([], [], color=".25", linestyle="--", label="Observed positive gain"),
        Line2D([], [], color=".45", linestyle=":", label="Estimated strict signs"),
    ]
    ax.set_xscale("log")
    ax.set_ylim(
        _finite_limits(
            _number(frame, "resolved_positive_fraction"),
            _number(frame, "estimated_strict_positive_fraction"),
            _number(frame, "stable_gain_positive_fraction"),
            include_zero=True,
        )
    )
    initial = _snr_reference(ax, frame, metadata)
    if initial is not None:
        handles.append(Line2D([], [], color=".45", linestyle=":", label=r"$\mathrm{SNR}_T$"))
    _legend(ax, handles, strip=True)
    return {"finite_summary_cells": observed, "estimated_and_resolved_distinct": True}


def _apply_saved_limits(ax, metadata, frame, entry):
    if entry.get("display_policy") == "finite_saved_values":
        # Combined and timestep views each retain every finite coordinate and
        # both zero guides; historical pooled limits are not slice limits.
        return
    if (entry["kind"] in {"four_terminal_grouped_cdf", "four_guidance_fit"}
            or (entry["kind"] == "four_prompt_chronological"
                and entry.get("prediction_domain") == "positive_noise_transitions")):
        # These renderers own complete support and reference/transition domains;
        # historical limits must not crop data or append a terminal sentinel.
        return
    for axis in ("x", "y"):
        if axis == "y" and entry.get("display_range_policy"):
            continue
        if axis == "x" and entry.get("plot_recipe_version") == PLOT_RECIPE_VERSION and entry["stem"] == "posterior_feedback_condition_margin":
            continue
        values = metadata.get(axis + "_limits")
        if values is None:
            continue
        if (
            len(values) != 2
            or not all(math.isfinite(float(v)) for v in values)
            or values[0] >= values[1]
        ):
            raise TheoryError("Invalid saved paper axis limits")
        if axis == "x":
            column = (
                "x"
                if entry["kind"] in {"scatter", "pair_loss", "injection_geometry", "four_terminal_scatter", "four_counterfactual"}
                else "value"
                if entry["kind"] in {"ecdf", "terminal_cdf", "terminal_components"}
                else "step_index"
                if entry.get("update_index")
                else "analytical_snr"
                if entry.get("analytical_snr")
                else "s"
                if entry["kind"] == "four_response"
                else "lambda"
                if entry["kind"] == "dose"
                else "snr"
            )
            columns = [column] if column in frame else []
            if entry["kind"] == "pair_loss":
                columns += [name for name in ("x_low", "x_high") if name in frame]
        else:
            columns = [
                name
                for name in (
                    "y",
                    "control_y",
                    "y_low",
                    "y_high",
                    "cdf",
                    "median",
                    "mean",
                    "q25",
                    "q75",
                    "minimum",
                    "maximum",
                    "fraction",
                    "upper_fraction",
                    "resolved_fraction",
                    "unknown_fraction",
                )
                if name in frame
            ]
        for column in columns:
            data = _number(frame, column)
            data = data[np.isfinite(data)]
            if axis == "x" and entry["kind"] in {"terminal_cdf", "terminal_components"}:
                data = data[data > 0]
            if len(data) and (
                data.min() < float(values[0]) - 1e-12
                or data.max() > float(values[1]) + 1e-12
            ):
                raise TheoryError("Saved paper axis limits would crop finite data")
        getattr(ax, "set_" + axis + "lim")(values)


def _curated_feedback_legend(ax, metadata, stats, frame):
    groups = [group for group in GROUPS if group in set(frame.group.astype(str))]
    for artist in list(ax.get_children()):
        if isinstance(artist, Legend):
            artist.remove()
    handles = [Line2D([], [], color=GROUP_COLORS[group], label=GROUP_LABELS[group]) for group in groups]
    condition_label = FEEDBACK_CONDITION_LABEL
    if stats.get("shared_zero_baseline"):
        positive, unresolved = metadata.get("condition_positive_counts", {}), metadata.get("condition_unresolved_counts", {})
        if any(group not in positive or group not in unresolved for group in GROUPS):
            raise TheoryError("Shared zero-condition display requires saved per-group positive and unresolved sample-transition counts")
        if any(int(positive[group]) != 0 for group in GROUPS):
            raise TheoryError("Saved positive-condition counts contradict the shared zero curve")
        counts = {}
        for group in groups:
            rows = frame.loc[frame.group.eq(group) & frame.metric.eq("condition")]
            total = _number(rows, "eligible_count")
            if not np.isfinite(total).all() or np.any(total < 0) or np.any(total != np.floor(total)):
                raise TheoryError("Shared zero-condition display requires saved integer eligible sample-transition counts")
            counts[group] = {"positive": int(positive[group]), "unresolved": int(unresolved[group]), "total": int(total.sum())}
            if counts[group]["unresolved"] > counts[group]["total"]:
                raise TheoryError("Unresolved condition counts exceed their saved population")
            for column in ("negative_count", "zero_count", "unavailable_count"):
                if column in rows:
                    counts[group][column] = int(_number(rows, column).sum())
        high, lower = counts[GROUPS[0]], counts[GROUPS[1]]
        condition_label = (
            FEEDBACK_CONDITION_LABEL + "\n" + r"$\Pr(\cdot)=0$ (both groups)"
            + "\n" + GROUP_LABELS[GROUPS[0]] + ": 0/" + str(high["total"])
            + "; unresolved " + str(high["unresolved"]) + "\n"
            + GROUP_LABELS[GROUPS[1]] + ": 0/" + str(lower["total"])
            + "; unresolved " + str(lower["unresolved"])
        )
        stats["shared_zero_group_counts"] = counts
        stats["shared_zero_count_unit"] = "Saved eligible sample-transitions, summed across native labels; not unique seeds"
    handles += [Line2D([], [], color=".25", linestyle="-", label=FEEDBACK_GAIN_LABEL),
                Line2D([], [], color=".25", linestyle="--", label=condition_label),
                Line2D([], [], color=".6", linewidth=6, alpha=.35, label="Unresolved feedback"),
                Line2D([], [], color=".45", linestyle=":", label="Condition upper bound"),
                Line2D([], [], color=".45", linestyle=":", label=r"$\mathrm{SNR}_T$")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2,
              frameon=False, fontsize=LEGEND_FONT_SIZE, handlelength=1.5, columnspacing=.7,
              handletextpad=.4, borderaxespad=0, labelspacing=.25)
    stats["legend_placement"] = "tight external top; complete time range remains unobscured"


def _active_presentation(ax, entry, metadata, stats, frame):
    """Keep legacy diagnostic artwork unchanged; active cards use captions."""
    if entry["kind"] == "four_terminal_grouped_cdf":
        # The grouped renderer owns its factorized legend and has no subtitle
        # or small annotations; never replace it with the legacy pooled key.
        ax.set_title("")
        stats["annotations_saved_in_caption"] = True
        return
    if entry.get("formula_version") != "four-stage-figures-1":
        return
    for artist in list(ax.texts):
        if artist.get_gid() != "curation-required-label":
            artist.remove()
    ax.set_title("")
    if entry["stem"] == "terminal_bound_coverage":
        names = ("actual", "observable", "reference")
        labels = tuple(TERMINAL_LABELS[name] for name in names)
        for line, label in zip(ax.lines[:3], labels):
            line.set_label(label)
        colors, styles = (".2", "#e5ba26", "#563580"), ("-", "--", "-.")
        zero = metadata["zero_mass"]
        infinity = metadata.get("infinite_mass", {name: 0. for name in names})
        show_mass = any(float(zero[name]) > 0 or float(infinity[name]) > 0 for name in names)
        handles = []
        for name, label, color, style in zip(names, labels, colors, styles):
            if show_mass:
                label += f"\nZero: {float(zero[name]):.3g}; +∞: {float(infinity[name]):.3g}"
            handles.append(Line2D([], [], color=color, linestyle=style, label=label))
        tolerance = metadata.get("predeclared_tolerance_rmse")
        if tolerance is not None:
            handles.append(Line2D([], [], color=".5", linestyle=":", label=r"$\tau=0$" if float(tolerance) == 0 else r"$\tau/\sqrt{d}$"))
        scope = {
            "original_clean_terminal_theorem": "Clean-update theorem",
            "finite_terminal_update_extension_deterministic": "Finite-step extension",
            "finite_terminal_update_extension_gaussian_noise_bound": "Finite-step extension\n(Gaussian probability bound)",
        }[metadata["terminal_scope"]]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.01),
                  bbox_transform=ax.transAxes, borderaxespad=0, frameon=False,
                  fontsize=LEGEND_FONT_SIZE, handlelength=1.8, labelspacing=.35)
        stats["legend_placement"] = "outside top"
        stats["terminal_scope_label"] = scope
        stats["terminal_scope_display"] = "caption_only"
        stats["legend_quantities"] = dict(zip(names, labels))
        stats["mass_disclosure"] = "Readable per-curve legend entries and exact caption metadata"
        stats["zero_mass_annotation_visible"] = False
        stats["zero_mass_legend_visible"] = show_mass
        stats["infinite_mass_legend_visible"] = any(float(infinity[name]) > 0 for name in names)
    if entry.get("plot_recipe_version") == PLOT_RECIPE_VERSION:
        if entry["stem"] == "initial_loss_recovery":
            legend = ax.get_legend()
            if legend is not None:
                names = {"Conditional": INITIAL_BRANCH_LABELS[0],
                         "Unconditional control": INITIAL_BRANCH_LABELS[1],
                         "Monte Carlo bootstrap intervals": "Bootstrap CI"}
                for label in legend.get_texts():
                    label.set_text(names.get(label.get_text(), label.get_text()))
            for axis in ax.figure.axes:
                if axis.get_label() == "<colorbar>":
                    axis.set_ylabel("SSCD")
            stats["colorbar_label"] = "SSCD"
            stats["color_population"] = "Pair mean terminal SSCD; aggregation remains defined in the caption"
        elif entry["stem"] == "posterior_feedback_over_time":
            _curated_feedback_legend(ax, metadata, stats, frame)
        elif entry["stem"] == "posterior_feedback_condition_margin":
            # Limits already include every plotted pair; include zero and fixed
            # padding on both sides without inventing any point or sign count.
            lower, upper = ax.get_xlim()
            lower, upper = min(float(lower), 0.), max(float(upper), 0.)
            span = upper - lower
            padding = .06 * span if span > 0 else .05
            ax.set_xlim(lower - padding, upper + padding)
            # Keep both zero guides; their meaning is described in the caption.
            for line in ax.lines:
                if line.get_label() in {"Zero gain", "Zero condition margin"}:
                    line.set_label("_nolegend_")
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            if handles:
                ax.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(.5, 1.01),
                          ncol=2, frameon=False, fontsize=LEGEND_FONT_SIZE, handlelength=1.5,
                          columnspacing=.7, handletextpad=.4, borderaxespad=0, labelspacing=.25)
            stats["zero_padding_policy"] = "6% of the full finite zero-inclusive displayed span on both sides; empty space is not an observation"
            stats["legend_placement"] = "tight external top" if handles else "none"
            stats["zero_guide_labels"] = "caption_only"
    if entry["stem"] in {"terminal_bound_coverage", "terminal_observable_bound", "final_reproduction_bound"}:
        stats["terminal_labels_abbreviated"] = True
        stats["terminal_correction_display"] = "caption_only; corrected numerical values retained"
        stats["terminal_scope"] = metadata.get("terminal_scope")
    # Split legends include an earlier legend retained via add_artist.
    for legend in (artist for artist in ax.get_children() if isinstance(artist, Legend)):
        for label in legend.get_texts():
            label.set_fontsize(LEGEND_FONT_SIZE)
        legend.get_title().set_fontsize(LEGEND_FONT_SIZE)
    stats["annotations_saved_in_caption"] = True


def _draw(entry, frame, metadata, config):
    # Both direct artist use and publication keep the theme local to these six
    # output routes. Unrelated renderers retain their existing rc settings.
    if paper_style.is_selected(entry):
        with plt.rc_context(paper_style.THEORY_STYLE):
            return _draw_impl(entry, frame, metadata, config)
    return _draw_impl(entry, frame, metadata, config)


def _sparse_signed_ticks(ax):
    """Reduce labels, preserving the complete symmetric-log transform."""
    transform = ax.yaxis.get_transform()
    base = float(transform.base)
    ax.yaxis.get_major_locator().set_params(numticks=7)
    def label(value, _position):
        if value == 0:
            return "$0$"
        exponent = math.log(abs(value), base)
        rounded = round(exponent)
        if not math.isclose(exponent, rounded, rel_tol=0., abs_tol=1e-8):
            return ""
        sign = "-" if value < 0 else ""
        return rf"${sign}{base:g}^{{{rounded}}}$"
    ax.yaxis.set_major_formatter(FuncFormatter(label))
    return {"base": base, "linthresh": float(transform.linthresh),
            "linscale": float(transform.linscale), "labels": "selected base powers and zero"}


def _draw_impl(entry, frame, metadata, config):
    if entry.get("parent_figure"):
        # Timestep companions share the parent's axes and marker styling;
        # parent_figure keeps pooled-only opacity separate from each slice.
        entry = {**entry, "stem": entry["parent_figure"]}
    if entry["stem"] in RETIRED_RENDER_STEMS or entry["kind"] == "four_motion":
        raise TheoryError("Requested figure render is retired; its saved scientific measurements remain available")
    missing = set(entry["required_columns"]) - set(frame)
    if missing:
        raise TheoryError(
            f"{entry['stem']}: compact table misses {sorted(missing)}. Run {recompute_command(config)}"
        )
    if entry.get("requires_theory_mean"):
        science = config.get("scientific_config", config)
        mode = science.get("mean_source", "cached-targets")
        expected_source = {
            "reference-min-snr": "minimum_snr_unconditional_reference_monte_carlo",
            "reference-initial": "initial_unconditional_reference_monte_carlo",
            "cached-targets": "declared_finite_bank_mean",
        }.get(mode)
        if expected_source is None:
            raise TheoryError(f"{entry['stem']}: unknown saved theory-mean source")
        if metadata.get("theory_mean_source") != expected_source or not metadata.get("theory_mean_sha256"):
            raise TheoryError(f"{entry['stem']}: saved theory mean differs from the requested mean estimator. Run {recompute_command(config)}")
        receipt = metadata.get("theory_mean", {})
        if receipt.get("vector_sha256") != metadata["theory_mean_sha256"] or receipt.get("source") != expected_source:
            raise TheoryError(f"{entry['stem']}: missing or inconsistent mean-estimation receipt. Run {recompute_command(config)}")
        if mode in {"reference-min-snr", "reference-initial"}:
            if receipt.get("sample_count") != science.get("num_mean_samples", 10000):
                raise TheoryError(f"{entry['stem']}: mean sample count differs from the saved configuration")
            if receipt.get("mean_seed") != science.get("mean_seed", 0):
                raise TheoryError(f"{entry['stem']}: mean seed differs from the saved configuration")
            level = receipt.get("level", {})
            if mode == "reference-initial":
                if level.get("step_index") != 0 or not receipt.get("estimator_hash"):
                    raise TheoryError(f"{entry['stem']}: missing first-step mean-estimation identity")
            else:
                valid = (receipt.get("estimator_hash") and level.get("source_range") == "analytical"
                         and level.get("grid_index") == 0 and "step_index" in level
                         and level["step_index"] is None and "timestep" in level
                         and level["timestep"] is None)
                try:
                    initial_snr = float(receipt["initial_snr"])
                    decades = float(science.get("reference_snr_decades", 6.))
                    expected_snr = initial_snr * 10. ** -decades
                    valid = (valid and math.isfinite(initial_snr) and initial_snr > 0
                             and 0 < decades <= 12 and math.isfinite(expected_snr) and expected_snr > 0
                             and float(receipt["reference_snr_decades"]) == decades)
                    for value in (level["snr"], receipt["estimation_snr"]):
                        valid = valid and math.isclose(float(value), expected_snr, rel_tol=1e-12, abs_tol=0.)
                    valid = (valid and math.isclose(float(level["alpha"]), math.sqrt(expected_snr / (1 + expected_snr)), rel_tol=1e-12, abs_tol=0.)
                             and math.isclose(float(level["sigma"]), 1 / math.sqrt(1 + expected_snr), rel_tol=1e-12, abs_tol=0.))
                    native_level = receipt["initial_level"]
                    valid = (valid and native_level.get("step_index") == 0
                             and math.isclose(float(native_level["snr"]), initial_snr, rel_tol=1e-12, abs_tol=0.))
                    if metadata.get("actual_initial_snr") is not None:
                        valid = valid and math.isclose(float(metadata["actual_initial_snr"]), initial_snr, rel_tol=1e-12, abs_tol=0.)
                except (KeyError, TypeError, ValueError, OverflowError):
                    valid = False
                if not valid:
                    raise TheoryError(f"{entry['stem']}: minimum-SNR mean-estimation receipt differs from the analytical grid. Run {recompute_command(config)}")
    if entry.get("comparison_centre") == "zero":
        if metadata.get("comparison_centre") != "zero" or metadata.get("declared_law_mean_unchanged") is not True:
            raise TheoryError(f"{entry['stem']}: missing saved zero-baseline measurement contract. Run {recompute_command(config)}")
    fig = plt.Figure()
    ax = fig.subplots()
    styled = paper_style.is_selected(entry)
    try:
        if styled:
            paper_style.apply_layout(fig, ax)
        else:
            ax.set_axisbelow(True)
            ax.grid(True, color=".9", linewidth=0.6)
        kind = entry["kind"]
        if kind in FOUR_STAGE_KINDS:
            stats = draw_four_stage(fig, ax, entry, frame, metadata, config)
        elif kind in EVIDENCE_KINDS:
            stats = draw_evidence(fig, ax, entry, frame, metadata, config)
        elif kind == "scatter":
            stats = _draw_scatter(fig, ax, entry, frame, metadata, config)
        elif kind == "ecdf":
            stats = _draw_ecdf(ax, entry, frame, metadata)
        elif kind == "fraction":
            stats = _draw_fraction(ax, frame, metadata)
        elif kind == "coverage":
            stats = _draw_coverage(ax, frame, metadata)
        else:
            stats = _draw_curves(ax, entry, frame, metadata)
        if entry["stem"] == "initial_loss_recovery":
            # Counts and qualification belong to the saved caption, not tiny text.
            for artist in list(ax.texts):
                artist.remove()
            if ax.get_legend() is not None:
                for label in ax.get_legend().get_texts():
                    label.set_fontsize(paper_style.LEGEND_SIZE if styled else LEGEND_FONT_SIZE)
            if {"control_y_low", "control_y_high"}.issubset(frame):
                x, low, high = (_number(frame, key) for key in ("x", "control_y_low", "control_y_high"))
                valid = np.isfinite(x) & np.isfinite(low) & np.isfinite(high)
                if np.any(valid & ((low < 0) | (low > high))):
                    raise TheoryError("Invalid saved paired unconditional bootstrap interval")
                if len(frame) <= 40:
                    ax.vlines(x[valid], low[valid], high[valid],
                              color=paper_style.COLORS["control"] if styled else ".65",
                              alpha=.35, linewidth=.6, zorder=2 if styled else None)
                if valid.any():
                    ax.set_ylim(0, max(ax.get_ylim()[1], float(high[valid].max()) * 1.06))
            stats["annotations_saved_in_caption"] = True
        _active_presentation(ax, entry, metadata, stats, frame)
        _apply_saved_limits(ax, metadata, frame, entry)
        label_size = paper_style.AXIS_LABEL_SIZE if styled else AXIS_LABEL_FONT_SIZE
        ax.set_xlabel(entry["axes"]["x"], fontsize=label_size)
        ax.set_ylabel(entry["axes"]["y"], fontsize=label_size)
        if styled:
            paper_style.apply_layout(fig, ax)
            stats["style_version"] = paper_style.STYLE_VERSION
            stats["layout"] = dict(paper_style.LAYOUT_RECEIPT)
            if entry["stem"] == "posterior_feedback_condition_margin":
                stats["signed_axis_display"] = _sparse_signed_ticks(ax)
            # Legacy presentation renames branch handles; apply the common
            # typography scale afterward without moving or merging legends.
            for legend in (item for item in ax.get_children() if isinstance(item, Legend)):
                for label in legend.get_texts():
                    label.set_fontsize(paper_style.LEGEND_SIZE)
                legend.get_title().set_fontsize(paper_style.LEGEND_SIZE)
        stats["axis_limits"] = {"x": list(ax.get_xlim()), "y": list(ax.get_ylim())}
        if entry.get("plot_recipe_version") == NOTATION_VERSION:
            stats["notation_version"] = NOTATION_VERSION
            stats["axis_labels"] = entry["axes"].copy()
        return fig, json_value(stats)
    except BaseException:
        plt.close(fig)
        raise


def _caption(entry):
    metadata = json_value(entry["measurement_metadata"])
    text = [
        "## " + str(Path(_output_route(entry.get("requested_outputs", entry["outputs"]))).with_suffix("")),
        "",
        entry["semantic_question"],
        "",
        f"Status: {entry['status']}.",
    ]
    if metadata.get("reason"):
        text.append(str(metadata["reason"]))
    if metadata.get("alias_of"):
        text.append(
            "Alias of "
            + str(metadata["alias_of"])
            + "; no duplicate figure is exported."
        )
    text += [
        "",
        entry["formula"],
        "",
        "Axis notation: " + entry["axis_notation"],
        "Normalization: " + str(entry["normalization"] if entry.get("kind") == "four_guidance_fit"
                                  else metadata.get("normalization", entry["normalization"])),
        "Weighting: " + str(metadata.get("weighting", entry["weighting"])),
        "Bands: "
        + str(
            entry.get("display_band_definition", metadata.get(
                "band_definition",
                metadata.get("band_meaning", entry["band_definition"]),
            ))
        ),
        "Groups: " + entry["group_rule"],
    ]
    if entry.get("style_version"):
        text.append("Appearance: " + entry["style_version"] + "; restricted magma SSCD palette, fixed [0,1] normalization. Higher-SSCD group: coral; lower-SSCD group: indigo. Final manuscript-size, grayscale and color-vision-deficiency inspection remains pending.")
    if entry.get("presentation_note"):
        text.append("Presentation: " + entry["presentation_note"])
    if entry.get("notation_details"):
        text.append("Notation details: " + entry["notation_details"])
    support = list(zip(entry.get("supporting_figure_slots", []), entry.get("supporting_figure_ids", [])))
    if support:
        destinations = {item["stem"]: item["outputs"]["pdf"] for item in paper_registry()}
        text.append("Supporting figures (in order): " + ", ".join(
            f"[{slot}: {stem}]({destinations[stem]})" for slot, stem in support if stem in destinations
        ) + ".")
    if entry.get("direct_statement"):
        for key in ("result_label", "manuscript_label", "x_definition", "y_definition", "input_source", "reference_law", "averaging_measure", "averaging_unit", "applicability_assumptions", "interpretation", "supporting_equation", "auxiliary_definitions", "manuscript_label_status", "source_label_status", "native_sweep_figure", "numerical_resolution_figure"):
            if key in entry:
                text.append(key.replace("_", " ").capitalize() + ": " + str(entry[key]))
    for key in (
        "counts",
        "initial_baseline_summary", "initial_baseline_json", "baseline_status", "baseline_summary",
        "comparison_centre", "declared_law_mean_unchanged", "mean_norm_rmse", "mean_norm_l2",
        "theory_mean", "theory_mean_source", "theory_mean_sha256", "theory_mean_interpretation",
        "bank_mean_norm_l2", "bank_mean_norm_rmse", "mean_offset_l2", "mean_offset_rmse",
        "zero_baseline_summary_table", "metric_definitions", "comparison_scope",
        "aggregation", "cohort_audit_table", "exclusion_reason_counts", "expected_seed_ids", "expected_seed_count", "terminal_output_prediction",
        "color_population", "baseline_uncertainty", "baseline_interpretation", "dose_grid", "y_transform",
        "cohort_policy", "endpoint_status_counts", "endpoint_contracts", "cfg_endpoint_label",
        "missing_outcome_curve_count", "response_status_table", "excluded_response_table", "prediction_steps",
        "prediction_domain", "manuscript_domain", "excluded_terminal_prediction_rows",
        "numerical_scope", "numerical_status_counts", "measurement_audit_table",
        "reference_definition", "missing_measurement_fields",
        "measurement_contract", "condition_formula", "comparison_interpretation",
        "condition_assessment_counts", "condition_display_policy",
        "display_policy", "timestep_slice", "timestep_export",
        "variation_definition", "zero_gap_convention",
        "reference_comparison_formula", "reference_comparison_scope",
        "missing_outcome_row_count", "chronological_schedule", "normalized_progress_definition", "denominator_column",
        "shape_group_counts", "shape_population", "shape_rule", "shape_table",
        "trajectory_peak_summary", "peak_summary_status", "peak_summary_details",
        "peak_descriptive_statistics",
        "prompt_peak_distribution", "prompt_shape_table", "mixed_prompt_shape_table", "motion_audit_table",
        "observed_bound_exceedance_count", "observed_comparison_policy", "inherited_audit_status", "bound_kind", "axis_scale", "log_policy", "zero_counts", "infinite_bound_count",
        "fixed_snapshots", "optional_network_scope", "improvement_statistics",
        "population_counts",
        "status_counts",
        "snapshot",
        "trend",
        "actual_initial_snr",
        "center_provenance",
        "comparison_status",
        "current_reference_audit",
        "independent_latent_tolerance",
        "independent_tolerance_rmse",
        "latent_dimension",
        "symlog_linthresh",
        "empty_groups",
        "reference_provenance",
        "clean_update_prerequisite",
        "theorem_claim_scope",
        "certification_status",
        "numerical_uncertainty",
        "excluded_structural_rows",
        "tolerance_grid_rmse",
        "saturation_to_one_count",
        "loss_statistical_unit",
        "gaussian_statistical_unit",
        "bootstrap_interval_definition",
        "bootstrap_scope",
        "bootstrap_policy",
        "analytical_snr_grid",
        "input_law_separation",
        "relative_error_statistics",
        "branch_error_statistics",
        "vector_residual_statistics",
        "relative_residual_statistics",
        "relative_residual_denominator_floor_rmse",
        "relative_residual_excluded_count",
        "source_precision_statistics",
        "verification_source_counts",
        "failed_independent_consistency_count",
        "excluded_direction_count",
        "exclusion_statuses",
        "decomposition_qa_counts",
        "incomplete_pair_count",
        "matched_control_status_counts",
        "geometry_levels",
        "reference_scale_rmse",
        "reference_scale_definition",
        "terminal_scope",
        "terminal_correction_sources", "terminal_certainty",
        "scope_counts",
        "original_clean_counts",
        "zero_mass",
        "grouped_zero_mass", "grouped_infinite_mass", "group_population",
        "group_common_population_table", "grouped_predeclared_coverage", "grouped_paired_ordering_audit",
        "group_sscd_exclusion_table", "pooled_cdf_audit_table", "pooled_metadata_audit_table",
        "manuscript_extension_required",
        "terminal_noise_scope",
        "terminal_noise_run_alpha",
        "predeclared_tolerance_rmse",
        "predeclared_tolerance_coverage",
        "condition_zero_overlap",
        "condition_positive_counts",
        "noise_probability_scope",
        "noise_bound_exceedance_count",
        "predeclared_coverage",
        "infinite_mass",
        "resolved_positive_counts",
        "reliable_positive_margin_negative_gain_count",
        "ordering_interpretation",
        "raw_ordering_violations",
        "excluded_status_counts",
        "certainty",
        "initial_measurement_scope",
        "native_sweep_figure",
        "display_window_definition",
        "sample_relative_error_quantiles",
        "pair_relative_error_quantiles",
        "pair_quantile_definition",
        "geometry_contour_coverage",
        "geometry_pair_table",
        "geometry_coverage_table",
        "injection_premise_table",
        "injection_premise",
        "condition_unresolved_counts",
        "condition_unresolved_fractions",
        "observed_condition_coverage",
        "condition_interpretation",
        "feedback_unresolved_fractions",
        "numerical_resolution",
        "numerical_resolution_table",
        "numerical_cause_counts_table",
        "numerical_coverage_table",
        "endpoint_contract",
        "numerical_layers",
        "numerical_resolution_scope",
        "fixed_cache_statistic_scope",
        "source_sensitivity_scope",
        "row_bound_audit_table",
        "row_bound_audit",
        "absolute_tolerance_coverage_table",
        "absolute_tolerance_grid_rmse",
        "absolute_tolerance_definition",
        "derived_quantity",
        "looseness_audit_table",
        "looseness_statistics",
        "looseness_identity",
        "paired_ordering_audit_table",
        "paired_ordering_audit",
        "complete_transfer_audit_table",
        "pinsker_tv_bound_unclipped_definition",
        "transfer_audit_scope",
    ):
        if key in metadata:
            text.append(
                key.replace("_", " ").capitalize()
                + ": "
                + json.dumps(metadata[key], sort_keys=True, ensure_ascii=True)
            )
    if "display_audit" in entry:
        text.append("Display: " + json.dumps(json_value(entry["display_audit"]), sort_keys=True))
    text += [
        "",
        f"Source: `{entry['source_table']}`; scientific hash `{entry['scientific_hash']}`; formula version `{entry['formula_version']}`.",
        "Full saved input/provenance and actual output hashes: `figure_manifest.json`.",
        "",
    ]
    return projected_error_display_notation("\n".join(text))


def _value_display_metadata(metadata):
    """Describe the artwork independently of saved numerical-sign audits."""
    metadata = copy.deepcopy(metadata)
    metadata.update(
        display_policy="finite_saved_values",
        condition_display_policy="Every finite saved x/y pair is displayed as a circular dot; no sign-resolution gate, marker distinction or status legend",
        condition_interpretation="The scatter shows saved numerical values without claiming certified condition signs or implications",
        endpoint_contract="x is the saved affine-path margin [||Delta||-mathcal{E}_t-mathcal{V}]/sqrt(d), where mathcal{E}_t=u_t dot (Delta_t-bar{Delta}_t) is signed and mathcal{V} is the positive part after the integrated unit-gap reference projection; y is the saved matched-endpoint log-probability gain. Sign classifications do not filter or alter the displayed values. Endpoint-transfer and scientific applicability remain separately recorded",
    )
    return metadata


def _timestep_views(entry, frame, metadata, config):
    """Slice immutable compact rows for additional figures; no new measurements."""
    policy = entry["per_timestep_exports"]
    column = policy["column"]
    if column not in frame:
        raise TheoryError(f"{entry['stem']}: per-timestep figures require saved {column}")
    steps = _number(frame, column)
    science = config.get("scientific_config", config)
    count = science.get("num_inference_steps")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise TheoryError("Per-timestep figures require a positive integer saved prediction count")
    if np.any(~np.isfinite(steps) | (steps < 0) | (steps >= count) | (steps != np.floor(steps))):
        raise TheoryError("Per-timestep figures have invalid saved step_index values")
    views = []
    for step in sorted(set(steps.astype(int))):
        selected = frame.loc[steps == step].copy()
        x, y = _number(selected, "x"), _number(selected, "y")
        finite = np.isfinite(x) & np.isfinite(y)
        name = policy["filename_template"].format(step_index=int(step))
        child = copy.deepcopy(entry)
        child.pop("per_timestep_exports", None)
        identity = entry["stem"] + "/" + name
        child.update(stem=identity, figure_id=identity, stable_stem=identity,
                     parent_figure=entry["stem"], paper_slot=None,
                     section="per_timestep_companion",
                     outputs={extension: policy["directory"] + "/" + name + "." + extension
                              for extension in ("png", "pdf")},
                     semantic_question=entry["semantic_question"] + f" Saved prediction k={int(step)}, manuscript t={count - int(step)}.")
        # Aggregate population and status counts belong to the parent figure.
        # Do not relabel those pooled receipts as statistics of this slice.
        child_metadata = {key: copy.deepcopy(metadata[key]) for key in (
            "symlog_linthresh", "normalization", "input_source", "measurement_contract",
            "condition_formula", "comparison_interpretation", "formula_version",
            "variation_definition", "zero_gap_convention",
            "reference_law", "reference_law_hash", "mathematical_scope") if key in metadata}
        child_metadata = _value_display_metadata(child_metadata)
        child_metadata.update(
            status="available" if finite.any() else "unavailable",
            reason=None if finite.any() else "This saved timestep has no finite margin/gain coordinate pairs; no points are fabricated",
            band_definition="None; individual saved values are shown as dots",
            weighting="One dot per finite saved sample at this timestep; no timestep or seed aggregation",
            timestep_slice={"step_index": int(step), "manuscript_t": count - int(step),
                            "saved_rows": len(selected), "finite_pairs": int(finite.sum()),
                            "nonfinite_pairs": int((~finite).sum()),
                            "native_timesteps": sorted(set(_number(selected, "timestep")[np.isfinite(_number(selected, "timestep"))].tolist())),
                            "snr_values": sorted(set(_number(selected, "snr")[np.isfinite(_number(selected, "snr"))].tolist()))},
        )
        views.append((child, selected, child_metadata))
    return views



def _output_route(outputs):
    """Stable display identity for paired or single-format figure exports."""
    return outputs.get("pdf", next(iter(outputs.values())))


def _export_formats(formats):
    if isinstance(formats, str):
        raise TheoryError("Figure formats must be a nonempty sequence of png/pdf names")
    requested = tuple(formats)
    if (not requested or len(set(requested)) != len(requested)
            or any(value not in {"png", "pdf"} for value in requested)):
        raise TheoryError("Figure formats must contain png and/or pdf exactly once")
    return tuple(value for value in ("png", "pdf") if value in requested)


def _finalize_presentations(pending, entries):
    """Align only display axes and export bounds, leaving compact inputs intact."""
    if not pending:
        return {}
    available = {_output_route(entry["outputs"]): entry for entry in entries if entry["outputs"]}
    by_stem = {available[_output_route(names)]["stem"]: (fig, available[_output_route(names)])
               for fig, names in pending}
    loss_stems = ("initial_loss_recovery", "corollary3_guidance_scale_vs_loss")
    if all(stem in by_stem for stem in loss_stems):
        pairs = [by_stem[stem] for stem in loss_stems]
        ranges = [tuple(fig.axes[0].get_xlim()) for fig, _entry in pairs]
        limits = (min(value[0] for value in ranges), max(value[1] for value in ranges))
        # Existing ranges already include each renderer's fixed padding and
        # displayed uncertainty endpoints. Their union cannot crop either plot.
        ticks = MaxNLocator(nbins=4).tick_values(*limits)
        ticks = ticks[(ticks >= limits[0]) & (ticks <= limits[1])]
        for fig, entry in pairs:
            ax = fig.axes[0]
            if ax.get_xscale() != "linear":
                raise TheoryError("Normalized-loss presentation requires its existing linear x axis")
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlim(limits)
            entry["display_audit"]["axis_limits"]["x"] = list(limits)
            entry["display_audit"]["shared_loss_axis"] = {
                "source_ranges": [list(value) for value in ranges],
                "limits": list(limits), "ticks": ticks.tolist(),
                "policy": "union of complete existing presentation ranges with their fixed padding; no values or square roots recalculated"}
    options = {}
    for _fig, names in pending:
        entry = available[_output_route(names)]
        entry["display_audit"]["export"] = {
            "bbox_inches": "tight", "pad_inches": paper_style.FIGURE_PAD_INCHES}
        entry["display_audit"] = json_value(entry["display_audit"])
        for extension, relative in names.items():
            options[relative] = {"bbox_inches": "tight"}
            if extension == "pdf" and entry["stem"] == "posterior_feedback_condition_margin":
                options[relative]["dpi"] = paper_style.DENSE_PDF_DPI
    return options

def render_paper(stage: Path, *, diagnostics=False, formats=("pdf",), figure_directory=None, publication_directory=None):
    """Render saved scalars; callers stage external PDFs and metadata together.

    Canonical bundles publish outside scientific outputs. Noncanonical low-level
    fixtures may explicitly exercise the legacy local renderer; active pipeline
    and CLI callers always supply the canonical external destination.
    """
    formats = _export_formats(formats)
    stage = Path(stage).absolute()
    if figure_directory is None:
        from utils.experiments.figure_paths import publication_directory as publication_path
        try:
            figure_directory = publication_path(stage)
        except ValueError:
            figure_directory = None
    external = figure_directory is not None
    figure_directory = Path(figure_directory).absolute() if external else stage
    publication_directory = Path(publication_directory or figure_directory).absolute()
    with StageProgress("Loading and validating saved figure inputs"):
        config, summary, audit, frames = load_paper_inputs(stage, diagnostics=diagnostics)
    counterfactual = bool(config.get("supplemental_config", {}).get("counterfactual_unconditional", config.get("scientific_config", config).get("counterfactual_unconditional", False)))
    registry = copy.deepcopy(paper_registry(diagnostics=diagnostics, counterfactual=counterfactual))
    for entry in registry:
        entry["outputs"] = {extension: entry["outputs"][extension] for extension in formats}
    previous_path = contained_path(stage, "figure_manifest.json")
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    owned = {**previous.get("preserved_files", {}), **previous.get("files", {})}
    previous_publication = previous.get("publication", {})
    if not isinstance(previous_publication, dict):
        raise TheoryError("Invalid external figure ownership receipt")
    if external and previous_publication.get("directory") == str(publication_directory):
        publication_owned = {**previous_publication.get("preserved_files", {}),
                             **previous_publication.get("files", {})}
    else:
        publication_owned = {}
    caption_path = contained_path(stage, "figure_captions.md")
    if caption_path.exists():
        if "figure_captions.md" not in owned:
            raise TheoryError("Refusing to replace unowned figure_captions.md")
        if file_sha256(caption_path) != owned["figure_captions.md"]:
            raise TheoryError(
                "Owned figure_captions.md was modified outside the renderer"
            )
    jobs = []
    for specification in registry:
        stem = specification["stem"]
        frame = frames.get(stem, pd.DataFrame())
        metadata = copy.deepcopy(summary["figures"][stem])
        if specification.get("display_policy") == "finite_saved_values":
            metadata = _value_display_metadata(metadata)
        jobs.append((specification, frame, metadata))
    pending = []
    entries, timestep_entries = [], []
    try:
        with plt.rc_context(PLOT_STYLE), tqdm(
            total=len(jobs), desc="[Theory] Preparing figures", unit="figure",
            dynamic_ncols=True, leave=True, disable=False,
        ) as progress:
            for specification, frame, metadata in jobs:
                progress.set_postfix_str(str(Path(_output_route(specification["outputs"])).with_suffix("")), refresh=True)
                entry = copy.deepcopy(specification)
                entry["style_version"] = paper_style.STYLE_VERSION
                if "notation_details" in entry:
                    entry["notation_details"] = entry["notation_details"].replace("Gold/purple", "Coral/indigo")
                stem = entry["stem"]
                source_stem = entry.get("parent_figure", stem)
                status = metadata["status"]
                if status == "unavailable" and not entry.get("parent_figure") and entry["category"] != "diagnostics" and not entry.get("allow_unavailable"):
                    raise TheoryError(
                        f"Required paper figure {stem} is unavailable: {metadata['reason']}. Run {recompute_command(config)}"
                    )
                entry.update(
                    status=status,
                    complete=status
                    in {
                        "available",
                        "complete",
                        "not_applicable",
                        "alias",
                        "unavailable",
                    },
                    measurement_metadata=metadata,
                    scientific_hash=config["scientific_hash"],
                    metric_schema_version=config["metric_schema_version"],
                    audit_version=audit.get(
                        "audit_version", audit.get("schema_version")
                    ),
                    provenance=copy.deepcopy(config.get("provenance", {})),
                    source_analysis=copy.deepcopy(config.get("source_analysis", {})),
                    plot_input=copy.deepcopy(summary.get("plot_data", {}).get(source_stem)),
                    requested_outputs=entry["outputs"].copy(),
                    output_hashes={},
                )
                if status in {"available", "complete"}:
                    for relative in entry["outputs"].values():
                        name = Path(relative).name if external else relative
                        path = contained_path(figure_directory, name)
                        destination_owned = publication_owned if external else owned
                        if path.exists() and name not in destination_owned:
                            raise TheoryError(
                                f"Refusing to replace unowned paper figure {relative}"
                            )
                        if path.exists() and file_sha256(path) != destination_owned[name]:
                            raise TheoryError(
                                f"Owned paper figure was modified outside the renderer: {relative}"
                            )
                    fig, stats = _draw(entry, frame, metadata, config)
                    pending.append((fig, entry["outputs"]))
                    entry["display_audit"] = json_value(stats)
                else:
                    entry["outputs"] = {}
                (timestep_entries if entry.get("parent_figure") else entries).append(entry)
                progress.update(1)
        all_entries = [*entries, *timestep_entries]
        with plt.rc_context(paper_style.THEORY_STYLE), StageProgress("Aligning figure layouts and loss axes"):
            export_options = _finalize_presentations(pending, all_entries)
        # Validate caption metadata before expensive PNG/PDF export. The same
        # native JSON values are retained in the eventual figure manifest.
        with StageProgress("Preparing figure captions"):
            destination_note = (f"Publication directory: `{publication_directory}`. "
                                "PDF basenames are relative to this directory; captions, ownership receipts, "
                                "and scientific tables remain in the source bundle.\n\n") if external else ""
            captions = "# Fixed paper figure captions\n\n" + destination_note + "\n".join(
                _caption(entry) for entry in all_entries
            )
        with plt.rc_context(paper_style.THEORY_STYLE), tqdm(
            total=sum(len(names) for _figure, names in pending),
            desc="[Theory] Exporting figures", unit="file",
            dynamic_ncols=True, leave=True, disable=False,
        ) as progress:
            def exported(relative, status):
                if status == "saving":
                    progress.set_postfix_str(relative, refresh=True)
                elif status == "saved":
                    progress.update(1)
            publish_options = {} if formats == ("png", "pdf") else {"formats": formats}
            if external:
                publication_pending = [(figure, {extension: Path(name).name for extension, name in names.items()})
                                       for figure, names in pending]
                publication_options = {Path(name).name: options for name, options in export_options.items()}
            else:
                publication_pending, publication_options = pending, export_options
            publish_figures(figure_directory, publication_pending, progress=exported,
                            export_options=publication_options, **publish_options)
        with StageProgress("Writing figure captions and manifest"):
            rendered_files = {
                (Path(relative).name if external else relative):
                    file_sha256(figure_directory / (Path(relative).name if external else relative))
                for entry in all_entries
                for relative in entry["outputs"].values()
            }
            for entry in all_entries:
                entry["output_hashes"] = {
                    format: rendered_files[Path(relative).name if external else relative]
                    for format, relative in entry["outputs"].items()
                }
                if external:
                    entry["publication_outputs"] = {format: Path(relative).name
                                                    for format, relative in entry["outputs"].items()}
            files = {} if external else rendered_files
            caption_path.write_text(captions)
            files["figure_captions.md"] = file_sha256(caption_path)
            manifest = {
                "schema_version": 2 if external else 1,
                "complete": True,
                "registry_version": REGISTRY_VERSION,
                "rendering_version": RENDERING_VERSION,
                "style_version": paper_style.STYLE_VERSION,
                "renderer_source_sha256": file_sha256(Path(__file__)),
                "renderer_sources": {
                    **{name: file_sha256(Path(__file__).with_name(name)) for name in ("paper_plotting.py", "evidence_plotting.py", "four_stage_plotting.py", "paper_notation.py", "paper_style.py")},
                    "publication_style.py": file_sha256(Path(__file__).parent.parent / "publication_style.py"),
                },
                "scientific_hash": config["scientific_hash"],
                "metric_schema_version": config["metric_schema_version"],
                "diagnostics_requested": diagnostics,
                "export_formats": list(formats),
                "manuscript_extension_required": any(
                    entry["measurement_metadata"].get("manuscript_extension_required") is True
                    for entry in entries
                ),
                "requested_counts": {
                    "main": sum(e["category"] == "main" for e in registry),
                    "appendix": sum(e["category"] == "appendix" for e in registry),
                    "diagnostics": sum(e["category"] == "diagnostics" for e in registry),
                    "timestep_figures": len(timestep_entries),
                },
                "figures": entries,
                "timestep_figures": timestep_entries,
                "files": files,
                "preserved_files": {name: digest for name, digest in owned.items() if name not in files},
                "render_retirements": list(RENDER_RETIREMENTS),
                "plot_recipe_version": PLOT_RECIPE_VERSION,
            }
            if external:
                manifest["publication"] = {
                    "directory": str(publication_directory), "files": rendered_files,
                    "preserved_files": {name: digest for name, digest in publication_owned.items()
                                        if name not in rendered_files},
                    "paths_relative_to": "publication.directory",
                    "metadata_location": "scientific bundle; publication contains figures only",
                }
            atomic_write_json(previous_path, manifest)
        return manifest
    finally:
        for figure, _names in pending:
            plt.close(figure)
