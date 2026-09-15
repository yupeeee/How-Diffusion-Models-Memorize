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
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
import numpy as np
import pandas as pd

from utils.common.io import atomic_write_json, file_sha256
from utils.experiments.plotting import (
    PLOT_STYLE,
    SCATTER_ALPHA,
    SCATTER_SIZE,
    SSCD_COLOR_RANGE,
    add_sscd_colorbar,
    publish_figures,
)
from .contracts import TheoryError
from .paper_contracts import contained_path, load_paper_inputs, recompute_command
from .paper_registry import GROUPS, GROUP_COLORS, GROUP_LABELS, REGISTRY_VERSION, paper_registry

RENDERING_VERSION = "paper-stix-6"
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
        fontsize=10,
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
        fontsize=10,
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
    keys = [key for key in ("group", "branch", "metric") if key in frame]
    if not keys:
        return [((), frame)]
    return list(frame.groupby(keys, sort=False, dropna=False))


def _identity(series, frame):
    keys = [key for key in ("group", "branch", "metric") if key in frame]
    values = series if isinstance(series, tuple) else (series,)
    return dict(zip(keys, values))


def _draw_scatter(fig, ax, entry, frame, metadata, config):
    x, y = _number(frame, "x"), _number(frame, "y")
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise TheoryError("Available scatter has no finite coordinate pairs")
    if entry.get("sscd"):
        colors = _number(frame, "terminal_sscd")
        colored = finite & np.isfinite(colors)
        ax.scatter(
            x[colored],
            y[colored],
            c=colors[colored],
            cmap="viridis",
            norm=Normalize(*SSCD_COLOR_RANGE, clip=True),
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            edgecolors="none",
            rasterized=True,
        )
        missing = finite & ~np.isfinite(colors)
        if missing.any():
            ax.scatter(
                x[missing],
                y[missing],
                color=".5",
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                edgecolors="none",
                rasterized=True,
            )
        add_sscd_colorbar(fig, ax, colors[colored])
    else:
        ax.scatter(
            x[finite],
            y[finite],
            color=GROUP_COLORS[GROUPS[1]],
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            edgecolors="none",
            rasterized=True,
        )
    if entry.get("equal"):
        limits = _finite_limits(
            x[finite], y[finite], include_zero=entry["stem"] == "initial_recovery"
        )
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.plot(
            limits,
            limits,
            color=".45",
            linestyle="--",
            linewidth=1,
            label=entry["reference"],
        )
    else:
        ax.set_xlim(_finite_limits(x[finite]))
        ax.set_ylim(_finite_limits(y[finite]))
    if entry.get("zero_guides"):
        ax.axhline(0, color=".65", linewidth=0.7)
        ax.axvline(0, color=".65", linewidth=0.7)
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
        axis.set_major_locator(MaxNLocator(nbins=5))
        _compact_ticks(axis)
    _legend(ax)
    return {
        "finite_pairs": int(finite.sum()),
        "missing_coordinate_pairs": int((~finite).sum()),
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
            linestyle="--" if branch == "unconditional" else "-",
            linewidth=1.4,
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
    ax.set_ylim(_finite_limits(*extrema, include_zero=entry.get("signed", False)))
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
        if entry["stem"] == "target_synchronization":
            _synchronization_legend(ax, handles[:outcome_count], handles[outcome_count:])
        else:
            _legend(ax, handles, strip=True)
    else:
        _legend(ax, strip=len(metric_names) > 2 or entry["kind"] == "dose")
    return {
        "finite_summary_cells": observed,
        "population": "all" if metric_diagnostic else "saved outcome cohorts",
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
    for axis in ("x", "y"):
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
                if entry["kind"] == "scatter"
                else "value"
                if entry["kind"] == "ecdf"
                else "step_index"
                if entry.get("update_index")
                else "analytical_snr"
                if entry.get("analytical_snr")
                else "lambda"
                if entry["kind"] == "dose"
                else "snr"
            )
            columns = [column] if column in frame else []
        else:
            columns = [
                name
                for name in (
                    "y",
                    "cdf",
                    "median",
                    "q25",
                    "q75",
                    "minimum",
                    "maximum",
                    "fraction",
                    "upper_fraction",
                )
                if name in frame
            ]
        for column in columns:
            data = _number(frame, column)
            data = data[np.isfinite(data)]
            if len(data) and (
                data.min() < float(values[0]) - 1e-12
                or data.max() > float(values[1]) + 1e-12
            ):
                raise TheoryError("Saved paper axis limits would crop finite data")
        getattr(ax, "set_" + axis + "lim")(values)


def _draw(entry, frame, metadata, config):
    missing = set(entry["required_columns"]) - set(frame)
    if missing:
        raise TheoryError(
            f"{entry['stem']}: compact table misses {sorted(missing)}. Run {recompute_command(config)}"
        )
    fig = plt.Figure()
    ax = fig.subplots()
    try:
        ax.set_axisbelow(True)
        ax.grid(True, color=".9", linewidth=0.6)
        kind = entry["kind"]
        if kind == "scatter":
            stats = _draw_scatter(fig, ax, entry, frame, metadata, config)
        elif kind == "ecdf":
            stats = _draw_ecdf(ax, entry, frame, metadata)
        elif kind == "fraction":
            stats = _draw_fraction(ax, frame, metadata)
        elif kind == "coverage":
            stats = _draw_coverage(ax, frame, metadata)
        else:
            stats = _draw_curves(ax, entry, frame, metadata)
        _apply_saved_limits(ax, metadata, frame, entry)
        ax.set_xlabel(entry["axes"]["x"])
        ax.set_ylabel(entry["axes"]["y"])
        stats["axis_limits"] = {"x": list(ax.get_xlim()), "y": list(ax.get_ylim())}
        return fig, stats
    except BaseException:
        plt.close(fig)
        raise


def _caption(entry):
    metadata = entry["measurement_metadata"]
    text = [
        f"## {entry['category']}/{entry['stem']}",
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
        "Normalization: " + str(metadata.get("normalization", entry["normalization"])),
        "Weighting: " + str(metadata.get("weighting", entry["weighting"])),
        "Bands: "
        + str(
            metadata.get(
                "band_definition",
                metadata.get("band_meaning", entry["band_definition"]),
            )
        ),
        "Groups: " + entry["group_rule"],
    ]
    for key in (
        "counts",
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
    ):
        if key in metadata:
            text.append(
                key.replace("_", " ").capitalize()
                + ": "
                + json.dumps(metadata[key], sort_keys=True, ensure_ascii=True)
            )
    if "display_audit" in entry:
        text.append("Display: " + json.dumps(entry["display_audit"], sort_keys=True))
    text += [
        "",
        f"Source: `{entry['source_table']}`; scientific hash `{entry['scientific_hash']}`; formula version `{entry['formula_version']}`.",
        "Full saved input/provenance and actual output hashes: `figure_manifest.json`.",
        "",
    ]
    return "\n".join(text)


def render_paper(stage: Path, *, diagnostics=False):
    """Export a complete staged paper set; caller owns the role lock/directory swap."""
    stage = Path(stage).absolute()
    config, summary, audit, frames = load_paper_inputs(stage, diagnostics=diagnostics)
    registry = paper_registry(diagnostics=diagnostics)
    previous_path = contained_path(stage, "figure_manifest.json")
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    owned = previous.get("files", {})
    caption_path = contained_path(stage, "figure_captions.md")
    if caption_path.exists():
        if "figure_captions.md" not in owned:
            raise TheoryError("Refusing to replace unowned figure_captions.md")
        if file_sha256(caption_path) != owned["figure_captions.md"]:
            raise TheoryError(
                "Owned figure_captions.md was modified outside the renderer"
            )
    pending = []
    entries = []
    try:
        with plt.rc_context(PLOT_STYLE):
            for specification in registry:
                entry = copy.deepcopy(specification)
                stem = entry["stem"]
                metadata = copy.deepcopy(summary["figures"][stem])
                status = metadata["status"]
                if status == "unavailable" and entry["category"] != "diagnostics":
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
                    plot_input=copy.deepcopy(summary.get("plot_data", {}).get(stem)),
                    requested_outputs=entry["outputs"].copy(),
                    output_hashes={},
                )
                if status in {"available", "complete"}:
                    for relative in entry["outputs"].values():
                        path = contained_path(stage, relative)
                        if path.exists() and relative not in owned:
                            raise TheoryError(
                                f"Refusing to replace unowned paper figure {relative}"
                            )
                        if path.exists() and file_sha256(path) != owned[relative]:
                            raise TheoryError(
                                f"Owned paper figure was modified outside the renderer: {relative}"
                            )
                    fig, stats = _draw(entry, frames[stem], metadata, config)
                    entry["display_audit"] = stats
                    pending.append((fig, entry["outputs"]))
                else:
                    entry["outputs"] = {}
                entries.append(entry)
            publish_figures(stage, pending)
        files = {
            relative: file_sha256(stage / relative)
            for entry in entries
            for relative in entry["outputs"].values()
        }
        for entry in entries:
            entry["output_hashes"] = {
                format: files[relative] for format, relative in entry["outputs"].items()
            }
        captions = "# Fixed paper figure captions\n\n" + "\n".join(
            _caption(entry) for entry in entries
        )
        caption_path.write_text(captions)
        files["figure_captions.md"] = file_sha256(caption_path)
        manifest = {
            "schema_version": 1,
            "complete": True,
            "registry_version": REGISTRY_VERSION,
            "rendering_version": RENDERING_VERSION,
            "renderer_source_sha256": file_sha256(Path(__file__)),
            "scientific_hash": config["scientific_hash"],
            "metric_schema_version": config["metric_schema_version"],
            "diagnostics_requested": diagnostics,
            "requested_counts": {
                "main": 4,
                "core_appendix": 8,
                "conditional_terminal": 2,
            },
            "figures": entries,
            "files": files,
        }
        atomic_write_json(previous_path, manifest)
        return manifest
    finally:
        for figure, _names in pending:
            plt.close(figure)
