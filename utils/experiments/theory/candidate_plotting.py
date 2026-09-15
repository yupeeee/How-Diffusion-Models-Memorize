"""Offline discovery gallery from saved scalar tables, without scientific loaders.

All numerical measurements and summary tables remain immutable. Figure output is
staged before publication and only files owned by this renderer may be replaced.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile
import textwrap

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .candidate_contracts import (
    read_tables,
    validate_candidate_bundle,
    recompute_command,
)
from .candidate_registry import (
    GROUPS,
    POPULATION_DESCRIPTIONS,
    BASELINE_SWEEP_WEIGHTING,
    TOLERANCES,
    candidate_registry,
    feedback_snapshots,
    synchronization_snapshots,
)
from .contracts import TheoryError, digest_file, read_object

STYLE = {
    "figure.figsize": (4.0, 4.0),
    "savefig.dpi": 150,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "text.usetex": False,
}
STYLE_VERSION = "candidate-stix-2"
CANDIDATE_AXIS_LABELS = {
    "PF01": {"y": "Candidate matched log-probability gain"},
    "PF02": {"y": "Normalized candidate log-odds gain"},
    "PF03": {"y": "Positive candidate-gain fraction"},
    "PF04": {
        "x": "Matched candidate target log odds",
        "y": "Guided candidate target log odds",
    },
    "PF05": {"y": "Candidate target-evidence alignment"},
    "PF06": {"y": "Candidate local log-probability gain"},
    "PF07": {"y": "Candidate positive coverage"},
    "PF08": {"y": "Candidate profile fraction"},
    "PF10": {"x": "Early normalized candidate gain"},
    "PF11": {"y": "Matched candidate log-odds regime"},
    "PF12": {"y": "Paired candidate-gain contrast"},
}
GROUP_COLORS = {"All": "#343434", "SSCD > 0.75": "#e5ba26", "SSCD <= 0.75": "#563580"}
PROMPT_KEYS = ["run_id", "original_index", "record_id", "target_id"]
SAMPLE_KEYS = PROMPT_KEYS + ["seed"]
LINTHRESH = 0.001
FIGURE_MANIFEST = "candidate_figure_manifest.json"
ALIASES = {
    "candidate_displacement_rmse": (
        "candidate_displacement_rmse",
        "local_displacement_rmse",
        "matched_displacement_rmse",
        "matched_shift_rmse",
    ),
    "current_unconditional_candidate_reference_error_rmse": (
        "current_unconditional_candidate_reference_error_rmse",
        "candidate_current_reference_error_u_rmse",
        "unconditional_reference_error_rmse",
    ),
    "candidate_radius_tail_rmse": (
        "candidate_radius_tail_rmse",
        "candidate_concentration_term_rmse",
        "candidate_radius_concentration_rmse",
    ),
}


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if value is pd.NA:
        return None
    return value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _numbers(frame, name):
    if name not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _group(frame, group):
    if group == "All":
        if "group" in frame:
            selected = frame[frame["group"].isin(["All", "all", "all_samples"])]
            return selected if len(selected) else frame
        return frame
    if "group" in frame:
        return frame[frame["group"] == group]
    values = _numbers(frame, "terminal_sscd")
    return frame[
        np.isfinite(values)
        & ((values > 0.75) if group == GROUPS[1] else (values <= 0.75))
    ]


def _weights(frame):
    keys = [key for key in PROMPT_KEYS if key in frame]
    if not keys or frame.empty:
        return np.ones(len(frame), dtype=float)
    counts = frame.groupby(keys, dropna=False)[keys[0]].transform("size")
    return 1 / counts.to_numpy(dtype=float)


def _quantiles(values, weights, quantiles=(0.25, 0.5, 0.75)):
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = np.asarray(values)[finite], np.asarray(weights)[finite]
    if not len(values):
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return values[
        np.minimum(np.searchsorted(cumulative, quantiles, side="left"), len(values) - 1)
    ]


def _counts(frame):
    result = {"source_rows": len(frame)}
    for name, keys in (("samples", SAMPLE_KEYS), ("prompts", PROMPT_KEYS)):
        actual = [key for key in keys if key in frame]
        if actual and (name != "prompts" or set(PROMPT_KEYS) <= set(frame)):
            result[name] = len(frame[actual].drop_duplicates())
    if "seed" in frame:
        result["distinct_seed_ids"] = int(frame.seed.nunique())
    if "target_id" in frame:
        result["distinct_targets"] = int(frame.target_id.nunique())
    if "terminal_sscd" in frame:
        scores = _numbers(frame, "terminal_sscd")
        result["sscd_below_display_range"] = int((scores < 0).sum())
        result["sscd_above_display_range"] = int((scores > 1).sum())
        result["sscd_unavailable"] = int((~np.isfinite(scores)).sum())
        keys = [key for key in PROMPT_KEYS if key in frame]
        if keys:
            high = set(
                map(tuple, frame.loc[scores > 0.75, keys].drop_duplicates().to_numpy())
            )
            low = set(
                map(
                    tuple,
                    frame.loc[np.isfinite(scores) & (scores <= 0.75), keys]
                    .drop_duplicates()
                    .to_numpy(),
                )
            )
            result["overlapping_outcome_group_prompts"] = len(high & low)
    statuses = {}
    for column in frame.columns:
        if column.endswith("_status") or column in {
            "candidate_profile_class",
            "reconstruction_method",
        }:
            statuses[column] = (
                frame[column].fillna("missing").astype(str).value_counts().to_dict()
            )
    if statuses:
        result["status_counts"] = statuses
    for name in ("prompt_count", "seed_count", "n_observations", "denominator_count"):
        if name in frame and len(frame):
            result["maximum_saved_summary_cell_" + name] = pd.to_numeric(
                frame[name], errors="coerce"
            ).max()
    return result


def _source(tables, name):
    if name.startswith("summaries/"):
        return tables["summaries"].get(name.split("/", 1)[1], pd.DataFrame())
    return tables.get(
        "trajectory" if name == "trajectory_metrics" else name, pd.DataFrame()
    )


def _relative_sources(manifest, source):
    names = manifest["numerical_files"]
    exact = source + ".parquet"
    if exact in names:
        return [exact]
    return [
        name
        for name in sorted(names)
        if name.startswith(source + "/") and name.endswith(".parquet")
    ]


def validate_candidates(bundle):
    """Check the shared numerical contract and all registered saved sources."""
    bundle = Path(bundle).resolve()
    manifest = validate_candidate_bundle(bundle)
    registry = read_object(bundle / "registry.json")
    designs = registry.get("designs", [])
    required_ids = {entry["id"] for entry in candidate_registry()["designs"]}
    ids = [entry.get("id") for entry in designs]
    if len(ids) != 34 or set(ids) != required_ids:
        raise TheoryError(
            "Candidate registry must contain exactly the 34 prescribed base designs. Run "
            + recompute_command(manifest["config"])
        )
    # Every source has an explicit saved table even when it has no eligible rows.
    required = {entry["source_table"] for entry in designs}
    for source in sorted(required):
        if not _relative_sources(manifest, source):
            raise TheoryError(
                f"Missing candidate source {source}; run {recompute_command(manifest['config'])}"
            )
    for card in designs:
        paths = _relative_sources(manifest, card["source_table"])
        columns, row_count = set(), 0
        for relative in paths:
            saved = pq.ParquetFile(bundle / relative)
            columns.update(saved.schema_arrow.names)
            row_count += saved.metadata.num_rows
        missing = set(card["required_columns"]) - columns
        if row_count and missing:
            raise TheoryError(
                f"Missing candidate fields for {card['id']}: {sorted(missing)}. Run {recompute_command(manifest['config'])}"
            )
    return manifest


def _variant_cards(registry, tables):
    trajectory = tables.get("trajectory", pd.DataFrame())
    prediction_steps = sorted(
        set(
            _numbers(trajectory, "step_index")[
                np.isfinite(_numbers(trajectory, "step_index"))
            ].astype(int)
        )
    )
    if "feedback_eligible" in trajectory:
        eligible = trajectory[trajectory.feedback_eligible.fillna(False).astype(bool)]
    elif "candidate_endpoint_status" in trajectory:
        eligible = trajectory[trajectory.candidate_endpoint_status != "not_applicable"]
    elif "candidate_gain_status" in trajectory:
        eligible = trajectory[trajectory.candidate_gain_status != "not_applicable"]
    else:
        eligible = trajectory
    eligible_steps = sorted(
        set(
            _numbers(eligible, "step_index")[
                np.isfinite(_numbers(eligible, "step_index"))
            ].astype(int)
        )
    )
    snapshots = feedback_snapshots(eligible_steps)
    sync = synchronization_snapshots(prediction_steps, eligible_steps)
    cards = []
    for original in registry["designs"]:
        kind = original.get("variants")
        variants = [{}]
        if kind == "groups":
            variants = [
                {
                    "group": group,
                    "suffix": ""
                    if group == "All"
                    else "__" + ("high_sscd" if group == GROUPS[1] else "lower_sscd"),
                }
                for group in GROUPS
            ]
        elif kind == "feedback_snapshots":
            variants = [
                {"step_index": step, "suffix": f"__k{step:03d}"} for step in snapshots
            ] or [{"suffix": "__no_eligible_snapshot"}]
        elif kind == "synchronization_snapshots_groups":
            variants = [
                {
                    "step_index": step,
                    "snapshot": name,
                    "group": group,
                    "suffix": f"__{name}_k{step:03d}"
                    + (
                        ""
                        if group == "All"
                        else "__"
                        + ("high_sscd" if group == GROUPS[1] else "lower_sscd")
                    ),
                }
                for name, step in sync
                for group in GROUPS
            ] or [{}]
        elif kind == "tolerances":
            variants = [
                {
                    "tolerance": tolerance,
                    "suffix": "__tol" + str(tolerance).replace(".", "p"),
                }
                for tolerance in TOLERANCES
            ]
        elif kind == "within_prompt":
            variants = [{}, {"within_prompt": True, "suffix": "__within_prompt"}]
        if original["id"] == "TS05":
            variants = [
                {
                    "group": group,
                    "suffix": ""
                    if group == "All"
                    else "__" + ("high_sscd" if group == GROUPS[1] else "lower_sscd"),
                }
                for group in GROUPS
            ]
        for variant in variants:
            card = copy.deepcopy(original)
            if card["id"] in POPULATION_DESCRIPTIONS:
                card["population"] = POPULATION_DESCRIPTIONS[card["id"]]
            if card["id"] == "UB04":
                card["weighting"] = BASELINE_SWEEP_WEIGHTING
            card.update(variant)
            card["variant_id"] = card["id"] + variant.get("suffix", "")
            namespace = card.get("namespace", card["family"])
            card["artifact_stem"] = namespace + "/" + card["variant_id"]
            if "step_index" in card:
                rows = trajectory[trajectory.step_index == card["step_index"]]
                card["snapshot_schedule"] = {
                    name: sorted(
                        set(
                            _numbers(rows, name)[
                                np.isfinite(_numbers(rows, name))
                            ].tolist()
                        )
                    )
                    for name in (
                        "step_index",
                        "timestep",
                        "snr",
                        "destination_timestep",
                        "destination_snr",
                    )
                }
            cards.append(card)
        if original.get("extra_geometry"):
            card = copy.deepcopy(original)
            card.update(
                variant_id="TR01__last_prediction_geometry",
                artifact_stem="last_prediction_geometry/TR01",
                clean_update=False,
                scope="last stored branch prediction",
                result="Last-prediction geometry",
                evidence_tag="learned_behavior",
                question="What is the two-term geometry of the last stored prediction?",
                formula="A=||m_c,last-x_star||/sqrt(d); B=(g-1)||Delta_last||/sqrt(d); no clean-endpoint attribution",
                affirmative_pattern_meaning="Small last-prediction conditional and residual-guidance terms.",
                does_not_establish="A final-reproduction certificate or a clean terminal update.",
            )
            cards.append(card)
    return cards, {"feedback": snapshots, "synchronization": sync}


def _limits(ax, x, y=None, *, equal=False, equality_label="Equal errors"):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if y is None:
        return
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if equal:
        values = np.concatenate((x, y))
        if len(values):
            low, high = float(min(0, values.min())), float(max(0, values.max()))
            gap = max(high - low, 1e-6) * 0.04
            ax.set_xlim(low - gap, high + gap)
            ax.set_ylim(low - gap, high + gap)
            ax.set_aspect("equal", adjustable="box")
            ax.plot(
                [low - gap, high + gap],
                [low - gap, high + gap],
                color=".45",
                lw=1,
                ls="--",
                label=equality_label,
            )


def _scatter(
    ax, frame, xcol, ycol, *, color=False, equal=False, equality_label="Equal errors"
):
    x, y = _numbers(frame, xcol), _numbers(frame, ycol)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return {"finite_observations": 0}
    if color:
        c = _numbers(frame, "terminal_sscd")
        missing = valid & ~np.isfinite(c)
        if missing.any():
            ax.scatter(
                x[missing],
                y[missing],
                s=7,
                color=".65",
                alpha=0.5,
                linewidths=0,
                rasterized=True,
            )
        colored = valid & np.isfinite(c)
        dots = ax.scatter(
            x[colored],
            y[colored],
            c=c[colored],
            cmap="viridis",
            norm=Normalize(0, 1),
            s=7,
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )
        ax.figure.colorbar(dots, ax=ax, pad=0.025).set_label("SSCD")
    else:
        ax.scatter(
            x[valid],
            y[valid],
            s=8,
            color="#55447e",
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
    _limits(ax, x[valid], y[valid], equal=equal, equality_label=equality_label)
    return {
        "finite_observations": int(valid.sum()),
        "rendered_observations": int(valid.sum()),
        "omitted_nonfinite": int((~valid).sum()),
        "all_data_ranges": {
            "x": [x[valid].min(), x[valid].max()],
            "y": [y[valid].min(), y[valid].max()],
        },
        "descriptive_y": {
            "negative": int((y[valid] < 0).sum()),
            "zero": int((y[valid] == 0).sum()),
            "positive": int((y[valid] > 0).sum()),
            "median": float(np.median(y[valid])),
        },
    }


def _ecdf(ax, frame, column, label, *, color=None, weighted=True):
    values = _numbers(frame, column)
    valid = np.isfinite(values)
    if "rank" in column:
        valid &= values >= 1
    if not valid.any():
        return 0
    valid_frame = frame.loc[valid]
    weights = _weights(valid_frame) if weighted else np.ones(len(valid_frame))
    values = values[valid]
    order = np.argsort(values, kind="stable")
    denominator = float(weights.sum())
    ax.step(
        values[order],
        np.cumsum(weights[order]) / denominator,
        where="post",
        label=label,
        color=color,
        lw=1.7,
    )
    ax.set_ylim(0, 1.02)
    return int(valid.sum())


def _binned(ax, frame, xcol, ycol, *, sscd=False):
    x, y = _numbers(frame, xcol), _numbers(frame, ycol)
    valid = np.isfinite(x) & np.isfinite(y)
    frame, x, y = frame[valid], x[valid], y[valid]
    if not len(frame):
        return []
    edges = (
        np.concatenate(([-np.inf], np.linspace(0, 1, 11), [np.inf]))
        if sscd
        else np.linspace(x.min(), x.max() + max(np.ptp(x), 1) * 1e-12, 11)
    )
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        select = (x >= lo) & (x < hi)
        if not select.any():
            continue
        weights = _weights(frame[select])
        q25, median, q75 = _quantiles(y[select], weights)
        location = _quantiles(x[select], weights, (0.5,))[0]
        rows.append(
            {
                "bin_left": lo,
                "bin_right": hi,
                "x": location,
                "median": median,
                "q25": q25,
                "q75": q75,
                "count": int(select.sum()),
            }
        )
    if rows:
        values = pd.DataFrame(rows)
        ax.errorbar(
            values.x,
            values["median"],
            yerr=[values["median"] - values.q25, values.q75 - values["median"]],
            color="#121212",
            lw=1,
            marker="o",
            ms=3,
            capsize=2,
            label="Median",
        )
    return rows


def _trend(ax, frame, xcol, ycol):
    x, y = _numbers(frame, xcol), _numbers(frame, ycol)
    select = np.isfinite(x) & np.isfinite(y)
    x, y, frame = x[select], y[select], frame[select]
    if len(x) < 2 or np.ptp(x) == 0:
        return {"status": "unavailable_seed_variation"}
    weights = _weights(frame)
    mean_x, mean_y = np.average(x, weights=weights), np.average(y, weights=weights)
    variance = np.average((x - mean_x) ** 2, weights=weights)
    slope = np.average((x - mean_x) * (y - mean_y), weights=weights) / variance
    intercept = mean_y - slope * mean_x
    limits = np.array([x.min(), x.max()])
    ax.plot(
        limits, intercept + slope * limits, color=".15", lw=1.3, label="Weighted trend"
    )
    return {
        "weighted_descriptive_slope": slope,
        "intercept": intercept,
        "inference": "Descriptive only; shared targets and repeated seed IDs are not independent replicates.",
    }


def _curve(ax, frame, xcol, label, *, color=None, linestyle="-", band=True):
    frame = frame.sort_values(xcol)
    x, y = _numbers(frame, xcol), _numbers(frame, "median")
    valid = np.isfinite(x) & np.isfinite(y)
    if xcol in {"snr", "analytical_snr"}:
        valid &= x > 0
    if not valid.any():
        return 0
    # Keep unavailable rows as NaN so a missing saved step is not bridged.
    x, y = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
    ax.plot(x, y, lw=1.6, color=color, ls=linestyle, label=label)
    if band and {"q25", "q75"} <= set(frame):
        low, high = _numbers(frame, "q25"), _numbers(frame, "q75")
        ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
    if {"minimum", "maximum"} <= set(frame):
        # All-data extrema, including values beyond the IQR, set display limits.
        low, high = _numbers(frame, "minimum"), _numbers(frame, "maximum")
        extrema = np.concatenate((low[np.isfinite(low)], high[np.isfinite(high)]))
        if len(extrema):
            finite_x = x[np.isfinite(x)]
            ax.update_datalim(
                np.array(
                    [[finite_x.min(), extrema.min()], [finite_x.max(), extrema.max()]]
                )
            )
            ax.autoscale_view()
    return int(valid.sum())


def _metric_rows(frame, metric):
    if "metric" not in frame:
        return frame.iloc[:0]
    return frame[frame.metric.isin(ALIASES.get(metric, (metric,)))]


def _timeseries(ax, frame, card):
    metrics = card.get("metrics", [])
    groups = [card["group"]] if "group" in card else list(GROUPS[1:])
    if "group" not in frame or not any(len(_group(frame, group)) for group in groups):
        groups = ["All"]
    count = 0
    for group in groups:
        for index, metric in enumerate(metrics):
            rows = _metric_rows(_group(frame, group), metric)
            name = {
                "conditional_target_error_rmse": "Conditional",
                "unconditional_target_error_rmse": "Unconditional",
                "branch_gap_rmse": "Branch gap",
                "current_unconditional_candidate_reference_error_rmse": "Unconditional reference error",
                "candidate_radius_tail_rmse": "Candidate radius term",
            }.get(metric, "")
            if card["id"] == "TS05":
                color = plt.get_cmap("tab10")(index)
                label, linestyle = name, "-"
            else:
                color = GROUP_COLORS[group]
                label = group + (" · " + name if len(metrics) > 1 else "")
                linestyle = "--" if metric == "unconditional_target_error_rmse" else "-"
            count += _curve(ax, rows, "snr", label, color=color, linestyle=linestyle)
    ax.set_xscale("log")
    if card["id"] in {"PF01", "PF02", "PF05", "PF09", "PF12"}:
        ax.axhline(0, color=".5", lw=0.8)
    if card["id"] in {"PF01", "PF02", "PF09", "PF12"}:
        ax.set_yscale("symlog", linthresh=LINTHRESH)
    return {
        "finite_summary_cells": count,
        "transformations": {
            "x": "log",
            "y": ax.get_yscale(),
            "linthresh": LINTHRESH if ax.get_yscale() == "symlog" else None,
        },
    }


def _coverage(ax, frame, card):
    count = 0
    if card["id"] == "PF03":
        for group in GROUPS[1:]:
            selected = _metric_rows(_group(frame, group), "candidate_gain_status")
            positive = selected[selected.status == "positive"].sort_values("snr")
            unresolved = selected[
                selected.status == "numerically_unresolved"
            ].sort_values("snr")
            if positive.empty:
                continue
            x, y = _numbers(positive, "snr"), _numbers(positive, "fraction")
            if not (np.isfinite(x) & np.isfinite(y)).any():
                continue
            ax.plot(x, y, color=GROUP_COLORS[group], label=group, lw=1.6)
            if not unresolved.empty:
                joined = positive[["step_index", "snr", "fraction"]].merge(
                    unresolved[["step_index", "fraction"]],
                    on="step_index",
                    suffixes=("_positive", "_unresolved"),
                )
                ax.fill_between(
                    joined.snr,
                    joined.fraction_positive,
                    joined.fraction_positive + joined.fraction_unresolved,
                    color=GROUP_COLORS[group],
                    alpha=0.18,
                )
            count += int((np.isfinite(x) & np.isfinite(y)).sum())
    elif card["id"] == "PF07":
        selected = _group(frame, "All")
        metrics = [
            "candidate_margin_original_status",
            "candidate_margin_combined_status",
            "candidate_margin_signed_error_status",
            "candidate_margin_projected_variation_status",
            "candidate_gain_status",
        ]
        labels = [
            "M0: original",
            "M1: combined errors",
            "M2: signed errors",
            "M3: projected variation",
            "Observed positive gain",
        ]
        for index, (metric, label) in enumerate(zip(metrics, labels)):
            rows = _metric_rows(selected, metric)
            rows = rows[rows.status == "positive"].sort_values("snr")
            if len(rows):
                ax.plot(
                    rows.snr,
                    rows.fraction,
                    color=plt.get_cmap("tab10")(index),
                    ls="--" if index == 4 else "-",
                    label=label,
                )
                count += int(np.isfinite(_numbers(rows, "fraction")).sum())
                if index < 4:
                    estimated = _metric_rows(
                        selected, metric.replace("_status", "_estimated_sign")
                    )
                    estimated = estimated[estimated.status == "positive"].sort_values(
                        "snr"
                    )
                    if len(estimated):
                        ax.plot(
                            estimated.snr,
                            estimated.fraction,
                            color=plt.get_cmap("tab10")(index),
                            ls=":",
                            lw=1,
                            alpha=0.8,
                        )
        ax.plot([], [], color=".4", ls=":", lw=1, label="Estimated signs (dotted)")
    else:
        selected = _metric_rows(
            _group(frame, card.get("group", "All")), "candidate_profile_class"
        )
        if not selected.empty:
            pivot = selected.pivot_table(
                index="snr", columns="status", values="fraction", aggfunc="first"
            ).sort_index()
            classes = [
                "increasing",
                "decreasing",
                "interior_peak",
                "flat",
                "numerically_unresolved",
            ]
            labels = [
                "Increasing",
                "Decreasing",
                "Interior peak",
                "Flat",
                "Unresolved",
            ]
            colors = ["#48997b", "#a96370", "#b89749", "#888888", "#c7c7c7"]
            ax.stackplot(
                pivot.index,
                *[
                    pivot.get(status, pd.Series(0, index=pivot.index))
                    .fillna(0)
                    .to_numpy()
                    for status in classes
                ],
                labels=labels,
                colors=colors,
                alpha=0.9,
            )
            count += int(np.isfinite(pivot.to_numpy(dtype=float)).any(axis=1).sum())
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    return {
        "finite_summary_cells": count,
        "saved_coverage_source": card["source_table"],
        "estimated_coverage_style": "Dotted lines are raw estimated signs; solid lines are resolved positive margins."
        if card["id"] == "PF07"
        else None,
        "uncertainty": "The PF03 band adds the saved unresolved fraction; it is not a confidence interval.",
    }


def _clean_rows(frame):
    if "terminal_clean_applicable" in frame:
        return frame[frame.terminal_clean_applicable.fillna(False).astype(bool)]
    return frame.iloc[:0]


def _draw(ax, card, frame, tables):
    identifier = card["id"]
    kind = card["rendering"]
    if kind == "timeseries":
        return _timeseries(ax, frame, card)
    if kind in {"gain_fraction", "margin_coverage", "profiles"}:
        return _coverage(ax, frame, card)
    if kind == "scatter":
        columns = list(card["required_columns"])
        if identifier == "PF10" and card.get("within_prompt"):
            columns = [
                "early_normalized_gain_centered",
                "subsequent_unconditional_improvement_centered",
            ]
            # Summary versions predating the final names remain explicit aliases.
            if columns[0] not in frame:
                columns = ["x_centered", "y_centered"]
            card["axes"] = {
                "x": "Within-prompt early candidate gain",
                "y": "Within-prompt subsequent improvement",
            }
        result = _scatter(
            ax,
            frame,
            columns[0],
            columns[1],
            color="terminal_sscd" in columns[2:],
            equal=card.get("equal", False),
            equality_label="Equal candidate evidence"
            if identifier == "PF04"
            else "Equal target errors",
        )
        if identifier == "PF04":
            # A shared x-axis multiplier otherwise overlaps the wrapped label
            # for late snapshots with large negative target log odds.
            for axis, column in ((ax.xaxis, columns[0]), (ax.yaxis, columns[1])):
                values = _numbers(frame, column)
                finite = values[np.isfinite(values)]
                if len(finite) and np.max(np.abs(finite)) >= 1e4:
                    axis.set_major_formatter(
                        FuncFormatter(
                            lambda value, position: (
                                f"{value:.2g}".replace("e+0", "e")
                                .replace("e+", "e")
                                .replace("e-0", "e-")
                            )
                        )
                    )
            result["tick_formatting"] = (
                "Large log-odds values use scientific notation per tick, without a shared offset."
            )
        if identifier in {"IR03", "PF10"}:
            result["trend"] = _trend(ax, frame, columns[0], columns[1])
            ax.axhline(0, color=".65", lw=0.8)
            ax.axvline(0, color=".65", lw=0.8)
        if identifier in {"IR02", "IA03"}:
            source = "initial_error_bins" if identifier == "IR02" else "sscd_bins"
            summary = tables["summaries"].get(source, pd.DataFrame())
            if identifier == "IA03" and "metric" in summary:
                summary = summary[summary.metric == "injection_relative_mismatch"]
            xname = (
                "conditional_error_median" if identifier == "IR02" else "sscd_median"
            )
            if xname in summary and "median" in summary:
                _curve(ax, summary, xname, "Median", color="black")
                result["fixed_bin_summary_source"] = "summaries/" + source + ".parquet"
            else:
                result["fixed_bin_summary_status"] = "unavailable_saved_summary"
        g = float(tables["manifest"].get("config", {}).get("guidance_scale", 1))
        if identifier == "IA01":
            ax.scatter(
                [g],
                [0],
                marker="x",
                s=38,
                color="black",
                label="Limiting aligned injection",
            )
        if identifier == "IA02":
            limits = ax.get_xlim()
            ax.plot(limits, limits, color=".35", ls="--", label="Unchanged coordinate")
            ax.plot(
                limits,
                np.array(limits) * g,
                color=".35",
                ls=":",
                label="Zero unconditional coordinate",
            )
            ax.set_xlim(limits)
        if identifier == "TR01" and card.get("clean_update"):
            tolerance = (
                tables["manifest"].get("config", {}).get("target_error_tolerance")
            )
            dimension = _numbers(frame, "latent_dimension")
            dimensions = np.unique(dimension[np.isfinite(dimension)])
            if tolerance is not None and g >= 1 and len(dimensions) == 1:
                normalized = float(tolerance) / math.sqrt(dimensions[0])
                ax.plot(
                    [0, normalized],
                    [normalized, 0],
                    color=".35",
                    ls="--",
                    label="Supplied tolerance",
                )
                result["independent_tolerance_rmse"] = normalized
            elif tolerance is not None:
                result["omitted_tolerance_reason"] = (
                    "Sufficient A+B region requires g>=1 and one known latent dimension."
                )
        return result
    if kind in {"rank_ecdf", "baseline_ecdf", "bank_ecdf", "terminal_ecdf"}:
        count = 0
        if kind == "rank_ecdf":
            for branch, color in (
                ("conditional", "#55447e"),
                ("unconditional", "#aa8b32"),
            ):
                count += _ecdf(
                    ax,
                    frame,
                    f"initial_{branch}_target_rank",
                    branch.capitalize(),
                    color=color,
                )
        elif kind in {"baseline_ecdf", "bank_ecdf"}:
            keys = [key for key in ("run_id", "seed") if key in frame]
            unique = frame.drop_duplicates(keys) if keys else frame
            count += _ecdf(
                ax,
                unique,
                "unconditional_center_rmse",
                "Initial unconditional",
                color="#55447e",
                weighted=False,
            )
            if kind == "bank_ecdf":
                count += _ecdf(
                    ax,
                    tables["bank_geometry"],
                    "candidate_atom_center_distance_rmse",
                    "Distinct candidate atoms",
                    color="#aa8b32",
                    weighted=False,
                )
        else:
            for column, label in (
                ("terminal_observed_bound_error_ratio", "Observed-geometry bound"),
                ("terminal_candidate_bound_error_ratio", "Candidate-theorem bound"),
            ):
                count += _ecdf(ax, frame, column, label)
            ax.set_xscale("symlog", linthresh=LINTHRESH)
        return {"finite_observations": count}
    if kind == "reference_snr":
        rows = []
        for snr, values in frame.groupby("analytical_snr"):
            q25, median, q75 = _quantiles(
                _numbers(values, "candidate_reference_movement_rmse"),
                np.ones(len(values)),
            )
            measured = _numbers(values, "candidate_reference_movement_rmse")
            measured = measured[np.isfinite(measured)]
            rows.append(
                {
                    "analytical_snr": snr,
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "minimum": float(measured.min()) if len(measured) else np.nan,
                    "maximum": float(measured.max()) if len(measured) else np.nan,
                }
            )
        count = (
            _curve(
                ax,
                pd.DataFrame(rows),
                "analytical_snr",
                "Analytical reference",
                color="#55447e",
            )
            if rows
            else 0
        )
        ax.set_xscale("log")
        if len(frame):
            ax.axvline(
                float(frame.analytical_snr.max()),
                color=".4",
                ls=":",
                label="Actual initial SNR",
            )
        return {"finite_summary_cells": count}
    if kind == "dose":
        count = 0
        for group in GROUPS[1:]:
            count += _curve(
                ax, _group(frame, group), "lambda", group, color=GROUP_COLORS[group]
            )
        g = float(tables["manifest"]["config"]["guidance_scale"])
        ax.axvline(0, color=".55", ls=":", lw=0.8, label="Matched unconditional")
        if g != 0 and 0 <= 1 / g <= 1:
            ax.axvline(1 / g, color=".35", ls="--", lw=0.8, label="Matched conditional")
        ax.axvline(1, color=".2", ls=":", lw=0.8, label="Actual CFG")
        ax.axhline(0, color=".6", lw=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_yscale("symlog", linthresh=LINTHRESH)
        return {
            "finite_summary_cells": count,
            "transformations": {"y": "symlog", "linthresh": LINTHRESH},
        }
    if kind in {"joint_tolerance", "terminal_tolerance"}:
        count = 0
        if kind == "joint_tolerance":
            groups = [card.get("group", "All")]
            for group in groups:
                selected = _group(frame, group).sort_values("tolerance")
                valid = np.isfinite(_numbers(selected, "tolerance")) & np.isfinite(
                    _numbers(selected, "fraction")
                )
                if valid.any():
                    ax.plot(
                        selected.tolerance,
                        selected.fraction,
                        label=group,
                        color=GROUP_COLORS[group],
                    )
                    count += int(valid.sum())
        else:
            selected = _group(frame, "All")
            for metric, rows in selected.groupby("metric"):
                rows = rows.sort_values("tolerance")
                label = {
                    "terminal_observed_error_rmse": "Observed endpoints",
                    "terminal_observed_bound_rmse": "Observed-geometry bound",
                    "terminal_candidate_bound_rmse": "Candidate-theorem bound",
                }.get(metric, metric.replace("_", " "))
                valid = np.isfinite(_numbers(rows, "tolerance")) & np.isfinite(
                    _numbers(rows, "fraction")
                )
                if valid.any():
                    ax.plot(rows.tolerance, rows.fraction, label=label)
                    count += int(valid.sum())
        ax.set_ylim(0, 1.02)
        return {"finite_summary_cells": count}
    if kind == "recovery":
        x, y = (
            _numbers(frame, "conditional_first_step"),
            _numbers(frame, "unconditional_first_step"),
        )
        cx = frame.conditional_status.eq("right_censored").to_numpy()
        cy = frame.unconditional_status.eq("right_censored").to_numpy()
        mx, my = ~np.isfinite(x) & ~cx, ~np.isfinite(y) & ~cy
        steps = _numbers(tables["trajectory"], "step_index")
        last = int(np.nanmax(steps)) if np.isfinite(steps).any() else 1
        spacing = max(2, 0.15 * (last + 1))
        never, missing = last + spacing, last + 2 * spacing
        # These coordinates are display categories, never saved recovery times.
        display_x = np.where(cx, never, np.where(mx, missing, x))
        display_y = np.where(cy, never, np.where(my, missing, y))
        ax.scatter(
            display_x,
            display_y,
            c=np.where(cx | cy | mx | my, 0.7, 0.3),
            cmap="Greys",
            vmin=0,
            vmax=1,
            s=12,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
        )
        ax.plot([0, last], [0, last], color=".45", ls="--", label="Same recovery step")
        limit = missing if (mx | my).any() else never
        ax.axvspan(last + 0.7, limit + 1, color=".95", zorder=0)
        ax.axhspan(last + 0.7, limit + 1, color=".95", zorder=0)
        ticks = sorted(
            set([0, last // 2, last, never] + ([missing] if (mx | my).any() else []))
        )
        labels = [
            "Never"
            if tick == never
            else "Missing"
            if tick == missing
            else str(int(tick))
            for tick in ticks
        ]
        ax.set_xticks(ticks, labels, fontsize=10)
        ax.set_yticks(ticks, labels, fontsize=10)
        ax.set_xlim(-1, limit + 1)
        ax.set_ylim(-1, limit + 1)
        return {
            "finite_observations": int((np.isfinite(x) & np.isfinite(y)).sum()),
            "rendered_observations": len(frame),
            "conditional_censored": int(cx.sum()),
            "unconditional_censored": int(cy.sum()),
            "both_censored": int((cx & cy).sum()),
            "conditional_unavailable": int(mx.sum()),
            "unconditional_unavailable": int(my.sum()),
            "censoring_display": "Separate Never and Missing boundary categories; no imputation of a finite recovery time.",
        }
    if kind == "heatmap":
        frame = _group(frame, "All")
        matrix = np.full((8, 10), np.nan)
        counts = np.zeros((8, 10), dtype=int)
        for row in frame.to_dict("records"):
            i, j = int(row["regime_bin"]), int(row["progress_bin"])
            if 0 <= i < 8 and 0 <= j < 10:
                matrix[i, j] = row["positive_fraction"]
                counts[i, j] = int(
                    row.get(
                        "denominator_count",
                        row.get("count", row.get("n_observations", 0)),
                    )
                )
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")
        artist = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap=cmap,
            interpolation="none",
        )
        ax.figure.colorbar(artist, ax=ax, pad=0.025).set_label("Positive fraction")
        for i in range(8):
            for j in range(10):
                if not np.isfinite(matrix[i, j]):
                    ax.text(
                        j, i, "—", ha="center", va="center", fontsize=7, color=".45"
                    )
                elif counts[i, j] < 20:
                    ax.text(
                        j,
                        i,
                        str(counts[i, j]),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if matrix[i, j] > 0.6 else "white",
                    )
        ax.set_xticks([0, 4, 9], ["0–.1", ".4–.5", ".9–1"])
        ax.set_yticks(
            range(8),
            ["<−10", "−10:−5", "−5:−2", "−2:0", "0:2", "2:5", "5:10", ">10"],
            fontsize=9,
        )
        return {
            "finite_summary_cells": int(np.isfinite(matrix).sum()),
            "cell_counts": counts.tolist(),
            "empty_cell_count": int((~np.isfinite(matrix)).sum()),
            "low_count_definition": "Fewer than 20 observations; count displayed. Empty cells are marked with a dash.",
        }
    if kind == "qa":
        count = 0
        metrics = {
            "independent_replay_rmse": "Independent vector residual",
            "update_rounding_sensitivity_rmse": "Source-precision envelope",
        }
        for metric, label in metrics.items():
            rows = _metric_rows(_group(frame, "All"), metric)
            count += _curve(ax, rows, "step_index", label, band=False)
        return {
            "finite_summary_cells": count,
            "qa_interpretation": "Independent scheduler replay only; subtraction-built counterfactual residuals do not independently verify the scheduler.",
        }
    raise TheoryError(f"Unimplemented candidate design: {card['variant_id']}")


def _availability(card, frame, tables):
    if card.get("clean_update") and _clean_rows(tables["endpoint"]).empty:
        endpoint = tables["endpoint"]
        status = "not_applicable"
        reason = "No endpoint satisfies the saved clean-update applicability checks."
        if "terminal_clean_status" in endpoint:
            reason += " " + "; ".join(
                sorted(set(endpoint.terminal_clean_status.dropna().astype(str)))
            )
        return status, reason
    if card["id"] == "PF07" and not tables["manifest"].get("complete", False):
        return (
            "unavailable",
            "The separately resumable integration stage has not completed; endpoint designs remain available.",
        )
    if card["id"] == "MD01":
        replay = _metric_rows(frame, "independent_replay_rmse")
        if not np.isfinite(_numbers(replay, "median")).any():
            return (
                "unavailable",
                "No independent deterministic scheduler vector replay is saved. Recovering a DDPM innovation from the measured endpoint or constructing x_cf by subtraction is not independent replay.",
            )
    if frame.empty:
        return "unavailable", "No saved rows for this prespecified subset."
    missing = [column for column in card["required_columns"] if column not in frame]
    if missing:
        return "unavailable", "Missing saved scalar columns: " + ", ".join(missing)
    return "available", "Saved numerical source is available."


def _displayed_summary_ranges(card, frame, tables):
    """Ranges of the plotted saved summary cells; never a pooled effect."""
    kind, identifier = card["rendering"], card["id"]
    source, xcol, ycol = card["source_table"], "snr", "median"
    groups = [card.get("group", "All")]
    if kind == "timeseries":
        groups = [card["group"]] if "group" in card else list(GROUPS[1:])
        if "group" not in frame or not any(
            len(_group(frame, group)) for group in groups
        ):
            groups = ["All"]
    elif kind == "qa":
        xcol = "step_index"
    elif kind == "dose":
        xcol, groups = "lambda", list(GROUPS[1:])
    elif kind in {"joint_tolerance", "terminal_tolerance"}:
        xcol, ycol = "tolerance", "fraction"
    elif kind in {"gain_fraction", "margin_coverage", "profiles"}:
        ycol = "fraction"
        if kind == "gain_fraction":
            groups = list(GROUPS[1:])
            frame = frame[frame.status == "positive"]
        elif kind == "margin_coverage":
            frame = frame[frame.status == "positive"]
        else:
            frame = frame[frame.status != "not_applicable"]
    elif kind == "reference_snr":
        source, xcol = "summaries/reference_sweep", "analytical_snr"
        frame = _source(tables, source)
    elif kind == "rank_ecdf":
        source, xcol, ycol = "summaries/retrieval_ecdf", "rank", "fraction"
        frame = _group(_source(tables, source), card.get("group", "All"))
    elif kind == "heatmap":
        xcol, ycol = "progress_bin", "positive_fraction"
        frame = frame[frame.regime_bin >= 0]
    elif identifier in {"IR02", "IA03"}:
        source = "summaries/" + (
            "initial_error_bins" if identifier == "IR02" else "sscd_bins"
        )
        xcol = "conditional_error_median" if identifier == "IR02" else "sscd_median"
        frame = _source(tables, source)
        if identifier == "IA03" and "metric" in frame:
            frame = frame[frame.metric == "injection_relative_mismatch"]
    else:
        return []
    if frame.empty or xcol not in frame or ycol not in frame:
        return []
    records = []
    for group in groups:
        subset = _group(frame, group)
        keys = [
            key for key in ("metric", "status", "branch", "regime_bin") if key in subset
        ]
        partitions = (
            subset.groupby(keys, dropna=False, sort=True) if keys else [((), subset)]
        )
        for identity, rows in partitions:
            if not isinstance(identity, tuple):
                identity = (identity,)
            x, y = _numbers(rows, xcol), _numbers(rows, ycol)
            valid = np.isfinite(x) & np.isfinite(y)
            if xcol in {"snr", "analytical_snr"}:
                valid &= x > 0
            item = {
                "group": group,
                "metric": identifier,
                **dict(zip(keys, identity)),
                "source_table": source,
                "x_column": xcol,
                "y_column": ycol,
                "scope": "range of displayed saved summary cells; not a pooled effect",
                "finite_summary_cell_count": int(valid.sum()),
                "x_min": float(x[valid].min()) if valid.any() else None,
                "x_max": float(x[valid].max()) if valid.any() else None,
                "y_min": float(y[valid].min()) if valid.any() else None,
                "y_max": float(y[valid].max()) if valid.any() else None,
            }
            for column, output, operation in (
                ("q25", "iqr_lower_min", np.min),
                ("q75", "iqr_upper_max", np.max),
                ("minimum", "all_data_minimum", np.min),
                ("maximum", "all_data_maximum", np.max),
            ):
                values = _numbers(rows, column)
                selected = values[valid & np.isfinite(values)]
                item[output] = float(operation(selected)) if len(selected) else None
            records.append(item)
    return records


def _nonfinite_status(card, tables):
    trajectory = tables.get("trajectory", pd.DataFrame())
    if card["id"] in {"PF02", "PF05", "PF10"}:
        column = (
            "candidate_current_alignment_status"
            if card["id"] == "PF05"
            else "candidate_direction_normalization_status"
        )
        statuses = set(
            trajectory.get(column, pd.Series(dtype=str)).dropna().astype(str)
        )
        normalizations = set(
            trajectory.get(
                "candidate_direction_normalization_status", pd.Series(dtype=str)
            )
            .dropna()
            .astype(str)
        )
        if normalizations and normalizations <= {"zero_displacement", "not_applicable"}:
            return (
                "not_applicable",
                "All saved directions are undefined because displacement is zero or the matched transition is inapplicable.",
            )
        if statuses and any("unresolved" in value for value in statuses):
            return (
                "numerically_unresolved",
                "Saved direction or alignment arithmetic is unresolved; no direction-normalized values are invented.",
            )
    return "unavailable", "Saved subset has no finite values for this design."


def _manifest_reference(value, key):
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    reference = {
        "representation": "authoritative analysis manifest field reference",
        "manifest_key": key,
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "hash_algorithm": "SHA-256",
        "serialization": "UTF-8 JSON; sorted keys; compact comma/colon separators; ensure_ascii=true; allow_nan=false",
    }
    if isinstance(value, (dict, list)):
        reference["entry_count"] = len(value)
    return reference


def _provenance(manifest, manifest_sha256=None):
    """Retain identities and verifiable links without repeating large inventories."""
    provenance = copy.deepcopy(manifest.get("provenance", {}))
    for key in (
        "config",
        "schema_version",
        "formula_version",
        "base_bundle",
        "base_analysis_hash",
        "endpoint_hash",
        "integration_hash",
        "source_root",
        "source_code",
        "center_metadata",
        "scheduler_adapter",
        "manuscript_sha256",
        "population_policy",
        "endpoint_policy",
        "integration_policy",
        "smoke_record_limit",
    ):
        if key in manifest:
            provenance[key] = copy.deepcopy(manifest[key])
    for key in ("source_metadata_files", "source_marker_inventory"):
        if key in manifest:
            provenance[key] = _manifest_reference(manifest[key], key)
    if "support_metadata" in manifest:
        support = manifest["support_metadata"]
        compact = {
            key: copy.deepcopy(value)
            for key, value in support.items()
            if key not in {"aliases", "atom_ids", "weights", "missing_candidates"}
        }
        compact["full_inventory_reference"] = _manifest_reference(
            support, "support_metadata"
        )
        for key in ("aliases", "atom_ids", "weights", "missing_candidates"):
            if key in support:
                compact[key + "_reference"] = _manifest_reference(
                    support[key], "support_metadata." + key
                )
        provenance["support_metadata"] = compact
    provenance["analysis_manifest"] = {
        "path": "../analysis_manifest.json",
        "sha256": manifest_sha256,
    }
    return provenance


def _figure(card, tables, stage):
    card["axes"].update(CANDIDATE_AXIS_LABELS.get(card["id"], {}))
    if card["axes"]["x"] == "SNR":
        card["axes"]["x"] = r"$\mathrm{SNR}_t$"
    frame = _source(tables, card["source_table"])
    if "step_index" in card and "step_index" in frame:
        frame = frame[frame.step_index == card["step_index"]]
    if "tolerance" in card and "tolerance" in frame:
        frame = frame[
            np.isclose(
                _numbers(frame, "tolerance"), card["tolerance"], atol=1e-12, rtol=0
            )
        ]
    if "group" in card:
        frame = _group(frame, card["group"])
    elif card["id"] == "PF11":
        frame = _group(frame, "All")
    if card["id"] == "IR03" or (card["id"] == "PF10" and card.get("within_prompt")):
        if "within_prompt_status" in frame:
            card["excluded_no_seed_variation_count"] = int(
                (frame.within_prompt_status != "usable_seed_variation").sum()
            )
            frame = frame[frame.within_prompt_status == "usable_seed_variation"]
    if card.get("clean_update") and card["source_table"] in {
        "endpoint",
        "summaries/terminal_ratios",
    }:
        frame = _clean_rows(frame)
    if "metric" in frame:
        metrics = card.get("metrics", [])
        if card["id"] == "MD01":
            metrics = ["independent_replay_rmse", "update_rounding_sensitivity_rmse"]
        elif card["id"] == "PF03":
            metrics = ["candidate_gain_status"]
        elif card["id"] == "PF07":
            metrics = ["candidate_gain_status"] + [
                f"candidate_margin_{stem}_{suffix}"
                for stem in (
                    "original",
                    "combined",
                    "signed_error",
                    "projected_variation",
                )
                for suffix in ("status", "estimated_sign")
            ]
        elif card["id"] == "PF08":
            metrics = ["candidate_profile_class"]
        if metrics:
            aliases = {
                alias for metric in metrics for alias in ALIASES.get(metric, (metric,))
            }
            frame = frame[frame.metric.isin(aliases)]
    card["counts"] = _counts(frame)
    if "group" in frame:
        card["group_counts"] = {
            str(group): _counts(rows)
            for group, rows in frame.groupby("group", sort=True)
        }
    if card["id"] == "IR04":
        card["retrieval_contract"] = {
            "bank_sizes": sorted(
                set(
                    _numbers(frame, "initial_retrieval_bank_size")[
                        np.isfinite(_numbers(frame, "initial_retrieval_bank_size"))
                    ].tolist()
                )
            ),
            "rank": "One plus the number of strictly closer distinct candidate atoms; Euclidean clean-estimate retrieval.",
            "tie_count_includes_target": True,
            "conditional_tied_rows": int(
                (_numbers(frame, "initial_conditional_target_tie_count") > 1).sum()
            ),
            "unconditional_tied_rows": int(
                (_numbers(frame, "initial_unconditional_target_tie_count") > 1).sum()
            ),
        }
    card["population_counts"] = _counts(tables["initial"])
    card["data_sources"] = _relative_sources(tables["manifest"], card["source_table"])
    card["additional_population_counts"] = {}
    for source in card.get("additional_sources", {}):
        card["data_sources"] += _relative_sources(tables["manifest"], source)
        card["additional_population_counts"][source] = _counts(_source(tables, source))
    card["data_hashes"] = {
        name: tables["manifest"]["numerical_files"][name]
        for name in card["data_sources"]
    }
    if card["id"] in {"IR02", "IA03"}:
        auxiliary = "summaries/" + (
            "initial_error_bins" if card["id"] == "IR02" else "sscd_bins"
        )
        for name in _relative_sources(tables["manifest"], auxiliary):
            card["data_sources"].append(name)
            card["data_hashes"][name] = tables["manifest"]["numerical_files"][name]
    card["provenance"] = _provenance(
        tables["manifest"], tables.get("analysis_manifest_sha256")
    )
    card["analysis_hash"] = tables["manifest"].get("analysis_hash")
    card["style_version"] = STYLE_VERSION
    card["renderer_source_sha256"] = digest_file(Path(__file__))
    card["descriptive_dependence"] = (
        "Prompts sharing a target and reused seed IDs are dependent; IQRs describe dispersion, with no independent-row confidence interval."
    )
    card["numeric_status"], card["status_reason"] = _availability(card, frame, tables)
    card["files"] = {}
    if card["numeric_status"] == "available":
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots()
            ax.set_axisbelow(True)
            ax.grid(True, color=".9", lw=0.6)
            try:
                stats = _draw(ax, card, frame, tables)
                finite = stats.get(
                    "rendered_observations",
                    stats.get(
                        "finite_observations", stats.get("finite_summary_cells", 0)
                    ),
                )
                card["descriptive_statistics"] = stats
                if finite:
                    ax.set_xlabel(textwrap.fill(card["axes"]["x"], 32))
                    ax.set_ylabel(textwrap.fill(card["axes"]["y"], 34))
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        # Keep 10-point legends legible without hiding observations.
                        ax.legend(
                            handles,
                            labels,
                            loc="upper center",
                            bbox_to_anchor=(0.5, -0.25),
                            frameon=False,
                            ncol=1,
                        )
                    for suffix in ("png", "pdf"):
                        relative = card["artifact_stem"] + "." + suffix
                        path = stage / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        kwargs = (
                            {"metadata": {"CreationDate": None, "ModDate": None}}
                            if suffix == "pdf"
                            else {}
                        )
                        fig.savefig(path, bbox_inches="tight", **kwargs)
                        card["files"][suffix] = relative
                    card["axis_limits"] = {
                        "x": list(ax.get_xlim()),
                        "y": list(ax.get_ylim()),
                    }
                    card["display_transformations"] = {
                        "x": ax.get_xscale(),
                        "y": ax.get_yscale(),
                        "symlog_linthresh": LINTHRESH
                        if "symlog" in {ax.get_xscale(), ax.get_yscale()}
                        else None,
                    }
                    description = stats.get("descriptive_y")
                    if description:
                        card["observed_result"] = (
                            f"{finite:,} finite observations; plotted y median {description['median']:.4g}, with {description['negative']:,} negative, {description['zero']:,} zero and {description['positive']:,} positive y values."
                        )
                    else:
                        column = (
                            "fraction"
                            if "fraction" in frame
                            else "positive_fraction"
                            if "positive_fraction" in frame
                            else "median"
                        )
                        measured = _numbers(frame, column)
                        measured = measured[np.isfinite(measured)]
                        if len(measured):
                            card["observed_result"] = (
                                f"{finite:,} finite saved cells; displayed {column} range {measured.min():.4g} to {measured.max():.4g}. Full counts and status denominators are linked."
                            )
                        else:
                            card["observed_result"] = (
                                f"{finite:,} finite observations; all finite signs are retained. Counts and measurement statuses are linked."
                            )
                else:
                    card["numeric_status"], card["status_reason"] = _nonfinite_status(
                        card, tables
                    )
            finally:
                plt.close(fig)
    if card["numeric_status"] != "available":
        card["observed_result"] = card["status_reason"]
    card["displayed_summary_series_ranges"] = (
        _displayed_summary_ranges(card, frame, tables)
        if card["numeric_status"] == "available"
        else None
    )
    card["files"]["json"] = card["artifact_stem"] + ".json"
    _write_json(stage / card["files"]["json"], card)
    return card


def _gallery(cards, manifest):
    def escape(value):
        return html.escape(str(value))

    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Theory candidate gallery</title>",
        "<style>body{font-family:Georgia,serif;background:#fafafa;color:#222;margin:2rem auto;max-width:1500px;padding:0 1rem}h1{font-size:1.7rem}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}.card{background:white;border:1px solid #ddd;padding:1rem;min-width:0}.card img{width:100%;height:360px;object-fit:contain}.tag{display:inline-block;background:#eee;padding:.2rem .4rem;margin:.2rem;font:12px sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.9rem}details{margin:.5rem 0}a{color:#493878}.missing{height:150px;display:grid;place-content:center;background:#f1f1f1;padding:1rem}nav a{margin-right:1rem}p{line-height:1.45}code{overflow-wrap:anywhere}</style>",
        "<h1>Theory-linked candidate discovery gallery</h1>",
        "<p>" + escape(json.dumps(manifest.get("config", {}), sort_keys=True)) + "</p>",
        "<p>34 prescribed base designs, with fixed snapshot, group and tolerance variants. Each figure is a separate axis. The gallery preserves adverse observations and unavailable designs. Descriptive bands are interquartile ranges.</p>",
        "<p><a href='comparison_summary.csv'>Comparison scorecard</a> · <a href='registry.json'>Design registry</a> · <a href='analysis_manifest.json'>Numerical provenance</a> · <a href='numerical_audit.json'>Numerical audit</a></p>",
        "<nav>"
        + " ".join(
            f"<a href='#{family}'>{family}</a>"
            for family in [
                "IR",
                "UB",
                "IA",
                "MD",
                "PF",
                "TS",
                "TR",
                "qa",
                "last_prediction_geometry",
            ]
        )
        + "</nav>",
    ]
    for family in [
        "IR",
        "UB",
        "IA",
        "MD",
        "PF",
        "TS",
        "TR",
        "qa",
        "last_prediction_geometry",
    ]:
        selected = [
            card for card in cards if card["artifact_stem"].split("/")[0] == family
        ]
        if not selected:
            continue
        parts.append(f"<h2 id='{family}'>{escape(family)}</h2><div class='cards'>")
        for card in selected:
            parts.append(
                "<article class='card'><h3>" + escape(card["variant_id"]) + "</h3>"
            )
            parts.append(
                "<span class='tag'>"
                + escape(card["evidence_tag"])
                + "</span><span class='tag'>"
                + escape(card["numeric_status"])
                + "</span>"
            )
            if "png" in card["files"]:
                parts.append(
                    f"<a href='{escape(card['files']['pdf'])}'><img loading='lazy' src='{escape(card['files']['png'])}' alt='{escape(card['question'])}'></a>"
                )
            else:
                parts.append(
                    "<div class='missing'>" + escape(card["status_reason"]) + "</div>"
                )
            parts.append(
                "<p><strong>"
                + escape(card["question"])
                + "</strong></p><pre>"
                + escape(card["formula"])
                + "</pre>"
            )
            parts.append(
                "<p>"
                + escape(card["observed_result"])
                + "</p><p>"
                + escape(card["result"])
                + " · <code>"
                + escape(card["manuscript_label"])
                + "</code></p>"
            )
            parts.append(
                "<details><summary>Interpretation, counts and provenance</summary><p>"
                + escape(card["affirmative_pattern_meaning"])
                + "</p><p>Does not establish: "
                + escape(card["does_not_establish"])
                + "</p><p>"
                + escape(card["population"])
                + "</p><p>"
                + escape(card["weighting"])
                + "</p><pre>"
                + escape(
                    json.dumps(
                        _json(
                            {
                                "plotted_source_counts": card["counts"],
                                "outcome_group_counts": card.get("group_counts", {}),
                                "additional_populations": card.get(
                                    "additional_population_counts", {}
                                ),
                                "full_evaluation_population": card["population_counts"],
                                "retrieval_contract": card.get("retrieval_contract"),
                                "descriptive_statistics": card.get(
                                    "descriptive_statistics", {}
                                ),
                            }
                        ),
                        indent=2,
                    )
                )
                + "</pre>"
            )
            if card.get("snapshot_schedule"):
                parts.append(
                    "<pre>Snapshot schedule: "
                    + escape(json.dumps(card["snapshot_schedule"]))
                    + "</pre>"
                )
            parts.append(
                "<p>Source tables: "
                + " · ".join(
                    f"<a href='{escape(name)}'>{escape(name)}</a>"
                    for name in card["data_sources"]
                )
                + "</p></details>"
            )
            parts.append(
                f"<p><a href='{escape(card['files']['json'])}'>Complete formula and provenance sidecar</a></p></article>"
            )
        parts.append("</div>")
    parts.append("</html>")
    return "\n".join(parts)


def _comparison(cards, manifest):
    rows = []
    for card in cards:
        stats = card.get("descriptive_statistics", {})
        ranges = card.get("displayed_summary_series_ranges")
        minima = [
            series["y_min"] for series in ranges or [] if series["y_min"] is not None
        ]
        maxima = [
            series["y_max"] for series in ranges or [] if series["y_max"] is not None
        ]
        rows.append(
            {
                "analysis_hash": manifest.get("analysis_hash"),
                **manifest.get("config", {}),
                "candidate_id": card["id"],
                "variant_id": card["variant_id"],
                "family": card["family"],
                "evidence_tag": card["evidence_tag"],
                "numeric_status": card["numeric_status"],
                "status_reason": card["status_reason"],
                "source_rows": card["counts"]["source_rows"],
                "represented_samples": card["counts"].get("samples"),
                "represented_prompts": card["counts"].get("prompts"),
                "finite_observations": stats.get("finite_observations"),
                "finite_summary_cells": stats.get("finite_summary_cells"),
                "rendered_observations": stats.get("rendered_observations"),
                "negative_y_count": stats.get("descriptive_y", {}).get("negative"),
                "positive_y_count": stats.get("descriptive_y", {}).get("positive"),
                "median_y": stats.get("descriptive_y", {}).get("median"),
                "displayed_summary_y_min": min(minima) if minima else None,
                "displayed_summary_y_max": max(maxima) if maxima else None,
                "displayed_summary_range_scope": "saved plotted summary-cell range; not a pooled effect"
                if ranges
                else None,
                "displayed_summary_series_ranges_json": json.dumps(
                    _json(ranges), sort_keys=True
                )
                if ranges
                else None,
                "weighted_descriptive_slope": stats.get("trend", {}).get(
                    "weighted_descriptive_slope"
                ),
                "question": card["question"],
                "estimand": card["formula"],
                "scope": card["scope"],
                "observed_result": card["observed_result"],
                "directness": "Direct measured local effect"
                if card["id"] in {"PF01", "PF02", "PF03", "PF04", "PF06"}
                else "See registered scientific question",
                "interpretability": card["affirmative_pattern_meaning"],
                "numerical_reliability": card["status_reason"],
                "cross_configuration_consistency": "Compare the same estimand in the separately named cross-configuration gallery; no pooled reference laws.",
                "sidecar": card["files"]["json"],
                "png": card["files"].get("png"),
            }
        )
    return pd.DataFrame(rows)


def _presentation_metadata(tables):
    source = "summaries/joint_tolerance.parquet"
    summary = tables.get("summaries", {}).get("joint_tolerance", pd.DataFrame())
    values = _numbers(summary, "tolerance")
    return {
        "joint_tolerance_grid": {
            "values": sorted(set(values[np.isfinite(values)].tolist())),
            "source_table": source,
            "source_sha256": tables["manifest"].get("numerical_files", {}).get(source),
            "units": "latent L2 target error / sqrt(d)",
            "selection": "All finite unique tolerance values saved in the source table, in ascending order.",
        },
        "population_descriptions": copy.deepcopy(POPULATION_DESCRIPTIONS),
        "UB04_weighting": BASELINE_SWEEP_WEIGHTING,
        "note": "Presentation metadata supplements the immutable numerical registry; numerical recipes and saved source files are unchanged.",
    }


def render_candidates(bundle):
    """Render every available candidate and an offline gallery from saved data."""
    bundle = Path(bundle).resolve()
    manifest = validate_candidates(bundle)
    tables = read_tables(bundle, include_dose=False, include_controls=False)
    tables["analysis_manifest_sha256"] = digest_file(bundle / "analysis_manifest.json")
    registry = read_object(bundle / "registry.json")
    cards, snapshots = _variant_cards(registry, tables)
    previous = (
        read_object(bundle / FIGURE_MANIFEST)
        if (bundle / FIGURE_MANIFEST).exists()
        else {}
    )
    owned = set(previous.get("files", {}))
    for name in owned:
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or name in manifest["numerical_files"]
        ):
            raise TheoryError(f"Unsafe managed candidate file: {name}")
    with tempfile.TemporaryDirectory(
        prefix=".candidate-render-", dir=bundle
    ) as temporary:
        stage = Path(temporary)
        rendered = [_figure(card, tables, stage) for card in cards]
        (stage / "index.html").write_text(_gallery(rendered, manifest))
        _comparison(rendered, manifest).to_csv(
            stage / "comparison_summary.csv", index=False
        )
        report = {
            "analysis_hash": manifest.get("analysis_hash"),
            "style_version": STYLE_VERSION,
            "base_design_count": 34,
            "variant_count": len(rendered),
            "fixed_snapshots": snapshots,
            "endpoint_complete": manifest.get("endpoint_complete"),
            "integration_status": manifest.get("integration_status"),
            "population_counts": _counts(tables["initial"]),
            "candidates": [
                {
                    "candidate_id": card["id"],
                    "variant_id": card["variant_id"],
                    "numeric_status": card["numeric_status"],
                    "reason": card["status_reason"],
                    "counts": card["counts"],
                }
                for card in rendered
            ],
            "rendering_contract": "Saved scalar tables only. No raw tensors, model loaders, posterior evaluations, center fitting or quadrature. All finite observations and full signed axis ranges retained.",
            "measurement_audit": manifest.get(
                "numerical_audit", manifest.get("audit", {})
            ),
        }
        # A producer-owned numerical audit is immutable; enrich a separate figure report.
        report_name = (
            "candidate_render_audit.json"
            if "numerical_audit.json" in manifest["numerical_files"]
            else "numerical_audit.json"
        )
        _write_json(stage / report_name, report)
        files = {
            path.relative_to(stage).as_posix(): digest_file(path)
            for path in stage.rglob("*")
            if path.is_file()
        }
        collisions = [
            name for name in files if (bundle / name).exists() and name not in owned
        ]
        if collisions:
            raise TheoryError(
                "Refusing to overwrite candidate files without renderer ownership: "
                + ", ".join(collisions)
            )
        for name in sorted(files):
            target = bundle / name
            if target.is_symlink() or not target.resolve().is_relative_to(bundle):
                raise TheoryError(f"Unsafe candidate publication path: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, target)
        # Archive only this renderer's superseded files, never unrelated figures.
        obsolete = sorted(owned - set(files))
        for name in obsolete:
            path = bundle / name
            if path.is_file() and not path.is_symlink():
                destination = (
                    bundle
                    / "archive"
                    / previous.get("style_version", "previous")
                    / name
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if digest_file(destination) != digest_file(path):
                        raise TheoryError(
                            f"Conflicting managed candidate archive: {destination}"
                        )
                    path.unlink()
                else:
                    os.replace(path, destination)
        result = {
            "analysis_hash": manifest.get("analysis_hash"),
            "style_version": STYLE_VERSION,
            "base_design_count": 34,
            "variant_count": len(rendered),
            "renderer_source_sha256": digest_file(Path(__file__)),
            "presentation_metadata": _presentation_metadata(tables),
            "files": files,
            "candidates": [
                {
                    "candidate_id": card["id"],
                    "variant_id": card["variant_id"],
                    "numeric_status": card["numeric_status"],
                    "files": card["files"],
                }
                for card in rendered
            ],
        }
        _write_json(bundle / FIGURE_MANIFEST, result)
    return result


def render_candidate_index(bundles, output_dir):
    """Separate cross-configuration gallery and like-for-like comparison table."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Cross-configuration theory candidates</title><style>body{font-family:Georgia,serif;max-width:1200px;margin:2rem auto;padding:1rem}table{border-collapse:collapse;width:100%}td,th{padding:.7rem;border:1px solid #ddd;text-align:left}a{color:#493878}</style><h1>Cross-configuration theory candidate discovery</h1><p>The same registered estimands are shown separately for each model and sampler. Candidate reference laws and latent scales are never pooled. No automatic figure selection or favorable-sign ranking is performed.</p><p><a href='comparison_summary.csv'>Complete comparison scorecard</a></p><table><tr><th>Configuration</th><th>Gallery</th><th>Available / registered variants</th></tr>"
    ]
    comparisons, indexes = [], []
    for bundle in sorted(map(lambda value: Path(value).resolve(), bundles)):
        manifest = validate_candidates(bundle)
        rendered = read_object(bundle / FIGURE_MANIFEST)
        for name, expected in rendered.get("files", {}).items():
            if digest_file(bundle / name) != expected:
                raise TheoryError(
                    f"Stale candidate figure in cross-configuration gallery: {bundle / name}"
                )
        relative = os.path.relpath(bundle / "index.html", output_dir)
        config = json.dumps(manifest["config"], sort_keys=True)
        available = sum(
            card["numeric_status"] == "available" for card in rendered["candidates"]
        )
        parts.append(
            f"<tr><td>{html.escape(config)}</td><td><a href='{html.escape(relative)}'>Open all candidate figures</a></td><td>{available} / {rendered['variant_count']}</td></tr>"
        )
        comparison = pd.read_csv(bundle / "comparison_summary.csv")
        comparison["gallery"] = relative
        comparisons.append(comparison)
        indexes.append(
            {
                "analysis_hash": manifest.get("analysis_hash"),
                "config": manifest["config"],
                "gallery": relative,
                "figure_manifest_sha256": digest_file(bundle / FIGURE_MANIFEST),
            }
        )
    parts.append("</table></html>")
    with tempfile.TemporaryDirectory(
        prefix=".candidate-index-", dir=output_dir
    ) as temporary:
        stage = Path(temporary)
        (stage / "index.html").write_text("\n".join(parts))
        pd.concat(comparisons, ignore_index=True).to_csv(
            stage / "comparison_summary.csv", index=False
        ) if comparisons else pd.DataFrame(
            columns=["candidate_id", "variant_id", "numeric_status"]
        ).to_csv(stage / "comparison_summary.csv", index=False)
        _write_json(
            stage / "cross_configuration_manifest.json",
            {
                "style_version": STYLE_VERSION,
                "configurations": indexes,
                "pooling": "None",
                "selection": "No automated winner selection",
            },
        )
        for path in stage.iterdir():
            os.replace(path, output_dir / path.name)
    return output_dir / "index.html"
