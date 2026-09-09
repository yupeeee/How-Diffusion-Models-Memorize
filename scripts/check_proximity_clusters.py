#!/usr/bin/env python3
"""Compare two proximity clusters with prompt-level behavior in line figures."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common.io import atomic_write_frame_csv  # noqa: E402
from utils.experiments.plotting import (  # noqa: E402
    AXIS_NUMBER_FONT_SIZE,
    CATEGORY_LINESTYLES,
    FIGURE_SIZE,
    LEGEND_FONT_SIZE,
    PLOT_STYLE,
    TEXT_FONT_SIZE,
    X_AXIS_LABEL,
    Y_AXIS_LABEL,
    _publish_figures,
    add_prompt_curves,
    add_sscd_colorbar,
    category_legend_label,
)

REQUIRED_COLUMNS = {
    "model_name",
    "original_index",
    "seed",
    "prompt",
    "kind",
    "generated_image_path",
    "generated_image_tile_index",
    "l2_norm",
    "sscd",
    "observation_status",
    "include_prompt",
}
AUDIT_COLUMNS = [
    "model_name",
    "original_index",
    "seed",
    "prompt",
    "kind",
    "generated_image_path",
    "generated_image_tile_index",
    "l2_norm",
    "sscd",
    "include_prompt",
    "kmeans_cluster",
    "computed_prompt_spearman",
    "prompt_rule",
    "prompt_low_cluster_fraction",
    "prompt_majority_cluster",
]
KIND_ORDER = ("MV", "RV", "TV", "N", "Other")
# Preserve category encodings when changing only the legend display order.
KIND_LINESTYLES = {
    "MV": CATEGORY_LINESTYLES[0],
    "RV": CATEGORY_LINESTYLES[2],
    "TV": CATEGORY_LINESTYLES[1],
    "N": CATEGORY_LINESTYLES[3],
    "Other": CATEGORY_LINESTYLES[4],
}
CLUSTER_COLORS = {
    "low_sscd_mode": "#7B3294",
    "high_sscd_mode": "#008837",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit deterministic k-means with k=2 to standardized L2/SSCD points, "
            "then compare its groups with each prompt's Spearman correlation."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        type=Path,
        help="Experiment proximity.csv or reference selection.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for one assignment CSV and three single-panel PNG/PDF views.",
    )
    parser.add_argument(
        "--plotted-only",
        action="store_true",
        help=(
            "Use include_prompt=True rows, exactly matching proximity_vs_sscd.png. "
            "The default uses all prompts so correlation-rule failures are included."
        ),
    )
    parser.add_argument(
        "--sscd-cutoff",
        type=float,
        default=0.2,
        help="Post-hoc cloud diagnostic only; it does not affect clustering.",
    )
    return parser


def load_points(path: Path, *, plotted_only: bool) -> tuple[pd.DataFrame, int, int]:
    expanded = path.expanduser()
    if not expanded.is_file() or expanded.is_symlink():
        raise ValueError(f"input CSV is missing or unsafe: {expanded}")
    frame = pd.read_csv(
        expanded.resolve(),
        dtype={"original_index": str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("input CSV is missing columns: " + ", ".join(missing))

    included = (
        frame["include_prompt"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .map({"true": True, "false": False})
    )
    if included.isna().any():
        raise ValueError("include_prompt must contain only true or false")
    frame["include_prompt"] = included.astype(bool)

    seeds = pd.to_numeric(frame["seed"], errors="coerce").astype("float64")
    if (~np.isfinite(seeds) | seeds.mod(1.0).ne(0.0)).any():
        raise ValueError("seed must contain only finite integers")
    frame["seed"] = seeds.astype("int64")
    frame["l2_norm"] = pd.to_numeric(frame["l2_norm"], errors="coerce")
    frame["sscd"] = pd.to_numeric(frame["sscd"], errors="coerce")
    if plotted_only:
        frame = frame.loc[frame["include_prompt"]].copy()
    if frame.empty:
        raise ValueError("no prompts remain in the requested scope")

    identity_counts = frame.groupby("original_index", sort=False)[
        ["prompt", "kind", "include_prompt"]
    ].nunique(dropna=False)
    if identity_counts.gt(1).any(axis=1).any():
        raise ValueError("prompt, kind, and include_prompt must be fixed per prompt ID")
    if frame.duplicated(["original_index", "seed"]).any():
        raise ValueError("duplicate (original_index, seed) observations found")

    complete = frame["observation_status"].astype(str).eq("complete")
    finite = np.isfinite(frame["l2_norm"]) & np.isfinite(frame["sscd"])
    complete_prompt = (
        (complete & finite)
        .groupby(frame["original_index"], sort=False)
        .transform("all")
    )
    excluded_prompts = frame.loc[~complete_prompt, "original_index"].nunique()
    frame = frame.loc[complete_prompt].copy()
    if frame.empty:
        raise ValueError("no fully complete prompts remain")

    seed_sets = frame.groupby("original_index", sort=False)["seed"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    expected_seeds = seed_sets.iloc[0]
    if not seed_sets.map(lambda value: value == expected_seeds).all():
        raise ValueError("complete prompts do not share one common seed set")
    if len(expected_seeds) < 2:
        raise ValueError("at least two seeds per prompt are required")
    return frame.reset_index(drop=True), len(expected_seeds), int(excluded_prompts)


def standardized_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[["l2_norm", "sscd"]].to_numpy(dtype=np.float64)
    standard_deviation = values.std(axis=0, ddof=0)
    if np.any(~np.isfinite(standard_deviation)) or np.any(standard_deviation == 0.0):
        raise ValueError("L2 and SSCD must each have nonzero finite variance")
    return (values - values.mean(axis=0)) / standard_deviation


def kmeans_two(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit dependency-free k=2 using deterministic first-PC extrema."""

    centered = values - values.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ right_vectors[0]
    endpoints = [int(np.argmin(projection)), int(np.argmax(projection))]
    if endpoints[0] == endpoints[1]:
        raise ValueError("k-means cannot initialize two distinct clusters")
    centers = values[endpoints].copy()
    labels = np.full(len(values), -1, dtype=np.int64)
    for _ in range(300):
        squared_distance = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        next_labels = squared_distance.argmin(axis=1)
        if any(not np.any(next_labels == label) for label in (0, 1)):
            raise ValueError("k-means formed an empty cluster")
        if np.array_equal(next_labels, labels):
            inertia = float(((values - centers[labels]) ** 2).sum())
            return labels, inertia
        labels = next_labels
        centers = np.vstack([values[labels == label].mean(axis=0) for label in (0, 1)])
    raise ValueError("k-means did not converge within 300 iterations")


def prompt_spearman(frame: pd.DataFrame) -> pd.Series:
    correlations: dict[str, float] = {}
    for original_index, group in frame.groupby("original_index", sort=False):
        l2_rank = group["l2_norm"].rank(method="average")
        sscd_rank = group["sscd"].rank(method="average")
        rho = l2_rank.corr(sscd_rank, method="pearson")
        correlations[str(original_index)] = (
            float(rho) if rho is not None and math.isfinite(rho) else math.nan
        )
    return pd.Series(correlations, dtype="float64")


def annotate(frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["_raw_cluster"] = labels
    low_cluster = int(
        result.groupby("_raw_cluster", sort=True)["sscd"].median().idxmin()
    )
    result["kmeans_cluster"] = np.where(
        result["_raw_cluster"].eq(low_cluster),
        "low_sscd_mode",
        "high_sscd_mode",
    )
    correlations = prompt_spearman(result)
    result["computed_prompt_spearman"] = result["original_index"].map(correlations)
    rho = result["computed_prompt_spearman"]
    result["prompt_rule"] = np.select(
        [rho.lt(0.0), rho.ge(0.0)],
        ["rho < 0", "rho >= 0"],
        default="rho undefined",
    )
    is_low = result["kmeans_cluster"].eq("low_sscd_mode")
    low_fraction = is_low.groupby(result["original_index"], sort=False).transform(
        "mean"
    )
    result["prompt_low_cluster_fraction"] = low_fraction
    result["prompt_majority_cluster"] = np.select(
        [low_fraction.gt(0.5), low_fraction.lt(0.5)],
        ["low_sscd_mode", "high_sscd_mode"],
        default="tie",
    )
    return result.drop(columns="_raw_cluster")


def normalized_kind(values: pd.Series) -> pd.Series:
    kinds = values.astype(str).str.strip().str.upper()
    return kinds.where(kinds.isin(KIND_ORDER[:-1]), KIND_ORDER[-1])


def print_report(
    frame: pd.DataFrame,
    *,
    source: Path,
    scope: str,
    seeds_per_prompt: int,
    excluded_prompts: int,
    cutoff: float,
    inertia: float,
    features: np.ndarray,
) -> None:
    explained = 1.0 - inertia / float((features**2).sum())
    print(f"Input: {source}")
    print(
        f"Scope: {scope}; observations={len(frame)}; "
        f"prompts={frame['original_index'].nunique()}; "
        f"seeds/prompt={seeds_per_prompt}; excluded incomplete prompts={excluded_prompts}"
    )
    print(f"Two-cluster standardized variance explained: {explained:.3f}")
    for cluster in ("low_sscd_mode", "high_sscd_mode"):
        group = frame.loc[frame["kmeans_cluster"].eq(cluster)]
        shares = normalized_kind(group["kind"]).value_counts(normalize=True)
        kinds = ", ".join(f"{kind}={shares.get(kind, 0.0):.1%}" for kind in KIND_ORDER)
        print(
            f"{cluster}: n={len(group)}, median L2={group['l2_norm'].median():.3f}, "
            f"median SSCD={group['sscd'].median():.3f}, "
            f"SSCD<{cutoff:g}={group['sscd'].lt(cutoff).mean():.1%}, "
            f"from rho<0 prompts={group['prompt_rule'].eq('rho < 0').mean():.1%}; "
            f"{kinds}"
        )

    prompts = frame.drop_duplicates("original_index", keep="first")
    table = pd.crosstab(
        prompts["prompt_majority_cluster"],
        prompts["prompt_rule"],
        margins=True,
    )
    split = prompts["prompt_low_cluster_fraction"].between(
        0.0, 1.0, inclusive="neither"
    )
    print("\nPrompt-majority cluster versus within-prompt Spearman sign:")
    print(table.to_string())
    print(f"Prompts with points in both clusters: {split.sum()}/{len(prompts)}")

    evaluable = prompts.loc[
        prompts["prompt_majority_cluster"].ne("tie")
        & prompts["prompt_rule"].ne("rho undefined")
    ]
    predicted_discard = evaluable["prompt_majority_cluster"].eq("low_sscd_mode")
    actually_violates = evaluable["prompt_rule"].eq("rho >= 0")
    precision = (
        actually_violates.loc[predicted_discard].mean()
        if predicted_discard.any()
        else math.nan
    )
    false_discard = predicted_discard.loc[~actually_violates].mean()
    print(
        "If low-mode majority means discard: "
        f"discard precision={precision:.1%}, "
        f"false-discard rate among rho<0 prompts={false_discard:.1%}"
    )


def plot_assignments(
    frame: pd.DataFrame, destination: Path, *, cutoff: float
) -> tuple[Path, ...]:
    """Save separate membership, Spearman-sign, and kind proximity profiles.

    SSCD controls color. Line style records the category of each segment's
    right endpoint after L2 ordering; curves never join different prompts.
    """
    plotted = frame.copy()
    plotted["plot_kind"] = normalized_kind(plotted["kind"])
    views = (
        ("kmeans_cluster", ("low_sscd_mode", "high_sscd_mode"), ""),
        ("prompt_rule", ("rho < 0", "rho >= 0", "rho undefined"), "_spearman"),
        ("plot_kind", KIND_ORDER, "_kind"),
    )
    styles = CATEGORY_LINESTYLES
    figures = []
    destinations: list[Path] = []
    with matplotlib.rc_context(PLOT_STYLE):
        try:
            for column, order, suffix in views:
                figure, axis = plt.subplots(figsize=FIGURE_SIZE)
                filenames = {
                    extension: f"{destination.stem}{suffix}.{extension}"
                    for extension in ("png", "pdf")
                }
                figures.append((figure, filenames))
                destinations.extend(
                    destination.parent / name for name in filenames.values()
                )
                category_styles = (
                    KIND_LINESTYLES
                    if column == "plot_kind"
                    else dict(zip(order, styles))
                )
                add_prompt_curves(
                    axis,
                    plotted,
                    category_column=column,
                    category_styles=category_styles,
                )
                add_sscd_colorbar(figure, axis, plotted["sscd"])
                handles = [
                    Line2D(
                        [],
                        [],
                        color="black",
                        linestyle=category_styles[label],
                        linewidth=1.0,
                        alpha=1.0,
                        label=(
                            category_legend_label(label)
                            if column == "plot_kind"
                            else label.replace("_", " ")
                        ),
                    )
                    for label in order
                    if plotted[column].eq(label).any()
                ]
                axis.axhline(cutoff, color="0.35", linestyle="--", linewidth=0.8)
                axis.set_xlabel(X_AXIS_LABEL, fontsize=TEXT_FONT_SIZE)
                axis.set_ylabel(Y_AXIS_LABEL, fontsize=TEXT_FONT_SIZE)
                axis.tick_params(
                    axis="both", which="both", labelsize=AXIS_NUMBER_FONT_SIZE
                )
                axis.grid(True, linewidth=0.6, alpha=0.18)
                if handles:
                    axis.legend(
                        handles=handles, frameon=False, fontsize=LEGEND_FONT_SIZE
                    )
                figure.tight_layout()
            _publish_figures(destination.parent, figures)
        finally:
            for figure, _filenames in figures:
                plt.close(figure)
    return tuple(destinations)


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    if not math.isfinite(arguments.sscd_cutoff):
        parser.error("--sscd-cutoff must be finite")
    source_argument = arguments.input_csv.expanduser()
    source = source_argument.resolve()
    output = arguments.output_dir.expanduser().resolve()
    if output == source.parent:
        parser.error("output directory must not be the input CSV directory")
    scope = "plotted prompts" if arguments.plotted_only else "all prompts"

    try:
        frame, seeds_per_prompt, excluded_prompts = load_points(
            source_argument, plotted_only=arguments.plotted_only
        )
        features = standardized_features(frame)
        labels, inertia = kmeans_two(features)
        annotated = annotate(frame, labels)
        assignments_path = output / "cluster_assignments.csv"
        figure_path = output / "kmeans_k2.png"
        atomic_write_frame_csv(annotated[AUDIT_COLUMNS], assignments_path)

        saved = pd.read_csv(
            assignments_path,
            dtype={"original_index": str},
            float_precision="round_trip",
        )
        print_report(
            saved,
            source=source,
            scope=scope,
            seeds_per_prompt=seeds_per_prompt,
            excluded_prompts=excluded_prompts,
            cutoff=arguments.sscd_cutoff,
            inertia=inertia,
            features=features,
        )
        figure_paths = plot_assignments(
            saved, figure_path, cutoff=arguments.sscd_cutoff
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Assignments: {assignments_path}")
    for path in figure_paths:
        print(f"Figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
