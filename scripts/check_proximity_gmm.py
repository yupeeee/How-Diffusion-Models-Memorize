#!/usr/bin/env python3
"""Inspect the current diagonal-GMM prompt selection in single-panel figures."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_proximity_clusters import (  # noqa: E402
    CLUSTER_COLORS,
    KIND_ORDER,
    KIND_LINESTYLES,
    normalized_kind,
)
from utils.common.io import atomic_write_frame_csv  # noqa: E402
from utils.data.proximity_gmm import (  # noqa: E402
    COMPONENT_NAMES,
    DEFAULT_REG_COVAR,
    GaussianMixtureFit,
    fit_gaussian_mixture,
    raw_parameters,
    standardize_features,
)
from utils.data.selection import (  # noqa: E402
    _gmm_decision,
    _normalize_csv,
    _prompt_spearman,
)
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
    "observation_status",
    "observation_error",
    "gmm_component",
    "gmm_low_mode_probability",
    "computed_prompt_spearman",
    "prompt_rule",
    "gmm_selection_decision",
    "gmm_reg_covar",
    "gmm_iterations",
    "gmm_log_likelihood",
]


def positive_finite_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the current deterministic two-component diagonal-covariance "
            "Gaussian mixture to every valid reference L2/SSCD observation. "
            "Keep complete prompts with any high-component seed; Spearman "
            "correlations are descriptive only."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        type=Path,
        help="Frozen reference selection.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for one assignment CSV and three single-panel PNG/PDF views.",
    )
    parser.add_argument(
        "--reg-covar",
        type=positive_finite_float,
        default=DEFAULT_REG_COVAR,
        help="Minimum covariance eigenvalue in standardized feature units.",
    )
    return parser


def _valid_observations(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["observation_status"].eq("complete")
        & np.isfinite(frame["l2_norm"])
        & np.isfinite(frame["sscd"])
    )


def load_reference_points(path: Path) -> tuple[pd.DataFrame, int, int]:
    """Load current reference rows without dropping valid incomplete siblings."""

    expanded = path.expanduser()
    if not expanded.is_file() or expanded.is_symlink():
        raise ValueError(f"input CSV is missing or unsafe: {expanded}")
    frame = _normalize_csv(
        pd.read_csv(
            expanded.resolve(),
            dtype={"original_index": str},
            keep_default_na=False,
            float_precision="round_trip",
        )
    )
    if frame.empty:
        raise ValueError("reference selection contains no prompts")
    identity_counts = frame.groupby("original_index", sort=False)[
        ["prompt", "kind", "include_prompt"]
    ].nunique(dropna=False)
    if identity_counts.gt(1).any(axis=1).any():
        raise ValueError("prompt, kind, and include_prompt must be fixed per prompt ID")
    if frame.duplicated(["original_index", "seed"]).any():
        raise ValueError("duplicate (original_index, seed) observations found")
    seed_sets = frame.groupby("original_index", sort=False)["seed"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    expected_seeds = seed_sets.iloc[0]
    if not seed_sets.map(lambda values: values == expected_seeds).all():
        raise ValueError("reference prompts do not share one common seed set")
    if len(expected_seeds) < 2:
        raise ValueError("at least two reference seeds per prompt are required")
    valid = _valid_observations(frame)
    if not valid.any():
        raise ValueError("reference selection has no valid observations")
    incomplete_prompts = (
        (~valid).groupby(frame["original_index"], sort=False).any().sum()
    )
    return frame.reset_index(drop=True), len(expected_seeds), int(incomplete_prompts)


def verify_prompt_correlations(
    frame: pd.DataFrame,
    computed: pd.Series,
) -> None:
    stored_column = None
    if "experiment_prompt_spearman" in frame.columns:
        stored_column = "experiment_prompt_spearman"
    elif "prompt_spearman" in frame.columns:
        stored_column = "prompt_spearman"
    if stored_column is None:
        return
    stored_text = frame[stored_column].astype(str).str.strip()
    stored_numeric = pd.to_numeric(stored_text, errors="coerce")
    invalid = stored_numeric.isna() & stored_text.ne("")
    if invalid.any() or np.isinf(stored_numeric).any():
        raise ValueError(f"{stored_column} must contain finite numbers or blanks")
    grouped = stored_numeric.groupby(frame["original_index"], sort=False)
    inconsistent = grouped.nunique(dropna=False).gt(1)
    if inconsistent.any():
        raise ValueError(
            f"{stored_column} changes within prompt IDs: "
            + ", ".join(map(str, inconsistent.index[inconsistent][:5]))
        )
    stored = grouped.agg(lambda values: values.iloc[0])
    aligned = computed.reindex(stored.index)
    stored_defined = stored.notna()
    computed_defined = aligned.notna()
    if not stored_defined.equals(computed_defined):
        raise ValueError(
            f"recomputed prompt Spearman definedness disagrees with {stored_column}"
        )
    if stored_defined.any() and not np.allclose(
        stored.loc[stored_defined],
        aligned.loc[stored_defined],
        rtol=0.0,
        atol=1e-12,
    ):
        difference = (
            (stored.loc[stored_defined] - aligned.loc[stored_defined]).abs().max()
        )
        raise ValueError(
            f"recomputed prompt Spearman disagrees with {stored_column}; "
            f"maximum difference={difference:.3e}"
        )


def annotate(
    frame: pd.DataFrame,
    fit: GaussianMixtureFit,
    *,
    reg_covar: float,
) -> pd.DataFrame:
    result = frame.copy()
    valid = _valid_observations(result)
    if fit.responsibilities.shape != (int(valid.sum()), 2):
        raise ValueError("GMM responsibilities do not match the valid observations")
    result["gmm_component"] = ""
    result["gmm_low_mode_probability"] = math.nan
    result.loc[valid, "gmm_component"] = np.asarray(COMPONENT_NAMES)[
        fit.responsibilities.argmax(axis=1)
    ]
    result.loc[valid, "gmm_low_mode_probability"] = fit.responsibilities[:, 0]

    correlations = pd.Series(
        {
            str(index): _prompt_spearman(group.to_dict(orient="records"))
            for index, group in result.groupby("original_index", sort=False)
        },
        dtype="float64",
    )
    verify_prompt_correlations(result, correlations)
    result["computed_prompt_spearman"] = result["original_index"].map(correlations)
    rho = result["computed_prompt_spearman"]
    result["prompt_rule"] = np.select(
        [rho.lt(0.0), rho.ge(0.0)],
        ["rho < 0", "rho >= 0"],
        default="rho undefined",
    )
    decisions: dict[str, str] = {}
    for index, group in result.groupby("original_index", sort=False):
        if not valid.loc[group.index].all():
            decisions[str(index)] = "unusable"
            continue
        included, _, _ = _gmm_decision(group["gmm_low_mode_probability"].tolist())
        decisions[str(index)] = "include" if included else "discard"
    result["gmm_selection_decision"] = result["original_index"].map(decisions)
    result["gmm_reg_covar"] = reg_covar
    result["gmm_iterations"] = fit.iterations
    result["gmm_log_likelihood"] = fit.log_likelihood
    return result


def print_report(
    frame: pd.DataFrame,
    *,
    fit: GaussianMixtureFit,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    source: Path,
    scope: str,
    seeds_per_prompt: int,
    excluded_prompts: int,
) -> None:
    raw_means, raw_covariances = raw_parameters(fit, feature_mean, feature_scale)
    print(f"Input: {source}")
    print(
        f"Scope: {scope}; fitted observations={len(fit.responsibilities)}; "
        f"prompts={frame['original_index'].nunique()}; "
        f"seeds/prompt={seeds_per_prompt}; unusable incomplete prompts={excluded_prompts}"
    )
    print(
        f"Converged in {fit.iterations} EM iterations; "
        f"log likelihood={fit.log_likelihood:.6f}"
    )
    for component_index, component_name in enumerate(COMPONENT_NAMES):
        group = frame.loc[frame["gmm_component"].eq(component_name)]
        covariance = raw_covariances[component_index]
        correlation = covariance[0, 1] / math.sqrt(covariance[0, 0] * covariance[1, 1])
        shares = normalized_kind(group["kind"]).value_counts(normalize=True)
        kinds = ", ".join(f"{kind}={shares.get(kind, 0.0):.1%}" for kind in KIND_ORDER)
        print(
            f"{component_name}: weight={fit.weights[component_index]:.3f}, "
            f"hard n={len(group)}, mean L2={raw_means[component_index, 0]:.3f}, "
            f"mean SSCD={raw_means[component_index, 1]:.3f}, "
            f"component correlation={correlation:.3f}; {kinds}"
        )
    uncertain = frame["gmm_low_mode_probability"].between(0.4, 0.6)
    print(f"Points with low-mode posterior in [0.4, 0.6]: {uncertain.mean():.1%}")

    prompts = frame.drop_duplicates("original_index", keep="first")
    table = pd.crosstab(
        prompts["gmm_selection_decision"], prompts["prompt_rule"], margins=True
    )
    hard_fraction = (
        frame["gmm_component"]
        .eq(COMPONENT_NAMES[0])
        .groupby(frame["original_index"], sort=False)
        .mean()
    )
    split_prompts = hard_fraction.between(0.0, 1.0, inclusive="neither").sum()
    print("\nGMM selection versus descriptive within-prompt Spearman sign:")
    print(table.to_string())
    print(
        f"Prompts with hard assignments in both components: {split_prompts}/{len(prompts)}"
    )

    undefined_rho = prompts["prompt_rule"].eq("rho undefined").sum()
    included_prompts = int(prompts["gmm_selection_decision"].eq("include").sum())
    print(
        f"Prompts included by any high-component seed: {included_prompts}/{len(prompts)}; "
        f"undefined rho (descriptive only)={undefined_rho}"
    )


def covariance_ellipse(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    component_index: int,
    standard_deviations: float,
) -> Ellipse:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    direction = eigenvectors[:, order[0]]
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    width, height = 2.0 * standard_deviations * np.sqrt(eigenvalues)
    return Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        color=CLUSTER_COLORS[COMPONENT_NAMES[component_index]],
        linewidth=1.0,
        alpha=0.8,
    )


def plot_assignments(
    frame: pd.DataFrame,
    destination: Path,
    *,
    component_means: np.ndarray,
    component_covariances: np.ndarray,
) -> tuple[Path, ...]:
    """Save separate component, Spearman-sign, and kind proximity profiles.

    SSCD controls color. Line style records the category of each segment's
    right endpoint after L2 ordering; curves never join different prompts.
    """

    plotted = frame.copy()
    plotted["plot_kind"] = normalized_kind(plotted["kind"])
    views = (
        ("gmm_component", COMPONENT_NAMES, ""),
        ("prompt_rule", ("rho < 0", "rho >= 0", "rho undefined"), "_spearman"),
        ("plot_kind", KIND_ORDER, "_kind"),
    )
    figures = []
    published_paths: list[Path] = []
    with matplotlib.rc_context(PLOT_STYLE):
        try:
            for column, order, suffix in views:
                figure, axis = plt.subplots(figsize=FIGURE_SIZE)
                filenames = {
                    extension: f"{destination.stem}{suffix}.{extension}"
                    for extension in ("png", "pdf")
                }
                figures.append((figure, filenames))
                published_paths.extend(
                    destination.parent / name for name in filenames.values()
                )
                category_styles = (
                    KIND_LINESTYLES
                    if column == "plot_kind"
                    else dict(zip(order, CATEGORY_LINESTYLES))
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
                if column == "gmm_component":
                    for component_index in range(2):
                        for standard_deviations in (1.0, 2.0):
                            axis.add_patch(
                                covariance_ellipse(
                                    component_means[component_index],
                                    component_covariances[component_index],
                                    component_index=component_index,
                                    standard_deviations=standard_deviations,
                                )
                            )
                axis.set_xlabel(X_AXIS_LABEL, fontsize=TEXT_FONT_SIZE)
                axis.set_ylabel(Y_AXIS_LABEL, fontsize=TEXT_FONT_SIZE)
                axis.tick_params(
                    axis="both", which="both", labelsize=AXIS_NUMBER_FONT_SIZE
                )
                axis.grid(True, linewidth=0.6, alpha=0.18)
                if handles:
                    axis.legend(
                        handles=handles,
                        frameon=False,
                        fontsize=LEGEND_FONT_SIZE,
                    )
                figure.tight_layout()
            _publish_figures(destination.parent, figures)
        finally:
            for figure, _filenames in figures:
                plt.close(figure)
    return tuple(published_paths)


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    source_argument = arguments.input_csv.expanduser()
    source = source_argument.resolve()
    output = arguments.output_dir.expanduser().resolve()
    if output == source.parent or source.parent in output.parents:
        parser.error("output directory must not be inside the input CSV directory")
    scope = "all valid reference observations"

    try:
        frame, seeds_per_prompt, excluded_prompts = load_reference_points(
            source_argument
        )
        valid = _valid_observations(frame)
        raw_values = frame.loc[valid, ["l2_norm", "sscd"]].to_numpy(dtype=np.float64)
        features, feature_mean, feature_scale = standardize_features(raw_values)
        fit = fit_gaussian_mixture(features, reg_covar=arguments.reg_covar)
        annotated = annotate(
            frame,
            fit,
            reg_covar=arguments.reg_covar,
        )
        assignments_path = output / "gmm_assignments.csv"
        figure_path = output / "gmm_k2.png"
        atomic_write_frame_csv(annotated[AUDIT_COLUMNS], assignments_path)

        saved = pd.read_csv(
            assignments_path,
            dtype={"original_index": str},
            float_precision="round_trip",
        )
        component_means, component_covariances = raw_parameters(
            fit, feature_mean, feature_scale
        )
        print_report(
            saved,
            fit=fit,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            source=source,
            scope=scope,
            seeds_per_prompt=seeds_per_prompt,
            excluded_prompts=excluded_prompts,
        )
        figure_paths = plot_assignments(
            saved.loc[_valid_observations(saved)].copy(),
            figure_path,
            component_means=component_means,
            component_covariances=component_covariances,
        )
    except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        parser.error(str(error))

    print(f"Assignments: {assignments_path}")
    for path in figure_paths:
        print(f"Figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
