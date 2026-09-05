#!/usr/bin/env python3
"""Inspect GMM evidence and prompt-level proximity decisions."""

from __future__ import annotations

import argparse
import math
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_proximity_clusters import (  # noqa: E402
    CLUSTER_COLORS,
    KIND_COLORS,
    KIND_ORDER,
    RULE_COLORS,
    load_points,
    normalized_kind,
    prompt_spearman,
)
from utils.common.io import atomic_write_bytes, atomic_write_frame_csv  # noqa: E402
from utils.data.proximity_gmm import (  # noqa: E402
    COMPONENT_NAMES,
    DEFAULT_REG_COVAR,
    GaussianMixtureFit,
    fit_gaussian_mixture,
    marginal_component_boundary,
    raw_parameters,
    standardize_features,
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
    "gmm_component",
    "gmm_low_mode_probability",
    "computed_prompt_spearman",
    "prompt_rule",
    "prompt_median_l2_norm",
    "gmm_high_proximity_evidence",
    "prompt_gmm_evidence_seed_count",
    "gmm_selection_decision",
    "gmm_reg_covar",
    "gmm_iterations",
    "gmm_log_likelihood",
    "sscd_marginal_boundary",
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
            "Fit a deterministic two-component full-covariance Gaussian mixture "
            "to standardized L2/SSCD points, derive its marginal SSCD boundary, "
            "and apply the prompt evidence-and-correlation rule."
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
        help="Directory for one assignment CSV and one comparison figure.",
    )
    parser.add_argument(
        "--reg-covar",
        type=positive_finite_float,
        default=DEFAULT_REG_COVAR,
        help="Minimum covariance eigenvalue in standardized feature units.",
    )
    return parser


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
    sscd_boundary: float,
) -> pd.DataFrame:
    result = frame.copy()
    low_probability = fit.responsibilities[:, 0]
    hard_component = fit.responsibilities.argmax(axis=1)
    result["gmm_component"] = np.asarray(COMPONENT_NAMES)[hard_component]
    result["gmm_low_mode_probability"] = low_probability

    correlations = prompt_spearman(result)
    verify_prompt_correlations(result, correlations)
    result["computed_prompt_spearman"] = result["original_index"].map(correlations)
    rho = result["computed_prompt_spearman"]
    result["prompt_rule"] = np.select(
        [rho.lt(0.0), rho.ge(0.0)],
        ["rho < 0", "rho >= 0"],
        default="rho undefined",
    )

    prompt_median_l2 = result.groupby("original_index", sort=False)[
        "l2_norm"
    ].transform("median")
    evidence = result["sscd"].gt(sscd_boundary) & result["l2_norm"].lt(prompt_median_l2)
    evidence_count = evidence.groupby(result["original_index"], sort=False).transform(
        "sum"
    )
    result["prompt_median_l2_norm"] = prompt_median_l2
    result["gmm_high_proximity_evidence"] = evidence
    result["prompt_gmm_evidence_seed_count"] = evidence_count
    result["gmm_selection_decision"] = np.select(
        [rho.lt(0.0) & evidence_count.ge(1), rho.notna()],
        ["include", "discard"],
        default="unusable",
    )
    result["gmm_reg_covar"] = reg_covar
    result["gmm_iterations"] = fit.iterations
    result["gmm_log_likelihood"] = fit.log_likelihood
    result["sscd_marginal_boundary"] = sscd_boundary
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
    sscd_boundary: float,
) -> None:
    raw_means, raw_covariances = raw_parameters(fit, feature_mean, feature_scale)
    print(f"Input: {source}")
    print(
        f"Scope: {scope}; observations={len(frame)}; "
        f"prompts={frame['original_index'].nunique()}; "
        f"seeds/prompt={seeds_per_prompt}; excluded incomplete prompts={excluded_prompts}"
    )
    print(
        f"Converged in {fit.iterations} EM iterations; "
        f"log likelihood={fit.log_likelihood:.6f}"
    )
    print(f"Weighted marginal SSCD boundary={sscd_boundary:.6f}")
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
    print("\nGMM evidence selection versus within-prompt Spearman sign:")
    print(table.to_string())
    print(
        f"Prompts with hard assignments in both components: {split_prompts}/{len(prompts)}"
    )

    undefined_rho = prompts["prompt_rule"].eq("rho undefined").sum()
    evidence_prompts = int(prompts["prompt_gmm_evidence_seed_count"].ge(1).sum())
    included_prompts = int(prompts["gmm_selection_decision"].eq("include").sum())
    print(
        "Prompts with high-SSCD/below-median-L2 evidence: "
        f"{evidence_prompts}/{len(prompts)}; included by evidence and rho < 0: "
        f"{included_prompts}/{len(prompts)}; undefined rho={undefined_rho}"
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
    title: str,
) -> None:
    plotted = frame.copy()
    plotted["plot_kind"] = normalized_kind(plotted["kind"])
    panels = (
        (
            "gmm_component",
            CLUSTER_COLORS,
            COMPONENT_NAMES,
            "Full-covariance GMM, k=2",
        ),
        (
            "prompt_rule",
            RULE_COLORS,
            ("rho < 0", "rho >= 0", "rho undefined"),
            "Within-prompt Spearman sign",
        ),
        ("plot_kind", KIND_COLORS, KIND_ORDER, "Prompt kind"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharex=True, sharey=True)
    for axis, (column, colors, order, panel_title) in zip(axes, panels, strict=True):
        for label in order:
            subset = plotted.loc[plotted[column].eq(label)]
            if subset.empty:
                continue
            axis.scatter(
                subset["l2_norm"],
                subset["sscd"],
                s=7,
                alpha=0.32,
                color=colors[label],
                edgecolors="none",
                label=label,
                rasterized=True,
            )
        axis.axhline(
            float(plotted["sscd_marginal_boundary"].iloc[0]),
            color="0.35",
            linestyle="--",
            linewidth=0.8,
        )
        axis.set_title(panel_title)
        axis.set_xlabel(r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|_2$")
        axis.grid(True, linewidth=0.5, alpha=0.15)
        axis.legend(frameon=False, fontsize=8, markerscale=1.5)
    for component_index in range(2):
        for standard_deviations in (1.0, 2.0):
            axes[0].add_patch(
                covariance_ellipse(
                    component_means[component_index],
                    component_covariances[component_index],
                    component_index=component_index,
                    standard_deviations=standard_deviations,
                )
            )
    axes[0].set_ylabel("SSCD")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    buffer = BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    finally:
        plt.close(figure)
    atomic_write_bytes(destination, buffer.getvalue())


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    source_argument = arguments.input_csv.expanduser()
    source = source_argument.resolve()
    output = arguments.output_dir.expanduser().resolve()
    if output == source.parent or source.parent in output.parents:
        parser.error("output directory must not be inside the input CSV directory")
    scope = "all complete reference prompts"

    try:
        frame, seeds_per_prompt, excluded_prompts = load_points(
            source_argument,
            plotted_only=False,
        )
        required_reference_columns = {
            "reference_run_name",
            "reference_generation_hash",
            "reference_sscd_hash",
        }
        missing_reference_columns = sorted(required_reference_columns - set(frame))
        if missing_reference_columns:
            raise ValueError(
                "input must be a frozen reference selection.csv; missing: "
                + ", ".join(missing_reference_columns)
            )
        raw_values = frame[["l2_norm", "sscd"]].to_numpy(dtype=np.float64)
        features, feature_mean, feature_scale = standardize_features(raw_values)
        fit = fit_gaussian_mixture(features, reg_covar=arguments.reg_covar)
        boundary_standardized = marginal_component_boundary(
            fit.weights, fit.means, fit.covariances, feature_index=1
        )
        sscd_boundary = float(
            feature_mean[1] + boundary_standardized * feature_scale[1]
        )
        annotated = annotate(
            frame,
            fit,
            reg_covar=arguments.reg_covar,
            sscd_boundary=sscd_boundary,
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
            sscd_boundary=sscd_boundary,
        )
        plot_assignments(
            saved,
            figure_path,
            component_means=component_means,
            component_covariances=component_covariances,
            title=f"{saved['model_name'].iloc[0]}: GMM vs prompt behavior ({scope})",
        )
    except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        parser.error(str(error))

    print(f"Assignments: {assignments_path}")
    print(f"Figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
