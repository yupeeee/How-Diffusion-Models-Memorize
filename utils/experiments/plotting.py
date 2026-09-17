"""Write proximity audit tables and selection diagnostic figures."""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.transforms import BboxBase  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
import numpy as np
import pandas as pd

from utils.common.io import atomic_write_frame_csv
from utils.data.proximity_gmm import COMPONENT_NAMES
from . import proximity_style
from .publication_style import FIGURE_DPI

__all__ = [
    "AnalysisStatistics",
    "CATEGORY_LINESTYLES",
    "PlottingError",
    "PROXIMITY_FIGURE_FILENAMES",
    "PROXIMITY_FIGURES",
    "PROXIMITY_PDF_FIGURES",
    "PROXIMITY_PDF_FILENAMES",
    "SELECTION_PDF_FILENAMES",
    "SELECTION_FIGURE_FILENAMES",
    "SELECTION_GMM_FIGURES",
    "add_prompt_curves",
    "add_sscd_colorbar",
    "category_legend_label",
    "covariance_ellipse",
    "publish_figures",
    "write_analysis_outputs",
    "write_gmm_fit_figure",
    "write_saved_analysis_figures",
    "write_selection_figure",
]

_KIND_ORDER = ("MV", "RV", "TV", "N", "Other / unlabeled")
_KIND_COLORS = {
    "MV": "C3",
    "RV": "C1",
    "TV": "C0",
    "N": "C2",
    "Other / unlabeled": "#7F7F7F",
}
FIGURE_SIZE = (4.0, 4.0)
TEXT_FONT_SIZE = 15
AXIS_LABEL_FONT_SIZE = 18
AXIS_NUMBER_FONT_SIZE = 15
LEGEND_FONT_SIZE = 12
SUMMARY_FONT_SIZE = 12
CATEGORY_LINESTYLES = (
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    (0, (5, 1, 1, 1, 1, 1)),
)
SCATTER_SIZE = 24
SCATTER_ALPHA = 0.8
CURVE_LINE_WIDTH = 1.0
CURVE_ALPHA = 0.35
COLORBAR_ALPHA = 1.0
SSCD_COLOR_RANGE = (0.0, 1.0)
FIGURE_PAD_INCHES = 0.05
X_AXIS_LABEL = r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|$"
Y_AXIS_LABEL = "SSCD"
EXPERIMENT_SPEARMAN_COLUMN = "experiment_prompt_spearman"
_FIGURE_FORMATS = ("png", "pdf")
PROXIMITY_FIGURES = {
    "selected": {
        "png": "proximity_vs_sscd.png",
        "pdf": "proximity_vs_sscd.pdf",
    },
    "all_prompts": {
        "png": "proximity_vs_sscd_all_prompts.png",
        "pdf": "proximity_vs_sscd_all_prompts.pdf",
    },
}
PROXIMITY_FIGURE_FILENAMES = tuple(
    PROXIMITY_FIGURES[view][file_format]
    for view in ("selected", "all_prompts")
    for file_format in _FIGURE_FORMATS
)
SELECTION_GMM_FIGURES = {
    "png": "proximity_vs_sscd_gmm_fit.png",
    "pdf": "proximity_vs_sscd_gmm_fit.pdf",
}
SELECTION_FIGURE_FILENAMES = PROXIMITY_FIGURE_FILENAMES + tuple(
    SELECTION_GMM_FIGURES[file_format] for file_format in _FIGURE_FORMATS
)
# Retain the paired catalog above as historical configuration metadata and for
# generic publisher clients. Active proximity entrypoints publish PDFs only.
PROXIMITY_PDF_FIGURES = {
    view: {"pdf": formats["pdf"]} for view, formats in PROXIMITY_FIGURES.items()
}
PROXIMITY_PDF_FILENAMES = tuple(formats["pdf"] for formats in PROXIMITY_PDF_FIGURES.values())
SELECTION_PDF_FILENAMES = (*PROXIMITY_PDF_FILENAMES, SELECTION_GMM_FIGURES["pdf"])
_GMM_COMPONENT_COLORS = {
    "low_sscd_mode": "#7B3294",
    "high_sscd_mode": "#008837",
}
_GMM_ELLIPSE_STANDARD_DEVIATIONS = (1.0, 2.0)

PLOT_STYLE = {
    "figure.figsize": FIGURE_SIZE,
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "font.family": "STIXGeneral",
    "font.size": TEXT_FONT_SIZE,
    "mathtext.fontset": "stix",
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "axes.titlesize": TEXT_FONT_SIZE,
    "figure.titlesize": TEXT_FONT_SIZE,
    "xtick.labelsize": AXIS_NUMBER_FONT_SIZE,
    "ytick.labelsize": AXIS_NUMBER_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
}


def category_legend_label(category: str) -> str:
    """Render typewriter category text without requiring an external TeX install."""
    if matplotlib.rcParams["text.usetex"]:
        return rf"\texttt{{{category}}}"
    # MathText does not support \texttt; \mathtt is its monospace equivalent.
    return rf"$\mathtt{{{category}}}$"


class PlottingError(RuntimeError):
    """The proximity analysis table cannot be validated or plotted."""


@dataclass(frozen=True, slots=True)
class AnalysisStatistics:
    """Prompt-level behavior among prompts retained by the frozen selection."""

    total_selected_prompts: int
    evaluable_selected_prompts: int
    negative_spearman_prompts: int
    negative_spearman_fraction: float | None
    median_spearman: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "total_selected_prompts": self.total_selected_prompts,
            "evaluable_selected_prompts": self.evaluable_selected_prompts,
            "negative_spearman_prompts": self.negative_spearman_prompts,
            "negative_spearman_fraction": self.negative_spearman_fraction,
            "median_spearman": self.median_spearman,
        }


def write_analysis_outputs(
    output_directory: str | Path,
    *,
    analysis: pd.DataFrame,
    figure_directory: str | Path | None = None,
) -> AnalysisStatistics:
    """Publish one seed-level CSV and comparable all/selected PDF scatter plots.

    Each seed row receives its prompt's experiment-seed Spearman value before
    publication. The saved CSV is then reloaded and validated; both figure
    annotations and returned statistics are computed only from that saved log.
    """

    _validate_analysis(analysis)
    annotated = _annotate_experiment_spearman(analysis)
    output = Path(output_directory)
    csv_path = output / "proximity.csv"
    atomic_write_frame_csv(annotated, csv_path)

    return write_saved_analysis_figures(output, figure_directory=figure_directory)


def write_saved_analysis_figures(
    output_directory: str | Path,
    *,
    figure_directory: str | Path | None = None,
) -> AnalysisStatistics:
    """Rebuild experiment proximity figures without rewriting their saved CSV."""

    output = Path(output_directory)
    csv_path = output / "proximity.csv"
    if not csv_path.is_file() or csv_path.is_symlink():
        raise PlottingError(f"saved proximity table is missing or unsafe: {csv_path}")
    try:
        saved_analysis = pd.read_csv(
            csv_path,
            dtype={
                "original_index": str,
                "record_id": str,
                "prompt": str,
                "kind": str,
                "target_image_sha256": str,
                "generated_image_path": str,
            },
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (OSError, TypeError, ValueError) as error:
        raise PlottingError(f"cannot read saved proximity table {csv_path}: {error}") from error
    _validate_analysis(saved_analysis)
    _validate_prompt_spearman(saved_analysis, column=EXPERIMENT_SPEARMAN_COLUMN)
    all_prompts = saved_analysis.copy()
    selected = saved_analysis.loc[saved_analysis["include_prompt"].astype(bool)].copy()
    with matplotlib.rc_context(PLOT_STYLE):
        return _write_scatter_views(
            Path(figure_directory) if figure_directory is not None else output,
            all_prompts=all_prompts,
            selected=selected,
            spearman_column=EXPERIMENT_SPEARMAN_COLUMN,
        )


def write_selection_figure(
    selection_directory: str | Path,
    *,
    output_directory: str | Path,
    figure_directory: str | Path | None = None,
) -> AnalysisStatistics:
    """Rebuild reference scatter plots from frozen ``selection.csv``.

    The frozen table remains the sole numerical log. The pre-discard view
    includes complete prompt groups regardless of their selection decision;
    prompt groups containing any failed observation or unplottable measurement
    are omitted.
    """

    selection = Path(selection_directory)
    output = Path(output_directory)
    csv_path = selection / "selection.csv"
    try:
        saved_selection = pd.read_csv(
            csv_path,
            dtype={"original_index": str},
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (OSError, ValueError) as error:
        raise PlottingError(
            f"cannot read frozen selection {csv_path}: {error}"
        ) from error
    if "prompt_spearman" not in saved_selection.columns:
        raise PlottingError("selection table is missing columns: prompt_spearman")
    _validate_inclusion(saved_selection)
    all_prompts = _fully_plottable_prompt_rows(saved_selection)
    _validate_measurements(all_prompts)
    _validate_prompt_spearman(all_prompts, column="prompt_spearman")
    selected = saved_selection.loc[
        saved_selection["include_prompt"].astype(bool)
    ].copy()
    _validate_measurements(selected)
    _validate_prompt_spearman(selected, column="prompt_spearman")
    with matplotlib.rc_context(PLOT_STYLE):
        return _write_scatter_views(
            Path(figure_directory) if figure_directory is not None else output,
            all_prompts=all_prompts,
            selected=selected,
            spearman_column="prompt_spearman",
        )


def write_gmm_fit_figure(
    selection_directory: str | Path,
    *,
    frame: pd.DataFrame,
    configuration: Mapping[str, object],
    figure_directory: str | Path | None = None,
) -> None:
    """Publish the stored GMM as a PDF without fitting it again."""

    output = Path(figure_directory) if figure_directory is not None else Path(selection_directory)
    with proximity_style.style_context(PLOT_STYLE):
        figure = _gmm_fit_figure(frame, configuration)
        try:
            _publish_figures(
                output, ((figure, {"pdf": SELECTION_GMM_FIGURES["pdf"]}),),
                formats=("pdf",),
                export_options={SELECTION_GMM_FIGURES["pdf"]: {"dpi": proximity_style.PDF_DPI}},
            )
        finally:
            plt.close(figure)


def _fully_plottable_prompt_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return prompt groups whose observations completed with finite measurements."""

    required = {"original_index", "l2_norm", "sscd", "observation_status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PlottingError("proximity table is missing columns: " + ", ".join(missing))
    if frame.empty:
        return frame.copy()
    row_is_plottable = frame["observation_status"].eq("complete")
    for column in ("l2_norm", "sscd"):
        row_is_plottable &= frame[column].map(
            lambda value: _finite_or_none(value) is not None
        )
    group_is_plottable = row_is_plottable.groupby(
        frame["original_index"], sort=False, dropna=False
    ).transform("all")
    return frame.loc[group_is_plottable].copy()


def _write_scatter_views(
    output: Path,
    *,
    all_prompts: pd.DataFrame,
    selected: pd.DataFrame,
    spearman_column: str,
) -> AnalysisStatistics:
    """Render paired views with identical all-prompt axes and local styling."""

    all_statistics = _prompt_statistics(all_prompts, column=spearman_column)
    selected_statistics = _prompt_statistics(selected, column=spearman_column)
    figures: list[Figure] = []
    try:
        with proximity_style.style_context(PLOT_STYLE):
            all_figure = _scatter_figure(all_prompts, all_statistics)
            figures.append(all_figure)
            selected_figure = _scatter_figure(selected, selected_statistics)
            figures.append(selected_figure)
            all_axes, selected_axes = all_figure.axes[0], selected_figure.axes[0]
            # Use the all-prompt autoscale, including its existing margins and
            # negative SSCD observations. The retained view cannot rescale it.
            limits = (all_axes.get_xlim(), all_axes.get_ylim())
            ticks = (all_axes.get_xticks(), all_axes.get_yticks())
            for axes in (all_axes, selected_axes):
                axes.set_xlim(limits[0])
                axes.set_ylim(limits[1])
                axes.xaxis.set_major_locator(FixedLocator(ticks[0]))
                axes.yaxis.set_major_locator(FixedLocator(ticks[1]))
            _publish_figures(
                output,
                (
                    (selected_figure, PROXIMITY_PDF_FIGURES["selected"]),
                    (all_figure, PROXIMITY_PDF_FIGURES["all_prompts"]),
                ),
                formats=("pdf",),
                export_options={
                    PROXIMITY_PDF_FIGURES[view]["pdf"]: {"dpi": proximity_style.PDF_DPI}
                    for view in ("selected", "all_prompts")
                },
            )
    finally:
        for figure in figures:
            plt.close(figure)
    print(f"[Proximity] Saved PDFs: {output}", flush=True)
    return selected_statistics


def _annotate_experiment_spearman(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=[EXPERIMENT_SPEARMAN_COLUMN], errors="ignore").copy()
    by_prompt = {
        original_index: _spearman(group)
        for original_index, group in result.groupby(
            "original_index", sort=False, dropna=False
        )
    }
    result[EXPERIMENT_SPEARMAN_COLUMN] = result["original_index"].map(by_prompt)
    return result


def _spearman(frame: pd.DataFrame) -> float | None:
    if len(frame) < 2:
        return None
    x = pd.to_numeric(frame["l2_norm"], errors="raise").astype("float64")
    y = pd.to_numeric(frame["sscd"], errors="raise").astype("float64")
    if x.nunique(dropna=False) < 2 or y.nunique(dropna=False) < 2:
        return None
    return _finite_or_none(
        x.rank(method="average").corr(y.rank(method="average"), method="pearson")
    )


def _prompt_statistics(
    selected: pd.DataFrame,
    *,
    column: str,
) -> AnalysisStatistics:
    prompt_rows = selected.drop_duplicates("original_index", keep="first")
    spearman = pd.to_numeric(prompt_rows[column], errors="coerce").astype("float64")
    evaluable = spearman.dropna()
    evaluable_count = int(len(evaluable))
    negative_count = int((evaluable < 0).sum())
    return AnalysisStatistics(
        total_selected_prompts=int(len(prompt_rows)),
        evaluable_selected_prompts=evaluable_count,
        negative_spearman_prompts=negative_count,
        negative_spearman_fraction=(
            float(negative_count / evaluable_count) if evaluable_count else None
        ),
        median_spearman=(
            _finite_or_none(evaluable.median()) if evaluable_count else None
        ),
    )


def _validate_measurements(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise PlottingError("proximity analysis input must be a pandas DataFrame")
    required = {"original_index", "l2_norm", "sscd"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PlottingError("proximity table is missing columns: " + ", ".join(missing))
    if frame["original_index"].isna().any():
        raise PlottingError("original_index must not be missing")
    if frame.empty:
        return
    for column in ("l2_norm", "sscd"):
        try:
            values = pd.to_numeric(frame[column], errors="raise").astype("float64")
        except (TypeError, ValueError) as error:
            raise PlottingError(f"{column} must contain only finite numbers") from error
        if not _finite_series(values):
            raise PlottingError(f"{column} must contain only finite numbers")


def _validate_analysis(frame: pd.DataFrame) -> None:
    _validate_measurements(frame)
    _validate_inclusion(frame)


def _validate_inclusion(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise PlottingError("proximity analysis input must be a pandas DataFrame")
    if "original_index" not in frame.columns:
        raise PlottingError("proximity table is missing columns: original_index")
    if frame["original_index"].isna().any():
        raise PlottingError("original_index must not be missing")
    if "include_prompt" not in frame.columns:
        raise PlottingError("proximity table is missing columns: include_prompt")
    included = frame["include_prompt"]
    if included.isna().any() or (
        not frame.empty and not pd.api.types.is_bool_dtype(included.dtype)
    ):
        raise PlottingError("include_prompt must contain only booleans")
    if (
        not frame.empty
        and frame.groupby("original_index", sort=False)["include_prompt"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise PlottingError("include_prompt must be constant within each prompt")


def _validate_prompt_spearman(frame: pd.DataFrame, *, column: str) -> None:
    if column not in frame.columns:
        raise PlottingError(f"proximity table is missing columns: {column}")
    values = frame[column]
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    blank = values.isna() | values.map(
        lambda value: isinstance(value, str) and not value.strip()
    )
    if ((~blank) & numeric.isna()).any() or not _finite_series(numeric.dropna()):
        raise PlottingError(f"{column} must be finite or blank")
    for original_index, group in frame.groupby(
        "original_index", sort=False, dropna=False
    ):
        expected = _spearman(group)
        observed = pd.to_numeric(group[column], errors="coerce").astype("float64")
        if expected is None:
            if observed.notna().any():
                raise PlottingError(
                    f"{column} must be blank for prompt {original_index}"
                )
            continue
        if observed.isna().any() or observed.nunique(dropna=False) != 1:
            raise PlottingError(
                f"{column} must be repeated for prompt {original_index}"
            )
        if not math.isclose(
            float(observed.iloc[0]), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise PlottingError(f"{column} differs for prompt {original_index}")


def _scatter_figure(
    frame: pd.DataFrame,
    statistics: AnalysisStatistics,
) -> Figure:
    """Render saved category observations; only display order may change."""
    with proximity_style.style_context(PLOT_STYLE):
        figure, axes = plt.subplots(figsize=proximity_style.CANVAS_SIZE_INCHES)
        try:
            proximity_style.apply_layout(figure, axes)
            order = proximity_style.display_order(frame)
            displayed = frame.iloc[order]
            kinds = displayed.get("kind", pd.Series(index=displayed.index, dtype=object)).map(_plot_kind)
            if not displayed.empty:
                axes.scatter(
                    displayed["l2_norm"].to_numpy(dtype=np.float64),
                    displayed["sscd"].to_numpy(dtype=np.float64),
                    s=proximity_style.SCATTER_SIZE,
                    color=[proximity_style.COLORS[kind] for kind in kinds],
                    alpha=proximity_style.SCATTER_ALPHA,
                    marker="o", edgecolors="none", rasterized=True, zorder=2,
                )
            categories = [kind for kind in _KIND_ORDER if kinds.eq(kind).any()]
            handles = [Line2D(
                [], [], linestyle="none", marker="o",
                markersize=math.sqrt(proximity_style.SCATTER_SIZE),
                markerfacecolor=proximity_style.COLORS[kind], markeredgecolor="none",
                alpha=1., label=category_legend_label(kind),
            ) for kind in categories]
            proximity_style.add_legend(axes, handles, categories)
            axes.set_xlabel(X_AXIS_LABEL, fontsize=proximity_style.AXIS_LABEL_FONT_SIZE)
            axes.set_ylabel(Y_AXIS_LABEL, fontsize=proximity_style.AXIS_LABEL_FONT_SIZE)
            proximity_style.add_summary_box(axes, _summary_text(statistics))
            figure._proximity_style_version = proximity_style.STYLE_VERSION
            return figure
        except BaseException:
            plt.close(figure)
            raise


def covariance_ellipse(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    standard_deviations: float,
    **properties: object,
) -> Ellipse:
    """Return a covariance contour whose angle follows its principal axis."""

    center = np.asarray(mean, dtype=np.float64)
    matrix = np.asarray(covariance, dtype=np.float64)
    if center.shape != (2,) or matrix.shape != (2, 2):
        raise PlottingError("GMM ellipse parameters have invalid shapes")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(matrix)):
        raise PlottingError("GMM ellipse parameters must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise PlottingError("GMM ellipse covariance is not symmetric")
    if (
        isinstance(standard_deviations, bool)
        or not math.isfinite(standard_deviations)
        or standard_deviations <= 0.0
    ):
        raise PlottingError("GMM ellipse scale must be finite and positive")
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError as error:
        raise PlottingError("GMM ellipse covariance decomposition failed") from error
    if np.any(eigenvalues <= 0.0):
        raise PlottingError("GMM ellipse covariance is not positive definite")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    direction = eigenvectors[:, order[0]]
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    width, height = 2.0 * standard_deviations * np.sqrt(eigenvalues)
    return Ellipse(
        xy=center,
        width=float(width),
        height=float(height),
        angle=angle,
        **properties,
    )


def _gmm_fit_plot_data(
    frame: pd.DataFrame,
    configuration: Mapping[str, object],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not isinstance(frame, pd.DataFrame):
        raise PlottingError("GMM fit input must be a pandas DataFrame")
    if not isinstance(configuration, Mapping):
        raise PlottingError("GMM selection configuration is invalid")
    fit = configuration.get("gmm_fit")
    if not isinstance(fit, Mapping):
        raise PlottingError("GMM selection configuration is missing gmm_fit")
    if fit.get("covariance_type") != "full":
        raise PlottingError("GMM fit covariance_type must be full")
    if fit.get("feature_names") != ["l2_norm", "sscd"]:
        raise PlottingError("GMM fit feature names are invalid")
    if fit.get("component_names") != list(COMPONENT_NAMES):
        raise PlottingError("GMM fit component names are invalid")

    required = {"observation_status", "gmm_component", "l2_norm", "sscd"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PlottingError(
            "selection table is missing columns: " + ", ".join(missing)
        )
    plotted = frame.loc[frame["observation_status"].eq("complete")].copy()
    _validate_measurements(plotted)
    components = plotted["gmm_component"].astype(str)
    if set(components) != set(COMPONENT_NAMES):
        raise PlottingError("GMM fit rows have invalid component assignments")
    usable_count = fit.get("usable_observation_count")
    if (
        isinstance(usable_count, bool)
        or not isinstance(usable_count, int)
        or usable_count != len(plotted)
    ):
        raise PlottingError("GMM fit observation count is inconsistent")

    try:
        feature_mean = np.asarray(fit["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(fit["feature_scale"], dtype=np.float64)
        means = np.asarray(fit["means_standardized"], dtype=np.float64)
        covariances = np.asarray(
            fit["covariances_standardized"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PlottingError("GMM fit arrays are invalid") from error
    if (
        feature_mean.shape != (2,)
        or feature_scale.shape != (2,)
        or means.shape != (2, 2)
        or covariances.shape != (2, 2, 2)
        or not all(
            np.all(np.isfinite(values))
            for values in (feature_mean, feature_scale, means, covariances)
        )
        or np.any(feature_scale <= 0.0)
    ):
        raise PlottingError("GMM fit arrays are invalid")

    raw_means = feature_mean + means * feature_scale
    raw_covariances = (
        covariances
        * feature_scale[None, :, None]
        * feature_scale[None, None, :]
    )
    for mean, covariance in zip(raw_means, raw_covariances, strict=True):
        covariance_ellipse(
            mean,
            covariance,
            standard_deviations=1.0,
        )
    return plotted, raw_means, raw_covariances


def _gmm_fit_figure(
    frame: pd.DataFrame,
    configuration: Mapping[str, object],
) -> Figure:
    """Style the saved two-component fit without changing its rows or geometry."""
    plotted, means, covariances = _gmm_fit_plot_data(frame, configuration)
    with proximity_style.style_context(PLOT_STYLE):
        figure, axes = plt.subplots(figsize=proximity_style.CANVAS_SIZE_INCHES)
        try:
            proximity_style.apply_layout(figure, axes)
            displayed = plotted.iloc[proximity_style.display_order(plotted)]
            axes.scatter(
                displayed["l2_norm"].to_numpy(dtype=np.float64),
                displayed["sscd"].to_numpy(dtype=np.float64),
                s=proximity_style.SCATTER_SIZE,
                color=[proximity_style.GMM_COMPONENT_COLORS[component]
                       for component in displayed["gmm_component"]],
                alpha=proximity_style.SCATTER_ALPHA,
                edgecolors="none", rasterized=True, zorder=2,
            )
            handles: list[Line2D] = []
            for component_index, component in enumerate(COMPONENT_NAMES):
                color = proximity_style.GMM_COMPONENT_COLORS[component]
                for standard_deviations in _GMM_ELLIPSE_STANDARD_DEVIATIONS:
                    axes.add_patch(
                        covariance_ellipse(
                            means[component_index], covariances[component_index],
                            standard_deviations=standard_deviations,
                            fill=False, edgecolor=color, linewidth=1.0,
                            alpha=0.9, zorder=3,
                        )
                    )
                axes.plot(
                    means[component_index, 0], means[component_index, 1],
                    linestyle="none", marker="x", markersize=6,
                    markeredgewidth=1.25, color=color, zorder=4,
                )
                label = "Low SSCD" if component == "low_sscd_mode" else "High SSCD"
                handles.append(
                    Line2D(
                        [], [], color=color, marker="o", linestyle="-",
                        markersize=math.sqrt(proximity_style.SCATTER_SIZE),
                        linewidth=1.0, label=label,
                    )
                )
            axes.set_xlabel(X_AXIS_LABEL)
            axes.set_ylabel(Y_AXIS_LABEL)
            proximity_style.add_legend(
                axes, handles, COMPONENT_NAMES,
                category_colors=proximity_style.GMM_COMPONENT_COLORS,
            )
            axes.autoscale_view()
            figure._proximity_style_version = proximity_style.GMM_STYLE_VERSION
            return figure
        except BaseException:
            plt.close(figure)
            raise


def add_prompt_curves(
    axis: object,
    frame: pd.DataFrame,
    *,
    alpha: float = CURVE_ALPHA,
    linewidth: float = CURVE_LINE_WIDTH,
    category_column: str | None = None,
    category_styles: Mapping[str, object] | None = None,
) -> LineCollection:
    """Connect cross-seed observations only within each prompt.

    Rows are ordered by L2 distance (and then seed, when available). Each
    segment is colored by the mean SSCD of its two endpoints. These curves are
    cross-seed proximity profiles, not sampling trajectories. When categories
    are supplied, a segment uses its right endpoint's category style.
    """

    required = {"original_index", "l2_norm", "sscd"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PlottingError("proximity table is missing columns: " + ", ".join(missing))
    if (category_column is None) != (category_styles is None):
        raise PlottingError(
            "category_column and category_styles must be supplied together"
        )
    if category_column is not None and category_column not in frame.columns:
        raise PlottingError(f"proximity table is missing columns: {category_column}")
    if category_styles is not None and not category_styles:
        raise PlottingError("category_styles must not be empty")

    segments: list[np.ndarray] = []
    colors: list[float] = []
    linestyles: list[object] = []
    for _original_index, group in frame.groupby(
        "original_index", sort=False, dropna=False
    ):
        order = ["l2_norm"]
        if "seed" in group.columns:
            order.append("seed")
        ordered = group.sort_values(order, kind="stable")
        points = ordered[["l2_norm", "sscd"]].to_numpy(dtype=np.float64)
        if len(points) < 2:
            continue
        prompt_segments = np.stack((points[:-1], points[1:]), axis=1)
        segments.extend(prompt_segments)
        if category_column is not None and category_styles is not None:
            categories = ordered[category_column].astype(str).to_numpy(dtype=object)
            unknown = sorted(set(categories) - set(category_styles))
            if unknown:
                raise PlottingError(
                    f"{category_column} contains unsupported categories: "
                    + ", ".join(unknown)
                )
            linestyles.extend(category_styles[str(value)] for value in categories[1:])
        colors.extend(((points[:-1, 1] + points[1:, 1]) / 2.0).tolist())
    norm = Normalize(vmin=SSCD_COLOR_RANGE[0], vmax=SSCD_COLOR_RANGE[1], clip=True)
    collection = LineCollection(
        segments,
        array=np.asarray(colors, dtype=np.float64),
        cmap="viridis",
        norm=norm,
        linewidths=linewidth,
        linestyles=linestyles if category_column is not None else "solid",
        alpha=alpha,
        rasterized=True,
        zorder=2,
    )
    axis.add_collection(collection, autolim=True)
    axis.autoscale_view()
    return collection


def add_sscd_colorbar(
    figure: Figure,
    axis: object,
    values: pd.Series | np.ndarray,
    *,
    cmap="viridis", norm=None, ticks=None, cax=None, size="5%", pad=0.1,
    labelsize=None, ticksize=None, rasterized=None,
) -> object:
    """Add an opaque SSCD colorbar; optional styling preserves legacy defaults."""

    if norm is None:
        norm = Normalize(vmin=SSCD_COLOR_RANGE[0], vmax=SSCD_COLOR_RANGE[1], clip=True)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.asarray(values, dtype=np.float64))
    # A regular figure colorbar follows the allocated subplot rectangle, which
    # can be taller than an equal-aspect scatter. Share the axes divider so the
    # colorbar tracks the actual plot box in both raster and vector exports.
    if cax is None:
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size=size, pad=pad)
    cax.set_label("<colorbar>")
    colorbar = figure.colorbar(mappable, cax=cax, ticks=ticks)
    if colorbar.solids is not None:
        colorbar.solids.set_alpha(COLORBAR_ALPHA)
        if rasterized is not None:
            colorbar.solids.set_rasterized(rasterized)
    colorbar.set_label("SSCD", fontsize=AXIS_LABEL_FONT_SIZE if labelsize is None else labelsize)
    colorbar.ax.tick_params(labelsize=AXIS_NUMBER_FONT_SIZE if ticksize is None else ticksize)
    return colorbar


def _summary_text(statistics: AnalysisStatistics) -> str:
    denominator = statistics.evaluable_selected_prompts
    fraction = (
        "undefined"
        if statistics.negative_spearman_fraction is None
        else f"{statistics.negative_spearman_fraction:.1%}"
    )
    median = (
        "undefined"
        if statistics.median_spearman is None
        else f"{statistics.median_spearman:.3f}"
    )
    evaluable_note = " evaluable" if denominator != statistics.total_selected_prompts else ""
    return (
        f"#Prompts: {statistics.total_selected_prompts}\n"
        + rf"Median $\rho$: {median}" + "\n"
        + rf"$\rho < 0$: {statistics.negative_spearman_prompts}/{denominator} "
        + f"({fraction}){evaluable_note}"
    )



def _export_options(figures, supplied):
    """Validate optional per-artifact presentation settings before any writes."""
    if supplied is None:
        return {}
    if not isinstance(supplied, Mapping):
        raise PlottingError("figure export options must map relative filenames to settings")
    known = {name for _figure, filenames in figures for name in filenames.values()}
    if set(supplied) - known:
        raise PlottingError("figure export options reference unknown output filenames")
    result = {}
    for name, settings in supplied.items():
        if not isinstance(settings, Mapping) or set(settings) - {"dpi", "bbox_inches"}:
            raise PlottingError("only dpi and bbox_inches may be overridden per figure artifact")
        options = dict(settings)
        if "dpi" in options and (isinstance(options["dpi"], bool)
                or not isinstance(options["dpi"], (int, float))
                or not math.isfinite(options["dpi"]) or options["dpi"] <= 0):
            raise PlottingError("figure export dpi must be finite and positive")
        if "bbox_inches" in options:
            box = options["bbox_inches"]
            if not (isinstance(box, str) and box == "tight") and not (
                isinstance(box, BboxBase) and np.isfinite(box.extents).all()
                and box.width > 0 and box.height > 0
            ):
                raise PlottingError("figure export bounds require tight or a finite positive inch Bbox")
        result[name] = options
    return result


def _publication_formats(formats):
    """Normalize a nonempty, duplicate-free PNG/PDF subset to export order."""
    if (not isinstance(formats, Sequence) or isinstance(formats, (str, bytes))
            or not formats or any(not isinstance(value, str) for value in formats)
            or len(set(formats)) != len(formats)
            or not set(formats).issubset(_FIGURE_FORMATS)):
        raise PlottingError("figure formats must be a nonempty, unique subset of png and pdf")
    return tuple(value for value in _FIGURE_FORMATS if value in formats)


def _publication_parents(output, figures, formats):
    """Validate only requested destinations before any directories or files exist."""
    destinations: set[Path] = set()
    parents: set[Path] = {output}
    for _figure, filenames in figures:
        if not isinstance(filenames, Mapping) or set(filenames) != set(formats):
            raise PlottingError("figure publication filenames must exactly match requested formats: " + ", ".join(formats))
        paths = {}
        for file_format in formats:
            try:
                relative = Path(filenames[file_format])
            except (TypeError, ValueError) as error:
                raise PlottingError("figure destination must be a relative filename") from error
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix != "." + file_format
                or not relative.stem
            ):
                raise PlottingError(f"unsafe figure destination: {relative}")
            destination = output / relative
            if destination in destinations:
                raise PlottingError(f"duplicate figure destination: {relative}")
            if destination.is_symlink():
                raise PlottingError(f"symlink figure destination: {destination}")
            if destination.exists() and not destination.is_file():
                raise PlottingError(f"figure destination is not a file: {destination}")
            destinations.add(destination)
            parents.add(destination.parent)
            paths[file_format] = relative.with_suffix("")
        if len(set(paths.values())) != 1:
            raise PlottingError("PNG and PDF figure destinations must share a stem")
    for parent in parents:
        for ancestor in (parent, *parent.parents):
            if ancestor.is_symlink():
                raise PlottingError(f"symlink figure parent: {ancestor}")
            if ancestor.exists() and not ancestor.is_dir():
                raise PlottingError(f"figure parent is not a directory: {ancestor}")
    return parents


def publish_figures(
    output: Path,
    figures: Sequence[tuple[Figure, Mapping[str, str]]],
    *,
    progress: Callable[[str, str], None] | None = None,
    export_options: Mapping[str, Mapping[str, object]] | None = None,
    formats: Sequence[str] = _FIGURE_FORMATS,
) -> None:
    """Publish requested artifacts through the existing recoverable transaction.

    By default filenames must provide same-stem PNG/PDF pairs. An explicit
    formats=("pdf",) or formats=("png",) selects one format, and each filename
    map must contain exactly those requested keys. Unrequested files are never
    read, replaced or removed. Nested names must be safe relative paths. The
    proximity entry points explicitly select PDF-only output through the private
    publisher. Callers own figure lifetime and must close figures in finally.
    Optional progress receives (relative filename, "saving" or "saved") around
    each export; it does not change the paired rollback/install transaction.
    Per-filename export_options may override only dpi and bbox_inches; callers
    must supply padding inside an explicit inch Bbox. Unspecified exports retain
    the existing 150 DPI, tight bounding box and configured padding.
    """
    formats = _publication_formats(formats)
    output = Path(output).absolute()
    parents = _publication_parents(output, figures, formats)
    overrides = _export_options(figures, export_options)
    for parent in sorted(parents, key=lambda path: len(path.parts)):
        parent.mkdir(parents=True, exist_ok=True)
    options = {}
    if progress is not None:
        options["progress"] = progress
    if export_options is not None:
        options["export_options"] = overrides
    if formats != _FIGURE_FORMATS:
        options["formats"] = formats
    _publish_figures(output, figures, **options)


def _publish_figures(
    output: Path,
    figures: Sequence[tuple[Figure, Mapping[str, str]]],
    *,
    progress: Callable[[str, str], None] | None = None,
    export_options: Mapping[str, Mapping[str, object]] | None = None,
    formats: Sequence[str] = _FIGURE_FORMATS,
) -> None:
    """Render every artifact, then install all files as one recoverable set."""

    formats = _publication_formats(formats)
    output = Path(output).absolute()
    parents = _publication_parents(output, figures, formats)
    overrides = _export_options(figures, export_options)
    for parent in sorted(parents, key=lambda path: len(path.parts)):
        parent.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path | None, Path]] = []
    committed = False
    try:
        for figure, filenames in figures:
            for file_format in formats:
                destination = output / filenames[file_format]
                temporary = _temporary_sibling(destination)
                staged.append((temporary, destination))
                if progress is not None:
                    progress(filenames[file_format], "saving")
                options = {"format": file_format, "dpi": FIGURE_DPI,
                           "bbox_inches": "tight", "pad_inches": FIGURE_PAD_INCHES}
                options.update(overrides.get(filenames[file_format], {}))
                figure.savefig(temporary, **options)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                if progress is not None:
                    progress(filenames[file_format], "saved")
        for _temporary, destination in staged:
            if destination.exists() and not (
                destination.is_file() or destination.is_symlink()
            ):
                raise PlottingError(
                    f"derived figure destination is not a regular file: {destination}"
                )
        for temporary, destination in staged:
            backup = None
            if destination.is_file() or destination.is_symlink():
                backup = _temporary_sibling(destination)
                os.replace(destination, backup)
            backups.append((backup, destination))
            os.replace(temporary, destination)
        committed = True
    except BaseException:
        for backup, destination in reversed(backups):
            if destination.is_file() or destination.is_symlink():
                destination.unlink()
            elif destination.exists():
                raise PlottingError(
                    "cannot restore derived figure because its destination became "
                    f"unsafe: {destination}"
                )
            if backup is not None:
                os.replace(backup, destination)
        raise
    finally:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)
        if committed:
            for backup, _destination in backups:
                if backup is not None:
                    backup.unlink(missing_ok=True)


def _plot_kind(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in _KIND_COLORS:
            return normalized
    return "Other / unlabeled"


def _finite_series(values: pd.Series) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
