"""Write proximity audit tables and comparable all/selected scatter plots."""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Mapping, Sequence
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
import numpy as np
import pandas as pd

from utils.common.io import atomic_write_frame_csv

__all__ = [
    "AnalysisStatistics",
    "CATEGORY_LINESTYLES",
    "PlottingError",
    "PROXIMITY_FIGURE_FILENAMES",
    "PROXIMITY_FIGURES",
    "add_prompt_curves",
    "add_sscd_colorbar",
    "category_legend_label",
    "write_analysis_outputs",
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
AXIS_NUMBER_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
SUMMARY_FONT_SIZE = 10
CATEGORY_LINESTYLES = (
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    (0, (5, 1, 1, 1, 1, 1)),
)
SCATTER_SIZE = 12
SCATTER_ALPHA = 0.35
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

PLOT_STYLE = {
    "figure.figsize": FIGURE_SIZE,
    "font.family": "STIXGeneral",
    "font.size": TEXT_FONT_SIZE,
    "mathtext.fontset": "stix",
    "axes.labelsize": TEXT_FONT_SIZE,
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
) -> AnalysisStatistics:
    """Publish one seed-level CSV and comparable all/selected PNG/PDF scatter plots.

    Each seed row receives its prompt's experiment-seed Spearman value before
    publication. The saved CSV is then reloaded and validated; both figure
    annotations and returned statistics are computed only from that saved log.
    """

    _validate_analysis(analysis)
    annotated = _annotate_experiment_spearman(analysis)
    output = Path(output_directory)
    csv_path = output / "proximity.csv"
    atomic_write_frame_csv(annotated, csv_path)

    saved_analysis = pd.read_csv(csv_path)
    _validate_analysis(saved_analysis)
    _validate_prompt_spearman(saved_analysis, column=EXPERIMENT_SPEARMAN_COLUMN)
    all_prompts = saved_analysis.copy()
    selected = saved_analysis.loc[saved_analysis["include_prompt"].astype(bool)].copy()
    with matplotlib.rc_context(PLOT_STYLE):
        return _write_scatter_views(
            output,
            all_prompts=all_prompts,
            selected=selected,
            spearman_column=EXPERIMENT_SPEARMAN_COLUMN,
        )


def write_selection_figure(
    selection_directory: str | Path,
) -> AnalysisStatistics:
    """Rebuild comparable all/selected scatter plots from frozen ``selection.csv``.

    The frozen table remains the sole numerical log. The pre-discard view
    includes complete prompt groups regardless of their selection decision;
    prompt groups containing any failed observation or unplottable measurement
    are omitted.
    """

    output = Path(selection_directory)
    csv_path = output / "selection.csv"
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
            output,
            all_prompts=all_prompts,
            selected=selected,
            spearman_column="prompt_spearman",
        )


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
    """Render both views with all-prompt limits, then publish their artifacts."""

    all_statistics = _prompt_statistics(all_prompts, column=spearman_column)
    selected_statistics = _prompt_statistics(selected, column=spearman_column)
    figures: list[Figure] = []
    try:
        all_figure = _scatter_figure(all_prompts, all_statistics)
        figures.append(all_figure)
        selected_figure = _scatter_figure(selected, selected_statistics)
        figures.append(selected_figure)
        all_axes, selected_axes = all_figure.axes[0], selected_figure.axes[0]
        selected_axes.set_xlim(all_axes.get_xlim())
        selected_axes.set_ylim(all_axes.get_ylim())
        _publish_figures(
            output,
            (
                (selected_figure, PROXIMITY_FIGURES["selected"]),
                (all_figure, PROXIMITY_FIGURES["all_prompts"]),
            ),
        )
    finally:
        for figure in figures:
            plt.close(figure)
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
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    kinds = frame.get("kind", pd.Series(index=frame.index, dtype=object)).map(
        _plot_kind
    )
    handles = []
    for kind in _KIND_ORDER:
        group = frame.loc[kinds.eq(kind)]
        if group.empty:
            continue
        color = _KIND_COLORS[kind]
        axes.scatter(
            group["l2_norm"].to_numpy(dtype=np.float64),
            group["sscd"].to_numpy(dtype=np.float64),
            s=SCATTER_SIZE,
            color=color,
            alpha=SCATTER_ALPHA,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=math.sqrt(SCATTER_SIZE),
                markerfacecolor=color,
                markeredgecolor="none",
                alpha=1.0,
                label=category_legend_label(kind),
            )
        )
    if handles:
        axes.legend(handles=handles, fontsize=LEGEND_FONT_SIZE)
    axes.set_xlabel(X_AXIS_LABEL, fontsize=TEXT_FONT_SIZE)
    axes.set_ylabel(Y_AXIS_LABEL, fontsize=TEXT_FONT_SIZE)
    axes.tick_params(axis="both", which="both", labelsize=AXIS_NUMBER_FONT_SIZE)
    axes.text(
        0.02,
        0.02,
        _summary_text(statistics),
        transform=axes.transAxes,
        ha="left",
        va="bottom",
        fontsize=SUMMARY_FONT_SIZE,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.85,
        },
    )
    axes.grid(True, which="both", linewidth=0.6, alpha=0.18)
    figure.tight_layout()
    return figure


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
) -> object:
    """Add the shared opaque [0, 1] SSCD colorbar."""

    norm = Normalize(vmin=SSCD_COLOR_RANGE[0], vmax=SSCD_COLOR_RANGE[1], clip=True)
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array(np.asarray(values, dtype=np.float64))
    colorbar = figure.colorbar(mappable, ax=axis)
    if colorbar.solids is not None:
        colorbar.solids.set_alpha(COLORBAR_ALPHA)
    colorbar.set_label("SSCD", fontsize=TEXT_FONT_SIZE)
    colorbar.ax.tick_params(labelsize=AXIS_NUMBER_FONT_SIZE)
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
    return (
        f"#prompts: {statistics.total_selected_prompts}\n"
        rf"$\rho < 0$: {statistics.negative_spearman_prompts}/{denominator} "
        f"({fraction})\n"
        rf"Median $\rho$: {median}"
    )


def _publish_figures(
    output: Path,
    figures: Sequence[tuple[Figure, Mapping[str, str]]],
) -> None:
    """Render every artifact, then install all files as one recoverable set."""

    output.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path | None, Path]] = []
    committed = False
    try:
        for figure, filenames in figures:
            for file_format in _FIGURE_FORMATS:
                destination = output / filenames[file_format]
                temporary = _temporary_sibling(destination)
                staged.append((temporary, destination))
                figure.savefig(
                    temporary,
                    format=file_format,
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=FIGURE_PAD_INCHES,
                )
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
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
