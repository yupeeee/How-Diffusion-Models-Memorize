"""Write the proximity audit table and selected-prompt scatter plot."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import pandas as pd

from utils.common.io import atomic_write_frame_csv

__all__ = [
    "AnalysisStatistics",
    "PlottingError",
    "write_analysis_outputs",
    "write_selection_figure",
]

_KIND_ORDER = ("MV", "TV", "RV", "N", "Other / unlabeled")
_KIND_COLORS = {
    "MV": "#D55E00",
    "TV": "#0072B2",
    "RV": "#009E73",
    "N": "#CC79A7",
    "Other / unlabeled": "#7F7F7F",
}
FIGURE_SIZE = (4.0, 4.0)
TEXT_FONT_SIZE = 15
AXIS_NUMBER_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
SUMMARY_FONT_SIZE = 10
LEGEND_MARKER_ALPHA = 1.0
SCATTER_SIZE = 8
SCATTER_ALPHA = 0.35
FIGURE_PAD_INCHES = 0.05
X_AXIS_LABEL = r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|_2$"
Y_AXIS_LABEL = "SSCD"
EXPERIMENT_SPEARMAN_COLUMN = "experiment_prompt_spearman"

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
    "legend.title_fontsize": LEGEND_FONT_SIZE,
}


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
    """Publish one complete seed-level CSV and one selected-prompt PNG.

    Each seed row receives its prompt's experiment-seed Spearman value before
    publication. The saved CSV is then reloaded and validated; both the figure
    annotation and returned statistics are computed only from that saved log.
    """

    _validate_analysis(analysis)
    annotated = _annotate_experiment_spearman(analysis)
    output = Path(output_directory)
    csv_path = output / "proximity.csv"
    atomic_write_frame_csv(annotated, csv_path)

    saved_analysis = pd.read_csv(csv_path)
    _validate_analysis(saved_analysis)
    _validate_prompt_spearman(saved_analysis, column=EXPERIMENT_SPEARMAN_COLUMN)
    selected = saved_analysis.loc[saved_analysis["include_prompt"].astype(bool)]
    statistics = _prompt_statistics(selected, column=EXPERIMENT_SPEARMAN_COLUMN)
    with matplotlib.rc_context(PLOT_STYLE):
        figure = _scatter_figure(selected, statistics)
        _save_png(figure, output / "proximity_vs_sscd.png")
    return statistics


def write_selection_figure(
    selection_directory: str | Path,
) -> AnalysisStatistics:
    """Rebuild the selected-prompt scatter from frozen ``selection.csv``.

    The frozen table remains the sole numerical log. Excluded or unusable
    prompt rows stay in that audit table but are not plotted.
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
    selected = saved_selection.loc[
        saved_selection["include_prompt"].astype(bool)
    ].copy()
    _validate_measurements(selected)
    _validate_prompt_spearman(selected, column="prompt_spearman")
    statistics = _prompt_statistics(selected, column="prompt_spearman")
    with matplotlib.rc_context(PLOT_STYLE):
        figure = _scatter_figure(selected, statistics)
        _save_png(figure, output / "proximity_vs_sscd.png")
    return statistics


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
    spearman = pd.to_numeric(prompt_rows[column], errors="raise").astype("float64")
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
    if (values.notna() & numeric.isna()).any() or not _finite_series(numeric.dropna()):
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
    if frame.empty or "kind" not in frame.columns:
        plotted = pd.Series(
            ["Other / unlabeled"] * len(frame), index=frame.index, dtype="object"
        )
    else:
        plotted = frame["kind"].map(_plot_kind)
    for label in _KIND_ORDER:
        subset = frame.loc[plotted == label]
        if subset.empty:
            continue
        axes.scatter(
            subset["l2_norm"].astype(float),
            subset["sscd"].astype(float),
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            color=_KIND_COLORS[label],
            edgecolors="none",
            label=label,
            rasterized=True,
            zorder=1 if label in {"N", "Other / unlabeled"} else 2,
        )
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
    if axes.collections:
        legend = axes.legend(
            title="Kind",
            loc="best",
            frameon=False,
            markerscale=1.5,
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_FONT_SIZE,
        )
        for handle in legend.legend_handles:
            handle.set_alpha(LEGEND_MARKER_ALPHA)
    axes.grid(True, which="both", linewidth=0.6, alpha=0.18)
    figure.tight_layout()
    return figure


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
        f"Selected prompts: {statistics.total_selected_prompts}\n"
        f"Evaluable prompts: {denominator}\n"
        f"rho < 0: {statistics.negative_spearman_prompts}/{denominator} "
        f"({fraction})\n"
        f"Median rho: {median}"
    )


def _save_png(figure: Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=FIGURE_PAD_INCHES,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


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
