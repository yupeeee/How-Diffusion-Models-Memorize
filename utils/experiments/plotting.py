"""Write proximity/SSCD tables, correlations, and scatter plots.

This module is deliberately statistical and presentational.  It does not know
how target-pair selection is constructed and it has no model, sampler, VAE, or
SSCD feature-extractor dependency.
"""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import pandas as pd
from tqdm import tqdm

from utils.common.io import (
    atomic_write_frame_csv,
    atomic_write_frame_parquet,
)

__all__ = [
    "AnalysisStatistics",
    "CorrelationStatistics",
    "PlottingError",
    "compute_correlations",
    "write_analysis_outputs",
]

_TYPE_ORDER = ("MV", "TV", "RV", "N", "Other / unlabeled")
_TYPE_COLORS = {
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
CORRELATION_FONT_SIZE = 10
LEGEND_MARKER_ALPHA = 1.0
SCATTER_SIZE = 8
SCATTER_ALPHA = 0.35
FIGURE_PAD_INCHES = 0.05
X_AXIS_LABEL = r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|_2/\sqrt{d}$"
Y_AXIS_LABEL = "SSCD"

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
    """A paired table cannot be validated, correlated, or plotted."""


@dataclass(frozen=True, slots=True)
class CorrelationStatistics:
    """Correlation results for one explicitly named point set."""

    prompts: int
    points: int
    pearson: float | None
    spearman: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "prompts": self.prompts,
            "points": self.points,
            "pearson": self.pearson,
            "spearman": self.spearman,
        }


@dataclass(frozen=True, slots=True)
class AnalysisStatistics:
    """Statistics returned after publishing the two analysis views."""

    all: CorrelationStatistics
    selected: CorrelationStatistics
    diagnostic_subsets: Mapping[str, CorrelationStatistics]

    def as_dict(self) -> dict[str, dict[str, int | float | None]]:
        values = {
            "all": self.all.as_dict(),
            "selected": self.selected.as_dict(),
        }
        values.update({name: value.as_dict() for name, value in self.diagnostic_subsets.items()})
        return values


def _view_progress(
    views: Mapping[str, pd.DataFrame],
) -> Iterable[tuple[str, pd.DataFrame]]:
    """Keep table and figure publication visible, including in captured logs."""

    return tqdm(
        views.items(),
        total=len(views),
        desc="[Proximity 2/2] Analysis views",
        unit="view",
        dynamic_ncols=True,
        leave=True,
        disable=False,
    )


def compute_correlations(frame: pd.DataFrame) -> CorrelationStatistics:
    """Correlate dimension-normalized terminal L2 with cached SSCD.

    Pearson and Spearman are computed here, and nowhere in selection or cache
    processing.  Undefined correlations (fewer than two points or a constant
    input) are represented by ``None``.
    """

    _validate_frame(frame)
    prompts = int(frame["original_index"].astype(str).nunique())
    points = int(len(frame))
    if points < 2:
        return CorrelationStatistics(prompts, points, None, None)
    x = pd.to_numeric(frame["latent_rmse"], errors="raise").astype("float64")
    y = pd.to_numeric(frame["sscd_cosine_similarity"], errors="raise").astype("float64")
    if not _finite_series(x) or not _finite_series(y):
        raise PlottingError("correlation inputs must be finite")
    if x.nunique(dropna=False) < 2 or y.nunique(dropna=False) < 2:
        return CorrelationStatistics(prompts, points, None, None)
    pearson = _finite_or_none(x.corr(y, method="pearson"))
    spearman = _finite_or_none(x.rank(method="average").corr(y.rank(method="average"), method="pearson"))
    return CorrelationStatistics(prompts, points, pearson, spearman)


def write_analysis_outputs(
    output_directory: str | Path,
    *,
    paired_all: pd.DataFrame,
    paired_selected: pd.DataFrame,
) -> AnalysisStatistics:
    """Atomically publish the all/selected table and figure pairs."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    views = {
        "all": paired_all,
        "selected": paired_selected,
    }
    statistics: dict[str, CorrelationStatistics] = {}
    for name, frame in _view_progress(views):
        _validate_frame(frame)
        statistics[name] = compute_correlations(frame)
        atomic_write_frame_csv(frame, output / f"paired_{name}.csv")
        atomic_write_frame_parquet(frame, output / f"paired_{name}.parquet")
        with matplotlib.rc_context(PLOT_STYLE):
            figure = _scatter_figure(frame, statistics[name])
            _save_figure_pair(
                figure,
                output / f"proximity_vs_sscd_{name}.png",
                output / f"proximity_vs_sscd_{name}.pdf",
            )
    selected_tv = _type_mask(paired_selected, "TV")
    diagnostic_subsets = {
        "non_tv_selected": compute_correlations(paired_selected.loc[~selected_tv]),
        "selected_tv_all_seeds": compute_correlations(paired_selected.loc[selected_tv]),
    }
    return AnalysisStatistics(
        all=statistics["all"],
        selected=statistics["selected"],
        diagnostic_subsets=diagnostic_subsets,
    )


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise PlottingError("paired analysis input must be a pandas DataFrame")
    required = {
        "original_index",
        "latent_rmse",
        "sscd_cosine_similarity",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PlottingError("paired table is missing columns: " + ", ".join(missing))
    if frame.empty:
        return
    for column in ("latent_rmse", "sscd_cosine_similarity"):
        values = pd.to_numeric(frame[column], errors="raise").astype("float64")
        if not _finite_series(values):
            raise PlottingError(f"{column} must contain only finite values")


def _scatter_figure(
    frame: pd.DataFrame,
    statistics: CorrelationStatistics,
) -> Figure:
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    if frame.empty:
        plotted = pd.Series([], dtype="object")
    else:
        label_column = (
            "webster_overfit_type_normalized"
            if "webster_overfit_type_normalized" in frame.columns
            else "webster_overfit_type_raw"
        )
        if label_column in frame.columns:
            plotted = frame[label_column].map(_plot_type)
        else:
            plotted = pd.Series(["Other / unlabeled"] * len(frame), index=frame.index)
    for label in _TYPE_ORDER:
        subset = frame.loc[plotted == label]
        if subset.empty:
            continue
        axes.scatter(
            subset["latent_rmse"].astype(float),
            subset["sscd_cosine_similarity"].astype(float),
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            color=_TYPE_COLORS[label],
            edgecolors="none",
            label=label,
            rasterized=True,
            zorder=1 if label in {"N", "Other / unlabeled"} else 2,
        )
    axes.set_xlabel(X_AXIS_LABEL, fontsize=TEXT_FONT_SIZE)
    axes.set_ylabel(Y_AXIS_LABEL, fontsize=TEXT_FONT_SIZE)
    axes.tick_params(
        axis="both",
        which="both",
        labelsize=AXIS_NUMBER_FONT_SIZE,
    )
    axes.text(
        0.02,
        0.02,
        (
            f"PCC: {_stat_text(statistics.pearson)}\n"
            f"Spearman rho: {_stat_text(statistics.spearman)}"
        ),
        transform=axes.transAxes,
        ha="left",
        va="bottom",
        fontsize=CORRELATION_FONT_SIZE,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.85,
        },
    )
    if axes.collections:
        legend = axes.legend(
            title="Webster type",
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


def _save_figure_pair(figure: Figure, png: Path, pdf: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    temporary_png = _temporary_sibling(png)
    temporary_pdf = _temporary_sibling(pdf)
    try:
        figure.savefig(
            temporary_png,
            format="png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=FIGURE_PAD_INCHES,
        )
        figure.savefig(
            temporary_pdf,
            format="pdf",
            bbox_inches="tight",
            pad_inches=FIGURE_PAD_INCHES,
        )
        for temporary in (temporary_png, temporary_pdf):
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary_png, png)
        os.replace(temporary_pdf, pdf)
    finally:
        plt.close(figure)
        temporary_png.unlink(missing_ok=True)
        temporary_pdf.unlink(missing_ok=True)


def _plot_type(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in _TYPE_COLORS:
            return normalized
    return "Other / unlabeled"


def _type_mask(frame: pd.DataFrame, label: str) -> pd.Series:
    column = (
        "webster_overfit_type_normalized"
        if "webster_overfit_type_normalized" in frame.columns
        else "webster_overfit_type_raw"
    )
    if column not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index, dtype=bool)
    return frame[column].map(_plot_type).eq(label)


def _finite_series(values: pd.Series) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stat_text(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
