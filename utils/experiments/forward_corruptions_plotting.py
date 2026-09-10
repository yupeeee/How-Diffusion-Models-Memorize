"""Validate and plot the forward-corruptions empirical illustration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np
import pandas as pd
import torch

from utils.common.io import CacheIOError, safe_torch_load
from utils.experiments.plotting import (
    AXIS_NUMBER_FONT_SIZE,
    CURVE_LINE_WIDTH as LINE_WIDTH,
    FIGURE_SIZE,
    LEGEND_FONT_SIZE,
    PLOT_STYLE,
    TEXT_FONT_SIZE,
    _publish_figures,
)

CSV_NAME = "observations.csv"
FIGURE_STEM = "forward_corruptions_generated_states"
OBSERVATION_COLUMNS = (
    "panel",
    "record_id",
    "original_index",
    "generation_seed",
    "step_index",
    "timestep",
    "is_final_clean_state",
    "denoising_progress",
    "alpha_t",
    "sigma_t",
    "forward_distance",
    "generated_distance",
)
SUMMARY_COLUMNS = (
    "panel",
    "step_index",
    "denoising_progress",
    "timestep",
    "alpha_t",
    "sigma_t",
    "forward_median",
    "forward_min",
    "forward_max",
    "generated_median",
    "generated_min",
    "generated_max",
)

PANEL_ORDER = ("memorized", "normal")
NUM_SEEDS = 20
RANGE_ALPHA = 0.2

_FORWARD_LABEL = "Forward-corrupted target"
_GENERATED_LABEL = "Generated state"
FIGURE_FILENAMES: Mapping[str, Mapping[str, str]] = {
    panel: {
        file_format: f"{FIGURE_STEM}_{panel}.{file_format}"
        for file_format in ("png", "pdf")
    }
    for panel in PANEL_ORDER
}
# Only the two superseded outputs owned by this experiment are removed.
_RETIRED_FIGURE_FILENAMES = (f"{FIGURE_STEM}.png", f"{FIGURE_STEM}.pdf")

DECODED_GALLERY_SIZE = (12.0, 4.0)
DECODED_FIGURE_FILENAMES: Mapping[str, Mapping[str, str]] = {
    panel: {
        file_format: f"{FIGURE_STEM}_{panel}_decoded.{file_format}"
        for file_format in ("png", "pdf")
    }
    for panel in PANEL_ORDER
}
_DECODED_SCHEMA_VERSION = 1
_DECODED_IMAGE_COUNT = 10
_DECODED_ROW_LABELS = ("Forward", "Generated", "Absolute difference")



class ForwardCorruptionsPlottingError(RuntimeError):
    """The saved forward-corruptions observations are invalid or unsafe."""


@dataclass(frozen=True)
class _DecodedPanel:
    record_id: str
    forward_images: torch.Tensor
    generated_images: torch.Tensor


@dataclass(frozen=True)
class _DecodedStates:
    num_inference_steps: int
    step_indices: tuple[int, ...]
    generation_seed: int
    panels: Mapping[str, _DecodedPanel]


def _payload_integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ForwardCorruptionsPlottingError(
            f"decoded-state {label} must be an integer >= {minimum}"
        )
    return value


def _decoded_image_tensor(value: object, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ForwardCorruptionsPlottingError(
            f"decoded-state {label} must be a Torch tensor"
        )
    if (
        value.dtype != torch.uint8
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.ndim != 4
        or value.shape[0] != _DECODED_IMAGE_COUNT
        or value.shape[-1] != 3
        or value.shape[1] < 1
        or value.shape[2] < 1
    ):
        raise ForwardCorruptionsPlottingError(
            f"decoded-state {label} must be a CPU RGB uint8 tensor with shape "
            f"[{_DECODED_IMAGE_COUNT}, H, W, 3]"
        )
    return value


def _validate_decoded_states_payload(payload: object) -> _DecodedStates:
    """Validate and normalize one safely loaded decoded-state gallery payload."""

    if not isinstance(payload, Mapping):
        raise ForwardCorruptionsPlottingError(
            "decoded-state artifact must contain a mapping"
        )
    schema_version = _payload_integer(
        payload.get("schema_version"), "schema_version", minimum=1
    )
    if schema_version != _DECODED_SCHEMA_VERSION:
        raise ForwardCorruptionsPlottingError(
            f"decoded-state schema_version must equal {_DECODED_SCHEMA_VERSION}"
        )
    num_inference_steps = _payload_integer(
        payload.get("num_inference_steps"), "num_inference_steps", minimum=1
    )
    generation_seed = _payload_integer(
        payload.get("generation_seed"), "generation_seed", minimum=0
    )
    raw_steps = payload.get("step_indices")
    if not isinstance(raw_steps, (list, tuple)) or len(raw_steps) != _DECODED_IMAGE_COUNT:
        raise ForwardCorruptionsPlottingError(
            f"decoded-state step_indices must contain exactly {_DECODED_IMAGE_COUNT} values"
        )
    step_indices = tuple(
        _payload_integer(value, "step_indices value", minimum=0)
        for value in raw_steps
    )
    if (
        step_indices[0] != 0
        or step_indices[-1] != num_inference_steps
        or any(left >= right for left, right in zip(step_indices, step_indices[1:]))
    ):
        raise ForwardCorruptionsPlottingError(
            "decoded-state step_indices must be unique and strictly increasing "
            "from 0 through num_inference_steps"
        )

    raw_panels = payload.get("panels")
    if not isinstance(raw_panels, Mapping) or set(raw_panels) != set(PANEL_ORDER):
        raise ForwardCorruptionsPlottingError(
            "decoded-state panels must contain exactly memorized and normal"
        )
    panels: dict[str, _DecodedPanel] = {}
    common_shape: tuple[int, ...] | None = None
    for panel in PANEL_ORDER:
        raw_panel = raw_panels[panel]
        if not isinstance(raw_panel, Mapping):
            raise ForwardCorruptionsPlottingError(
                f"decoded-state panel {panel} must contain a mapping"
            )
        record_id = raw_panel.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ForwardCorruptionsPlottingError(
                f"decoded-state panel {panel} record_id must be nonempty"
            )
        forward = _decoded_image_tensor(
            raw_panel.get("forward_images"), f"{panel} forward_images"
        )
        generated = _decoded_image_tensor(
            raw_panel.get("generated_images"), f"{panel} generated_images"
        )
        if forward.shape != generated.shape:
            raise ForwardCorruptionsPlottingError(
                f"decoded-state panel {panel} image tensor shapes differ"
            )
        shape = tuple(forward.shape)
        if common_shape is not None and shape != common_shape:
            raise ForwardCorruptionsPlottingError(
                "decoded-state image tensor shapes differ across panels"
            )
        common_shape = shape
        panels[panel] = _DecodedPanel(
            record_id=record_id.strip(),
            forward_images=forward,
            generated_images=generated,
        )
    return _DecodedStates(
        num_inference_steps=num_inference_steps,
        step_indices=step_indices,
        generation_seed=generation_seed,
        panels=panels,
    )


def _render_decoded_figure(payload: _DecodedStates, panel: str) -> Figure:
    """Three matched image rows; RGB differences use one fixed display scale."""
    values = payload.panels[panel]
    forward = values.forward_images.numpy()
    generated = values.generated_images.numpy()
    difference = np.abs(
        forward.astype(np.int16) - generated.astype(np.int16)
    ).astype(np.uint8)
    with matplotlib.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(3, _DECODED_IMAGE_COUNT, figsize=DECODED_GALLERY_SIZE)
        try:
            for row, images in enumerate((forward, generated, difference)):
                for column, (step, pixels) in enumerate(
                    zip(payload.step_indices, images, strict=True)
                ):
                    axis = axes[row, column]
                    axis.imshow(pixels, interpolation="nearest", vmin=0, vmax=255)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    axis.grid(False)
                    for spine in axis.spines.values():
                        spine.set_visible(False)
                    if column == 0:
                        label = _DECODED_ROW_LABELS[row].replace(
                            "Absolute difference", "Absolute\ndifference"
                        )
                        axis.set_ylabel(label, fontsize=TEXT_FONT_SIZE)
                    if row == 2:
                        axis.set_xlabel(
                            f"$t={payload.num_inference_steps - step}$",
                            fontsize=AXIS_NUMBER_FONT_SIZE,
                        )
            figure.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.2)
            return figure
        except BaseException:
            plt.close(figure)
            raise


def plot_saved_decoded_states(
    tensor_path: str | Path, output_directory: str | Path
) -> tuple[Path, ...]:
    """Reload RGB logs and publish one 3x10 PDF/PNG gallery per fixed prompt."""
    source = Path(tensor_path)
    if source.is_symlink() or not source.is_file():
        raise ForwardCorruptionsPlottingError(
            f"decoded-state artifact is missing or unsafe: {source}"
        )
    try:
        payload = _validate_decoded_states_payload(safe_torch_load(source))
    except (CacheIOError, ValueError, OSError) as error:
        raise ForwardCorruptionsPlottingError(
            f"cannot read decoded-state artifact {source}: {error}"
        ) from error
    output = Path(output_directory)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ForwardCorruptionsPlottingError(f"gallery output directory is unsafe: {output}")
    destinations = tuple(
        output / DECODED_FIGURE_FILENAMES[panel][file_format]
        for panel in PANEL_ORDER
        for file_format in ("png", "pdf")
    )
    for destination in destinations:
        if destination.is_symlink() or (
            destination.exists() and not destination.is_file()
        ):
            raise ForwardCorruptionsPlottingError(
                f"gallery destination is unsafe: {destination}"
            )
    figures: list[tuple[Figure, Mapping[str, str]]] = []
    try:
        for panel in PANEL_ORDER:
            figures.append(
                (_render_decoded_figure(payload, panel), DECODED_FIGURE_FILENAMES[panel])
            )
        _publish_figures(output, figures)
    finally:
        for figure, _ in figures:
            plt.close(figure)
    return destinations


def _numeric_series(
    frame: pd.DataFrame, name: str, *, integer: bool = False
) -> pd.Series:
    values = pd.to_numeric(frame[name], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ForwardCorruptionsPlottingError(
            f"observation column {name} must be finite and numeric"
        )
    if integer and not values.map(lambda value: float(value).is_integer()).all():
        raise ForwardCorruptionsPlottingError(
            f"observation column {name} must contain integers"
        )
    return values.astype(np.int64) if integer else values.astype(np.float64)


def _boolean_series(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false")).all():
        raise ForwardCorruptionsPlottingError(
            "observation column is_final_clean_state must contain booleans"
        )
    return normalized.eq("true")


def _require_complete_grid(frame: pd.DataFrame, step_count: int) -> None:
    key_columns = ["panel", "generation_seed", "step_index"]
    if frame.duplicated(key_columns).any():
        raise ForwardCorruptionsPlottingError(
            "observations contain duplicate panel-seed-step rows"
        )
    seeds = {
        panel: tuple(
            sorted(frame.loc[frame["panel"].eq(panel), "generation_seed"].unique())
        )
        for panel in PANEL_ORDER
    }
    for panel, values in seeds.items():
        if len(values) != NUM_SEEDS:
            raise ForwardCorruptionsPlottingError(
                f"panel {panel} has {len(values)}/{NUM_SEEDS} generation seeds"
            )
    if seeds[PANEL_ORDER[0]] != seeds[PANEL_ORDER[1]]:
        raise ForwardCorruptionsPlottingError(
            "memorized and normal panels must use the same generation seeds"
        )
    expected = {
        (panel, seed, step)
        for panel in PANEL_ORDER
        for seed in seeds[PANEL_ORDER[0]]
        for step in range(step_count)
    }
    observed = set(frame[key_columns].itertuples(index=False, name=None))
    if observed != expected:
        missing = len(expected - observed)
        unexpected = len(observed - expected)
        raise ForwardCorruptionsPlottingError(
            "observations do not form a complete panel-seed-step Cartesian grid "
            f"({missing} missing, {unexpected} unexpected)"
        )


def validate_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical observations after enforcing the full empirical design."""

    if tuple(frame.columns) != OBSERVATION_COLUMNS:
        missing = sorted(set(OBSERVATION_COLUMNS) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(OBSERVATION_COLUMNS))
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        if not details:
            details.append("column order differs")
        raise ForwardCorruptionsPlottingError(
            "observation CSV schema differs (" + "; ".join(details) + ")"
        )
    if frame.empty:
        raise ForwardCorruptionsPlottingError("observation CSV is empty")

    validated = frame.copy()
    validated["panel"] = validated["panel"].astype(str).str.strip().str.lower()
    if set(validated["panel"]) != set(PANEL_ORDER):
        raise ForwardCorruptionsPlottingError(
            "panel must contain exactly memorized and normal observations"
        )
    for name in ("record_id", "original_index"):
        validated[name] = validated[name].astype(str).str.strip()
        if validated[name].eq("").any():
            raise ForwardCorruptionsPlottingError(
                f"observation column {name} must be nonempty"
            )
    for panel in PANEL_ORDER:
        panel_rows = validated.loc[validated["panel"].eq(panel)]
        if len(panel_rows[["record_id", "original_index"]].drop_duplicates()) != 1:
            raise ForwardCorruptionsPlottingError(
                f"panel {panel} must describe exactly one prompt-target pair"
            )

    for name in ("generation_seed", "step_index", "timestep"):
        validated[name] = _numeric_series(validated, name, integer=True)
    if (validated["generation_seed"] < 0).any():
        raise ForwardCorruptionsPlottingError(
            "observation generation seeds must be nonnegative"
        )
    if (validated["step_index"] < 0).any():
        raise ForwardCorruptionsPlottingError(
            "observation step indices must be nonnegative"
        )
    validated["is_final_clean_state"] = _boolean_series(
        validated["is_final_clean_state"]
    )
    for name in (
        "denoising_progress",
        "alpha_t",
        "sigma_t",
        "forward_distance",
        "generated_distance",
    ):
        validated[name] = _numeric_series(validated, name)
    for name in ("alpha_t", "sigma_t", "forward_distance", "generated_distance"):
        if (validated[name] < 0.0).any():
            raise ForwardCorruptionsPlottingError(
                f"observation column {name} must be nonnegative"
            )

    final_step = int(validated["step_index"].max())
    if final_step < 1:
        raise ForwardCorruptionsPlottingError(
            "observations require at least one scheduler step and one final state"
        )
    step_count = final_step + 1
    _require_complete_grid(validated, step_count)
    expected_final = validated["step_index"].eq(final_step)
    if not validated["is_final_clean_state"].equals(expected_final):
        raise ForwardCorruptionsPlottingError(
            "is_final_clean_state must identify exactly step_index T"
        )
    expected_progress = validated["step_index"].to_numpy(dtype=float) / final_step
    if not np.allclose(
        validated["denoising_progress"].to_numpy(dtype=float),
        expected_progress,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ForwardCorruptionsPlottingError(
            "denoising_progress must equal step_index / T"
        )

    final = validated.loc[expected_final]
    if not final["timestep"].eq(-1).all():
        raise ForwardCorruptionsPlottingError(
            "the final clean state must use timestep -1"
        )
    if not final["alpha_t"].eq(1.0).all() or not final["sigma_t"].eq(0.0).all():
        raise ForwardCorruptionsPlottingError(
            "the final clean state must use alpha_t=1 and sigma_t=0"
        )
    if not final["forward_distance"].eq(0.0).all():
        raise ForwardCorruptionsPlottingError(
            "the final clean state's forward distance must be exactly zero"
        )
    prefinal = validated.loc[~expected_final]
    if (prefinal["timestep"] < 0).any():
        raise ForwardCorruptionsPlottingError(
            "scheduler timesteps before the final clean state must be nonnegative"
        )

    schedule_columns = (
        "timestep",
        "denoising_progress",
        "alpha_t",
        "sigma_t",
        "is_final_clean_state",
    )
    for name in schedule_columns:
        counts = validated.groupby("step_index", sort=True)[name].nunique(dropna=False)
        if not counts.eq(1).all():
            raise ForwardCorruptionsPlottingError(
                f"observation {name} differs across panels or seeds"
            )
    schedule = validated.drop_duplicates("step_index").sort_values(
        "step_index", kind="stable"
    )
    pre_timesteps = schedule.loc[
        ~schedule["is_final_clean_state"], "timestep"
    ].to_numpy(dtype=np.int64)
    if len(pre_timesteps) > 1 and not (np.diff(pre_timesteps) < 0).all():
        raise ForwardCorruptionsPlottingError(
            "scheduler timesteps must be strictly descending before the final state"
        )

    validated["_panel_order"] = validated["panel"].map(
        {panel: index for index, panel in enumerate(PANEL_ORDER)}
    )
    return (
        validated.sort_values(
            ["_panel_order", "generation_seed", "step_index"], kind="stable"
        )
        .drop(columns="_panel_order")
        .reset_index(drop=True)
    )


def _summarize_validated(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for panel in PANEL_ORDER:
        panel_rows = frame.loc[frame["panel"].eq(panel)]
        for step_index, step in panel_rows.groupby("step_index", sort=True):
            forward = step["forward_distance"].to_numpy(dtype=float)
            generated = step["generated_distance"].to_numpy(dtype=float)
            rows.append(
                {
                    "panel": panel,
                    "step_index": int(step_index),
                    "denoising_progress": float(step["denoising_progress"].iloc[0]),
                    "timestep": int(step["timestep"].iloc[0]),
                    "alpha_t": float(step["alpha_t"].iloc[0]),
                    "sigma_t": float(step["sigma_t"].iloc[0]),
                    "forward_median": float(np.median(forward)),
                    "forward_min": float(np.min(forward)),
                    "forward_max": float(np.max(forward)),
                    "generated_median": float(np.median(generated)),
                    "generated_min": float(np.min(generated)),
                    "generated_max": float(np.max(generated)),
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def summarize_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize the observed 20-seed distribution with median and full range."""

    return _summarize_validated(validate_observations(frame))


def _render_figure(summary: pd.DataFrame, panel: str, *, upper_limit: float) -> Figure:
    """Render one case with the same axes and project style as the other case."""
    if panel not in PANEL_ORDER or not np.isfinite(upper_limit) or upper_limit <= 0:
        raise ForwardCorruptionsPlottingError("invalid figure case or axis limit")
    with matplotlib.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        try:
            values = summary.loc[summary["panel"].eq(panel)].sort_values(
                "step_index", kind="stable"
            )
            # Plot denoising-step indices T,...,0, not training-schedule indices.
            num_steps = int(values["step_index"].max())
            timesteps = num_steps - values["step_index"].to_numpy(dtype=float)
            for ensemble, color, linestyle in (
                ("forward", "C0", "--"),
                ("generated", "C1", "-"),
            ):
                axis.fill_between(
                    timesteps,
                    values[f"{ensemble}_min"].to_numpy(dtype=float),
                    values[f"{ensemble}_max"].to_numpy(dtype=float),
                    color=color,
                    alpha=RANGE_ALPHA,
                    linewidth=0.0,
                )
                axis.plot(
                    timesteps,
                    values[f"{ensemble}_median"].to_numpy(dtype=float),
                    color=color,
                    linestyle=linestyle,
                    linewidth=LINE_WIDTH,
                    alpha=1.0,
                )
            handles = (
                Line2D(
                    [],
                    [],
                    color="C0",
                    linestyle="--",
                    linewidth=LINE_WIDTH,
                    alpha=1.0,
                    label=_FORWARD_LABEL,
                ),
                Line2D(
                    [],
                    [],
                    color="C1",
                    linestyle="-",
                    linewidth=LINE_WIDTH,
                    alpha=1.0,
                    label=_GENERATED_LABEL,
                ),
            )
            axis.legend(
                handles=handles, loc="best", frameon=False, fontsize=LEGEND_FONT_SIZE
            )
            axis.set_xlim(float(timesteps.max()), 0.0)
            axis.set_ylim(0.0, upper_limit)
            axis.set_xlabel(r"$t$", fontsize=TEXT_FONT_SIZE)
            axis.set_ylabel(
                r"$\|\mathbf{x}_t-\mathbf{x}^{\star}\|$",
                fontsize=TEXT_FONT_SIZE,
            )
            axis.tick_params(axis="both", which="both", labelsize=AXIS_NUMBER_FONT_SIZE)
            axis.grid(True, which="both", alpha=0.18, linewidth=0.6)
            figure.tight_layout()
            return figure
        except BaseException:
            plt.close(figure)
            raise


def plot_saved_results(
    csv_path: str | Path,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Reload the measurements and publish two single-axis figures as PNG/PDF."""

    source = Path(csv_path)
    if source.is_symlink() or not source.is_file():
        raise ForwardCorruptionsPlottingError(
            f"saved observation CSV is missing or unsafe: {source}"
        )
    try:
        frame = pd.read_csv(
            source,
            dtype={
                "panel": str,
                "record_id": str,
                "original_index": str,
            },
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ForwardCorruptionsPlottingError(
            f"cannot read saved observation CSV {source}: {error}"
        ) from error
    validated = validate_observations(frame)
    summary = _summarize_validated(validated)
    output = Path(output_directory)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ForwardCorruptionsPlottingError(
            f"figure output directory is unsafe: {output}"
        )
    destinations = tuple(
        output / FIGURE_FILENAMES[panel][file_format]
        for panel in PANEL_ORDER
        for file_format in ("png", "pdf")
    )
    retired = tuple(output / name for name in _RETIRED_FIGURE_FILENAMES)
    for destination in (*destinations, *retired):
        if destination.is_symlink() or (
            destination.exists() and not destination.is_file()
        ):
            raise ForwardCorruptionsPlottingError(
                f"figure destination is unsafe: {destination}"
            )
    # Keep both separate figures directly comparable, including their y limits.
    upper_limit = max(
        1.0, 1.05 * float(summary[["forward_max", "generated_max"]].to_numpy().max())
    )
    figures: list[tuple[Figure, Mapping[str, str]]] = []
    try:
        for panel in PANEL_ORDER:
            figures.append(
                (
                    _render_figure(summary, panel, upper_limit=upper_limit),
                    FIGURE_FILENAMES[panel],
                )
            )
        _publish_figures(output, figures)
        for destination in retired:
            destination.unlink(missing_ok=True)
    finally:
        for figure, _ in figures:
            plt.close(figure)
    return destinations



__all__ = [
    "CSV_NAME",
    "FIGURE_STEM",
    "FIGURE_FILENAMES",
    "ForwardCorruptionsPlottingError",
    "OBSERVATION_COLUMNS",
    "SUMMARY_COLUMNS",
    "plot_saved_results",
    "plot_saved_decoded_states",
    "summarize_observations",
    "validate_observations",
]
