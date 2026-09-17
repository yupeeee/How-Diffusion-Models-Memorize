"""Contracts for the forward-corruptions empirical illustration."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import PolyCollection
import numpy as np
from matplotlib.figure import Figure
import pandas as pd
import pytest
import torch

from utils.experiments import forward_corruptions_plotting as plotting


def _observations(*, steps: int = 3) -> pd.DataFrame:
    timesteps = np.linspace(900, 100, num=steps, dtype=np.int64).tolist() + [-1]
    rows: list[dict[str, object]] = []
    for panel_position, panel in enumerate(plotting.PANEL_ORDER):
        for seed in range(plotting.NUM_SEEDS):
            for step_index in range(steps + 1):
                final = step_index == steps
                remaining = (steps - step_index) / steps
                rows.append(
                    {
                        "panel": panel,
                        "record_id": f"record-{panel}",
                        "original_index": str(10 + panel_position),
                        "generation_seed": seed,
                        "step_index": step_index,
                        "timestep": timesteps[step_index],
                        "is_final_clean_state": final,
                        "denoising_progress": step_index / steps,
                        "alpha_t": 1.0 if final else 0.2 + 0.6 * step_index / steps,
                        "sigma_t": 0.0 if final else 0.9 - 0.6 * step_index / steps,
                        "forward_distance": (
                            0.0
                            if final
                            else remaining * (2.0 + seed / 10.0 + panel_position)
                        ),
                        "generated_distance": (
                            0.3 + remaining * (3.0 + seed / 20.0 + panel_position)
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=plotting.OBSERVATION_COLUMNS)


def test_validate_observations_accepts_only_complete_common_seed_grid() -> None:
    shuffled = _observations().sample(frac=1.0, random_state=7).reset_index(drop=True)
    validated = plotting.validate_observations(shuffled)

    assert tuple(validated.columns) == plotting.OBSERVATION_COLUMNS
    assert len(validated) == 2 * plotting.NUM_SEEDS * 4
    assert validated["panel"].tolist()[: plotting.NUM_SEEDS * 4] == ["memorized"] * (
        plotting.NUM_SEEDS * 4
    )
    assert validated.groupby("panel")["generation_seed"].nunique().eq(20).all()
    assert validated.groupby("panel")["step_index"].unique().map(tuple).to_dict() == {
        "memorized": (0, 1, 2, 3),
        "normal": (0, 1, 2, 3),
    }
    final = validated["is_final_clean_state"]
    assert validated.loc[final, "timestep"].eq(-1).all()
    assert validated.loc[final, "forward_distance"].eq(0.0).all()


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate"),
        (lambda frame: frame.iloc[:-1], "Cartesian grid"),
        (
            lambda frame: frame.assign(
                generation_seed=np.where(
                    frame["panel"].eq("normal") & frame["generation_seed"].eq(19),
                    20,
                    frame["generation_seed"],
                )
            ),
            "same generation seeds",
        ),
        (
            lambda frame: frame.assign(
                forward_distance=np.where(
                    frame["is_final_clean_state"], 0.01, frame["forward_distance"]
                )
            ),
            "exactly zero",
        ),
        (
            lambda frame: frame.assign(
                timestep=np.where(frame["is_final_clean_state"], 0, frame["timestep"])
            ),
            "timestep -1",
        ),
        (
            lambda frame: frame.assign(
                generated_distance=np.where(
                    frame.index == 0, -0.01, frame["generated_distance"]
                )
            ),
            "generated_distance must be nonnegative",
        ),
        (
            lambda frame: frame.assign(
                alpha_t=np.where(
                    frame["panel"].eq("normal")
                    & frame["generation_seed"].eq(0)
                    & frame["step_index"].eq(1),
                    0.41,
                    frame["alpha_t"],
                )
            ),
            "alpha_t differs",
        ),
    ),
)
def test_validate_observations_rejects_corrupt_design(
    mutate: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(plotting.ForwardCorruptionsPlottingError, match=message):
        plotting.validate_observations(mutate(_observations()).reset_index(drop=True))


def test_summary_uses_observed_min_median_max_not_confidence_intervals() -> None:
    observations = _observations()
    selected = observations.loc[
        observations["panel"].eq("memorized") & observations["step_index"].eq(0)
    ]
    summary = plotting.summarize_observations(observations)
    row = summary.loc[
        summary["panel"].eq("memorized") & summary["step_index"].eq(0)
    ].iloc[0]

    assert tuple(summary.columns) == plotting.SUMMARY_COLUMNS
    assert len(summary) == 8
    assert row["forward_min"] == pytest.approx(selected["forward_distance"].min())
    assert row["forward_median"] == pytest.approx(selected["forward_distance"].median())
    assert row["forward_max"] == pytest.approx(selected["forward_distance"].max())
    assert row["generated_min"] == pytest.approx(selected["generated_distance"].min())
    assert row["generated_median"] == pytest.approx(
        selected["generated_distance"].median()
    )
    assert row["generated_max"] == pytest.approx(selected["generated_distance"].max())


@pytest.mark.parametrize("panel", plotting.PANEL_ORDER)
@pytest.mark.parametrize("num_steps", [3, 50])
def test_each_panel_is_an_independent_styled_single_axis_figure(
    panel: str, num_steps: int
) -> None:
    summary = plotting.summarize_observations(_observations(steps=num_steps))
    upper_limit = 1.05 * max(
        summary["forward_max"].max(), summary["generated_max"].max()
    )
    figure = plotting._render_figure(summary, panel, upper_limit=upper_limit)
    try:
        assert tuple(figure.get_size_inches()) == pytest.approx(plotting.FIGURE_SIZE)
        assert plotting.FIGURE_SIZE == (4.0, 4.0)
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_title() == ""
        assert figure._suptitle is None
        assert figure.texts == []
        assert not axis.texts
        assert len(axis.lines) == 2
        assert [line.get_linestyle() for line in axis.lines] == ["--", "-"]
        assert [line.get_color() for line in axis.lines] == ["C0", "C1"]
        assert [line.get_linewidth() for line in axis.lines] == pytest.approx(
            [plotting.LINE_WIDTH, plotting.LINE_WIDTH]
        )
        assert plotting.LINE_WIDTH == pytest.approx(1.0)
        ranges = [
            item for item in axis.collections if isinstance(item, PolyCollection)
        ]
        assert len(ranges) == 2
        assert all(item.get_alpha() == pytest.approx(0.2) for item in ranges)
        assert axis.get_xlim() == pytest.approx((num_steps, 0.0))
        for line in axis.lines:
            np.testing.assert_array_equal(
                line.get_xdata(), np.arange(num_steps, -1, -1)
            )
        assert axis.get_ylim() == pytest.approx((0.0, upper_limit))
        assert axis.get_xlabel() == r"$t$"
        assert axis.get_ylabel() == r"$\|\mathbf{x}_t-\mathbf{x}^{\star}\|$"
        assert axis.xaxis.label.get_fontsize() == pytest.approx(18)
        assert axis.yaxis.label.get_fontsize() == pytest.approx(18)
        assert all(
            label.get_fontsize() == pytest.approx(15)
            for label in axis.get_xticklabels()
        )
        assert all(
            label.get_fontsize() == pytest.approx(15)
            for label in axis.get_yticklabels()
        )
        legend = axis.get_legend()
        assert legend is not None
        assert not legend.get_frame_on()
        assert [text.get_text() for text in legend.get_texts()] == [
            "Forward-corrupted target",
            "Generated state",
        ]
        assert all(
            text.get_fontsize() == pytest.approx(12) for text in legend.get_texts()
        )
        assert all(
            handle.get_alpha() in (None, 1.0) for handle in legend.legend_handles
        )
        assert axis.xaxis.label.get_fontfamily() == ["STIXGeneral"]
        assert axis.yaxis.label.get_fontfamily() == ["STIXGeneral"]
    finally:
        matplotlib.pyplot.close(figure)


def test_plot_saved_results_reloads_csv_and_writes_four_panel_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / plotting.CSV_NAME
    _observations().to_csv(csv_path, index=False)
    original_csv = csv_path.read_bytes()
    output = tmp_path / "figures"
    output.mkdir()
    old_paths = (
        output / f"{plotting.FIGURE_STEM}.png",
        output / f"{plotting.FIGURE_STEM}.pdf",
    )
    for path in old_paths:
        path.write_bytes(b"retired combined figure")
    unrelated = output / "notes.txt"
    unrelated.write_text("belongs to the user", encoding="utf-8")

    save_calls = []
    original_savefig = Figure.savefig

    def tracked_savefig(figure, *args, **kwargs):
        save_calls.append((id(figure), len(figure.axes), figure.axes[0].get_ylim(), kwargs))
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", tracked_savefig)
    destinations = plotting.plot_saved_results(csv_path, output)

    assert len(save_calls) == 4
    assert len({call[0] for call in save_calls}) == 2
    assert all(call[1] == 1 for call in save_calls)
    assert len({call[2] for call in save_calls}) == 1
    for _, _, _, kwargs in save_calls:
        assert kwargs["dpi"] == 150
        assert kwargs["bbox_inches"] == "tight"
        assert kwargs["pad_inches"] == pytest.approx(0.05)

    assert destinations == (
        output / plotting.FIGURE_FILENAMES["memorized"]["png"],
        output / plotting.FIGURE_FILENAMES["memorized"]["pdf"],
        output / plotting.FIGURE_FILENAMES["normal"]["png"],
        output / plotting.FIGURE_FILENAMES["normal"]["pdf"],
    )
    assert {path.name for path in output.iterdir()} == {
        *(path.name for path in destinations),
        unrelated.name,
    }
    assert destinations[0].read_bytes().startswith(b"\x89PNG")
    assert destinations[1].read_bytes().startswith(b"%PDF")
    assert destinations[2].read_bytes().startswith(b"\x89PNG")
    assert destinations[3].read_bytes().startswith(b"%PDF")
    assert not any(path.exists() for path in old_paths)
    assert unrelated.read_text(encoding="utf-8") == "belongs to the user"
    assert csv_path.read_bytes() == original_csv


def test_failed_publication_preserves_retired_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / plotting.CSV_NAME
    _observations().to_csv(csv_path, index=False)
    output = tmp_path / "figures"
    output.mkdir()
    retired = [output / f"{plotting.FIGURE_STEM}.{suffix}" for suffix in ("png", "pdf")]
    for path in retired:
        path.write_bytes(b"previous figure")

    def fail_publish(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(plotting, "_publish_figures", fail_publish)
    with pytest.raises(OSError, match="publication failure"):
        plotting.plot_saved_results(csv_path, output)

    assert all(path.read_bytes() == b"previous figure" for path in retired)


def test_retired_figure_symlink_is_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / plotting.CSV_NAME
    _observations().to_csv(csv_path, index=False)
    output = tmp_path / "figures"
    output.mkdir()
    unrelated = tmp_path / "unrelated.png"
    unrelated.write_bytes(b"unrelated contents")
    retired = output / f"{plotting.FIGURE_STEM}.png"
    retired.symlink_to(unrelated)

    with pytest.raises(plotting.ForwardCorruptionsPlottingError, match="unsafe"):
        plotting.plot_saved_results(csv_path, output)

    assert retired.is_symlink()
    assert unrelated.read_bytes() == b"unrelated contents"


def _decoded_payload():
    forward = torch.zeros((10, 4, 4, 3), dtype=torch.uint8)
    generated = torch.full_like(forward, 240)
    forward[:, 0, 0] = 255
    return {
        "schema_version": 1,
        "num_inference_steps": 50,
        "generation_seed": 0,
        "step_indices": [0, 6, 11, 17, 22, 28, 33, 39, 44, 50],
        "panels": {
            panel: {
                "record_id": panel,
                "forward_images": forward.clone(),
                "generated_images": generated.clone(),
            }
            for panel in plotting.PANEL_ORDER
        },
    }


@pytest.mark.parametrize("panel", plotting.PANEL_ORDER)
def test_decoded_gallery_has_matched_rows_and_fixed_scale_difference(panel):
    raw = _decoded_payload()
    payload = plotting._validate_decoded_states_payload(raw)
    figure = plotting._render_decoded_figure(payload, panel)
    try:
        assert tuple(figure.get_size_inches()) == pytest.approx((12, 4))
        assert len(figure.axes) == 30
        assert not figure.texts
        forward = raw["panels"][panel]["forward_images"].numpy()
        generated = raw["panels"][panel]["generated_images"].numpy()
        difference = np.abs(
            forward.astype(np.int16) - generated.astype(np.int16)
        ).astype(np.uint8)
        for row, images in enumerate((forward, generated, difference)):
            for column, expected in enumerate(images):
                axis = figure.axes[row * 10 + column]
                assert axis.get_title() == ""
                assert not axis.texts
                assert len(axis.images) == 1
                np.testing.assert_array_equal(axis.images[0].get_array(), expected)
                assert axis.images[0].get_clim() == (0, 255)
                assert not axis.get_xticks().size and not axis.get_yticks().size
                if row == 2:
                    remaining = 50 - raw["step_indices"][column]
                    assert axis.get_xlabel() == f"$t={remaining}$"
                    assert axis.xaxis.label.get_fontsize() == 15
        assert [figure.axes[row * 10].get_ylabel() for row in range(3)] == [
            "Forward", "Generated", "Absolute\ndifference"
        ]
        assert figure.axes[20].yaxis.label.get_fontsize() == 18
        assert figure.axes[20].yaxis.label.get_fontfamily() == ["STIXGeneral"]
    finally:
        matplotlib.pyplot.close(figure)


def test_decoded_gallery_reloads_logs_and_preserves_distance_figures(
    tmp_path, monkeypatch
):
    source = tmp_path / "decoded_states.pt"
    torch.save(_decoded_payload(), source)
    before = source.read_bytes()
    output = tmp_path / "figures"
    output.mkdir()
    old = output / plotting.FIGURE_FILENAMES["memorized"]["png"]
    old.write_bytes(b"existing distance figure")
    save_calls = []
    savefig = Figure.savefig

    def tracked(figure, *args, **kwargs):
        save_calls.append((len(figure.axes), kwargs))
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", tracked)
    paths = plotting.plot_saved_decoded_states(source, output)
    assert paths == tuple(
        output / plotting.DECODED_FIGURE_FILENAMES[panel][fmt]
        for panel in plotting.PANEL_ORDER for fmt in ("png", "pdf")
    )
    assert len(save_calls) == 4
    for count, kwargs in save_calls:
        assert count == 30
        assert kwargs["dpi"] == 150
        assert kwargs["bbox_inches"] == "tight"
        assert kwargs["pad_inches"] == pytest.approx(0.05)
    for path in paths:
        signature = b"%PDF" if path.suffix == ".pdf" else b"\x89PNG"
        assert path.read_bytes().startswith(signature)
    assert old.read_bytes() == b"existing distance figure"
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "corruption",
    ["missing_panel", "dtype", "shape", "duplicate_step", "endpoint", "seed", "id"],
)
def test_decoded_gallery_rejects_invalid_logged_images(tmp_path, corruption):
    payload = _decoded_payload()
    panel = payload["panels"]["memorized"]
    if corruption == "missing_panel":
        del payload["panels"]["normal"]
    elif corruption == "dtype":
        panel["forward_images"] = panel["forward_images"].float()
    elif corruption == "shape":
        panel["generated_images"] = panel["generated_images"][:9]
    elif corruption == "duplicate_step":
        payload["step_indices"][1] = 0
    elif corruption == "endpoint":
        payload["step_indices"][-1] = 49
    elif corruption == "seed":
        payload["generation_seed"] = -1
    else:
        panel["record_id"] = ""
    source = tmp_path / "decoded_states.pt"
    torch.save(payload, source)
    with pytest.raises(plotting.ForwardCorruptionsPlottingError):
        plotting.plot_saved_decoded_states(source, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


def test_plot_saved_results_rejects_schema_instead_of_filtering(tmp_path: Path) -> None:
    csv_path = tmp_path / plotting.CSV_NAME
    _observations().drop(columns="forward_distance").to_csv(csv_path, index=False)

    with pytest.raises(
        plotting.ForwardCorruptionsPlottingError, match="schema differs"
    ):
        plotting.plot_saved_results(csv_path, tmp_path / "figures")
