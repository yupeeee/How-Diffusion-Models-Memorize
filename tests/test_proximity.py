"""Offline proximity integration and plotting tests."""

from __future__ import annotations

import ast
from collections import Counter
import inspect
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
import torch

from scripts.compute_proximity import build_parser as build_proximity_parser
from utils.common.io import atomic_torch_save, atomic_write_json, file_sha256
from utils.data import selection as selection_module
from utils.experiments.cache import CompletedGenerationRecord, GenerationPaths
from utils.experiments import plotting as plotting_module
from utils.experiments import proximity as proximity_module
from utils.experiments.plotting import write_analysis_outputs, write_selection_figure
from utils.experiments.proximity import (
    ANALYSIS_COLUMNS,
    OBSERVATION_COLUMNS,
    ProximityError,
    ProximityPaths,
    _annotate_selection,
    _failed_reference_rows,
    _paired_rows,
    _record_progress,
    _seed_role,
)
from utils.models.latent import compute_latent_distances

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "sdv1_ddim_g7.5_T50_N20"


def _proximity_summary(figure):
    return next(text for text in [*figure.texts, *figure.axes[0].texts]
                if text.get_text().startswith("#Prompts:"))


def _proximity_category_colors():
    cmap = plotting_module.matplotlib.colormaps["magma"]
    return {"MV": cmap(.70), "RV": cmap(.20), "TV": "#168C91",
            "N": "#536B8A", "Other / unlabeled": "#858B93"}


def _run_args(
    *,
    model_name: str = "sdv1",
    scheduler_name: str = "ddim",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_seeds: int = 20,
    seed_start: int = 0,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
        "seed_start": seed_start,
    }


def test_compute_proximity_parser_supports_overwrite_and_gmm_only() -> None:
    parser = build_proximity_parser()

    defaults = parser.parse_args([])
    assert defaults.selection_strategy == "gmm"
    assert defaults.overwrite is False
    assert defaults.plot is False
    assert parser.parse_args(["--overwrite"]).overwrite is True
    assert parser.parse_args(["--plot"]).plot is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--plot", "--overwrite"])
    assert (
        parser.parse_args(["--selection-strategy", "gmm"]).selection_strategy == "gmm"
    )
    for unsupported in ("spearman", "gmm-evidence", "unsupported"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--selection-strategy", unsupported])
    assert (
        inspect.signature(proximity_module.run_proximity)
        .parameters["overwrite"]
        .default
        is inspect.Parameter.empty
    )


def test_compute_proximity_plot_dispatches_to_read_only_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import compute_proximity as compute_proximity_script

    calls: list[dict[str, object]] = []
    result = SimpleNamespace(
        paths=SimpleNamespace(summary_json=tmp_path / "summary.json"),
        exit_code=0,
    )

    def plot(_root: Path, **arguments: object) -> object:
        calls.append(arguments)
        return result

    monkeypatch.setattr(compute_proximity_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(proximity_module, "plot_proximity", plot)
    monkeypatch.setattr(
        proximity_module,
        "run_proximity",
        lambda *_args, **_kwargs: pytest.fail("--plot must not dispatch computation"),
    )

    assert (
        compute_proximity_script.main(
            [
                "--plot",
                "--model",
                "sdv2",
                "--scheduler",
                "ddpm",
                "--g",
                "3.25",
                "--T",
                "17",
                "--N",
                "7",
                "--seed-start",
                "7",
                "--selection-strategy",
                "gmm",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "model_name": "sdv2",
            "scheduler_name": "ddpm",
            "guidance_scale": 3.25,
            "num_inference_steps": 17,
            "num_seeds": 7,
            "seed_start": 7,
            "selection_strategy": "gmm",
        }
    ]


def test_proximity_progress_remains_visible_when_stderr_is_captured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert list(_record_progress(["one", "two"])) == ["one", "two"]
    assert "[Proximity] Records" in capsys.readouterr().err


def test_sscd_reuse_ignores_generation_marker_audit_hash_only(
    tmp_path: Path,
) -> None:
    generation = GenerationPaths(tmp_path / "logs" / "synthetic")
    generation.create()
    index = "7"
    scores = torch.tensor([0.25, 0.75], dtype=torch.float32)
    score_path = generation.run_directory / "sscd" / f"{index}.pt"
    score_hash = atomic_torch_save(scores, score_path)
    marker_path = generation.run_directory / "sscd_record" / f"{index}.json"
    marker = {
        "original_index": index,
        "record_id": "sdv1-0007",
        "source_row_number": 3,
        "prompt_raw": "selected prompt",
        "generation_record_sha256": "0" * 64,
        "generation_scientific_config_hash": "a" * 64,
        "generation_latent_sha256": "b" * 64,
        "target_image_sha256": "c" * 64,
        "sscd_configuration_hash": "d" * 64,
        "num_seeds": 2,
        "seeds": [0, 1],
        "score_path": str(score_path),
        "score_sha256": score_hash,
        "score_shape": [2],
        "score_dtype": "float32",
    }
    atomic_write_json(marker_path, marker)
    metadata = {
        "record_id": "sdv1-0007",
        "prompt_raw": "selected prompt",
        "scientific_config_hash": "a" * 64,
        "tensor_file_sha256": {"latent": "b" * 64},
        "target_image_sha256": "c" * 64,
        "num_seeds": 2,
        "seeds": [0, 1],
    }
    completed = CompletedGenerationRecord(
        index,
        3,
        generation.record_path(index),
        metadata,
    )
    record = proximity_module._GenerationRecord(
        metadata,
        completed,
        generation,
        "logs/synthetic/image/7.png",
    )

    loaded = proximity_module._sscd_scores(
        tmp_path,
        record,
        {"scientific_config_hash": "a" * 64},
        {"configuration_hash": "d" * 64},
    )
    torch.testing.assert_close(loaded, scores)

    marker["generation_latent_sha256"] = "e" * 64
    atomic_write_json(marker_path, marker)
    with pytest.raises(ProximityError, match="generation_latent_sha256"):
        proximity_module._sscd_scores(
            tmp_path,
            record,
            {"scientific_config_hash": "a" * 64},
            {"configuration_hash": "d" * 64},
        )

    class Progress:
        def __init__(self) -> None:
            self.updates: list[dict[str, object]] = []

        def set_postfix(self, **values: object) -> None:
            self.updates.append(values)

    progress = Progress()
    completed = SimpleNamespace(
        original_index="fallback-index",
        metadata={"record_id": "record-17"},
    )
    proximity_module._set_record_progress(progress, completed, "unusable")
    assert progress.updates == [
        {"record": "record-17", "status": "unusable", "refresh": True}
    ]


def test_canonical_latent_distance_handles_terminal_endpoints() -> None:
    trajectory = torch.full((2, 3, 1, 2, 2), 100.0, dtype=torch.float32)
    trajectory[0, -1], trajectory[1, -1] = 1.0, 2.0
    distances = compute_latent_distances(
        trajectory[:, -1], torch.zeros((1, 2, 2), dtype=torch.float32)
    )
    torch.testing.assert_close(distances.l2_norms, torch.tensor([2.0, 4.0]))


def test_output_paths_do_not_define_a_per_prompt_proximity_cache(
    tmp_path: Path,
) -> None:
    experiment_run = Path(RUN_NAME) / "experiment_S0_N20"
    paths = ProximityPaths.build(
        tmp_path.resolve(),
        experiment_run,
        role="experiment",
        seed_start=0,
        num_seeds=20,
    )
    reference = ProximityPaths.frozen_selection(
        tmp_path.resolve(),
        tmp_path.resolve() / "logs" / RUN_NAME / "reference_S20_N20",
        tmp_path.resolve() / "data/webster/selection/sdv1/reference_S20_N20",
    )
    reference_outputs = proximity_module._reference_output_paths(
        tmp_path.resolve(),
        tmp_path.resolve() / "logs" / RUN_NAME / "reference_S20_N20",
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        selection_strategy="gmm",
    )

    assert paths.output_directory == (
        tmp_path.resolve() / "outputs" / RUN_NAME / "proximity/experiment_S0_N20"
    )
    with pytest.raises(
        selection_module.TargetPairSelectionError,
        match="selection_strategy must be one of: gmm",
    ):
        ProximityPaths.build(
            tmp_path.resolve(),
            experiment_run,
            role="experiment",
            seed_start=0,
            num_seeds=20,
            selection_strategy="spearman",
        )
    assert paths.run_config_json.name == "run_config.json"
    assert reference.run_config_json.name == "config.json"
    assert reference.summary_json.name == "summary.json"
    assert reference_outputs.output_directory == (
        tmp_path.resolve() / "outputs" / RUN_NAME / "proximity/reference_S20_N20"
    )
    assert reference_outputs.output_directory != reference.output_directory
    assert not hasattr(paths, "records_directory")
    assert not hasattr(paths, "result_path")


def test_scheduler_name_requires_a_mapping_with_a_name() -> None:
    assert proximity_module._scheduler_name({"scheduler": {"name": "ddim"}}) == "ddim"
    for invalid in ({}, {"scheduler": "ddim"}, {"scheduler": {}}):
        with pytest.raises(ProximityError, match="scheduler name is missing"):
            proximity_module._scheduler_name(invalid)


def test_proximity_has_no_inference_or_per_prompt_result_cache() -> None:
    path = ROOT / "utils/experiments/proximity.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not {
        name
        for name in imported
        if name.startswith(
            (
                "diffusers",
                "transformers",
                "utils.models.loading",
                "utils.models.sampling",
            )
        )
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "compute_latent_distances" in calls
    assert not calls.intersection(
        {"load_model_components", "sample_trajectory", "atomic_torch_save"}
    )
    for obsolete in (
        "SCHEMA_VERSION",
        "schema_version",
        "paired_selected",
        "threshold_diagnostics",
        "selected_tv.csv",
        "records_directory",
        "noise_prediction",
    ):
        assert obsolete not in source


def _role(**overrides: object) -> str:
    values = _run_args()
    values.update(overrides)
    return _seed_role(**values)


@pytest.mark.parametrize("num_seeds", (1, 3, 20, 37))
@pytest.mark.parametrize(
    ("scheduler_name", "guidance_scale", "num_inference_steps"),
    (("ddim", 7.5, 50), ("ddpm", 3.25, 17)),
)
def test_seed_roles_accept_dynamic_disjoint_ranges_and_sampler_values(
    num_seeds: int,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
) -> None:
    sampler = {
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
    }
    assert _role(**sampler) == "experiment"
    assert _role(**sampler, seed_start=num_seeds) == "reference"


def test_seed_roles_preserve_experiment_science_configuration() -> None:
    assert _role() == "experiment"
    assert _role(scheduler_name="ddpm", guidance_scale=3.0, num_seeds=4) == "experiment"


@pytest.mark.parametrize(
    ("seed_start", "seeds"),
    (
        (0, [0, 1, 2]),
        (3, [3, 4, 5]),
    ),
)
def test_n3_generation_cache_uses_exact_experiment_and_reference_ranges(
    seed_start: int,
    seeds: list[int],
) -> None:
    configuration = {
        "scientific_config": {
            "model_cli_name": "sdv1",
            "scheduler": {"name": "ddim"},
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "num_seeds": 3,
            "seeds": seeds,
            "latent_shape": [4, 64, 64],
        }
    }
    proximity_module._validate_generation_invocation(
        configuration,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=3,
        seed_start=seed_start,
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {"num_seeds": 0, "seed_start": 0},
        {"num_seeds": -1, "seed_start": 0},
        {"seed_start": 1, "num_seeds": 19},
        {"seed_start": 10, "num_seeds": 20},
        {"seed_start": 20, "num_seeds": 19},
        {"seed_start": 20, "scheduler_name": "unsupported"},
    ),
)
def test_seed_roles_reject_overlap_and_invalid_reference(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ProximityError, match="first N seeds"):
        _role(**overrides)


def test_missing_prerequisites_report_seed_aware_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_seeds = 7
    arguments = _run_args(
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=17,
        num_seeds=num_seeds,
        seed_start=num_seeds,
    )
    with pytest.raises(ProximityError) as missing_generation:
        proximity_module.run_proximity(tmp_path, overwrite=False, **arguments)
    message = str(missing_generation.value)
    assert "--scheduler ddpm --g 3.25 --T 17 --N 7 --seed-start 7" in message
    assert "--downscale" not in message

    run_name = "sdv1_ddpm_g3.25_T17_N7"
    run = tmp_path / "logs" / run_name / "reference_S7_N7"
    run.mkdir(parents=True)
    (run / "run_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(proximity_module, "require_generation_run", lambda _: {})
    monkeypatch.setattr(
        proximity_module,
        "_validate_generation_invocation",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ProximityError) as missing_sscd:
        proximity_module.run_proximity(tmp_path, overwrite=False, **arguments)
    assert "--scheduler ddpm --g 3.25 --T 17 --N 7 --seed-start 7" in str(
        missing_sscd.value
    )


def test_reference_plot_proximity_uses_only_frozen_selection_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "selection"
    directory.mkdir()
    summary = {
        "complete": True,
        "selection_hash": "a" * 64,
        "selection_strategy": "gmm",
    }
    artifacts = {
        "selection.csv": b"immutable selection table",
        "config.json": b"immutable selection configuration",
        "summary.json": json.dumps(summary).encode(),
    }
    for filename, contents in artifacts.items():
        (directory / filename).write_bytes(contents)
    before = {
        filename: ((directory / filename).read_bytes(), (directory / filename).stat())
        for filename in artifacts
    }
    frame = pd.DataFrame({"sentinel": [1]})
    configuration = {"selection_hash": "a" * 64}
    selection = SimpleNamespace(
        sha256="a" * 64,
        frame=frame,
        configuration=configuration,
    )
    loaded: list[dict[str, object]] = []

    def load_selection(_root: Path, **identity: object) -> object:
        loaded.append(identity)
        return selection

    monkeypatch.setattr(
        selection_module,
        "target_pair_selection_directory",
        lambda *_args, **_kwargs: directory,
    )
    monkeypatch.setattr(selection_module, "load_target_pair_selection", load_selection)
    for name in (
        "generation_paths",
        "require_generation_run",
        "list_completed_records",
        "_load_sscd_config",
        "reference_completion_fingerprint",
    ):
        monkeypatch.setattr(
            proximity_module,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"reference plot-only must not call {_name}"
            ),
        )
    plotted: list[tuple[str, object, object]] = []

    def plot_selection(
        path: str | Path, *, output_directory: str | Path, figure_directory: str | Path
    ) -> plotting_module.AnalysisStatistics:
        output = Path(figure_directory)
        assert len(example_calls) == 1 and example_calls[0][-1] is True
        plotted.append(("scatter", Path(path), output))
        output.mkdir(parents=True)
        for filename in plotting_module.PROXIMITY_PDF_FILENAMES:
            (output / filename).write_bytes(b"scatter figure")
        return plotting_module.AnalysisStatistics(1, 1, 1, 1.0, -1.0)

    def plot_gmm(
        path: str | Path, *, frame: pd.DataFrame, configuration: object, figure_directory: str | Path
    ) -> None:
        output = Path(figure_directory)
        plotted.append(("gmm", output, (frame, configuration)))
        for filename in (plotting_module.SELECTION_GMM_FIGURES["pdf"],):
            (output / filename).write_bytes(b"GMM figure")

    monkeypatch.setattr(proximity_module, "write_selection_figure", plot_selection)
    monkeypatch.setattr(proximity_module, "write_gmm_fit_figure", plot_gmm)
    example_calls: list[tuple[ProximityPaths, Path, int, bool]] = []

    def write_examples(paths: ProximityPaths, *, table_path: Path, num_seeds: int,
                       validate_only: bool = False) -> None:
        example_calls.append((paths, table_path, num_seeds, validate_only))

    monkeypatch.setattr(proximity_module, "_write_examples", write_examples)

    result = proximity_module.plot_proximity(
        tmp_path,
        **_run_args(
            model_name="sdv2",
            scheduler_name="ddpm",
            guidance_scale=3.25,
            num_inference_steps=17,
            num_seeds=7,
            seed_start=7,
        ),
    )

    assert result.exit_code == 0
    assert result.paths.output_directory == directory
    output = (
        tmp_path
        / "figures/sdv2_ddpm_g3.25_T17_N7/proximity/reference_S7_N7"
    )
    assert loaded == [
        {
            "model_name": "sdv2",
            "scheduler_name": "ddpm",
            "guidance_scale": 3.25,
            "num_inference_steps": 17,
            "num_seeds": 7,
            "selection_strategy": "gmm",
        }
    ]
    assert plotted == [
        ("scatter", directory, output),
        ("gmm", output, (frame, configuration)),
    ]
    assert len(example_calls) == 2
    example_paths, example_table, example_seeds, preflight = example_calls[0]
    assert preflight is True
    assert example_calls[1] == (example_paths, example_table, example_seeds, False)
    assert example_paths.figure_directory == output
    assert example_paths.output_directory == (
        tmp_path / "outputs/sdv2_ddpm_g3.25_T17_N7/proximity/reference_S7_N7"
    )
    assert example_paths.generation_run == result.paths.generation_run
    assert example_table == directory / "selection.csv" and example_seeds == 7
    for filename, (contents, stat) in before.items():
        path = directory / filename
        assert path.read_bytes() == contents
        assert path.stat().st_mtime_ns == stat.st_mtime_ns
        assert path.stat().st_ctime_ns == stat.st_ctime_ns
    assert {path.name for path in directory.iterdir()} == set(artifacts)
    assert {path.name for path in output.iterdir()} == set(
        plotting_module.SELECTION_PDF_FILENAMES
    )
    assert not (output / "examples").exists()
    assert not (tmp_path / "logs").exists()


def test_experiment_loads_only_the_n_matched_frozen_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "experiment"
    run.mkdir()
    run_config = run / "run_config.json"
    run_config.write_text("{}", encoding="utf-8")
    (run / "sscd_config.json").write_text("{}", encoding="utf-8")
    cache = SimpleNamespace(
        run_directory=run,
        run_config=run_config,
        record_directory=run / "record",
    )
    monkeypatch.setattr(
        proximity_module, "generation_paths", lambda _root, **_kwargs: cache
    )
    monkeypatch.setattr(proximity_module, "require_generation_run", lambda _: {})
    monkeypatch.setattr(
        proximity_module,
        "_validate_generation_invocation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(proximity_module, "_load_sscd_config", lambda *_args: {})
    monkeypatch.setattr(proximity_module, "list_completed_records", lambda _: [])
    loaded: list[dict[str, object]] = []

    def load_selection(_root: Path, **identity: object) -> object:
        loaded.append(identity)
        return object()

    monkeypatch.setattr(selection_module, "load_target_pair_selection", load_selection)
    with pytest.raises(ProximityError, match="no completed generation records"):
        proximity_module.run_proximity(
            tmp_path,
            overwrite=False,
            **_run_args(
                scheduler_name="ddpm",
                guidance_scale=3.25,
                num_inference_steps=17,
                num_seeds=3,
            ),
        )
    assert loaded == [
        {
            "model_name": "sdv1",
            "scheduler_name": "ddpm",
            "guidance_scale": 3.25,
            "num_inference_steps": 17,
            "num_seeds": 3,
            "selection_strategy": "gmm",
        },
    ]


def _record(
    *, index: str = "42", seeds: list[int] | None = None, kind: str = "TV"
) -> SimpleNamespace:
    seed_values = [20, 23, 39] if seeds is None else seeds
    completed = SimpleNamespace(original_index=index, source_row_number=7)
    return SimpleNamespace(
        completed=completed,
        generated_image_path=f"logs/run/image/{index}.png",
        metadata={
            "record_id": f"sdv1-{index}",
            "prompt_raw": "prompt",
            "webster_overfit_type": kind,
            "target_image_sha256": "a" * 64,
            "num_seeds": len(seed_values),
            "seeds": seed_values,
        },
    )


def test_paired_rows_preserve_seed_position_and_audit_fields() -> None:
    rows = _paired_rows(
        _record(),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.4, 0.5, 0.6]),
        model_name="sdv1",
    )

    assert set(rows[0]) == set(OBSERVATION_COLUMNS)
    assert [row["seed"] for row in rows] == [20, 23, 39]
    assert [row["generated_image_tile_index"] for row in rows] == [0, 1, 2]
    assert [row["l2_norm"] for row in rows] == pytest.approx([1.0, 2.0, 3.0])
    assert [row["sscd"] for row in rows] == pytest.approx([0.4, 0.5, 0.6])
    assert {row["observation_status"] for row in rows} == {"complete"}
    assert {row["observation_error"] for row in rows} == {""}


def test_reference_cache_failure_becomes_explicit_unusable_observations(
    tmp_path: Path,
) -> None:
    reference_seeds = list(range(7, 14))
    completed = SimpleNamespace(
        original_index="42",
        source_row_number=7,
        metadata={
            "record_id": "sdv1-42",
            "prompt_raw": "prompt",
            "webster_overfit_type": "MV",
            "target_image_sha256": "a" * 64,
        },
    )
    rows = _failed_reference_rows(
        tmp_path,
        SimpleNamespace(image_path=lambda index: tmp_path / f"image/{index}.png"),
        completed,
        {"scientific_config": {"seeds": reference_seeds}},
        RuntimeError("score missing"),
        model_name="sdv1",
    )

    assert len(rows) == 7
    assert [row["seed"] for row in rows] == reference_seeds
    assert {row["kind"] for row in rows} == {"MV"}
    assert {row["observation_status"] for row in rows} == {"cache_error"}
    assert {row["observation_error"] for row in rows} == {"RuntimeError: score missing"}
    assert all(pd.isna(row["l2_norm"]) and pd.isna(row["sscd"]) for row in rows)


def _observations(index: str, *, count: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "sdv1",
                "original_index": index,
                "record_id": f"sdv1-{index}",
                "source_row_number": int(index),
                "seed": seed,
                "prompt": f"prompt-{index}",
                "kind": "N",
                "target_image_sha256": "a" * 64,
                "generated_image_path": f"logs/experiment/image/{index}.png",
                "generated_image_tile_index": seed,
                "l2_norm": float(seed + 1),
                "sscd": float(count - seed),
                "observation_status": "complete",
                "observation_error": "",
            }
            for seed in range(count)
        ],
        columns=OBSERVATION_COLUMNS,
    )


def _selection(indices: tuple[str, ...]) -> SimpleNamespace:
    prompts = pd.DataFrame(
        [
            {
                "model_name": "sdv1",
                "original_index": index,
                "record_id": f"sdv1-{index}",
                "source_row_number": int(index),
                "prompt": f"prompt-{index}",
                "kind": "N",
                "target_image_sha256": "a" * 64,
                "selection_strategy": "gmm",
                "prompt_spearman": -1.0 if index == "1" else 1.0,
                "include_prompt": index == "1",
                "selection_status": (
                    "included_proximity_rule"
                    if index == "1"
                    else "discarded_proximity_rule"
                ),
                "selection_reason": (
                    "no_low_sscd_mode_majority"
                    if index == "1"
                    else "majority_reference_seeds_in_low_sscd_mode"
                ),
            }
            for index in indices
        ]
    )
    return SimpleNamespace(
        prompt_frame=prompts,
        configuration={
            "selection_policy": selection_module.selection_policy("gmm"),
            "selection_strategy": "gmm",
        },
        sha256="c" * 64,
    )


def _saved_experiment_plot_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], ProximityPaths, SimpleNamespace]:
    arguments = _run_args(num_seeds=2)
    selection = _selection(("1", "2"))
    generation_run = tmp_path / proximity_module.generation_log_relative_path(
        **arguments
    )
    run_name = proximity_module.generation_run_name(**arguments)
    paths = ProximityPaths.build(
        tmp_path.resolve(),
        generation_run,
        output_run_name=run_name,
        role="experiment",
        seed_start=0,
        num_seeds=2,
        selection_strategy="gmm",
    )
    generation = {
        "scientific_config_hash": "a" * 64,
        "scientific_config": {
            "model_cli_name": "sdv1",
            "scheduler": {"name": "ddim"},
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "num_seeds": 2,
            "seeds": [0, 1],
        },
    }
    configuration = proximity_module._analysis_configuration(
        paths,
        run_name,
        generation,
        {"configuration_hash": "b" * 64},
        selection,
    )
    paths.output_directory.mkdir(parents=True)
    atomic_write_json(paths.run_config_json, configuration)
    analysis = _annotate_selection(
        pd.concat([_observations("1"), _observations("2")], ignore_index=True),
        selection,
    )
    write_analysis_outputs(paths.output_directory, analysis=analysis, figure_directory=paths.figure_directory)
    return arguments, paths, selection


def test_experiment_plot_proximity_reads_only_saved_contract_and_renders_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, paths, selection = _saved_experiment_plot_fixture(tmp_path)
    frozen = tmp_path / "frozen-selection"
    frozen.mkdir()
    for filename in ("selection.csv", "config.json", "summary.json"):
        (frozen / filename).write_bytes(f"immutable {filename}".encode())
    paths.summary_json.write_bytes(b"unrelated saved summary")
    unrelated = paths.output_directory / "notes.txt"
    unrelated.write_bytes(b"preserve me")
    watched = tuple(
        path
        for directory in (frozen, paths.output_directory)
        for path in directory.iterdir()
        if path.is_file()
    )
    before = {path: (path.read_bytes(), path.stat()) for path in watched}

    loaded: list[dict[str, object]] = []

    def load_selection(_root: Path, **identity: object) -> object:
        loaded.append(identity)
        return selection

    monkeypatch.setattr(selection_module, "load_target_pair_selection", load_selection)
    json_reads: list[Path] = []
    real_read_json = proximity_module.read_json

    def read_json(path: str | Path) -> object:
        normalized = Path(path)
        json_reads.append(normalized)
        if normalized != paths.run_config_json:
            pytest.fail(f"experiment plot-only unexpectedly read JSON: {normalized}")
        return real_read_json(normalized)

    monkeypatch.setattr(proximity_module, "read_json", read_json)
    csv_reads: list[Path] = []
    real_read_csv = proximity_module.pd.read_csv

    def read_csv(path: str | Path, **options: object) -> pd.DataFrame:
        normalized = Path(path)
        csv_reads.append(normalized)
        if normalized != paths.output_directory / "proximity.csv":
            pytest.fail(f"experiment plot-only unexpectedly read CSV: {normalized}")
        return real_read_csv(normalized, **options)

    monkeypatch.setattr(proximity_module.pd, "read_csv", read_csv)
    for name in (
        "generation_paths",
        "require_generation_run",
        "list_completed_records",
        "_load_sscd_config",
        "reference_completion_fingerprint",
        "write_analysis_outputs",
        "write_selection_figure",
        "write_gmm_fit_figure",
        "atomic_write_bytes",
        "atomic_write_frame_csv",
        "atomic_write_json",
    ):
        monkeypatch.setattr(
            proximity_module,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"experiment plot-only must not call {_name}"
            ),
        )
    rendered: list[Path] = []
    statistics = plotting_module.AnalysisStatistics(1, 1, 1, 1.0, -1.0)

    def render(directory: str | Path, *, figure_directory: str | Path) -> plotting_module.AnalysisStatistics:
        assert Path(figure_directory) == paths.figure_directory
        assert example_calls == [(paths, paths.output_directory / "proximity.csv", 2, True)]
        rendered.append(Path(directory))
        return statistics

    monkeypatch.setattr(proximity_module, "write_saved_analysis_figures", render)
    example_calls: list[tuple[ProximityPaths, Path, int, bool]] = []

    def write_examples(observed: ProximityPaths, *, table_path: Path, num_seeds: int,
                       validate_only: bool = False) -> None:
        example_calls.append((observed, table_path, num_seeds, validate_only))

    monkeypatch.setattr(proximity_module, "_write_examples", write_examples)

    result = proximity_module.plot_proximity(tmp_path, **arguments)

    assert result.exit_code == 0
    assert result.paths == paths
    assert loaded == [
        {
            "model_name": "sdv1",
            "scheduler_name": "ddim",
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "num_seeds": 2,
            "selection_strategy": "gmm",
        }
    ]
    assert json_reads == [paths.run_config_json]
    assert csv_reads == [paths.output_directory / "proximity.csv"]
    assert rendered == [paths.output_directory]
    assert example_calls == [
        (paths, paths.output_directory / "proximity.csv", 2, True),
        (paths, paths.output_directory / "proximity.csv", 2, False),
    ]
    assert result.values == {
        "complete": True,
        "plot_only": True,
        "selection_hash": "c" * 64,
        "prompt_spearman_summary": statistics.as_dict(),
    }
    assert set(watched) == {
        path
        for directory in (frozen, paths.output_directory)
        for path in directory.iterdir()
        if path.is_file()
    }
    for path, (contents, stat) in before.items():
        assert path.read_bytes() == contents
        assert path.stat().st_mtime_ns == stat.st_mtime_ns
        assert path.stat().st_ctime_ns == stat.st_ctime_ns
    assert unrelated.read_bytes() == b"preserve me"
    assert not paths.generation_run.exists()


def test_experiment_plot_proximity_rejects_incompatible_selection_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, paths, selection = _saved_experiment_plot_fixture(tmp_path)
    configuration = json.loads(paths.run_config_json.read_text(encoding="utf-8"))
    configuration["selection_hash"] = "d" * 64
    atomic_write_json(paths.run_config_json, configuration)
    monkeypatch.setattr(
        selection_module,
        "load_target_pair_selection",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        proximity_module,
        "write_saved_analysis_figures",
        lambda *_args, **_kwargs: pytest.fail(
            "incompatible saved provenance must be rejected before rendering"
        ),
    )

    with pytest.raises(ProximityError, match="differs at: selection_hash"):
        proximity_module.plot_proximity(tmp_path, **arguments)


def test_experiment_plot_proximity_rejects_table_different_from_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, paths, selection = _saved_experiment_plot_fixture(tmp_path)
    table_path = paths.output_directory / "proximity.csv"
    table = pd.read_csv(table_path, keep_default_na=False)
    table.loc[table["original_index"].eq(1), "prompt"] = "changed prompt"
    table.to_csv(table_path, index=False)
    monkeypatch.setattr(
        selection_module,
        "load_target_pair_selection",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        proximity_module,
        "write_saved_analysis_figures",
        lambda *_args, **_kwargs: pytest.fail(
            "incompatible saved table must be rejected before rendering"
        ),
    )

    with pytest.raises(ProximityError, match="frozen selection differ at prompt"):
        proximity_module.plot_proximity(tmp_path, **arguments)


def test_experiment_rows_receive_frozen_whole_prompt_decisions() -> None:
    observations = pd.concat(
        [_observations("1"), _observations("2")], ignore_index=True
    )
    annotated = _annotate_selection(observations, _selection(("1", "2")))

    assert tuple(annotated) == ANALYSIS_COLUMNS
    assert annotated.groupby("original_index")["include_prompt"].unique().map(
        tuple
    ).to_dict() == {"1": (True,), "2": (False,)}
    assert "prompt_gmm_evidence_seed_count" not in annotated
    assert set(annotated["selection_strategy"]) == {"gmm"}
    assert len(annotated) == 4


def test_partial_failed_experiment_preserves_valid_rows_only() -> None:
    annotated = _annotate_selection(
        _observations("1"), _selection(("1", "2")), require_complete=False
    )
    assert len(annotated) == 2
    with pytest.raises(ProximityError, match="every frozen prompt"):
        _annotate_selection(_observations("1"), _selection(("1", "2")))


def test_frozen_reference_fast_path_does_not_touch_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "selection"
    directory.mkdir()
    summary = {
        "complete": True,
        "selection_hash": "a" * 64,
        "selection_strategy": "gmm",
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    path_identities: list[dict[str, object]] = []
    load_identities: list[dict[str, object]] = []
    fingerprint = {
        "generation": {"count": 1, "sha256": "b" * 64},
        "sscd": {"count": 1, "sha256": "c" * 64},
    }
    loaded_frame = pd.DataFrame({"sentinel": [1]})
    loaded_configuration = {"reference_completion_fingerprint": fingerprint}

    def selection_directory(_root: Path, **identity: object) -> Path:
        path_identities.append(identity)
        return directory

    def load_selection(_root: Path, **identity: object) -> SimpleNamespace:
        load_identities.append(identity)
        return SimpleNamespace(
            sha256="a" * 64,
            frame=loaded_frame,
            configuration=loaded_configuration,
        )

    monkeypatch.setattr(
        selection_module, "target_pair_selection_directory", selection_directory
    )
    monkeypatch.setattr(selection_module, "load_target_pair_selection", load_selection)
    fingerprint_runs: list[Path] = []

    def completion_fingerprint(run: Path) -> dict[str, object]:
        fingerprint_runs.append(run)
        return fingerprint

    monkeypatch.setattr(
        proximity_module,
        "reference_completion_fingerprint",
        completion_fingerprint,
    )
    plotted: list[tuple[Path, Path]] = []

    def plot_selection(
        directory: str | Path, *, output_directory: str | Path, figure_directory: str | Path
    ) -> plotting_module.AnalysisStatistics:
        plotted.append((Path(directory), Path(figure_directory)))
        return plotting_module.AnalysisStatistics(1, 1, 1, 1.0, -1.0)

    monkeypatch.setattr(proximity_module, "write_selection_figure", plot_selection)
    gmm_plotted: list[tuple[Path, pd.DataFrame, object]] = []

    def plot_gmm(
        directory: str | Path,
        *,
        frame: pd.DataFrame,
        configuration: object,
        figure_directory: str | Path,
    ) -> None:
        gmm_plotted.append((Path(figure_directory), frame, configuration))

    monkeypatch.setattr(proximity_module, "write_gmm_fit_figure", plot_gmm)
    exported: list[tuple[ProximityPaths, Path, int]] = []

    def write_examples(
        paths: ProximityPaths, *, table_path: Path, num_seeds: int
    ) -> None:
        exported.append((paths, table_path, num_seeds))

    monkeypatch.setattr(proximity_module, "_write_examples", write_examples)
    result = proximity_module._frozen_reference_result(
        tmp_path,
        tmp_path / "reference",
        "sdv1",
        "ddpm",
        3.25,
        17,
        7,
        "gmm",
    )
    assert result is not None
    assert result.exit_code == 0
    assert result.paths.output_directory == directory
    output = (
        tmp_path
        / "outputs/sdv1_ddpm_g3.25_T17_N7/proximity/reference_S7_N7"
    )
    assert result.values == summary
    assert result.paths.summary_json == directory / "summary.json"
    figure_output = tmp_path / "figures/sdv1_ddpm_g3.25_T17_N7/proximity/reference_S7_N7"
    assert plotted == [(directory, figure_output)]
    assert len(gmm_plotted) == 1
    gmm_directory, gmm_frame, gmm_configuration = gmm_plotted[0]
    assert gmm_directory == figure_output
    assert gmm_frame is loaded_frame
    assert gmm_configuration is loaded_configuration
    assert [identity["selection_strategy"] for identity in path_identities] == ["gmm"]
    assert [identity["selection_strategy"] for identity in load_identities] == ["gmm"]
    assert fingerprint_runs == [tmp_path / "reference"]
    assert len(exported) == 1
    output_paths, table_path, exported_seeds = exported[0]
    assert output_paths.output_directory == output
    assert output_paths.generation_run == tmp_path / "reference"
    assert table_path == directory / "selection.csv"
    assert exported_seeds == 7


def test_frozen_reference_rejects_changed_markers_before_plotting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "selection"
    directory.mkdir()
    stored = {
        "generation": {"count": 1, "sha256": "a" * 64},
        "sscd": {"count": 1, "sha256": "b" * 64},
    }
    observed = {
        "generation": {"count": 1, "sha256": "c" * 64},
        "sscd": stored["sscd"],
    }
    monkeypatch.setattr(
        selection_module,
        "target_pair_selection_directory",
        lambda *_args, **_kwargs: directory,
    )
    monkeypatch.setattr(
        selection_module,
        "load_target_pair_selection",
        lambda *_args, **_kwargs: SimpleNamespace(
            sha256="d" * 64,
            frame=pd.DataFrame({"sentinel": [1]}),
            configuration={"reference_completion_fingerprint": stored},
        ),
    )
    monkeypatch.setattr(
        proximity_module,
        "reference_completion_fingerprint",
        lambda _run: observed,
    )
    plotted: list[Path] = []
    monkeypatch.setattr(
        proximity_module,
        "write_selection_figure",
        lambda path: plotted.append(Path(path)),
    )
    gmm_plotted: list[Path] = []
    monkeypatch.setattr(
        proximity_module,
        "write_gmm_fit_figure",
        lambda path, **_kwargs: gmm_plotted.append(Path(path)),
    )

    with pytest.raises(ProximityError) as captured:
        proximity_module._frozen_reference_result(
            tmp_path,
            tmp_path / "reference",
            "sdv1",
            "ddpm",
            3.25,
            17,
            7,
            "gmm",
        )

    assert plotted == []
    assert gmm_plotted == []
    assert directory.is_dir()
    message = str(captured.value)
    assert "changed marker groups: generation" in message
    assert "run_all.sh --overwrite" not in message
    assert (
        "./compute_proximity.sh --model sdv1 --scheduler ddpm --g 3.25 "
        "--T 17 --N 7 --seed-start 7 --selection-strategy gmm --overwrite" in message
    )


def test_frozen_reference_reports_cache_only_rebuild_for_old_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "selection"
    directory.mkdir()
    monkeypatch.setattr(
        selection_module,
        "target_pair_selection_directory",
        lambda *_args, **_kwargs: directory,
    )
    monkeypatch.setattr(
        selection_module,
        "load_target_pair_selection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            selection_module.TargetPairSelectionError(
                "selection.csv has an invalid schema"
            )
        ),
    )

    with pytest.raises(ProximityError) as captured:
        proximity_module._frozen_reference_result(
            tmp_path,
            tmp_path / "reference",
            "sdv2",
            "ddim",
            7.5,
            50,
            20,
            "gmm",
        )

    message = str(captured.value)
    assert "selection.csv has an invalid schema" in message
    assert (
        "./compute_proximity.sh --model sdv2 --scheduler ddim --g 7.5 "
        "--T 50 --N 20 --seed-start 20 --selection-strategy gmm --overwrite" in message
    )
    assert "without regenerating trajectories or SSCD" in message
    assert (
        "./run_all.sh --model sdv2 --scheduler ddim --g 7.5 --T 50 --N 20 "
        "--selection-strategy gmm" in message
    )
    assert "--overwrite-selection" not in message


def test_reference_overwrite_recomputes_cached_observations_and_replaces_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "reference"
    run.mkdir()
    run_config = run / "run_config.json"
    run_config.write_text("{}", encoding="utf-8")
    (run / "sscd_config.json").write_text("{}", encoding="utf-8")
    cache = SimpleNamespace(
        run_directory=run,
        run_config=run_config,
        record_directory=run / "record",
    )
    generation = {"scientific_config": {"seeds": [3, 4, 5]}}
    completed = SimpleNamespace(
        original_index="42",
        source_row_number=7,
        metadata={"record_id": "sdv1-42"},
    )
    record = _record(index="42", seeds=[3, 4, 5])
    frozen = SimpleNamespace(paths=SimpleNamespace(), values={})
    fast_path_calls: list[tuple[object, ...]] = []
    built: list[dict[str, object]] = []

    def frozen_result(*arguments: object) -> object:
        fast_path_calls.append(arguments)
        return frozen

    def build_selection(_root: Path, **arguments: object) -> None:
        built.append(arguments)

    monkeypatch.setattr(
        proximity_module, "generation_paths", lambda _root, **_kwargs: cache
    )
    monkeypatch.setattr(
        proximity_module, "require_generation_run", lambda _cache: generation
    )
    monkeypatch.setattr(
        proximity_module,
        "_validate_generation_invocation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        proximity_module, "_load_sscd_config", lambda *_args: {"cached": True}
    )
    monkeypatch.setattr(
        proximity_module, "list_completed_records", lambda _cache: [completed]
    )
    monkeypatch.setattr(
        proximity_module, "_require_generation_coverage", lambda *_args: None
    )
    monkeypatch.setattr(proximity_module, "_validated_record", lambda *_args: record)
    monkeypatch.setattr(
        proximity_module,
        "_terminal_l2",
        lambda _record: torch.tensor([1.0, 2.0, 3.0]),
    )
    monkeypatch.setattr(
        proximity_module,
        "_sscd_scores",
        lambda *_args: torch.tensor([0.9, 0.5, 0.1]),
    )
    monkeypatch.setattr(
        proximity_module,
        "_reference_records_frame",
        lambda *_args: pd.DataFrame({"original_index": ["42"]}),
    )
    monkeypatch.setattr(
        selection_module, "build_target_pair_selection", build_selection
    )
    monkeypatch.setattr(proximity_module, "_frozen_reference_result", frozen_result)
    arguments = _run_args(num_seeds=3, seed_start=3)

    assert (
        proximity_module.run_proximity(tmp_path, overwrite=True, **arguments) is frozen
    )
    assert len(fast_path_calls) == 1
    assert len(built) == 1
    assert built[0]["overwrite"] is True
    observations = built[0]["paired_frame"]
    assert isinstance(observations, pd.DataFrame)
    assert observations[["l2_norm", "sscd"]].to_numpy().tolist() == [
        [1.0, pytest.approx(0.9)],
        [2.0, pytest.approx(0.5)],
        [3.0, pytest.approx(0.1)],
    ]


def test_analysis_configuration_and_summary_preserve_selection_strategy(
    tmp_path: Path,
) -> None:
    paths = ProximityPaths.build(
        tmp_path.resolve(),
        Path(RUN_NAME) / "experiment_S0_N20",
        role="experiment",
        seed_start=0,
        num_seeds=20,
        selection_strategy="gmm",
    )
    generation = {
        "scientific_config_hash": "a" * 64,
        "scientific_config": {
            "model_cli_name": "sdv1",
            "scheduler": {"name": "ddim"},
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "num_seeds": 20,
            "seeds": list(range(20)),
        },
    }
    sscd = {"configuration_hash": "b" * 64}
    selection = _selection(("1", "2"))
    config = proximity_module._analysis_configuration(
        paths, RUN_NAME, generation, sscd, selection
    )
    assert config["selection_hash"] == "c" * 64
    assert config["selection_policy"] == selection_module.selection_policy("gmm")
    assert config["selection_strategy"] == "gmm"
    assert config["outputs"] == {
        "table": "proximity.csv",
        "figures": plotting_module.PROXIMITY_FIGURES,
    }
    assert "schema_version" not in config
    assert "source_provenance" not in config

    analysis = _annotate_selection(
        pd.concat([_observations("1"), _observations("2")], ignore_index=True),
        selection,
    )
    summary = proximity_module._summary(
        paths,
        RUN_NAME,
        generation,
        sscd,
        selection,
        analysis,
        duration=1.25,
        statistics=plotting_module.AnalysisStatistics(1, 1, 1, 1.0, -1.0),
    )
    assert summary["selection_hash"] == "c" * 64
    assert summary["selection_policy"] == selection_module.selection_policy("gmm")
    assert summary["selection_strategy"] == "gmm"
    assert summary["figures"] == {
        scope: {
            file_format: (paths.figure_directory / filename)
            .relative_to(tmp_path.resolve())
            .as_posix()
            for file_format, filename in formats.items()
        }
        for scope, formats in plotting_module.PROXIMITY_PDF_FIGURES.items()
    }


def test_configuration_never_accepts_a_different_contract(tmp_path: Path) -> None:
    path = tmp_path / "run_config.json"
    existing = {
        "selection_hash": "a" * 64,
        "outputs": {"figure": "proximity_vs_sscd.png"},
    }
    desired = {
        "selection_hash": "a" * 64,
        "outputs": {
            "table": "proximity.csv",
            "figures": plotting_module.PROXIMITY_FIGURES,
        },
    }
    proximity_module._write_configuration(path, existing)
    proximity_module._write_configuration(path, desired)
    assert json.loads(path.read_text(encoding="utf-8")) == desired
    with pytest.raises(ProximityError, match="incompatible"):
        proximity_module._write_configuration(
            path,
            {**desired, "selection_hash": "b" * 64},
        )
    replacement = {**desired, "selection_hash": "b" * 64}
    proximity_module._write_configuration(path, replacement, overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8")) == replacement


def test_saved_analysis_plotting_regenerates_figures_without_rewriting_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _annotate_selection(
        pd.concat([_observations("1"), _observations("2")], ignore_index=True),
        _selection(("1", "2")),
    )
    expected = write_analysis_outputs(tmp_path, analysis=analysis)
    csv_path = tmp_path / "proximity.csv"
    csv_contents = csv_path.read_bytes()
    csv_stat = csv_path.stat()
    for filename in plotting_module.PROXIMITY_PDF_FILENAMES:
        (tmp_path / filename).unlink()
    monkeypatch.setattr(
        plotting_module,
        "atomic_write_frame_csv",
        lambda *_args, **_kwargs: pytest.fail(
            "saved-analysis plotting must not rewrite proximity.csv"
        ),
    )

    actual = plotting_module.write_saved_analysis_figures(tmp_path)

    assert actual == expected
    assert csv_path.read_bytes() == csv_contents
    assert csv_path.stat().st_mtime_ns == csv_stat.st_mtime_ns
    assert csv_path.stat().st_ctime_ns == csv_stat.st_ctime_ns
    assert {
        path.name for path in tmp_path.iterdir()
    } == {"proximity.csv", *plotting_module.PROXIMITY_PDF_FILENAMES}
    for filename in plotting_module.PROXIMITY_PDF_FILENAMES:
        artifact = tmp_path / filename
        header = b"%PDF" if artifact.suffix == ".pdf" else b"\x89PNG"
        assert artifact.read_bytes().startswith(header)


def test_pdf_cleanup_preserves_frozen_catalog_and_historical_pngs(
    tmp_path: Path,
) -> None:
    assert plotting_module.PROXIMITY_FIGURES == {
        "selected": {
            "png": "proximity_vs_sscd.png",
            "pdf": "proximity_vs_sscd.pdf",
        },
        "all_prompts": {
            "png": "proximity_vs_sscd_all_prompts.png",
            "pdf": "proximity_vs_sscd_all_prompts.pdf",
        },
    }
    assert plotting_module.PROXIMITY_FIGURE_FILENAMES == (
        "proximity_vs_sscd.png",
        "proximity_vs_sscd.pdf",
        "proximity_vs_sscd_all_prompts.png",
        "proximity_vs_sscd_all_prompts.pdf",
    )
    for filename in plotting_module.PROXIMITY_FIGURE_FILENAMES:
        (tmp_path / filename).write_bytes(b"old figure")
    unrelated = tmp_path / "proximity.csv"
    unrelated.write_bytes(b"keep")

    proximity_module._remove_figure_outputs(tmp_path)

    assert unrelated.read_bytes() == b"keep"
    assert {path.name for path in tmp_path.iterdir()} == {"proximity.csv", "proximity_vs_sscd.png", "proximity_vs_sscd_all_prompts.png"}
    assert plotting_module.PROXIMITY_PDF_FILENAMES == ("proximity_vs_sscd.pdf", "proximity_vs_sscd_all_prompts.pdf")


def _gmm_figure_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.DataFrame(
        {
            "original_index": ["low", "low", "high", "high", "incomplete", "incomplete"],
            "l2_norm": [1.0, 2.0, 4.0, 5.0, 2.5, float("nan")],
            "sscd": [0.1, 0.2, 0.8, 0.7, 0.2, float("nan")],
            "observation_status": [
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "missing",
            ],
            "gmm_component": [
                "low_sscd_mode",
                "low_sscd_mode",
                "high_sscd_mode",
                "high_sscd_mode",
                "low_sscd_mode",
                "",
            ],
        }
    )
    configuration: dict[str, object] = {
        "gmm_fit": {
            "feature_names": ["l2_norm", "sscd"],
            "component_names": ["low_sscd_mode", "high_sscd_mode"],
            "covariance_type": "full",
            "feature_mean": [10.0, 0.5],
            "feature_scale": [2.0, 0.25],
            "means_standardized": [[-1.0, -1.0], [1.0, 1.0]],
            "covariances_standardized": [
                [[1.0, 0.6], [0.6, 0.5]],
                [[0.7, -0.3], [-0.3, 0.4]],
            ],
            "usable_observation_count": 5,
        }
    }
    return frame, configuration


def test_gmm_fit_figure_writes_only_pdf_from_all_fitted_reference_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_rgba
    from utils.experiments import proximity_style
    from matplotlib.patches import Ellipse

    frame, configuration = _gmm_figure_inputs()
    captured: list[object] = []
    export_calls: list[dict[str, object]] = []
    real_publish = plotting_module._publish_figures

    def publish(output: Path, figures: object, **options) -> None:
        items = tuple(figures)  # type: ignore[arg-type]
        captured.extend(figure for figure, _filenames in items)
        export_calls.append(options)
        real_publish(output, items, **options)

    monkeypatch.setattr(plotting_module, "_publish_figures", publish)
    plotting_module.write_gmm_fit_figure(
        tmp_path,
        frame=frame,
        configuration=configuration,
    )

    assert plotting_module.SELECTION_FIGURE_FILENAMES == (
        *plotting_module.PROXIMITY_FIGURE_FILENAMES,
        plotting_module.SELECTION_GMM_FIGURES["png"],
        plotting_module.SELECTION_GMM_FIGURES["pdf"],
    )
    assert {path.name for path in tmp_path.iterdir()} == set(
        (plotting_module.SELECTION_GMM_FIGURES["pdf"],)
    )
    assert not any(
        path.is_dir() or path.name.startswith(".") for path in tmp_path.iterdir()
    )
    for file_format, filename in {"pdf": plotting_module.SELECTION_GMM_FIGURES["pdf"]}.items():
        header = b"%PDF" if file_format == "pdf" else b"\x89PNG"
        assert (tmp_path / filename).read_bytes().startswith(header)

    assert len(captured) == 1
    axis = captured[0].axes[0]  # type: ignore[union-attr]
    scatter_groups = [
        collection
        for collection in axis.collections
        if isinstance(collection, PathCollection)
    ]
    assert len(scatter_groups) == 1
    fitted = frame.loc[frame["observation_status"].eq("complete")]
    points = scatter_groups[0]
    observed_xy = [tuple(pair) for pair in points.get_offsets().tolist()]
    assert Counter(observed_xy) == Counter(map(tuple, fitted[["l2_norm", "sscd"]].to_numpy()))
    assert (2.5, 0.2) in observed_xy
    source_components = {(float(row.l2_norm), float(row.sscd)): row.gmm_component
                         for row in fitted.itertuples()}
    for xy, color in zip(observed_xy, points.get_facecolors(), strict=True):
        assert color == pytest.approx(to_rgba(
            proximity_style.GMM_COMPONENT_COLORS[source_components[xy]], alpha=.35))
    assert points.get_sizes().tolist() == [10.]
    assert points.get_alpha() == .35 and points.get_rasterized() is True
    assert len(export_calls) == 1
    assert export_calls[0]["formats"] == ("pdf",)
    assert export_calls[0]["export_options"][plotting_module.SELECTION_GMM_FIGURES["pdf"]]["dpi"] == 150

    fit = configuration["gmm_fit"]
    assert isinstance(fit, dict)
    feature_mean = np.asarray(fit["feature_mean"], dtype=np.float64)
    feature_scale = np.asarray(fit["feature_scale"], dtype=np.float64)
    means = feature_mean + np.asarray(
        fit["means_standardized"], dtype=np.float64
    ) * feature_scale
    covariances = (
        np.asarray(fit["covariances_standardized"], dtype=np.float64)
        * feature_scale[None, :, None]
        * feature_scale[None, None, :]
    )
    assert np.all(np.abs(covariances[:, 0, 1]) > 1e-6)
    ellipses = [patch for patch in axis.patches if isinstance(patch, Ellipse)]
    assert len(ellipses) == 4
    for component_index in range(2):
        one_sigma, two_sigma = ellipses[2 * component_index : 2 * component_index + 2]
        np.testing.assert_allclose(one_sigma.center, means[component_index])
        np.testing.assert_allclose(two_sigma.center, means[component_index])
        assert two_sigma.width == pytest.approx(2.0 * one_sigma.width)
        assert two_sigma.height == pytest.approx(2.0 * one_sigma.height)
    assert all(
        not math.isclose(ellipse.angle % 90.0, 0.0, abs_tol=1e-10)
        for ellipse in ellipses
    )


def test_gmm_fit_figure_rejects_invalid_configuration_without_writing(
    tmp_path: Path,
) -> None:
    frame, configuration = _gmm_figure_inputs()
    malformed = json.loads(json.dumps(configuration))
    malformed["gmm_fit"]["means_standardized"] = [[0.0, 0.0]]
    nonsymmetric = json.loads(json.dumps(configuration))
    nonsymmetric["gmm_fit"]["covariances_standardized"][0][0][1] += 0.1

    for name, invalid, message in (
        ("malformed", malformed, "GMM fit arrays are invalid"),
        ("nonsymmetric", nonsymmetric, "covariance is not symmetric"),
    ):
        output = tmp_path / name
        with pytest.raises(plotting_module.PlottingError, match=message):
            plotting_module.write_gmm_fit_figure(
                output,
                frame=frame,
                configuration=invalid,
            )
        assert not output.exists()


def test_figure_publication_stages_every_format_before_replacing_destinations(
    tmp_path: Path,
) -> None:
    for filename in plotting_module.PROXIMITY_FIGURE_FILENAMES:
        (tmp_path / filename).write_bytes(b"previous complete figure")
    save_calls: list[Path] = []

    class FakeFigure:
        def savefig(self, destination: Path, **_kwargs: object) -> None:
            path = Path(destination)
            save_calls.append(path)
            path.write_bytes(b"new staged figure")
            if len(save_calls) == 3:
                raise RuntimeError("synthetic PDF/PNG render failure")

    with pytest.raises(RuntimeError, match="synthetic PDF/PNG render failure"):
        plotting_module._publish_figures(
            tmp_path,
            (
                (FakeFigure(), plotting_module.PROXIMITY_FIGURES["selected"]),
                (FakeFigure(), plotting_module.PROXIMITY_FIGURES["all_prompts"]),
            ),
        )

    assert len(save_calls) == 3
    assert all(
        (tmp_path / filename).read_bytes() == b"previous complete figure"
        for filename in plotting_module.PROXIMITY_FIGURE_FILENAMES
    )
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_figure_publication_rolls_back_a_partial_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = {
        filename: f"previous {filename}".encode()
        for filename in plotting_module.PROXIMITY_FIGURE_FILENAMES
    }
    for filename, contents in previous.items():
        (tmp_path / filename).write_bytes(contents)

    real_replace = plotting_module.os.replace
    destinations = {tmp_path / filename for filename in previous}
    install_count = 0
    failed = False

    def replace_with_one_failure(source: str | Path, destination: str | Path) -> None:
        nonlocal failed, install_count
        target = Path(destination)
        if target in destinations:
            install_count += 1
            if install_count == 2 and not failed:
                failed = True
                raise OSError("synthetic install failure")
        real_replace(source, destination)

    monkeypatch.setattr(plotting_module.os, "replace", replace_with_one_failure)

    class FakeFigure:
        def savefig(self, destination: Path, **_kwargs: object) -> None:
            Path(destination).write_bytes(b"new staged figure")

    with pytest.raises(OSError, match="synthetic install failure"):
        plotting_module._publish_figures(
            tmp_path,
            (
                (FakeFigure(), plotting_module.PROXIMITY_FIGURES["selected"]),
                (FakeFigure(), plotting_module.PROXIMITY_FIGURES["all_prompts"]),
            ),
        )

    assert failed
    assert {
        filename: (tmp_path / filename).read_bytes() for filename in previous
    } == previous
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_selection_figure_publishes_selected_and_completed_finite_group_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = pd.DataFrame(
        {
            "original_index": [
                "included",
                "included",
                "discarded",
                "discarded",
                "unusable",
                "unusable",
                "failed",
                "failed",
            ],
            "prompt": [
                "kept",
                "kept",
                "dropped",
                "dropped",
                "bad",
                "bad",
                "failed but finite",
                "failed but finite",
            ],
            "kind": ["MV", "MV", "TV", "TV", "N", "N", "RV", "RV"],
            "seed": [3, 4, 3, 4, 3, 4, 3, 4],
            "generated_image_path": [
                "included.png",
                "included.png",
                "discarded.png",
                "discarded.png",
                "unusable.png",
                "unusable.png",
                "failed.png",
                "failed.png",
            ],
            "generated_image_tile_index": [0, 1, 0, 1, 0, 1, 0, 1],
            "l2_norm": [
                1.0,
                2.0,
                8.0,
                8.0,
                20.0,
                float("nan"),
                15.0,
                16.0,
            ],
            "sscd": [0.9, 0.1, 0.5, 0.5, 0.8, float("nan"), 0.4, 0.3],
            "observation_status": [
                "complete",
                "complete",
                "complete",
                "complete",
                "cache_error",
                "cache_error",
                "complete",
                "cache_error",
            ],
            "selection_strategy": ["gmm"] * 8,
            "prompt_spearman": [
                -1.0,
                -1.0,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                -1.0,
                -1.0,
            ],
            "include_prompt": [True, True, False, False, False, False, False, False],
            "selection_status": [
                "included_proximity_rule",
                "included_proximity_rule",
                "discarded_proximity_rule",
                "discarded_proximity_rule",
                "unusable_reference_observations",
                "unusable_reference_observations",
                "unusable_reference_observations",
                "unusable_reference_observations",
            ],
            "selection_reason": [
                "no_low_sscd_mode_majority",
                "no_low_sscd_mode_majority",
                "majority_reference_seeds_in_low_sscd_mode",
                "majority_reference_seeds_in_low_sscd_mode",
                "cache_error",
                "cache_error",
                "cache_error",
                "cache_error",
            ],
        }
    )
    selection.to_csv(tmp_path / "selection.csv", index=False)

    read_paths: list[Path] = []
    plotted_frames: list[pd.DataFrame] = []
    plotted_figures: list[object] = []
    real_read_csv = plotting_module.pd.read_csv
    real_scatter_figure = plotting_module._scatter_figure

    def read_csv_spy(path: str | Path, *args: object, **kwargs: object) -> pd.DataFrame:
        read_paths.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    def scatter_spy(
        frame: pd.DataFrame,
        statistics: plotting_module.AnalysisStatistics,
    ) -> object:
        plotted_frames.append(frame.copy(deep=True))
        figure = real_scatter_figure(frame, statistics)
        plotted_figures.append(figure)
        return figure

    monkeypatch.setattr(plotting_module.pd, "read_csv", read_csv_spy)
    monkeypatch.setattr(plotting_module, "_scatter_figure", scatter_spy)

    output = tmp_path / "derived"
    statistics = write_selection_figure(tmp_path, output_directory=output)

    assert statistics.as_dict() == {
        "total_selected_prompts": 1,
        "evaluable_selected_prompts": 1,
        "negative_spearman_prompts": 1,
        "negative_spearman_fraction": pytest.approx(1.0),
        "median_spearman": pytest.approx(-1.0),
    }
    assert read_paths == [tmp_path / "selection.csv"]
    assert len(plotted_frames) == 2
    assert len(plotted_figures) == 2
    all_frame, selected_frame = plotted_frames
    assert set(selected_frame["original_index"]) == {"included"}
    assert set(all_frame["original_index"]) == {"included", "discarded"}
    assert all_frame[["l2_norm", "sscd"]].notna().all().all()
    all_axes = plotted_figures[0].axes[0]
    selected_axes = plotted_figures[1].axes[0]
    assert selected_axes.get_xlim() == pytest.approx(all_axes.get_xlim())
    assert selected_axes.get_ylim() == pytest.approx(all_axes.get_ylim())
    assert 8.0 < all_axes.get_xlim()[1] < 20.0
    # Failed rows make CSV measurement columns object-typed; scatter coordinates
    # must remain numeric rather than becoming Matplotlib category positions.
    assert len(all_axes.collections) == 1
    assert Counter(map(tuple, all_axes.collections[0].get_offsets().tolist())) == Counter(
        [(1.0, 0.9), (2.0, 0.1), (8.0, 0.5), (8.0, 0.5)])
    assert _proximity_summary(plotted_figures[1]).get_text().startswith("#Prompts: 1")
    assert _proximity_summary(plotted_figures[0]).get_text().startswith("#Prompts: 2")
    assert {path.name for path in tmp_path.iterdir()} == {"selection.csv", "derived"}
    assert {path.name for path in output.iterdir()} == set(
        plotting_module.PROXIMITY_PDF_FILENAMES
    )
    assert not any(path.is_dir() for path in output.iterdir())
    assert (output / "proximity_vs_sscd.pdf").read_bytes().startswith(b"%PDF")
    assert (
        (output / "proximity_vs_sscd_all_prompts.pdf")
        .read_bytes()
        .startswith(b"%PDF")
    )


def test_selection_figure_requires_observation_status(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "original_index": ["included", "included"],
            "l2_norm": [1.0, 2.0],
            "sscd": [0.9, 0.1],
            "prompt_spearman": [-1.0, -1.0],
            "include_prompt": [True, True],
        }
    ).to_csv(tmp_path / "selection.csv", index=False)

    output = tmp_path / "derived"
    with pytest.raises(plotting_module.PlottingError, match="observation_status"):
        write_selection_figure(tmp_path, output_directory=output)
    assert not output.exists()


def test_analysis_outputs_publish_prompt_level_summary_from_saved_seed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = pd.DataFrame(
        {
            "original_index": [
                "a",
                "a",
                "b",
                "b",
                "c",
                "c",
                "d",
                "d",
            ],
            "prompt": [
                "negative",
                "negative",
                "positive",
                "positive",
                "undefined",
                "undefined",
                "excluded",
                "excluded",
            ],
            "kind": ["N", "N", "TV", "TV", "RV", "RV", "MV", "MV"],
            "seed": [0, 1, 0, 1, 0, 1, 0, 1],
            "generated_image_path": [
                "a.png",
                "a.png",
                "b.png",
                "b.png",
                "c.png",
                "c.png",
                "d.png",
                "d.png",
            ],
            "generated_image_tile_index": [0, 1, 0, 1, 0, 1, 0, 1],
            "l2_norm": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
            "sscd": [4.0, 3.0, 1.0, 4.0, 2.0, 3.0, 9.0, 8.0],
            "selection_strategy": ["gmm"] * 8,
            "prompt_spearman": [
                -0.8,
                -0.8,
                -0.5,
                -0.5,
                -0.2,
                -0.2,
                0.4,
                0.4,
            ],
            "include_prompt": [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ],
            "selection_reason": [
                "no_low_sscd_mode_majority",
                "no_low_sscd_mode_majority",
                "no_low_sscd_mode_majority",
                "no_low_sscd_mode_majority",
                "no_low_sscd_mode_majority",
                "no_low_sscd_mode_majority",
                "majority_reference_seeds_in_low_sscd_mode",
                "majority_reference_seeds_in_low_sscd_mode",
            ],
        }
    )
    original_style = {
        key: plotting_module.matplotlib.rcParams[key]
        for key in ("font.family", "font.size", "mathtext.fontset")
    }
    figures: list[object] = []
    styles: list[dict[str, object]] = []
    save_calls: list[dict[str, object]] = []
    tight_layout_calls: list[int] = []
    read_paths: list[Path] = []
    real_subplots = plotting_module.plt.subplots
    real_read_csv = plotting_module.pd.read_csv

    def read_csv_spy(path: str | Path, *args: object, **kwargs: object) -> pd.DataFrame:
        csv_path = Path(path)
        assert csv_path.is_file()
        read_paths.append(csv_path)
        return real_read_csv(path, *args, **kwargs)

    def subplots_spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        figure, axis = real_subplots(*args, **kwargs)
        real_savefig, real_tight_layout = figure.savefig, figure.tight_layout

        def savefig_spy(*args: object, **kwargs: object) -> object:
            save_calls.append(dict(kwargs))
            return real_savefig(*args, **kwargs)

        def tight_layout_spy(*args: object, **kwargs: object) -> object:
            tight_layout_calls.append(1)
            return real_tight_layout(*args, **kwargs)

        monkeypatch.setattr(figure, "savefig", savefig_spy)
        monkeypatch.setattr(figure, "tight_layout", tight_layout_spy)
        figures.append(figure)
        styles.append(
            {
                "font.family": tuple(
                    plotting_module.matplotlib.rcParams["font.family"]
                ),
                "font.size": plotting_module.matplotlib.rcParams["font.size"],
                "mathtext.fontset": plotting_module.matplotlib.rcParams[
                    "mathtext.fontset"
                ],
            }
        )
        return figure, axis

    monkeypatch.setattr(plotting_module.plt, "subplots", subplots_spy)
    monkeypatch.setattr(plotting_module.pd, "read_csv", read_csv_spy)
    statistics = write_analysis_outputs(tmp_path, analysis=analysis)

    assert statistics.as_dict() == {
        "total_selected_prompts": 3,
        "evaluable_selected_prompts": 2,
        "negative_spearman_prompts": 1,
        "negative_spearman_fraction": pytest.approx(0.5),
        "median_spearman": pytest.approx(0.0),
    }
    assert read_paths == [tmp_path / "proximity.csv"]
    saved = real_read_csv(tmp_path / "proximity.csv")
    expected = analysis.copy()
    expected["experiment_prompt_spearman"] = [
        -1.0,
        -1.0,
        1.0,
        1.0,
        None,
        None,
        -1.0,
        -1.0,
    ]
    pd.testing.assert_frame_equal(saved, expected)
    assert (
        saved.loc[saved["original_index"].eq("c"), "experiment_prompt_spearman"]
        .isna()
        .all()
    )
    assert {path.name for path in tmp_path.iterdir()} == {
        "proximity.csv",
        *plotting_module.PROXIMITY_PDF_FILENAMES,
    }
    assert {
        key: plotting_module.matplotlib.rcParams[key] for key in original_style
    } == original_style
    expected_style = {
        "font.family": ("STIXGeneral",),
        "font.size": 15.0,
        "mathtext.fontset": "stix",
    }
    assert styles == [expected_style, expected_style]
    assert len(figures) == 2
    assert not tight_layout_calls  # The paired figures keep a fixed physical data box.
    all_figure, figure = figures
    for panel in figures:
        width, height = panel.get_size_inches()
        position = panel.axes[0].get_position()
        assert (position.width * width, position.height * height) == pytest.approx((3.1, 3.1))
    assert tuple(figure.get_size_inches()) == pytest.approx((4.45, 4.2))
    assert tuple(figure.get_size_inches()) == pytest.approx(tuple(all_figure.get_size_inches()))
    assert all(getattr(panel, "_proximity_canvas_extent_artist", None) is None for panel in figures)
    assert figure._suptitle is None
    axis = figure.axes[0]
    all_axis = all_figure.axes[0]
    assert axis.get_title() == ""
    assert axis.get_xscale() == axis.get_yscale() == "linear"
    assert axis.get_xlabel() == r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|$"
    assert axis.get_ylabel() == "SSCD"
    assert axis.xaxis.label.get_fontsize() == 18
    assert axis.yaxis.label.get_fontsize() == 18
    assert axis.xaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert axis.yaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert all(label.get_fontsize() == 15 for label in axis.get_xticklabels())
    assert all(label.get_fontsize() == 15 for label in axis.get_yticklabels())
    summary = _proximity_summary(figure)
    all_summary = _proximity_summary(all_figure)
    assert summary.get_text() == (
        "#Prompts: 3\n" + r"Median $\rho$: 0.000" + "\n"
        + r"$\rho < 0$: 1/2 (50.0%) evaluable")
    assert all_summary.get_text() == (
        "#Prompts: 4\n" + r"Median $\rho$: -1.000" + "\n"
        + r"$\rho < 0$: 2/3 (66.7%) evaluable")
    assert axis.get_xlim() == pytest.approx(all_axis.get_xlim())
    assert axis.get_ylim() == pytest.approx(all_axis.get_ylim())
    assert axis.get_xticks() == pytest.approx(all_axis.get_xticks())
    assert axis.get_yticks() == pytest.approx(all_axis.get_yticks())
    assert all_axis.get_xlim()[1] > 7.0
    assert all_axis.get_ylim()[1] > 9.0
    assert "PCC" not in summary.get_text()
    for panel, label in ((figure, summary), (all_figure, all_summary)):
        assert label.get_horizontalalignment() == "left"
        assert label.get_verticalalignment() == "bottom"
        assert label.get_position() == pytest.approx((.02, .02))
        assert label.get_transform() is panel.axes[0].transAxes
        box = label.get_bbox_patch()
        assert box is not None and box.get_alpha() == .85
        assert box.get_facecolor() == pytest.approx((1., 1., 1., .85))
        assert box.get_edgecolor() == pytest.approx((.75, .75, .75, .85))
        assert box.get_boxstyle().pad == .3
        assert label.get_fontsize() == 12
        assert label.get_fontfamily() == ["STIXGeneral"]
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_rgba

    assert len(figure.axes) == len(all_figure.axes) == 1
    for plotted_axis, kinds in (
        (all_axis, ("MV", "RV", "TV", "N")),
        (axis, ("RV", "TV", "N")),
    ):
        assert not plotted_axis.lines
        legend = plotted_axis.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == [
            plotting_module.category_legend_label(kind) for kind in kinds
        ]
        assert all(text.get_fontsize() == 12 for text in legend.get_texts())
        assert legend.get_title().get_text() == ""
        assert not legend.get_frame_on()
        assert legend._loc == 1  # upper right, inside the axes
        assert legend._ncols == 1
        expected_colors = _proximity_category_colors()
        assert len(plotted_axis.collections) == 1
        points = plotted_axis.collections[0]
        assert isinstance(points, PathCollection)
        assert points.get_alpha() == pytest.approx(0.35)
        assert points.get_sizes().tolist() == [10.0]
        assert points.get_rasterized()
        assert len(points.get_edgecolors()) == 0
        for handle, kind in zip(legend.legend_handles, kinds, strict=True):
            assert handle.get_alpha() == 1.0
            assert handle.get_linestyle() == "None"
            assert to_rgba(handle.get_markerfacecolor()) == pytest.approx(
                to_rgba(expected_colors[kind]))
        source = analysis if plotted_axis is all_axis else analysis.loc[analysis["include_prompt"]]
        expected = Counter((float(row.l2_norm), float(row.sscd),
                            tuple(to_rgba(expected_colors[row.kind], alpha=.35)))
                           for row in source.itertuples())
        actual = Counter((float(x), float(y), tuple(color))
                         for (x, y), color in zip(points.get_offsets(), points.get_facecolors(), strict=True))
        assert actual == expected
    assert save_calls == [
        {
            "format": "pdf",
            "dpi": 150,
            "bbox_inches": "tight",
            "pad_inches": 0.05,
        },
        {
            "format": "pdf",
            "dpi": 150,
            "bbox_inches": "tight",
            "pad_inches": 0.05,
        },
    ]
    for filename in plotting_module.PROXIMITY_PDF_FILENAMES:
        artifact = tmp_path / filename
        assert artifact.is_file()
        expected_header = b"%PDF" if artifact.suffix == ".pdf" else b"\x89PNG"
        assert artifact.read_bytes().startswith(expected_header)
    assert all(
        not plotting_module.plt.fignum_exists(plotted.number) for plotted in figures
    )


def test_proximity_scatter_preserves_singletons_and_normalizes_categories() -> None:
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_rgba

    frame = pd.DataFrame(
        {
            "original_index": [str(index) for index in range(8)],
            "seed": [0] * 8,
            "l2_norm": list(range(8)),
            "sscd": [0.1 * index for index in range(8)],
            "kind": [" mv ", "rv", "TV", "n", "unknown", "", None, float("nan")],
        }
    )
    statistics = plotting_module.AnalysisStatistics(8, 0, 0, None, None)
    with plotting_module.matplotlib.rc_context(plotting_module.PLOT_STYLE):
        figure = plotting_module._scatter_figure(frame, statistics)
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert not axis.lines
        assert len(axis.collections) == 1
        kinds = ("MV", "RV", "TV", "N", "Other / unlabeled")
        colors = _proximity_category_colors()
        from matplotlib.legend import Legend
        legends = [item for item in axis.get_children() if isinstance(item, Legend)]
        assert len(legends) == 1
        assert legends[0]._loc == 1 and legends[0]._ncols == 1
        assert [label.get_text() for legend in legends for label in legend.get_texts()] == [
            plotting_module.category_legend_label(kind) for kind in kinds]
        assert legends[0].get_title().get_text() == ""
        points = axis.collections[0]
        assert isinstance(points, PathCollection)
        assert len(points.get_offsets()) == len(frame)
        expected = {index: kind for index, kind in enumerate(kinds[:4])}
        expected.update({index: "Other / unlabeled" for index in range(4, 8)})
        for (x, y), color in zip(points.get_offsets(), points.get_facecolors(), strict=True):
            assert y == pytest.approx(.1 * x)
            assert color == pytest.approx(to_rgba(colors[expected[int(x)]], alpha=.35))
        assert all(handle.get_alpha() == 1.0 for legend in legends for handle in legend.legend_handles)
    finally:
        plotting_module.plt.close(figure)


@pytest.mark.parametrize("empty", (False, True))
def test_proximity_scatter_handles_missing_categories_and_empty_frames(
    empty: bool,
) -> None:
    from matplotlib.colors import to_rgba

    frame = pd.DataFrame(
        {
            "original_index": [] if empty else ["singleton"],
            "l2_norm": [] if empty else [2.0],
            "sscd": [] if empty else [0.7],
        }
    )
    statistics = plotting_module.AnalysisStatistics(len(frame), 0, 0, None, None)
    with plotting_module.matplotlib.rc_context(plotting_module.PLOT_STYLE):
        figure = plotting_module._scatter_figure(frame, statistics)
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert not axis.lines
        if empty:
            assert not axis.collections
            assert axis.get_legend() is None
        else:
            assert len(axis.collections) == 1
            points = axis.collections[0]
            assert points.get_offsets().tolist() == [[2.0, 0.7]]
            assert points.get_facecolors()[0] == pytest.approx(
                to_rgba("#858B93", alpha=0.35)
            )
            assert [text.get_text() for text in axis.get_legend().get_texts()] == [
                plotting_module.category_legend_label("Other / unlabeled")
            ]
    finally:
        plotting_module.plt.close(figure)


@pytest.mark.parametrize("use_tex", (False, True))
def test_category_labels_use_typewriter_text_for_the_active_renderer(
    use_tex: bool,
) -> None:
    assert plotting_module._KIND_ORDER[:4] == ("MV", "RV", "TV", "N")
    with plotting_module.matplotlib.rc_context({"text.usetex": use_tex}):
        for category in ("MV", "RV", "TV", "N", "Other"):
            expected = (
                rf"\texttt{{{category}}}" if use_tex else rf"$\mathtt{{{category}}}$"
            )
            assert plotting_module.category_legend_label(category) == expected


@pytest.mark.parametrize("method", ("kmeans", "gmm"))
def test_proximity_kind_legend_order_and_typewriter_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    import numpy as np
    from scripts import check_proximity_clusters, check_proximity_gmm

    diagnostic = check_proximity_clusters if method == "kmeans" else check_proximity_gmm
    # Input order is deliberately different from the requested legend order.
    kinds = ("TV", "Other", "N", "MV", "RV")
    frame = pd.DataFrame(
        [
            {
                "original_index": str(prompt),
                "seed": seed,
                "l2_norm": float(prompt * 4 + seed),
                "sscd": 0.1 + prompt * 0.15,
                "kind": kind,
                "prompt_rule": "rho < 0",
                "kmeans_cluster": "low_sscd_mode",
                "gmm_component": "low_sscd_mode",
            }
            for prompt, kind in enumerate(kinds)
            for seed in range(2)
        ]
    )
    captured = []
    monkeypatch.setattr(
        diagnostic,
        "_publish_figures",
        lambda _output, figures: captured.extend(figures),
    )
    options = (
        {"cutoff": 0.4}
        if method == "kmeans"
        else {
            "component_means": np.asarray([[2.0, 0.2], [6.0, 0.8]]),
            "component_covariances": np.asarray([np.eye(2), np.eye(2)]) * 0.01,
        }
    )
    with plotting_module.matplotlib.rc_context({"text.usetex": False}):
        diagnostic.plot_assignments(frame, tmp_path / f"{method}.png", **options)
    assert len(captured) == 3
    legend = captured[2][0].axes[0].get_legend()
    assert [label.get_text() for label in legend.get_texts()] == [
        rf"$\mathtt{{{kind}}}$" for kind in ("MV", "RV", "TV", "N", "Other")
    ]
    assert all(label.get_fontsize() == 12 for label in legend.get_texts())
    assert all(handle.get_alpha() == 1.0 for handle in legend.legend_handles)
    # Reordering the legend must not swap the existing RV and TV encodings.
    assert [handle.get_linestyle() for handle in legend.legend_handles] == [
        "-",
        "-.",
        "--",
        ":",
        "--",
    ]
    for figure, _filenames in captured[:2]:
        assert all(
            "mathtt" not in label.get_text() and "texttt" not in label.get_text()
            for label in figure.axes[0].get_legend().get_texts()
        )


def test_gmm_diagnostic_writes_three_single_panel_prompt_line_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from matplotlib.collections import LineCollection, PathCollection
    from scripts import check_proximity_gmm as diagnostic

    frame = pd.DataFrame(
        [
            {
                "original_index": str(prompt),
                "seed": seed,
                "l2_norm": float(3 - seed + prompt * 4),
                "sscd": (0.1, 0.8, 0.2)[seed],
                "kind": ("MV", "N")[prompt],
                "gmm_component": (
                    "low_sscd_mode",
                    "high_sscd_mode",
                    "low_sscd_mode",
                )[seed],
                "prompt_rule": ("rho < 0", "rho >= 0")[prompt],
            }
            for prompt in range(2)
            for seed in range(3)
        ]
    )
    captured = []
    original_publish = diagnostic._publish_figures

    def publish(output: Path, figures: object) -> None:
        captured.extend(figures)
        original_publish(output, figures)

    monkeypatch.setattr(diagnostic, "_publish_figures", publish)
    paths = diagnostic.plot_assignments(
        frame,
        tmp_path / "gmm_k2.png",
        component_means=np.asarray([[2.0, 0.2], [6.0, 0.8]]),
        component_covariances=np.asarray(
            [
                [[0.2, 0.0], [0.0, 0.01]],
                [[0.3, 0.0], [0.0, 0.02]],
            ]
        ),
    )

    assert {path.name for path in paths} == {
        f"gmm_k2{suffix}.{extension}"
        for suffix in ("", "_spearman", "_kind")
        for extension in ("png", "pdf")
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert len(captured) == 3
    for view_index, (figure, _filenames) in enumerate(captured):
        np.testing.assert_allclose(figure.get_size_inches(), (4.0, 4.0))
        assert len(figure.axes) == 2
        axis, colorbar_axis = figure.axes
        assert not axis.get_title() and figure._suptitle is None
        assert not any(isinstance(item, PathCollection) for item in axis.collections)
        curves = [item for item in axis.collections if isinstance(item, LineCollection)]
        assert len(curves) == 1
        curve = curves[0]
        assert len(curve.get_segments()) == 4
        assert curve.get_segments()[0].tolist() == [[1.0, 0.2], [2.0, 0.8]]
        assert curve.get_segments()[1].tolist() == [[2.0, 0.8], [3.0, 0.1]]
        assert curve.get_array().tolist() == pytest.approx([0.5, 0.45, 0.5, 0.45])
        assert len(curve.get_linestyles()) == 4
        styles = curve.get_linestyles()
        if view_index == 0:
            # Adjacent seeds have different component labels.
            assert styles[0][1] is not None
            assert styles[1][1] is None
        else:
            # Spearman sign and kind are constant within each prompt.
            assert all(styles[index][1] is None for index in (0, 1))
            assert all(styles[index][1] is not None for index in (2, 3))
        assert curve.cmap.name == "viridis"
        assert (curve.norm.vmin, curve.norm.vmax) == (0.0, 1.0)
        assert colorbar_axis.get_ylabel() == "SSCD"
        assert colorbar_axis.collections[-1].get_alpha() == 1.0
        assert axis.xaxis.label.get_fontsize() == 18
        assert all(
            handle.get_alpha() == 1.0 for handle in axis.get_legend().legend_handles
        )
        assert len(axis.patches) == (4 if view_index == 0 else 0)


def _example_group_rows(
    index: str,
    distances: list[float],
    *,
    include: bool,
    source_row_number: int,
    selection_status: str | None = None,
) -> list[dict[str, object]]:
    status = selection_status or (
        "included_proximity_rule" if include else "discarded_proximity_rule"
    )
    return [
        {
            "original_index": index,
            "record_id": f"sdv1-{index}",
            "source_row_number": source_row_number,
            "prompt": f"prompt {index}",
            "seed": seed,
            "l2_norm": distance,
            "sscd": (seed - 10) / 20.0,
            "observation_status": "complete",
            "include_prompt": include,
            "selection_status": status,
        }
        for seed, distance in enumerate(distances)
    ]


def test_example_prompts_rank_prompt_means_and_preserve_all_twenty_seeds() -> None:
    rows = [
        *_example_group_rows(
            "kept-high", [9.0] * 20, include=True, source_row_number=3
        ),
        # Its individual maximum is larger, but its prompt mean is only 5.
        *_example_group_rows(
            "kept-spiky",
            [100.0, *([0.0] * 19)],
            include=True,
            source_row_number=2,
        ),
        *_example_group_rows("kept-low", [2.0] * 20, include=True, source_row_number=1),
        *_example_group_rows(
            "discarded-high", [8.0] * 20, include=False, source_row_number=6
        ),
        *_example_group_rows(
            "discarded-middle", [4.0] * 20, include=False, source_row_number=5
        ),
        *_example_group_rows(
            "discarded-low", [1.0] * 20, include=False, source_row_number=4
        ),
        # Incomplete and unusable groups cannot become examples.
        *_example_group_rows(
            "incomplete", [20.0] * 19, include=True, source_row_number=7
        ),
        *_example_group_rows(
            "unusable",
            [30.0] * 20,
            include=False,
            source_row_number=8,
            selection_status="unusable_reference_observations",
        ),
    ]
    failed = _example_group_rows(
        "failed", [40.0] * 20, include=True, source_row_number=9
    )
    failed[-1]["observation_status"] = "cache_error"
    analysis = pd.DataFrame([*rows, *failed])
    ranked = proximity_module._example_prompts(analysis, num_seeds=20)
    shuffled = proximity_module._example_prompts(
        analysis.sample(frac=1.0, random_state=17), num_seeds=20
    )
    assert shuffled == ranked

    assert [
        (example["group"], example["rank"], example["original_index"])
        for example in ranked
    ] == [
        ("retained", "highest", "kept-high"),
        ("retained", "median", "kept-spiky"),
        ("retained", "lowest", "kept-low"),
        ("discarded", "highest", "discarded-high"),
        ("discarded", "median", "discarded-middle"),
        ("discarded", "lowest", "discarded-low"),
    ]
    assert ranked[1]["mean_l2_norm"] == pytest.approx(5.0)
    assert ranked[1]["l2_norms"] == [100.0, *([0.0] * 19)]
    assert all(example["mean_sscd"] == pytest.approx(-0.025) for example in ranked)
    assert all(example["seeds"] == list(range(20)) for example in ranked)
    assert len({example["original_index"] for example in ranked}) == len(ranked)


def test_example_prompts_reject_nonfinite_sscd_in_an_eligible_group() -> None:
    rows = _example_group_rows(
        "invalid-sscd", [1.0] * 20, include=True, source_row_number=1
    )
    rows[7]["sscd"] = float("nan")

    with pytest.raises(
        ProximityError, match="example SSCD scores are invalid: invalid-sscd"
    ):
        proximity_module._example_prompts(pd.DataFrame(rows), num_seeds=20)


def test_example_prompts_use_deterministic_lower_median_and_no_duplicates() -> None:
    four_kept = pd.DataFrame(
        [
            *_example_group_rows("low", [1.0] * 3, include=True, source_row_number=9),
            *_example_group_rows("tie-b", [2.0] * 3, include=True, source_row_number=8),
            *_example_group_rows("tie-a", [2.0] * 3, include=True, source_row_number=7),
            *_example_group_rows("high", [4.0] * 3, include=True, source_row_number=6),
        ]
    )
    ranked = proximity_module._example_prompts(four_kept, num_seeds=3)
    assert [(item["rank"], item["original_index"]) for item in ranked] == [
        ("highest", "high"),
        ("median", "tie-a"),
        ("lowest", "low"),
    ]

    two_discarded = pd.DataFrame(
        [
            *_example_group_rows(
                "first", [1.0] * 3, include=False, source_row_number=1
            ),
            *_example_group_rows(
                "second", [2.0] * 3, include=False, source_row_number=2
            ),
        ]
    )
    ranked = proximity_module._example_prompts(two_discarded, num_seeds=3)
    assert [(item["rank"], item["original_index"]) for item in ranked] == [
        ("highest", "second"),
        ("lowest", "first"),
    ]

    singleton = pd.DataFrame(
        _example_group_rows("only", [1.0] * 3, include=False, source_row_number=1)
    )
    ranked = proximity_module._example_prompts(singleton, num_seeds=3)
    assert [(item["rank"], item["original_index"]) for item in ranked] == [
        ("median", "only")
    ]


def _example_png_bytes(
    size: tuple[int, int], *, color: tuple[int, int, int] = (32, 64, 128)
) -> bytes:
    with Image.new("RGB", size, color=color) as image:
        image.paste((240, 96, 16), (0, 0, max(1, size[0] // 2), size[1]))
        image.putpixel((size[0] - 1, size[1] - 1), (17, 91, 205))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


@pytest.mark.parametrize(
    "original_size,scale,max_edge,expected_size",
    [
        ((800, 640), 0.75, None, (600, 480)),
        ((1201, 803), 0.75, None, (901, 602)),
        ((1024, 768), 1.0, 256, (256, 192)),
        ((768, 1024), 1.0, 256, (192, 256)),
        ((201, 149), 1.0, 256, (201, 149)),
        ((256, 256), 1.0, 256, (256, 256)),
        ((301, 199), 1.0, 256, (256, 169)),
        ((1, 1), 0.75, None, (1, 1)),
        ((1, 1000), 1.0, 256, (1, 256)),
        ((800, 400), 0.75, 256, (256, 128)),
        ((2, 3), 0.75, None, (2, 2)),
    ],
)
def test_example_png_scales_dimensions_and_preserves_pixels_with_lanczos(
    original_size: tuple[int, int],
    scale: float,
    max_edge: int | None,
    expected_size: tuple[int, int],
) -> None:
    content = _example_png_bytes(original_size)
    exported, observed_original, observed_exported = proximity_module._example_png(
        content, scale=scale, max_edge=max_edge
    )
    assert observed_original == original_size
    assert observed_exported == expected_size
    assert min(observed_exported) >= 1
    assert observed_exported[0] <= original_size[0]
    assert observed_exported[1] <= original_size[1]
    if max_edge is not None:
        assert max(observed_exported) <= max_edge
    with Image.open(io.BytesIO(content)) as source:
        expected = source.resize(expected_size, resample=Image.Resampling.LANCZOS)
        expected_png = io.BytesIO()
        expected.save(expected_png, format="PNG", optimize=True)
    assert exported == expected_png.getvalue()
    with Image.open(io.BytesIO(exported)) as image:
        assert image.format == "PNG"
        assert image.size == expected_size
        assert image.mode == "RGB"


@pytest.mark.parametrize("content", [b"", b"not an image", b"\x89PNG\r\n\x1a\ninvalid"])
def test_example_png_rejects_undecodable_sources(content: bytes) -> None:
    with pytest.raises(ProximityError):
        proximity_module._example_png(content, scale=0.75)


def _example_export_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    ProximityPaths,
    Path,
    dict[str, tuple[Path, bytes, Path, bytes]],
    list[tuple[str, dict[str, object]]],
]:
    root = tmp_path.resolve()
    generation_run = root / "logs" / "synthetic" / "experiment_S0_N20"
    cache = GenerationPaths(generation_run)
    cache.create()
    output = root / "outputs" / "synthetic" / "proximity" / "experiment_S0_N20"
    paths = ProximityPaths(root, generation_run, output)
    target_directory = root / "data" / "webster" / "images"
    target_directory.mkdir(parents=True)
    candidates = (
        ("kept-high", True, 9.0),
        ("kept-middle", True, 5.0),
        ("kept-low", True, 1.0),
        ("discarded-high", False, 8.0),
        ("discarded-middle", False, 4.0),
        ("discarded-low", False, 2.0),
    )
    rows: list[dict[str, object]] = []
    sources: dict[str, tuple[Path, bytes, Path, bytes]] = {}
    markers: dict[str, dict[str, object]] = {}
    for source_row_number, (index, included, distance) in enumerate(candidates):
        generated = cache.image_path(index)
        generated_bytes = _example_png_bytes(
            (800, 640), color=(source_row_number, 64, 128)
        )
        generated.write_bytes(generated_bytes)
        target = target_directory / f"{index}.png"
        target_bytes = _example_png_bytes(
            (1024, 768), color=(128, source_row_number, 64)
        )
        target.write_bytes(target_bytes)
        target_hash = file_sha256(target)
        prompt_rows = _example_group_rows(
            index,
            [distance] * 20,
            include=included,
            source_row_number=source_row_number,
        )
        for tile_index, row in enumerate(prompt_rows):
            row.update(
                {
                    "generated_image_path": generated.relative_to(root).as_posix(),
                    "generated_image_tile_index": tile_index,
                    "target_image_sha256": target_hash,
                }
            )
        rows.extend(prompt_rows)
        sources[index] = (generated, generated_bytes, target, target_bytes)
        markers[index] = {
            "num_seeds": 20,
            "seeds": list(range(20)),
            "target_image_path": str(target),
            "target_image_sha256": target_hash,
            "preview_image_sha256": file_sha256(generated),
        }
    table_path = output / "proximity.csv"
    table_path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(table_path, index=False)
    monkeypatch.setattr(
        proximity_module,
        "require_generation_run",
        lambda observed: {
            "scientific_config_hash": "a" * 64,
            "scientific_config": {
                "num_seeds": 20,
                "seeds": list(range(20)),
            },
        },
    )
    validation_calls: list[tuple[str, dict[str, object]]] = []

    def validate_record(
        observed: GenerationPaths, index: object, **options: object
    ) -> SimpleNamespace:
        assert observed.run_directory == generation_run
        normalized = str(index)
        validation_calls.append((normalized, options))
        return SimpleNamespace(valid=True, errors=(), metadata=markers[normalized])

    monkeypatch.setattr(proximity_module, "validate_generation_record", validate_record)
    return paths, table_path, sources, validation_calls


def test_write_examples_downscales_ranked_pairs_and_preserves_source_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, table_path, sources, validation_calls = _example_export_fixture(
        tmp_path, monkeypatch
    )
    notes = paths.output_directory / "examples/retained/highest_l2_training.notes.txt"
    notes.parent.mkdir(parents=True)
    notes.write_text("unrelated user note", encoding="utf-8")
    old_png = notes.with_name("highest_l2_training.png")
    old_png.write_bytes(b"preserved historical example PNG")
    old_png_before = (old_png.read_bytes(), old_png.stat().st_mtime_ns)
    table_before = (table_path.read_bytes(), table_path.stat().st_mtime_ns)
    monkeypatch.setattr(
        proximity_module,
        "safe_torch_load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("example export must not load latent tensors")
        ),
    )

    assert (
        proximity_module._write_examples(paths, table_path=table_path, num_seeds=20)
        is None
    )

    manifest = json.loads(
        (paths.output_directory / "examples/manifest.json").read_text(encoding="utf-8")
    )
    assert {
        key: manifest[key]
        for key in ("ranking", "median_rule", "num_seeds", "source_csv")
    } == {
        "ranking": "mean_terminal_l2_across_seeds",
        "median_rule": "lower_middle_prompt",
        "num_seeds": 20,
        "source_csv": table_path.relative_to(paths.project_root).as_posix(),
    }
    assert [
        (entry["group"], entry["rank"], entry["original_index"])
        for entry in manifest["examples"]
    ] == [
        ("retained", "highest", "kept-high"),
        ("retained", "median", "kept-middle"),
        ("retained", "lowest", "kept-low"),
        ("discarded", "highest", "discarded-high"),
        ("discarded", "median", "discarded-middle"),
        ("discarded", "lowest", "discarded-low"),
    ]
    gallery = paths.figure_directory / "examples"
    assert manifest["format"] == "pdf"
    assert manifest["figure_directory"] == gallery.relative_to(paths.project_root).as_posix()
    assert manifest["image_export"] == {
        "format": "pdf",
        "pdf_dpi": 150,
        "pad_inches": .05,
        "generated_scale": 0.75,
        "training_max_edge": 256,
        "resampling": "lanczos",
        "png_optimize": True,
    }
    cached = pd.read_csv(table_path)
    for entry in manifest["examples"]:
        index = entry["original_index"]
        generated, generated_bytes, target, target_bytes = sources[index]
        assert entry["seeds"] == list(range(20))
        cached_prompt = cached.loc[cached["original_index"].eq(index)].sort_values(
            "seed"
        )
        assert entry["l2_norms"] == cached_prompt["l2_norm"].astype(float).tolist()
        assert entry["mean_l2_norm"] == pytest.approx(cached_prompt["l2_norm"].mean())
        assert entry["mean_sscd"] == pytest.approx(cached_prompt["sscd"].mean())
        assert (
            entry["generated_source_path"]
            == generated.relative_to(paths.project_root).as_posix()
        )
        assert (
            entry["training_source_path"]
            == target.relative_to(paths.project_root).as_posix()
        )
        generated_output = (
            gallery / entry["generated_image_path"]
        )
        training_output = (
            gallery / entry["training_image_path"]
        )
        assert generated.read_bytes() == generated_bytes
        assert target.read_bytes() == target_bytes
        assert generated_output.read_bytes() != generated_bytes
        assert training_output.read_bytes() != target_bytes
        assert entry["generated_image_sha256"] == file_sha256(generated_output)
        assert entry["training_image_sha256"] == file_sha256(training_output)
        assert entry["generated_source_sha256"] == file_sha256(generated)
        assert entry["training_source_sha256"] == file_sha256(target)
        assert entry["target_image_sha256"] == file_sha256(target)
        assert entry["generated_source_size"] == [800, 640]
        assert entry["generated_image_size"] == [600, 480]
        assert entry["training_source_size"] == [1024, 768]
        assert entry["training_image_size"] == [256, 192]
        assert generated_output.suffix == training_output.suffix == ".pdf"
        assert generated_output.read_bytes().startswith(b"%PDF")
        assert training_output.read_bytes().startswith(b"%PDF")
        assert Path(entry["generated_image_path"]).parent.name == entry["group"]
        assert Path(entry["training_image_path"]).parent.name == entry["group"]
    assert notes.read_text(encoding="utf-8") == "unrelated user note"
    assert (old_png.read_bytes(), old_png.stat().st_mtime_ns) == old_png_before
    assert (table_path.read_bytes(), table_path.stat().st_mtime_ns) == table_before
    assert (gallery / "retained").is_dir() and (gallery / "discarded").is_dir()
    assert not (gallery / "kept").exists()
    assert len(list(gallery.rglob("*.pdf"))) == 12
    assert not list(gallery.rglob("*.png"))
    assert {item.suffix for item in gallery.rglob("*") if item.is_file()} == {".pdf"}
    assert not list((paths.output_directory / "examples").rglob("*.pdf"))
    assert len(validation_calls) == 6
    assert all(
        options["load_tensors"] is False
        and options["tensor_names"] == ()
        and options["require_preview"] is True
        and options["verify_file_hashes"] is True
        for _index, options in validation_calls
    )


def test_write_examples_keeps_small_training_dimensions_and_preserves_cached_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, table_path, sources, _validation_calls = _example_export_fixture(
        tmp_path, monkeypatch
    )
    target = sources["kept-high"][2]
    original = _example_png_bytes((1, 1))
    target.write_bytes(original)
    reencoded, original_size, exported_size = proximity_module._example_png(
        original, scale=1.0, max_edge=256
    )
    assert original_size == exported_size == (1, 1)
    assert len(reencoded) >= len(original)
    assert reencoded != original
    source_hash = file_sha256(target)

    validation = proximity_module.validate_generation_record(
        GenerationPaths(paths.generation_run), "kept-high"
    )
    validation.metadata["target_image_sha256"] = source_hash
    table = pd.read_csv(table_path, dtype={"original_index": str})
    table.loc[table["original_index"].eq("kept-high"), "target_image_sha256"] = (
        source_hash
    )
    table.to_csv(table_path, index=False)
    proximity_module._write_examples(paths, table_path=table_path, num_seeds=20)

    gallery = paths.output_directory / "examples"
    manifest = json.loads((gallery / "manifest.json").read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["examples"] if item["original_index"] == "kept-high"
    )
    output = paths.figure_directory / "examples" / entry["training_image_path"]
    assert output.suffix == ".pdf" and output.read_bytes().startswith(b"%PDF")
    assert target.read_bytes() == original
    assert entry["training_source_size"] == entry["training_image_size"] == [1, 1]
    assert entry["training_image_sha256"] == file_sha256(output)
    assert entry["training_source_sha256"] == source_hash
    assert entry["target_image_sha256"] == source_hash


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("target", "cached paired training image file is invalid"),
        ("montage", "preview image SHA-256 differs"),
    ),
)
def test_write_examples_rejects_corrupt_pairs_before_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    message: str,
) -> None:
    paths, table_path, sources, _validation_calls = _example_export_fixture(
        tmp_path, monkeypatch
    )
    if corruption == "target":
        sources["kept-high"][2].write_bytes(b"corrupt target")
    else:
        monkeypatch.setattr(
            proximity_module,
            "validate_generation_record",
            lambda *_args, **_kwargs: SimpleNamespace(
                valid=False,
                errors=("preview image SHA-256 differs",),
                metadata=None,
            ),
        )

    with pytest.raises(ProximityError, match=message):
        proximity_module._write_examples(paths, table_path=table_path, num_seeds=20)
    assert not (paths.output_directory / "examples/manifest.json").exists()
    assert not list((paths.output_directory / "examples").glob("*/*.png"))
    assert not list((paths.figure_directory / "examples").rglob("*.pdf"))


def test_write_examples_preserves_existing_gallery_when_late_source_cannot_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, table_path, sources, _validation_calls = _example_export_fixture(
        tmp_path, monkeypatch
    )
    proximity_module._write_examples(paths, table_path=table_path, num_seeds=20)
    gallery = paths.output_directory / "examples"
    figure_gallery = paths.figure_directory / "examples"
    original_outputs = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for directory in (gallery, figure_gallery)
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert gallery / "manifest.json" in original_outputs
    assert sum(path.suffix == ".pdf" for path in original_outputs) == 12

    corrupted_source = sources["discarded-low"][0]
    corrupted_source.write_bytes(b"checksum-valid but undecodable montage")
    validation = proximity_module.validate_generation_record(
        GenerationPaths(paths.generation_run), "discarded-low"
    )
    validation.metadata["preview_image_sha256"] = file_sha256(corrupted_source)
    with pytest.raises(ProximityError):
        proximity_module._write_examples(paths, table_path=table_path, num_seeds=20)

    after_outputs = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for directory in (gallery, figure_gallery)
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert after_outputs == original_outputs


def test_external_pdf_publication_preserves_cached_tables_and_historical_images(tmp_path, monkeypatch):
    paths = ProximityPaths.build(
        tmp_path, Path(RUN_NAME) / "experiment_S0_N20",
        role="experiment", seed_start=0, num_seeds=20)
    analysis = _annotate_selection(
        pd.concat([_observations("1"), _observations("2")], ignore_index=True),
        _selection(("1", "2")))
    paths.output_directory.mkdir(parents=True)
    historical = paths.output_directory / "proximity_vs_sscd.png"
    historical.write_bytes(b"preserved historical image")
    captured = []
    def publish(output, figures, *, formats, **options):
        assert formats == ("pdf",)
        assert Path(output) == paths.figure_directory
        Path(output).mkdir(parents=True, exist_ok=True)
        for _figure, names in figures:
            assert set(names) == {"pdf"}
            (Path(output) / names["pdf"]).write_bytes(b"saved PDF fixture")
            captured.append(names["pdf"])
    monkeypatch.setattr(plotting_module, "_publish_figures", publish)
    expected = write_analysis_outputs(
        paths.output_directory, analysis=analysis, figure_directory=paths.figure_directory)
    assert paths.figure_directory == tmp_path / "figures" / RUN_NAME / "proximity/experiment_S0_N20"
    assert sorted(captured) == sorted(plotting_module.PROXIMITY_PDF_FILENAMES)
    assert {path.name for path in paths.output_directory.iterdir()} == {"proximity.csv", historical.name}
    before = {path.name: (file_sha256(path), path.stat().st_mtime_ns)
              for path in paths.output_directory.iterdir()}
    monkeypatch.setattr(plotting_module, "atomic_write_frame_csv",
                        lambda *args, **kwargs: pytest.fail("plot-only rewrote cached scalar CSV"))
    assert plotting_module.write_saved_analysis_figures(
        paths.output_directory, figure_directory=paths.figure_directory) == expected
    assert {path.name: (file_sha256(path), path.stat().st_mtime_ns)
            for path in paths.output_directory.iterdir()} == before
    assert {path.suffix for path in paths.figure_directory.iterdir()} == {".pdf"}


def test_external_pdf_failure_keeps_previous_publication_and_cache(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    paths = ProximityPaths.build(
        tmp_path, Path(RUN_NAME) / "experiment_S0_N20",
        role="experiment", seed_start=0, num_seeds=20)
    paths.figure_directory.mkdir(parents=True)
    old = {name: ("old " + name).encode() for name in plotting_module.PROXIMITY_PDF_FILENAMES}
    for name, content in old.items():
        (paths.figure_directory / name).write_bytes(content)
    # Fail the second PDF after one has staged; neither old PDF may be replaced.
    calls = []
    def fail_second(self, destination, **options):
        assert options["format"] == "pdf"
        calls.append(Path(destination))
        Path(destination).write_bytes(b"new staged PDF")
        if len(calls) == 2:
            raise RuntimeError("injected second PDF failure")
    monkeypatch.setattr(Figure, "savefig", fail_second)
    frame = pd.DataFrame({"original_index": ["p", "p"], "seed": [0, 1],
                          "kind": ["TV", "TV"], "l2_norm": [1., 2.], "sscd": [.8, -.1],
                          "prompt_spearman": [-1., -1.]})
    before = frame.copy(deep=True)
    with pytest.raises(RuntimeError, match="second PDF"):
        plotting_module._write_scatter_views(
            paths.figure_directory, all_prompts=frame, selected=frame,
            spearman_column="prompt_spearman")
    assert {path.name: path.read_bytes() for path in paths.figure_directory.iterdir()} == old
    assert not paths.output_directory.exists()
    pd.testing.assert_frame_equal(frame, before)
    assert not plotting_module.plt.get_fignums()


def _example_publication_snapshot(paths: ProximityPaths) -> dict[Path, tuple[bytes, int]]:
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for directory in (paths.output_directory / "examples", paths.figure_directory / "examples")
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_example_plot_preflight_decodes_sources_without_encoding_or_writing(tmp_path, monkeypatch):
    paths, table_path, sources, validation_calls = _example_export_fixture(tmp_path, monkeypatch)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in tmp_path.rglob("*") if path.is_file()}

    def forbidden(*_args, **_kwargs):
        pytest.fail("example preflight must not encode, publish, or load latent tensors")

    for name in ("_example_png", "_example_pdf", "_publish_example_payloads",
                 "atomic_write_bytes", "atomic_write_json", "safe_torch_load"):
        monkeypatch.setattr(proximity_module, name, forbidden)

    assert proximity_module._write_examples(
        paths, table_path=table_path, num_seeds=20, validate_only=True,
    ) is None

    assert len(validation_calls) == len(sources) == 6
    assert all(options["load_tensors"] is False and options["tensor_names"] == ()
               and options["require_preview"] is True and options["verify_file_hashes"] is True
               for _index, options in validation_calls)
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not paths.figure_directory.exists()
    assert not (paths.output_directory / "examples").exists()


def test_example_late_pdf_encoding_failure_preserves_manifest_and_all_old_pdfs(tmp_path, monkeypatch):
    paths, table_path, sources, _calls = _example_export_fixture(tmp_path, monkeypatch)
    proximity_module._write_examples(paths, table_path=table_path, num_seeds=20)
    before = _example_publication_snapshot(paths)
    source_before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                     for generated, _generated_bytes, target, _target_bytes in sources.values()
                     for path in (generated, target)}
    encode = proximity_module._example_pdf
    encoded = []

    def fail_last_image(content):
        encoded.append(len(content))
        if len(encoded) == 12:
            raise ProximityError("injected final example PDF encoding failure")
        return encode(content)

    monkeypatch.setattr(proximity_module, "_example_pdf", fail_last_image)
    with pytest.raises(ProximityError, match="final example PDF encoding failure"):
        proximity_module._write_examples(paths, table_path=table_path, num_seeds=20)

    assert len(encoded) == 12
    assert _example_publication_snapshot(paths) == before
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in source_before} == source_before
    assert not list((paths.figure_directory / "examples").rglob("*.png"))


@pytest.mark.parametrize("failed_artifact", ["last_pdf", "manifest"])
def test_example_late_install_failure_restores_old_pdfs_and_manifest(tmp_path, monkeypatch, failed_artifact):
    paths, table_path, sources, _calls = _example_export_fixture(tmp_path, monkeypatch)
    proximity_module._write_examples(paths, table_path=table_path, num_seeds=20)
    old_png = paths.output_directory / "examples/retained/highest_l2_generated.png"
    old_png.parent.mkdir(parents=True, exist_ok=True)
    old_png.write_bytes(b"preserved historical gallery PNG")
    before = _example_publication_snapshot(paths)
    source_before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                     for generated, _generated_bytes, target, _target_bytes in sources.values()
                     for path in (generated, target)}
    table_before = (table_path.read_bytes(), table_path.stat().st_mtime_ns)
    encode = proximity_module._example_pdf
    # Valid trailing PDF comments distinguish replacement payloads from the old
    # publication, so rollback is tested even with deterministic image exports.
    monkeypatch.setattr(proximity_module, "_example_pdf",
                        lambda content: encode(content) + b"\n% replacement fixture\n")
    target = (paths.output_directory / "examples/manifest.json" if failed_artifact == "manifest"
              else paths.figure_directory / "examples/discarded/lowest_l2_training.pdf")
    replace = proximity_module.os.replace
    failures = []

    def fail_once(source, destination):
        if Path(destination) == target and not failures:
            failures.append((Path(source), Path(destination)))
            raise OSError("injected late example installation failure")
        return replace(source, destination)

    monkeypatch.setattr(proximity_module.os, "replace", fail_once)
    with pytest.raises((OSError, ProximityError), match="late example installation failure"):
        proximity_module._write_examples(paths, table_path=table_path, num_seeds=20)

    assert len(failures) == 1
    assert _example_publication_snapshot(paths) == before
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in source_before} == source_before
    assert (table_path.read_bytes(), table_path.stat().st_mtime_ns) == table_before
    assert not list((paths.figure_directory / "examples").rglob("*.png"))
