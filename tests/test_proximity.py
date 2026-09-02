"""Offline proximity, plotting-statistics, and supplied-cache parity tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from utils.experiments import proximity as proximity_module
from utils.experiments.plotting import (
    _view_progress,
    compute_correlations,
    write_analysis_outputs,
)
from utils.experiments.proximity import (
    ProximityError,
    ProximityPaths,
    _paired_rows,
    _record_progress,
    _seed_role,
    _selected_frame,
)
from utils.models.latent import compute_latent_distances


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "sdv1_ddim_g7.5_T50_N20"


def test_proximity_progress_remains_visible_when_stderr_is_captured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert list(_record_progress(["one", "two"])) == ["one", "two"]
    views = {
        "all": pd.DataFrame(),
        "selected": pd.DataFrame(),
    }
    assert [name for name, _ in _view_progress(views)] == list(views)

    output = capsys.readouterr().err
    assert "[Proximity 1/2] Records" in output
    assert "[Proximity 2/2] Analysis views" in output
    assert output.count("2/2") >= 2


def test_canonical_latent_distance_handles_terminal_endpoints() -> None:
    trajectory = torch.full((2, 3, 1, 2, 2), 100.0, dtype=torch.float32)
    trajectory[0, -1] = 1.0
    trajectory[1, -1] = 2.0
    target = torch.zeros((1, 2, 2), dtype=torch.float32)
    distances = compute_latent_distances(trajectory[:, -1], target)
    torch.testing.assert_close(distances.l2_norms, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(distances.latent_rmse, torch.tensor([1.0, 2.0]))


def test_scientific_output_namespaces_share_the_experiment_parent(tmp_path: Path) -> None:
    proximity_directory = tmp_path.resolve() / "outputs" / RUN_NAME / "proximity"
    proximity_directory.mkdir(parents=True)
    legacy_config = proximity_directory / "run_config.json"
    legacy_config.write_text("legacy-schema-one", encoding="utf-8")

    experiment_run = Path(RUN_NAME) / "experiment_S0_N20"
    paths = ProximityPaths.build(tmp_path.resolve(), experiment_run)
    reference = ProximityPaths.build(
        tmp_path.resolve(),
        Path(RUN_NAME) / "reference_S20_N20",
        output_run_name=RUN_NAME,
        role="reference",
        seed_start=20,
        num_seeds=20,
    )

    assert paths.generation_run == tmp_path.resolve() / "logs" / experiment_run
    assert reference.generation_run == (
        tmp_path.resolve() / "logs" / RUN_NAME / "reference_S20_N20"
    )
    assert paths.output_directory == proximity_directory / "experiment_S0_N20"
    assert reference.output_directory == proximity_directory / "reference_S20_N20"
    assert paths.result_path("42") == paths.output_directory / "records/42.pt"
    assert legacy_config.read_text(encoding="utf-8") == "legacy-schema-one"
    assert paths.run_config_json != legacy_config


def test_analysis_configuration_refreshes_audit_provenance_only_after_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run_config.json"
    existing = {
        "analysis": "proximity",
        "selection_hash": "a" * 64,
        "source_provenance": {
            "utils/data/selection.py": "b" * 64,
            "utils/experiments/proximity.py": "c" * 64,
        },
    }
    desired = {
        **existing,
        "source_provenance": {
            **existing["source_provenance"],
            "utils/data/selection.py": "d" * 64,
        },
    }
    path.write_text(json.dumps(existing), encoding="utf-8")

    proximity_module._write_or_validate_configuration(path, desired)
    assert json.loads(path.read_text(encoding="utf-8")) == existing
    proximity_module._write_or_validate_configuration(
        path,
        desired,
        refresh_source_provenance=True,
    )
    assert json.loads(path.read_text(encoding="utf-8")) == desired

    changed_core_source = {
        **desired,
        "source_provenance": {
            **desired["source_provenance"],
            "utils/experiments/proximity.py": "e" * 64,
        },
    }
    proximity_module._write_or_validate_configuration(path, changed_core_source)
    assert json.loads(path.read_text(encoding="utf-8")) == desired

    changed_science = {**desired, "selection_hash": "f" * 64}
    with pytest.raises(ProximityError, match="incompatible"):
        proximity_module._write_or_validate_configuration(path, changed_science)


def test_correlation_statistics_are_explicit_and_dimension_normalized() -> None:
    frame = pd.DataFrame(
        {
            "original_index": ["a", "a", "b"],
            "latent_rmse": [1.0, 2.0, 3.0],
            "sscd_cosine_similarity": [4.0, 2.0, 0.0],
        }
    )
    statistics = compute_correlations(frame)
    assert (statistics.prompts, statistics.points) == (2, 3)
    assert statistics.pearson == pytest.approx(-1.0)
    assert statistics.spearman == pytest.approx(-1.0)


def test_proximity_has_no_model_loader_sampler_or_inference_dependency() -> None:
    path = ROOT / "utils/experiments/proximity.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("diffusers", "transformers", "utils.models.loading", "utils.models.sampling")
    assert not {name for name in imported if name.startswith(forbidden)}
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called_names.intersection(
        {"load_model_components", "load_vae_from_generation_config", "sample_trajectory"}
    )
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "compute_terminal_proximity" not in function_names
    assert "compute_latent_distances" in called_names


def _legacy_authoritative_all_table() -> Path:
    return (
        ROOT
        / "outputs"
        / RUN_NAME
        / "definition1_target_proximity"
        / "from_generation_cache_v3_e56f6c1b4dea003c"
        / "paired_proximity_sscd_all.parquet"
    )


def _final_all_table() -> Path:
    return ROOT / "outputs" / RUN_NAME / "proximity" / "experiment_S0_N20" / "paired_all.parquet"


def test_final_proximity_values_match_authoritative_cache_row_for_row() -> None:
    legacy_path = _legacy_authoritative_all_table()
    final_path = _final_all_table()
    if not legacy_path.is_file() or not final_path.is_file():
        pytest.skip("both the supplied cache and rebuilt final proximity table are required")

    comparison_columns = [
        "original_index",
        "record_id",
        "source_row_number",
        "seed",
        "prompt_raw",
        "latent_l2",
        "latent_rmse",
        "sscd_cosine_similarity",
    ]

    def normalized(path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path, columns=comparison_columns)
        frame["original_index"] = frame["original_index"].astype(str)
        return frame.sort_values(
            ["original_index", "seed"], kind="stable"
        ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        normalized(final_path),
        normalized(legacy_path),
        check_exact=True,
        check_dtype=False,
    )


def _role(**overrides: object) -> str:
    configuration: dict[str, object] = {
        "model_name": "sdv1",
        "scheduler_name": "ddim",
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "num_seeds": 20,
        "seed_start": 0,
    }
    configuration.update(overrides)
    return _seed_role(**configuration)


def test_seed_roles_accept_only_disjoint_experiment_or_exact_reference() -> None:
    assert _role() == "experiment"
    assert (
        _role(scheduler_name="ddpm", guidance_scale=3.0, num_seeds=4)
        == "experiment"
    )
    assert _role(seed_start=20) == "reference"


@pytest.mark.parametrize(
    "overrides",
    (
        {"seed_start": 1, "num_seeds": 19},
        {"seed_start": 0, "num_seeds": 21},
        {"seed_start": 10, "num_seeds": 20},
        {"seed_start": 20, "num_seeds": 19},
        {"seed_start": 20, "scheduler_name": "ddpm"},
    ),
)
def test_seed_roles_reject_partial_overlap_and_noncanonical_reference(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ProximityError, match="partial overlap"):
        _role(**overrides)


def test_missing_prerequisites_report_seed_aware_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ProximityError) as missing_generation:
        proximity_module.run_proximity(tmp_path, seed_start=20)
    generation_message = str(missing_generation.value)
    assert "./generate.sh --model sdv1 --scheduler ddim" in generation_message
    assert "--T 50 --N 20 --seed-start 20 --downscale 4" in generation_message

    run_directory = (
        tmp_path
        / "logs/sdv1_ddim_g7.5_T50_N20/reference_S20_N20"
    )
    run_directory.mkdir(parents=True)
    (run_directory / "run_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(proximity_module, "require_generation_run", lambda _: {})
    monkeypatch.setattr(
        proximity_module,
        "_validate_generation_invocation",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ProximityError) as missing_sscd:
        proximity_module.run_proximity(tmp_path, seed_start=20)
    sscd_message = str(missing_sscd.value)
    assert "./sscd.sh --model sdv1 --scheduler ddim" in sscd_message
    assert "--T 50 --N 20 --seed-start 20" in sscd_message


def test_paired_rows_preserve_explicit_seed_position_mapping(tmp_path: Path) -> None:
    record = SimpleNamespace(
        original_index="42",
        source_row_number=7,
        metadata={
            "record_id": "sdv1-0042",
            "prompt_raw": "prompt",
            "webster_overfit_type": "TV",
            "target_image_sha256": "a" * 64,
            "seeds": [20, 23, 39],
        },
    )
    rows = _paired_rows(
        record,
        {
            "latent_l2": torch.tensor([1.0, 2.0, 3.0]),
            "latent_rmse": torch.tensor([0.1, 0.2, 0.3]),
        },
        torch.tensor([0.4, 0.5, 0.6]),
    )

    assert [row["seed"] for row in rows] == [20, 23, 39]
    assert [row["latent_l2"] for row in rows] == pytest.approx([1.0, 2.0, 3.0])
    assert [row["sscd_cosine_similarity"] for row in rows] == pytest.approx(
        [0.4, 0.5, 0.6]
    )


def test_selected_table_keeps_every_experiment_seed_for_included_prompts() -> None:
    paired = pd.DataFrame(
        [
            {"original_index": index, "seed": seed}
            for index in ("included-a", "excluded", "included-b")
            for seed in range(20)
        ]
    )
    selection = SimpleNamespace(
        frame=pd.DataFrame(
            {"original_index": ["included-a", "excluded", "included-b"]}
        ),
        included_indices=frozenset({"included-a", "included-b"}),
    )

    selected = _selected_frame(paired, selection)

    assert selected.groupby("original_index")["seed"].apply(tuple).to_dict() == {
        "included-a": tuple(range(20)),
        "included-b": tuple(range(20)),
    }
    assert len(selected) == 40


def test_analysis_outputs_publish_no_evaluation_alias(tmp_path: Path) -> None:
    paired_all = pd.DataFrame(
        {
            "original_index": ["a", "a", "b", "b"],
            "seed": [0, 1, 0, 1],
            "latent_rmse": [1.0, 2.0, 3.0, 4.0],
            "sscd_cosine_similarity": [4.0, 3.0, 2.0, 1.0],
            "webster_overfit_type_normalized": ["N", "N", "TV", "TV"],
        }
    )
    paired_selected = paired_all.loc[paired_all["original_index"].eq("a")]

    statistics = write_analysis_outputs(
        tmp_path,
        paired_all=paired_all,
        paired_selected=paired_selected,
    )

    assert set(statistics.as_dict()) == {
        "all",
        "selected",
        "non_tv_selected",
        "selected_tv_all_seeds",
    }
    for name in ("all", "selected"):
        assert (tmp_path / f"paired_{name}.csv").is_file()
        assert (tmp_path / f"paired_{name}.parquet").is_file()
        assert (tmp_path / f"proximity_vs_sscd_{name}.png").is_file()
        assert (tmp_path / f"proximity_vs_sscd_{name}.pdf").is_file()
    assert not list(tmp_path.glob("*evaluation*"))
