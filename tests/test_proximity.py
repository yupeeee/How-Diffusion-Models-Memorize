"""Offline proximity, plotting-statistics, and supplied-cache parity tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from utils.experiments import plotting as plotting_module
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


def test_analysis_outputs_publish_theorem_aligned_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    original_style = {
        key: plotting_module.matplotlib.rcParams[key]
        for key in ("font.family", "font.size", "mathtext.fontset")
    }
    figures: list[object] = []
    styles: list[dict[str, object]] = []
    save_calls: list[dict[str, object]] = []
    tight_layout_calls: list[int] = []
    real_subplots = plotting_module.plt.subplots

    def subplots_spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        figure, axis = real_subplots(*args, **kwargs)
        real_savefig = figure.savefig
        real_tight_layout = figure.tight_layout

        def savefig_spy(*save_args: object, **save_kwargs: object) -> object:
            save_calls.append(dict(save_kwargs))
            return real_savefig(*save_args, **save_kwargs)

        def tight_layout_spy(*layout_args: object, **layout_kwargs: object) -> object:
            tight_layout_calls.append(1)
            return real_tight_layout(*layout_args, **layout_kwargs)

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

    assert {
        key: plotting_module.matplotlib.rcParams[key] for key in original_style
    } == original_style
    assert (
        styles
        == [
            {
                "font.family": ("STIXGeneral",),
                "font.size": 15.0,
                "mathtext.fontset": "stix",
            }
        ]
        * 2
    )
    assert len(figures) == 2
    assert sum(tight_layout_calls) == 2
    for figure in figures:
        assert tuple(figure.get_size_inches()) == pytest.approx((4.0, 4.0))
        assert figure._suptitle is None
        axis = figure.axes[0]
        assert axis.get_title() == ""
        assert axis.get_xscale() == "linear"
        assert axis.get_yscale() == "linear"
        assert axis.get_xlabel() == (
            r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|_2/\sqrt{d}$"
        )
        assert axis.get_ylabel() == "SSCD"
        assert axis.xaxis.label.get_fontsize() == 15
        assert axis.yaxis.label.get_fontsize() == 15
        assert axis.xaxis.label.get_fontfamily() == ["STIXGeneral"]
        assert axis.yaxis.label.get_fontfamily() == ["STIXGeneral"]
        assert all(label.get_fontsize() == 12 for label in axis.get_xticklabels())
        assert all(label.get_fontsize() == 12 for label in axis.get_yticklabels())
        assert len(axis.texts) == 1
        assert axis.texts[0].get_text() == "PCC: -1.000\nSpearman rho: -1.000"
        assert axis.texts[0].get_position() == pytest.approx((0.02, 0.02))
        assert axis.texts[0].get_horizontalalignment() == "left"
        assert axis.texts[0].get_verticalalignment() == "bottom"
        assert axis.texts[0].get_fontsize() == 10
        assert axis.texts[0].get_fontfamily() == ["STIXGeneral"]
        legend = axis.get_legend()
        assert legend is not None
        assert legend.get_title().get_fontsize() == 10
        assert all(text.get_fontsize() == 10 for text in legend.get_texts())
        assert not legend.get_frame_on()
        assert all(
            handle.get_alpha() == pytest.approx(1.0)
            for handle in legend.legend_handles
        )
        assert all(
            collection.get_alpha() == pytest.approx(0.35)
            for collection in axis.collections
        )
    assert (
        save_calls.count(
            {
                "format": "png",
                "dpi": 300,
                "bbox_inches": "tight",
                "pad_inches": 0.05,
            }
        )
        == 2
    )
    assert (
        save_calls.count(
            {
                "format": "pdf",
                "bbox_inches": "tight",
                "pad_inches": 0.05,
            }
        )
        == 2
    )


def _selection_configuration_for_proximity() -> dict[str, object]:
    from utils.data import selection as selection_module

    return {
        "schema_version": selection_module.SELECTION_SCHEMA_VERSION,
        "selection_policy": selection_module.SELECTION_POLICY,
        "boundary": selection_module.REFERENCE_SSCD_BOUNDARY,
        "category_rules": selection_module._category_rules(),
        "selection_seeds": list(selection_module.SELECTION_SEEDS),
        "reference_validation_seeds": list(
            selection_module.REFERENCE_VALIDATION_SEEDS
        ),
    }


def test_proximity_configuration_records_the_complete_selection_contract() -> None:
    selection = SimpleNamespace(
        configuration=_selection_configuration_for_proximity(),
        sha256="c" * 64,
    )
    paths = ProximityPaths.build(
        ROOT,
        Path(RUN_NAME) / "experiment_S0_N20",
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
    configuration = proximity_module._analysis_configuration(
        paths,
        RUN_NAME,
        generation,
        {"configuration_hash": "b" * 64},
        selection,
    )

    assert configuration["schema_version"] == 4
    assert configuration["selection_policy"] == selection.configuration[
        "selection_policy"
    ]
    assert configuration["selection_contract"] == selection.configuration
    assert configuration["selection_contract"]["category_rules"]["TV"][
        "comparison_operator"
    ] == ">="
    assert configuration["selection_contract"]["category_rules"]["N"][
        "comparison_operator"
    ] == ">="


@pytest.mark.parametrize(
    ("schema_version", "selection_policy"),
    [
        (2, "target_pair_selection"),
        (3, "target_pair_selection_tv_ge_0_25_n_lt_0_25"),
    ],
)
def test_stale_selection_error_names_only_current_derived_locations(
    tmp_path: Path,
    schema_version: int,
    selection_policy: str,
) -> None:
    from utils.data.selection import target_pair_selection_directory

    selection_directory = target_pair_selection_directory(
        tmp_path, model_name="sdv1"
    )
    selection_directory.mkdir(parents=True)
    (selection_directory / "config.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "selection_policy": selection_policy,
            }
        ),
        encoding="utf-8",
    )
    output_directory = (
        tmp_path
        / "outputs/sdv1_ddim_g7.5_T50_N20/proximity/experiment_S0_N20"
    )

    with pytest.raises(ProximityError) as captured:
        proximity_module._preexisting_selection(
            tmp_path,
            model_name="sdv1",
            role="experiment",
            output_directory=output_directory,
        )

    message = str(captured.value)
    assert "Stale frozen target-pair selection" in message
    assert [
        line.removeprefix("- ") for line in message.splitlines() if line.startswith("- ")
    ] == [str(selection_directory), str(output_directory)]


def test_stale_proximity_configuration_names_only_its_derived_directory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "proximity/experiment_S0_N20/run_config.json"
    path.parent.mkdir(parents=True)
    existing = {
        "schema_version": 2,
        "analysis": "proximity",
        "selection_policy": "target_pair_selection",
        "selection_hash": "a" * 64,
    }
    path.write_text(json.dumps(existing), encoding="utf-8")
    desired = {
        **existing,
        "schema_version": 4,
        "selection_policy": _selection_configuration_for_proximity()[
            "selection_policy"
        ],
        "selection_contract": _selection_configuration_for_proximity(),
    }

    with pytest.raises(ProximityError) as captured:
        proximity_module._write_or_validate_configuration(path, desired)

    assert str(captured.value).splitlines()[-1] == f"- {path.parent}"
    assert json.loads(path.read_text(encoding="utf-8")) == existing


def test_proximity_copies_both_tv_and_n_selection_sidecars(tmp_path: Path) -> None:
    from utils.data.selection import target_pair_selection_directory

    source = target_pair_selection_directory(tmp_path, model_name="sdv1")
    source.mkdir(parents=True)
    filenames = (
        "selection.csv",
        "selected_tv.csv",
        "excluded_tv.csv",
        "selected_n.csv",
        "excluded_n.csv",
        "threshold_diagnostics.csv",
    )
    for filename in filenames:
        (source / filename).write_text(filename, encoding="utf-8")
    output = tmp_path / "analysis"
    output.mkdir()
    selection = SimpleNamespace(root=tmp_path, model_name="sdv1")

    proximity_module._write_selection_outputs(output, selection)

    assert {
        path.name: path.read_text(encoding="utf-8")
        for path in output.iterdir()
    } == {filename: filename for filename in filenames}


def test_proximity_summary_counts_tv_and_n_prompts_not_seed_rows() -> None:
    selection = SimpleNamespace(
        frame=pd.DataFrame(
            {
                "original_index": ["tv-in", "tv-out", "n-in", "n-out", "mv"],
                "webster_overfit_type_normalized": ["TV", "TV", "N", "N", "MV"],
                "include_target_pair": [True, False, True, False, True],
                "selection_validation_agree": [True, False, False, True, None],
            }
        )
    )

    assert proximity_module._selection_prompt_counts(selection) == {
        "included_tv_prompts": 1,
        "excluded_tv_prompts": 1,
        "included_n_prompts": 1,
        "excluded_n_prompts": 1,
        "tv_selection_validation_disagreements": 1,
        "n_selection_validation_disagreements": 1,
    }


def test_experiment_inputs_cannot_change_frozen_selection_or_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from utils.data import selection as selection_module

    frozen_frame = pd.DataFrame(
        {
            "original_index": ["tv", "n"],
            "selection_mean_sscd": [0.4, 0.1],
            "reference_validation_mean_sscd": [0.2, 0.3],
            "selection_validation_agree": [False, False],
            "include_target_pair": [True, True],
        }
    )
    frozen = SimpleNamespace(
        frame=frozen_frame,
        diagnostics=pd.DataFrame(
            {
                "category": ["TV", "N"],
                "selection_validation_disagreement_count": [1, 1],
            }
        ),
        sha256="d" * 64,
    )
    monkeypatch.setattr(
        selection_module,
        "load_target_pair_selection",
        lambda _root, *, model_name: frozen,
    )
    common = {
        "model_name": "sdv1",
        "scheduler_name": "ddim",
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "seed_start": 0,
        "generation_config": {"ignored": "experiment"},
        "sscd_config": {"ignored": "experiment"},
    }

    first = proximity_module._selection_for_run(
        tmp_path,
        **common,
        num_seeds=2,
        paired_all=pd.DataFrame(
            {
                "seed": [0, 1],
                "latent_rmse": [0.0, 1000.0],
                "sscd_cosine_similarity": [-1.0, 1.0],
            }
        ),
    )
    second = proximity_module._selection_for_run(
        tmp_path,
        **common,
        num_seeds=20,
        paired_all=pd.DataFrame(
            {
                "seed": list(range(20)),
                "latent_rmse": list(reversed(range(20))),
                "sscd_cosine_similarity": [0.99] * 20,
            }
        ),
    )

    assert first is frozen
    assert second is frozen
    assert first.sha256 == second.sha256 == "d" * 64
    pd.testing.assert_frame_equal(first.frame, second.frame, check_exact=True)
    pd.testing.assert_frame_equal(
        first.diagnostics,
        second.diagnostics,
        check_exact=True,
    )
