"""Offline proximity integration and plotting tests."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from scripts.compute_proximity import build_parser as build_proximity_parser
from utils.common.io import atomic_torch_save, atomic_write_json
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


def test_compute_proximity_parser_supports_overwrite_and_strategies() -> None:
    parser = build_proximity_parser()

    defaults = parser.parse_args([])
    assert defaults.selection_strategy == "gmm"
    assert defaults.overwrite is False
    assert parser.parse_args(["--overwrite"]).overwrite is True
    assert (
        parser.parse_args(["--selection-strategy", "gmm"]).selection_strategy == "gmm"
    )
    assert (
        parser.parse_args(["--selection-strategy", "gmm-evidence"])
        .selection_strategy
        == "gmm-evidence"
    )
    assert (
        parser.parse_args(["--selection-strategy", "spearman"]).selection_strategy
        == "spearman"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--selection-strategy", "unsupported"])
    assert (
        inspect.signature(proximity_module.run_proximity)
        .parameters["overwrite"]
        .default
        is inspect.Parameter.empty
    )


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
    spearman_paths = ProximityPaths.build(
        tmp_path.resolve(),
        experiment_run,
        role="experiment",
        seed_start=0,
        num_seeds=20,
        selection_strategy="spearman",
    )
    gmm_evidence_paths = ProximityPaths.build(
        tmp_path.resolve(),
        experiment_run,
        role="experiment",
        seed_start=0,
        num_seeds=20,
        selection_strategy="gmm-evidence",
    )
    reference = ProximityPaths.frozen_selection(
        tmp_path.resolve(),
        tmp_path.resolve() / "logs" / RUN_NAME / "reference_S20_N20",
        tmp_path.resolve() / "data/webster/selection/sdv1/reference_S20_N20",
    )

    assert paths.output_directory == (
        tmp_path.resolve() / "outputs" / RUN_NAME / "proximity/gmm/experiment_S0_N20"
    )
    assert spearman_paths.output_directory == (
        tmp_path.resolve()
        / "outputs"
        / RUN_NAME
        / "proximity/spearman/experiment_S0_N20"
    )
    assert gmm_evidence_paths.output_directory == (
        tmp_path.resolve()
        / "outputs"
        / RUN_NAME
        / "proximity/gmm-evidence/experiment_S0_N20"
    )
    assert paths.run_config_json.name == "run_config.json"
    assert reference.run_config_json.name == "config.json"
    assert reference.summary_json.name == "summary.json"
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
            selection_strategy="spearman",
        )
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
            selection_strategy="gmm-evidence",
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
        {
            "model_name": "sdv1",
            "scheduler_name": "ddpm",
            "guidance_scale": 3.25,
            "num_inference_steps": 17,
            "num_seeds": 3,
            "selection_strategy": "spearman",
        },
        {
            "model_name": "sdv1",
            "scheduler_name": "ddpm",
            "guidance_scale": 3.25,
            "num_inference_steps": 17,
            "num_seeds": 3,
            "selection_strategy": "gmm-evidence",
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
                "selection_strategy": "gmm-evidence",
                "prompt_spearman": -1.0 if index == "1" else 1.0,
                "prompt_gmm_evidence_seed_count": (1 if index == "1" else 0),
                "include_prompt": index == "1",
                "selection_status": (
                    "included_proximity_rule"
                    if index == "1"
                    else "discarded_proximity_rule"
                ),
                "selection_reason": (
                    "gmm_high_proximity_evidence_and_prompt_spearman_lt_0"
                    if index == "1"
                    else "prompt_spearman_ge_0"
                ),
            }
            for index in indices
        ]
    )
    return SimpleNamespace(
        prompt_frame=prompts,
        configuration={
            "selection_policy": selection_module.selection_policy("gmm-evidence"),
            "selection_strategy": "gmm-evidence",
        },
        sha256="c" * 64,
    )


def test_experiment_rows_receive_frozen_whole_prompt_decisions() -> None:
    observations = pd.concat(
        [_observations("1"), _observations("2")], ignore_index=True
    )
    annotated = _annotate_selection(observations, _selection(("1", "2")))

    assert tuple(annotated) == ANALYSIS_COLUMNS
    assert annotated.groupby("original_index")["include_prompt"].unique().map(
        tuple
    ).to_dict() == {"1": (True,), "2": (False,)}
    assert annotated.groupby("original_index")[
        "prompt_gmm_evidence_seed_count"
    ].unique().map(tuple).to_dict() == {"1": (1,), "2": (0,)}
    assert set(annotated["selection_strategy"]) == {"gmm-evidence"}
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
        "selection_strategy": "spearman",
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    path_identities: list[dict[str, object]] = []
    load_identities: list[dict[str, object]] = []
    fingerprint = {
        "generation": {"count": 1, "sha256": "b" * 64},
        "sscd": {"count": 1, "sha256": "c" * 64},
    }

    def selection_directory(_root: Path, **identity: object) -> Path:
        path_identities.append(identity)
        return directory

    def load_selection(_root: Path, **identity: object) -> SimpleNamespace:
        load_identities.append(identity)
        return SimpleNamespace(
            sha256="a" * 64,
            configuration={"reference_completion_fingerprint": fingerprint},
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
    plotted: list[Path] = []

    def plot_selection(directory: str | Path) -> plotting_module.AnalysisStatistics:
        plotted.append(Path(directory))
        return plotting_module.AnalysisStatistics(1, 1, 1, 1.0, -1.0)

    monkeypatch.setattr(proximity_module, "write_selection_figure", plot_selection)
    result = proximity_module._frozen_reference_result(
        tmp_path,
        tmp_path / "reference",
        "sdv1",
        "ddpm",
        3.25,
        17,
        7,
        "spearman",
    )
    assert result is not None
    assert result.exit_code == 0
    assert result.paths.output_directory == directory
    assert plotted == [directory]
    assert [identity["selection_strategy"] for identity in path_identities] == [
        "spearman"
    ]
    assert [identity["selection_strategy"] for identity in load_identities] == [
        "spearman"
    ]
    assert fingerprint_runs == [tmp_path / "reference"]
    assert not (tmp_path / "outputs").exists()


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

    with pytest.raises(ProximityError) as captured:
        proximity_module._frozen_reference_result(
            tmp_path,
            tmp_path / "reference",
            "sdv1",
            "ddpm",
            3.25,
            17,
            7,
            "spearman",
        )

    assert plotted == []
    assert directory.is_dir()
    message = str(captured.value)
    assert "changed marker groups: generation" in message
    assert "run_all.sh --overwrite" not in message
    assert (
        "./compute_proximity.sh --model sdv1 --scheduler ddpm --g 3.25 "
        "--T 17 --N 7 --seed-start 7 --selection-strategy spearman --overwrite"
        in message
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
        selection_strategy="gmm-evidence",
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
    assert config["selection_policy"] == selection_module.selection_policy(
        "gmm-evidence"
    )
    assert config["selection_strategy"] == "gmm-evidence"
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
    assert summary["selection_policy"] == selection_module.selection_policy(
        "gmm-evidence"
    )
    assert summary["selection_strategy"] == "gmm-evidence"
    assert summary["figures"] == {
        scope: {
            file_format: (paths.output_directory / filename)
            .relative_to(tmp_path.resolve())
            .as_posix()
            for file_format, filename in formats.items()
        }
        for scope, formats in plotting_module.PROXIMITY_FIGURES.items()
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


def test_figure_catalog_and_failure_cleanup_cover_only_four_known_outputs(
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
    assert {path.name for path in tmp_path.iterdir()} == {"proximity.csv"}


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
            "selection_strategy": ["gmm-evidence"] * 8,
            "prompt_gmm_evidence_seed_count": [
                1,
                1,
                0,
                0,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ],
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
                "gmm_high_proximity_evidence_and_prompt_spearman_lt_0",
                "gmm_high_proximity_evidence_and_prompt_spearman_lt_0",
                "no_gmm_high_proximity_evidence",
                "no_gmm_high_proximity_evidence",
                "cache_error",
                "cache_error",
                "cache_error",
                "cache_error",
            ],
        }
    )
    selection.to_csv(tmp_path / "selection.csv", index=False)

    read_paths: list[Path] = []
    plotted_frames: dict[str, pd.DataFrame] = {}
    plotted_figures: dict[str, object] = {}
    real_read_csv = plotting_module.pd.read_csv
    real_scatter_figure = plotting_module._scatter_figure

    def read_csv_spy(path: str | Path, *args: object, **kwargs: object) -> pd.DataFrame:
        read_paths.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    def scatter_spy(
        frame: pd.DataFrame,
        statistics: plotting_module.AnalysisStatistics,
        *,
        population_label: str,
    ) -> object:
        plotted_frames[population_label] = frame.copy(deep=True)
        figure = real_scatter_figure(
            frame,
            statistics,
            population_label=population_label,
        )
        plotted_figures[population_label] = figure
        return figure

    monkeypatch.setattr(plotting_module.pd, "read_csv", read_csv_spy)
    monkeypatch.setattr(plotting_module, "_scatter_figure", scatter_spy)

    statistics = write_selection_figure(tmp_path)

    assert statistics.as_dict() == {
        "total_selected_prompts": 1,
        "evaluable_selected_prompts": 1,
        "negative_spearman_prompts": 1,
        "negative_spearman_fraction": pytest.approx(1.0),
        "median_spearman": pytest.approx(-1.0),
    }
    assert read_paths == [tmp_path / "selection.csv"]
    assert set(plotted_frames) == {"Prompts before discard", "Selected prompts"}
    assert set(plotted_frames["Selected prompts"]["original_index"]) == {"included"}
    assert set(plotted_frames["Prompts before discard"]["original_index"]) == {
        "included",
        "discarded",
    }
    assert (
        plotted_frames["Prompts before discard"][["l2_norm", "sscd"]]
        .notna()
        .all()
        .all()
    )
    selected_axes = plotted_figures["Selected prompts"].axes[0]
    all_axes = plotted_figures["Prompts before discard"].axes[0]
    assert selected_axes.get_xlim() == pytest.approx(all_axes.get_xlim())
    assert selected_axes.get_ylim() == pytest.approx(all_axes.get_ylim())
    assert 8.0 < all_axes.get_xlim()[1] < 20.0
    assert selected_axes.texts[0].get_text().startswith("Selected prompts: 1\n")
    assert all_axes.texts[0].get_text().startswith("Prompts before discard: 2\n")
    assert {path.name for path in tmp_path.iterdir()} == {
        "selection.csv",
        *plotting_module.PROXIMITY_FIGURE_FILENAMES,
    }
    assert not any(path.is_dir() for path in tmp_path.iterdir())
    assert (tmp_path / "proximity_vs_sscd.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "proximity_vs_sscd.pdf").read_bytes().startswith(b"%PDF")
    assert (
        (tmp_path / "proximity_vs_sscd_all_prompts.png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )
    assert (
        (tmp_path / "proximity_vs_sscd_all_prompts.pdf")
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

    with pytest.raises(plotting_module.PlottingError, match="observation_status"):
        write_selection_figure(tmp_path)


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
            "selection_strategy": ["gmm-evidence"] * 8,
            "prompt_gmm_evidence_seed_count": [
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
            ],
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
                "gmm_high_proximity_evidence_and_prompt_spearman_lt_0",
                "gmm_high_proximity_evidence_and_prompt_spearman_lt_0",
                "gmm_high_proximity_evidence_and_prompt_spearman_lt_0",
                "gmm_high_proximity_evidence_and_prompt_spearman_lt_0",
                "gmm_high_proximity_evidence_and_prompt_spearman_lt_0",
                "gmm_high_proximity_evidence_and_prompt_spearman_lt_0",
                "no_gmm_high_proximity_evidence",
                "no_gmm_high_proximity_evidence",
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
        *plotting_module.PROXIMITY_FIGURE_FILENAMES,
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
    assert sum(tight_layout_calls) == 2
    all_figure, figure = figures
    assert tuple(figure.get_size_inches()) == pytest.approx((4.0, 4.0))
    assert figure._suptitle is None
    axis = figure.axes[0]
    all_axis = all_figure.axes[0]
    assert axis.get_title() == ""
    assert axis.get_xscale() == axis.get_yscale() == "linear"
    assert axis.get_xlabel() == r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|_2$"
    assert axis.get_ylabel() == "SSCD"
    assert axis.xaxis.label.get_fontsize() == 15
    assert axis.yaxis.label.get_fontsize() == 15
    assert axis.xaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert axis.yaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert all(label.get_fontsize() == 12 for label in axis.get_xticklabels())
    assert all(label.get_fontsize() == 12 for label in axis.get_yticklabels())
    assert len(axis.texts) == 1
    assert axis.texts[0].get_text() == (
        "Selected prompts: 3\n"
        "Evaluable prompts: 2\n"
        "rho < 0: 1/2 (50.0%)\n"
        "Median rho: 0.000"
    )
    assert all_axis.texts[0].get_text() == (
        "Prompts before discard: 4\n"
        "Evaluable prompts: 3\n"
        "rho < 0: 2/3 (66.7%)\n"
        "Median rho: -1.000"
    )
    assert axis.get_xlim() == pytest.approx(all_axis.get_xlim())
    assert axis.get_ylim() == pytest.approx(all_axis.get_ylim())
    assert all_axis.get_xlim()[1] > 7.0
    assert all_axis.get_ylim()[1] > 9.0
    assert "PCC" not in axis.texts[0].get_text()
    assert axis.texts[0].get_position() == pytest.approx((0.02, 0.02))
    assert axis.texts[0].get_horizontalalignment() == "left"
    assert axis.texts[0].get_verticalalignment() == "bottom"
    assert axis.texts[0].get_fontsize() == 10
    assert axis.texts[0].get_fontfamily() == ["STIXGeneral"]
    legend = axis.get_legend()
    all_legend = all_axis.get_legend()
    assert legend is not None
    assert all_legend is not None
    assert legend.get_title().get_text() == ""
    assert all_legend.get_title().get_text() == ""
    assert [text.get_text() for text in legend.get_texts()] == ["TV", "RV", "N"]
    assert [text.get_text() for text in all_legend.get_texts()] == [
        "MV",
        "TV",
        "RV",
        "N",
    ]
    assert all(text.get_fontsize() == 10 for text in legend.get_texts())
    assert not legend.get_frame_on()
    assert all(
        handle.get_alpha() == pytest.approx(1.0) for handle in legend.legend_handles
    )
    assert all(
        collection.get_alpha() == pytest.approx(0.35) for collection in axis.collections
    )
    assert save_calls == [
        {
            "format": "png",
            "dpi": 300,
            "bbox_inches": "tight",
            "pad_inches": 0.05,
        },
        {
            "format": "pdf",
            "dpi": 300,
            "bbox_inches": "tight",
            "pad_inches": 0.05,
        },
        {
            "format": "png",
            "dpi": 300,
            "bbox_inches": "tight",
            "pad_inches": 0.05,
        },
        {
            "format": "pdf",
            "dpi": 300,
            "bbox_inches": "tight",
            "pad_inches": 0.05,
        },
    ]
    for filename in plotting_module.PROXIMITY_FIGURE_FILENAMES:
        artifact = tmp_path / filename
        assert artifact.is_file()
        expected_header = b"%PDF" if artifact.suffix == ".pdf" else b"\x89PNG"
        assert artifact.read_bytes().startswith(expected_header)
    assert all(
        not plotting_module.plt.fignum_exists(plotted.number) for plotted in figures
    )
