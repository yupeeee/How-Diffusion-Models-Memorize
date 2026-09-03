"""Offline tests for fixed model-specific target-pair selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from utils.common.io import canonical_hash
from utils.data import selection as selection_module
from utils.data.selection import (
    REFERENCE_VALIDATION_SEEDS,
    EXCLUDED_N_TARGET_UNSUPPORTED,
    EXCLUDED_TV_TARGET_UNSUPPORTED,
    INCLUDED_N_TARGET_SUPPORTED,
    INCLUDED_NON_TV,
    INCLUDED_TV_TARGET_SUPPORTED,
    INVALID_RECORD,
    MISSING_REFERENCE_SSCD,
    REFERENCE_GUIDANCE_SCALE,
    REFERENCE_NUM_INFERENCE_STEPS,
    REFERENCE_NUM_SEEDS,
    REFERENCE_SEED_START,
    REFERENCE_SEEDS,
    REFERENCE_SCHEDULER,
    REFERENCE_SSCD_BOUNDARY,
    SELECTION_COLUMNS,
    SELECTION_POLICY,
    SELECTION_SCHEMA_VERSION,
    SELECTION_SEEDS,
    THRESHOLD_SENSITIVITY,
    TV_MEAN_SSCD_THRESHOLD,
    FrozenTargetPairSelectionError,
    StaleTargetPairSelectionError,
    TargetPairSelectionError,
    TargetPairSelectionMissingError,
    apply_target_pair_selection,
    build_target_pair_selection,
    build_threshold_diagnostics,
    compute_target_pair_selection_hash,
    ensure_reference_target_pair_selection,
    exact_two_class_otsu,
    is_reference_configuration,
    load_target_pair_selection,
    normalize_webster_overfit_type,
    reference_run_name,
    reference_run_path,
    reference_selection_command,
    target_pair_selection_directory,
)


def _reference_configs(
    model_name: str = "sdv1", *, preprocessing_hash: str = "c" * 64
) -> tuple[dict[str, object], dict[str, object]]:
    dataset_model = "realisticvision" if model_name == "realvis" else model_name
    science: dict[str, object] = {
        "model_cli_name": model_name,
        "dataset_model": dataset_model,
        "scheduler": {"name": "ddim"},
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "num_seeds": 20,
        "seeds": list(REFERENCE_SEEDS),
    }
    generation_hash = canonical_hash(science)
    generation = {
        "scientific_config": science,
        "scientific_config_hash": generation_hash,
    }
    sscd: dict[str, object] = {
        "generation_scientific_config_hash": generation_hash,
        "num_seeds": 20,
        "seeds": list(REFERENCE_SEEDS),
        "sscd_checkpoint_sha256": "b" * 64,
        "sscd_preprocessing_hash": preprocessing_hash,
    }
    sscd["configuration_hash"] = canonical_hash(sscd)
    return generation, sscd


def _set_scheduler_default_order(
    generation: dict[str, object],
    sscd: dict[str, object],
    defaulted_keys: tuple[str, ...],
) -> None:
    science = generation["scientific_config"]
    assert isinstance(science, dict)
    science["scheduler"] = {
        "name": "ddim",
        "config": {"_use_default_values": list(defaulted_keys)},
    }
    generation_hash = canonical_hash(science)
    generation["scientific_config_hash"] = generation_hash
    sscd["generation_scientific_config_hash"] = generation_hash
    sscd.pop("configuration_hash", None)
    sscd["configuration_hash"] = canonical_hash(sscd)


def _records(
    labels: tuple[str, ...] = ("N", "TV", "TV", "TV"),
    *,
    model_name: str = "sdv1",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position, label in enumerate(labels):
        index = str(10_000 + position)
        rows.append(
            {
                "model_name": (
                    "realisticvision" if model_name == "realvis" else model_name
                ),
                "original_index": index,
                "record_id": f"{model_name}-{position:04d}",
                "source_row_number": position,
                "prompt_raw": f"prompt {position}",
                "webster_overfit_type": label,
                "target_image_sha256": hashlib.sha256(index.encode()).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def _paired(
    records: pd.DataFrame,
    *,
    selection_means: tuple[float, ...] = (0.01, 0.249999, 0.25, 0.30),
    reference_validation_means: tuple[float, ...] = (0.99, 0.99, -0.50, 0.00),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record, selection_mean, reference_validation_mean in zip(
        records.to_dict(orient="records"),
        selection_means,
        reference_validation_means,
        strict=True,
    ):
        for seed in REFERENCE_SEEDS:
            rows.append(
                {
                    "original_index": record["original_index"],
                    "seed": seed,
                    "sscd_cosine_similarity": (
                        selection_mean if seed in SELECTION_SEEDS else reference_validation_mean
                    ),
                    "latent_rmse": 1000.0 - seed,
                    "pearson_correlation": -0.99,
                }
            )
    return pd.DataFrame(rows)


def _build(
    root: Path,
    *,
    model_name: str = "sdv1",
    records: pd.DataFrame | None = None,
    paired: pd.DataFrame | None = None,
    preprocessing_hash: str = "c" * 64,
):
    records = _records(model_name=model_name) if records is None else records
    paired = _paired(records) if paired is None else paired
    generation, sscd = _reference_configs(
        model_name, preprocessing_hash=preprocessing_hash
    )
    return build_target_pair_selection(
        root,
        model_name=model_name,
        paired_frame=paired,
        records_frame=records,
        reference_run_config=generation,
        sscd_config=sscd,
    )


def test_fixed_constants_and_reference_identity_are_exact() -> None:
    assert REFERENCE_SSCD_BOUNDARY == 0.25
    assert TV_MEAN_SSCD_THRESHOLD == 0.25
    assert REFERENCE_SCHEDULER == "ddim"
    assert REFERENCE_GUIDANCE_SCALE == 7.5
    assert REFERENCE_NUM_INFERENCE_STEPS == 50
    assert REFERENCE_NUM_SEEDS == 20
    assert REFERENCE_SEED_START == 20
    assert REFERENCE_SEEDS == tuple(range(20, 40))
    assert SELECTION_SEEDS == tuple(range(20, 30))
    assert REFERENCE_VALIDATION_SEEDS == tuple(range(30, 40))
    assert SELECTION_POLICY == "target_pair_selection_tv_ge_0_25_n_ge_0_25"
    assert SELECTION_SCHEMA_VERSION == 4
    assert reference_run_name("sdv1") == "sdv1_ddim_g7.5_T50_S20_N20"
    assert reference_run_path("sdv1").as_posix() == (
        "logs/sdv1_ddim_g7.5_T50_N20/reference_S20_N20"
    )
    assert is_reference_configuration(
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=20,
    )
    assert not is_reference_configuration(
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=0,
    )
    assert not is_reference_configuration(
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=20,
    )


def test_tv_n_boundaries_and_unconditional_categories_use_only_selection_half(
    tmp_path: Path,
) -> None:
    records = _records(
        labels=("TV", "TV", "TV", "N", "N", "N", "MV", "RV", "other")
    )
    selection = _build(
        tmp_path,
        records=records,
        paired=_paired(
            records,
            selection_means=(
                0.249999,
                0.25,
                0.30,
                0.249999,
                0.25,
                0.30,
                0.99,
                0.99,
                0.99,
            ),
            reference_validation_means=(
                0.99,
                -0.50,
                0.00,
                0.99,
                -0.50,
                0.00,
                -10.0,
                -10.0,
                -10.0,
            ),
        ),
    )
    frame = selection.frame.set_index("original_index")

    assert selection.included_indices == frozenset(
        {"10001", "10002", "10004", "10005", "10006", "10007", "10008"}
    )
    assert selection.excluded_indices == frozenset({"10000", "10003"})
    assert selection.selected_tv_indices == frozenset({"10001", "10002"})
    assert selection.excluded_tv_indices == frozenset({"10000"})
    assert selection.selected_n_indices == frozenset({"10004", "10005"})
    assert selection.excluded_n_indices == frozenset({"10003"})

    assert frame.at["10000", "selection_status"] == EXCLUDED_TV_TARGET_UNSUPPORTED
    assert frame.at["10001", "selection_status"] == INCLUDED_TV_TARGET_SUPPORTED
    assert frame.at["10003", "selection_status"] == EXCLUDED_N_TARGET_UNSUPPORTED
    assert frame.at["10004", "selection_status"] == INCLUDED_N_TARGET_SUPPORTED
    assert frame.at["10003", "selection_reason"] == "n_selection_mean_sscd_lt_0_25"
    assert frame.at["10003", "target_semantics"] == (
        "n_target_unsupported_under_frozen_reference_criterion"
    )
    assert frame.at["10004", "selection_reason"] == "n_selection_mean_sscd_ge_0_25"
    assert frame.at["10004", "target_semantics"] == (
        "n_target_supported_under_frozen_reference_criterion"
    )
    assert frame.at["10006", "selection_status"] == INCLUDED_NON_TV
    assert frame.at["10007", "selection_status"] == INCLUDED_NON_TV
    assert frame.at["10008", "selection_status"] == INCLUDED_NON_TV

    assert frame.at["10000", "comparison_operator"] == ">="
    assert frame.at["10003", "comparison_operator"] == ">="
    assert not bool(frame.at["10000", "selection_rule_passes"])
    assert bool(frame.at["10001", "selection_rule_passes"])
    assert not bool(frame.at["10003", "selection_rule_passes"])
    assert bool(frame.at["10004", "selection_rule_passes"])
    assert not bool(frame.at["10001", "selection_validation_agree"])
    assert not bool(frame.at["10003", "selection_validation_agree"])
    assert frame.at["10001", "selection_mean_sscd"] == pytest.approx(0.25)
    assert frame.at["10004", "selection_mean_sscd"] == pytest.approx(0.25)
    assert frame.at["10001", "reference_validation_mean_sscd"] == pytest.approx(-0.5)
    summary = json.loads(
        (
            target_pair_selection_directory(tmp_path, model_name="sdv1")
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["selected_n_reference_validation_mean_ge_0_25_count"] == 0
    assert summary["all_selected_n_reference_validation_mean_ge_0_25"] is False
    assert "all_selected_n_reference_validation_mean_lt_0_25" not in summary


@pytest.mark.parametrize(
    ("raw_label", "normalized", "included", "status"),
    (
        ("normal", "N", True, INCLUDED_N_TARGET_SUPPORTED),
        ("non-memorized", "N", True, INCLUDED_N_TARGET_SUPPORTED),
        ("template", "TV", True, INCLUDED_TV_TARGET_SUPPORTED),
    ),
)
def test_canonical_webster_aliases_use_category_boundary_rules(
    tmp_path: Path,
    raw_label: str,
    normalized: str,
    included: bool,
    status: str,
) -> None:
    assert normalize_webster_overfit_type(raw_label) == normalized
    records = _records(labels=(raw_label,))
    selection = _build(
        tmp_path,
        records=records,
        paired=_paired(
            records,
            selection_means=(REFERENCE_SSCD_BOUNDARY,),
            reference_validation_means=(REFERENCE_SSCD_BOUNDARY,),
        ),
    )
    row = selection.frame.iloc[0]

    assert row["webster_overfit_type_normalized"] == normalized
    assert bool(row["include_target_pair"]) is included
    assert row["selection_status"] == status
    assert row["comparison_operator"] == ({"N": ">=", "TV": ">="}[normalized])


def test_exact_empty_raw_prompt_is_not_treated_as_an_invalid_pair(tmp_path: Path) -> None:
    records = _records(labels=("N",))
    records.loc[0, "prompt_raw"] = ""
    selection = _build(
        tmp_path,
        records=records,
        paired=_paired(records, selection_means=(0.25,), reference_validation_means=(0.25,)),
    )
    row = selection.frame.iloc[0]
    assert row["prompt_raw"] == ""
    assert row["selection_status"] == INCLUDED_N_TARGET_SUPPORTED
    assert bool(row["include_target_pair"])


def test_reference_validation_scores_never_change_inclusion_or_selection_hash(
    tmp_path: Path,
) -> None:
    records = _records()
    first = _build(tmp_path / "first", records=records, paired=_paired(records))
    changed = _paired(
        records,
        reference_validation_means=(-100.0, -100.0, 100.0, 100.0),
    )
    second = _build(tmp_path / "second", records=records, paired=changed)

    assert first.included_indices == second.included_indices
    assert first.sha256 == second.sha256
    assert not first.frame["reference_validation_mean_sscd"].equals(
        second.frame["reference_validation_mean_sscd"]
    )
    assert not first.frame["reference_validation_rule_passes"].equals(
        second.frame["reference_validation_rule_passes"]
    )
    assert not first.frame["selection_validation_agree"].equals(
        second.frame["selection_validation_agree"]
    )
    assert not first.diagnostics.equals(second.diagnostics)
    pd.testing.assert_series_equal(
        first.frame["selection_rule_passes"],
        second.frame["selection_rule_passes"],
    )


def test_proximity_and_correlation_never_control_selection_or_hash(
    tmp_path: Path,
) -> None:
    records = _records()
    paired = _paired(records)
    first = _build(tmp_path / "first", records=records, paired=paired)
    changed = paired.copy()
    changed["latent_rmse"] *= -1_000_000
    changed["pearson_correlation"] = 1.0
    second = _build(tmp_path / "second", records=records, paired=changed)

    assert first.included_indices == second.included_indices
    assert first.sha256 == second.sha256


def test_experiment_seed_values_change_neither_selection_nor_validation(
    tmp_path: Path,
) -> None:
    records = _records(labels=("N", "N", "TV"))
    reference = _paired(
        records,
        selection_means=(0.10, 0.30, 0.30),
        reference_validation_means=(0.20, 0.40, 0.40),
    )
    selection = _build(tmp_path, records=records, paired=reference)
    frozen_diagnostics = selection.frame.loc[
        :,
        [
            "selection_mean_sscd",
            "reference_validation_mean_sscd",
            "selection_rule_passes",
            "reference_validation_rule_passes",
            "selection_validation_agree",
        ],
    ].copy(deep=True)

    experiment = reference.copy(deep=True)
    experiment["seed"] -= 20
    first_selected = apply_target_pair_selection(experiment, selection)
    experiment["sscd_cosine_similarity"] *= -1_000_000
    experiment["latent_rmse"] += 12345
    second_selected = apply_target_pair_selection(experiment, selection)

    pd.testing.assert_frame_equal(
        first_selected.loc[:, ["original_index", "seed"]],
        second_selected.loc[:, ["original_index", "seed"]],
    )
    pd.testing.assert_frame_equal(
        selection.frame.loc[:, frozen_diagnostics.columns], frozen_diagnostics
    )
    assert selection.excluded_n_indices == frozenset({"10000"})


def test_filtering_keeps_or_removes_every_seed_at_prompt_scope(tmp_path: Path) -> None:
    records = _records(labels=("N", "N", "TV"))
    paired = _paired(
        records,
        selection_means=(0.10, 0.30, 0.30),
        reference_validation_means=(0.10, 0.30, 0.30),
    )
    selection = _build(tmp_path, records=records, paired=paired)

    selected = apply_target_pair_selection(paired, selection)
    assert selected.groupby("original_index").size().to_dict() == {
        "10001": 20,
        "10002": 20,
    }
    assert "10000" not in set(selected["original_index"])
    assert len(paired.loc[paired["original_index"].eq("10000")]) == 20
    assert selection.excluded_n_indices == frozenset({"10000"})


def test_missing_selection_names_reference_run_and_exact_command(tmp_path: Path) -> None:
    expected_command = (
        "./generate.sh --model sdv2 --scheduler ddim --g 7.5 "
        "--T 50 --N 20 --seed-start 20 --downscale 4\n"
        "./sscd.sh --model sdv2 --scheduler ddim --g 7.5 "
        "--T 50 --N 20 --seed-start 20\n"
        "./compute_proximity.sh --model sdv2 --scheduler ddim --g 7.5 "
        "--T 50 --N 20 --seed-start 20"
    )
    assert reference_selection_command("sdv2") == expected_command
    with pytest.raises(TargetPairSelectionMissingError) as captured:
        load_target_pair_selection(tmp_path, model_name="sdv2")
    message = str(captured.value)
    assert (
        "logs/sdv2_ddim_g7.5_T50_N20/reference_S20_N20" in message
    )
    assert expected_command in message


def test_legacy_schema_one_selection_is_preserved_and_reported_incompatible(
    tmp_path: Path,
) -> None:
    legacy_directory = tmp_path / "data/webster/selection/sdv1"
    legacy_directory.mkdir(parents=True)
    legacy_config = {"schema_version": 1, "sentinel": "preserve me"}
    config_path = legacy_directory / "config.json"
    config_path.write_text(json.dumps(legacy_config), encoding="utf-8")

    with pytest.raises(
        TargetPairSelectionMissingError,
        match="Preserved schema-1 selection is incompatible",
    ):
        load_target_pair_selection(tmp_path, model_name="sdv1")

    assert json.loads(config_path.read_text(encoding="utf-8")) == legacy_config
    assert not (legacy_directory / "reference_S20_N20").exists()


@pytest.mark.parametrize(
    ("schema_version", "selection_policy"),
    (
        (2, "target_pair_selection"),
        (3, "target_pair_selection_tv_ge_0_25_n_lt_0_25"),
    ),
)
def test_prior_policy_selection_is_preserved_and_rejected_before_sidecars(
    tmp_path: Path, schema_version: int, selection_policy: str,
) -> None:
    directory = target_pair_selection_directory(tmp_path, model_name="sdv1")
    directory.mkdir(parents=True)
    config = {
        "schema_version": schema_version,
        "selection_policy": selection_policy,
        "selection_hash": "a" * 64,
        "sentinel": "preserve me",
    }
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(StaleTargetPairSelectionError) as captured:
        load_target_pair_selection(tmp_path, model_name="sdv1")

    message = str(captured.value)
    assert "Stale prior-policy frozen target-pair selection" in message
    assert f"- {directory}" in message
    assert "selected_n.csv" not in message
    assert json.loads(config_path.read_text(encoding="utf-8")) == config


def test_model_specific_directories_and_selection_are_not_shared(tmp_path: Path) -> None:
    sdv1 = _build(tmp_path, model_name="sdv1")
    assert target_pair_selection_directory(
        tmp_path, model_name="sdv1"
    ).is_dir()
    with pytest.raises(TargetPairSelectionMissingError):
        load_target_pair_selection(tmp_path, model_name="sdv2")

    realvis = _build(tmp_path, model_name="realvis")
    assert realvis.model_name == "realvis"
    realvis_directory = target_pair_selection_directory(tmp_path, model_name="realvis")
    assert realvis_directory.name == "reference_S20_N20"
    assert realvis_directory.parent.name == "realisticvision"
    assert sdv1.sha256 != realvis.sha256
    assert set(realvis.frame["threshold"]) == {0.25}


def test_configuration_serializes_category_rules_and_identical_inputs_hash_stably(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")

    assert first.sha256 == second.sha256
    assert first.configuration == second.configuration
    config = first.configuration
    assert config["boundary"] == 0.25
    assert config["threshold"] == 0.25
    assert config["selection_seeds"] == list(range(20, 30))
    assert config["reference_validation_seeds"] == list(range(30, 40))
    rules = config["category_rules"]
    assert isinstance(rules, dict)
    assert rules["TV"]["comparison_operator"] == ">="
    assert rules["TV"]["include_when"] == "mean_reference_sscd >= boundary"
    assert rules["N"]["comparison_operator"] == ">="
    assert rules["N"]["include_when"] == "mean_reference_sscd >= boundary"
    assert rules["MV"]["decision"] == "unconditional_include_preserved"
    assert rules["RV"]["decision"] == "unconditional_include_preserved"


def test_frozen_selection_reuses_identical_evidence_and_rejects_changes(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path)
    rebuilt = _build(tmp_path)
    assert rebuilt.sha256 == first.sha256
    loaded = ensure_reference_target_pair_selection(tmp_path, model_name="sdv1")
    assert loaded.sha256 == first.sha256

    records = _records()
    changed = _paired(records, selection_means=(0.01, 0.30, 0.25, 0.30))
    with pytest.raises(FrozenTargetPairSelectionError):
        _build(tmp_path, records=records, paired=changed)


def test_frozen_selection_reuses_exact_evidence_across_aggregate_hash_drift(
    tmp_path: Path,
) -> None:
    records = _records()
    paired = _paired(records)
    first_generation, first_sscd = _reference_configs()
    _set_scheduler_default_order(
        first_generation,
        first_sscd,
        ("prediction_type", "thresholding", "sample_max_value"),
    )
    first = build_target_pair_selection(
        tmp_path,
        model_name="sdv1",
        paired_frame=paired,
        records_frame=records,
        reference_run_config=first_generation,
        sscd_config=first_sscd,
    )
    directory = target_pair_selection_directory(tmp_path, model_name="sdv1")
    frozen_bytes = {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }

    second_generation, second_sscd = _reference_configs()
    _set_scheduler_default_order(
        second_generation,
        second_sscd,
        ("sample_max_value", "prediction_type", "thresholding"),
    )
    assert (
        second_generation["scientific_config_hash"]
        != first_generation["scientific_config_hash"]
    )
    assert second_sscd["configuration_hash"] != first_sscd["configuration_hash"]
    reused = build_target_pair_selection(
        tmp_path,
        model_name="sdv1",
        paired_frame=paired,
        records_frame=records,
        reference_run_config=second_generation,
        sscd_config=second_sscd,
    )
    assert reused.sha256 == first.sha256
    assert frozen_bytes == {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }

    changed = paired.copy(deep=True)
    mask = changed["seed"].eq(SELECTION_SEEDS[0])
    changed.loc[mask.idxmax(), "sscd_cosine_similarity"] += 1e-6
    with pytest.raises(FrozenTargetPairSelectionError):
        build_target_pair_selection(
            tmp_path,
            model_name="sdv1",
            paired_frame=changed,
            records_frame=records,
            reference_run_config=second_generation,
            sscd_config=second_sscd,
        )


@pytest.mark.parametrize(
    "field",
    ("sscd_checkpoint_sha256", "sscd_preprocessing_hash"),
)
def test_frozen_selection_rejects_changed_sscd_contract_with_equal_scores(
    tmp_path: Path,
    field: str,
) -> None:
    records = _records()
    paired = _paired(records)
    _build(tmp_path, records=records, paired=paired)
    generation, sscd = _reference_configs()
    sscd[field] = "d" * 64
    sscd.pop("configuration_hash")
    sscd["configuration_hash"] = canonical_hash(sscd)

    with pytest.raises(FrozenTargetPairSelectionError):
        build_target_pair_selection(
            tmp_path,
            model_name="sdv1",
            paired_frame=paired,
            records_frame=records,
            reference_run_config=generation,
            sscd_config=sscd,
        )


@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_frozen_selection_rejects_inexact_config_keys(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / mutation
    _build(root)
    config_path = (
        target_pair_selection_directory(root, model_name="sdv1")
        / "config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if mutation == "extra":
        config["paper_outcome"] = {"pearson": 1.0}
        expected = "unexpected: paper_outcome"
    else:
        config.pop("decision_scope")
        expected = "missing: decision_scope"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TargetPairSelectionError, match=expected):
        load_target_pair_selection(root, model_name="sdv1")

def test_selection_publication_failure_never_exposes_an_incomplete_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_csv(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(selection_module, "atomic_write_frame_csv", fail_csv)
    with pytest.raises(OSError, match="synthetic publication failure"):
        _build(tmp_path)

    directory = target_pair_selection_directory(tmp_path, model_name="sdv1")
    assert not directory.exists()
    assert not list(directory.parent.glob(f".{directory.name}.*.tmp"))


@pytest.mark.parametrize(
    "artifact",
    (
        "selection.parquet",
        "selection.csv",
        "selected_tv.csv",
        "excluded_tv.csv",
        "selected_n.csv",
        "excluded_n.csv",
        "threshold_diagnostics.csv",
        "config.json",
        "summary.json",
    ),
)
def test_frozen_selection_rejects_tampered_manifest_and_sidecars(
    tmp_path: Path, artifact: str
) -> None:
    root = tmp_path / artifact.replace(".", "_")
    _build(root)
    directory = target_pair_selection_directory(root, model_name="sdv1")
    path = directory / artifact
    if artifact == "selection.parquet":
        frame = pd.read_parquet(path)
        frame.loc[0, "prompt_raw"] = "tampered prompt"
        frame.to_parquet(path, index=False)
    elif artifact == "selection.csv":
        frame = pd.read_csv(path, dtype={"original_index": str})
        frame.loc[0, "selection_reason"] = "tampered_reason"
        frame.to_csv(path, index=False)
    elif artifact in {
        "selected_tv.csv",
        "excluded_tv.csv",
        "selected_n.csv",
        "excluded_n.csv",
    }:
        frame = pd.read_csv(path, dtype={"original_index": str})
        if frame.empty:
            frame.drop(columns=[frame.columns[-1]]).to_csv(path, index=False)
        else:
            frame.iloc[1:].to_csv(path, index=False)
    elif artifact == "threshold_diagnostics.csv":
        frame = pd.read_csv(path)
        frame.loc[0, "included_prompt_count"] += 1
        frame.to_csv(path, index=False)
    else:
        values = json.loads(path.read_text(encoding="utf-8"))
        key = "threshold" if artifact == "config.json" else "included_prompt_count"
        values[key] = -1
        path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(TargetPairSelectionError):
        load_target_pair_selection(root, model_name="sdv1")


def test_n_sidecars_are_one_row_per_prompt_and_empty_files_keep_schema(
    tmp_path: Path,
) -> None:
    records = _records(labels=("N", "N", "N"))
    selection = _build(
        tmp_path / "mixed",
        records=records,
        paired=_paired(
            records,
            selection_means=(0.10, 0.25, 0.30),
            reference_validation_means=(0.10, 0.25, 0.30),
        ),
    )
    directory = target_pair_selection_directory(
        tmp_path / "mixed", model_name="sdv1"
    )
    selected_n = pd.read_csv(directory / "selected_n.csv", dtype={"original_index": str})
    excluded_n = pd.read_csv(directory / "excluded_n.csv", dtype={"original_index": str})
    assert tuple(selected_n.columns) == SELECTION_COLUMNS
    assert tuple(excluded_n.columns) == SELECTION_COLUMNS
    assert selected_n["original_index"].tolist() == ["10001", "10002"]
    assert excluded_n["original_index"].tolist() == ["10000"]
    assert not selected_n["original_index"].duplicated().any()
    assert not excluded_n["original_index"].duplicated().any()
    assert selection.selected_n_indices == frozenset({"10001", "10002"})
    assert selection.excluded_n_indices == frozenset({"10000"})

    tv_only_records = _records(labels=("TV",))
    _build(
        tmp_path / "empty",
        records=tv_only_records,
        paired=_paired(
            tv_only_records,
            selection_means=(0.30,),
            reference_validation_means=(0.30,),
        ),
    )
    empty_directory = target_pair_selection_directory(
        tmp_path / "empty", model_name="sdv1"
    )
    for filename in ("selected_n.csv", "excluded_n.csv"):
        empty = pd.read_csv(empty_directory / filename)
        assert empty.empty
        assert tuple(empty.columns) == SELECTION_COLUMNS


def test_local_csv_parquet_and_json_inputs_are_supported(tmp_path: Path) -> None:
    records = _records()
    paired = _paired(records)
    generation, sscd = _reference_configs()
    cache = tmp_path / "cache"
    cache.mkdir()
    paired.to_parquet(cache / "paired.parquet", index=False)
    records.to_csv(cache / "records.csv", index=False)
    (cache / "run.json").write_text(json.dumps(generation), encoding="utf-8")
    (cache / "sscd.json").write_text(json.dumps(sscd), encoding="utf-8")

    selection = build_target_pair_selection(
        tmp_path,
        model_name="sdv1",
        paired_frame=cache / "paired.parquet",
        records_frame=cache / "records.csv",
        reference_run_config=cache / "run.json",
        sscd_config=cache / "sscd.json",
    )
    assert len(selection.frame) == 4
    assert selection.included_indices == frozenset({"10002", "10003"})


def test_hash_covers_prompt_target_sscd_hash_and_threshold_but_not_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _build(tmp_path)
    config = selection.configuration

    changed_prompt = selection.frame.copy(deep=True)
    changed_prompt.loc[0, "prompt_raw"] = "different prompt"
    changed_target = selection.frame.copy(deep=True)
    changed_target.loc[0, "target_image_sha256"] = "d" * 64
    kwargs = {
        "model_name": "sdv1",
        "reference_generation_hash": config["reference_generation_hash"],
        "reference_sscd_hash": config["reference_sscd_hash"],
    }
    assert compute_target_pair_selection_hash(changed_prompt, **kwargs) != selection.sha256
    assert compute_target_pair_selection_hash(changed_target, **kwargs) != selection.sha256
    assert compute_target_pair_selection_hash(
        selection.frame, **{**kwargs, "reference_sscd_hash": "e" * 64}
    ) != selection.sha256
    assert compute_target_pair_selection_hash(
        selection.frame, **kwargs, threshold=0.30
    ) != selection.sha256

    irrelevant = selection.frame.copy(deep=True)
    irrelevant["latent_proximity"] = range(len(irrelevant))
    irrelevant["pearson_correlation"] = 1.0
    assert compute_target_pair_selection_hash(irrelevant, **kwargs) == selection.sha256

    monkeypatch.setattr(selection_module, "SELECTION_POLICY", "old_tv_only_policy")
    assert compute_target_pair_selection_hash(selection.frame, **kwargs) != selection.sha256


def test_missing_scores_and_invalid_pairs_have_explicit_statuses(tmp_path: Path) -> None:
    records = _records(labels=("TV", "N"))
    records.loc[1, "target_image_sha256"] = "not-a-hash"
    paired = _paired(
        records,
        selection_means=(0.5, 0.1),
        reference_validation_means=(0.5, 0.1),
    )
    paired = paired.loc[
        ~((paired["original_index"] == "10000") & (paired["seed"] == 29))
    ]
    selection = _build(tmp_path, records=records, paired=paired)
    frame = selection.frame.set_index("original_index")
    assert frame.at["10000", "selection_status"] == MISSING_REFERENCE_SSCD
    assert frame.at["10001", "selection_status"] == INVALID_RECORD
    assert not bool(frame.at["10000", "include_target_pair"])
    assert not bool(frame.at["10001", "include_target_pair"])


def test_missing_or_nonfinite_selection_scores_never_partially_average_n(
    tmp_path: Path,
) -> None:
    records = _records(labels=("N", "N"))
    paired = _paired(
        records,
        selection_means=(0.10, 0.10),
        reference_validation_means=(0.10, 0.10),
    )
    paired = paired.loc[
        ~((paired["original_index"] == "10000") & (paired["seed"] == 29))
    ].copy()
    paired.loc[
        (paired["original_index"] == "10001") & (paired["seed"] == 28),
        "sscd_cosine_similarity",
    ] = float("inf")

    selection = _build(tmp_path, records=records, paired=paired)
    frame = selection.frame.set_index("original_index")
    for index in ("10000", "10001"):
        assert frame.at[index, "selection_status"] == MISSING_REFERENCE_SSCD
        assert not bool(frame.at[index, "include_target_pair"])
        assert pd.isna(frame.at[index, "selection_mean_sscd"])
        assert pd.isna(frame.at[index, "selection_rule_passes"])
        assert frame.at[index, "target_semantics"] == "n_target_support_unresolved"


def test_incomplete_validation_block_is_diagnostic_only_and_not_averaged(
    tmp_path: Path,
) -> None:
    records = _records(labels=("N",))
    paired = _paired(
        records,
        selection_means=(0.30,),
        reference_validation_means=(0.10,),
    )
    paired = paired.loc[~paired["seed"].eq(39)]
    selection = _build(tmp_path, records=records, paired=paired)
    row = selection.frame.iloc[0]

    assert bool(row["include_target_pair"])
    assert bool(row["selection_rule_passes"])
    assert pd.isna(row["reference_validation_mean_sscd"])
    assert pd.isna(row["reference_validation_rule_passes"])
    assert pd.isna(row["selection_validation_agree"])


def test_duplicate_required_reference_score_is_rejected(tmp_path: Path) -> None:
    records = _records(labels=("N",))
    paired = _paired(
        records,
        selection_means=(0.10,),
        reference_validation_means=(0.10,),
    )
    paired = pd.concat([paired, paired.iloc[[0]]], ignore_index=True)

    with pytest.raises(TargetPairSelectionError, match="Duplicate score for 10000"):
        _build(tmp_path, records=records, paired=paired)


def test_reference_configs_are_strict_and_other_runs_only_load_frozen(
    tmp_path: Path,
) -> None:
    generation, sscd = _reference_configs()
    bad = json.loads(json.dumps(generation))
    bad["scientific_config"]["scheduler"]["name"] = "ddpm"
    bad["scientific_config_hash"] = canonical_hash(bad["scientific_config"])
    records = _records()
    with pytest.raises(TargetPairSelectionError, match="scheduler"):
        build_target_pair_selection(
            tmp_path,
            model_name="sdv1",
            paired_frame=_paired(records),
            records_frame=records,
            reference_run_config=bad,
            sscd_config=sscd,
        )
    frozen = _build(tmp_path)
    assert not is_reference_configuration(
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=1.0,
        num_inference_steps=12,
        num_seeds=3,
    )
    assert load_target_pair_selection(tmp_path, model_name="sdv1").sha256 == frozen.sha256


def test_exact_sorted_otsu_is_diagnostic_and_fixed_threshold_stays_point_25() -> None:
    lower = 0.23537377268075943
    upper = 0.2684394229203463
    values = [lower] * 165 + [upper] * 38
    diagnostic = exact_two_class_otsu(values)
    assert diagnostic is not None
    assert diagnostic.threshold == pytest.approx(0.25190659780055286)
    assert diagnostic.lower_neighbor == lower
    assert diagnostic.upper_neighbor == upper
    assert diagnostic.lower_count == 165
    assert diagnostic.upper_count == 38

    table = build_threshold_diagnostics(
        values,
        total_tv_count=203,
        n_selection_means=(0.10, 0.25, 0.30),
        n_validation_means=(0.30, 0.10, 0.30),
        total_n_count=3,
    )
    assert set(table["category"]) == {"TV", "N"}
    for _, category_rows in table.groupby("category"):
        assert tuple(category_rows["threshold"]) == THRESHOLD_SENSITIVITY
    fixed = table.loc[table["is_fixed_threshold"]].set_index("category")
    assert fixed.at["TV", "threshold"] == 0.25
    assert fixed.at["TV", "comparison_operator"] == ">="
    assert fixed.at["TV", "included_prompt_count"] == 38
    assert fixed.at["TV", "excluded_prompt_count"] == 165
    assert fixed.at["TV", "otsu_threshold"] != fixed.at["TV", "threshold"]
    assert fixed.at["N", "comparison_operator"] == ">="
    assert fixed.at["N", "included_prompt_count"] == 2
    assert fixed.at["N", "excluded_prompt_count"] == 1
    assert fixed.at["N", "selection_validation_agreement_count"] == 1
    assert fixed.at["N", "selection_validation_disagreement_count"] == 2


def test_synthetic_acceptance_fixture_selects_38_tv_and_38_total(
    tmp_path: Path,
) -> None:
    labels = ("N",) * 145 + ("TV",) * 203
    records = _records(labels)
    means = (0.10,) * 145 + (0.23537377268075943,) * 165 + (
        0.2684394229203463,
    ) * 38
    held_out = (0.10,) * 310 + (0.30,) * 38
    paired = _paired(records, selection_means=means, reference_validation_means=held_out)
    selection = _build(tmp_path, records=records, paired=paired)

    assert len(selection.selected_tv_indices) == 38
    assert len(selection.included_indices) == 38
    selected_tv = selection.frame.loc[
        selection.frame["original_index"].isin(selection.selected_tv_indices)
    ]
    assert selected_tv["reference_validation_mean_sscd"].ge(0.25).all()
    summary = json.loads(
        (
            target_pair_selection_directory(tmp_path, model_name="sdv1")
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["selected_tv_prompt_count"] == 38
    assert summary["included_tv_prompt_count"] == 38
    assert summary["excluded_tv_prompt_count"] == 165
    assert summary["included_n_prompt_count"] == 0
    assert summary["excluded_n_prompt_count"] == 145
    assert summary["tv_selection_validation_disagreement_count"] == 0
    assert summary["n_selection_validation_disagreement_count"] == 0
    assert summary["included_prompt_count"] == 38
    assert summary["all_selected_tv_reference_validation_mean_ge_0_25"] is True


def test_supplied_frozen_selection_has_internal_seed20_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = target_pair_selection_directory(root, model_name="sdv1")
    if not directory.is_dir():
        pytest.skip("supplied frozen SDv1 selection is not present")
    supplied_config = json.loads(
        (directory / "config.json").read_text(encoding="utf-8")
    )
    if supplied_config.get("schema_version") in {2, 3}:
        with pytest.raises(StaleTargetPairSelectionError) as captured:
            load_target_pair_selection(root, model_name="sdv1")
        assert f"- {directory}" in str(captured.value)
        return

    selection = load_target_pair_selection(root, model_name="sdv1")
    frame = selection.frame
    config = selection.configuration
    included = frame["include_target_pair"].astype(bool)
    tv = frame["webster_overfit_type_normalized"].eq("TV")
    normal = frame["webster_overfit_type_normalized"].eq("N")
    selected_tv = frame.loc[included & tv]

    assert config["schema_version"] == SELECTION_SCHEMA_VERSION
    assert config["reference_run_name"] == "sdv1_ddim_g7.5_T50_S20_N20"
    assert config["reference_seed_start"] == REFERENCE_SEED_START
    assert config["reference_seeds"] == list(REFERENCE_SEEDS)
    assert config["selection_seeds"] == list(SELECTION_SEEDS)
    assert config["reference_validation_seeds"] == list(REFERENCE_VALIDATION_SEEDS)
    assert config["selection_hash"] == selection.sha256
    assert frame["reference_run_name"].eq(config["reference_run_name"]).all()
    assert frame["selection_hash"].eq(selection.sha256).all()

    expected_included = frozenset(frame.loc[included, "original_index"].astype(str))
    expected_excluded = frozenset(frame.loc[~included, "original_index"].astype(str))
    expected_selected_tv = frozenset(
        frame.loc[included & tv, "original_index"].astype(str)
    )
    assert selection.included_indices == expected_included
    assert selection.excluded_indices == expected_excluded
    assert selection.selected_tv_indices == expected_selected_tv
    assert selected_tv["selection_mean_sscd"].ge(TV_MEAN_SSCD_THRESHOLD).all()
    assert frame.loc[included & normal, "selection_mean_sscd"].ge(
        REFERENCE_SSCD_BOUNDARY
    ).all()
    assert frame.loc[~included & normal, "selection_mean_sscd"].lt(
        REFERENCE_SSCD_BOUNDARY
    ).all()

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_prompt_count"] == len(frame)
    assert summary["included_prompt_count"] == int(included.sum())
    assert summary["excluded_prompt_count"] == int((~included).sum())
    assert summary["selected_tv_prompt_count"] == int((included & tv).sum())
    assert summary["excluded_tv_prompt_count"] == int((~included & tv).sum())
    assert summary["included_n_prompt_count"] == int((included & normal).sum())
    assert summary["excluded_n_prompt_count"] == int((~included & normal).sum())
