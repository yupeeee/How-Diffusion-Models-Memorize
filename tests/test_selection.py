"""Tests for category-blind Spearman prompt selection."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from utils.common.io import canonical_hash
from utils.data import selection as selection_module
from utils.data.selection import (
    DISCARDED_PROXIMITY_RULE,
    INCLUDED_PROXIMITY_RULE,
    SELECTION_COLUMNS,
    SELECTION_POLICY,
    UNUSABLE_REFERENCE_OBSERVATIONS,
    FrozenTargetPairSelectionError,
    TargetPairSelectionError,
    TargetPairSelectionMissingError,
    apply_target_pair_selection,
    build_target_pair_selection,
    is_reference_configuration,
    load_target_pair_selection,
    reference_run_name,
    reference_run_path,
    reference_selection_command,
    spearman_correlation,
    target_pair_selection_directory,
)

DEFAULT_NUM_SEEDS = 3
DEFAULT_SCHEDULER = "ddim"
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 50


def _identity(
    model_name: str = "sdv1",
    *,
    scheduler_name: str = DEFAULT_SCHEDULER,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    num_seeds: int = DEFAULT_NUM_SEEDS,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
    }


def _reference_seeds(num_seeds: int) -> tuple[int, ...]:
    return tuple(range(num_seeds, 2 * num_seeds))


def _configs(
    model_name: str = "sdv1",
    *,
    scheduler_name: str = DEFAULT_SCHEDULER,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    num_seeds: int = DEFAULT_NUM_SEEDS,
) -> tuple[dict[str, object], dict[str, object]]:
    dataset_model = "realisticvision" if model_name == "realvis" else model_name
    reference_seeds = _reference_seeds(num_seeds)
    science: dict[str, object] = {
        "model_cli_name": model_name,
        "dataset_model": dataset_model,
        "scheduler": {"name": scheduler_name},
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
        "seeds": list(reference_seeds),
    }
    generation_hash = canonical_hash(science)
    generation = {
        "scientific_config": science,
        "scientific_config_hash": generation_hash,
    }
    sscd: dict[str, object] = {
        "generation_scientific_config_hash": generation_hash,
        "num_seeds": num_seeds,
        "seeds": list(reference_seeds),
        "sscd_checkpoint_sha256": "b" * 64,
        "sscd_preprocessing_hash": "c" * 64,
    }
    sscd["configuration_hash"] = canonical_hash(sscd)
    return generation, sscd


def _records(
    kinds: tuple[str, ...] = ("MV", "RV", "TV", "N", "other"),
    *,
    model_name: str = "sdv1",
) -> pd.DataFrame:
    dataset_model = "realisticvision" if model_name == "realvis" else model_name
    return pd.DataFrame(
        [
            {
                "model_name": dataset_model,
                "original_index": str(10_000 + position),
                "record_id": f"{model_name}-{position}",
                "source_row_number": position,
                "prompt": f"prompt {position}",
                "kind": kind,
                "target_image_sha256": hashlib.sha256(
                    str(position).encode()
                ).hexdigest(),
            }
            for position, kind in enumerate(kinds)
        ]
    )


def _paired(
    records: pd.DataFrame,
    directions: tuple[str, ...] | None = None,
    *,
    num_seeds: int = DEFAULT_NUM_SEEDS,
) -> pd.DataFrame:
    directions = directions or tuple("negative" for _ in range(len(records)))
    rows: list[dict[str, object]] = []
    for record, direction in zip(
        records.to_dict(orient="records"), directions, strict=True
    ):
        for position, seed in enumerate(_reference_seeds(num_seeds)):
            l2 = float(position + 1)
            if direction == "negative":
                sscd = float(num_seeds - position) / num_seeds
            elif direction == "positive":
                sscd = float(position + 1) / num_seeds
            elif direction == "constant_l2":
                l2, sscd = 1.0, float(position + 1) / num_seeds
            elif direction == "constant_sscd":
                sscd = 0.5
            else:
                raise AssertionError(direction)
            rows.append(
                {
                    **record,
                    "seed": seed,
                    "l2_norm": l2,
                    "sscd": sscd,
                    "observation_status": "complete",
                    "observation_error": "",
                }
            )
    return pd.DataFrame(rows)


def _build(
    root: Path,
    *,
    model_name: str = "sdv1",
    scheduler_name: str = DEFAULT_SCHEDULER,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    records: pd.DataFrame | None = None,
    paired: pd.DataFrame | None = None,
):
    records = _records(model_name=model_name) if records is None else records
    paired = _paired(records, num_seeds=num_seeds) if paired is None else paired
    identity = _identity(
        model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
    )
    generation, sscd = _configs(**identity)
    return build_target_pair_selection(
        root,
        **identity,
        paired_frame=paired,
        records_frame=records,
        reference_run_config=generation,
        sscd_config=sscd,
    )


@pytest.mark.parametrize("num_seeds", (3, 7))
@pytest.mark.parametrize(
    ("scheduler_name", "guidance_scale", "num_inference_steps"),
    (("ddim", 7.5, 50), ("ddpm", 3.25, 17)),
)
def test_reference_contract_is_dynamic_and_has_no_split_or_threshold(
    num_seeds: int,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
) -> None:
    assert SELECTION_POLICY == "prompt_spearman_l2_sscd_lt_zero"
    for obsolete in (
        "REFERENCE_SCHEDULER",
        "REFERENCE_GUIDANCE_SCALE",
        "REFERENCE_NUM_INFERENCE_STEPS",
        "REFERENCE_SEED_START",
        "REFERENCE_NUM_SEEDS",
        "REFERENCE_SEEDS",
        "SELECTION_SEEDS",
        "REFERENCE_VALIDATION_SEEDS",
        "REFERENCE_SSCD_BOUNDARY",
        "SELECTION_SCHEMA_VERSION",
    ):
        assert not hasattr(selection_module, obsolete)

    expected_name = (
        f"sdv1_{scheduler_name}_g{guidance_scale}_T{num_inference_steps}"
        f"_S{num_seeds}_N{num_seeds}"
    )
    expected_path = (
        f"logs/sdv1_{scheduler_name}_g{guidance_scale}_T{num_inference_steps}"
        f"_N{num_seeds}/reference_S{num_seeds}_N{num_seeds}"
    )
    identity = _identity(
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
    )
    assert reference_run_name(**identity) == expected_name
    assert reference_run_path(**identity).as_posix() == expected_path
    assert is_reference_configuration(
        **identity,
        seed_start=num_seeds,
    )
    assert not is_reference_configuration(
        **identity,
        seed_start=num_seeds + 1,
    )


@pytest.mark.parametrize("num_seeds", (0, True, 2**62 + 1))
def test_seed_count_must_leave_room_for_the_full_reference_block(
    num_seeds: object,
) -> None:
    with pytest.raises(
        TargetPairSelectionError, match="valid experiment and reference"
    ):
        reference_run_name(**_identity(num_seeds=num_seeds))
    assert not is_reference_configuration(
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=num_seeds,
        seed_start=num_seeds,
    )


def test_every_kind_uses_the_same_spearman_rule(tmp_path: Path) -> None:
    records = _records()
    selection = _build(
        tmp_path,
        records=records,
        paired=_paired(
            records,
            ("negative", "positive", "negative", "positive", "negative"),
        ),
    )
    prompts = selection.prompt_frame.set_index("original_index")

    assert list(prompts["kind"]) == ["MV", "RV", "TV", "N", "UNKNOWN"]
    assert selection.included_indices == frozenset({"10000", "10002", "10004"})
    assert selection.excluded_indices == frozenset({"10001", "10003"})
    assert prompts.loc["10000", "prompt_spearman"] == pytest.approx(-1.0)
    assert prompts.loc["10001", "prompt_spearman"] == pytest.approx(1.0)
    assert prompts.loc["10000", "selection_status"] == INCLUDED_PROXIMITY_RULE
    assert prompts.loc["10001", "selection_status"] == DISCARDED_PROXIMITY_RULE
    assert prompts.loc["10000", "selection_reason"] == "prompt_spearman_lt_0"
    assert prompts.loc["10001", "selection_reason"] == "prompt_spearman_ge_0"


@pytest.mark.parametrize("num_seeds", (3, 7))
def test_single_csv_is_seed_level_and_points_to_dynamic_montage_tiles(
    tmp_path: Path,
    num_seeds: int,
) -> None:
    records = _records(kinds=("TV",))
    paired = _paired(records, num_seeds=num_seeds)
    selection = _build(tmp_path, num_seeds=num_seeds, records=records, paired=paired)
    directory = target_pair_selection_directory(
        tmp_path, **_identity(num_seeds=num_seeds)
    )

    assert selection.num_seeds == num_seeds
    assert tuple(selection.frame.columns) == SELECTION_COLUMNS
    assert len(selection.frame) == num_seeds
    assert len(selection.prompt_frame) == 1
    assert tuple(selection.frame["seed"]) == _reference_seeds(num_seeds)
    assert tuple(selection.frame["generated_image_tile_index"]) == tuple(
        range(num_seeds)
    )
    assert selection.frame["generated_image_path"].unique().tolist() == [
        f"logs/sdv1_ddim_g7.5_T50_N{num_seeds}/"
        f"reference_S{num_seeds}_N{num_seeds}/image/10000.png"
    ]
    assert selection.frame["prompt"].eq("prompt 0").all()
    assert selection.frame["kind"].eq("TV").all()
    assert selection.frame["l2_norm"].tolist() == pytest.approx(
        list(range(1, num_seeds + 1))
    )
    assert selection.frame["observation_status"].eq("complete").all()
    assert selection.frame["observation_error"].eq("").all()
    assert {path.name for path in directory.iterdir()} == {
        "selection.csv",
        "config.json",
        "summary.json",
    }
    assert not any(path.is_dir() for path in directory.iterdir())

    summary = json.loads((directory / "summary.json").read_text())
    assert summary["complete"] is True
    assert summary["reference_observation_count"] == num_seeds
    assert summary["total_prompt_count"] == 1
    assert summary["included_prompt_count"] == 1
    config = json.loads((directory / "config.json").read_text())
    assert config["reference_seed_start"] == num_seeds
    assert config["reference_num_seeds"] == num_seeds
    assert config["reference_seeds"] == list(_reference_seeds(num_seeds))
    assert "generated_image_tile_columns" not in config
    assert config["selection_metric"] == "spearman(l2_norm,sscd)"
    assert config["include_when"] == "prompt_spearman < 0"
    assert config["kind_affects_selection"] is False
    assert "schema_version" not in config
    assert "threshold" not in config
    assert "category_rules" not in config


def test_all_configured_seeds_affect_one_correlation_and_hash(tmp_path: Path) -> None:
    num_seeds = 7
    records = _records(kinds=("N",))
    first_rows = _paired(records, num_seeds=num_seeds)
    changed_rows = first_rows.copy()
    changed_rows.loc[changed_rows["seed"].eq(2 * num_seeds - 1), "sscd"] = 1.0

    first = _build(
        tmp_path / "first", num_seeds=num_seeds, records=records, paired=first_rows
    )
    changed = _build(
        tmp_path / "changed", num_seeds=num_seeds, records=records, paired=changed_rows
    )
    expected = spearman_correlation(
        changed_rows["l2_norm"].tolist(), changed_rows["sscd"].tolist()
    )

    assert first.prompt_frame.iloc[0]["prompt_spearman"] == pytest.approx(-1.0)
    assert changed.prompt_frame.iloc[0]["prompt_spearman"] == pytest.approx(expected)
    assert changed.sha256 != first.sha256


@pytest.mark.parametrize(
    "case",
    (
        "missing",
        "duplicate",
        "nonfinite_l2",
        "nonfinite_sscd",
        "constant_l2",
        "constant_sscd",
        "reported_failure",
    ),
)
def test_bad_reference_evidence_is_logged_and_never_included(
    tmp_path: Path, case: str
) -> None:
    records = _records(kinds=("MV",))
    paired = _paired(records)
    if case == "missing":
        paired = paired.loc[paired["seed"].ne(DEFAULT_NUM_SEEDS)].reset_index(drop=True)
    elif case == "duplicate":
        paired = pd.concat([paired, paired.iloc[[0]]], ignore_index=True)
    elif case == "nonfinite_l2":
        paired.loc[0, "l2_norm"] = math.inf
    elif case == "nonfinite_sscd":
        paired.loc[0, "sscd"] = math.nan
    elif case == "constant_l2":
        paired["l2_norm"] = 1.0
    elif case == "constant_sscd":
        paired["sscd"] = 0.5
    elif case == "reported_failure":
        paired.loc[0, ["l2_norm", "sscd"]] = math.nan
        paired.loc[0, "observation_status"] = "cache_failure"
        paired.loc[0, "observation_error"] = "trajectory tensor is missing"

    selection = _build(tmp_path, records=records, paired=paired)
    prompt = selection.prompt_frame.iloc[0]

    assert selection.included_indices == frozenset()
    assert selection.excluded_indices == frozenset({"10000"})
    assert prompt["selection_status"] == UNUSABLE_REFERENCE_OBSERVATIONS
    assert math.isnan(prompt["prompt_spearman"])
    assert len(selection.frame) == DEFAULT_NUM_SEEDS
    if case not in {"constant_l2", "constant_sscd"}:
        failures = selection.frame.loc[
            selection.frame["observation_status"].ne("complete")
        ]
        assert len(failures) >= 1
        assert failures["observation_error"].str.len().gt(0).all()


def test_apply_selection_filters_whole_prompts_not_seeds(tmp_path: Path) -> None:
    records = _records(kinds=("TV", "N"))
    selection = _build(
        tmp_path,
        records=records,
        paired=_paired(records, ("negative", "positive")),
    )
    experiment = pd.DataFrame(
        {
            "original_index": ["10000"] * DEFAULT_NUM_SEEDS
            + ["10001"] * DEFAULT_NUM_SEEDS,
            "seed": list(range(DEFAULT_NUM_SEEDS)) * 2,
            "value": list(range(2 * DEFAULT_NUM_SEEDS)),
        }
    )

    selected = apply_target_pair_selection(experiment, selection)
    assert len(selected) == DEFAULT_NUM_SEEDS
    assert set(selected["original_index"]) == {"10000"}
    assert set(selected["seed"]) == set(range(DEFAULT_NUM_SEEDS))

    with pytest.raises(TargetPairSelectionError, match="absent"):
        apply_target_pair_selection(
            pd.DataFrame({"original_index": ["unknown"]}), selection
        )


def test_tied_ranks_use_average_rank_spearman() -> None:
    rho = spearman_correlation([1, 1, 2, 3], [4, 3, 2, 1])
    assert rho == pytest.approx(-0.9486832980505138)
    assert math.isnan(spearman_correlation([1, 1], [2, 3]))
    assert math.isnan(spearman_correlation([1, 2], [3, math.inf]))


def test_frozen_selection_reuses_identical_and_rejects_changed_evidence(
    tmp_path: Path,
) -> None:
    records = _records(kinds=("RV",))
    paired = _paired(records)
    first = _build(tmp_path, records=records, paired=paired)
    before = {
        path.name: path.read_bytes()
        for path in target_pair_selection_directory(tmp_path, **_identity()).iterdir()
    }
    second = _build(tmp_path, records=records, paired=paired)
    assert second.sha256 == first.sha256
    assert before == {
        path.name: path.read_bytes()
        for path in target_pair_selection_directory(tmp_path, **_identity()).iterdir()
    }

    changed = paired.copy()
    changed.loc[0, "sscd"] = 0.001
    with pytest.raises(FrozenTargetPairSelectionError):
        _build(tmp_path, records=records, paired=changed)


def test_load_rejects_tampered_csv_and_summary(tmp_path: Path) -> None:
    _build(tmp_path / "csv")
    directory = target_pair_selection_directory(tmp_path / "csv", **_identity())
    frame = pd.read_csv(directory / "selection.csv")
    frame.loc[0, "l2_norm"] += 1.0
    frame.to_csv(directory / "selection.csv", index=False)
    with pytest.raises(TargetPairSelectionError):
        load_target_pair_selection(tmp_path / "csv", **_identity())

    _build(tmp_path / "summary")
    directory = target_pair_selection_directory(tmp_path / "summary", **_identity())
    summary = json.loads((directory / "summary.json").read_text())
    summary["complete"] = False
    (directory / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(TargetPairSelectionError, match="summary"):
        load_target_pair_selection(tmp_path / "summary", **_identity())


def test_reference_configuration_and_seed_range_are_strict(tmp_path: Path) -> None:
    records = _records(kinds=("TV",))
    generation, sscd = _configs()
    science = dict(generation["scientific_config"])
    science["seeds"] = list(range(DEFAULT_NUM_SEEDS + 1, 2 * DEFAULT_NUM_SEEDS + 1))
    generation["scientific_config"] = science
    generation["scientific_config_hash"] = canonical_hash(science)
    sscd["generation_scientific_config_hash"] = generation["scientific_config_hash"]
    sscd.pop("configuration_hash")
    sscd["configuration_hash"] = canonical_hash(sscd)
    with pytest.raises(TargetPairSelectionError, match="generation differs"):
        build_target_pair_selection(
            tmp_path,
            **_identity(),
            paired_frame=_paired(records),
            records_frame=records,
            reference_run_config=generation,
            sscd_config=sscd,
        )

    outside = _paired(records)
    outside.loc[0, "seed"] = 0
    with pytest.raises(
        TargetPairSelectionError,
        match=f"outside {DEFAULT_NUM_SEEDS}--{2 * DEFAULT_NUM_SEEDS - 1}",
    ):
        _build(tmp_path / "outside", records=records, paired=outside)


def test_csv_inputs_and_empty_prompt_round_trip(tmp_path: Path) -> None:
    records = _records(kinds=("normal",))
    records.loc[0, "prompt"] = ""
    paired = _paired(records)
    records_path, paired_path = tmp_path / "records.csv", tmp_path / "paired.csv"
    records.to_csv(records_path, index=False)
    paired.to_csv(paired_path, index=False)
    generation, sscd = _configs()

    selection = build_target_pair_selection(
        tmp_path,
        **_identity(),
        paired_frame=paired_path,
        records_frame=records_path,
        reference_run_config=generation,
        sscd_config=sscd,
    )
    assert selection.prompt_frame.iloc[0]["prompt"] == ""
    assert selection.prompt_frame.iloc[0]["kind"] == "N"
    assert (
        load_target_pair_selection(tmp_path, **_identity()).sha256 == selection.sha256
    )


def test_missing_selection_error_names_exact_reference_commands(tmp_path: Path) -> None:
    num_seeds = 7
    identity = _identity(
        "sdv2",
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=17,
        num_seeds=num_seeds,
    )
    expected = (
        "./generate.sh --model sdv2 --scheduler ddpm --g 3.25 --T 17 "
        "--N 7 --seed-start 7\n"
        "./sscd.sh --model sdv2 --scheduler ddpm --g 3.25 --T 17 "
        "--N 7 --seed-start 7\n"
        "./compute_proximity.sh --model sdv2 --scheduler ddpm --g 3.25 "
        "--T 17 --N 7 --seed-start 7"
    )
    assert reference_selection_command(**identity) == expected
    with pytest.raises(TargetPairSelectionMissingError) as captured:
        load_target_pair_selection(tmp_path, **identity)
    assert expected in str(captured.value)


def test_model_specific_frozen_directories_are_distinct(tmp_path: Path) -> None:
    sdv1 = _build(
        tmp_path,
        model_name="sdv1",
        records=_records(kinds=("MV",), model_name="sdv1"),
    )
    realvis_records = _records(kinds=("MV",), model_name="realvis")
    realvis = _build(
        tmp_path,
        model_name="realvis",
        records=realvis_records,
        paired=_paired(realvis_records),
    )
    assert sdv1.root == realvis.root
    assert sdv1.sha256 != realvis.sha256
    assert target_pair_selection_directory(
        tmp_path, **_identity("sdv1")
    ) != target_pair_selection_directory(tmp_path, **_identity("realvis"))


def test_sampler_specific_frozen_directories_and_hashes_are_distinct(
    tmp_path: Path,
) -> None:
    default = _build(tmp_path)
    alternate_identity = _identity(
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=17,
    )
    alternate = _build(
        tmp_path,
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=17,
    )

    assert default.sha256 != alternate.sha256
    assert target_pair_selection_directory(
        tmp_path, **_identity()
    ) != target_pair_selection_directory(tmp_path, **alternate_identity)
    assert load_target_pair_selection(tmp_path, **_identity()).sha256 == default.sha256
    assert (
        load_target_pair_selection(tmp_path, **alternate_identity).sha256
        == alternate.sha256
    )
