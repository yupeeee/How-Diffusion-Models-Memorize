"""Freeze prompt selection from terminal-L2/SSCD proximity evidence.

Selection uses all reference seeds N--2N-1, where N is the experiment seed
count. A prompt is included exactly when
its within-prompt Spearman correlation between L2 distance and SSCD is finite
and negative. Webster kind is audit metadata and never affects the decision.
"""

from __future__ import annotations

import math
import operator
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from utils.common.cli import (
    generation_cache_namespace,
    generation_cache_parent_name,
    generation_run_name,
    stable_float,
    validate_seed_block,
)
from utils.common.io import (
    atomic_write_frame_csv,
    atomic_write_json,
    canonical_hash,
    read_json,
)
from utils.data.webster import normalize_webster_type
from utils.experiments.cache import generation_log_relative_path

SELECTION_POLICY = "prompt_spearman_l2_sscd_lt_zero"
INCLUDED_PROXIMITY_RULE = "included_proximity_rule"
DISCARDED_PROXIMITY_RULE = "discarded_proximity_rule"
UNUSABLE_REFERENCE_OBSERVATIONS = "unusable_reference_observations"

SELECTION_COLUMNS = tuple(
    "model_name original_index record_id source_row_number seed prompt kind "
    "target_image_sha256 generated_image_path generated_image_tile_index "
    "l2_norm sscd observation_status observation_error prompt_spearman include_prompt "
    "selection_status selection_reason reference_run_name "
    "reference_generation_hash reference_sscd_hash selection_policy selection_hash".split()
)
_IDENTITY_COLUMNS = tuple(
    "original_index record_id source_row_number prompt kind target_image_sha256".split()
)
_MODELS = {"sdv1": "sdv1", "sdv2": "sdv2", "realvis": "realisticvision"}
_HASH = re.compile(r"[0-9a-f]{64}")
_FILES = ("selection.csv", "config.json", "summary.json")


class TargetPairSelectionError(RuntimeError):
    """Selection evidence or a frozen selection is invalid."""


class TargetPairSelectionMissingError(TargetPairSelectionError):
    """The required frozen selection does not exist."""


class FrozenTargetPairSelectionError(TargetPairSelectionError):
    """New evidence conflicts with an existing frozen selection."""


@dataclass(frozen=True, slots=True)
class TargetPairSelection:
    """A validated seed-level selection audit."""

    root: Path
    model_name: str
    scheduler_name: str
    guidance_scale: float
    num_inference_steps: int
    num_seeds: int
    frame: pd.DataFrame
    configuration: Mapping[str, object]
    sha256: str

    @property
    def prompt_frame(self) -> pd.DataFrame:
        return (
            self.frame.drop_duplicates("original_index").reset_index(drop=True).copy()
        )

    @property
    def included_indices(self) -> frozenset[str]:
        rows = self.prompt_frame
        return frozenset(rows.loc[rows["include_prompt"], "original_index"].astype(str))

    @property
    def excluded_indices(self) -> frozenset[str]:
        rows = self.prompt_frame
        return frozenset(
            rows.loc[~rows["include_prompt"], "original_index"].astype(str)
        )


FrameInput = pd.DataFrame | str | Path
ConfigurationInput = Mapping[str, object] | str | Path


def is_reference_configuration(
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int,
) -> bool:
    """Return whether CLI values identify the N-seed reference run."""

    try:
        _, _, _, _, count = _reference_values(
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            num_seeds,
        )
        first_seed = operator.index(seed_start)
    except (TargetPairSelectionError, TypeError, ValueError, OverflowError):
        return False
    return not isinstance(seed_start, bool) and first_seed == count


def reference_run_name(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> str:
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    return generation_run_name(
        model,
        scheduler,
        guidance,
        steps,
        count,
        seed_start=count,
    )


def reference_run_path(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> Path:
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    return generation_log_relative_path(
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        seed_start=count,
    )


def reference_selection_command(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> str:
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    common = (
        f"--model {model} --scheduler {scheduler} --g {stable_float(guidance)} "
        f"--T {steps} "
        f"--N {count} --seed-start {count}"
    )
    return (
        f"./generate.sh {common}\n./sscd.sh {common}\n./compute_proximity.sh {common}"
    )


def target_pair_selection_directory(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> Path:
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    parent = generation_cache_parent_name(model, scheduler, guidance, steps, count)
    namespace = generation_cache_namespace(
        model, scheduler, guidance, steps, count, seed_start=count
    )
    return (
        Path(root).expanduser().resolve()
        / "data/webster/selection"
        / _MODELS[model]
        / parent
        / namespace
    )


def build_target_pair_selection(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    paired_frame: FrameInput,
    reference_run_config: ConfigurationInput | None = None,
    sscd_config: ConfigurationInput | None = None,
    records_frame: FrameInput | None = None,
) -> TargetPairSelection:
    """Build and atomically freeze the seed-level audit."""

    project_root = Path(root).expanduser().resolve()
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    run = project_root / reference_run_path(model, scheduler, guidance, steps, count)
    generation = _read_config(
        reference_run_config, run / "run_config.json", "generation config"
    )
    sscd = _read_config(sscd_config, run / "sscd_config.json", "SSCD config")
    generation_hash = _validate_generation(
        generation, model, scheduler, guidance, steps, count
    )
    sscd_hash = _validate_sscd(sscd, generation_hash, count)

    paired = _read_frame(paired_frame, "paired reference observations")
    source = paired if records_frame is None else _read_frame(records_frame, "records")
    records = _prompt_records(source, derived=records_frame is None, model=model)
    observations = _observations(
        paired, set(records["original_index"]), num_seeds=count
    )
    manifest = _manifest(
        records,
        observations,
        model,
        scheduler,
        guidance,
        steps,
        generation_hash,
        sscd_hash,
        count,
    )
    digest = compute_target_pair_selection_hash(
        manifest,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        reference_generation_hash=generation_hash,
        reference_sscd_hash=sscd_hash,
    )
    manifest["selection_hash"] = digest
    config = _selection_config(
        model,
        scheduler,
        guidance,
        steps,
        count,
        generation_hash,
        sscd_hash,
        sscd["sscd_checkpoint_sha256"],
        sscd["sscd_preprocessing_hash"],
        digest,
    )
    selection = TargetPairSelection(
        root=project_root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        frame=manifest,
        configuration=config,
        sha256=digest,
    )
    validate_target_pair_selection(selection)

    destination = target_pair_selection_directory(
        project_root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
    )
    if destination.exists() or destination.is_symlink():
        frozen = load_target_pair_selection(
            project_root,
            model_name=model,
            scheduler_name=scheduler,
            guidance_scale=guidance,
            num_inference_steps=steps,
            num_seeds=count,
        )
        if frozen.sha256 != digest:
            raise FrozenTargetPairSelectionError(
                f"Reference evidence would change frozen selection {destination}: "
                f"{frozen.sha256} != {digest}"
            )
        return frozen
    _write_selection(selection, destination)
    return load_target_pair_selection(
        project_root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
    )


def load_target_pair_selection(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> TargetPairSelection:
    """Load and fully validate the three frozen artifacts."""

    project_root = Path(root).expanduser().resolve()
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    directory = target_pair_selection_directory(
        project_root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
    )
    if not directory.exists():
        _raise_missing(project_root, model, scheduler, guidance, steps, count)
    if not directory.is_dir() or directory.is_symlink():
        raise TargetPairSelectionError(f"Invalid selection directory: {directory}")
    missing = [name for name in _FILES if not (directory / name).is_file()]
    if missing:
        raise TargetPairSelectionError(
            f"Incomplete frozen selection {directory}; missing: {', '.join(missing)}"
        )
    if any((directory / name).is_symlink() for name in _FILES):
        raise TargetPairSelectionError(
            "Frozen selection artifacts must not be symlinks"
        )
    try:
        frame = pd.read_csv(
            directory / "selection.csv",
            dtype={
                "original_index": str,
                "record_id": str,
                "prompt": str,
                "kind": str,
                "target_image_sha256": str,
                "generated_image_path": str,
            },
            keep_default_na=False,
            float_precision="round_trip",
        )
        config = read_json(directory / "config.json")
        summary = read_json(directory / "summary.json")
    except (OSError, RuntimeError, ValueError) as error:
        raise TargetPairSelectionError(
            f"Cannot read selection {directory}: {error}"
        ) from error
    frame = _normalize_csv(frame)
    digest = config.get("selection_hash")
    if not _is_hash(digest):
        raise TargetPairSelectionError("config.json selection_hash is invalid")
    selection = TargetPairSelection(
        root=project_root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        frame=frame,
        configuration=config,
        sha256=str(digest),
    )
    validate_target_pair_selection(selection)
    if summary != _summary(selection):
        raise TargetPairSelectionError("Frozen selection summary is inconsistent")
    return selection


def validate_target_pair_selection(selection: TargetPairSelection) -> None:
    """Validate fixed provenance, seed groups, decisions, and content hash."""

    model, scheduler, guidance, steps, count = _reference_values(
        selection.model_name,
        selection.scheduler_name,
        selection.guidance_scale,
        selection.num_inference_steps,
        selection.num_seeds,
    )
    frame, config = selection.frame, selection.configuration
    if tuple(frame.columns) != SELECTION_COLUMNS or frame.empty:
        raise TargetPairSelectionError("selection.csv has an invalid schema")
    if not pd.api.types.is_bool_dtype(frame["include_prompt"]):
        raise TargetPairSelectionError("include_prompt must contain booleans")
    expected = _selection_config(
        model,
        scheduler,
        guidance,
        steps,
        count,
        str(config.get("reference_generation_hash")),
        str(config.get("reference_sscd_hash")),
        config.get("sscd_checkpoint_sha256"),
        config.get("sscd_preprocessing_hash"),
        selection.sha256,
    )
    if dict(config) != expected:
        raise TargetPairSelectionError(
            "config.json does not match the selection contract"
        )
    if any(
        not _is_hash(config[key])
        for key in (
            "reference_generation_hash",
            "reference_sscd_hash",
            "sscd_checkpoint_sha256",
            "sscd_preprocessing_hash",
            "selection_hash",
        )
    ):
        raise TargetPairSelectionError("Selection provenance hash is invalid")
    fixed = {
        "model_name": model,
        "reference_run_name": reference_run_name(
            model, scheduler, guidance, steps, count
        ),
        "reference_generation_hash": config["reference_generation_hash"],
        "reference_sscd_hash": config["reference_sscd_hash"],
        "selection_policy": SELECTION_POLICY,
        "selection_hash": selection.sha256,
    }
    if any(not frame[column].eq(value).all() for column, value in fixed.items()):
        raise TargetPairSelectionError("Selection row provenance is inconsistent")
    expected_order = frame.sort_values(
        ["source_row_number", "original_index", "seed"], kind="stable"
    ).reset_index(drop=True)
    if not frame.reset_index(drop=True).equals(expected_order):
        raise TargetPairSelectionError("selection.csv rows are not canonically ordered")
    for index, group in frame.groupby("original_index", sort=False):
        _validate_prompt_group(
            str(index), group, model, scheduler, guidance, steps, count
        )
    digest = compute_target_pair_selection_hash(
        frame,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        reference_generation_hash=str(config["reference_generation_hash"]),
        reference_sscd_hash=str(config["reference_sscd_hash"]),
    )
    if digest != selection.sha256:
        raise TargetPairSelectionError(
            f"Selection hash mismatch: {selection.sha256} != {digest}"
        )


def apply_target_pair_selection(
    frame: pd.DataFrame, selection: TargetPairSelection
) -> pd.DataFrame:
    """Keep every input row belonging to an included prompt."""

    if "original_index" not in frame:
        raise TargetPairSelectionError("Table is missing original_index")
    result = frame.copy(deep=True)
    indices = result["original_index"].map(_index)
    known = set(selection.prompt_frame["original_index"])
    unknown = sorted(set(indices) - known)
    if unknown:
        raise TargetPairSelectionError(
            "Prompts absent from selection: " + ", ".join(unknown[:5])
        )
    return result.loc[indices.isin(selection.included_indices)].reset_index(drop=True)


def compute_target_pair_selection_hash(
    frame: pd.DataFrame,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    reference_generation_hash: str,
    reference_sscd_hash: str,
) -> str:
    """Hash every seed observation and deterministic prompt decision."""

    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    reference_seeds = _reference_seeds(count)
    if not _is_hash(reference_generation_hash) or not _is_hash(reference_sscd_hash):
        raise TargetPairSelectionError("Selection provenance hash is invalid")
    columns = tuple(
        column for column in SELECTION_COLUMNS if column != "selection_hash"
    )
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise TargetPairSelectionError("Cannot hash without: " + ", ".join(missing))
    rows = [
        {column: _json(row[column]) for column in columns}
        for row in frame.sort_values(
            ["source_row_number", "original_index", "seed"], kind="stable"
        )
        .loc[:, columns]
        .to_dict(orient="records")
    ]
    return canonical_hash(
        {
            "selection_policy": SELECTION_POLICY,
            "model_name": model,
            "dataset_model_name": _MODELS[model],
            "reference_scheduler": scheduler,
            "reference_guidance_scale": guidance,
            "reference_num_inference_steps": steps,
            "reference_run_name": reference_run_name(
                model, scheduler, guidance, steps, count
            ),
            "reference_generation_hash": reference_generation_hash,
            "reference_sscd_hash": reference_sscd_hash,
            "reference_seeds": list(reference_seeds),
            "selection_metric": "spearman(l2_norm,sscd)",
            "include_when": "prompt_spearman < 0",
            "rows": rows,
        }
    )


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Spearman rho as Pearson correlation of average ranks."""

    if len(left) != len(right) or len(left) < 2:
        return math.nan
    values = [float(value) for value in (*left, *right)]
    if not all(math.isfinite(value) for value in values):
        return math.nan
    left_rank = pd.Series(left, dtype="float64").rank(method="average").tolist()
    right_rank = pd.Series(right, dtype="float64").rank(method="average").tolist()
    left_mean = math.fsum(left_rank) / len(left_rank)
    right_mean = math.fsum(right_rank) / len(right_rank)
    left_centered = [value - left_mean for value in left_rank]
    right_centered = [value - right_mean for value in right_rank]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return math.nan
    return (
        math.fsum(x * y for x, y in zip(left_centered, right_centered, strict=True))
        / denominator
    )


def _manifest(
    records: pd.DataFrame,
    observations: Mapping[str, Mapping[int, Mapping[str, object]]],
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    generation_hash: str,
    sscd_hash: str,
    num_seeds: int,
) -> pd.DataFrame:
    count = _num_seeds(num_seeds)
    reference_seeds = _reference_seeds(count)
    rows: list[dict[str, object]] = []
    for record in records.to_dict(orient="records"):
        index = str(record["original_index"])
        per_seed = []
        for tile, seed in enumerate(reference_seeds):
            observed = observations.get(index, {}).get(
                seed,
                {
                    "l2_norm": math.nan,
                    "sscd": math.nan,
                    "observation_status": "missing",
                    "observation_error": "reference observation is missing",
                },
            )
            per_seed.append(
                {"seed": seed, "generated_image_tile_index": tile, **observed}
            )
        rho, include, status, reason = _decision(per_seed)
        common = {
            "model_name": model,
            **{column: record[column] for column in _IDENTITY_COLUMNS},
            "generated_image_path": (
                reference_run_path(model, scheduler, guidance, steps, count)
                / "image"
                / f"{index}.png"
            ).as_posix(),
            "prompt_spearman": rho,
            "include_prompt": include,
            "selection_status": status,
            "selection_reason": reason,
            "reference_run_name": reference_run_name(
                model, scheduler, guidance, steps, count
            ),
            "reference_generation_hash": generation_hash,
            "reference_sscd_hash": sscd_hash,
            "selection_policy": SELECTION_POLICY,
            "selection_hash": "",
        }
        rows.extend({**common, **observed} for observed in per_seed)
    return pd.DataFrame(rows, columns=SELECTION_COLUMNS).sort_values(
        ["source_row_number", "original_index", "seed"],
        kind="stable",
        ignore_index=True,
    )


def _decision(
    observations: Sequence[Mapping[str, object]],
) -> tuple[float, bool, str, str]:
    issues = sorted(
        {
            str(row["observation_status"])
            for row in observations
            if row["observation_status"] != "complete"
        }
    )
    if issues:
        return (
            math.nan,
            False,
            UNUSABLE_REFERENCE_OBSERVATIONS,
            "invalid_reference_observations:" + ",".join(issues),
        )
    l2 = [float(row["l2_norm"]) for row in observations]
    sscd = [float(row["sscd"]) for row in observations]
    if len(set(l2)) == 1:
        return math.nan, False, UNUSABLE_REFERENCE_OBSERVATIONS, "constant_l2_norm"
    if len(set(sscd)) == 1:
        return math.nan, False, UNUSABLE_REFERENCE_OBSERVATIONS, "constant_sscd"
    rho = spearman_correlation(l2, sscd)
    if not math.isfinite(rho):
        return (
            math.nan,
            False,
            UNUSABLE_REFERENCE_OBSERVATIONS,
            "undefined_prompt_spearman",
        )
    if rho < 0.0:
        return rho, True, INCLUDED_PROXIMITY_RULE, "prompt_spearman_lt_0"
    return rho, False, DISCARDED_PROXIMITY_RULE, "prompt_spearman_ge_0"


def _prompt_records(source: pd.DataFrame, *, derived: bool, model: str) -> pd.DataFrame:
    missing = sorted(set(_IDENTITY_COLUMNS) - set(source.columns))
    if missing:
        suffix = " or provide records_frame" if derived else ""
        raise TargetPairSelectionError(
            "Prompt identity lacks: " + ", ".join(missing) + suffix
        )
    if "model_name" in source:
        observed = {str(value) for value in source["model_name"] if not _missing(value)}
        if not observed or not observed.issubset({model, _MODELS[model]}):
            raise TargetPairSelectionError(f"Prompt records do not belong to {model}")
    frame = source.loc[:, _IDENTITY_COLUMNS].copy()
    frame["original_index"] = frame["original_index"].map(_index)
    if derived:
        rows = []
        for index, group in frame.groupby("original_index", sort=False):
            row = {"original_index": index}
            for column in _IDENTITY_COLUMNS[1:]:
                values = [_scalar(value) for value in group[column]]
                if len({repr(value) for value in values}) != 1:
                    raise TargetPairSelectionError(
                        f"Prompt identity differs for {index}: {column}"
                    )
                row[column] = values[0]
            rows.append(row)
        frame = pd.DataFrame(rows, columns=_IDENTITY_COLUMNS)
    elif frame["original_index"].duplicated().any():
        raise TargetPairSelectionError("records_frame must have one row per prompt")
    for row in frame.to_dict(orient="records"):
        if not isinstance(row["record_id"], str) or not row["record_id"].strip():
            raise TargetPairSelectionError("record_id must be a non-empty string")
        if not isinstance(row["prompt"], str):
            raise TargetPairSelectionError("prompt must be a string")
        if _integer(row["source_row_number"], "source_row_number") < 0:
            raise TargetPairSelectionError("source_row_number must be non-negative")
        if not _is_hash(row["target_image_sha256"]):
            raise TargetPairSelectionError("target_image_sha256 is invalid")
    frame["source_row_number"] = frame["source_row_number"].map(
        lambda value: _integer(value, "source_row_number")
    )
    frame["kind"] = frame["kind"].map(normalize_webster_type)
    return frame.sort_values(
        ["source_row_number", "original_index"], kind="stable", ignore_index=True
    )


def _observations(
    paired: pd.DataFrame,
    known: set[str],
    *,
    num_seeds: int,
) -> dict[str, dict[int, dict[str, object]]]:
    reference_seeds = frozenset(_reference_seeds(num_seeds))
    required = {
        "original_index",
        "seed",
        "l2_norm",
        "sscd",
        "observation_status",
        "observation_error",
    }
    missing = sorted(required - set(paired.columns))
    if missing:
        raise TargetPairSelectionError(
            "Paired reference observations lack: " + ", ".join(missing)
        )
    grouped: dict[str, dict[int, list[Mapping[str, object]]]] = {}
    for row in paired.loc[:, sorted(required)].to_dict(orient="records"):
        index = _index(row["original_index"])
        if index not in known:
            raise TargetPairSelectionError(f"Unknown paired prompt {index}")
        seed = _integer(row["seed"], "seed")
        if seed not in reference_seeds:
            first, last = min(reference_seeds), max(reference_seeds)
            raise TargetPairSelectionError(
                f"Reference seed outside {first}--{last}: {seed}"
            )
        grouped.setdefault(index, {}).setdefault(seed, []).append(row)
    result: dict[str, dict[int, dict[str, object]]] = {}
    for index, seeds in grouped.items():
        result[index] = {}
        for seed, rows in seeds.items():
            if len(rows) != 1:
                result[index][seed] = {
                    "l2_norm": math.nan,
                    "sscd": math.nan,
                    "observation_status": "duplicate",
                    "observation_error": (
                        f"duplicate reference observations for prompt {index}, seed {seed}"
                    ),
                }
                continue
            row = rows[0]
            source_status = str(row["observation_status"]).strip()
            source_error = str(row["observation_error"]).strip()
            if source_status != "complete":
                result[index][seed] = {
                    "l2_norm": math.nan,
                    "sscd": math.nan,
                    "observation_status": source_status or "failed",
                    "observation_error": source_error or "reference observation failed",
                }
                continue
            l2, l2_issue = _metric(row["l2_norm"], "l2_norm")
            sscd, sscd_issue = _metric(row["sscd"], "sscd")
            issues = [issue for issue in (l2_issue, sscd_issue) if issue]
            result[index][seed] = {
                "l2_norm": l2,
                "sscd": sscd,
                "observation_status": "complete" if not issues else "+".join(issues),
                "observation_error": "" if not issues else ",".join(issues),
            }
    return result


def _metric(value: object, name: str) -> tuple[float, str | None]:
    if _missing(value) or value == "":
        return math.nan, f"missing_{name}"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan, f"nonnumeric_{name}"
    if not math.isfinite(number):
        return math.nan, f"nonfinite_{name}"
    if name == "l2_norm" and number < 0:
        return math.nan, "negative_l2_norm"
    if name == "sscd" and not -1.00001 <= number <= 1.00001:
        return math.nan, "out_of_range_sscd"
    return number, None


def _validate_prompt_group(
    index: str,
    group: pd.DataFrame,
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    num_seeds: int,
) -> None:
    count = _num_seeds(num_seeds)
    if len(group) != count:
        raise TargetPairSelectionError(
            f"Prompt {index} must contain exactly {count} rows"
        )
    seeds = tuple(_integer(value, "seed") for value in group["seed"])
    tiles = tuple(
        _integer(value, "tile") for value in group["generated_image_tile_index"]
    )
    if seeds != _reference_seeds(count) or tiles != tuple(range(count)):
        raise TargetPairSelectionError(
            f"Prompt {index} seed or tile sequence is invalid"
        )
    image = (
        reference_run_path(model, scheduler, guidance, steps, count)
        / "image"
        / f"{index}.png"
    ).as_posix()
    if not group["generated_image_path"].eq(image).all():
        raise TargetPairSelectionError(f"Prompt {index} image path is invalid")
    constant = (
        "record_id",
        "source_row_number",
        "prompt",
        "kind",
        "target_image_sha256",
        "prompt_spearman",
        "include_prompt",
        "selection_status",
        "selection_reason",
    )
    for column in constant:
        values = [_json(value) for value in group[column]]
        if any(value != values[0] for value in values[1:]):
            raise TargetPairSelectionError(f"Prompt {index} has inconsistent {column}")
    if group.iloc[0]["kind"] not in {"MV", "RV", "TV", "N", "UNKNOWN"}:
        raise TargetPairSelectionError(f"Prompt {index} kind is invalid")
    observations = []
    for row in group.to_dict(orient="records"):
        status = str(row["observation_status"])
        error = str(row["observation_error"])
        l2 = _optional_float(row["l2_norm"], "l2_norm")
        sscd = _optional_float(row["sscd"], "sscd")
        if status == "complete" and (math.isnan(l2) or math.isnan(sscd) or error):
            raise TargetPairSelectionError(
                f"Prompt {index} has an invalid complete row"
            )
        if status != "complete" and not error:
            raise TargetPairSelectionError(
                f"Prompt {index} has an unexplained failed row"
            )
        observations.append({"l2_norm": l2, "sscd": sscd, "observation_status": status})
    rho, include, status, reason = _decision(observations)
    stored_rho = _optional_float(group.iloc[0]["prompt_spearman"], "prompt_spearman")
    first = group.iloc[0]
    if not _same_float(rho, stored_rho):
        raise TargetPairSelectionError(f"Prompt {index} Spearman value is inconsistent")
    if (
        bool(first["include_prompt"]) != include
        or first["selection_status"] != status
        or first["selection_reason"] != reason
    ):
        raise TargetPairSelectionError(f"Prompt {index} decision is inconsistent")


def _selection_config(
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    num_seeds: int,
    generation_hash: str,
    sscd_hash: str,
    checkpoint_hash: object,
    preprocessing_hash: object,
    selection_hash: str,
) -> dict[str, object]:
    count = _num_seeds(num_seeds)
    reference_seeds = _reference_seeds(count)
    return {
        "selection_policy": SELECTION_POLICY,
        "model_name": model,
        "dataset_model_name": _MODELS[model],
        "reference_run_name": reference_run_name(
            model, scheduler, guidance, steps, count
        ),
        "reference_run_path": reference_run_path(
            model, scheduler, guidance, steps, count
        ).as_posix(),
        "reference_scheduler": scheduler,
        "reference_guidance_scale": guidance,
        "reference_num_inference_steps": steps,
        "reference_seed_start": count,
        "reference_num_seeds": count,
        "reference_seeds": list(reference_seeds),
        "selection_metric": "spearman(l2_norm,sscd)",
        "include_when": "prompt_spearman < 0",
        "decision_scope": "whole_prompt",
        "kind_affects_selection": False,
        "unusable_evidence_is_included": False,
        "generated_image_path_base": "project_root",
        "generated_image_tile_order": "zero_based_row_major_reference_seed_order",
        "reference_generation_hash": generation_hash,
        "reference_sscd_hash": sscd_hash,
        "sscd_checkpoint_sha256": checkpoint_hash,
        "sscd_preprocessing_hash": preprocessing_hash,
        "selection_hash": selection_hash,
    }


def _validate_generation(
    config: Mapping[str, object],
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    num_seeds: int,
) -> str:
    count = _num_seeds(num_seeds)
    science, digest = (
        config.get("scientific_config"),
        config.get("scientific_config_hash"),
    )
    if (
        not isinstance(science, Mapping)
        or not _is_hash(digest)
        or canonical_hash(science) != digest
    ):
        raise TargetPairSelectionError(
            "Reference generation scientific hash is invalid"
        )
    scheduler_metadata = science.get("scheduler")
    scheduler_name = (
        scheduler_metadata.get("name")
        if isinstance(scheduler_metadata, Mapping)
        else scheduler_metadata
    )
    expected = {
        "model_cli_name": model,
        "dataset_model": _MODELS[model],
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "num_seeds": count,
        "seeds": list(_reference_seeds(count)),
    }
    wrong = [key for key, value in expected.items() if science.get(key) != value]
    if scheduler_name != scheduler:
        wrong.append("scheduler")
    if wrong:
        raise TargetPairSelectionError(
            "Reference generation differs at: " + ", ".join(wrong)
        )
    return str(digest)


def _validate_sscd(
    config: Mapping[str, object], generation_hash: str, num_seeds: int
) -> str:
    from utils.experiments.sscd import sscd_configuration_hash

    count = _num_seeds(num_seeds)

    digest = config.get("configuration_hash")
    if not _is_hash(digest) or sscd_configuration_hash(config) != digest:
        raise TargetPairSelectionError("Reference SSCD configuration hash is invalid")
    expected = {
        "generation_scientific_config_hash": generation_hash,
        "num_seeds": count,
        "seeds": list(_reference_seeds(count)),
    }
    wrong = [key for key, value in expected.items() if config.get(key) != value]
    if wrong:
        raise TargetPairSelectionError("Reference SSCD differs at: " + ", ".join(wrong))
    if any(
        not _is_hash(config.get(key))
        for key in ("sscd_checkpoint_sha256", "sscd_preprocessing_hash")
    ):
        raise TargetPairSelectionError("Reference SSCD provenance hash is invalid")
    return str(digest)


def _write_selection(selection: TargetPairSelection, directory: Path) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.", suffix=".tmp", dir=directory.parent
        )
    )
    try:
        atomic_write_frame_csv(selection.frame, temporary / "selection.csv")
        atomic_write_json(temporary / "config.json", selection.configuration)
        atomic_write_json(temporary / "summary.json", _summary(selection))
        descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, directory)
        descriptor = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _summary(selection: TargetPairSelection) -> dict[str, object]:
    prompts = selection.prompt_frame
    included = prompts["include_prompt"]
    unusable = prompts["selection_status"].eq(UNUSABLE_REFERENCE_OBSERVATIONS)
    return {
        "complete": True,
        "selection_policy": SELECTION_POLICY,
        "model_name": selection.model_name,
        "selection_hash": selection.sha256,
        "reference_generation_hash": selection.configuration[
            "reference_generation_hash"
        ],
        "reference_sscd_hash": selection.configuration["reference_sscd_hash"],
        "reference_observation_count": len(selection.frame),
        "total_prompt_count": len(prompts),
        "included_prompt_count": int(included.sum()),
        "discarded_prompt_count": int((~included & ~unusable).sum()),
        "unusable_prompt_count": int(unusable.sum()),
    }


def _read_config(
    source: ConfigurationInput | None, default: Path, label: str
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = default if source is None else Path(source)
    if not path.is_file():
        raise TargetPairSelectionError(f"Missing {label}: {path}")
    try:
        return read_json(path)
    except RuntimeError as error:
        raise TargetPairSelectionError(f"Invalid {label}: {error}") from error


def _read_frame(source: FrameInput, label: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)
    path = Path(source)
    if not path.is_file() or path.suffix.casefold() != ".csv":
        raise TargetPairSelectionError(f"{label} must be a DataFrame or CSV")
    try:
        return pd.read_csv(
            path,
            dtype={"original_index": str},
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (OSError, ValueError) as error:
        raise TargetPairSelectionError(
            f"Cannot read {label} {path}: {error}"
        ) from error


def _normalize_csv(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != SELECTION_COLUMNS:
        raise TargetPairSelectionError("selection.csv has an invalid schema")
    result = frame.copy()
    for column in ("source_row_number", "seed", "generated_image_tile_index"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    for column in ("l2_norm", "sscd", "prompt_spearman"):
        result[column] = result[column].map(
            lambda value, name=column: _optional_float(value, name)
        )
    if not pd.api.types.is_bool_dtype(result["include_prompt"]):
        values = {"True": True, "False": False}
        if not result["include_prompt"].isin(values).all():
            raise TargetPairSelectionError("include_prompt must contain booleans")
        result["include_prompt"] = result["include_prompt"].map(values).astype(bool)
    return result.loc[:, SELECTION_COLUMNS]


def _optional_float(value: object, label: str) -> float:
    if _missing(value) or value == "":
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TargetPairSelectionError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise TargetPairSelectionError(f"{label} must be finite or blank")
    return number


def _same_float(left: float, right: float) -> bool:
    return (math.isnan(left) and math.isnan(right)) or left == right


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TargetPairSelectionError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise TargetPairSelectionError(f"{label} must be an integer") from error
        if not math.isfinite(number) or not number.is_integer():
            raise TargetPairSelectionError(f"{label} must be an integer")
        return int(number)


def _index(value: object) -> str:
    if _missing(value):
        raise TargetPairSelectionError("original_index is missing")
    text = str(value).strip()
    if not text or text in {".", ".."} or Path(text).name != text or "\\" in text:
        raise TargetPairSelectionError(f"Unsafe original_index: {value!r}")
    return text


def _num_seeds(value: object) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        count = operator.index(value)
        validate_seed_block(count, count)
    except (TypeError, ValueError) as error:
        raise TargetPairSelectionError(
            "num_seeds must define a valid experiment and reference seed block"
        ) from error
    return count


def _reference_values(
    model_name: object,
    scheduler_name: object,
    guidance_scale: object,
    num_inference_steps: object,
    num_seeds: object,
) -> tuple[str, str, float, int, int]:
    model = _model(model_name)
    count = _num_seeds(num_seeds)
    try:
        if not isinstance(scheduler_name, str):
            raise TypeError
        if isinstance(guidance_scale, bool):
            raise TypeError
        guidance = float(guidance_scale)
        if not math.isfinite(guidance):
            raise ValueError
        if isinstance(num_inference_steps, bool):
            raise TypeError
        steps = operator.index(num_inference_steps)
        generation_run_name(
            model,
            scheduler_name,
            guidance,
            steps,
            count,
            seed_start=count,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise TargetPairSelectionError(
            "model, scheduler, guidance scale, steps, and N must define a valid "
            "reference run"
        ) from error
    return model, scheduler_name, guidance, steps, count


def _reference_seeds(num_seeds: int) -> tuple[int, ...]:
    count = _num_seeds(num_seeds)
    return tuple(range(count, 2 * count))


def _model(value: object) -> str:
    if not isinstance(value, str) or value not in _MODELS:
        raise TargetPairSelectionError(
            f"Unknown model {value!r}; expected one of: {', '.join(_MODELS)}"
        )
    return value


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _scalar(value: object) -> object:
    if _missing(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _json(value: object) -> object:
    if _missing(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json(item) for item in value]
    return value


def _raise_missing(
    root: str | Path,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> None:
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    directory = target_pair_selection_directory(
        root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
    )
    raise TargetPairSelectionMissingError(
        f"Frozen target-pair selection is missing: {directory}\n"
        f"Required reference run: "
        f"{reference_run_path(model, scheduler, guidance, steps, count).as_posix()}\n"
        f"Create it with:\n"
        f"{reference_selection_command(model, scheduler, guidance, steps, count)}"
    )
