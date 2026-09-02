"""Cache-only terminal latent proximity and target-pair analysis."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from utils.common.cli import generation_run_name, stable_float
from utils.common.io import (
    CacheIOError,
    atomic_copy,
    atomic_torch_save,
    atomic_write_frame_csv,
    atomic_write_json,
    canonical_json,
    file_sha256,
    read_json,
    safe_torch_load,
    utc_now,
)
from utils.models.latent import compute_latent_distances

from .cache import (
    CompletedGenerationRecord,
    GenerationCacheError,
    GenerationPaths,
    generation_paths,
    list_completed_records,
    require_generation_run,
    safe_index,
    validate_generation_record,
)
from .plotting import AnalysisStatistics, write_analysis_outputs

SCHEMA_VERSION = 2
SELECTION_POLICY = "target_pair_selection"

PAIRED_COLUMNS = tuple(
    "original_index record_id source_row_number seed prompt_raw "
    "webster_overfit_type_raw target_image_sha256 latent_l2 latent_rmse "
    "sscd_cosine_similarity".split()
)
FAILED_COLUMNS = tuple(
    "original_index record_id source_row_number issue_type reason generation_record_path sscd_record_path".split()
)


class ProximityError(RuntimeError):
    """The local cache or frozen selection cannot support this analysis."""


@dataclass(frozen=True, slots=True)
class ProximityPaths:
    project_root: Path
    generation_run: Path
    output_directory: Path

    @classmethod
    def build(
        cls,
        root: Path,
        generation_run: str | Path,
        *,
        output_run_name: str | None = None,
        role: str = "experiment",
        seed_start: int = 0,
        num_seeds: int = 20,
    ) -> "ProximityPaths":
        namespace = f"{role}_S{seed_start}_N{num_seeds}"
        generation = Path(generation_run)
        if not generation.is_absolute():
            generation = root / "logs" / generation
        if output_run_name is None:
            try:
                output_parent = generation.relative_to(root / "logs").parts[0]
            except (ValueError, IndexError) as error:
                raise ValueError(
                    "output_run_name is required for a generation run outside logs"
                ) from error
        else:
            output_parent = output_run_name
        output = root / "outputs" / output_parent / "proximity" / namespace
        return cls(root, generation, output)

    @property
    def records_directory(self) -> Path:
        return self.output_directory / "records"

    @property
    def failed_csv(self) -> Path:
        return self.output_directory / "failed.csv"

    @property
    def run_config_json(self) -> Path:
        return self.output_directory / "run_config.json"

    @property
    def summary_json(self) -> Path:
        return self.output_directory / "summary.json"

    def result_path(self, original_index: str) -> Path:
        return self.records_directory / f"{safe_index(original_index)}.pt"


@dataclass(frozen=True, slots=True)
class ProximitySummary:
    paths: ProximityPaths
    values: Mapping[str, object]

    @property
    def exit_code(self) -> int:
        return 0 if bool(self.values.get("complete")) else 1


@dataclass(frozen=True, slots=True)
class _GenerationRecord:
    metadata: Mapping[str, object]
    original_index: str
    source_row_number: int
    latent_path: Path
    target_latent_path: Path
    marker_sha256: str


def _record_progress(values: Sequence[Any]) -> Iterable[Any]:
    """Keep per-record proximity work visible in terminals and captured logs."""

    return tqdm(
        values,
        total=len(values),
        desc="[Proximity 1/2] Records",
        unit="record",
        dynamic_ncols=True,
        leave=True,
        disable=False,
    )


def _seed_role(
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int,
) -> str:
    """Accept only disjoint experiment runs or the exact selection reference."""

    from utils.data.selection import is_reference_configuration

    identity = {
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
        "seed_start": seed_start,
    }
    if is_reference_configuration(**identity):
        return "reference"
    if seed_start == 0 and 0 < num_seeds <= 20:
        return "experiment"
    raise ProximityError(
        "seed range must be an experiment starting at 0 with N <= 20, or the "
        "exact selection reference --scheduler ddim --g 7.5 --T 50 --N 20 "
        "--seed-start 20; partial overlap with reference seeds 20--39 is "
        "forbidden"
    )


def _preexisting_selection(
    root: Path, *, model_name: str, role: str
) -> Any | None:
    """Load selection early so incompatible output config fails before writes."""

    from utils.data.selection import (
        load_target_pair_selection,
        target_pair_selection_directory,
    )

    directory = target_pair_selection_directory(root, model_name=model_name)
    if role == "reference" and not directory.exists():
        return None
    return load_target_pair_selection(root, model_name=model_name)


def run_proximity(
    project_root: str | Path,
    *,
    model_name: str = "sdv1",
    scheduler_name: str = "ddim",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_seeds: int = 20,
    seed_start: int = 0,
) -> ProximitySummary:
    """Validate local caches, compute proximity, freeze/apply selection, and plot."""

    started = time.monotonic()
    root = Path(project_root).expanduser().resolve()
    role = _seed_role(
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        seed_start=seed_start,
    )
    cache_paths = generation_paths(
        root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        seed_start=seed_start,
    )
    run_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start,
    )
    output_run_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start=0,
    )
    paths = ProximityPaths.build(
        root,
        cache_paths.run_directory,
        output_run_name=output_run_name,
        role=role,
        seed_start=seed_start,
        num_seeds=num_seeds,
    )
    generation_command = _generation_command(
        model_name, scheduler_name, guidance_scale, num_inference_steps,
        num_seeds, seed_start,
    )
    if not cache_paths.run_config.exists() and not cache_paths.run_config.is_symlink():
        raise ProximityError(
            f"required generation configuration is absent: {cache_paths.run_config}\n"
            f"Create it with:\n{generation_command}"
        )
    generation_config = require_generation_run(cache_paths)
    _validate_generation_invocation(
        generation_config,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        seed_start=seed_start,
    )
    sscd_config_path = paths.generation_run / "sscd_config.json"
    sscd_command = _sscd_command(
        model_name, scheduler_name, guidance_scale, num_inference_steps,
        num_seeds, seed_start,
    )
    if not sscd_config_path.exists() and not sscd_config_path.is_symlink():
        raise ProximityError(
            f"required cached SSCD configuration is absent: {sscd_config_path}\n"
            f"Create it with:\n{sscd_command}"
        )
    sscd_config = _load_sscd_config(paths.generation_run, generation_config)
    preexisting_selection = _preexisting_selection(
        root, model_name=model_name, role=role
    )
    if preexisting_selection is not None:
        desired = _analysis_configuration(
            paths, run_name, generation_config, sscd_config, preexisting_selection
        )
        _write_or_validate_configuration(paths.run_config_json, desired)
    elif paths.run_config_json.exists() or paths.run_config_json.is_symlink():
        raise ProximityError(
            "analysis configuration exists without its frozen reference selection: "
            f"{paths.run_config_json}"
        )
    paths.records_directory.mkdir(parents=True, exist_ok=True)
    records = list_completed_records(cache_paths)
    if not records:
        raise ProximityError(
            f"no completed generation records found in {paths.generation_run / 'record'}\n"
            f"Create them with:\n{_generation_command(
                model_name, scheduler_name, guidance_scale, num_inference_steps,
                num_seeds, seed_start,
            )}"
        )

    paired_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for completed in _record_progress(records):
        try:
            record = _validated_generation_record(cache_paths, completed, generation_config)
            proximity = _load_or_compute_result(paths, record, generation_config)
            scores = _validated_sscd_scores(root, paths.generation_run, record, generation_config, sscd_config)
            paired_rows.extend(_paired_rows(record, proximity, scores))
        except (
            CacheIOError,
            GenerationCacheError,
            KeyError,
            OSError,
            ProximityError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            failures.append(_failure_row(completed, error, paths.generation_run))

    atomic_write_frame_csv(pd.DataFrame(failures, columns=FAILED_COLUMNS), paths.failed_csv)
    if failures:
        values = {
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "run_name": run_name,
            "generation_scientific_config_hash": generation_config["scientific_config_hash"],
            "sscd_configuration_hash": sscd_config["configuration_hash"],
            "failed_rows": len(failures),
            "failed_csv": _display_path(paths.failed_csv, root),
            "run_duration_seconds": time.monotonic() - started,
            "finished_at_utc": utc_now(),
        }
        atomic_write_json(paths.summary_json, values)
        return ProximitySummary(paths, values)

    paired_all = pd.DataFrame(paired_rows, columns=PAIRED_COLUMNS)
    paired_all = paired_all.sort_values(["source_row_number", "seed"], kind="stable").reset_index(drop=True)
    selection = _selection_for_run(
        root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        seed_start=seed_start,
        paired_all=paired_all,
        generation_config=generation_config,
        sscd_config=sscd_config,
    )
    paired_all = _annotate_selection(paired_all, selection)
    paired_selected = _selected_frame(paired_all, selection)
    run_config = _analysis_configuration(
        paths, run_name, generation_config, sscd_config, selection
    )
    # Validate the scientific identity before copying or publishing any tables.
    _write_or_validate_configuration(
        paths.run_config_json,
        run_config,
        refresh_source_provenance=True,
    )
    _write_selection_outputs(paths.output_directory, selection)
    statistics = write_analysis_outputs(
        paths.output_directory,
        paired_all=paired_all,
        paired_selected=paired_selected,
    )
    values = _complete_summary(
        paths,
        run_name,
        generation_config,
        sscd_config,
        selection,
        statistics,
        time.monotonic() - started,
    )
    atomic_write_json(paths.summary_json, values)
    return ProximitySummary(paths, values)


def _validate_generation_invocation(config: Mapping[str, object], **expected: object) -> None:
    science = config.get("scientific_config")
    if not isinstance(science, Mapping):
        raise ProximityError("generation scientific_config is missing")
    seeds = science.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ProximityError("generation seeds are missing")
    scheduler = science.get("scheduler")
    scheduler_name = scheduler.get("name") if isinstance(scheduler, Mapping) else science.get("scheduler_name")
    observed = {
        "model_name": science.get("model_cli_name"),
        "scheduler_name": scheduler_name,
        "guidance_scale": science.get("guidance_scale"),
        "num_inference_steps": science.get("num_inference_steps"),
        "num_seeds": science.get("num_seeds"),
        "seed_start": seeds[0],
    }
    for key, value in expected.items():
        if canonical_json(observed.get(key)) != canonical_json(value):
            raise ProximityError(f"generation configuration {key} differs")
    start = int(expected["seed_start"])
    count = int(expected["num_seeds"])
    if seeds != list(range(start, start + count)):
        raise ProximityError("generation seeds differ from seed-start and N")
    _latent_shape(science.get("latent_shape"))


def _load_sscd_config(run: Path, generation: Mapping[str, object]) -> dict[str, Any]:
    from utils.experiments.sscd import sscd_configuration_hash

    path = run / "sscd_config.json"
    if not path.is_file() or path.is_symlink():
        raise ProximityError(f"required cached SSCD configuration is absent: {path}")
    config = read_json(path)
    stored_hash = config.get("configuration_hash")
    if not _sha256(stored_hash) or sscd_configuration_hash(config) != stored_hash:
        raise ProximityError("SSCD configuration hash is invalid")
    science = generation["scientific_config"]
    assert isinstance(science, Mapping)
    expected = {
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "num_seeds": science["num_seeds"],
        "seeds": science["seeds"],
    }
    for key, value in expected.items():
        if canonical_json(config.get(key)) != canonical_json(value):
            raise ProximityError(f"SSCD configuration {key} differs")
    return config


def _validated_generation_record(
    paths: GenerationPaths,
    completed: CompletedGenerationRecord,
    config: Mapping[str, object],
) -> _GenerationRecord:
    record = completed.metadata
    marker_path = completed.marker_path
    science = config["scientific_config"]
    assert isinstance(science, Mapping)
    index = completed.original_index
    validation = validate_generation_record(
        paths,
        index,
        expected_scientific_hash=str(config["scientific_config_hash"]),
        load_tensors=False,
    )
    if not validation.valid:
        raise ProximityError("invalid completed generation record: " + "; ".join(validation.errors))
    expected = {
        "scientific_config_hash": config["scientific_config_hash"],
        "model_cli_name": science["model_cli_name"],
        "guidance_scale": science["guidance_scale"],
        "num_inference_steps": science["num_inference_steps"],
        "num_seeds": science["num_seeds"],
        "seeds": science["seeds"],
        "latent_shape": science["latent_shape"],
    }
    scheduler = science.get("scheduler")
    expected["scheduler_name"] = (
        scheduler.get("name") if isinstance(scheduler, Mapping) else science.get("scheduler_name")
    )
    for key, value in expected.items():
        if canonical_json(record.get(key)) != canonical_json(value):
            raise ProximityError(f"generation marker {key} differs")
    if not isinstance(record.get("record_id"), str) or not record["record_id"]:
        raise ProximityError("generation marker record_id is invalid")
    if not isinstance(record.get("prompt_raw"), str):
        raise ProximityError("generation marker prompt_raw is invalid")
    return _GenerationRecord(
        record,
        index,
        completed.source_row_number,
        paths.latent_path(index),
        paths.target_latent_path(index),
        file_sha256(marker_path),
    )


def _load_or_compute_result(
    paths: ProximityPaths,
    record: _GenerationRecord,
    config: Mapping[str, object],
) -> Mapping[str, object]:
    result_path = paths.result_path(record.original_index)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "original_index": record.original_index,
        "generation_record_sha256": record.marker_sha256,
        "generation_scientific_config_hash": config["scientific_config_hash"],
        "generation_latent_sha256": record.metadata["tensor_file_sha256"]["latent"],  # type: ignore[index]
        "generation_target_latent_sha256": record.metadata["tensor_file_sha256"]["target_latent"],  # type: ignore[index]
        "seeds": record.metadata["seeds"],
        "latent_shape": record.metadata["latent_shape"],
    }
    if result_path.is_file() and not result_path.is_symlink():
        cached = safe_torch_load(result_path)
        if (
            isinstance(cached, Mapping)
            and all(canonical_json(cached.get(key)) == canonical_json(value) for key, value in expected.items())
            and _valid_distance_vectors(
                cached,
                int(record.metadata["num_seeds"]),
                expected["latent_shape"],
            )
        ):
            return cached
    trajectory = safe_torch_load(record.latent_path)
    target = safe_torch_load(record.target_latent_path)
    if not isinstance(trajectory, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ProximityError("generation proximity inputs must be plain tensors")
    expected_shape = _latent_shape(record.metadata["latent_shape"])
    expected_trajectory = (
        int(record.metadata["num_seeds"]),
        int(record.metadata["num_inference_steps"]) + 1,
        *expected_shape,
    )
    if tuple(trajectory.shape) != expected_trajectory or tuple(target.shape) != expected_shape:
        raise ProximityError("cached latent tensor shape differs from marker")
    distances = compute_latent_distances(trajectory[:, -1], target)
    payload = {
        **expected,
        "latent_l2": distances.l2_norms,
        "latent_rmse": distances.latent_rmse,
    }
    atomic_torch_save(payload, result_path)
    return payload


def _validated_sscd_scores(
    root: Path,
    run: Path,
    record: _GenerationRecord,
    generation: Mapping[str, object],
    config: Mapping[str, object],
) -> torch.Tensor:
    marker_path = run / "sscd_record" / f"{record.original_index}.json"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise ProximityError(f"cached SSCD marker is absent: {marker_path}")
    marker = read_json(marker_path)
    expected = {
        "original_index": record.original_index,
        "record_id": record.metadata["record_id"],
        "source_row_number": record.source_row_number,
        "prompt_raw": record.metadata["prompt_raw"],
        "generation_record_sha256": record.marker_sha256,
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "generation_latent_sha256": record.metadata["tensor_file_sha256"]["latent"],  # type: ignore[index]
        "target_image_sha256": record.metadata.get("target_image_sha256"),
        "sscd_configuration_hash": config["configuration_hash"],
        "num_seeds": record.metadata["num_seeds"],
        "seeds": record.metadata["seeds"],
        "score_shape": [record.metadata["num_seeds"]],
        "score_dtype": "float32",
    }
    for key, value in expected.items():
        if canonical_json(marker.get(key)) != canonical_json(value):
            raise ProximityError(f"SSCD marker {key} differs")
    for key in (
        "sscd_model_name",
        "sscd_checkpoint_sha256",
        "sscd_feature_dimension",
        "sscd_input_size",
        "sscd_preprocessing_hash",
    ):
        if key in config and canonical_json(marker.get(key)) != canonical_json(config[key]):
            raise ProximityError(f"SSCD marker {key} differs")
    score_path = _canonical_record_path(
        root,
        run,
        marker.get("score_path"),
        run / "sscd" / f"{record.original_index}.pt",
    )
    _validate_cached_file(score_path, marker.get("score_sha256"), "SSCD score")
    score = safe_torch_load(score_path)
    count = int(record.metadata["num_seeds"])
    if (
        not isinstance(score, torch.Tensor)
        or score.dtype != torch.float32
        or score.device.type != "cpu"
        or not score.is_contiguous()
        or tuple(score.shape) != (count,)
        or not bool(torch.isfinite(score).all())
    ):
        raise ProximityError("cached SSCD score tensor is invalid")
    if bool((score < -1.00001).any()) or bool((score > 1.00001).any()):
        raise ProximityError("cached SSCD score is outside the cosine range")
    return score


def _paired_rows(
    record: _GenerationRecord,
    proximity: Mapping[str, object],
    scores: torch.Tensor,
) -> list[dict[str, object]]:
    l2 = proximity["latent_l2"]
    rmse = proximity["latent_rmse"]
    assert isinstance(l2, torch.Tensor) and isinstance(rmse, torch.Tensor)
    common = {
        "original_index": record.original_index,
        "record_id": record.metadata["record_id"],
        "source_row_number": record.source_row_number,
        "prompt_raw": record.metadata["prompt_raw"],
        "webster_overfit_type_raw": record.metadata.get("webster_overfit_type"),
        "target_image_sha256": record.metadata.get("target_image_sha256"),
    }
    return [
        {
            **common,
            "seed": int(seed),
            "latent_l2": float(l2[position]),
            "latent_rmse": float(rmse[position]),
            "sscd_cosine_similarity": float(scores[position]),
        }
        for position, seed in enumerate(record.metadata["seeds"])
    ]


def _selection_for_run(root: Path, **values: object) -> Any:
    from utils.data.selection import (
        ensure_reference_target_pair_selection,
        is_reference_configuration,
        load_target_pair_selection,
    )

    identity = {
        key: values[key]
        for key in (
            "model_name",
            "scheduler_name",
            "guidance_scale",
            "num_inference_steps",
            "num_seeds",
            "seed_start",
        )
    }
    if is_reference_configuration(**identity):
        return ensure_reference_target_pair_selection(
            root,
            model_name=str(values["model_name"]),
            paired_frame=values["paired_all"],
            reference_run_config=values["generation_config"],
            sscd_config=values["sscd_config"],
        )
    return load_target_pair_selection(root, model_name=str(values["model_name"]))


def _annotate_selection(frame: pd.DataFrame, selection: Any) -> pd.DataFrame:
    selection_frame = selection.frame.copy(deep=True)
    selection_frame["original_index"] = selection_frame["original_index"].astype(str)
    frame = frame.copy(deep=True)
    frame["original_index"] = frame["original_index"].astype(str)
    fields = [
        "original_index",
        "webster_overfit_type_normalized",
        "include_target_pair",
        "selection_status",
        "selection_reason",
        "target_semantics",
        "selection_hash",
    ]
    annotated = frame.merge(
        selection_frame[fields],
        on="original_index",
        how="left",
        validate="many_to_one",
    )
    if annotated["include_target_pair"].isna().any():
        missing = sorted(annotated.loc[annotated["include_target_pair"].isna(), "original_index"].unique())
        raise ProximityError("frozen selection has no row for: " + ", ".join(missing))
    return annotated


def _selected_frame(frame: pd.DataFrame, selection: Any) -> pd.DataFrame:
    from utils.data.selection import apply_target_pair_selection

    return apply_target_pair_selection(frame, selection)


def _write_selection_outputs(output_directory: Path, selection: Any) -> None:
    """Copy only validated schema-2 selection sidecars into this analysis."""

    from utils.data.selection import target_pair_selection_directory

    source_directory = target_pair_selection_directory(
        selection.root, model_name=selection.model_name
    )
    for filename in (
        "selection.csv",
        "selected_tv.csv",
        "excluded_tv.csv",
        "threshold_diagnostics.csv",
    ):
        source = source_directory / filename
        if not source.is_file() or source.is_symlink():
            raise ProximityError(f"frozen selection sidecar is invalid: {source}")
        atomic_copy(source, output_directory / filename)


def _analysis_configuration(
    paths: ProximityPaths,
    run_name: str,
    generation: Mapping[str, object],
    sscd: Mapping[str, object],
    selection: Any,
) -> dict[str, object]:
    science = generation["scientific_config"]
    assert isinstance(science, Mapping)
    seeds = science.get("seeds")
    is_experiment = isinstance(seeds, list) and bool(seeds) and seeds[0] == 0
    source_files = (
        Path(__file__),
        paths.project_root / "utils/experiments/plotting.py",
        paths.project_root / "utils/data/selection.py",
        paths.project_root / "utils/models/latent.py",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "proximity",
        "source": "validated_generation_and_sscd_caches_only",
        "run_name": run_name,
        "generation_run_path": _display_path(paths.generation_run, paths.project_root),
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "sscd_configuration_hash": sscd["configuration_hash"],
        "selection_policy": SELECTION_POLICY,
        "selection_hash": selection.sha256,
        "model_name": science["model_cli_name"],
        "scheduler_name": _scheduler_name(science),
        "guidance_scale": science["guidance_scale"],
        "num_inference_steps": science["num_inference_steps"],
        "num_seeds": science["num_seeds"],
        "seeds": science["seeds"],
        "seed_role": "experiment" if is_experiment else "selection_reference",
        "terminal_latent_index": -1,
        "raw_distance": "euclidean_l2",
        "dimension_normalized_distance": "latent_l2 / sqrt(d)",
        "paper_facing_table": "paired_selected" if is_experiment else None,
        "duplicates_generation_tensors": False,
        "source_provenance": {
            _display_path(source, paths.project_root): file_sha256(source)
            for source in source_files
        },
    }


def _write_or_validate_configuration(
    path: Path,
    desired: Mapping[str, object],
    *,
    refresh_source_provenance: bool = False,
) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ProximityError(f"unsafe existing analysis configuration: {path}")
        existing = read_json(path)
        if canonical_json(existing) == canonical_json(desired):
            return
        if _same_analysis_contract(existing, desired):
            if refresh_source_provenance:
                atomic_write_json(path, desired)
            return
        raise ProximityError(f"existing analysis configuration is incompatible: {path}")
    atomic_write_json(path, desired)


def _same_analysis_contract(
    existing: Mapping[str, object],
    desired: Mapping[str, object],
) -> bool:
    """Compare schema-versioned science while treating source hashes as audit data."""

    existing_values = dict(existing)
    desired_values = dict(desired)
    existing_values.pop("source_provenance", None)
    desired_values.pop("source_provenance", None)
    return canonical_json(existing_values) == canonical_json(desired_values)


def _complete_summary(
    paths: ProximityPaths,
    run_name: str,
    generation: Mapping[str, object],
    sscd: Mapping[str, object],
    selection: Any,
    statistics: AnalysisStatistics,
    duration: float,
) -> dict[str, object]:
    science = generation["scientific_config"]
    assert isinstance(science, Mapping)
    seeds = science.get("seeds")
    is_experiment = isinstance(seeds, list) and bool(seeds) and seeds[0] == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "run_name": run_name,
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "sscd_configuration_hash": sscd["configuration_hash"],
        "selection_policy": SELECTION_POLICY,
        "selection_hash": selection.sha256,
        "included_prompts": len(selection.included_indices),
        "excluded_prompts": len(selection.excluded_indices),
        "selected_tv_prompts": len(selection.selected_tv_indices),
        "tables": {
            name: {"prompts": value.prompts, "points": value.points}
            for name, value in (
                ("paired_all", statistics.all),
                ("paired_selected", statistics.selected),
            )
        },
        "correlations": statistics.as_dict(),
        "paper_facing_correlation": (
            statistics.selected.as_dict() if is_experiment else None
        ),
        "failed_rows": 0,
        "run_duration_seconds": duration,
        "finished_at_utc": utc_now(),
        "output_directory": _display_path(paths.output_directory, paths.project_root),
    }


def _valid_distance_vectors(
    value: Mapping[str, object], count: int, latent_shape: object
) -> bool:
    for key in ("latent_l2", "latent_rmse"):
        tensor = value.get(key)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != (count,)
            or not bool(torch.isfinite(tensor).all())
        ):
            return False
    l2 = value["latent_l2"]
    rmse = value["latent_rmse"]
    assert isinstance(l2, torch.Tensor) and isinstance(rmse, torch.Tensor)
    if bool((l2 < 0).any()) or bool((rmse < 0).any()):
        return False
    denominator = math.sqrt(math.prod(_latent_shape(latent_shape)))
    return torch.equal(rmse, l2 / denominator)


def _validate_cached_file(path: Path, expected_hash: object, label: str) -> None:
    if not _sha256(expected_hash):
        raise ProximityError(f"{label} SHA-256 is invalid")
    if not path.is_file() or path.is_symlink():
        raise ProximityError(f"missing safe {label}: {path}")
    if file_sha256(path) != expected_hash:
        raise ProximityError(f"{label} SHA-256 differs")


def _canonical_record_path(root: Path, run: Path, value: object, expected: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ProximityError("cached artifact path is invalid")
    given = Path(value)
    candidates = [given.resolve()] if given.is_absolute() else [(run / given).resolve(), (root / given).resolve()]
    target = expected.resolve()
    if target not in candidates:
        raise ProximityError(f"cached artifact path is noncanonical: {value}")
    return target


def _failure_row(completed: CompletedGenerationRecord, error: Exception, run: Path) -> dict[str, object]:
    metadata = completed.metadata
    index = completed.original_index
    return {
        "original_index": index,
        "record_id": metadata.get("record_id"),
        "source_row_number": metadata.get("source_row_number"),
        "issue_type": type(error).__name__,
        "reason": str(error),
        "generation_record_path": str(completed.marker_path),
        "sscd_record_path": str(run / "sscd_record" / f"{index}.json"),
    }


def _latent_shape(value: object) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProximityError("latent_shape is invalid")
    shape = tuple(value)
    if len(shape) != 3 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape):
        raise ProximityError("latent_shape is invalid")
    return shape  # type: ignore[return-value]


def _scheduler_name(science: Mapping[str, object]) -> object:
    scheduler = science.get("scheduler")
    return scheduler.get("name") if isinstance(scheduler, Mapping) else science.get("scheduler_name")


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sscd_command(
    model: str, scheduler: str, g: float, steps: int, seeds: int, seed_start: int
) -> str:
    return (
        f"./sscd.sh --model {model} --scheduler {scheduler} "
        f"--g {stable_float(g)} --T {steps} --N {seeds} "
        f"--seed-start {seed_start}"
    )


def _generation_command(
    model: str, scheduler: str, g: float, steps: int, seeds: int, seed_start: int
) -> str:
    return (
        f"./generate.sh --model {model} --scheduler {scheduler} "
        f"--g {stable_float(g)} --T {steps} --N {seeds} "
        f"--seed-start {seed_start} --downscale 4"
    )
