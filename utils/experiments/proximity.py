"""Build or apply frozen prompt selection from generation and SSCD caches."""

from __future__ import annotations

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
    atomic_write_frame_csv,
    atomic_write_json,
    canonical_json,
    file_sha256,
    read_json,
    safe_torch_load,
    utc_now,
)
from utils.data.webster import normalize_webster_type
from utils.models.latent import compute_latent_distances

from .cache import (
    CompletedGenerationRecord,
    GenerationCacheError,
    GenerationPaths,
    generation_paths,
    list_completed_records,
    require_generation_run,
)
from .plotting import (
    AnalysisStatistics,
    write_analysis_outputs,
    write_selection_figure,
)

OBSERVATION_COLUMNS = tuple(
    "model_name original_index record_id source_row_number seed prompt kind "
    "target_image_sha256 generated_image_path generated_image_tile_index "
    "l2_norm sscd observation_status observation_error".split()
)
ANALYSIS_COLUMNS = OBSERVATION_COLUMNS + tuple(
    "prompt_spearman include_prompt selection_status selection_reason".split()
)
FAILED_COLUMNS = tuple(
    "original_index record_id source_row_number issue_type reason "
    "generation_record_path sscd_record_path".split()
)


class ProximityError(RuntimeError):
    """The local generation, SSCD, or frozen selection cache is unusable."""


@dataclass(frozen=True, slots=True)
class ProximityPaths:
    project_root: Path
    generation_run: Path
    output_directory: Path
    configuration_filename: str = "run_config.json"

    @classmethod
    def build(
        cls,
        root: Path,
        generation_run: str | Path,
        *,
        output_run_name: str | None = None,
        role: str,
        seed_start: int,
        num_seeds: int,
    ) -> "ProximityPaths":
        generation = Path(generation_run)
        if not generation.is_absolute():
            generation = root / "logs" / generation
        if output_run_name is None:
            try:
                output_run_name = generation.relative_to(root / "logs").parts[0]
            except (ValueError, IndexError) as error:
                raise ValueError(
                    "output_run_name is required outside the logs directory"
                ) from error
        namespace = f"{role}_S{seed_start}_N{num_seeds}"
        output = root / "outputs" / output_run_name / "proximity" / namespace
        return cls(root, generation, output)

    @classmethod
    def frozen_selection(
        cls, root: Path, generation_run: Path, selection_directory: Path
    ) -> "ProximityPaths":
        return cls(root, generation_run, selection_directory, "config.json")

    @property
    def failed_csv(self) -> Path:
        return self.output_directory / "failed.csv"

    @property
    def run_config_json(self) -> Path:
        return self.output_directory / self.configuration_filename

    @property
    def summary_json(self) -> Path:
        return self.output_directory / "summary.json"


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
    completed: CompletedGenerationRecord
    paths: GenerationPaths
    generated_image_path: str
    marker_sha256: str


def _record_progress(values: Sequence[Any]) -> Iterable[Any]:
    return tqdm(
        values,
        total=len(values),
        desc="[Proximity] Records",
        unit="record",
        dynamic_ncols=True,
        leave=True,
        disable=False,
    )


def _set_record_progress(
    progress: Any,
    completed: CompletedGenerationRecord,
    status: str,
) -> None:
    record_id = completed.metadata.get("record_id") or completed.original_index
    progress.set_postfix(record=record_id, status=status, refresh=True)


def _seed_role(
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int,
) -> str:
    from utils.data.selection import is_reference_configuration

    if is_reference_configuration(
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        seed_start=seed_start,
    ):
        return "reference"
    if (
        isinstance(num_seeds, int)
        and not isinstance(num_seeds, bool)
        and num_seeds > 0
        and seed_start == 0
    ):
        return "experiment"
    raise ProximityError(
        "seed range must be the first N seeds (--seed-start 0) for the "
        "experiment, or the next N seeds (--seed-start N) for the matching "
        "selection reference; N must be positive"
    )


def run_proximity(
    project_root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int,
) -> ProximitySummary:
    """Build on reference seeds N..2N-1 or report a frozen experiment."""

    started = time.monotonic()
    root = Path(project_root).expanduser().resolve()
    arguments = {
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
        "seed_start": seed_start,
    }
    role = _seed_role(**arguments)
    cache = generation_paths(root, **arguments)
    run_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start,
    )
    parent_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start=0,
    )
    paths = ProximityPaths.build(
        root,
        cache.run_directory,
        output_run_name=parent_name,
        role=role,
        seed_start=seed_start,
        num_seeds=num_seeds,
    )

    if role == "reference":
        frozen = _frozen_reference_result(
            root,
            cache.run_directory,
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            num_seeds,
        )
        if frozen is not None:
            return frozen

    generation_command = _generation_command(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start,
    )
    if not cache.run_config.is_file() or cache.run_config.is_symlink():
        raise ProximityError(
            f"required generation configuration is absent: {cache.run_config}\n"
            f"Create it with:\n{generation_command}"
        )
    generation = require_generation_run(cache)
    _validate_generation_invocation(generation, **arguments)
    sscd_path = cache.run_directory / "sscd_config.json"
    if not sscd_path.is_file() or sscd_path.is_symlink():
        raise ProximityError(
            f"required cached SSCD configuration is absent: {sscd_path}\n"
            f"Create it with:\n{
                _sscd_command(
                    model_name,
                    scheduler_name,
                    guidance_scale,
                    num_inference_steps,
                    num_seeds,
                    seed_start,
                )
            }"
        )
    sscd = _load_sscd_config(cache.run_directory, generation)

    selection = None
    if role == "experiment":
        from utils.data.selection import load_target_pair_selection

        selection = load_target_pair_selection(
            root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=num_seeds,
        )

    records = list_completed_records(cache)
    if not records:
        raise ProximityError(
            f"no completed generation records found in {cache.record_directory}\n"
            f"Create them with:\n{generation_command}"
        )
    _require_generation_coverage(cache, records)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    progress = _record_progress(records)
    for completed in progress:
        _set_record_progress(progress, completed, "processing")
        try:
            record = _validated_record(root, cache, completed, generation)
            l2 = _terminal_l2(record)
            scores = _sscd_scores(root, record, generation, sscd)
            rows.extend(_paired_rows(record, l2, scores, model_name=model_name))
            _set_record_progress(progress, completed, "complete")
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
            if role == "reference":
                try:
                    failed_rows = _failed_reference_rows(
                        root,
                        cache,
                        completed,
                        generation,
                        error,
                        model_name=model_name,
                    )
                except Exception:
                    _set_record_progress(progress, completed, "failed")
                    raise
                rows.extend(failed_rows)
                _set_record_progress(progress, completed, "unusable")
            else:
                failures.append(_failure_row(completed, error, cache.run_directory))
                _set_record_progress(progress, completed, "failed")

    observations = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).sort_values(
        ["source_row_number", "original_index", "seed"],
        kind="stable",
        ignore_index=True,
    )
    if role == "reference":
        from utils.data.selection import build_target_pair_selection

        build_target_pair_selection(
            root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=num_seeds,
            paired_frame=observations,
            records_frame=_reference_records_frame(records, model_name),
            reference_run_config=generation,
            sscd_config=sscd,
        )
        result = _frozen_reference_result(
            root,
            cache.run_directory,
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            num_seeds,
        )
        assert result is not None
        return result

    assert selection is not None
    analysis = _annotate_selection(
        observations, selection, require_complete=not failures
    )
    configuration = _analysis_configuration(
        paths, run_name, generation, sscd, selection
    )
    _write_configuration(paths.run_config_json, configuration)
    if failures:
        _remove_optional(paths.output_directory / "proximity_vs_sscd.png")
        atomic_write_frame_csv(analysis, paths.output_directory / "proximity.csv")
        atomic_write_frame_csv(
            pd.DataFrame(failures, columns=FAILED_COLUMNS), paths.failed_csv
        )
        values = _summary(
            paths,
            run_name,
            generation,
            sscd,
            selection,
            analysis,
            duration=time.monotonic() - started,
            failures=failures,
        )
    else:
        _remove_optional(paths.failed_csv)
        statistics = write_analysis_outputs(paths.output_directory, analysis=analysis)
        values = _summary(
            paths,
            run_name,
            generation,
            sscd,
            selection,
            analysis,
            duration=time.monotonic() - started,
            statistics=statistics,
        )
    atomic_write_json(paths.summary_json, values)
    return ProximitySummary(paths, values)


def _frozen_reference_result(
    root: Path,
    generation_run: Path,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> ProximitySummary | None:
    from utils.data.selection import (
        load_target_pair_selection,
        target_pair_selection_directory,
    )

    directory = target_pair_selection_directory(
        root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
    )
    if not directory.exists() and not directory.is_symlink():
        return None
    selection = load_target_pair_selection(
        root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
    )
    paths = ProximityPaths.frozen_selection(root, generation_run, directory)
    summary = read_json(paths.summary_json)
    if summary.get("selection_hash") != selection.sha256:
        raise ProximityError("frozen selection summary hash differs")
    write_selection_figure(directory)
    return ProximitySummary(paths, summary)


def _validate_generation_invocation(
    config: Mapping[str, object], **expected: object
) -> None:
    science = config.get("scientific_config")
    if not isinstance(science, Mapping):
        raise ProximityError("generation scientific_config is missing")
    seeds = science.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ProximityError("generation seeds are missing")
    observed = {
        "model_name": science.get("model_cli_name"),
        "scheduler_name": _scheduler_name(science),
        "guidance_scale": science.get("guidance_scale"),
        "num_inference_steps": science.get("num_inference_steps"),
        "num_seeds": science.get("num_seeds"),
        "seed_start": seeds[0],
    }
    wrong = [
        key
        for key, value in expected.items()
        if canonical_json(observed.get(key)) != canonical_json(value)
    ]
    if wrong:
        raise ProximityError("generation configuration differs at: " + ", ".join(wrong))
    start, count = int(expected["seed_start"]), int(expected["num_seeds"])
    if seeds != list(range(start, start + count)):
        raise ProximityError("generation seeds differ from seed-start and N")
    _latent_shape(science.get("latent_shape"))


def _load_sscd_config(run: Path, generation: Mapping[str, object]) -> dict[str, Any]:
    from utils.experiments.sscd import sscd_configuration_hash

    config = read_json(run / "sscd_config.json")
    digest = config.get("configuration_hash")
    if not _sha256(digest) or sscd_configuration_hash(config) != digest:
        raise ProximityError("SSCD configuration hash is invalid")
    science = generation["scientific_config"]
    assert isinstance(science, Mapping)
    expected = {
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "num_seeds": science["num_seeds"],
        "seeds": science["seeds"],
    }
    if any(
        canonical_json(config.get(key)) != canonical_json(value)
        for key, value in expected.items()
    ):
        raise ProximityError("SSCD configuration differs from generation")
    return config


def _require_generation_coverage(
    paths: GenerationPaths, records: Sequence[CompletedGenerationRecord]
) -> None:
    path = paths.summary_json
    if not path.is_file() or path.is_symlink():
        raise ProximityError(f"required generation summary is absent: {path}")
    summary = read_json(path)
    if (
        summary.get("failed_rows") != 0
        or summary.get("completed_rows") != len(records)
        or summary.get("selected_rows") != len(records)
    ):
        raise ProximityError(
            "generation did not complete every selected prompt; rerun generation"
        )


def _validated_record(
    root: Path,
    paths: GenerationPaths,
    completed: CompletedGenerationRecord,
    config: Mapping[str, object],
) -> _GenerationRecord:
    metadata = completed.metadata
    index = completed.original_index
    science = config["scientific_config"]
    assert isinstance(science, Mapping)
    expected = {
        "scientific_config_hash": config["scientific_config_hash"],
        "model_cli_name": science["model_cli_name"],
        "scheduler_name": _scheduler_name(science),
        "guidance_scale": science["guidance_scale"],
        "num_inference_steps": science["num_inference_steps"],
        "num_seeds": science["num_seeds"],
        "seeds": science["seeds"],
        "latent_shape": science["latent_shape"],
        "preview_image_path": f"image/{index}.png",
    }
    wrong = [
        key
        for key, value in expected.items()
        if canonical_json(metadata.get(key)) != canonical_json(value)
    ]
    if wrong:
        raise ProximityError("generation marker differs at: " + ", ".join(wrong))
    hashes = metadata.get("tensor_file_sha256")
    if not isinstance(hashes, Mapping):
        raise ProximityError("generation tensor hashes are missing")
    _require_cached_file(paths.latent_path(index), hashes.get("latent"), "latent")
    _require_cached_file(
        paths.target_latent_path(index),
        hashes.get("target_latent"),
        "target latent",
    )
    _require_cached_file(
        paths.image_path(index),
        metadata.get("preview_image_sha256"),
        "generated montage",
    )
    return _GenerationRecord(
        metadata,
        completed,
        paths,
        _display_path(paths.image_path(index), root),
        file_sha256(completed.marker_path),
    )


def _terminal_l2(record: _GenerationRecord) -> torch.Tensor:
    trajectory = safe_torch_load(
        record.paths.latent_path(record.completed.original_index)
    )
    target = safe_torch_load(
        record.paths.target_latent_path(record.completed.original_index)
    )
    if not isinstance(trajectory, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ProximityError("generation proximity inputs must be tensors")
    shape = _latent_shape(record.metadata["latent_shape"])
    expected = (
        int(record.metadata["num_seeds"]),
        int(record.metadata["num_inference_steps"]) + 1,
        *shape,
    )
    if tuple(trajectory.shape) != expected or tuple(target.shape) != shape:
        raise ProximityError("generation latent shape differs from marker")
    return compute_latent_distances(trajectory[:, -1], target).l2_norms


def _sscd_scores(
    root: Path,
    record: _GenerationRecord,
    generation: Mapping[str, object],
    config: Mapping[str, object],
) -> torch.Tensor:
    index = record.completed.original_index
    run = record.paths.run_directory
    marker_path = run / "sscd_record" / f"{index}.json"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise ProximityError(f"cached SSCD marker is absent: {marker_path}")
    marker = read_json(marker_path)
    expected = {
        "original_index": index,
        "record_id": record.metadata["record_id"],
        "source_row_number": record.completed.source_row_number,
        "prompt_raw": record.metadata["prompt_raw"],
        "generation_record_sha256": record.marker_sha256,
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "generation_latent_sha256": record.metadata["tensor_file_sha256"]["latent"],
        "target_image_sha256": record.metadata.get("target_image_sha256"),
        "sscd_configuration_hash": config["configuration_hash"],
        "num_seeds": record.metadata["num_seeds"],
        "seeds": record.metadata["seeds"],
        "score_shape": [record.metadata["num_seeds"]],
        "score_dtype": "float32",
    }
    wrong = [
        key
        for key, value in expected.items()
        if canonical_json(marker.get(key)) != canonical_json(value)
    ]
    if wrong:
        raise ProximityError("SSCD marker differs at: " + ", ".join(wrong))
    score_path = _canonical_score_path(
        root, run, marker.get("score_path"), run / "sscd" / f"{index}.pt"
    )
    if (
        not _sha256(marker.get("score_sha256"))
        or not score_path.is_file()
        or score_path.is_symlink()
        or file_sha256(score_path) != marker["score_sha256"]
    ):
        raise ProximityError("cached SSCD score file is invalid")
    scores = safe_torch_load(score_path)
    count = int(record.metadata["num_seeds"])
    if (
        not isinstance(scores, torch.Tensor)
        or scores.dtype != torch.float32
        or scores.device.type != "cpu"
        or not scores.is_contiguous()
        or tuple(scores.shape) != (count,)
        or not bool(torch.isfinite(scores).all())
        or bool((scores < -1.00001).any())
        or bool((scores > 1.00001).any())
    ):
        raise ProximityError("cached SSCD score tensor is invalid")
    return scores


def _paired_rows(
    record: _GenerationRecord,
    l2: torch.Tensor,
    scores: torch.Tensor,
    *,
    model_name: str,
) -> list[dict[str, object]]:
    count = int(record.metadata["num_seeds"])
    if (
        tuple(l2.shape) != (count,)
        or not bool(torch.isfinite(l2).all())
        or tuple(scores.shape) != (count,)
    ):
        raise ProximityError("L2 and SSCD values do not align with seeds")
    common = {
        "model_name": model_name,
        "original_index": record.completed.original_index,
        "record_id": record.metadata["record_id"],
        "source_row_number": record.completed.source_row_number,
        "prompt": record.metadata["prompt_raw"],
        "kind": normalize_webster_type(record.metadata.get("webster_overfit_type")),
        "target_image_sha256": record.metadata.get("target_image_sha256"),
        "generated_image_path": record.generated_image_path,
    }
    return [
        {
            **common,
            "seed": int(seed),
            "generated_image_tile_index": position,
            "l2_norm": float(l2[position]),
            "sscd": float(scores[position]),
            "observation_status": "complete",
            "observation_error": "",
        }
        for position, seed in enumerate(record.metadata["seeds"])
    ]


def _reference_records_frame(
    records: Sequence[CompletedGenerationRecord], model_name: str
) -> pd.DataFrame:
    rows = []
    for completed in records:
        metadata = completed.metadata
        row = {
            "model_name": model_name,
            "original_index": completed.original_index,
            "record_id": metadata.get("record_id"),
            "source_row_number": completed.source_row_number,
            "prompt": metadata.get("prompt_raw"),
            "kind": normalize_webster_type(metadata.get("webster_overfit_type")),
            "target_image_sha256": metadata.get("target_image_sha256"),
        }
        if (
            not isinstance(row["record_id"], str)
            or not row["record_id"]
            or not isinstance(row["prompt"], str)
            or not _sha256(row["target_image_sha256"])
        ):
            raise ProximityError(
                "cannot establish prompt identity for reference record "
                f"{completed.original_index}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _failed_reference_rows(
    root: Path,
    paths: GenerationPaths,
    completed: CompletedGenerationRecord,
    generation: Mapping[str, object],
    error: Exception,
    *,
    model_name: str,
) -> list[dict[str, object]]:
    identity = _reference_records_frame([completed], model_name).iloc[0].to_dict()
    science = generation["scientific_config"]
    assert isinstance(science, Mapping)
    common = {
        **identity,
        "generated_image_path": _display_path(
            paths.image_path(completed.original_index), root
        ),
    }
    message = f"{type(error).__name__}: {error}"
    return [
        {
            **common,
            "seed": int(seed),
            "generated_image_tile_index": position,
            "l2_norm": float("nan"),
            "sscd": float("nan"),
            "observation_status": "cache_error",
            "observation_error": message,
        }
        for position, seed in enumerate(science["seeds"])
    ]


def _annotate_selection(
    frame: pd.DataFrame,
    selection: Any,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    prompts = selection.prompt_frame.copy(deep=True)
    prompts["original_index"] = prompts["original_index"].astype(str)
    observations = frame.copy(deep=True)
    observations["original_index"] = observations["original_index"].astype(str)
    identity = (
        "model_name",
        "record_id",
        "source_row_number",
        "prompt",
        "kind",
        "target_image_sha256",
    )
    decisions = (
        "prompt_spearman",
        "include_prompt",
        "selection_status",
        "selection_reason",
    )
    required = {"original_index", *identity, *decisions}
    missing = sorted(required - set(prompts))
    if missing:
        raise ProximityError(
            "frozen selection prompt table is missing: " + ", ".join(missing)
        )
    annotated = observations.merge(
        prompts.loc[:, ["original_index", *identity, *decisions]],
        on="original_index",
        how="left",
        suffixes=("", "_selection"),
        validate="many_to_one",
    )
    if annotated["include_prompt"].isna().any():
        raise ProximityError("experiment contains a prompt absent from selection")
    observed, frozen = (
        set(observations["original_index"]),
        set(prompts["original_index"]),
    )
    if require_complete and observed != frozen:
        raise ProximityError("experiment does not contain every frozen prompt")
    for field in identity:
        other = f"{field}_selection"
        if any(
            canonical_json(left) != canonical_json(right)
            for left, right in zip(annotated[field], annotated[other], strict=True)
        ):
            raise ProximityError(f"experiment and frozen selection differ at {field}")
        annotated = annotated.drop(columns=other)
    annotated["include_prompt"] = annotated["include_prompt"].astype(bool)
    return annotated.loc[:, ANALYSIS_COLUMNS].sort_values(
        ["source_row_number", "original_index", "seed"],
        kind="stable",
        ignore_index=True,
    )


def _analysis_configuration(
    paths: ProximityPaths,
    run_name: str,
    generation: Mapping[str, object],
    sscd: Mapping[str, object],
    selection: Any,
) -> dict[str, object]:
    science = generation["scientific_config"]
    assert isinstance(science, Mapping)
    return {
        "analysis": "proximity",
        "source": "validated_generation_and_sscd_caches_only",
        "run_name": run_name,
        "generation_run_path": _display_path(paths.generation_run, paths.project_root),
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "sscd_configuration_hash": sscd["configuration_hash"],
        "selection_policy": selection.configuration["selection_policy"],
        "selection_hash": selection.sha256,
        "model_name": science["model_cli_name"],
        "scheduler_name": _scheduler_name(science),
        "guidance_scale": science["guidance_scale"],
        "num_inference_steps": science["num_inference_steps"],
        "num_seeds": science["num_seeds"],
        "seeds": science["seeds"],
        "terminal_latent_index": -1,
        "distance": "euclidean_l2",
        "outputs": {
            "table": "proximity.csv",
            "figure": "proximity_vs_sscd.png",
        },
    }


def _write_configuration(path: Path, desired: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ProximityError(f"unsafe existing analysis configuration: {path}")
        if canonical_json(read_json(path)) != canonical_json(desired):
            raise ProximityError(
                f"existing analysis configuration is incompatible: {path}"
            )
        return
    atomic_write_json(path, desired)


def _summary(
    paths: ProximityPaths,
    run_name: str,
    generation: Mapping[str, object],
    sscd: Mapping[str, object],
    selection: Any,
    analysis: pd.DataFrame,
    *,
    duration: float,
    statistics: AnalysisStatistics | None = None,
    failures: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    prompts = selection.prompt_frame
    included = prompts["include_prompt"].astype(bool)
    unusable = prompts["selection_status"].eq("unusable_reference_observations")
    values: dict[str, object] = {
        "complete": not failures,
        "run_name": run_name,
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "sscd_configuration_hash": sscd["configuration_hash"],
        "selection_policy": selection.configuration["selection_policy"],
        "selection_hash": selection.sha256,
        "total_prompt_count": len(prompts),
        "included_prompt_count": int(included.sum()),
        "discarded_prompt_count": int((~included & ~unusable).sum()),
        "unusable_prompt_count": int(unusable.sum()),
        "observation_count": len(analysis),
        "included_observation_count": int(analysis["include_prompt"].sum()),
        "failed_rows": len(failures),
        "run_duration_seconds": duration,
        "finished_at_utc": utc_now(),
        "proximity_csv": _display_path(
            paths.output_directory / "proximity.csv", paths.project_root
        ),
    }
    if failures:
        values["failed_csv"] = _display_path(paths.failed_csv, paths.project_root)
    else:
        assert statistics is not None
        values["prompt_spearman_summary"] = statistics.as_dict()
        values["figure"] = _display_path(
            paths.output_directory / "proximity_vs_sscd.png", paths.project_root
        )
    return values


def _canonical_score_path(root: Path, run: Path, value: object, expected: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ProximityError("cached score path is invalid")
    given = Path(value)
    candidates = (
        [given.resolve()]
        if given.is_absolute()
        else [(run / given).resolve(), (root / given).resolve()]
    )
    target = expected.resolve()
    if target not in candidates:
        raise ProximityError(f"cached score path is noncanonical: {value}")
    return target


def _require_cached_file(path: Path, digest: object, label: str) -> None:
    if (
        not _sha256(digest)
        or not path.is_file()
        or path.is_symlink()
        or file_sha256(path) != digest
    ):
        raise ProximityError(f"cached {label} file is invalid")


def _failure_row(
    completed: CompletedGenerationRecord, error: Exception, run: Path
) -> dict[str, object]:
    return {
        "original_index": completed.original_index,
        "record_id": completed.metadata.get("record_id"),
        "source_row_number": completed.metadata.get("source_row_number"),
        "issue_type": type(error).__name__,
        "reason": str(error),
        "generation_record_path": str(completed.marker_path),
        "sscd_record_path": str(
            run / "sscd_record" / f"{completed.original_index}.json"
        ),
    }


def _remove_optional(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise ProximityError(f"derived output is not a regular file: {path}")


def _latent_shape(value: object) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProximityError("latent_shape is invalid")
    shape = tuple(value)
    if len(shape) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in shape
    ):
        raise ProximityError("latent_shape is invalid")
    return shape  # type: ignore[return-value]


def _scheduler_name(science: Mapping[str, object]) -> str:
    scheduler = science.get("scheduler")
    name = scheduler.get("name") if isinstance(scheduler, Mapping) else None
    if not isinstance(name, str) or not name:
        raise ProximityError("generation scheduler name is missing")
    return name


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sscd_command(
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    seeds: int,
    seed_start: int,
) -> str:
    return (
        f"./sscd.sh --model {model} --scheduler {scheduler} "
        f"--g {stable_float(guidance)} --T {steps} --N {seeds} "
        f"--seed-start {seed_start}"
    )


def _generation_command(
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    seeds: int,
    seed_start: int,
) -> str:
    return (
        f"./generate.sh --model {model} --scheduler {scheduler} "
        f"--g {stable_float(guidance)} --T {steps} --N {seeds} "
        f"--seed-start {seed_start}"
    )
