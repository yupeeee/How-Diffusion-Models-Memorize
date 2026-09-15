"""Shared cache-only candidate measurements with independent resumable stages.

The endpoint cache opens each protected record once and stores compact scalar
segment logits, never reconstructed clean trajectories. Integration consumes
those saved logits and the unchanged derived candidate bank. Main figures and
schema-3 numerical publications are read-only inputs to this extension.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import io
from importlib.metadata import version as package_version
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import torch

from utils.common.io import (
    CacheIOError,
    atomic_write_bytes,
    atomic_write_frame_parquet,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.models.devices import (
    configure_worker_cpu_threads,
    round_robin_shard,
    worker_count_for_tasks,
)
from .cache_reader import discover_sources, load_record
from .candidate_contracts import (
    FORMULA_VERSION,
    ROW_KEYS,
    SCHEMA_VERSION,
    candidate_parent,
    publish_candidate_index,
    read_tables,
    validate_candidate_bundle,
)
from .contracts import TheoryError, find_analysis_bundle, validate_source_metadata
from .metrics import clean_estimates
from .progress import RecordProgress, install_progress_queue, report_worker_record
from .reduce import (
    _json_safe,
    _resolve_theory_devices,
    _scalar_rows,
    _validate_shard,
    _worker_sources,
)
from .scheduler_adapter import SchedulerAdapter
from .support import FiniteSupport

AUXILIARY = ("center.pt", "support.pt", "evaluation_initial.pt", "worker_schedule.pt")
ENDPOINT_SOURCE_FILES = (
    "candidate_feedback.py",
    "candidate_metrics.py",
    "candidate_reduce.py",
    "cache_reader.py",
    "metrics.py",
    "scheduler_adapter.py",
    "support.py",
    "feedback.py",
)
SHARED_ENDPOINT_VERSION = "shared-base-candidate-endpoints-1"
SHARED_ENDPOINT_DIRECTORY = "candidate_endpoint_extension"


def _copy_immutable(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and file_sha256(source) == file_sha256(destination):
        return
    temporary = destination.with_name("." + destination.name + f".{os.getpid()}.copy")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _source_hashes(names):
    directory = Path(__file__).parent
    return {name: file_sha256(directory / name) for name in names}


def _save_segments(destination, entries):
    arrays, index = {}, []
    for i, (keys, segment) in enumerate(entries):
        item = {"keys": keys, "arrays": {}, "values": {}}
        for name, value in segment.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                key = f"b{i}_{name}"
                if value.dtype.hasobject:
                    raise TheoryError(
                        f"Object array cannot enter segment cache: {name}"
                    )
                arrays[key] = value
                item["arrays"][name] = key
            else:
                item["values"][name] = _json_safe(value)
        index.append(item)
    arrays["__index__"] = np.asarray(json.dumps(index, allow_nan=False))
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    atomic_write_bytes(destination, stream.getvalue())


def _read_segments(path, device):
    with np.load(path, allow_pickle=False) as stored:
        for entry in json.loads(str(stored["__index__"])):
            payload = dict(entry["values"])
            payload.update(
                {
                    name: torch.as_tensor(stored[key], device=device)
                    for name, key in entry["arrays"].items()
                }
            )
            yield entry["keys"], payload


def _list_rows(payload, count):
    """Keep dose/control vectors as list cells, with one row per sample-step."""
    rows = [dict() for _ in range(count)]
    for name, value in payload.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.ndim >= 2 and len(value) == count:
                values = value.tolist()
            elif (
                value.ndim == 1
                and name
                not in {
                    "lambda",
                    "lambda_grid",
                    "dose_lambda",
                    "control_atom_indices",
                    "control_atoms",
                }
                and len(value) == count
            ):
                values = value.tolist()
            else:
                values = [value.tolist()] * count
        elif isinstance(value, (list, tuple)):
            values = [list(value)] * count
        else:
            values = [value] * count
        for row, item in zip(rows, values, strict=True):
            row[name] = _json_safe(item)
    return rows


def _worker_context(cache, device, candidate_chunk_size, query_chunk_size):
    payload = {}
    metadata = read_json(cache / "cache_identity.json")
    for name in AUXILIARY:
        path = cache / name
        if file_sha256(path) != metadata["auxiliary_files"][name]:
            raise TheoryError(f"Changed candidate auxiliary input: {path}")
        payload[name] = safe_torch_load(path)
    support_meta = read_json(cache / "support_metadata.json")
    support = FiniteSupport(
        payload["support.pt"].to(device),
        support_meta["atom_ids"],
        support_meta["aliases"],
        weights=support_meta["weights"],
        candidate_chunk=candidate_chunk_size,
        query_chunk=query_chunk_size,
    )
    schedule = payload["worker_schedule.pt"]
    adapter = SchedulerAdapter(
        schedule, recorded_diffusers_version=metadata.get("recorded_diffusers_version")
    )
    return (
        support,
        payload["center.pt"].to(device),
        schedule,
        adapter,
        payload["evaluation_initial.pt"],
    )


def _endpoint_record(
    sources,
    record,
    base_bundle,
    support,
    center,
    schedule,
    adapter,
    *,
    device,
    query_chunk_size,
    bank_hash,
    loaded_record=None,
    base_frames=None,
):
    from .candidate_feedback import endpoint_metrics, unavailable_endpoint
    from .candidate_metrics import basic_step_metrics, initial_retrieval_metrics

    # A fused base worker supplies this immutable record and its freshly reduced
    # frames. Legacy standalone callers retain the existing validated disk path.
    z, epsilon_u, epsilon_c, target = (
        load_record(sources.experiment, record)
        if loaded_record is None
        else loaded_record
    )
    if base_frames is None:
        base_path = (
            base_bundle / "trajectory_metrics" / f"part-{record.original_index}.parquet"
        )
        base_manifest = read_json(base_bundle / "manifest.json")
        if (
            file_sha256(base_path)
            != base_manifest["numerical_files"][
                base_path.relative_to(base_bundle).as_posix()
            ]
        ):
            raise TheoryError(f"Changed base scalar shard: {base_path}")
        base = pd.read_parquet(base_path)
    else:
        if loaded_record is None or len(base_frames) != 3:
            raise TheoryError(
                "Shared endpoint reduction requires one loaded record and all three base frames"
            )
        base = base_frames[0]
    _validate_shard(
        base,
        record,
        sources.config["num_seeds"],
        sources.config["num_inference_steps"],
        sources.runs["experiment"]["scientific_config_hash"],
    )
    lookup = base.set_index(["seed", "step_index"])
    count, steps = z.shape[:2]
    steps -= 1
    g = sources.config["guidance_scale"]
    target = target.to(device)
    support_target_id = str(base.support_target_id.iloc[0])
    rows, doses, controls, segments = [], [], [], []
    for k in range(steps):
        coef = adapter.coefficients(k)
        for start in range(0, count, query_chunk_size):
            end = min(count, start + query_chunk_size)
            state, next_state = (
                z[start:end, k].to(device),
                z[start:end, k + 1].to(device),
            )
            epsu, epsc = (
                epsilon_u[start:end, k].to(device),
                epsilon_c[start:end, k].to(device),
            )
            mu, mc, _, _ = clean_estimates(state, epsu, epsc, coef.alpha, coef.sigma, g)
            seeds = record.metadata["seeds"][start:end]
            source_rows = [
                lookup.loc[(seed, k)].to_dict() | {"seed": seed, "step_index": k}
                for seed in seeds
            ]
            basic = basic_step_metrics(
                mu,
                mc,
                target,
                g,
                center=center if k == 0 else None,
                source_epsilon=torch.finfo(epsu.dtype).eps,
            )
            if k == 0:
                if support_target_id in support.aliases:
                    basic.update(
                        initial_retrieval_metrics(mu, mc, support, support_target_id)
                    )
                else:
                    basic.update(
                        initial_retrieval_status="unavailable_target_missing_from_candidate_bank",
                        initial_conditional_target_rank=None,
                        initial_unconditional_target_rank=None,
                        initial_conditional_target_tie_count=None,
                        initial_unconditional_target_tie_count=None,
                        initial_conditional_retrieval_valid=False,
                        initial_unconditional_retrieval_valid=False,
                    )
            for row, additional in zip(
                source_rows, _scalar_rows(basic, end - start), strict=True
            ):
                row.update(additional)
            eligible = (
                k < steps - 1
                and coef.affine
                and coef.B > 0
                and coef.destination_sigma > 0
            )
            if eligible:
                _, matched, _ = adapter.matched_update(
                    state, next_state, epsu, epsc, g, k
                )
                sensitivity = torch.tensor(
                    [
                        row["clean_estimate_rounding_sensitivity_rmse"]
                        for row in source_rows
                    ],
                    dtype=torch.float64,
                    device=device,
                ) * math.sqrt(target.numel())
                batch = endpoint_metrics(
                    support,
                    state,
                    mc,
                    mu,
                    matched,
                    next_state,
                    support_target_id,
                    current_alpha=coef.alpha,
                    current_sigma=coef.sigma,
                    destination_alpha=coef.destination_alpha,
                    destination_sigma=coef.destination_sigma,
                    guidance=g,
                    kappa=coef.B,
                    bank_hash=bank_hash,
                    source_error_l2=sensitivity,
                )
            else:
                batch = unavailable_endpoint(
                    end - start,
                    device=device,
                    reason="terminal_or_nonaffine_or_nonpositive_destination",
                    guidance=g,
                )
            additions = _scalar_rows(batch.scalars, end - start)
            keys = []
            for row, extra in zip(source_rows, additions, strict=True):
                for name, value in extra.items():
                    if name in row:
                        row.setdefault("base_v3_" + name, row[name])
                    row[name] = value
                row["candidate_endpoint_eligible"] = eligible
                target_atom = support.aliases.get(support_target_id)
                row["candidate_target_atom_id"] = (
                    support.atom_ids[target_atom] if target_atom is not None else None
                )
                keys.append({name: row[name] for name in ROW_KEYS})
            contexts = [
                key
                | {
                    name: row[name]
                    for name in ("snr", "terminal_sscd", "record_id", "target_group")
                    if name in row
                }
                | {"source_trajectory_terminal_sscd": row["terminal_sscd"]}
                for key, row in zip(keys, source_rows, strict=True)
            ]
            if batch.dose:
                doses.extend(
                    key | value
                    for key, value in zip(
                        contexts, _list_rows(batch.dose, end - start), strict=True
                    )
                )
            if batch.controls:
                controls.extend(
                    key | value
                    for key, value in zip(
                        contexts, _list_rows(batch.controls, end - start), strict=True
                    )
                )
            if batch.segment:
                segments.append((keys, batch.segment))
            rows.extend(source_rows)
    frame = (
        pd.DataFrame(rows).sort_values(["seed", "step_index"]).reset_index(drop=True)
    )
    _validate_shard(
        frame,
        record,
        count,
        steps,
        sources.runs["experiment"]["scientific_config_hash"],
    )
    initial = frame[frame.step_index.eq(0)].copy()
    old_initial = (
        pd.read_parquet(
            base_bundle / "initial_shards" / f"part-{record.original_index}.parquet"
        )
        if base_frames is None
        else base_frames[1]
    )
    for name in old_initial.columns.difference(initial.columns):
        initial[name] = old_initial.set_index("seed").loc[initial.seed, name].to_numpy()
    endpoint = frame[frame.step_index.eq(steps - 1)].copy()
    old_endpoint = (
        pd.read_parquet(
            base_bundle / "endpoint_shards" / f"part-{record.original_index}.parquet"
        )
        if base_frames is None
        else base_frames[2]
    )
    for name in old_endpoint.columns.difference(endpoint.columns):
        endpoint[name] = (
            old_endpoint.set_index("seed").loc[endpoint.seed, name].to_numpy()
        )
    # The existing protected proximity check/joins remain present in old endpoint scalars.
    proximity = sources.proximity[
        sources.proximity.original_index.eq(record.original_index)
    ].set_index("seed")
    for data in (initial, endpoint, frame):
        data["terminal_proximity_l2"] = [
            float(proximity.loc[seed, "l2_norm"]) for seed in data.seed
        ]
        data["terminal_proximity_rmse"] = data.terminal_proximity_l2 / math.sqrt(
            target.numel()
        )
    return (
        frame,
        initial,
        endpoint,
        pd.DataFrame(doses),
        pd.DataFrame(controls),
        segments,
    )


def _stage_paths(directory, record, stage):
    ident = record.original_index
    if stage == "endpoints":
        return [
            directory / name / f"part-{ident}.parquet"
            for name in (
                "trajectory_metrics",
                "initial_shards",
                "endpoint_shards",
                "dose_shards",
                "controls_shards",
            )
        ] + [directory / "segments" / f"part-{ident}.npz"]
    return [directory / "integration_shards" / f"part-{ident}.parquet"]


def prepare_shared_endpoint_extension(
    base_bundle,
    *,
    analysis_hash,
    config,
    support_metadata,
    candidate_chunk_size,
    query_chunk_size,
    backend_types,
):
    """Pin a provisional derived-only endpoint recipe before the base pass."""
    from .candidate_feedback import ENDPOINT_POLICY

    identity = {
        "version": SHARED_ENDPOINT_VERSION,
        "base_analysis_hash": analysis_hash,
        "config": config,
        "support_metadata_hash": canonical_hash(support_metadata),
        "source_code": _source_hashes(ENDPOINT_SOURCE_FILES),
        "endpoint_policy": ENDPOINT_POLICY,
        "chunk_sizes": [candidate_chunk_size, query_chunk_size],
        "backend_types": backend_types,
    }
    contract = identity | {"extension_hash": canonical_hash(identity)}
    path = Path(base_bundle) / SHARED_ENDPOINT_DIRECTORY / "extension_identity.json"
    try:
        existing = read_json(path) if path.is_file() else None
    except CacheIOError:
        existing = None
    if existing != contract:
        atomic_write_json(path, contract)
    return contract


def _shared_endpoint_contract(base_bundle):
    path = Path(base_bundle) / SHARED_ENDPOINT_DIRECTORY / "extension_identity.json"
    if path.is_symlink():
        raise TheoryError("Shared endpoint identity cannot be a symbolic link")
    contract = read_json(path)
    identity = {
        name: value for name, value in contract.items() if name != "extension_hash"
    }
    if (
        contract.get("version") != SHARED_ENDPOINT_VERSION
        or canonical_hash(identity) != contract.get("extension_hash")
        or contract.get("source_code") != _source_hashes(ENDPOINT_SOURCE_FILES)
    ):
        raise TheoryError("Shared endpoint recipe changed or is incompatible")
    return contract


def shared_endpoint_record_valid(base_bundle, record):
    """Validate compact extension bytes, never reopen a raw trajectory."""
    try:
        contract = _shared_endpoint_contract(base_bundle)
        directory = Path(base_bundle) / SHARED_ENDPOINT_DIRECTORY
        paths = _stage_paths(directory, record, "endpoints")
        stamp = directory / "completion/endpoints" / f"{record.original_index}.json"
        if stamp.is_symlink() or not stamp.is_file():
            return False
        if not all(path.is_file() and not path.is_symlink() for path in paths):
            return False
        checkpoint = read_json(stamp)
        return checkpoint.get("stage_hash") == contract[
            "extension_hash"
        ] and checkpoint.get("files") == {
            path.relative_to(directory).as_posix(): file_sha256(path) for path in paths
        }
    except (CacheIOError, OSError, ValueError, TheoryError):
        return False


def stage_shared_endpoint_record(
    sources,
    record,
    base_bundle,
    support,
    center,
    schedule,
    adapter,
    *,
    device,
    query_chunk_size,
    bank_hash,
    loaded_record,
    base_frames,
):
    """Consume the base worker's already-loaded record before releasing it."""
    contract = _shared_endpoint_contract(base_bundle)
    if (
        contract["config"] != sources.config
        or contract["chunk_sizes"][1] != query_chunk_size
    ):
        raise TheoryError("Shared endpoint worker differs from its pinned recipe")
    directory = Path(base_bundle) / SHARED_ENDPOINT_DIRECTORY
    result = _endpoint_record(
        sources,
        record,
        base_bundle,
        support,
        center,
        schedule,
        adapter,
        device=device,
        query_chunk_size=query_chunk_size,
        bank_hash=bank_hash,
        loaded_record=loaded_record,
        base_frames=base_frames,
    )
    paths = _stage_paths(directory, record, "endpoints")
    for frame, path in zip(result[:5], paths[:5], strict=True):
        atomic_write_frame_parquet(frame, path)
    _save_segments(paths[-1], result[-1])
    atomic_write_json(
        directory / "completion/endpoints" / f"{record.original_index}.json",
        {
            "stage_hash": contract["extension_hash"],
            "files": {
                path.relative_to(directory).as_posix(): file_sha256(path)
                for path in paths
            },
        },
    )


def shared_endpoint_extension_manifest(base_bundle, records):
    """The final base publication pins every extension completion marker."""
    contract = _shared_endpoint_contract(base_bundle)
    if not all(shared_endpoint_record_valid(base_bundle, record) for record in records):
        raise TheoryError(
            "Shared endpoint extension is incomplete; no base publication"
        )
    directory = Path(base_bundle) / SHARED_ENDPOINT_DIRECTORY
    return {
        "complete": True,
        "contract": contract,
        "contract_file_sha256": file_sha256(directory / "extension_identity.json"),
        "record_indices": sorted(record.original_index for record in records),
        "completion_files": {
            f"completion/endpoints/{record.original_index}.json": file_sha256(
                directory / "completion/endpoints" / f"{record.original_index}.json"
            )
            for record in records
        },
    }


def import_shared_endpoint_extension(
    base_bundle,
    base_manifest,
    destination,
    records,
    endpoint_hash,
    *,
    endpoint_sources,
    candidate_chunk_size,
    query_chunk_size,
    backend_types,
):
    """Import verified fused results into the unchanged ordinary endpoint cache."""
    provenance = base_manifest.get("candidate_endpoint_extension")
    if provenance is None:
        return False
    directory = Path(base_bundle) / SHARED_ENDPOINT_DIRECTORY
    contract = _shared_endpoint_contract(base_bundle)
    if (
        not provenance.get("complete")
        or provenance.get("contract") != contract
        or provenance.get("contract_file_sha256")
        != file_sha256(directory / "extension_identity.json")
        or contract["base_analysis_hash"] != base_manifest["analysis_hash"]
        or contract["config"] != base_manifest["config"]
        or contract["source_code"] != endpoint_sources
        or contract["chunk_sizes"] != [candidate_chunk_size, query_chunk_size]
        or contract["backend_types"] != backend_types
        or contract["support_metadata_hash"]
        != canonical_hash(read_json(Path(base_bundle) / "support_metadata.json"))
    ):
        raise TheoryError(
            "Shared endpoint provenance differs from requested endpoint recipe"
        )
    for record in records:
        relative_stamp = f"completion/endpoints/{record.original_index}.json"
        source_stamp = directory / relative_stamp
        if (
            record.original_index not in provenance["record_indices"]
            or not shared_endpoint_record_valid(base_bundle, record)
            or provenance["completion_files"].get(relative_stamp)
            != file_sha256(source_stamp)
        ):
            raise TheoryError(
                f"Incomplete or changed shared endpoints: {record.original_index}; resume base preparation"
            )
        paths = _stage_paths(directory, record, "endpoints")
        copied = _stage_paths(destination, record, "endpoints")
        for source, target in zip(paths, copied, strict=True):
            _copy_immutable(source, target)
        checkpoint = {
            "stage_hash": endpoint_hash,
            "files": {
                path.relative_to(destination).as_posix(): file_sha256(path)
                for path in copied
            },
        }
        target_stamp = destination / relative_stamp
        if not target_stamp.is_file() or read_json(target_stamp) != checkpoint:
            atomic_write_json(target_stamp, checkpoint)
    return True


def _stage_worker(
    *,
    sources,
    records,
    cache,
    base_bundle,
    destination,
    stage,
    stage_hash,
    device,
    worker_index,
    worker_count,
    candidate_chunk_size,
    query_chunk_size,
    progress=None,
):
    started = time.perf_counter()
    threads = configure_worker_cpu_threads(worker_count)
    concrete = torch.device(device)
    if concrete.type == "cuda":
        torch.cuda.set_device(concrete)
    report = {
        "worker_index": worker_index,
        "device": str(concrete),
        "pid": os.getpid(),
        "cpu_threads": threads,
        "assigned_indices": [r.original_index for r in records],
        "status_by_index": {},
        "failed_rows": [],
    }
    report_path = destination / "worker_reports" / stage / f"part-{worker_index}.json"
    try:
        support, center, schedule, adapter, _ = _worker_context(
            cache, concrete, candidate_chunk_size, query_chunk_size
        )
        bank_hash = read_json(cache / "support_metadata.json")["tensor_sha256"]
        for record in records:
            paths = _stage_paths(destination, record, stage)
            stamp = destination / "completion" / stage / f"{record.original_index}.json"
            try:
                resumed = False
                if stamp.is_file() and all(p.is_file() for p in paths):
                    checkpoint = read_json(stamp)
                    resumed = checkpoint.get(
                        "stage_hash"
                    ) == stage_hash and checkpoint.get("files") == {
                        p.relative_to(destination).as_posix(): file_sha256(p)
                        for p in paths
                    }
                if not resumed:
                    with torch.inference_mode():
                        if stage == "endpoints":
                            result = _endpoint_record(
                                sources,
                                record,
                                base_bundle,
                                support,
                                center,
                                schedule,
                                adapter,
                                device=concrete,
                                query_chunk_size=query_chunk_size,
                                bank_hash=bank_hash,
                            )
                            for frame, path in zip(result[:5], paths[:5], strict=True):
                                atomic_write_frame_parquet(frame, path)
                            _save_segments(paths[-1], result[-1])
                        else:
                            from .candidate_integration import (
                                integration_metrics,
                                unavailable_integration,
                            )

                            segment_path = (
                                cache / "segments" / f"part-{record.original_index}.npz"
                            )
                            checkpoint = read_json(
                                cache
                                / "completion/endpoints"
                                / f"{record.original_index}.json"
                            )
                            if (
                                file_sha256(segment_path)
                                != checkpoint["files"][
                                    segment_path.relative_to(cache).as_posix()
                                ]
                            ):
                                raise TheoryError(
                                    f"Changed segment payload: {segment_path}"
                                )
                            rows = []
                            for keys, payload in _read_segments(segment_path, concrete):
                                values = integration_metrics(
                                    support, payload, bank_hash=bank_hash
                                )
                                rows.extend(
                                    key | row
                                    for key, row in zip(
                                        keys,
                                        _scalar_rows(values, len(keys)),
                                        strict=True,
                                    )
                                )
                            population = pd.read_parquet(
                                cache
                                / "trajectory_metrics"
                                / f"part-{record.original_index}.parquet",
                                columns=ROW_KEYS,
                            )
                            defaults = _scalar_rows(
                                unavailable_integration(
                                    len(population),
                                    device=concrete,
                                    reason="endpoint_segment_not_applicable_or_unavailable",
                                ),
                                len(population),
                            )
                            all_rows = {
                                tuple(key[name] for name in ROW_KEYS): key | default
                                for key, default in zip(
                                    population.to_dict("records"), defaults, strict=True
                                )
                            }
                            seen = set()
                            for row in rows:
                                key = tuple(row[name] for name in ROW_KEYS)
                                if key not in all_rows or key in seen:
                                    raise TheoryError(
                                        "Orphan/duplicate integrated sample identity"
                                    )
                                seen.add(key)
                                all_rows[key].update(row)
                            atomic_write_frame_parquet(
                                pd.DataFrame(all_rows.values()), paths[0]
                            )
                    atomic_write_json(
                        stamp,
                        {
                            "stage_hash": stage_hash,
                            "files": {
                                p.relative_to(destination).as_posix(): file_sha256(p)
                                for p in paths
                            },
                        },
                    )
                report["status_by_index"][record.original_index] = (
                    "resumed" if resumed else "reduced"
                )
            except Exception as error:
                report["status_by_index"][record.original_index] = "failed"
                report["failed_rows"].append(
                    {
                        "original_index": record.original_index,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
            report["duration_seconds"] = time.perf_counter() - started
            atomic_write_json(report_path, _json_safe(report))
            if progress:
                progress(
                    record.original_index,
                    report["status_by_index"][record.original_index],
                    concrete,
                )
    except Exception as error:
        for record in records:
            if record.original_index not in report["status_by_index"]:
                report["status_by_index"][record.original_index] = "failed"
                report["failed_rows"].append(
                    {
                        "original_index": record.original_index,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                if progress:
                    progress(record.original_index, "failed", concrete)
    report["duration_seconds"] = time.perf_counter() - started
    atomic_write_json(report_path, _json_safe(report))
    return report


def _subprocess(**kwargs):
    return _stage_worker(progress=report_worker_record, **kwargs)


def _dispatch(
    *,
    sources,
    records,
    cache,
    base_bundle,
    destination,
    stage,
    stage_hash,
    devices,
    candidate_chunk_size,
    query_chunk_size,
):
    workers = worker_count_for_tasks(devices, len(records))
    if not workers:
        raise TheoryError("Candidate analysis has no retained records")
    shards = [
        round_robin_shard(records, worker_index=i, worker_count=workers)
        for i in range(workers)
    ]
    arguments = [
        dict(
            sources=_worker_sources(sources, shard),
            records=shard,
            cache=cache,
            base_bundle=base_bundle,
            destination=destination,
            stage=stage,
            stage_hash=stage_hash,
            device=str(devices[i]),
            worker_index=i,
            worker_count=workers,
            candidate_chunk_size=candidate_chunk_size,
            query_chunk_size=query_chunk_size,
        )
        for i, shard in enumerate(shards)
    ]
    context = get_context("spawn")
    print(
        f"[candidates] {stage}: {len(records)} records on {workers} device(s)",
        flush=True,
    )
    with RecordProgress(
        total=len(records), devices=workers, context=context
    ) as progress:
        if workers == 1:
            results = [_stage_worker(progress=progress.report, **arguments[0])]
        else:
            with ProcessPoolExecutor(
                max_workers=workers - 1,
                mp_context=context,
                initializer=install_progress_queue,
                initargs=(progress.queue,),
            ) as executor:
                futures = [
                    executor.submit(_subprocess, **args) for args in arguments[1:]
                ]
                results = [_stage_worker(progress=progress.report, **arguments[0])]
                for i, future in enumerate(futures, 1):
                    try:
                        results.append(future.result())
                    except Exception as error:
                        results.append(
                            {
                                "worker_index": i,
                                "device": str(devices[i]),
                                "status_by_index": {
                                    r.original_index: "failed" for r in shards[i]
                                },
                                "failed_rows": [
                                    {
                                        "original_index": r.original_index,
                                        "error": str(error),
                                        "traceback": traceback.format_exc(),
                                    }
                                    for r in shards[i]
                                ],
                            }
                        )
        for result in results:
            for record_id, status in result["status_by_index"].items():
                progress.report(record_id, status, result["device"])
    failed = [row for result in results for row in result["failed_rows"]]
    atomic_write_json(
        destination / f"{stage}_execution.json",
        {"stage": stage, "workers": results, "failed_records": len(failed)},
    )
    if failed:
        atomic_write_bytes(
            destination / f"{stage}_failed.csv",
            pd.DataFrame(failed).to_csv(index=False).encode(),
        )
        raise TheoryError(
            f"{len(failed)} candidate {stage} records failed; saved completed shards are resumable. See {destination}/{stage}_failed.csv"
        )
    return results


def _reference_tables(
    cache, base_bundle, device, candidate_chunk_size, query_chunk_size
):
    from .candidate_metrics import (
        bank_geometry,
        initial_reference_metrics,
        reference_only_sweep,
    )

    support, center, schedule, adapter, evaluation = _worker_context(
        cache, device, candidate_chunk_size, query_chunk_size
    )
    z, epsu = (value.to(device) for value in evaluation)
    coef = adapter.coefficients(0)
    mu, _, _, _ = clean_estimates(z, epsu, epsu, coef.alpha, coef.sigma, 1.0)
    baseline = (
        pd.read_parquet(base_bundle / "plotdata/unconditional_center.parquet")
        .sort_values("seed")
        .reset_index(drop=True)
    )
    if len(baseline) != len(z) or baseline.seed.duplicated().any():
        raise TheoryError(
            "Candidate baseline must contain one row per unique evaluation seed"
        )
    additions = _scalar_rows(
        initial_reference_metrics(z, mu, support, coef.alpha, coef.sigma), len(z)
    )
    for column in additions[0]:
        baseline[column] = [row[column] for row in additions]
    initial_snr = coef.alpha**2 / coef.sigma**2
    atomic_write_frame_parquet(baseline, cache / "reference_initial.parquet")
    atomic_write_frame_parquet(
        reference_only_sweep(z, baseline.seed.tolist(), support, initial_snr),
        cache / "reference_snr.parquet",
    )
    atomic_write_frame_parquet(
        bank_geometry(support, center), cache / "bank_geometry.parquet"
    )


def _scalar_hashes(bundle):
    excluded = {
        "analysis_manifest.json",
        "cache_complete.json",
        "index.json",
        "figure_manifest.json",
        "candidate_figure_manifest.json",
        "candidate_render_audit.json",
        "comparison_summary.csv",
    }
    return {
        path.relative_to(bundle).as_posix(): file_sha256(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
        and path.suffix in {".parquet", ".csv", ".json"}
        and path.name not in excluded
        and not any(
            part
            in {
                "IR",
                "UB",
                "IA",
                "MD",
                "PF",
                "TS",
                "TR",
                "qa",
                "last_prediction_geometry",
                "archive",
            }
            for part in path.relative_to(bundle).parts
        )
    }


def _concatenate(directory, records, shard_dir, destination):
    frames = [
        pd.read_parquet(directory / shard_dir / f"part-{record.original_index}.parquet")
        for record in records
    ]
    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["original_index", "seed"])
        .reset_index(drop=True)
    )
    atomic_write_frame_parquet(frame, directory / destination)


def _write_candidate_manifest(
    bundle, metadata, *, complete=False, integration_status="pending"
):
    value = metadata | {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "figure_suite": "candidates",
        "endpoint_complete": True,
        "complete": bool(complete),
        "integration_status": integration_status,
        "numerical_files": _scalar_hashes(bundle),
    }
    atomic_write_json(bundle / "analysis_manifest.json", _json_safe(value))
    return value


def _summarize(bundle, metadata, *, complete, integration_status):
    from .candidate_summaries import build_candidate_summaries

    # Only the final write below publishes a completed scientific bundle.
    # A reader must not mistake previous endpoint summaries for integrated ones.
    _write_candidate_manifest(
        bundle, metadata, complete=False, integration_status=integration_status
    )
    tables = read_tables(bundle)
    summaries = build_candidate_summaries(
        tables["trajectory"],
        initial=tables["initial"],
        baseline=tables["reference_initial"],
        endpoint=tables["endpoint"],
        dose=tables["dose"],
        controls=tables["controls"],
        eligible_steps=metadata["eligible_steps"],
        reference_sweep=tables["reference_snr"],
    )
    for name, frame in summaries.items():
        if not isinstance(frame, pd.DataFrame):
            raise TheoryError(f"Candidate summary {name} must be a scalar DataFrame")
        atomic_write_frame_parquet(frame, bundle / "summaries" / f"{name}.parquet")
    trajectory = tables["trajectory"]
    numerical = {
        "endpoint_population": len(trajectory),
        "endpoint_status_counts": trajectory.candidate_gain_status.value_counts(
            dropna=False
        ).to_dict(),
        "retained_prompt_count": trajectory.original_index.nunique(),
        "retained_sample_count": len(tables["initial"]),
        "integration_status": integration_status,
        "no_positive_trend_required": True,
        "base_v3_condition_counts": trajectory.candidate_condition_status.value_counts(
            dropna=False
        ).to_dict()
        if "candidate_condition_status" in trajectory
        else {},
        "evidence": "descriptive_candidate_reference_and_learned_behavior; no_automatic_figure_selection",
    }
    for name in trajectory:
        if name.endswith("_status") and name.startswith("candidate_margin_"):
            numerical[name] = trajectory[name].value_counts(dropna=False).to_dict()
    atomic_write_json(bundle / "numerical_audit.json", _json_safe(numerical))
    del tables, trajectory
    return _write_candidate_manifest(
        bundle, metadata, complete=complete, integration_status=integration_status
    )


def _ensure_sources_unchanged(sources):
    for relative, expected in sources.metadata_files.items():
        if file_sha256(sources.root / relative) != expected:
            raise TheoryError(
                f"Protected source metadata changed during candidate reduction: {relative}"
            )


def run_candidates(
    project_root,
    *,
    base_bundle=None,
    recompute=False,
    device="auto",
    candidate_chunk_size=256,
    query_chunk_size=16,
    render=True,
    endpoint_only=False,
    smoke_record_limit=None,
    **config,
):
    """Extend valid base scalars; only changed stage identities need recomputation.

    endpoint_only and smoke_record_limit are explicit internal audit interfaces;
    CLI full mode always requires every requested stage and frozen selected row.
    A failed replacement never publishes over an existing completed candidate.
    """
    from .candidate_feedback import ENDPOINT_POLICY
    from .candidate_integration import INTEGRATION_POLICY
    from .candidate_registry import (
        candidate_registry,
        feedback_snapshots,
        synchronization_snapshots,
    )
    from .plotting import validate_bundle

    devices = _resolve_theory_devices(device)
    sources = discover_sources(project_root, **config)
    base_bundle = (
        Path(base_bundle)
        if base_bundle is not None
        else find_analysis_bundle(project_root, **config)
    )
    base_manifest = validate_bundle(base_bundle, expected_config=sources.config)
    validate_source_metadata(base_bundle, project_root=project_root)
    if base_manifest["selection_hash"] != sources.selection.sha256:
        raise TheoryError(
            "Candidate source selection differs from the completed main analysis"
        )
    pdf = sources.root / "revised.pdf"
    expected_pdf = "fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f"
    if pdf.exists() and file_sha256(pdf) != expected_pdf:
        raise TheoryError(
            "revised.pdf differs from the audited candidate formula mapping"
        )
    records = sorted(sources.selected, key=lambda r: r.original_index)
    if smoke_record_limit is not None:
        records = records[:smoke_record_limit]
    endpoint_sources = _source_hashes(ENDPOINT_SOURCE_FILES)
    endpoint_identity = {
        "base_analysis_hash": base_bundle.name,
        "base_manifest_sha256": file_sha256(base_bundle / "manifest.json"),
        "config": sources.config,
        "source_code": endpoint_sources,
        "endpoint_policy": ENDPOINT_POLICY,
        "backend_types": sorted({d.type for d in devices}),
        "chunk_sizes": [candidate_chunk_size, query_chunk_size],
        "record_indices": [r.original_index for r in records],
        "smoke_record_limit": smoke_record_limit,
        "manuscript_sha256": expected_pdf,
        "package_versions": {
            name: package_version(name)
            for name in ("torch", "numpy", "pandas", "pyarrow")
        },
    }
    endpoint_hash = canonical_hash(endpoint_identity)
    parent = candidate_parent(project_root, **sources.config)
    cache = parent / ".endpoint_cache" / endpoint_hash
    cache.mkdir(parents=True, exist_ok=True)
    auxiliary = {
        name: base_manifest["auxiliary_tensor_files"][name] for name in AUXILIARY
    }
    for name in AUXILIARY:
        if file_sha256(base_bundle / name) != auxiliary[name]:
            raise TheoryError(f"Base auxiliary hash differs: {base_bundle / name}")
        _copy_immutable(base_bundle / name, cache / name)
    for name in ("center_metadata.json", "support_metadata.json"):
        _copy_immutable(base_bundle / name, cache / name)
    _copy_immutable(
        base_bundle / "manifest.json", cache / "base_analysis_manifest.json"
    )
    version = (
        sources.runs["experiment"]["scientific_config"]
        .get("package_versions", {})
        .get("diffusers")
    )
    atomic_write_json(
        cache / "cache_identity.json",
        endpoint_identity
        | {
            "endpoint_hash": endpoint_hash,
            "auxiliary_files": auxiliary,
            "recorded_diffusers_version": version,
        },
    )
    cached = (
        read_json(cache / "cache_complete.json")
        if (cache / "cache_complete.json").exists()
        else None
    )
    if cached and cached.get("endpoint_hash") == endpoint_hash:
        valid_cache = all(
            (cache / relative).is_file() and file_sha256(cache / relative) == expected
            for relative, expected in cached["numerical_files"].items()
        )
        if valid_cache:
            # Segment payloads are deliberately outside the plot-only file contract.
            # Audit them here so a missing payload repairs just its endpoint shard.
            for record in records:
                stamp = cache / "completion/endpoints" / f"{record.original_index}.json"
                checkpoint = read_json(stamp) if stamp.is_file() else {}
                paths = _stage_paths(cache, record, "endpoints")
                if checkpoint.get("stage_hash") != endpoint_hash or not all(
                    path.is_file() for path in paths
                ):
                    valid_cache = False
                    break
                if checkpoint.get("files") != {
                    path.relative_to(cache).as_posix(): file_sha256(path)
                    for path in paths
                }:
                    valid_cache = False
                    break
        if not valid_cache:
            cached = None
    if cached and cached.get("endpoint_hash") == endpoint_hash:
        endpoint_execution = cached["execution"]
    else:
        with torch.inference_mode():
            _reference_tables(
                cache, base_bundle, devices[0], candidate_chunk_size, query_chunk_size
            )
        import_shared_endpoint_extension(
            base_bundle,
            base_manifest,
            cache,
            records,
            endpoint_hash,
            endpoint_sources=endpoint_sources,
            candidate_chunk_size=candidate_chunk_size,
            query_chunk_size=query_chunk_size,
            backend_types=sorted({d.type for d in devices}),
        )
        endpoint_execution = _dispatch(
            sources=sources,
            records=records,
            cache=cache,
            base_bundle=base_bundle,
            destination=cache,
            stage="endpoints",
            stage_hash=endpoint_hash,
            devices=devices,
            candidate_chunk_size=candidate_chunk_size,
            query_chunk_size=query_chunk_size,
        )
        _concatenate(cache, records, "initial_shards", "initial.parquet")
        _concatenate(cache, records, "endpoint_shards", "endpoint.parquet")
        _ensure_sources_unchanged(sources)
        cached = {
            "endpoint_hash": endpoint_hash,
            "execution": endpoint_execution,
            "numerical_files": _scalar_hashes(cache),
        }
        atomic_write_json(cache / "cache_complete.json", cached)
    integration_identity = {
        "endpoint_hash": endpoint_hash,
        "policy": INTEGRATION_POLICY,
        "source_code": _source_hashes(("candidate_integration.py",)),
    }
    integration_hash = canonical_hash(integration_identity)
    registry = candidate_registry()
    scientific_identity = {
        "endpoint_hash": endpoint_hash,
        "integration": integration_identity,
        "fixed_numerical_recipe": {
            key: value
            for key, value in registry["fixed_recipe"].items()
            if key not in {"sscd_color", "display_subsampling"}
        },
        "summary_sources": _source_hashes(
            ("candidate_summaries.py", "candidate_contracts.py")
        ),
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
    }
    digest = canonical_hash(scientific_identity)
    published = parent / digest
    existing = (published / "analysis_manifest.json").is_file()
    replacing = existing and read_json(published / "analysis_manifest.json").get(
        "complete"
    )
    if replacing and not recompute:
        validate_candidate_bundle(
            published, expected_config=sources.config, require_complete=True
        )
        if render:
            from .candidate_plotting import render_candidates

            render_candidates(published)
        if smoke_record_limit is None:
            publish_candidate_index(project_root, sources.config, published)
        return published
    bundle = parent / (".recompute-" + digest) if replacing else published
    bundle.mkdir(parents=True, exist_ok=True)
    if replacing:
        # Preserve valid integration shards while rebuilding derived summaries.
        for directory in ("integration_shards", "completion/integration"):
            if (published / directory).exists():
                for source in (published / directory).glob("*"):
                    if source.is_file():
                        _copy_immutable(source, bundle / source.relative_to(published))
    for relative in cached["numerical_files"]:
        if (
            relative.startswith(("completion/", "worker_reports/"))
            or relative == "cache_identity.json"
        ):
            continue
        _copy_immutable(cache / relative, bundle / relative)
    atomic_write_json(bundle / "registry.json", registry)
    atomic_write_json(bundle / "analysis_identity.json", scientific_identity)
    schedule = safe_torch_load(cache / "worker_schedule.pt")
    adapter = SchedulerAdapter(schedule, recorded_diffusers_version=version)
    steps = sources.config["num_inference_steps"]
    eligible = [
        k
        for k in range(steps - 1)
        if adapter.coefficients(k).affine
        and adapter.coefficients(k).B > 0
        and adapter.coefficients(k).destination_sigma > 0
    ]
    metadata = {
        "analysis_hash": digest,
        "config": sources.config,
        "base_analysis_hash": base_bundle.name,
        "base_bundle": str(base_bundle),
        "endpoint_hash": endpoint_hash,
        "integration_hash": integration_hash,
        "source_root": str(sources.root),
        "source_metadata_files": sources.metadata_files,
        "source_marker_inventory": sources.marker_inventory,
        "source_code": endpoint_sources
        | scientific_identity["summary_sources"]
        | integration_identity["source_code"],
        "endpoint_policy": ENDPOINT_POLICY,
        "integration_policy": INTEGRATION_POLICY,
        "registry_version": registry.get("registry_version"),
        "manuscript_sha256": expected_pdf,
        "center_metadata": read_json(cache / "center_metadata.json"),
        "support_metadata": read_json(cache / "support_metadata.json"),
        "scheduler_adapter": adapter.metadata(),
        "eligible_steps": eligible,
        "feedback_snapshots": feedback_snapshots(eligible),
        "synchronization_snapshots": synchronization_snapshots(
            list(range(steps)), eligible
        ),
        "expected_counts": {
            "prompts": len(records),
            "seeds_per_prompt": sources.config["num_seeds"],
            "initial_rows": len(records) * sources.config["num_seeds"],
            "endpoint_rows": len(records) * sources.config["num_seeds"],
            "trajectory_rows": len(records) * sources.config["num_seeds"] * steps,
        },
        "execution": {
            "requested_device": str(device),
            "resolved_devices": [str(d) for d in devices],
            "endpoint_workers": endpoint_execution,
        },
        "population_policy": "all_frozen_selected_prompts_and_all_evaluation_seeds; no_outcome_pruning",
        "integration_payload_cache": str(cache),
        "smoke_record_limit": smoke_record_limit,
    }
    _summarize(bundle, metadata, complete=False, integration_status="pending")
    if render:
        from .candidate_plotting import render_candidates

        render_candidates(bundle)
    if endpoint_only:
        return bundle
    try:
        integration_execution = _dispatch(
            sources=sources,
            records=records,
            cache=cache,
            base_bundle=base_bundle,
            destination=bundle,
            stage="integration",
            stage_hash=integration_hash,
            devices=devices,
            candidate_chunk_size=candidate_chunk_size,
            query_chunk_size=query_chunk_size,
        )
    except Exception:
        _write_candidate_manifest(
            bundle, metadata, complete=False, integration_status="failed_resumable"
        )
        raise
    metadata["execution"]["integration_workers"] = integration_execution
    _ensure_sources_unchanged(sources)
    _summarize(bundle, metadata, complete=True, integration_status="complete")
    validate_candidate_bundle(
        bundle, expected_config=sources.config, require_complete=True
    )
    if render:
        from .candidate_plotting import render_candidates

        render_candidates(bundle)
    if replacing:
        archive = parent / "archived_analyses"
        archive.mkdir(exist_ok=True)
        previous = archive / (digest + "-" + str(time.time_ns()))
        os.replace(published, previous)
        try:
            os.replace(bundle, published)
        except BaseException:
            os.replace(previous, published)
            raise
        bundle = published
    if smoke_record_limit is None:
        publish_candidate_index(project_root, sources.config, bundle)
    return bundle
