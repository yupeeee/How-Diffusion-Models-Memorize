"""Author-invoked shared stages for seven direct comparisons.

The learned stage completes and releases its replicas before this module starts
analytical workers. One worker owns a whole record and reduces all vector
statements in one read. Only paths, metadata and status receipts cross IPC.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
import traceback

import pandas as pd
import torch

from utils.common.io import (
    atomic_torch_save, atomic_write_frame_parquet, atomic_write_json,
    canonical_hash, file_sha256, safe_torch_load,
)
from utils.models.devices import configure_worker_cpu_threads, round_robin_shard, worker_count_for_tasks
from .cache_reader import discover_sources, load_record, load_schedule, load_scores
from .candidate_integration import DEFAULT_INTEGRATION
from .contracts import DIRECT_FORMULA_VERSION, TheoryError, read_object
from .paper_contracts import PaperPaths, contained_path
from .progress import RecordProgress, install_progress_queue, report_worker_record

TABLES = ("initial", "trajectory", "matched_updates", "terminal")
ANALYSIS_SCHEMA_VERSION = 1
INTEGRATION_PAYLOAD_SCHEMA_VERSION = 2


def core_recipe():
    """Vector observation recipe, independent of integrator, probes and rendering."""
    directory = Path(__file__).parent
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "formula_version": DIRECT_FORMULA_VERSION,
        "source_code": {name: file_sha256(directory / name) for name in (
            "direct_reduce.py", "direct_measurements.py", "reference_law.py",
            "scheduler_adapter.py", "support.py", "feedback.py", "candidate_feedback.py",
            "metrics.py", "cache_reader.py",
        )},
    }


def analytical_recipe():
    """Separately resumable original-variation and signed-gain recipe."""
    directory = Path(__file__).parent
    return {"formula_version": DIRECT_FORMULA_VERSION, "integration": asdict(DEFAULT_INTEGRATION),
            "source_code": {name: file_sha256(directory / name) for name in (
                "direct_integration.py", "candidate_integration.py", "feedback.py", "support.py",
            )}}


def _record_key(record):
    return canonical_hash({"record": record.metadata})


def _paths(bundle, record):
    root = contained_path(bundle, "records/" + _record_key(record))
    return root, {name: root / (name + ".parquet") for name in TABLES}


def _valid_record(bundle, record, analysis_hash):
    root, paths = _paths(bundle, record)
    marker = root / "complete.json"
    if not marker.is_file() or marker.is_symlink():
        return False
    try:
        value = read_object(marker)
        files = value.get("files", {})
        required = {path.relative_to(root).as_posix() for path in paths.values()} | {"audit.json"}
        required.update(value.get("payloads", []))
        return (value.get("complete") is True
                and value.get("analysis_hash") == analysis_hash
                and value.get("record_hash") == _record_key(record)
                and set(files) == required
                and all(contained_path(root, name).is_file() and
                        file_sha256(contained_path(root, name)) == digest for name, digest in files.items()))
    except (TheoryError, OSError, ValueError):
        return False


def _validate_rows(tables, record, config):
    n, steps = config["num_seeds"], config["num_inference_steps"]
    meta = record.metadata
    for name, expected in (("initial", n), ("trajectory", n * steps),
                           ("matched_updates", n * steps), ("terminal", n)):
        frame = tables[name]
        if len(frame) != expected:
            raise TheoryError(f"{record.original_index}/{name}: {len(frame)} rows, expected {expected}; records must not be filtered by outcome")
        keys = ["run_id", "original_index", "record_id", "target_id", "seed", "step_index"]
        if not set(keys) <= set(frame) or frame.duplicated(keys).any():
            raise TheoryError(f"Invalid/duplicated direct scalar identity: {record.original_index}/{name}")
        expected_identity = {"run_id": meta["scientific_config_hash"], "original_index": record.original_index,
                             "record_id": meta["record_id"], "target_id": meta["target_image_sha256"]}
        for key, value in expected_identity.items():
            if not frame[key].astype(str).eq(str(value)).all():
                raise TheoryError(f"Direct scalar {key} differs: {record.original_index}/{name}")
        step_grid = [0] if name == "initial" else [steps - 1] if name == "terminal" else range(steps)
        grid = {(str(seed), int(step)) for seed in meta["seeds"] for step in step_grid}
        actual = set(zip(frame.seed.astype(str), frame.step_index))
        if actual != grid:
            raise TheoryError(f"Direct scalar native step/seed grid differs: {record.original_index}/{name}")


def _cpu_payload(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _cpu_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_payload(item) for item in value]
    return value


def _save_record(root, paths, tables, record, identity, *, payloads=()):
    for name, path in paths.items():
        atomic_write_frame_parquet(tables[name], path)
    atomic_write_json(root / "audit.json", tables["audit"])
    names = [path.relative_to(root).as_posix() for path in paths.values()] + ["audit.json"] + list(payloads)
    atomic_write_json(root / "complete.json", {
        "analysis_hash": identity, "record_hash": _record_key(record),
        "complete": tables["audit"].get("complete", False) is True,
        "files": {name: file_sha256(contained_path(root, name)) for name in names}, "payloads": list(payloads),
    })


def _integrate_record(core, destination, record, law, config, integration_hash):
    """Only saved sufficient statistics and scalars are read; no raw trajectory."""
    from .direct_integration import integrate_proposition5_payload
    core_root, core_paths = _paths(core, record)
    root, paths = _paths(destination, record)
    tables = {name: pd.read_parquet(path) for name, path in core_paths.items()}
    core_marker = read_object(core_root / "complete.json")
    failures = []
    matched = tables["matched_updates"]
    pending = matched.get("direct_prop5_status", pd.Series("", index=matched.index)).eq("integration_required_not_computed")
    expected = {(int(row.step_index), str(row.seed)) for _, row in matched.loc[pending].iterrows()}
    covered = set()
    for relative in core_marker.get("payloads", []):
        payload_path = contained_path(core_root, relative)
        if file_sha256(payload_path) != core_marker["files"][relative]:
            raise TheoryError("Saved direct integration payload changed")
        item = safe_torch_load(payload_path)
        if item.get("schema_version") != INTEGRATION_PAYLOAD_SCHEMA_VERSION:
            raise TheoryError("Saved direct integration payload schema differs; recompute the vector core")
        seeds = list(map(str, item["seeds"]))
        keys = {(int(item["step_index"]), seed) for seed in seeds}
        if not seeds or len(seeds) != len(set(seeds)) or not keys <= expected or keys & covered:
            raise TheoryError("Saved integration payload step/seed coverage is duplicated or differs from pending observations")
        covered.update(keys)
        if item["payload"].get("bank_hash") != law.law_hash:
            raise TheoryError("Saved integration payload reference law differs from the loaded law")
        # Diagnostic scalars (including pending NaN and saturated log values)
        # already live in Parquet. Keep them out of the finite tensor payload,
        # and recover their original logical seed order for this integration.
        selected = matched.loc[matched.step_index.eq(item["step_index"])
                               & matched.seed.astype(str).isin(seeds)]
        indexed = selected.set_index(selected.seed.astype(str))
        if not indexed.index.is_unique or set(indexed.index) != set(seeds):
            raise TheoryError("Saved integration scalar rows do not uniquely match the payload seeds")
        selected = indexed.loc[seeds]
        base_metrics = {name: selected[name].tolist() for name in selected if name.startswith("direct_prop5_")}
        result = integrate_proposition5_payload(law.support, item["payload"], base_metrics, config=DEFAULT_INTEGRATION)
        for column, value in result.items():
            if isinstance(value, torch.Tensor) and value.ndim != 0 and value.shape != (len(seeds),):
                raise TheoryError(f"Integration result is not a per-seed scalar: {column}")
            if isinstance(value, list) and len(value) != len(seeds):
                raise TheoryError(f"Integration result seed count differs: {column}")
        if result.get("direct_prop5_status") == "failed_integration":
            failures.append({"step_index": item["step_index"], "seeds": item["seeds"],
                             "reason": result.get("direct_prop5_integration_error", "integration failed")})
        for name in ("initial", "matched_updates", "terminal"):
            frame = tables[name]
            mask = frame.step_index.eq(item["step_index"]) & frame.seed.astype(str).isin(map(str, item["seeds"]))
            selected = frame.loc[mask]
            if selected.empty:
                continue
            by_seed = {str(seed): i for i, seed in enumerate(item["seeds"])}
            offsets = [by_seed[str(seed)] for seed in selected.seed]
            for column, value in result.items():
                if not column.startswith("direct_prop5_"):
                    continue
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()
                    value = value.item() if value.ndim == 0 else value.tolist()
                values = [value[offset] for offset in offsets] if isinstance(value, list) else [value] * len(offsets)
                if column not in frame:
                    frame[column] = pd.Series([None] * len(frame), dtype=object)
                elif any(isinstance(value, (str, bool)) for value in values):
                    frame[column] = frame[column].astype(object)
                frame.loc[mask, column] = values
    if covered != expected:
        failures.append({"reason": "Missing integration payload coverage", "keys": sorted(expected - covered)})
    matched = tables["matched_updates"]
    if "direct_prop5_status" in matched:
        pending = matched.direct_prop5_status.eq("integration_required_not_computed")
        for _, row in matched.loc[pending].iterrows():
            failures.append({"step_index": int(row.step_index), "seeds": [int(row.seed)],
                             "reason": "Required original variation has no completed saved integration payload"})
    _validate_rows(tables, record, config)
    tables["audit"] = {"complete": not failures, "core_complete": True, "integration_complete": not failures,
                       "failed_updates": failures, "core_marker_sha256": file_sha256(core_root / "complete.json"),
                       "integration_hash": integration_hash, "payload_count": len(core_marker.get("payloads", []))}
    _save_record(root, paths, tables, record, integration_hash)
    if failures:
        raise TheoryError("Direct integration incomplete; scalar rows retained for integration-only retry")
    return root


def _read_law(path, digest, device, candidate_chunk_size, query_chunk_size):
    from .reference_law import ReferenceLaw
    from .support import FiniteSupport
    if path.is_symlink() or file_sha256(path) != digest:
        raise TheoryError("Shared direct reference law payload changed")
    payload = safe_torch_load(path)
    metadata = payload["metadata"]
    support = FiniteSupport(payload["atoms"].to(device), metadata["atom_ids"], metadata["aliases"],
                            weights=payload["weights"].to(device),
                            candidate_chunk=candidate_chunk_size, query_chunk=query_chunk_size)
    support.weights = payload["weights"].to(device).clone()
    support.log_weights = support.weights.log()
    return ReferenceLaw(support, metadata, payload["law_hash"])


def _run_analytical_worker(*, sources, records, bundle, analysis_hash, law_path,
                           law_digest, schedule_path, schedule_digest, config,
                           device, worker_count, candidate_chunk_size, query_chunk_size,
                           integration_directory=None, integration_hash=None):
    from .direct_measurements import measure_direct_record
    from .scheduler_adapter import SchedulerAdapter
    from utils.experiments.cache import validate_generation_record
    configure_worker_cpu_threads(worker_count)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    law = _read_law(Path(law_path), law_digest, device, candidate_chunk_size, query_chunk_size)
    if file_sha256(schedule_path) != schedule_digest:
        raise TheoryError("Shared saved scheduler payload changed")
    schedule = safe_torch_load(schedule_path)
    version = sources.runs["experiment"]["scientific_config"].get("package_versions", {}).get("diffusers")
    adapter = SchedulerAdapter(schedule, recorded_diffusers_version=version)
    integration_hash = integration_hash or canonical_hash(analytical_recipe())
    integration_directory = integration_directory or bundle / "integrations" / integration_hash
    receipts = []
    for record in records:
        root, paths = _paths(bundle, record)
        status = "failed"
        try:
            resumed = _valid_record(bundle, record, analysis_hash)
            if resumed:
                check = validate_generation_record(sources.experiment, record.original_index,
                                                   load_tensors=False, require_preview=False)
                if not check.valid:
                    raise TheoryError(f"Protected source changed: {check.errors}")
            else:
                payloads = []
                def capture(step, start, stop, payload, base_metrics):
                    name = f"payloads/step-{step:04d}-seeds-{start:06d}-{stop:06d}.pt"
                    atomic_torch_save(_cpu_payload({"schema_version": INTEGRATION_PAYLOAD_SCHEMA_VERSION,
                                                   "step_index": step, "seeds": record.metadata["seeds"][start:stop],
                                                   "payload": payload}), contained_path(root, name))
                    payloads.append(name)
                log = load_record(sources.experiment, record)
                scores = load_scores(sources, record)
                with torch.inference_mode():
                    tables = measure_direct_record(
                        log, record, law, config, adapter=adapter, terminal_sscd=scores,
                        integration=False, payload_callback=capture,
                    )
                del log
                _validate_rows(tables, record, config)
                tables["audit"]["complete"] = tables["audit"].get("core_complete", tables["audit"].get("complete", False))
                _save_record(root, paths, tables, record, analysis_hash, payloads=payloads)
                if not tables["audit"]["complete"]:
                    errors = tables["audit"].get("failed_updates", [])
                    reasons = Counter(item.get("error", item.get("reason", "Unknown core failure")) for item in errors)
                    details = "; ".join(f"{count} batches: {reason}" for reason, count in reasons.most_common(3))
                    raise TheoryError(f"Direct vector measurement incomplete: {details or 'see retained core audit'}")
            integral_resumed = _valid_record(integration_directory, record, integration_hash)
            if integral_resumed:
                integrated_root, _ = _paths(integration_directory, record)
                audit = read_object(integrated_root / "audit.json")
                integral_resumed = audit.get("core_marker_sha256") == file_sha256(root / "complete.json")
            if not integral_resumed:
                _integrate_record(bundle, integration_directory, record, law, config, integration_hash)
            status = "resumed" if resumed and integral_resumed else "reduced"
            receipts.append({"record_id": record.metadata["record_id"], "original_index": record.original_index,
                             "status": status, "device": str(device), "directory": str(root),
                             "core_resumed": resumed, "integration_resumed": integral_resumed})
        except Exception as error:
            failure = {"record_id": record.metadata["record_id"], "original_index": record.original_index,
                       "status": "failed", "device": str(device), "reason": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc(), "directory": str(root)}
            atomic_write_json(root / "failure.json", failure)
            receipts.append(failure)
        finally:
            report_worker_record(record.original_index, status, device)
    return receipts


def run_direct_analysis(project_root, *, config, device="auto", probe_batch_size=8,
                        candidate_chunk_size=256, query_chunk_size=16):
    """Resume validated task shards; missing integrals never invalidate learned probes."""
    from .direct_probes import run_direct_probes
    from .reference_law import build_reference_law
    from .reduce import _resolve_theory_devices, _worker_sources
    root = Path(project_root).absolute()
    sources = discover_sources(root, **config)
    records = sources.selected
    if not records:
        raise TheoryError("No selected complete experiment records for direct comparisons")
    schedule = load_schedule(sources)
    law = build_reference_law(sources, {**config, "candidate_chunk_size": candidate_chunk_size,
                                     "query_chunk_size": query_chunk_size})
    paper = PaperPaths.build(root, **config).output_directory
    backing = contained_path(paper.parent.parent, "theory_measurements")
    probes = run_direct_probes(root, records=records, support=law, config=config,
                              output_directory=backing / "direct_probes", device=device,
                              probe_batch_size=probe_batch_size, sources=sources)
    # The probe executor has joined and released all denoiser replicas here.
    identity = {"recipe": core_recipe(), "reference_law_hash": law.law_hash,
                "source_metadata_hash": sources.metadata_hash(),
                "record_hashes": [_record_key(r) for r in records],
                "guidance_scale": config["guidance_scale"], "target_error_tolerance": config.get("target_error_tolerance")}
    analysis_hash = canonical_hash(identity)
    bundle = contained_path(backing, "direct_analysis/" + analysis_hash)
    bundle.mkdir(parents=True, exist_ok=True)
    integration_identity = {"core_hash": analysis_hash, "recipe": analytical_recipe()}
    integration_hash = canonical_hash(integration_identity)
    integrated_bundle = contained_path(bundle, "integrations/" + integration_hash)
    law_path, schedule_path = bundle / "reference_law.pt", bundle / "schedule.pt"
    law_digest = atomic_torch_save({"atoms": law.support.atoms.cpu(), "weights": law.support.weights.cpu(),
                                   "metadata": law.metadata, "law_hash": law.law_hash}, law_path)
    schedule_digest = atomic_torch_save(schedule, schedule_path)
    devices = _resolve_theory_devices(device)
    count = worker_count_for_tasks(devices, len(records))
    shards = [round_robin_shard(records, worker_index=k, worker_count=count) for k in range(count)]
    context = get_context("spawn")
    receipts = []
    with RecordProgress(total=len(records), devices=count, context=context) as progress:
        # Spawn even a single analytical worker: parent owns only coordination,
        # and no model allocation from a prior stage can survive in a worker.
        with ProcessPoolExecutor(max_workers=count, mp_context=context,
                                 initializer=install_progress_queue, initargs=(progress.queue,)) as executor:
            futures = {executor.submit(
                _run_analytical_worker, sources=_worker_sources(sources, shard), records=shard,
                bundle=bundle, analysis_hash=analysis_hash, law_path=law_path, law_digest=law_digest,
                integration_directory=integrated_bundle, integration_hash=integration_hash,
                schedule_path=schedule_path, schedule_digest=schedule_digest, config=config,
                device=str(devices[k]), worker_count=count, candidate_chunk_size=candidate_chunk_size,
                query_chunk_size=query_chunk_size,
            ): (k, shard) for k, shard in enumerate(shards)}
            for future in as_completed(futures):
                k, shard = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = [{"record_id": r.metadata["record_id"], "original_index": r.original_index,
                               "status": "failed", "device": str(devices[k]), "reason": str(error)} for r in shard]
                receipts.extend(result)
                for receipt in result:
                    progress.report(receipt["original_index"], receipt["status"], receipt["device"])
    failed = [receipt for receipt in receipts if receipt["status"] == "failed"]
    manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION, "formula_version": DIRECT_FORMULA_VERSION,
        "analysis_hash": integration_hash, "core_hash": analysis_hash,
        "identity": integration_identity, "core_identity": identity, "complete": False,
        "reference_law": law.metadata, "workers": receipts,
    }
    atomic_write_json(integrated_bundle / "analysis_manifest.json", manifest)
    atomic_write_frame_parquet(pd.DataFrame(failed, columns=["record_id", "original_index", "status", "device", "reason"]), integrated_bundle / "failed.parquet")
    if failed:
        reasons = Counter(receipt["reason"] for receipt in failed)
        details = "; ".join(f"{count} records: {reason}" for reason, count in reasons.most_common(3))
        raise TheoryError(f"{len(failed)}/{len(records)} direct records incomplete. {details}. "
                          f"Valid independent shards retained. See {integrated_bundle / 'failed.parquet'} "
                          "and per-record audit.json; rerun to resume missing work")
    tables = dict(probes["tables"])
    files = dict(probes["files"])
    for name in TABLES:
        frames = [pd.read_parquet(_paths(integrated_bundle, record)[1][name]) for record in records]
        tables[name] = pd.concat(frames, ignore_index=True)
        path = integrated_bundle / (name + ".parquet")
        atomic_write_frame_parquet(tables[name], path)
        files[name] = {"path": str(path), "sha256": file_sha256(path), "rows": len(tables[name])}
    manifest["complete"] = True
    manifest["files"] = {name: files[name] for name in TABLES}
    atomic_write_json(integrated_bundle / "analysis_manifest.json", manifest)
    provenance = {"formula_version": DIRECT_FORMULA_VERSION, "analysis_hash": integration_hash,
                  "core_hash": analysis_hash,
                  "analysis_manifest": str(integrated_bundle / "analysis_manifest.json"),
                  "analysis_manifest_sha256": file_sha256(integrated_bundle / "analysis_manifest.json"),
                  "reference_law": law.metadata, "reference_law_hash": law.law_hash,
                  "probe_stage": probes["provenance"],
                  "record_audits": {"complete_records": len(records), "failed_records": 0,
                                    "directory": str(integrated_bundle / "records"),
                                    "reading_policy": "Per-record audit.json is analysis-only; plot summaries store aggregate statuses."},
                  "scheduler_recipe": identity["recipe"],
                  "manuscript_sha256": "fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f",
                  "matching_latex_source": "not available; verified statement labels and formulas from author brief",
                  "execution": {"devices": list(map(str, devices[:count])), "worker_count": count,
                                "probe_batch_size": probe_batch_size, "candidate_chunk_size": candidate_chunk_size,
                                "query_chunk_size": query_chunk_size},
                  "input_sources": ["forward_target", "gaussian_probe", "generated_state"]
                                    + (["forward_marginal"] if config["measure_unconditional_loss"] else [])}
    return {"tables": tables, "provenance": provenance, "files": files, "directory": integrated_bundle}
