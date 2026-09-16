"""Separately resumable supplements for the fixed seven-statement evidence suite.

The existing learned probes, vector cores, and original variation integrals are
left under their existing scientific identities. Only missing supplements read
preserved vectors. A single parent progress bar coordinates all new devices.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from importlib.metadata import version as package_version
from pathlib import Path
import math
import traceback

import pandas as pd
import torch

from utils.common.io import atomic_write_frame_parquet, atomic_write_json, canonical_hash, file_sha256, read_json, safe_torch_load
from utils.models.devices import configure_worker_cpu_threads, round_robin_shard, worker_count_for_tasks
from .contracts import EVIDENCE_FORMULA_VERSION, TheoryError
from .paper_contracts import contained_path, publication_lock, reject_symlinks
from .progress import RecordProgress, install_progress_queue, report_worker_record

SUPPLEMENT_SCHEMA_VERSION = 1
KEYS = ["run_id", "original_index", "record_id", "target_id", "seed", "step_index"]


def _sources(*names):
    base = Path(__file__).parent
    return {name: file_sha256(base / name) for name in names}


def _verify_sources(expected):
    if _sources(*expected) != expected:
        raise TheoryError("Evidence source code changed during analysis; retry with the current recipe")


def _complete(root, identity):
    root = reject_symlinks(root)
    marker = root / "complete.json"
    if not marker.is_file() or marker.is_symlink():
        return False
    try:
        value = read_json(marker)
        return (value.get("schema_version") == SUPPLEMENT_SCHEMA_VERSION and value.get("complete") is True
                and value.get("identity") == identity and bool(value.get("files"))
                and all(contained_path(root, name).is_file()
                        and file_sha256(contained_path(root, name)) == item["sha256"]
                        for name, item in value["files"].items()))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError):
        return False


def _save(root, identity, frames, audit=None):
    root = reject_symlinks(root)
    files = {}
    for name, frame in frames.items():
        path = contained_path(root, name + ".parquet")
        atomic_write_frame_parquet(frame, path)
        files[path.name] = {"sha256": file_sha256(path), "rows": len(frame)}
    atomic_write_json(root / "complete.json", {
        "schema_version": SUPPLEMENT_SCHEMA_VERSION, "identity": identity,
        "complete": True, "files": files, "audit": audit or {},
    })


def _join(base, extra):
    """An additive exact-key join, with no outcome-dependent row removal."""
    if extra.empty and base.empty:
        return base.copy()
    if not set(KEYS) <= set(extra) or extra.duplicated(KEYS).any() or base.duplicated(KEYS).any():
        raise TheoryError("Evidence supplement has missing or duplicate logical identities")
    for key in KEYS:
        if key != "step_index":
            base = base.assign(**{key: base[key].astype(str)})
            extra = extra.assign(**{key: extra[key].astype(str)})
    old = set(base.columns) - set(KEYS)
    additions = [name for name in extra if name not in KEYS]
    if old.intersection(additions):
        raise TheoryError("Evidence supplement attempted to overwrite an existing measurement")
    joined = base.merge(extra, how="outer", on=KEYS, validate="one_to_one", indicator=True, sort=False)
    if not joined._merge.eq("both").all() or len(joined) != len(base):
        raise TheoryError("Evidence supplement must preserve every retained sample identity")
    return joined.drop(columns="_merge")


def _gaussian_control(log, record, science, schedule, initial):
    """Use initial vectors only when they exactly match the Gaussian probe bank."""
    from utils.models.sampling import make_initial_noise
    states, epsilon_u, _, target = log
    bank = make_initial_noise(science["seeds"], science["latent_shape"])
    expected = bank.to(dtype=states.dtype) * float(schedule["init_noise_sigma"])
    matched = float(schedule["init_noise_sigma"]) == 1.0 and torch.equal(states[:, 0], expected)
    result = initial[KEYS].copy()
    result["unconditional_target_error_rmse"] = math.nan
    result["gaussian_control_status"] = "unavailable_saved_initialization_not_same_gaussian_input"
    if matched:
        cumulative = float(schedule["alphas_cumprod_t"][0])
        mu = (states[:, 0].double() - math.sqrt(1.0 - cumulative) * epsilon_u[:, 0].double()) / math.sqrt(cumulative)
        values = (mu - target.double()).flatten(1).norm(dim=1) / math.sqrt(target.numel())
        by_seed = dict(zip(map(str, record.metadata["seeds"]), values.tolist(), strict=True))
        result["unconditional_target_error_rmse"] = [by_seed[str(seed)] for seed in result.seed]
        result["gaussian_control_status"] = "same_saved_gaussian_input_canonical_epsilon"
    return result


def _worker(*, sources, records, tasks, law_path, law_digest, schedule_path, schedule_digest,
            direct_directory, config, run_scope, terminal_update_count,
            device, worker_count, candidate_chunk_size, query_chunk_size):
    from .cache_reader import load_record
    from .direct_reduce import _paths, _read_law
    from .evidence_measurements import measure_evidence_record
    from .evidence_reference import gaussian_bank, reference_grid_rows
    from .scheduler_adapter import SchedulerAdapter

    expected_sources = {}
    for task in tasks:
        recipe = task["identity"].get("reference", task["identity"].get("recipe"))
        expected_sources.update(recipe["source_code"])
    _verify_sources(expected_sources)
    configure_worker_cpu_threads(worker_count)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    law = _read_law(Path(law_path), law_digest, device, candidate_chunk_size, query_chunk_size)
    if file_sha256(schedule_path) != schedule_digest:
        raise TheoryError("Saved schedule changed before evidence supplements")
    schedule = safe_torch_load(schedule_path)
    science = sources.runs["experiment"]["scientific_config"]
    version = science.get("package_versions", {}).get("diffusers")
    adapter = SchedulerAdapter(schedule, recorded_diffusers_version=version)
    bank = gaussian_bank(science, schedule)
    records = {str(record.original_index): record for record in records}
    receipts = []
    try:
        for task in tasks:
            status = "failed"
            try:
                path = Path(task["path"])
                if _complete(path, task["identity"]):
                    status = "resumed"
                elif task["kind"] == "reference":
                    frame = reference_grid_rows(law, bank, seeds=science["seeds"], run_id=run_scope,
                                                snr=task["snr"], grid_index=task["grid_index"],
                                                initial_snr=task["initial_snr"],
                                                reference_scale_rmse=task["identity"]["reference"]["reference_scale_rmse"])
                    _verify_sources(expected_sources)
                    _save(path, task["identity"], {"reference_analytical": frame})
                    status = "reduced"
                else:
                    record = records[task["record_index"]]
                    core_root, core_paths = _paths(direct_directory, record)
                    if file_sha256(core_root / "complete.json") != task["identity"]["core_marker_sha256"]:
                        raise TheoryError("Saved direct scalar source changed before supplement")
                    tables = {name: pd.read_parquet(core_paths[name]) for name in ("initial", "trajectory", "terminal")}
                    log = load_record(sources.experiment, record)
                    with torch.inference_mode():
                        extra = measure_evidence_record(log, record, law, config, adapter,
                                                        terminal_update_count, run_scope, tables)
                        control = _gaussian_control(log, record, science, schedule, tables["initial"])
                    del log
                    frames = {name: extra[name] for name in ("initial", "trajectory", "terminal")}
                    frames["gaussian_control"] = control
                    if extra["audit"].get("complete") is not True:
                        raise TheoryError(f"Evidence supplement incomplete: {extra['audit']}")
                    for name in ("initial", "trajectory", "terminal"):
                        _join(tables[name], frames[name])
                    _verify_sources(expected_sources)
                    _save(path, task["identity"], frames, extra["audit"])
                    status = "reduced"
                receipts.append({"task_hash": task["task_hash"], "status": status, "device": str(device)})
            except Exception as error:
                receipt = {"task_hash": task["task_hash"], "status": "failed", "device": str(device),
                           "reason": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
                atomic_write_json(Path(task["path"]) / "failure.json", receipt)
                receipts.append(receipt)
            finally:
                report_worker_record(task["task_hash"], status, device)
    finally:
        del law, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return receipts


def run_evidence_analysis(project_root, *, config, device="auto", probe_batch_size=8,
                          candidate_chunk_size=256, query_chunk_size=16):
    from .cache_reader import discover_sources
    from .direct_reduce import run_direct_analysis, _paths, _read_law
    from .evidence_reference import REFERENCE_SWEEP_VERSION, reference_grid, reference_atoms
    from .reduce import _resolve_theory_devices, _worker_sources

    direct = run_direct_analysis(project_root, config=config, device=device, probe_batch_size=probe_batch_size,
                                 candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size)
    sources = discover_sources(project_root, **config)
    records = sources.selected
    direct_directory = Path(direct["directory"])
    core_directory = direct_directory.parent.parent
    backing = core_directory.parent.parent
    law_path, schedule_path = core_directory / "reference_law.pt", core_directory / "schedule.pt"
    law_digest, schedule_digest = file_sha256(law_path), file_sha256(schedule_path)
    schedule = safe_torch_load(schedule_path)
    science = sources.runs["experiment"]["scientific_config"]
    cumulative = float(schedule["alphas_cumprod_t"][0])
    initial_snr = cumulative / (1.0 - cumulative)
    grid = reference_grid(initial_snr, config["reference_snr_decades"])
    run_scope = sources.runs["experiment"]["scientific_config_hash"]
    terminal_update_count = sum(len(record.metadata["seeds"]) for record in records)
    # One CPU float64 scale prevents device-dependent last bits from changing
    # a shared scalar reference line across independently resumed grid tasks.
    law = _read_law(law_path, law_digest, "cpu", candidate_chunk_size, query_chunk_size)
    atom_table = reference_atoms(law)
    del law
    reference_identity = {
        "formula_version": REFERENCE_SWEEP_VERSION, "reference_law_hash": direct["provenance"]["reference_law_hash"],
        "source_code": _sources("evidence_reference.py", "reference_law.py", "support.py", "../../models/sampling.py"),
        "gaussian_seeds": science["seeds"], "latent_shape": science["latent_shape"],
        "gaussian_storage_dtype": science["scientific_tensor_storage"]["dtype"],
        "init_noise_sigma": float(schedule["init_noise_sigma"]), "run_id": run_scope,
        "initial_snr": initial_snr, "grid": grid.tolist(),
        "reference_scale_rmse": float(atom_table.reference_scale_rmse.iloc[0]),
    }
    reference_root = contained_path(backing, "reference_analytical/" + canonical_hash(reference_identity))
    supplement_recipe = {
        "formula_version": EVIDENCE_FORMULA_VERSION,
        "source_code": _sources("evidence_reduce.py", "evidence_measurements.py"),
        "reference_law_hash": direct["provenance"]["reference_law_hash"], "schedule_sha256": schedule_digest,
        "terminal_noise_run_alpha": config["terminal_noise_run_alpha"],
        "quantile_implementation": {"library": "scipy.stats.chi2.isf", "scipy_version": package_version("scipy")},
        "terminal_update_count": terminal_update_count, "terminal_run_scope": run_scope,
        "guidance_scale": config["guidance_scale"],
    }
    supplement_root = contained_path(backing, "evidence_supplements/" + canonical_hash(supplement_recipe))
    tasks = []
    for index, snr in enumerate(grid):
        identity = {"reference": reference_identity, "grid_index": index, "snr": float(snr)}
        tasks.append({"kind": "reference", "identity": identity, "task_hash": canonical_hash(identity),
                      "path": str(reference_root / "tasks" / f"{index:03d}"), "grid_index": index,
                      "snr": float(snr), "initial_snr": initial_snr})
    for record in records:
        source_root, _ = _paths(direct_directory, record)
        identity = {"recipe": supplement_recipe, "record": record.metadata,
                    "core_marker_sha256": file_sha256(source_root / "complete.json")}
        task_hash = canonical_hash(identity)
        tasks.append({"kind": "record", "identity": identity, "task_hash": task_hash,
                      "path": str(supplement_root / "records" / task_hash), "record_index": str(record.original_index)})
    with publication_lock(supplement_root):
        pending = [task for task in tasks if not _complete(task["path"], task["identity"])]
        devices = tuple(_resolve_theory_devices(device)) if pending else ()
        count = worker_count_for_tasks(devices, len(pending)) if pending else 0
        context = get_context("spawn")
        receipts = []
        with RecordProgress(total=len(tasks), devices=count, context=context) as progress:
            pending_ids = {task["task_hash"] for task in pending}
            for task in tasks:
                if task["task_hash"] not in pending_ids:
                    progress.report(task["task_hash"], "resumed", "saved")
            if count:
                with ProcessPoolExecutor(max_workers=count, mp_context=context,
                                         initializer=install_progress_queue, initargs=(progress.queue,)) as executor:
                    futures = {}
                    for worker_index in range(count):
                        shard = round_robin_shard(pending, worker_index=worker_index, worker_count=count)
                        indices = {task.get("record_index") for task in shard}
                        chosen = [record for record in records if str(record.original_index) in indices]
                        future = executor.submit(_worker, sources=_worker_sources(sources, chosen), records=chosen,
                            tasks=shard, law_path=law_path, law_digest=law_digest, schedule_path=schedule_path,
                            schedule_digest=schedule_digest, direct_directory=direct_directory, config=config,
                            run_scope=run_scope, terminal_update_count=terminal_update_count,
                            device=str(devices[worker_index]), worker_count=count,
                            candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size)
                        futures[future] = (worker_index, shard)
                    for future in as_completed(futures):
                        worker_index, shard = futures[future]
                        try:
                            result = future.result()
                        except Exception as error:
                            result = [{"task_hash": task["task_hash"], "status": "failed",
                                       "device": str(devices[worker_index]), "reason": str(error)} for task in shard
                                      if not _complete(task["path"], task["identity"])]
                        receipts.extend(result)
                        for item in result:
                            progress.report(item["task_hash"], item["status"], item["device"])
        incomplete = [task for task in tasks if not _complete(task["path"], task["identity"])]
        if incomplete:
            failed = [item for item in receipts if item["status"] == "failed"]
            atomic_write_json(supplement_root / "failed.json", {"incomplete": len(incomplete), "failures": failed})
            reasons = Counter(item["reason"] for item in failed)
            raise TheoryError(f"{len(incomplete)} evidence tasks incomplete: {dict(reasons)}; "
                              f"completed probes/integrals retained; see {supplement_root / 'failed.json'}")
        _verify_sources(reference_identity["source_code"])
        _verify_sources(supplement_recipe["source_code"])
        tables = dict(direct["tables"])
        supplemental_tables = {}
        for name in ("initial", "trajectory", "terminal"):
            extra = pd.concat([pd.read_parquet(Path(task["path"]) / (name + ".parquet"))
                               for task in tasks if task["kind"] == "record"], ignore_index=True)
            tables[name] = _join(tables[name], extra)
            supplemental_tables[name + "_evidence"] = extra
        control = pd.concat([pd.read_parquet(Path(task["path"]) / "gaussian_control.parquet")
                             for task in tasks if task["kind"] == "record"], ignore_index=True)
        gaussian = tables["gaussian_conditional"].copy()
        for key in KEYS:
            if key != "step_index":
                gaussian[key], control[key] = gaussian[key].astype(str), control[key].astype(str)
        gaussian = gaussian.merge(control, how="left", on=KEYS, validate="one_to_one", sort=False)
        exact = gaussian.terminal_sscd_matched_initialization.fillna(False).astype(bool)
        gaussian.loc[~exact, "unconditional_target_error_rmse"] = math.nan
        gaussian.loc[~exact, "gaussian_control_status"] = "unavailable_not_same_saved_gaussian_input"
        tables["gaussian_conditional"] = gaussian
        supplemental_tables["gaussian_control_evidence"] = control
        tables["reference_analytical"] = pd.concat([
            pd.read_parquet(Path(task["path"]) / "reference_analytical.parquet")
            for task in tasks if task["kind"] == "reference"], ignore_index=True)
        tables["reference_atoms"] = atom_table
        identity = {"supplement_recipe": supplement_recipe, "reference_recipe": reference_identity,
                    "direct_analysis_hash": direct["provenance"]["analysis_hash"],
                    "task_hashes": [task["task_hash"] for task in tasks]}
        digest = canonical_hash(identity)
        collection = contained_path(supplement_root, "collections/" + digest)
        files = dict(direct["files"])
        # Keep original scalar-history paths authoritative. Persist only additive
        # columns; the exact join recipe makes the enriched tables reproducible
        # without copying the complete trajectory or Gaussian histories.
        supplemental_tables.update({name: tables[name] for name in ("reference_analytical", "reference_atoms")})
        for name, frame in supplemental_tables.items():
            path = collection / (name + ".parquet")
            atomic_write_frame_parquet(frame, path)
            files[name] = {"path": str(path), "sha256": file_sha256(path), "rows": len(frame)}
        logical_joins = {name: {"base": name, "supplement": name + "_evidence", "keys": KEYS,
                               "how": "exact_one_to_one_additive", "preserves_every_base_row": True}
                         for name in ("initial", "trajectory", "terminal")}
        logical_joins["gaussian_conditional"] = {
            "base": "gaussian_conditional", "supplement": "gaussian_control_evidence", "keys": KEYS,
            "how": "left_one_to_one", "preserves_every_base_row": True,
            "eligibility": "Only step 0 controls whose original terminal_sscd_matched_initialization is true; otherwise control is unavailable",
        }
        provenance = {**direct["provenance"], "analysis_hash": digest,
                      "direct_analysis_hash": direct["provenance"]["analysis_hash"],
                      "evidence_recipe": identity, "reference_grid": reference_identity,
                      "terminal_noise_run_scope": {"alpha": config["terminal_noise_run_alpha"],
                          "m": terminal_update_count, "run_id": run_scope,
                          "scope": "all_retained_final_updates_in_one_scientific_run; no_trajectory_independence_required"},
                      "plan_selection_scope": "fixed_after_review_of_SDv1_DDIM; no_held_out_confirmation_claim"}
        atomic_write_json(collection / "manifest.json", {"schema_version": SUPPLEMENT_SCHEMA_VERSION,
                          "complete": True, "identity": identity, "files": files, "logical_joins": logical_joins, "provenance": provenance})
        return {"tables": tables, "files": files, "logical_joins": logical_joins, "provenance": provenance, "directory": collection}
