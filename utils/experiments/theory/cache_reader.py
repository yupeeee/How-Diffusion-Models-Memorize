"""Validated, outcome-independent access to preserved generation/SSCD caches."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.common.io import canonical_hash, file_sha256, read_json, safe_torch_load
from utils.data.selection import (
    load_target_pair_selection,
    reference_completion_fingerprint,
    target_pair_selection_directory,
)
from utils.experiments.cache import (
    generation_paths,
    require_generation_run,
    list_completed_records,
    validate_generation_record,
)
from .contracts import TheoryError, numerical_config


@dataclass
class Sources:
    root: Path
    config: dict
    experiment: object
    reference: object
    runs: dict
    records: dict
    selection: object
    proximity: pd.DataFrame
    sscd_config: dict
    metadata_files: dict
    marker_inventory: dict

    @property
    def selected(self):
        return [
            record
            for record in self.records["experiment"]
            if record.original_index in self.selection.included_indices
        ]

    def metadata_hash(self):
        return canonical_hash(
            {"files": self.metadata_files, "inventory": self.marker_inventory}
        )


def _same(actual, expected, label):
    if actual != expected:
        raise TheoryError(
            f"{label} differs: observed {actual!r}, expected {expected!r}"
        )


def _science_contract(science):
    return {k: v for k, v in science.items() if k != "seeds"}


def discover_sources(project_root, **config) -> Sources:
    """Validate complete frozen metadata before any trajectory is opened."""
    root = Path(project_root).resolve()
    c = numerical_config(**config)
    args = {
        key: c[key]
        for key in (
            "model_name",
            "scheduler_name",
            "guidance_scale",
            "num_inference_steps",
            "num_seeds",
        )
    }
    count, steps = c["num_seeds"], c["num_inference_steps"]
    selection = load_target_pair_selection(root, **args)
    paths = {
        role: generation_paths(root, **args, seed_start=start)
        for role, start in (("experiment", 0), ("reference", count))
    }
    runs, records, metadata_files, inventories, sscd_configs = {}, {}, {}, {}, {}

    def remember(path):
        if not path.is_file() or path.is_symlink():
            raise TheoryError(f"Missing/unsafe source artifact: {path}")
        metadata_files[path.relative_to(root).as_posix()] = file_sha256(path)

    for role, path in paths.items():
        runs[role] = require_generation_run(path)
        science = runs[role]["scientific_config"]
        for key, value in {
            "model_cli_name": c["model_name"],
            "guidance_scale": c["guidance_scale"],
            "num_inference_steps": steps,
            "num_seeds": count,
            "stored_prediction_type": "epsilon",
            "trajectory_order": "noise_to_image",
        }.items():
            _same(science.get(key), value, f"{role} generation {key}")
        _same(
            science.get("scheduler", {}).get("name"),
            c["scheduler_name"],
            f"{role} scheduler",
        )
        seeds = (
            list(range(0, count))
            if role == "experiment"
            else list(range(count, 2 * count))
        )
        _same(science.get("seeds"), seeds, f"{role} seed block")
        records[role] = list_completed_records(path)
        if not records[role]:
            raise TheoryError(
                f"No complete {role} generation records: {path.run_directory}"
            )
        remember(path.run_config)
        sscd_path = path.run_directory / "sscd_config.json"
        sscd = read_json(sscd_path)
        # Reuse the protected path-normalizing SSCD scientific hash contract.
        from utils.experiments.sscd import sscd_configuration_hash

        _same(
            sscd.get("configuration_hash"),
            sscd_configuration_hash(sscd),
            f"{role} SSCD configuration hash",
        )
        _same(
            sscd.get("generation_scientific_config_hash"),
            runs[role]["scientific_config_hash"],
            f"{role} SSCD generation hash",
        )
        _same(sscd.get("seeds"), seeds, f"{role} SSCD seeds")
        sscd_configs[role] = sscd
        remember(sscd_path)
        for subdir in ("record", "sscd_record"):
            directory = path.run_directory / subdir
            inventories[directory.relative_to(root).as_posix()] = sorted(
                p.name for p in directory.glob("*.json")
            )
        for record in records[role]:
            validation = validate_generation_record(
                path,
                record.original_index,
                expected_scientific_hash=runs[role]["scientific_config_hash"],
                load_tensors=False,
                require_preview=False,
                verify_file_hashes=False,
            )
            if not validation.valid:
                raise TheoryError(
                    f"{role}/{record.original_index}: {'; '.join(validation.errors)}"
                )
            _same(
                record.metadata.get("seeds"),
                seeds,
                f"{role}/{record.original_index} seeds",
            )
            for key in (
                "model_id",
                "model_revision",
                "vae_id",
                "vae_revision",
                "stored_prediction_type",
                "native_prediction_type",
                "target_preprocessing",
                "latent_shape",
            ):
                _same(
                    record.metadata.get(key),
                    science.get(key),
                    f"{role}/{record.original_index} {key}",
                )
            for key in (
                "original_index",
                "record_id",
                "prompt_raw",
                "target_image_sha256",
            ):
                if not isinstance(record.metadata.get(key), str):
                    raise TheoryError(
                        f"{role}/{record.original_index} {key} must remain a string"
                    )
            remember(record.marker_path)
            score_marker = (
                path.run_directory / "sscd_record" / f"{record.original_index}.json"
            )
            # A nonselected support atom does not require its own experiment SSCD.
            if score_marker.exists():
                remember(score_marker)
        for record in records[role]:
            target = path.target_latent_path(record.original_index)
            if (
                file_sha256(target)
                != record.metadata["tensor_file_sha256"]["target_latent"]
            ):
                raise TheoryError(f"Target-latent identity differs: {target}")
    _same(
        _science_contract(runs["experiment"]["scientific_config"]),
        _science_contract(runs["reference"]["scientific_config"]),
        "reference/experiment latent and sampler contracts",
    )
    _same(
        selection.configuration["reference_generation_hash"],
        runs["reference"]["scientific_config_hash"],
        "frozen reference generation hash",
    )
    _same(
        selection.configuration["reference_sscd_hash"],
        sscd_configs["reference"]["configuration_hash"],
        "frozen reference SSCD hash",
    )
    _same(
        selection.configuration["reference_completion_fingerprint"],
        reference_completion_fingerprint(paths["reference"].run_directory),
        "frozen reference completion evidence",
    )
    selection_dir = target_pair_selection_directory(root, **args)
    for name in ("selection.csv", "config.json", "summary.json"):
        remember(selection_dir / name)
    selected = selection.included_indices
    available = {record.original_index for record in records["experiment"]}
    if selected - available:
        raise TheoryError(
            f"Selected experiment records missing: {sorted(selected - available)}"
        )
    if not selected:
        raise TheoryError(
            "Frozen selection contains no eligible prompts; no analysis was published"
        )
    prompt_rows = selection.prompt_frame.set_index("original_index")
    reference_by_id = {r.original_index: r for r in records["reference"]}
    for record in records["experiment"]:
        if record.original_index not in selected:
            continue
        ref = reference_by_id.get(record.original_index)
        if ref is None:
            raise TheoryError(
                f"Selected reference record missing: {record.original_index}"
            )
        for key in ("record_id", "prompt_raw", "target_image_sha256"):
            _same(
                record.metadata.get(key),
                ref.metadata.get(key),
                f"paired reference/experiment {record.original_index} {key}",
            )
        if (
            record.metadata["tensor_file_sha256"]["target_latent"]
            != ref.metadata["tensor_file_sha256"]["target_latent"]
        ):
            left = safe_torch_load(
                paths["experiment"].target_latent_path(record.original_index)
            )
            right = safe_torch_load(
                paths["reference"].target_latent_path(record.original_index)
            )
            if (
                not isinstance(left, torch.Tensor)
                or not isinstance(right, torch.Tensor)
                or not torch.equal(left, right)
            ):
                raise TheoryError(
                    f"Paired reference/experiment target-latent identity differs: {record.original_index}"
                )
        for key, column in (
            ("record_id", "record_id"),
            ("prompt_raw", "prompt"),
            ("target_image_sha256", "target_image_sha256"),
        ):
            _same(
                record.metadata.get(key),
                prompt_rows.loc[record.original_index, column],
                f"selection identity {record.original_index} {key}",
            )
    proximity_dir = (
        root
        / "outputs"
        / paths["experiment"].run_directory.parent.name
        / "proximity"
        / paths["experiment"].run_directory.name
    )
    proximity_config = read_json(proximity_dir / "run_config.json")
    for key, value in {
        "selection_hash": selection.sha256,
        "generation_scientific_config_hash": runs["experiment"][
            "scientific_config_hash"
        ],
        "sscd_configuration_hash": sscd_configs["experiment"]["configuration_hash"],
        "seeds": list(range(count)),
        "distance": "euclidean_l2",
    }.items():
        _same(proximity_config.get(key), value, f"protected proximity {key}")
    proximity = pd.read_csv(
        proximity_dir / "proximity.csv",
        dtype={"original_index": str, "record_id": str, "prompt": str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    if proximity.duplicated(["original_index", "seed"]).any():
        raise TheoryError("Duplicate protected proximity observation identity")
    for record in records["experiment"]:
        if record.original_index not in selected:
            continue
        frame = proximity[proximity.original_index.eq(record.original_index)]
        _same(
            sorted(frame.seed.tolist()),
            list(range(count)),
            f"proximity seed coverage {record.original_index}",
        )
        for key, value in {
            "record_id": record.metadata["record_id"],
            "prompt": record.metadata["prompt_raw"],
            "target_image_sha256": record.metadata["target_image_sha256"],
            "observation_status": "complete",
            "include_prompt": True,
        }.items():
            if not frame[key].eq(value).all():
                raise TheoryError(
                    f"Protected proximity {key} mismatch: {record.original_index}"
                )
        if not np.isfinite(frame[["l2_norm", "sscd"]].to_numpy(dtype=float)).all():
            raise TheoryError(
                f"Missing/nonfinite proximity scores: {record.original_index}"
            )
    remember(proximity_dir / "run_config.json")
    remember(proximity_dir / "proximity.csv")
    return Sources(
        root,
        c,
        paths["experiment"],
        paths["reference"],
        runs,
        records,
        selection,
        proximity,
        sscd_configs,
        metadata_files,
        inventories,
    )


def load_schedule(sources):
    schedules = []
    for role in ("experiment", "reference"):
        paths = getattr(sources, role)
        payload = safe_torch_load(paths.schedule)
        if not isinstance(payload, dict):
            raise TheoryError(f"Invalid schedule: {paths.schedule}")
        science = sources.runs[role]["scientific_config"]
        for key, expected in (
            ("scheduler_name", science["scheduler"]["name"]),
            ("scheduler_class", science["scheduler"]["class"]),
            ("trajectory_order", "noise_to_image"),
        ):
            _same(payload.get(key), expected, f"{role} schedule {key}")
        s0 = payload.get("init_noise_sigma")
        if (
            not isinstance(s0, (int, float))
            or isinstance(s0, bool)
            or not math.isfinite(s0)
            or s0 <= 0
        ):
            raise TheoryError(
                f"{role} schedule init_noise_sigma must be finite and positive"
            )
        timesteps = payload.get("timesteps")
        if (
            not isinstance(timesteps, torch.Tensor)
            or timesteps.dtype != torch.int64
            or timesteps.shape != (sources.config["num_inference_steps"],)
        ):
            raise TheoryError(
                f"{role} saved schedule T/integer timestep contract differs"
            )
        for key in ("stored_prediction_type", "native_prediction_type"):
            _same(payload.get(key), science.get(key), f"{role} schedule {key}")
        _same(
            payload.get("scheduler_config"),
            science["scheduler"]["config"],
            f"{role} saved scheduler configuration",
        )
        schedules.append(payload)
    # Hash canonical tensor values, never infer schedule coefficients from filenames.
    if canonical_hash(schedules[0]) != canonical_hash(schedules[1]):
        raise TheoryError("Reference/experiment saved schedules differ")
    return schedules[0]


def load_record(paths, record, *, initial_only=False, verify_hashes=True):
    """One payload load per record. Initial-only readers memory map CPU tensors.

    Hash validation is independent of mmap. Mapping lets reference-center checks
    touch only the initial states/predictions, without allocating full rollouts.
    """
    result = validate_generation_record(
        paths,
        record.original_index,
        expected_scientific_hash=record.metadata["scientific_config_hash"],
        load_tensors=False,
        require_preview=False,
        verify_file_hashes=verify_hashes,
    )
    if not result.valid:
        raise TheoryError(
            f"Invalid generation record {record.original_index}: {'; '.join(result.errors)}"
        )

    def load(path):
        if initial_only:
            return torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        return safe_torch_load(path)

    z = load(paths.latent_path(record.original_index))
    predictions = load(paths.noise_prediction_path(record.original_index))
    target = safe_torch_load(paths.target_latent_path(record.original_index))
    if not isinstance(predictions, tuple) or len(predictions) != 2:
        raise TheoryError(
            f"Expected canonical (epsilon_u, epsilon_c): {record.original_index}"
        )
    u, c = predictions
    tensors = {
        "latent": z,
        "unconditional_noise_predictions": u,
        "conditional_noise_predictions": c,
        "target_latent": target,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or list(
            tensor.shape
        ) != record.metadata["tensor_shapes"].get(name):
            raise TheoryError(f"{record.original_index} {name} shape mismatch")
        if str(tensor.dtype).removeprefix("torch.") != record.metadata[
            "tensor_dtypes"
        ].get(name):
            raise TheoryError(f"{record.original_index} {name} dtype mismatch")
    n, t = len(record.metadata["seeds"]), record.metadata["num_inference_steps"]
    if (
        z.ndim != 5
        or z.shape[:2] != (n, t + 1)
        or u.shape != c.shape
        or u.shape != (n, t, *z.shape[2:])
        or target.shape != z.shape[2:]
    ):
        raise TheoryError(
            f"T+1 state / T branch alignment differs: {record.original_index}"
        )
    if initial_only:
        z, u, c = z[:, 0].clone(), u[:, 0].clone(), c[:, 0].clone()
    if not all(bool(torch.isfinite(v).all()) for v in (z, u, c, target)):
        raise TheoryError(f"Nonfinite source tensor: {record.original_index}")
    return z, u, c, target


def load_scores(sources, record):
    from utils.experiments.sscd import SSCDPaths, _read_cached_scores

    paths = SSCDPaths(sources.experiment.run_directory)
    scores = _read_cached_scores(
        paths,
        record,
        sources.sscd_config["experiment"]["configuration_hash"],
        range(sources.config["num_seeds"]),
    )
    if scores is None:
        raise TheoryError(f"Missing same-seed terminal SSCD: {record.original_index}")
    metadata = read_json(paths.marker_path(record.original_index))
    for key in ("prompt_raw", "target_image_sha256"):
        _same(
            metadata.get(key),
            record.metadata.get(key),
            f"SSCD {record.original_index} {key}",
        )
    frame = sources.proximity[
        sources.proximity.original_index.eq(record.original_index)
    ].set_index("seed")
    for offset, seed in enumerate(record.metadata["seeds"]):
        if abs(float(scores[offset]) - float(frame.loc[seed, "sscd"])) > 2e-7:
            raise TheoryError(
                f"Same-seed SSCD/proximity mismatch: {record.original_index}, seed {seed}"
            )
    return scores
