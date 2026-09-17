"""One explicit finite reference law for direct statements and learned probes.

The runtime loaders below only consume declared local sources. They neither
identify a training distribution from a checkpoint nor search for extra data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import torch

from utils.common.io import canonical_hash, file_sha256, read_json, safe_torch_load
from .support import FiniteSupport, _tensor_hash

REFERENCE_LAW_VERSION = "direct-reference-law-2"
_SINGLE_TARGET = "assumed_single_target_conditional_idealization_not_inferred_from_SSCD"


def _row_norm(value):
    scale = value.abs().amax(dim=1, keepdim=True)
    normalized = value / torch.where(scale > 0, scale, torch.ones_like(scale))
    return normalized.square().sum(dim=1).sqrt() * scale[:, 0]


def _safe_path(path):
    path = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Reference-law source must not traverse a symlink: {path}")
    return path


@dataclass
class ReferenceLaw:
    """CPU-pickleable wrapper around the existing finite Gaussian posterior."""
    support: FiniteSupport
    metadata: dict
    law_hash: str
    estimated_mean_vector: torch.Tensor | None = None

    def __post_init__(self):
        receipt = self.metadata.get("theory_mean", {})
        if receipt.get("source") in {"initial_unconditional_reference_monte_carlo",
                                    "minimum_snr_unconditional_reference_monte_carlo"}:
            if self.estimated_mean_vector is None:
                raise ValueError("Estimated reference mean receipt lacks its saved vector")
            value = self.estimated_mean_vector.to(device=self.support.flat.device, dtype=torch.float64)
            if tuple(value.shape) != self.support.latent_shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Estimated reference mean shape or values differ from the reference space")
            if _tensor_hash(value) != receipt.get("vector_sha256"):
                raise ValueError("Estimated reference mean vector differs from its receipt")
            self.estimated_mean_vector = value
        elif self.estimated_mean_vector is not None:
            raise ValueError("An estimated mean vector requires an explicit reference-estimation receipt")

    @property
    def theory_mean_vector(self):
        """Selected comparison/injection centre; the atom-law mean stays exact."""
        return self.mean_vector if self.estimated_mean_vector is None else self.estimated_mean_vector

    @property
    def theory_mean_metadata(self):
        return self.metadata.get("theory_mean", {
            "source": "declared_finite_bank_mean", "sample_count": None,
            "vector_sha256": self.metadata.get("mean_sha256"),
            "definition": "Exact declared weighted atom mean",
        })

    @property
    def theory_mean_offset_l2(self):
        return (self.mean_vector - self.theory_mean_vector).norm()

    def to_payload(self):
        payload = {"atoms": self.support.atoms.cpu(), "weights": self.support.weights.cpu(),
                   "metadata": self.metadata, "law_hash": self.law_hash}
        if self.estimated_mean_vector is not None:
            payload["estimated_mean_vector"] = self.estimated_mean_vector.cpu()
        return payload

    @property
    def mean_vector(self):
        return (self.support.weights @ self.support.flat).reshape(self.support.latent_shape)

    @property
    def max_atom_norm(self):
        return float(_row_norm(self.support.flat).max())

    def posterior_mean(self, z, alpha, sigma):
        return self.support.posterior_mean(z, alpha, sigma)

    def posterior(self, z, target_id, alpha, sigma, unconditional=None):
        return self.support.evaluate(
            z, target_id, alpha, sigma, unconditional, include_mean=True
        )

    def target_radius(self, target_id):
        index = self.support._target(target_id)
        return float(_row_norm(self.support.target_geometry(index)[0]).max())

    def radius_for(self, target):
        value = torch.as_tensor(target, dtype=torch.float64, device=self.support.flat.device)
        if tuple(value.shape) != self.support.latent_shape or not bool(torch.isfinite(value).all()):
            raise ValueError("Reference-law target shape/values differ")
        return float(_row_norm(self.support.flat - value.reshape(1, -1)).max())

    def target_id_for(self, target, preferred=None, *, required=True):
        value = torch.as_tensor(target, dtype=torch.float64, device=self.support.flat.device)
        if tuple(value.shape) != self.support.latent_shape or not bool(torch.isfinite(value).all()):
            raise ValueError("Reference-law target latent shape/values differ")
        if preferred is not None and str(preferred) in self.support.aliases:
            index = self.support.aliases[str(preferred)]
            if not torch.equal(value, self.support.atoms[index]):
                raise ValueError("Declared target alias points to a different exact atom")
            return str(preferred)
        matches = (self.support.flat == value.reshape(1, -1)).all(dim=1).nonzero().flatten()
        if len(matches):
            return self.support.atom_ids[int(matches[0])]
        if required:
            raise ValueError("Target has no positive-mass atom in the declared reference law")
        return None

    def to(self, device):
        support = FiniteSupport(
            self.support.atoms.to(device), self.support.atom_ids, self.support.aliases,
            weights=self.support.weights.to(device),
            source_record_counts=(self.support.source_record_counts
                                  or self.metadata.get("source_record_multiplicities")),
            candidate_chunk=self.support.candidate_chunk, query_chunk=self.support.query_chunk,
        )
        # Device placement must preserve the exact declared floating masses.
        support.weights = self.support.weights.to(device).clone()
        support.log_weights = support.weights.log()
        estimated_mean = None if self.estimated_mean_vector is None else self.estimated_mean_vector.to(device)
        return ReferenceLaw(support, dict(self.metadata), self.law_hash, estimated_mean)


def reference_law_from_atoms(atoms, atom_ids, *, weights=None, provenance=None,
                             preprocessing=None, scope="declared_finite_reference_law",
                             candidate_chunk_size=256, query_chunk_size=16, device=None):
    """Deduplicate exact atoms and aggregate supplied probability masses.

    An omitted weight vector means equal mass per supplied source record.
    Explicit weights mean masses on supplied rows. Both aggregate the masses
    of exact duplicate atoms without dropping source-record multiplicity.
    Zero-mass-only atoms are excluded from the mathematical support explicitly.
    """
    values = list(atoms)
    ids = [str(value) for value in atom_ids]
    if len(values) != len(ids) or not values or len(set(ids)) != len(ids):
        raise ValueError("A nonempty finite law requires unique source atom IDs")
    support = FiniteSupport.from_candidates(
        zip(ids, values), candidate_chunk=candidate_chunk_size, query_chunk=query_chunk_size, device=device
    )
    excluded = []
    if weights is not None:
        supplied = torch.as_tensor(weights, dtype=torch.float64, device=support.flat.device)
        if supplied.shape != (len(ids),) or not bool(torch.isfinite(supplied).all() & (supplied >= 0).all()) or not bool((supplied > 0).any()):
            raise ValueError("Reference masses must be finite/nonnegative with positive total mass")
        # Scaling before summation avoids overflow without changing the law.
        scaled = supplied / supplied.max()
        if bool(((supplied > 0) & (scaled == 0)).any()):
            raise ValueError("Explicit positive reference mass underflows float64 normalization")
        supplied = scaled
        masses = torch.zeros(support.size, dtype=torch.float64, device=support.flat.device)
        for key, mass in zip(ids, supplied):
            masses[support.aliases[key]] += mass
        keep = masses > 0
        mapping = {old: new for new, old in enumerate(keep.nonzero().flatten().tolist())}
        excluded = [key for key, index in support.aliases.items() if index not in mapping]
        support = FiniteSupport(
            support.atoms[keep], [key for key, use in zip(support.atom_ids, keep.tolist()) if use],
            {key: mapping[index] for key, index in support.aliases.items() if index in mapping},
            weights=masses[keep], candidate_chunk=candidate_chunk_size, query_chunk=query_chunk_size,
        )
    if not bool((support.weights > 0).all() & torch.isfinite(support.weights).all()):
        raise ValueError("Normalized positive reference weights are not representable")
    return _reference_law_from_support(
        support, provenance=provenance, preprocessing=preprocessing, scope=scope,
        excluded=excluded,
    )


def _reference_law_from_support(support, *, provenance=None, preprocessing=None,
                                scope="declared_finite_reference_law", excluded=()):
    """Attach a law receipt without re-deduplicating or reweighting its atoms."""
    metadata = {
        **support.metadata(), "version": REFERENCE_LAW_VERSION,
        "evidence": "declared_finite_reference_law",
        "reference_law_scope": scope,
        "preprocessing": preprocessing or {}, "provenance": provenance or {},
        "single_target_assumption": _SINGLE_TARGET,
        "zero_mass_source_ids": list(excluded),
        "mean_definition": "exact_declared_weighted_atom_mean_not_model_output_center",
        "mean_sha256": _tensor_hash(support.weights @ support.flat),
        "max_atom_norm_l2": float(_row_norm(support.flat).max()),
    }
    identity = {key: value for key, value in metadata.items() if key not in {"candidate_chunk", "query_chunk", "complexity"}}
    return ReferenceLaw(support, metadata, canonical_hash(identity))


def _prompt_scope(rows, law):
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    groups, missing = {}, 0
    for row in rows:
        exact_prompt = str(row["prompt_raw"])
        candidate_id = row["candidate_id"]
        index = None if candidate_id is None else law.support.aliases.get(str(candidate_id))
        identity = law.support.atom_ids[index] if index is not None else row.get("target_atom_sha256")
        if index is None:
            missing += 1
        groups.setdefault(exact_prompt, set()).add(str(identity))
    conflicts = [
        {"prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
         "distinct_target_atom_ids": sorted(identities),
         "scope": "observed_exact_prompt_maps_to_multiple_cached_atoms; Eq6_not_established"}
        for prompt, identities in groups.items() if len(identities) > 1
    ]
    return {"assumption": _SINGLE_TARGET, "exact_prompt_count": len(groups),
            "ambiguous_exact_prompt_count": len(conflicts), "ambiguities": conflicts,
            "source_records_without_positive_reference_atom": missing,
            "record_policy": "retain_all_records_without_assumption_based_selection"}


def build_reference_law(sources, config, device="cpu", *, allow_mean_compute=False,
                        mean_device=None, mean_batch_size=8):
    """Build once at analysis runtime, from one declared source and fixed masses."""
    # Posterior chunk sizes control execution, not the scientific cache identity.
    # Strip them from a local copy before forwarding config to the mean estimator.
    config = dict(config)
    mode = config.get("reference_law", "cached-targets")
    manifest_path = config.get("reference_manifest")
    chunks = {"candidate_chunk_size": config.pop("candidate_chunk_size", 256),
              "query_chunk_size": config.pop("query_chunk_size", 16)}
    science = sources.runs["experiment"]["scientific_config"]
    preprocessing = {key: science.get(key) for key in (
        "vae_id", "vae_revision", "latent_shape", "target_preprocessing", "target_latent_definition"
    )}
    if mode == "cached-targets":
        if manifest_path is not None:
            raise ValueError("reference_manifest is forbidden for cached-targets law")
        from .reduce import build_support
        support, source_metadata, rows = build_support(sources, **chunks, device=device)
        # Keep the already aggregated empirical prior. Rebuilding from the
        # distinct atom list would incorrectly replace multiplicities by 1/K.
        placed = FiniteSupport(
            support.atoms, support.atom_ids, support.aliases, weights=support.weights,
            source_record_counts=support.source_record_counts,
            candidate_chunk=chunks["candidate_chunk_size"], query_chunk=chunks["query_chunk_size"],
        )
        placed.weights = support.weights.clone()
        placed.log_weights = placed.weights.log()
        law = _reference_law_from_support(
            placed,
            provenance={key: value for key, value in source_metadata.items()
                        if key not in {"candidate_chunk", "query_chunk", "complexity"}},
            preprocessing=preprocessing,
            scope="D_K_all_compatible_complete_cached_source_records_preselection_empirical_multiplicity",
        )
    elif mode == "manifest":
        if manifest_path is None:
            raise ValueError("A reference manifest is required for manifest mode")
        path = _safe_path(manifest_path)
        specification = read_json(path)
        if specification.get("schema_version") != 1:
            raise ValueError("Unsupported explicit reference-law manifest schema")
        if specification.get("preprocessing") != preprocessing:
            raise ValueError("Reference-law manifest preprocessing/VAE contract differs")
        provenance = specification.get("provenance")
        if not isinstance(provenance, dict) or not provenance.get("source_identity"):
            raise ValueError("Reference-law manifest requires explicit source provenance")
        if not isinstance(specification.get("declared_complete"), bool):
            raise ValueError("Reference-law completeness must be explicitly declared, never inferred")
        entries = specification.get("atoms")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Reference-law manifest has no atom population")
        atoms, ids, masses, rows = [], [], [], []
        for entry in entries:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Reference atom payload must be contained by its manifest")
            source = _safe_path(path.parent / relative)
            if not source.is_file() or file_sha256(source) != entry.get("sha256"):
                raise ValueError(f"Missing or changed reference atom: {source}")
            atom = safe_torch_load(source)
            if not isinstance(atom, torch.Tensor) or list(atom.shape) != science["latent_shape"]:
                raise ValueError("Reference atom latent shape/type differs")
            atoms.append(atom)
            ids.append(str(entry["id"]))
            masses.append(entry["weight"])
        law = reference_law_from_atoms(
            atoms, ids, weights=masses, preprocessing=preprocessing,
            provenance={**provenance, "manifest_sha256": file_sha256(path),
                        "declared_complete": specification["declared_complete"],
                        "completeness_scope": "explicit_source_claim_not_inferred_or_independently_established"},
            scope="manifest_declared_finite_reference_law", device=device, **chunks,
        )
        # Audit retained source prompts against exact manifest atoms, preserving
        # missing target atoms as a scope issue rather than silently adding them.
        for record in sources.records["experiment"]:
            source = _safe_path(sources.experiment.target_latent_path(record.original_index))
            if file_sha256(source) != record.metadata["tensor_file_sha256"]["target_latent"]:
                raise ValueError("Cached target changed during reference-law assumption audit")
            target = safe_torch_load(source)
            key = law.target_id_for(target, required=False)
            rows.append({"candidate_id": key, "prompt_raw": record.metadata["prompt_raw"],
                         "target_atom_sha256": _tensor_hash(target)})
    else:
        raise ValueError("reference_law must be cached-targets or manifest")
    if config.get("mean_source", "cached-targets") in {"reference-min-snr", "reference-initial"}:
        # Estimation is an explicit run-level stage. A worker may only load the
        # already completed immutable estimate, never launch nested computation.
        from .reference_mean import estimate_reference_mean
        estimated = estimate_reference_mean(
            sources, config, reference_law=law,
            device=device if mean_device is None else mean_device,
            batch_size=mean_batch_size, allow_compute=allow_mean_compute,
        )
        receipt = estimated["metadata"]
        keys = ("source", "mean_source", "estimator_hash", "vector_sha256", "sample_count", "mean_seed",
                "initial_snr", "initial_level", "estimation_snr", "reference_snr_decades", "reference_grid_definition",
                "level", "mean_norm_l2", "mean_norm_rmse",
                "mean_mc_standard_error_l2", "mean_mc_standard_error_rmse",
                "mean_mc_standard_error_max_coordinate", "split_half_difference_rmse",
                "split_half_counts", "uncertainty_scope", "reference_atom_law_hash",
                "definition", "bias_scope")
        law.metadata["theory_mean"] = {key: receipt[key] for key in keys if key in receipt}
        law.estimated_mean_vector = estimated["vector"].to(device=law.support.flat.device, dtype=torch.float64)
        law.__post_init__()
    else:
        law.metadata["theory_mean"] = {
            "source": "declared_finite_bank_mean", "sample_count": None,
            "vector_sha256": law.metadata["mean_sha256"],
            "definition": "Exact declared weighted atom mean",
        }
    law.metadata["prompt_assumption_audit"] = _prompt_scope(rows, law)
    law.law_hash = canonical_hash({key: value for key, value in law.metadata.items()
                                  if key not in {"candidate_chunk", "query_chunk", "complexity"}})
    return law.to(device) if str(device) != "cpu" else law
