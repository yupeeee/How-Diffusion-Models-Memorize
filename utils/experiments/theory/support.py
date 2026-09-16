"""Explicit uniform finite-support Gaussian posterior; no learned posterior claim.

Storage is O(M*d). Evaluation uses candidate and query chunks with matrix
multiplication: O(Q*M*d) work and O(query_chunk*(candidate_chunk+d)) scratch.
No complete posterior histories or query-by-candidate-by-latent arrays are kept.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import torch

EVIDENCE = "candidate_distribution_diagnostic"


def _tensor_hash(tensor):
    canonical = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    canonical = torch.where(canonical == 0, torch.zeros_like(canonical), canonical)
    return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()


class FiniteSupport:
    def __init__(
        self,
        atoms,
        atom_ids,
        aliases,
        *,
        candidate_chunk=256,
        query_chunk=32,
        weights=None,
    ):
        self.atoms = torch.as_tensor(atoms).detach().to(dtype=torch.float64)
        self.latent_shape = tuple(self.atoms.shape[1:])
        self.flat = self.atoms.reshape(self.atoms.shape[0], -1)
        self.size, self.dimension = self.flat.shape
        if self.size == 0 or not bool(torch.isfinite(self.flat).all()):
            raise ValueError("Support must contain finite target atoms")
        self.atom_ids = [str(x) for x in atom_ids]
        self.aliases = {str(k): int(v) for k, v in aliases.items()}
        self.candidate_chunk, self.query_chunk = int(candidate_chunk), int(query_chunk)
        if min(self.candidate_chunk, self.query_chunk) < 1:
            raise ValueError("Support chunk sizes must be positive")
        if weights is None:
            self.weights = torch.full(
                (self.size,),
                1 / self.size,
                dtype=torch.float64,
                device=self.flat.device,
            )
        else:
            self.weights = torch.as_tensor(
                weights, dtype=torch.float64, device=self.flat.device
            )
            if self.weights.shape != (self.size,) or not bool(
                torch.isfinite(self.weights).all() & (self.weights > 0).all()
            ):
                raise ValueError(
                    "Every declared support weight must be finite and positive"
                )
            self.weights = self.weights / self.weights.sum()
        self.log_weights = self.weights.log()

    @classmethod
    def from_candidates(
        cls,
        candidates: Iterable[tuple[str, torch.Tensor]],
        *,
        identities=None,
        device=None,
        **kwargs,
    ):
        """Collapse identical latent atoms; conflicting repeated IDs/identities fail.

        Identities, when provided, map candidate IDs to target-image identities.
        Different aliases for one identity must have identical cached latents.
        """
        canonical = sorted(
            (
                (
                    str(key),
                    torch.as_tensor(value)
                    .detach()
                    .to(device=device, dtype=torch.float64),
                )
                for key, value in candidates
            ),
            key=lambda x: x[0],
        )
        if not canonical:
            raise ValueError("Cannot construct an empty recovered candidate support")
        shape = canonical[0][1].shape
        atoms, atom_ids, aliases, hashes, seen_identity = [], [], {}, {}, {}
        for key, atom in canonical:
            if atom.shape != shape:
                raise ValueError(
                    "Candidate targets use incompatible latent contracts/shapes"
                )
            digest = _tensor_hash(atom)
            if key in aliases and not torch.equal(atoms[aliases[key]], atom):
                raise ValueError(f"Candidate ID {key!r} has conflicting target latents")
            identity = (
                str(identities[key])
                if identities is not None and key in identities
                else key
            )
            if identity in seen_identity and seen_identity[identity] != digest:
                raise ValueError(
                    f"Target identity {identity!r} has conflicting cached latents"
                )
            seen_identity[identity] = digest
            if digest not in hashes:
                hashes[digest] = len(atoms)
                atoms.append(atom.contiguous())
                atom_ids.append(key)
            aliases[key] = hashes[digest]
        return cls(torch.stack(atoms), atom_ids, aliases, **kwargs)

    def metadata(self):
        return {
            "evidence": EVIDENCE,
            "support_size": self.size,
            "latent_dimension": self.dimension,
            "latent_shape": list(self.latent_shape),
            "candidate_mean_l2": float((self.weights @ self.flat).norm()),
            "candidate_mean_sha256": _tensor_hash(self.weights @ self.flat),
            "atom_ids": self.atom_ids,
            "aliases": self.aliases,
            "weights": self.weights.detach().cpu().tolist(),
            "weight_definition": (
                "predeclared_uniform_unique_atoms"
                if bool(torch.all(self.weights == self.weights[0]))
                else "predeclared_explicit_weights"
            ),
            "deduplication": "exact_float64_latent_values_and_consistent_target_identity",
            "tensor_sha256": _tensor_hash(self.atoms),
            "source_qualification": "all_compatible_recovered_cached_targets; not_known_training_distribution",
            "candidate_chunk": self.candidate_chunk,
            "query_chunk": self.query_chunk,
            "complexity": "O(queries*support_size*latent_dimension) work; bounded query/candidate chunks",
        }

    def _queries(self, z):
        z = torch.as_tensor(z, dtype=torch.float64, device=self.flat.device)
        if tuple(z.shape[-len(self.latent_shape) :]) != self.latent_shape:
            raise ValueError(
                "Query latent shape differs from declared support contract"
            )
        lead = z.shape[: -len(self.latent_shape)]
        return z.reshape(-1, self.dimension), lead

    def _target(self, target_id):
        try:
            return self.aliases[str(target_id)]
        except KeyError as error:
            raise ValueError(
                f"Evaluated target {target_id!r} is absent from declared support"
            ) from error

    def target_geometry(self, target):
        """Cache one record's target geometry, with O(K*d) bounded storage."""
        if getattr(self, "_geometry_target", None) != target:
            self._geometry_difference = self.flat - self.flat[target]
            self._geometry_squared_l2 = self._geometry_difference.square().sum(dim=1)
            self._geometry_target = target
        return self._geometry_difference, self._geometry_squared_l2

    def _relative_logits(self, query, target, start, end, alpha, sigma):
        geometry, squared_l2 = self.target_geometry(target)
        difference = geometry[start:end]
        # Relative-to-target logits remove the common query norm and improve
        # precision at high SNR. Gaussian exponent uses RAW squared L2.
        logits = (alpha / (sigma * sigma)) * (
            (query - alpha * self.flat[target]) @ difference.T
        )
        logits -= (alpha * alpha / (2 * sigma * sigma)) * squared_l2[start:end]
        logits += self.log_weights[start:end] - self.log_weights[target]
        return logits, difference

    def evaluate(self, z, target_id, alpha, sigma, unconditional=None, *, include_mean=False):
        query, lead = self._queries(z)
        means = torch.full_like(query, torch.nan) if include_mean else None
        target = self.aliases.get(str(target_id))
        a, s = float(alpha), float(sigma)
        keys = [
            "target_log_probability",
            "target_log_complement",
            "target_log_odds",
            "candidate_unconditional_reference_error_l2",
            "candidate_radius_l2",
            "posterior_entropy",
            "maximum_posterior",
            "target_rank",
            "non_target_mass",
            "posterior_mean_target_rmse",
            "posterior_weighted_target_radius_rmse",
            "support_radius_rmse",
            "network_to_surrogate_rmse",
            "surrogate_triangle_bound_rmse",
            "surrogate_concentration_bound_rmse",
            "surrogate_triangle_slack_rmse",
        ]
        result = {
            key: torch.full(
                (len(query),), torch.nan, dtype=torch.float64, device=query.device
            )
            for key in keys
        }
        result["support_evidence"] = EVIDENCE
        if target is None:
            result["support_status"] = (
                "target_absent_from_declared_candidate_distribution"
            )
            return {
                k: v.reshape(lead) if isinstance(v, torch.Tensor) else v
                for k, v in result.items()
            }
        if not (math.isfinite(a) and math.isfinite(s) and a >= 0 and s > 0):
            result["support_status"] = (
                "invalid_destination_or_current_noise: sigma_must_be_positive"
            )
            return {
                k: v.reshape(lead) if isinstance(v, torch.Tensor) else v
                for k, v in result.items()
            }
        if not bool(torch.isfinite(query).all()):
            result["support_status"] = "invalid_nonfinite_query"
            return {
                k: v.reshape(lead) if isinstance(v, torch.Tensor) else v
                for k, v in result.items()
            }
        result["support_status"] = (
            "valid" if self.size > 1 else "single_atom_log_odds_undefined"
        )
        unconditional_flat = (
            None if unconditional is None else self._queries(unconditional)[0]
        )
        if unconditional_flat is not None and unconditional_flat.shape != query.shape:
            raise ValueError("Unconditional estimate/query batch mismatch")
        radius = self.target_geometry(target)[1].sqrt() / math.sqrt(self.dimension)
        support_radius = radius.max()
        for qs in range(0, len(query), self.query_chunk):
            qe = min(qs + self.query_chunk, len(query))
            q = query[qs:qe]
            logz = torch.full(
                (len(q),), -torch.inf, dtype=torch.float64, device=q.device
            )
            log_comp = torch.full_like(logz, -torch.inf)
            maximum, rank = torch.full_like(logz, -torch.inf), torch.ones_like(logz)
            for start in range(0, self.size, self.candidate_chunk):
                end = min(start + self.candidate_chunk, self.size)
                logits, _ = self._relative_logits(q, target, start, end, a, s)
                logz = torch.logaddexp(logz, torch.logsumexp(logits, dim=1))
                maximum = torch.maximum(maximum, logits.max(dim=1).values)
                rank += (logits > 0).sum(dim=1)
                if start <= target < end:
                    logits[:, target - start] = -torch.inf
                log_comp = torch.logaddexp(log_comp, torch.logsumexp(logits, dim=1))
            mean_difference = torch.zeros_like(q)
            weighted_radius, entropy = torch.zeros_like(logz), torch.zeros_like(logz)
            for start in range(0, self.size, self.candidate_chunk):
                end = min(start + self.candidate_chunk, self.size)
                logits, difference = self._relative_logits(q, target, start, end, a, s)
                logp = logits - logz[:, None]
                probability = logp.exp()
                mean_difference += probability @ difference
                weighted_radius += probability @ radius[start:end]
                entropy -= torch.where(probability > 0, probability * logp, 0).sum(
                    dim=1
                )
            if means is not None:
                means[qs:qe] = self.flat[target] + mean_difference
            mean_error = mean_difference.norm(dim=1) / math.sqrt(self.dimension)
            # sigmoid(log_comp) is stable even when target p rounds to one.
            non_target_mass = torch.sigmoid(log_comp)
            values = {
                "target_log_probability": -torch.logaddexp(
                    torch.zeros_like(log_comp), log_comp
                ),
                "target_log_complement": -torch.logaddexp(
                    torch.zeros_like(log_comp), -log_comp
                ),
                "candidate_radius_l2": support_radius.expand_as(logz)
                * math.sqrt(self.dimension),
                "target_log_odds": (
                    -log_comp if self.size > 1 else torch.full_like(log_comp, torch.nan)
                ),
                "posterior_entropy": entropy,
                "maximum_posterior": (maximum - logz).exp(),
                "target_rank": rank,
                "non_target_mass": non_target_mass,
                "posterior_mean_target_rmse": mean_error,
                "posterior_weighted_target_radius_rmse": weighted_radius,
                "support_radius_rmse": support_radius.expand_as(logz),
                "surrogate_concentration_bound_rmse": support_radius * non_target_mass,
            }
            if unconditional_flat is not None:
                u = unconditional_flat[qs:qe]
                discrepancy = (u - self.flat[target] - mean_difference).norm(
                    dim=1
                ) / math.sqrt(self.dimension)
                values.update(
                    network_to_surrogate_rmse=discrepancy,
                    candidate_unconditional_reference_error_l2=discrepancy
                    * math.sqrt(self.dimension),
                    surrogate_triangle_bound_rmse=discrepancy + weighted_radius,
                    surrogate_triangle_slack_rmse=discrepancy
                    + weighted_radius
                    - (u - self.flat[target]).norm(dim=1) / math.sqrt(self.dimension),
                )
            for key, value in values.items():
                result[key][qs:qe] = value
        output = {
            k: v.reshape(lead) if isinstance(v, torch.Tensor) else v
            for k, v in result.items()
        }
        if means is not None:
            output["posterior_mean"] = means.reshape(*lead, *self.latent_shape)
        return output

    def posterior_mean(self, z, alpha, sigma):
        """Transient vector mean from the same chunked posterior as scalar metrics."""
        result = self.evaluate(z, self.atom_ids[0], alpha, sigma, include_mean=True)
        if "posterior_mean" not in result:
            raise ValueError("Posterior mean unavailable: " + result["support_status"])
        return result["posterior_mean"]

    def feedback(self, z_u, z_g, target_id, alpha, sigma):
        """Matched target log-odds gain, stabilized via competitor log weights."""
        query, lead = self._queries(z_u)
        guided, guided_lead = self._queries(z_g)
        if guided_lead != lead:
            raise ValueError("Matched unconditional and guided state batches differ")
        target = self._target(target_id)
        a, s = float(alpha), float(sigma)
        keys = (
            "margin_min",
            "margin_max",
            "log_odds_gain",
            "log_probability_gain",
            "feedback_numerical_tolerance",
            "margin_condition",
            "positive_gain",
            "margin_bound_covered",
            "worst_competitor_index",
        )
        result = {
            k: torch.full(
                (len(query),), torch.nan, dtype=torch.float64, device=query.device
            )
            for k in keys
        }
        result["feedback_evidence"] = EVIDENCE
        if self.size < 2 or not (
            math.isfinite(a) and math.isfinite(s) and a >= 0 and s > 0
        ):
            result["feedback_status"] = "single_atom_or_zero_invalid_destination_sigma"
            return {
                k: v.reshape(lead) if isinstance(v, torch.Tensor) else v
                for k, v in result.items()
            }
        if not bool(torch.isfinite(query).all() & torch.isfinite(guided).all()):
            result["feedback_status"] = "invalid_nonfinite_matched_or_guided_state"
            result = {
                k: v.reshape(lead) if isinstance(v, torch.Tensor) else v
                for k, v in result.items()
            }
            result.update(
                {
                    "guided_" + k: v
                    for k, v in self.evaluate(z_g, target_id, a, s).items()
                }
            )
            return result
        result["feedback_status"] = "valid; sign_flags_zero_indicate_numerical_tie"
        for qs in range(0, len(query), self.query_chunk):
            qe = min(qs + self.query_chunk, len(query))
            q, displacement = query[qs:qe], guided[qs:qe] - query[qs:qe]
            log_comp = torch.full(
                (len(q),), -torch.inf, dtype=torch.float64, device=q.device
            )
            lo, hi = (
                torch.full_like(log_comp, torch.inf),
                torch.full_like(log_comp, -torch.inf),
            )
            worst = torch.full_like(log_comp, torch.nan)
            scale = torch.ones_like(log_comp)
            for start in range(0, self.size, self.candidate_chunk):
                end = min(start + self.candidate_chunk, self.size)
                logits, difference = self._relative_logits(q, target, start, end, a, s)
                margins = -(a / (s * s)) * (displacement @ difference.T)
                mask = torch.arange(start, end, device=q.device) != target
                if not bool(mask.any()):
                    continue
                logits, margins = logits[:, mask], margins[:, mask]
                ids = torch.arange(start, end, device=q.device)[mask]
                log_comp = torch.logaddexp(log_comp, torch.logsumexp(logits, dim=1))
                local_lo, index = margins.min(dim=1)
                worst = torch.where(local_lo < lo, ids[index].double(), worst)
                lo, hi = (
                    torch.minimum(lo, local_lo),
                    torch.maximum(hi, margins.max(dim=1).values),
                )
                scale = torch.maximum(
                    scale,
                    torch.maximum(
                        logits.abs().max(1).values, margins.abs().max(1).values
                    ),
                )
            weighted_log = torch.full_like(log_comp, -torch.inf)
            for start in range(0, self.size, self.candidate_chunk):
                end = min(start + self.candidate_chunk, self.size)
                logits, difference = self._relative_logits(q, target, start, end, a, s)
                margins = -(a / (s * s)) * (displacement @ difference.T)
                mask = torch.arange(start, end, device=q.device) != target
                if bool(mask.any()):
                    weighted_log = torch.logaddexp(
                        weighted_log,
                        torch.logsumexp(
                            logits[:, mask] - log_comp[:, None] - margins[:, mask],
                            dim=1,
                        ),
                    )
            gain = -weighted_log
            tolerance = (
                128
                * torch.finfo(torch.float64).eps
                * scale
                * max(1, math.log2(self.size + 1))
            )
            result["margin_min"][qs:qe], result["margin_max"][qs:qe] = lo, hi
            result["log_odds_gain"][qs:qe] = gain
            result["log_probability_gain"][qs:qe] = -torch.logaddexp(
                torch.zeros_like(gain), log_comp - gain
            ) + torch.logaddexp(torch.zeros_like(gain), log_comp)
            result["feedback_numerical_tolerance"][qs:qe] = tolerance
            result["margin_condition"][qs:qe] = torch.where(
                lo > tolerance, 1, torch.where(lo < -tolerance, -1, 0)
            )
            result["positive_gain"][qs:qe] = torch.where(
                gain > tolerance, 1, torch.where(gain < -tolerance, -1, 0)
            )
            result["margin_bound_covered"][qs:qe] = (
                (gain >= lo - tolerance) & (gain <= hi + tolerance)
            ).double()
            result["worst_competitor_index"][qs:qe] = worst
        result = {
            k: v.reshape(lead) if isinstance(v, torch.Tensor) else v
            for k, v in result.items()
        }
        for prefix, state in (("matched_", z_u), ("guided_", z_g)):
            result.update(
                {
                    prefix + k: v
                    for k, v in self.evaluate(state, target_id, a, s).items()
                }
            )
        return result
