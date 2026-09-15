"""Literal manuscript statements and honest cache-only applicability.

No tensor or model imports: this module is also safe in a copied scalar bundle.
The paper's ``e_t(empty)`` is a *reference* error, not unconditional target error.
Candidate-distribution quantities must never be supplied to the paper evaluators.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

REGISTRY_SCHEMA_VERSION = 2
FORMULA_VERSION = "revised-section3-v2-proposition5"
EVIDENCE_CLASSES = frozenset(
    {
        "algebraic_qa",
        "finite_noise_observation",
        "model_center_diagnostic",
        "identified_training_reference",
        "not_applicable",
        "candidate_distribution_diagnostic",
        "unavailable",
    }
)
STATEMENT_IDS = (
    "initial_recovery",
    "unconditional_center",
    "target_injection",
    "matched_displacement",
    "posterior_feedback",
    "target_synchronization",
    "terminal_reproduction",
)
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3] / "docs/theory/statement_registry.json"
)


def validate_statement_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate seven formal mappings, four default designs and two diagnostics."""
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError("Unsupported statement registry schema_version")
    if registry.get("formula_version") != FORMULA_VERSION:
        raise ValueError(
            "Incompatible statement registry formula_version; rebuild analysis"
        )
    statements = registry.get("statements")
    if not isinstance(statements, list):
        raise ValueError("Statement registry must contain a statements list")
    found: list[str] = []
    main_figures: list[str] = []
    for statement in statements:
        required = (
            "semantic_id",
            "label",
            "exact_statement",
            "equations",
            "assumptions",
            "observables",
            "unavailable_quantities",
            "evidence_classification",
            "main_figure",
        )
        if not isinstance(statement, dict) or any(
            key not in statement for key in required
        ):
            raise ValueError("Incomplete formal statement in registry")
        found.append(statement["semantic_id"])
        if statement["evidence_classification"] not in EVIDENCE_CLASSES:
            raise ValueError(
                f"Invalid evidence classification: {statement['semantic_id']}"
            )
        figure = statement["main_figure"]
        if figure is not None:
            if not isinstance(figure, dict) or not isinstance(figure.get("id"), str):
                raise ValueError("Invalid main_figure contract")
            main_figures.append(figure["id"])
            if figure["id"] != statement["semantic_id"] or figure.get("formats") != [
                "png",
                "pdf",
            ]:
                raise ValueError(
                    "Default figures retain semantic IDs and PNG/PDF formats"
                )
        diagnostic = statement.get("diagnostic_figure")
        expected_diagnostic = {
            "target_injection": "target_injection",
            "terminal_reproduction": "terminal_terms",
        }.get(statement["semantic_id"])
        if (diagnostic or {}).get("id") != expected_diagnostic:
            raise ValueError(
                "Diagnostic inventory differs from the corrective contract"
            )
        for observable in statement["observables"]:
            if observable.get("evidence_classification") not in EVIDENCE_CLASSES:
                raise ValueError(
                    f"Observable has no evidence classification: {observable}"
                )
        for quantity in statement["unavailable_quantities"]:
            if not quantity.get("reason"):
                raise ValueError("Unavailable paper quantity must have a reason")
    if (
        tuple(found) != STATEMENT_IDS
        or set(main_figures)
        != {
            "initial_recovery",
            "unconditional_center",
            "posterior_feedback",
            "target_synchronization",
        }
        or len(main_figures) != 4
    ):
        raise ValueError(
            "Registry requires seven revised results and exactly four designated default figures"
        )
    return dict(registry)


def load_statement_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Load the checked-in registry, or an independently copied bundle registry."""
    source = DEFAULT_REGISTRY_PATH if path is None else Path(path)
    with source.open(encoding="utf-8") as handle:
        registry = json.load(handle)
    return validate_statement_registry(registry)


def guidance_domain_status(guidance: float) -> dict[str, Any]:
    """The manuscript assumes fixed g > 1; sampler/algebra domains are separate."""
    guidance = float(guidance)
    valid = math.isfinite(guidance) and guidance > 1
    return {
        "paper_guidance_domain": "g > 1",
        "paper_guidance_domain_satisfied": valid,
        "paper_guidance_domain_reason": None
        if valid
        else "manuscript assumes finite fixed g > 1",
        "frozen_state_guidance_algebra_applicable": math.isfinite(guidance),
    }


def paper_unavailable_metrics(
    guidance: float,
    *,
    terminal_clean_update: bool | None = None,
    branch_gap_l2: float | None = None,
    conditional_target_error_l2: float | None = None,
) -> dict[str, Any]:
    """Fields emitted for ordinary preserved caches, without invented references.

    The observable feedback LHS is retained even when the RHS is unidentified.
    Conditional target error is reported separately: interpreting it as a paper
    reference error requires the unverified single-target training-law condition.
    """
    reason = "full training distribution, true posterior mean/mass and support radius are not identified"
    result: dict[str, Any] = guidance_domain_status(guidance)
    result.update(
        {
            "paper_population_forward_loss": None,
            "paper_population_forward_loss_status": "unavailable",
            "paper_population_forward_loss_reason": "requires independent forward-corruption model evaluations and expectation",
            "paper_training_mean": None,
            "paper_training_mean_reason": "reference-initial center is not the training-distribution mean",
            "paper_single_target_assumption_verified": False,
            "paper_conditional_reference_error_l2": None,
            "paper_conditional_reference_error_reason": "training-law single-target condition in Equation 6 is not verified",
            "observed_conditional_target_error_l2": conditional_target_error_l2,
            "paper_unconditional_reference_error_l2": None,
            "paper_unconditional_reference_error_reason": reason,
            "paper_target_posterior": None,
            "paper_training_support_radius_l2": None,
            "paper_feedback_lhs_l2": branch_gap_l2,
            "paper_feedback_rhs_l2": None,
            "paper_feedback_slack_l2": None,
            "paper_feedback_condition_satisfied": None,
            "paper_feedback_cross_step_variation_l2": None,
            "paper_feedback_status": "unavailable",
            "paper_feedback_reason": "Equation 15 requires true posterior means along a continuous matched segment; "
            + reason,
            "paper_synchronization_bound_l2": None,
            "paper_synchronization_status": "unavailable",
            "paper_synchronization_reason": reason,
            "paper_terminal_clean_update_satisfied": terminal_clean_update,
            "paper_terminal_bound_l2": None,
            "paper_terminal_bound_rmse": None,
            "paper_terminal_conditional_term_l2": None,
            "paper_terminal_unconditional_term_l2": None,
            "paper_terminal_concentration_term_l2": None,
            "paper_terminal_bound_status": "unavailable",
            "paper_terminal_bound_reason": reason
            + "; Equation 5 and g > 1 are additional applicability conditions",
        }
    )
    return result


def _identified_inputs(
    values: Mapping[str, float | None],
    *,
    identified_training_distribution: bool,
    single_target_verified: bool,
) -> str | None:
    if not identified_training_distribution:
        return "full training distribution unidentified; finite candidate distributions are surrogates"
    if not single_target_verified:
        return "single-target training-law condition in Equation 6 is unverified"
    if any(value is None for value in values.values()):
        return "missing identified paper input: " + ", ".join(
            key for key, value in values.items() if value is None
        )
    if any(
        not math.isfinite(float(value)) or float(value) < 0 for value in values.values()
    ):
        return "paper inputs must be finite and nonnegative"
    return None


def posterior_feedback_condition(
    branch_gap_l2: float | None,
    conditional_reference_error_l2: float | None,
    unconditional_reference_error_l2: float | None,
    cross_step_reference_variation_l2: float | None,
    *,
    guidance: float,
    kappa: float,
    destination_sigma: float,
    identified_training_distribution: bool = False,
    single_target_verified: bool = False,
) -> dict[str, Any]:
    """Proposition 5: ||Delta|| >= e(c) + e(empty) + V (raw L2).

    V must be Equation 15's identified integral, not a sampled-grid surrogate.
    ``destination_sigma > 0`` excludes the endpoint, where the proposition is
    not stated. Guidance and kappa restrictions follow the actual manuscript.
    """
    values = {
        "branch_gap_l2": branch_gap_l2,
        "conditional_reference_error_l2": conditional_reference_error_l2,
        "unconditional_reference_error_l2": unconditional_reference_error_l2,
        "cross_step_reference_variation_l2": cross_step_reference_variation_l2,
    }
    reason = _identified_inputs(
        values,
        identified_training_distribution=identified_training_distribution,
        single_target_verified=single_target_verified,
    )
    if not guidance_domain_status(guidance)["paper_guidance_domain_satisfied"]:
        reason = "Proposition 5 assumes g > 1"
    if (
        not math.isfinite(kappa)
        or kappa <= 0
        or not math.isfinite(destination_sigma)
        or destination_sigma <= 0
    ):
        reason = "Proposition 5 requires kappa > 0 and positive destination noise"
    if reason:
        return {
            "lhs_l2": branch_gap_l2,
            "rhs_l2": None,
            "slack_l2": None,
            "satisfied": None,
            "strictly_satisfied": None,
            "status": "unavailable",
            "reason": reason,
        }
    rhs = (
        float(conditional_reference_error_l2)
        + float(unconditional_reference_error_l2)
        + float(cross_step_reference_variation_l2)
    )
    slack = float(branch_gap_l2) - rhs
    return {
        "lhs_l2": float(branch_gap_l2),
        "rhs_l2": rhs,
        "slack_l2": slack,
        "satisfied": slack >= 0,
        "strictly_satisfied": slack > 0,
        "status": "identified_paper_inputs",
        "reason": None,
    }


def synchronization_bound(
    conditional_reference_error_l2: float | None,
    unconditional_reference_error_l2: float | None,
    training_support_radius_l2: float | None,
    target_posterior: float | None,
    *,
    identified_training_distribution: bool = False,
    single_target_verified: bool = False,
) -> dict[str, Any]:
    """Lemma 6: ||Delta|| <= e(c) + e(empty) + R (1-p)."""
    values = {
        "conditional_reference_error_l2": conditional_reference_error_l2,
        "unconditional_reference_error_l2": unconditional_reference_error_l2,
        "training_support_radius_l2": training_support_radius_l2,
        "target_posterior": target_posterior,
    }
    reason = _identified_inputs(
        values,
        identified_training_distribution=identified_training_distribution,
        single_target_verified=single_target_verified,
    )
    if target_posterior is not None and not 0 <= target_posterior <= 1:
        reason = "target_posterior must lie in [0,1]"
    if reason:
        return {
            "bound_l2": None,
            "concentration_term_l2": None,
            "status": "unavailable",
            "reason": reason,
        }
    concentration = float(training_support_radius_l2) * (1 - float(target_posterior))
    return {
        "bound_l2": float(conditional_reference_error_l2)
        + float(unconditional_reference_error_l2)
        + concentration,
        "concentration_term_l2": concentration,
        "status": "identified_paper_inputs",
        "reason": None,
    }


def terminal_reproduction_bound(
    conditional_reference_error_l2: float | None,
    unconditional_reference_error_l2: float | None,
    training_support_radius_l2: float | None,
    target_posterior: float | None,
    *,
    guidance: float,
    terminal_clean_update: bool,
    identified_training_distribution: bool = False,
    single_target_verified: bool = False,
) -> dict[str, Any]:
    """Theorem 7 literally, distinct from the operational affine scheduler bound.

    Returns each summand. No endpoint observation is an input to this bound.
    """
    sync = synchronization_bound(
        conditional_reference_error_l2,
        unconditional_reference_error_l2,
        training_support_radius_l2,
        target_posterior,
        identified_training_distribution=identified_training_distribution,
        single_target_verified=single_target_verified,
    )
    reason = sync["reason"]
    if not guidance_domain_status(guidance)["paper_guidance_domain_satisfied"]:
        reason = "Theorem 7 assumes g > 1; use separately named absolute-coefficient operational bound otherwise"
    if not terminal_clean_update:
        reason = (
            "Equation 5 is not satisfied: terminal a_1 = 0 and kappa_1 = 1 required"
        )
    if reason:
        return {
            "bound_l2": None,
            "conditional_term_l2": None,
            "unconditional_term_l2": None,
            "concentration_term_l2": None,
            "status": "unavailable",
            "reason": reason,
        }
    conditional = guidance * float(conditional_reference_error_l2)
    unconditional = (guidance - 1) * float(unconditional_reference_error_l2)
    concentration = (guidance - 1) * sync["concentration_term_l2"]
    return {
        "bound_l2": conditional + unconditional + concentration,
        "conditional_term_l2": conditional,
        "unconditional_term_l2": unconditional,
        "concentration_term_l2": concentration,
        "status": "identified_paper_inputs",
        "reason": None,
    }
